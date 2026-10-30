"""Visual sourcing: per-scene stock footage and photos (Pexels, Pixabay), your own media folder,
and AI-generated images (OpenAI gpt-image-1 / DALL·E, Pollinations, Agnes). Clips are never reused across
videos, portrait footage is preferred, and a scene always gets *something* (animated gradient)."""

from __future__ import annotations

import base64
import concurrent.futures
import logging
import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from PIL import Image, UnidentifiedImageError

from . import ffmpeg
from .config import Settings
from .utils import PermanentError, download, http, raise_for_status, redact, with_retries

log = logging.getLogger("purffle")

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
STOP_WORDS = {"the", "a", "an", "of", "and", "in", "on", "with", "for", "to", "at", "by", "from", "closeup"}
STOCK_SOURCES = {"pexels", "pexels-photos", "pixabay", "pixabay-photos", "local"}
AI_SOURCES = {"openai-images", "ai", "dalle", "gpt-image", "pollinations", "agnes"}
MAX_DOWNLOAD_ATTEMPTS = 3  # per query before moving on; one bad CDN response must not discard the rest
ANIMALS = {"octopus", "octopi", "shark", "whale", "dolphin", "fish", "turtle", "crab", "jellyfish", "seal",
           "penguin", "cat", "dog", "bird", "insect", "spider", "snake", "lion", "tiger", "elephant"}


@dataclass
class MediaItem:
    source: str
    id: str
    kind: str                     # video | image | generated
    url: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0
    credit: str = ""
    path: Path | None = None
    query: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.id}"

    @property
    def portrait(self) -> bool:
        return self.height > self.width


# ------------------------------------------------------------------------------------------ Pexels
def _pexels_pick_file(files: list[dict], prefer_4k: bool) -> dict | None:
    files = [f for f in files if f.get("link") and f.get("width") and f.get("height")
             and f.get("file_type", "video/mp4") == "video/mp4"]
    if not files:
        return None
    cap = 4096 if prefer_4k else 2160

    def short(f):
        return min(f["width"], f["height"])
    good = [f for f in files if 1080 <= short(f) <= cap]
    if good:
        return min(good, key=short) if not prefer_4k else max(good, key=short)
    return max(files, key=short)


def pexels_videos(query: str, key: str, prefer_4k: bool = False, orientation: str = "portrait") -> list[MediaItem]:
    params = {"query": query, "per_page": 15, "size": "medium"}
    if orientation:
        params["orientation"] = orientation
    r = http().get("https://api.pexels.com/videos/search", params=params, headers={"Authorization": key}, timeout=30)
    raise_for_status(r, "Pexels video search")
    items = []
    for v in r.json().get("videos", []):
        f = _pexels_pick_file(v.get("video_files", []), prefer_4k)
        if not f:
            continue
        items.append(MediaItem("pexels", str(v["id"]), "video", f["link"], f["width"], f["height"],
                               float(v.get("duration") or 0), (v.get("user") or {}).get("name", ""), query=query,
                               extra={"page_url": v.get("url", "")}))
    return items


def pexels_photos(query: str, key: str, orientation: str = "portrait") -> list[MediaItem]:
    params = {"query": query, "per_page": 15}
    if orientation:
        params["orientation"] = orientation
    r = http().get("https://api.pexels.com/v1/search", params=params, headers={"Authorization": key}, timeout=30)
    raise_for_status(r, "Pexels photo search")
    items = []
    for p in r.json().get("photos", []):
        src = p.get("src") or {}
        url = src.get("original")
        if not url:
            continue
        url += ("&" if "?" in url else "?") + "auto=compress&cs=tinysrgb&h=2400"
        items.append(MediaItem("pexels-photo", str(p["id"]), "image", url, p.get("width", 0), p.get("height", 0),
                               credit=p.get("photographer", ""), query=query,
                               extra={"description": p.get("alt", ""), "page_url": p.get("url", "")}))
    return items


# ------------------------------------------------------------------------------------------ Pixabay
def _tags(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = re.split(r"[,;]", str(value or ""))
    return [str(t).strip() for t in raw if str(t).strip()]


def pixabay_videos(query: str, key: str, prefer_4k: bool = False) -> list[MediaItem]:
    r = http().get("https://pixabay.com/api/videos/", timeout=30, params={
        "key": key, "q": query[:100], "per_page": 20, "safesearch": "true"})
    raise_for_status(r, "Pixabay video search")
    order = ["large", "medium", "small"] if prefer_4k else ["medium", "large", "small"]
    items = []
    for h in r.json().get("hits", []):
        vids = h.get("videos") or {}
        f = next((vids[q] for q in order if (vids.get(q) or {}).get("url")), None)
        if not f:
            continue
        items.append(MediaItem("pixabay", str(h["id"]), "video", f["url"], f.get("width", 0), f.get("height", 0),
                               float(h.get("duration") or 0), h.get("user", ""), query=query,
                               extra={"tags": _tags(h.get("tags")), "page_url": h.get("pageURL", "")}))
    return items


def pixabay_photos(query: str, key: str, orientation: str = "portrait") -> list[MediaItem]:
    pix = {"portrait": "vertical", "landscape": "horizontal"}.get(orientation, "all")
    r = http().get("https://pixabay.com/api/", timeout=30, params={
        "key": key, "q": query[:100], "image_type": "photo", "orientation": pix,
        "per_page": 20, "safesearch": "true"})
    raise_for_status(r, "Pixabay photo search")
    return [MediaItem("pixabay-photo", str(h["id"]), "image", h["largeImageURL"], h.get("imageWidth", 0),
                      h.get("imageHeight", 0), credit=h.get("user", ""), query=query,
                      extra={"tags": _tags(h.get("tags")), "page_url": h.get("pageURL", "")})
            for h in r.json().get("hits", []) if h.get("largeImageURL")]


# ------------------------------------------------------------------------------------------ AI images
def openai_image(prompt: str, dest: Path, key: str, model: str, orientation: str = "portrait") -> Path:
    body = {"model": model, "prompt": prompt, "n": 1}
    if model.startswith("dall-e"):
        size = {"portrait": "1024x1792", "landscape": "1792x1024"}.get(orientation, "1024x1024")
        body.update(size=size, response_format="b64_json", quality="hd" if model == "dall-e-3" else "standard")
    else:
        size = {"portrait": "1024x1536", "landscape": "1536x1024"}.get(orientation, "1024x1024")
        body.update(size=size, quality="medium")

    def _do():
        r = http().post("https://api.openai.com/v1/images/generations", json=body, timeout=240,
                        headers={"Authorization": f"Bearer {key}"})
        raise_for_status(r, "OpenAI image")
        d = r.json()["data"][0]
        if d.get("b64_json"):
            dest.write_bytes(base64.b64decode(d["b64_json"]))
        else:
            download(d["url"], dest)
        return dest

    return with_retries(_do, attempts=2, label="OpenAI image")


def agnes_image(prompt: str, dest: Path, key: str, model: str = "agnes-image-2.5-flash",
                orientation: str = "portrait") -> Path:
    """Generate an image through Agnes AI's OpenAI-compatible image endpoint."""
    size = {"portrait": "1024x1536", "landscape": "1536x1024"}.get(orientation, "1024x1024")
    body = {"model": model, "prompt": prompt, "size": size, "n": 1}

    def _do():
        r = http().post(
            "https://apihub.agnes-ai.com/v1/images/generations",
            json=body,
            headers={"Authorization": f"Bearer {key}"},
            timeout=240,
        )
        raise_for_status(r, "Agnes AI image")
        data = r.json().get("data") or []
        if not data:
            raise RuntimeError("Agnes AI image response contained no data")
        item = data[0]
        encoded = item.get("b64_json") or item.get("b64")
        if encoded:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(base64.b64decode(encoded))
        elif item.get("url"):
            download(item["url"], dest, timeout=180, max_bytes=30 * 1024 * 1024)
        else:
            raise RuntimeError("Agnes AI image response contained neither image URL nor base64 data")
        validate_generated_image(dest)
        return dest

    return with_retries(_do, attempts=2, label="Agnes AI image")


_POLLINATIONS_LOCK = threading.Lock()


def pollinations_image(prompt: str, dest: Path, width: int, height: int, key: str = "") -> Path:
    url = (f"https://image.pollinations.ai/prompt/{quote(prompt[:900])}"
           f"?width={width}&height={height}&nologo=true&model=flux&seed={random.randint(1, 10**9)}")
    headers = {"Authorization": f"Bearer {key}"} if key else None
    # One request at a time: parallel scenes would all be rate-limited. Since September 2026 anonymous
    # use is limited to roughly one image every few minutes (HTTP 402 "payment required" otherwise).
    with _POLLINATIONS_LOCK:
        return download(url, dest, headers=headers, timeout=180, max_bytes=30 * 1024 * 1024)


# ------------------------------------------------------------------------------------------ selection
def simplify_query(query: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z]+", query.lower()) if w not in STOP_WORDS]
    return " ".join(words[:2])


def _shape(item: MediaItem) -> str:
    if item.height > item.width:
        return "portrait"
    return "landscape" if item.width > item.height else "square"


def _metadata_text(item: MediaItem) -> str:
    """Provider-supplied content metadata. A photographer's name is a credit, not a description."""
    tags = item.extra.get("tags") or []
    if isinstance(tags, str):
        tags = _tags(tags)
    desc = item.extra.get("description") or ""
    return " ".join([*map(str, tags), str(desc)]).strip()


def _relevance(item: MediaItem, query: str, topic: str = "") -> float:
    """0..1 overlap between the scene/topic and the provider's tags/description.

    Items without metadata are not penalised (some providers return only a URL); they score 0 and the
    other signals (video, portrait, resolution, duration) decide.
    """
    meta = set(re.findall(r"[a-z]+", _metadata_text(item).lower())) - STOP_WORDS
    if not meta:
        return 0.0
    wanted = set(re.findall(r"[a-z]+", f"{query} {topic}".lower())) - STOP_WORDS
    if not wanted:
        return 0.0
    overlap = len(meta & wanted)
    if not overlap:
        return 0.0
    recall = overlap / len(wanted)
    precision = overlap / min(len(meta), 8)
    score = min(1.0, 0.75 * recall + 0.25 * precision)
    if query.lower().strip() and query.lower().strip() in _metadata_text(item).lower():
        score = min(1.0, score + 0.2)
    return score


def _score(item: MediaItem, need: float, orientation: str = "portrait", relevance: float = 0.0) -> float:
    s = relevance * 4.0
    if item.kind == "video":
        s += 5.0  # a moving clip is preferred over a still when both are relevant
    if _shape(item) == orientation or (orientation == "square" and item.width and item.height):
        s += 3
    if item.kind == "video":
        if item.duration >= need:
            s += 2
        elif item.duration >= need * 0.6:
            s += 1
        else:
            s -= 2
    if min(item.width, item.height) >= 1080:
        s += 1
    return s + random.uniform(0, 1.2)  # a little variety between similar candidates


def validate_download(kind: str, path: Path) -> None:
    """Fail fast on an HTML error page or a truncated video saved with a media extension."""
    if kind == "video":
        try:
            info = ffmpeg.probe(path)
        except Exception as e:
            raise RuntimeError(f"downloaded video is unreadable: {path.name} ({e})") from e
        if not info.get("has_video") or info.get("duration", 0) <= 0:
            raise RuntimeError(f"downloaded video has no usable video stream: {path.name}")
        return
    try:
        with Image.open(path) as im:
            im.verify()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise RuntimeError(f"downloaded image is unreadable: {path.name} ({e})") from e


def validate_generated_image(path: Path, min_dimension: int = 256,
                             max_bytes: int = 30 * 1024 * 1024) -> tuple[int, int]:
    """Reject an AI-image response that is not a usable, reasonably sized image."""
    if not path.exists():
        raise RuntimeError(f"generated image was not written: {path.name}")
    size = path.stat().st_size
    if size < 1024:
        raise RuntimeError(f"generated image is too small to be usable: {path.name} ({size} bytes)")
    if size > max_bytes:
        raise RuntimeError(f"generated image is larger than the {max_bytes} byte limit: {path.name}")
    try:
        with Image.open(path) as im:
            width, height = im.size
            fmt = (im.format or "").upper()
            im.verify()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise RuntimeError(f"generated image is unreadable: {path.name} ({e})") from e
    if fmt not in {"PNG", "JPEG", "WEBP"}:
        raise RuntimeError(f"generated image format {fmt or 'unknown'} is not supported: {path.name}")
    if min(width, height) < min_dimension:
        raise RuntimeError(f"generated image dimensions {width}x{height} are below "
                           f"the {min_dimension}px minimum: {path.name}")
    return width, height


class Visuals:
    def __init__(self, settings: Settings, used_keys: set[str] | None = None):
        self.s = settings
        self.used = set(used_keys or ())
        self.lock = threading.Lock()
        self._search_cache: dict[tuple[str, str], list[MediaItem]] = {}
        self._search_kinds: dict[tuple[str, str], dict[str, int]] = {}
        self._warned_keys: set[str] = set()
        self.sources = [x for x in settings.visual_sources if x]
        if settings.offline:
            self.sources = ["local"]
        self.credits: set[str] = set()
        self.reports: dict[int, dict] = {}

    def _drop_source(self, source: str) -> None:
        with self.lock:
            if source in self.sources:
                self.sources = [x for x in self.sources if x != source]

    def _warn_missing_key(self, source: str, env: str) -> None:
        """A selected source without a key must not be retried for every scene."""
        if env in self._warned_keys:
            return
        self._warned_keys.add(env)
        log.warning("Visual source '%s' is selected but %s is not set; it cannot return media and will be "
                    "skipped while the next selected source is tried. Set %s in .env or drop %s from "
                    "VISUAL_SOURCES/--visuals.", source, env, env, source)
        self._drop_source(source)

    def _optional(self, fn, source: str, query: str) -> list[MediaItem]:
        """A secondary request (still-image fallback) must not discard usable video results."""
        try:
            return fn()
        except Exception as e:
            log.debug("%s photo fallback for %r failed: %s", source, query, redact(e))
            return []

    # -- search per source (cached per query) --
    def _search(self, source: str, query: str) -> list[MediaItem]:
        ck = (source, query.lower())
        if ck in self._search_cache:
            return self._search_cache[ck]
        s = self.s
        items: list[MediaItem] = []
        failed = False
        try:
            o = s.orientation
            if source == "pexels":
                if not s.pexels_api_key:
                    self._warn_missing_key(source, "PEXELS_API_KEY")
                    failed = True
                else:
                    vids = pexels_videos(query, s.pexels_api_key, s.prefer_4k, orientation=o)
                    if len(vids) < 3:
                        vids += pexels_videos(query, s.pexels_api_key, s.prefer_4k, orientation="")
                    # Keep a still option in the pool; scoring still strongly prefers a usable video.
                    items = vids + self._optional(lambda: pexels_photos(query, s.pexels_api_key, orientation=o),
                                                  source, query)
            elif source == "pexels-photos":
                if not s.pexels_api_key:
                    self._warn_missing_key(source, "PEXELS_API_KEY")
                    failed = True
                else:
                    items = pexels_photos(query, s.pexels_api_key, orientation=o)
            elif source == "pixabay":
                if not s.pixabay_api_key:
                    self._warn_missing_key(source, "PIXABAY_API_KEY")
                    failed = True
                else:
                    items = (pixabay_videos(query, s.pixabay_api_key, s.prefer_4k)
                             + self._optional(lambda: pixabay_photos(query, s.pixabay_api_key, orientation=o),
                                              source, query))
            elif source == "pixabay-photos":
                if not s.pixabay_api_key:
                    self._warn_missing_key(source, "PIXABAY_API_KEY")
                    failed = True
                else:
                    items = pixabay_photos(query, s.pixabay_api_key, orientation=o)
            elif source == "local":
                items = self._local(query)
            seen: set[str] = set()
            items = [i for i in items if not (i.key in seen or seen.add(i.key))]
            self._search_kinds[ck] = {
                "video": sum(i.kind == "video" for i in items),
                "image": sum(i.kind == "image" for i in items),
            }
        except PermanentError as e:
            log.warning("%s search failed permanently (%s); disabling it for this video", source, redact(e))
            self._drop_source(source)
            failed = True
        except Exception as e:
            log.warning("%s search for %r failed: %s", source, query, redact(e))
            failed = True
        # Only remember a result that reflects the provider's answer. A transient network error must not
        # poison this query for the rest of the run (the next scene may retry and succeed).
        if not failed:
            self._search_cache[ck] = items
        return items

    def _filter_relevant(self, candidates: list[MediaItem], query: str, topic: str) -> list[MediaItem]:
        """Reject results with provider metadata that clearly contradicts the scene.

        Only content metadata (Pixabay tags, Pexels alt text) is considered. Credits are names, not
        descriptions. Results without metadata are kept because there is no evidence against them.
        If every result is rejected, the caller gets an empty list so it can report why and move on to
        the next source instead of silently using irrelevant footage.
        """
        if not candidates:
            return candidates
        wanted = set(re.findall(r"[a-z]+", f"{query} {topic}".lower())) - STOP_WORDS
        target_animals = wanted & ANIMALS
        filtered: list[MediaItem] = []
        for item in candidates:
            meta = set(re.findall(r"[a-z]+", _metadata_text(item).lower())) - STOP_WORDS
            if not meta:
                filtered.append(item)
                continue
            found_animals = meta & ANIMALS
            if target_animals and found_animals and not (target_animals & found_animals):
                log.debug("Rejecting %s: metadata describes %s, scene asks for %s",
                          item.key, sorted(found_animals), sorted(target_animals))
                continue
            if target_animals and _relevance(item, query, topic) == 0:
                log.debug("Rejecting %s: no metadata overlap for specific subject %s", item.key, sorted(target_animals))
                continue
            filtered.append(item)
        if not filtered:
            log.warning("No metadata-relevant result for %r (rejected %d); trying the next source",
                        query, len(candidates))
        return filtered

    def _local(self, query: str) -> list[MediaItem]:
        root = Path(self.s.media_dir)
        if not root.is_dir():
            return []
        words = set(simplify_query(query).split()) | set(query.lower().split())
        items = []
        for p in root.rglob("*"):
            ext = p.suffix.lower()
            if ext not in VIDEO_EXT | IMAGE_EXT:
                continue
            stem = set(re.findall(r"[a-z]+", p.stem.lower()))
            item = MediaItem("local", str(p), "video" if ext in VIDEO_EXT else "image", path=p, query=query)
            item.extra["overlap"] = len(words & stem)
            items.append(item)
        items.sort(key=lambda i: i.extra["overlap"], reverse=True)
        return items

    def _claim(self, candidates: list[MediaItem], need: float, query: str = "", topic: str = "") -> MediaItem | None:
        with self.lock:
            fresh = [c for c in candidates if c.key not in self.used]
            if not fresh:
                return None
            if fresh[0].source == "local":
                best = fresh[0] if fresh[0].extra.get("overlap") else random.choice(fresh)
            else:
                best = max(fresh, key=lambda c: _score(
                    c, need, self.s.orientation, _relevance(c, query, topic)))
            self.used.add(best.key)
            return best

    def _fetch(self, item: MediaItem, dest_dir: Path, idx: int) -> MediaItem:
        if item.source == "local":
            return item

        ext = ".mp4" if item.kind == "video" else ".jpg"
        dest = dest_dir / f"scene{idx:02d}_{item.source}_{item.id}{ext}"

        try:
            if item.source.startswith("pixabay"):
                r = http().get(
                    item.url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154.0 Safari/537.36",
                        "Referer": "https://pixabay.com/",
                    },
                    timeout=120,
                    stream=True,
                )
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            else:
                download(item.url, dest)

            if not dest.exists() or dest.stat().st_size < 10000:
                raise RuntimeError(f"download produced an invalid file: {dest}")
            validate_download(item.kind, dest)
        except Exception:
            dest.unlink(missing_ok=True)
            raise

        item.path = dest
        item.extra["bytes"] = dest.stat().st_size
        item.extra["validated"] = True
        return item

    def _ai(self, source: str, prompt: str, dest_dir: Path, idx: int) -> MediaItem | None:
        s = self.s
        full = f"{prompt}. {s.visual_style}. No text, no letters, no watermark."
        dest = dest_dir / f"scene{idx:02d}_ai.png"
        try:
            if source in ("openai-images", "ai", "dalle", "gpt-image"):
                if not s.openai_api_key:
                    self._warn_missing_key(source, "OPENAI_API_KEY")
                    return None
                openai_image(full, dest, s.openai_api_key, s.image_model, s.orientation)
            elif source == "pollinations":
                pollinations_image(full, dest.with_suffix(".jpg"), s.width, s.height, s.pollinations_api_key)
                dest = dest.with_suffix(".jpg")
            elif source == "agnes":
                if not s.agnes_api_key:
                    self._warn_missing_key(source, "AGNES_API_KEY")
                    return None
                agnes_image(full, dest, s.agnes_api_key, s.agnes_image_model, s.orientation)
            else:
                return None
        except PermanentError as e:
            # A refused key or an exhausted free tier won't recover within this video: stop asking.
            if source in self.sources:
                self._drop_source(source)
                hint = (" Anonymous Pollinations use is now heavily rate-limited; set POLLINATIONS_API_KEY "
                        "or use Pexels/Pixabay (free keys)." if source == "pollinations" and " 402" in str(e)
                        else "")
                log.warning("AI images from %s refused (%s); skipping it for this video.%s",
                            source, redact(e)[:160], hint)
            return None
        except Exception as e:
            log.warning("AI image (%s) for scene %d failed: %s", source, idx + 1, redact(e))
            return None
        log.info("Scene %d: %s image", idx + 1, source)
        return MediaItem(source, dest.name, "image", path=dest, width=s.width, height=s.height, query=prompt)

    def for_scene(self, idx: int, query: str, image_prompt: str, topic: str, need: float, dest_dir: Path,
                  *, prefer_image: bool = False, narration: str = "") -> MediaItem:
        queries = []
        for q in (query, simplify_query(query), simplify_query(topic), topic):
            q = (q or "").strip()
            if q and q.lower() not in [x.lower() for x in queries]:
                queries.append(q)
        configured = list(dict.fromkeys(self.sources))
        stock_sources = [x for x in configured if x in STOCK_SOURCES]
        ai_sources = [x for x in configured if x in AI_SOURCES]
        unknown_sources = [x for x in configured if x not in STOCK_SOURCES | AI_SOURCES]
        report = {"scene": idx + 1, "query": query, "topic": topic, "attempts": [], "chosen": None, "reason": ""}

        def note(**kw) -> None:
            with self.lock:
                report["attempts"].append(kw)

        tried: list[str] = []
        ai_tried: set[str] = set()
        hybrid_prompt = image_prompt or query
        if prefer_image and narration.strip():
            hybrid_prompt = f"{hybrid_prompt}. Scene context: {narration.strip()}"

        # Hybrid image scene: generate a topic-specific still before looking for stock. The normal
        # stock-first rule still applies to every other scene, so AI cannot pre-empt footage there.
        if prefer_image:
            for source in ai_sources:
                tried.append(source)
                ai_tried.add(source)
                item = self._ai(source, hybrid_prompt, dest_dir, idx)
                if item:
                    report["chosen"] = {"source": item.source, "id": item.id, "kind": item.kind,
                                        "query": hybrid_prompt}
                    report["reason"] = "hybrid mode: generated an AI image for this scene"
                    note(source=source, query=hybrid_prompt, result="hybrid-image",
                         reason=report["reason"])
                    log.info("Scene %d: %s generated image (hybrid)", idx + 1, source)
                    with self.lock:
                        self.reports[idx] = report
                    return item
                note(source=source, query=hybrid_prompt, result="failed",
                     reason="AI image failed; falling back to stock")
            if not ai_sources:
                note(source="", query=hybrid_prompt, result="miss",
                     reason="hybrid scene but no AI image source is configured")
            else:
                log.warning("Scene %d: no hybrid AI image was generated; falling back to stock footage",
                            idx + 1)

        # Stock first: a real clip (or photo) must be considered before generating an image, regardless of
        # where the AI source sits in VISUAL_SOURCES. This is what stops an image provider from silently
        # pre-empting the Pixabay/Pexels clips the user selected.
        for source in stock_sources:
            tried.append(source)
            if source not in self.sources:
                note(source=source, query="", result="skipped", reason="source unavailable")
                continue
            for q in queries:
                if source not in self.sources:
                    break
                raw = self._search(source, q)
                kinds = self._search_kinds.get((source, q.lower()), {})
                cand = self._filter_relevant(raw, q, topic)
                attempts = 0
                while cand and attempts < MAX_DOWNLOAD_ATTEMPTS:
                    item = self._claim(cand, need, q, topic)
                    if not item:
                        break
                    attempts += 1
                    try:
                        self._fetch(item, dest_dir, idx)
                    except Exception as e:
                        note(source=source, query=q, found=len(raw), usable=len(cand), result="download-failed",
                             reason=redact(e), videos=kinds.get("video", 0), images=kinds.get("image", 0))
                        log.warning("Scene %d: download failed for %s: %s", idx + 1, item.key, redact(e))
                        continue
                    if item.source.startswith("pexels"):
                        self.credits.add("Pexels")
                    elif item.source.startswith("pixabay"):
                        self.credits.add("Pixabay")
                    report["chosen"] = {"source": item.source, "id": item.id, "kind": item.kind, "query": q,
                                        "width": item.width, "height": item.height, "duration": item.duration}
                    if item.kind == "video":
                        reason = "relevant stock video selected"
                        log.info("Scene %d: %s video (%r, %sx%s, %.1fs)", idx + 1, item.source, q,
                                 item.width, item.height, item.duration)
                    else:
                        reason = "no suitable stock video; using a stock photo"
                        log.warning("Scene %d: %s photo (%r): no suitable stock video clip was usable", idx + 1,
                                    item.source, q)
                    report["reason"] = (f"hybrid AI image unavailable; {reason}" if prefer_image else reason)
                    with self.lock:
                        self.reports[idx] = report
                    return item
                reason = ("no search results" if not raw else
                          "all results rejected by relevance metadata" if not cand else
                          "all matching media already used" if not attempts else
                          "all matching media failed to download")
                note(source=source, query=q, found=len(raw), usable=len(cand), result="miss",
                     reason=reason, videos=kinds.get("video", 0), images=kinds.get("image", 0))
                continue
            note(source=source, query="", result="exhausted", reason="no usable result for any query",
                 videos=sum(self._search_kinds.get((source, q.lower()), {}).get("video", 0) for q in queries),
                 images=sum(self._search_kinds.get((source, q.lower()), {}).get("image", 0) for q in queries))

        stock_reason = ("no usable stock video or photo" if stock_sources else "no stock source selected")
        for source in ai_sources:
            if source in ai_tried:
                continue
            tried.append(source)
            item = self._ai(source, image_prompt or query, dest_dir, idx)
            if item:
                report["chosen"] = {"source": item.source, "id": item.id, "kind": item.kind,
                                    "query": image_prompt or query}
                report["reason"] = f"{stock_reason}; generated an AI image instead"
                note(source=source, query=image_prompt or query, result="fallback-image", reason=stock_reason)
                log.warning("Scene %d: %s — using %s image (%r) instead", idx + 1, stock_reason, source,
                            image_prompt or query)
                with self.lock:
                    self.reports[idx] = report
                return item
            note(source=source, query=image_prompt or query, result="failed", reason="AI image failed")
        for source in unknown_sources:
            tried.append(source)
            note(source=source, query="", result="skipped", reason="unknown visual source")

        report["reason"] = f"{stock_reason}; no AI image was available"
        log.warning("Scene %d: no real visual for %r after trying %s — using an animated gradient. "
                    "Add a free Pexels/Pixabay key, an image provider, or clips in MEDIA_DIR.",
                    idx + 1, query, ", ".join(tried) or "no source")
        with self.lock:
            self.reports[idx] = report
        return MediaItem("generated", f"gradient-{idx}", "generated", query=query)

    def for_scenes(self, jobs: list[tuple], topic: str, dest_dir: Path) -> list[MediaItem]:
        """jobs: [(search_query, image_prompt, seconds_needed[, narration])] -> one item per scene.

        In ``hybrid`` mode every Nth scene (starting with the second) is image-first, producing a
        stock -> AI -> stock -> AI timeline. If no AI provider is configured or generation fails, that
        scene falls back to stock. All other scenes keep the existing stock-first behaviour.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        stock = any(s in STOCK_SOURCES for s in self.sources)
        hybrid = self.s.visual_mode == "hybrid"
        if hybrid and not any(s in AI_SOURCES for s in self.sources):
            log.warning("Hybrid mode is enabled but no AI image source is configured; add agnes (or another "
                        "AI source) to VISUAL_SOURCES/--visuals. Continuing with stock footage.")
            hybrid = False
        image_slots = set(range(1, len(jobs), max(2, self.s.hybrid_image_every))) if hybrid else set()
        workers = 2 if hybrid else (4 if stock else 2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = []
            for i, job in enumerate(jobs):
                q, p, need = job[:3]
                narration = str(job[3]) if len(job) > 3 else ""
                futs.append(ex.submit(self.for_scene, i, q, p, topic, need, dest_dir,
                                      prefer_image=i in image_slots, narration=narration))
            out: list[MediaItem] = []
            for i, f in enumerate(futs):
                try:
                    out.append(f.result())
                except Exception as e:  # one bad scene must not abort the whole video
                    log.error("Scene %d: visual lookup failed: %s; using an animated gradient", i + 1, redact(e))
                    out.append(MediaItem("generated", f"gradient-{i}", "generated", query=jobs[i][0]))
            return out
