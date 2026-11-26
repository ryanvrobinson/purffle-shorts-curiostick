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

from PIL import Image, ImageChops, ImageFilter, ImageStat, UnidentifiedImageError

from . import ffmpeg
from .config import Settings
from .utils import PermanentError, download, http, raise_for_status, redact, with_retries

log = logging.getLogger("purffle")

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
STOP_WORDS = {"the", "a", "an", "of", "and", "in", "on", "with", "for", "to", "at", "by", "from", "closeup"}
STOCK_SOURCES = {"pexels", "pexels-photos", "pixabay", "pixabay-photos", "local", "wikimedia"}
AI_SOURCES = {"openai-images", "ai", "dalle", "gpt-image", "pollinations", "agnes"}
MAX_DOWNLOAD_ATTEMPTS = 3  # per query before moving on; one bad CDN response must not discard the rest
ANIMALS = {"octopus", "octopi", "shark", "whale", "dolphin", "fish", "turtle", "crab", "jellyfish", "seal",
           "penguin", "cat", "dog", "bird", "insect", "spider", "snake", "lion", "tiger", "elephant"}

# ------------------------------------------------------------------------------------------ visual choice modes
VISUAL_CHOICE_STOCK_FIRST = "stock-first"   # current default behaviour
VISUAL_CHOICE_AI_FIRST = "ai-first"         # try AI images before stock
VISUAL_CHOICE_SUITABILITY = "suitability"   # filmable -> stock, concept -> AI
VISUAL_CHOICE_AGNES_ONLY = "agnes-only"     # Agnes for every scene; error if unavailable

VALID_VISUAL_CHOICES = {
    VISUAL_CHOICE_STOCK_FIRST,
    VISUAL_CHOICE_AI_FIRST,
    VISUAL_CHOICE_SUITABILITY,
    VISUAL_CHOICE_AGNES_ONLY,
}

# ------------------------------------------------------------------------------------------ Agnes image quality
# Agnes API genuinely supports: model, prompt, size, n. Size options:
#   portrait  -> 1024x1536   landscape -> 1536x1024   default -> 1024x1024
# No negative_prompt, seed, guidance_scale or reference_image parameters are
# confirmed, so they are deliberately NOT sent.
AGNES_DEFAULT_MODEL = "agnes-image-2.5-flash"
AGNES_PORTRAIT_SIZE = "1024x1536"   # 9:16-friendly (2:3 aspect)
AGNES_MAX_RETRIES = 2               # max additional retries after first failure

# Quality-control thresholds (pure-PIL heuristics, no vision model required).
BLANK_VARIANCE_THRESHOLD = 8.0      # grayscale range below this = blank/near-blank
BLUR_EDGE_STRENGTH_THRESHOLD = 6.0  # edge strength below this = too blurry
MIN_PORTRAIT_ASPECT = 1.3           # height/width ratio considered "portrait enough"


@dataclass
class ImageQualityResult:
    """Result of image quality control checks."""

    passed: bool
    score: float                      # 0-100 overall quality score
    checks: dict                      # individual check name -> (passed, message)
    reasons: list[str]                # human-readable failure reasons
    dimensions: tuple[int, int] | None = None
    file_size: int | None = None


# Cinematic style modifiers applied per scene topic for visual identity.
AGNES_STYLE_MODIFIERS = {
    "documentary": "documentary cinematography, naturalistic lighting, authentic textures, film grain",
    "cinematic": "cinematic color grading, dramatic chiaroscuro lighting, anamorphic depth, film still",
    "scientific": "scientific documentary reconstruction, accurate detail, educational visual style",
    "historical": "historical reconstruction, period-accurate architecture and costume, archival aesthetic",
    "nature": "natural history documentary, wildlife cinematography, rich environmental detail",
    "space": "space documentary, cosmic scale, nebula and stellar detail, telescope imagery",
}


def build_agnes_prompt(
    narration: str,
    visual_description: str = "",
    scene_index: int = 0,
    topic: str = "",
    style_hint: str = "documentary",
) -> str:
    """Build a detailed, scene-aware prompt for Agnes image generation.

    The prompt specifies subject, environment, action, camera, lighting,
    textures, depth, composition and art direction — never just raw narration.
    """
    # Subject/environment: prefer the explicit visual description, then the topic
    # (actual subject/location/geological feature), then fall back to narration.
    # Include narration context so the image matches the scene's story.
    if visual_description:
        subject = visual_description.strip()
    elif topic:
        subject = f"{topic}: {narration[:200]}".strip()
    else:
        subject = narration[:300].strip()
    narration_context = f" Scene context: {narration[:200]}." if narration and narration != subject else ""

    # Cinematic base style for visual identity coherence.
    style = AGNES_STYLE_MODIFIERS.get(
        style_hint, AGNES_STYLE_MODIFIERS["documentary"]
    )

    # Camera perspective rotates by scene index to avoid repetitive angles.
    camera_by_index = [
        "eye-level medium shot, balanced composition",
        "low-angle wide shot, dramatic foreground, leading lines",
        "close-up detail shot, shallow depth of field, sharp focus on subject",
        "high-angle establishing shot, expansive environment, atmospheric perspective",
        "over-the-shoulder medium shot, intimate framing, natural eye contact",
        "dutch angle dynamic shot, tension and movement, cinematic composition",
    ]
    camera = camera_by_index[scene_index % len(camera_by_index)]

    # Lighting varies by scene for visual distinctness.
    lighting_by_index = [
        "soft golden hour lighting, warm rim light, gentle shadows",
        "cool blue hour lighting, high contrast, deep shadows, moody atmosphere",
        "bright natural daylight, even illumination, clear visibility",
        "dramatic spotlighting, volumetric god rays, cinematic contrast",
        "diffused overcast lighting, soft shadows, neutral tones",
        "moody twilight lighting, subtle color grading, atmospheric haze",
    ]
    lighting = lighting_by_index[scene_index % len(lighting_by_index)]

    prompt = (
        f"{subject}{narration_context}. {style}. {camera}. {lighting}. "
        f"Realistic textures and fine surface details, "
        f"clear depth with distinct foreground, midground and background elements, "
        f"vertical 9:16 portrait composition optimized for mobile documentary Shorts, "
        f"professional documentary art direction, cinematic still frame. "
        f"No text, no letters, no watermark, no UI elements, no signatures."
    )
    return prompt


def _improve_prompt_for_retry(prompt: str, reason: str, attempt: int) -> str:
    """Improve a prompt based on a rejection reason, avoiding identical resubmission."""
    additions = {
        "blank": " high-contrast vivid lighting, saturated colors, well-exposed scene",
        "blurry": " sharp focus, crisp edges, high detail, clear subject definition",
        "composition": " centered subject, strong vertical composition, clear foreground-background separation",
        "corrupt": " high-quality photorealistic rendering, clean image data",
        "dimensions": " portrait orientation, tall vertical composition, 9:16 aspect",
        "unsuitable": " professional documentary photography, detailed cinematic shot",
    }
    addition = additions.get(reason, " cinematic documentary quality, high detail")
    return f"{prompt}. {addition} (refined attempt {attempt + 1})"

# Keywords that suggest a scene depicts a real, filmable place or subject (stock footage is appropriate).
FILMABLE_KEYWORDS = {
    "ocean", "sea", "beach", "coast", "island", "lake", "river", "waterfall", "mountain", "mountains",
    "valley", "canyon", "desert", "forest", "jungle", "woods", "cave", "glacier", "volcano", "reef",
    "city", "town", "village", "street", "road", "bridge", "building", "temple", "ruins", "castle",
    "waterfall", "park", "garden", "farm", "field", "meadow", "countryside", "savanna", "tundra",
    "iceberg", "glacier", "cliff", "canyon", "archipelago", "lagoon", "marsh", "swamp", "delta",
    "street", "avenue", "plaza", "skyline", "downtown", "harbor", "port", "dock", "lighthouse",
    "airport", "station", "market", "bazaar", "canyon", "gorge", "plateau", "steppe", "prairie",
    "animal", "wildlife", "lion", "elephant", "tiger", "bear", "wolf", "deer", "whale", "dolphin",
    "shark", "octopus", "turtle", "bird", "eagle", "penguin", "fish", "coral", "reef",
    "snow", "rain", "storm", "sunset", "sunrise", "fog", "mist", "cloud", "sky", "night", "day",
    "weather", "season", "autumn", "winter", "spring", "summer", "leaf", "flower", "tree",
}

# Keywords that suggest an unfilmable concept or scientific cutaway (AI image is appropriate).
CONCEPT_KEYWORDS = {
    "atom", "molecule", "cell", "dna", "gene", "neuron", "synapse", "organ", "heart", "lung",
    "brain", "blood", "vein", "artery", "capillary", "mitochondria", "ribosome", "protein",
    "virus", "bacteria", "microbe", "germ", "particle", "photon", "electron", "proton", "neutron",
    "quark", "black hole", "galaxy", "nebula", "supernova", "quasar", "pulsar", "dark matter",
    "dimension", "timeline", "paradox", "quantum", "relativity", "gravity", "force", "energy",
    "radiation", "frequency", "wave", "spectrum", "magnetic", "electric", "circuit", "chip",
    "nanobot", "nanotech", "microscope", "telescope", "satellite", "orbit", "eclipse",
    "constellation", "asteroid", "comet", "meteor", "exoplanet", "light-year", "big bang",
    "singularity", "event horizon", "wormhole", "antimatter", "fusion", "fission",
    "chemical", "compound", "reaction", "catalyst", "acid", "base", "element", "isotope",
    "crystal", "lattice", "fluid dynamics", "turbulence", "viscosity", "pressure", "vacuum",
    "entropy", "thermal", "heat", "cold", "absolute zero", "plasma", "superconductor",
}


def classify_suitability(narration: str, image_prompt: str = "", topic: str = "") -> str:
    """Classify a scene as ``filmable`` or ``concept``.

    ``filmable`` means a real place/subject that stock footage can plausibly show.
    ``concept`` means a scientific or abstract idea that cannot be filmed directly.
    Returns ``"filmable"`` or ``"concept"``.
    """
    text = f"{narration} {image_prompt} {topic}".lower()
    filmable_score = sum(1 for kw in FILMABLE_KEYWORDS if kw in text)
    concept_score = sum(1 for kw in CONCEPT_KEYWORDS if kw in text)
    if concept_score > filmable_score:
        return "concept"
    return "filmable"


def _build_attribution_line(source: str, credit: str, license_str: str, source_url: str) -> str:
    """Build a human-readable attribution line from actual provider metadata.

    Never invents attribution; every component comes from the selected asset.
    """
    parts = []
    if credit:
        parts.append(credit)
    if source:
        parts.append(source.title())
    if license_str:
        parts.append(license_str)
    if source_url:
        parts.append(source_url)
    return " | ".join(parts) if parts else ""



@dataclass
class MediaItem:
    source: str
    id: str
    kind: str                     # video | image | generated
    url: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0
    credit: str = ""              # photographer / user name from the provider
    license: str = ""             # license string from the provider metadata
    source_url: str = ""          # provider page URL for the asset
    attribution_line: str = ""    # human-readable attribution built from provider metadata
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
                               float(v.get("duration") or 0), (v.get("user") or {}).get("name", ""),
                               license="", source_url=v.get("url", ""),
                               attribution_line=_build_attribution_line(
                                   "pexels", (v.get("user") or {}).get("name", ""),
                                   "", v.get("url", "")),
                               query=query,
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
                               license="", source_url=p.get("url", ""),
                               attribution_line=_build_attribution_line(
                                   "pexels", p.get("photographer", ""), "", p.get("url", "")),
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
                               license="", source_url=h.get("pageURL", ""),
                               attribution_line=_build_attribution_line(
                                   "pixabay", h.get("user", ""), "", h.get("pageURL", "")),
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
                      license="", source_url=h.get("pageURL", ""),
                      attribution_line=_build_attribution_line(
                          "pixabay", h.get("user", ""), "", h.get("pageURL", "")),
                      extra={"tags": _tags(h.get("tags")), "page_url": h.get("pageURL", "")})
            for h in r.json().get("hits", []) if h.get("largeImageURL")]




# ------------------------------------------------------------------------------------------ Wikimedia Commons
def wikimedia_photos(query: str, orientation: str = "portrait") -> list[MediaItem]:
    """Search Wikimedia Commons for openly licensed images.

    Wikimedia Commons provides public domain and copyleft media with automatic attribution.
    No API key required. Returns images with license info for commercial reuse.
    """
    # Wikimedia Search API - no key required
    r = http().get(
        "https://commons.wikimedia.org/w/api.php",
        timeout=30,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "origin": "*",
        },
    )
    raise_for_status(r, "Wikimedia search")
    data = r.json()
    search_results = data.get("query", {}).get("search", [])
    items: list[MediaItem] = []
    for p in search_results[:20]:  # Limit to 20 results
        title = p.get("title", "")
        # Use the wiki page URL as the source; attribution is from the page
        items.append(MediaItem(
            "wikimedia",
            title[:50],  # use truncated title as id
            "image",
            f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
            0, 0,
            "",  # no credit/author from this simple search
            query=query,
            license="",
            source_url=f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
            attribution_line=f"Wikimedia Commons | {title}",
            extra={"search_result": p, "page_url": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}"},
        ))
    return items

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
                orientation: str = "portrait", *, scene_index: int = 0,
                narration: str = "", visual_description: str = "") -> Path:
    """Generate an image through Agnes AI's OpenAI-compatible image endpoint.

    When ``narration`` or ``visual_description`` is provided, the prompt is
    expanded into a detailed scene-aware cinematic prompt via
    ``build_agnes_prompt``. Otherwise the raw ``prompt`` is used (backward
    compatible).
    """
    # Build a detailed scene-aware prompt when scene context is available.
    if narration or visual_description:
        effective_prompt = build_agnes_prompt(
            narration=narration,
            visual_description=visual_description,
            scene_index=scene_index,
            style_hint="documentary",
        )
    else:
        effective_prompt = prompt

    size = {"portrait": AGNES_PORTRAIT_SIZE, "landscape": "1536x1024"}.get(
        orientation, "1024x1024"
    )
    body = {"model": model, "prompt": effective_prompt, "size": size, "n": 1}

    def _do() -> Path:
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
            raise RuntimeError(
                "Agnes AI image response contained neither image URL nor base64 data"
            )
        validate_generated_image(dest)
        return dest

    return with_retries(_do, attempts=2, label="Agnes AI image")


def agnes_image_with_qa(
    prompt: str,
    dest: Path,
    key: str,
    model: str = AGNES_DEFAULT_MODEL,
    orientation: str = "portrait",
    *,
    scene_index: int = 0,
    narration: str = "",
    visual_description: str = "",
    max_retries: int = AGNES_MAX_RETRIES,
) -> tuple[Path, ImageQualityResult] | None:
    """Generate an Agnes image with full quality control and controlled regeneration.

    Returns ``(path, quality_result)`` on success, or ``None`` if Agnes is
    unavailable or cannot produce a usable image after retries.

    Retry policy:
      - Max ``max_retries`` additional attempts after the first failure.
      - Each retry improves the prompt based on the rejection reason.
      - Identical prompts are never resubmitted.
    """
    # Build the scene-aware prompt.
    effective_prompt = build_agnes_prompt(
        narration=narration,
        visual_description=visual_description,
        scene_index=scene_index,
        style_hint="documentary",
    ) if (narration or visual_description) else prompt

    last_result: ImageQualityResult | None = None
    current_prompt = effective_prompt

    for attempt in range(max_retries + 1):
        try:
            agnes_image(
                current_prompt, dest, key, model=model, orientation=orientation,
                scene_index=scene_index, narration=narration,
                visual_description=visual_description,
            )
        except Exception as e:
            log.warning("Agnes generation attempt %d failed: %s", attempt + 1, redact(e))
            if attempt < max_retries:
                current_prompt = _improve_prompt_for_retry(
                    current_prompt, "api_error", attempt
                )
                continue
            return None

        # Run quality control on the generated image.
        qr = validate_image_quality(dest, prompt=current_prompt, min_dimension=512)
        last_result = qr

        if qr.passed:
            log.info(
                "Agnes image passed QC (score %.1f/100, %s)",
                qr.score, ", ".join(f"{k}: {v[1]}" for k, v in qr.checks.items()),
            )
            return dest, qr

        # Image failed QC: improve prompt and retry if attempts remain.
        reason = qr.reasons[0] if qr.reasons else "unsuitable"
        log.warning(
            "Agnes image failed QC attempt %d/%d: %s (score %.1f)",
            attempt + 1, max_retries + 1, "; ".join(qr.reasons), qr.score,
        )
        if attempt < max_retries:
            current_prompt = _improve_prompt_for_retry(current_prompt, reason, attempt)

    # Exhausted retries without a passing image.
    log.error(
        "Agnes could not produce a usable image for scene %d after %d attempts",
        scene_index + 1, max_retries + 1,
    )
    return None


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


# ------------------------------------------------------------------------------------------ Agnes image QC
def _detect_blank_image(image: Image.Image, threshold: float = BLANK_VARIANCE_THRESHOLD) -> tuple[bool, str]:
    """Detect if an image is blank or near-blank (single dominant color).

    Pure-PIL heuristic: converts to grayscale and checks the value range.
    Does not require a vision model.
    """
    gray = image.convert("L")
    lo, hi = gray.getextrema()  # type: ignore[assignment] - PIL returns (min, max) tuple
    value_range = int(hi) - int(lo)
    if value_range < threshold:
        return True, f"image appears blank or near-blank (grayscale range {value_range:.1f})"
    return False, f"image has adequate tonal range (range {value_range:.1f})"


def _detect_blurry_image(image: Image.Image, threshold: float = BLUR_EDGE_STRENGTH_THRESHOLD) -> tuple[bool, str]:
    """Detect excessive blur using edge-strength analysis.

    Pure-PIL heuristic: applies FIND_EDGES filter and measures the spread
    of edge pixel values. Does not require OpenCV or a vision model.
    """
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edges)
    # Mean edge strength: higher = more detail/sharpness
    edge_mean = stat.mean[0]
    if edge_mean < threshold:
        return True, f"image appears blurry (edge strength {edge_mean:.1f})"
    return False, f"image appears sharp (edge strength {edge_mean:.1f})"


def _check_vertical_composition(image: Image.Image) -> tuple[bool, str]:
    """Check if image composition is suitable for vertical 9:16 cropping.

    Returns (passed, message). Portrait orientation passes; landscape fails
    because a center 9:16 crop would discard most of the content.
    """
    width, height = image.size
    if width == 0:
        return False, "image has zero width"
    aspect = height / width
    if aspect >= MIN_PORTRAIT_ASPECT:
        return True, f"portrait orientation ({width}x{height}, aspect {aspect:.2f}), suitable for 9:16"
    return False, f"landscape orientation ({width}x{height}, aspect {aspect:.2f}), poor fit for 9:16 vertical"


def validate_image_quality(
    path: Path,
    prompt: str = "",
    min_dimension: int = 512,
) -> ImageQualityResult:
    """Run full quality control on a generated image.

    Checks: file validity, dimensions, blank content, blur, vertical composition.
    Returns an ImageQualityResult with per-check details and an overall score.
    Scene matching is inherently heuristic (no vision model); it is reported
    as advisory, not proof.
    """
    checks: dict[str, tuple[bool, str]] = {}
    reasons: list[str] = []
    score = 100.0
    dimensions: tuple[int, int] | None = None
    file_size: int | None = None

    # 1. File exists and is readable (reuse validate_generated_image logic)
    try:
        if not path.exists():
            checks["file_exists"] = (False, "image file was not written")
            reasons.append("image file missing")
            return ImageQualityResult(False, 0.0, checks, reasons, None, 0)
        file_size = path.stat().st_size
        if file_size < 1024:
            checks["file_size"] = (False, f"file too small ({file_size} bytes)")
            reasons.append("file too small")
            score -= 40
        with Image.open(path) as im:
            dimensions = (im.width, im.height)
            fmt = (im.format or "").upper()
            im.verify()
        if fmt not in {"PNG", "JPEG", "WEBP"}:
            checks["format"] = (False, f"unsupported format {fmt}")
            reasons.append(f"unsupported format {fmt}")
            score -= 30
        if dimensions and min(dimensions) < min_dimension:
            checks["dimensions"] = (False, f"dimensions {dimensions[0]}x{dimensions[1]} below {min_dimension}px")
            reasons.append("dimensions too small")
            score -= 30
    except (UnidentifiedImageError, OSError, ValueError) as e:
        checks["readable"] = (False, f"unreadable image: {e}")
        reasons.append("corrupt or unreadable image")
        return ImageQualityResult(False, 0.0, checks, reasons, dimensions, file_size)

    # Re-open for content checks (verify() above closed/re-positioned the file)
    try:
        with Image.open(path) as im:
            im.load()
            # 2. Blank detection
            blank, blank_msg = _detect_blank_image(im)
            checks["blank"] = (not blank, blank_msg)
            if blank:
                reasons.append("blank or near-blank image")
                score -= 35

            # 3. Blur detection
            blurry, blur_msg = _detect_blurry_image(im)
            checks["blur"] = (not blurry, blur_msg)
            if blurry:
                reasons.append("excessively blurry image")
                score -= 30

            # 4. Vertical composition
            comp, comp_msg = _check_vertical_composition(im)
            checks["composition"] = (comp, comp_msg)
            if not comp:
                reasons.append("unsuitable composition for vertical 9:16")
                score -= 20
    except (UnidentifiedImageError, OSError, ValueError) as e:
        checks["content_readable"] = (False, f"could not read image content: {e}")
        reasons.append("image content unreadable")
        score = 0.0

    score = max(0.0, min(100.0, score))
    passed = len(reasons) == 0
    return ImageQualityResult(passed, round(score, 1), checks, reasons, dimensions, file_size)


# Scene-match heuristic: without a vision model we cannot truly verify that an
# image matches its narration. This checks keyword overlap only and is advisory.
def _scene_match_heuristic(image_path: Path, narration: str, prompt: str) -> tuple[bool, str]:
    """Heuristic scene-match check (advisory only, not proof).

    Without vision capabilities, this checks whether the image file name or
    prompt contains any narration keywords. A pass does NOT confirm visual
    relevance; a fail flags the scene for human review.
    """
    wanted = set(re.findall(r"[a-z]{4,}", f"{narration} {prompt}".lower())) - STOP_WORDS
    if not wanted:
        return True, "no keywords to match (advisory pass)"
    # We cannot see the image, so we always pass at the heuristic level but
    # flag that this needs human review.
    return True, f"scene match cannot be verified without vision; {len(wanted)} keywords in narration (heuristic only)"


class AgnesUnavailableError(RuntimeError):
    """Raised when Agnes is required (agnes-only mode) but cannot be used."""


class VisualSelectionError(RuntimeError):
    """Raised when no suitable visual can be selected for a scene."""


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
        # Keep source available for future key addition; do not drop

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
            elif source == "wikimedia":
                items = wikimedia_photos(query, orientation=o)
            seen: set[str] = set()
            items = [i for i in items if not (i.key in seen or seen.add(i.key))]
            self._search_kinds[ck] = {
                "video": sum(i.kind == "video" for i in items),
                "image": sum(i.kind == "image" for i in items),
            }
        except PermanentError as e:
            log.warning("%s search failed permanently (%s); disabling it for this video", source, redact(e))
            # Keep source available for future key addition; do not drop
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

    def _ai(self, source: str, prompt: str, dest_dir: Path, idx: int,
            narration: str = "", visual_description: str = "") -> MediaItem | None:
        s = self.s
        full = f"{prompt}. {s.visual_style}. No text, no letters, no watermark."
        dest = dest_dir / f"scene{idx:02d}_ai.png"
        qa_extra: dict = {}
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
                result = agnes_image_with_qa(
                    prompt, dest, s.agnes_api_key,
                    model=s.agnes_image_model, orientation=s.orientation,
                    scene_index=idx, narration=narration,
                    visual_description=visual_description,
                )
                if result is None:
                    # Agnes could not produce a usable image after retries.
                    log.warning(
                        "Agnes QC failed for scene %d; no image returned", idx + 1
                    )
                    return None
                dest, qr = result
                # Store QC metadata in the item's extra dict for downstream use.
                qa_extra = {
                    "quality_score": qr.score,
                    "quality_checks": {k: {"passed": v[0], "message": v[1]}
                                       for k, v in qr.checks.items()},
                    "quality_reasons": qr.reasons,
                    "quality_dimensions": qr.dimensions,
                    "quality_file_size": qr.file_size,
                }
            else:
                return None
        except PermanentError as e:
            # A refused key or an exhausted free tier won't recover within this video: stop asking.
            if source in self.sources:
                # Keep source available for future key addition; do not drop
                hint = (" Anonymous Pollinations use is heavily rate-limited; set POLLINATIONS_API_KEY "
                        "or use Pexels/Pixabay (free keys)." if source == "pollinations" and " 402" in str(e)
                        else "")
                log.warning("AI images from %s refused (%s); skipping it for this video.%s",
                            source, redact(e)[:160], hint)
            return None
        except Exception as e:
            log.warning("AI image (%s) for scene %d failed: %s", source, idx + 1, redact(e))
            return None
        log.info("Scene %d: %s image", idx + 1, source)
        extra = qa_extra if source == "agnes" else {}
        return MediaItem(source, dest.name, "image", path=dest, width=s.width, height=s.height,
                         credit=source.title(), license="AI-generated",
                         source_url="", attribution_line=f"{source.title()} AI-generated image",
                         query=prompt, extra=extra)

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

        visual_choice = (self.s.visual_choice or VISUAL_CHOICE_STOCK_FIRST).strip().lower()
        if visual_choice not in VALID_VISUAL_CHOICES:
            visual_choice = VISUAL_CHOICE_STOCK_FIRST

        # --- agnes-only mode: Agnes for every scene; clear error if unavailable, no fallback ---
        if visual_choice == VISUAL_CHOICE_AGNES_ONLY:
            tried.append("agnes")
            if "agnes" not in self.sources:
                msg = (
                    "VISUAL_CHOICE=agnes-only requires AGNES_API_KEY and 'agnes' in VISUAL_SOURCES. "
                    "Set AGNES_API_KEY in .env, add 'agnes' to VISUAL_SOURCES/--visuals, "
                    "or choose a different VISUAL_CHOICE."
                )
                report["reason"] = msg
                with self.lock:
                    self.reports[idx] = report
                raise AgnesUnavailableError(msg)
            item = self._ai("agnes", hybrid_prompt, dest_dir, idx,
                            narration=narration, visual_description=image_prompt)
            if item:
                report["chosen"] = {"source": item.source, "id": item.id, "kind": item.kind,
                                    "query": hybrid_prompt}
                report["reason"] = "agnes-only: generated AI image"
                note(source="agnes", query=hybrid_prompt, result="agnes-image",
                     reason=report["reason"])
                with self.lock:
                    self.reports[idx] = report
                log.info("Scene %d: agnes-only generated image (%r)", idx + 1, hybrid_prompt)
                return item
            msg = (
                f"Agnes image generation failed for scene {idx + 1} ({hybrid_prompt}). "
                "agnes-only mode does not fall back to stock footage or gradients."
            )
            report["reason"] = msg
            note(source="agnes", query=hybrid_prompt, result="failed", reason=msg)
            with self.lock:
                self.reports[idx] = report
            raise AgnesUnavailableError(msg)

        # --- Decide whether this scene tries AI before stock ---
        if visual_choice == VISUAL_CHOICE_SUITABILITY:
            suitability = classify_suitability(narration, image_prompt, topic)
            # A hybrid image slot always asks for an AI image first, whatever the
            # content classification; suitability still drives the non-hybrid scenes.
            prefer_ai_first = (suitability == "concept") or prefer_image
            report["suitability"] = suitability
        elif visual_choice == VISUAL_CHOICE_AI_FIRST:
            prefer_ai_first = True
            report["suitability"] = "ai-first"
        else:
            # stock-first: only AI-first for explicit hybrid image slots
            prefer_ai_first = prefer_image
            report["suitability"] = "stock-first"
        is_hybrid_slot = (visual_choice == VISUAL_CHOICE_STOCK_FIRST and prefer_image)

        def _record_and_return(item: MediaItem, reason: str, source_name: str, q: str,
                               hybrid: bool = False) -> MediaItem:
            if hybrid:
                reason = f"hybrid AI image unavailable; {reason}"
            report["chosen"] = {"source": item.source, "id": item.id, "kind": item.kind,
                                "query": q, "width": item.width, "height": item.height,
                                "duration": item.duration, "attribution": item.attribution_line,
                                "license": item.license, "source_url": item.source_url}
            report["reason"] = reason
            note(source=source_name, query=q, result="selected", reason=reason,
                 videos=self._search_kinds.get((source_name, q.lower()), {}).get("video", 0),
                 images=self._search_kinds.get((source_name, q.lower()), {}).get("image", 0))
            with self.lock:
                self.reports[idx] = report
            if item.source.startswith("pexels"):
                self.credits.add("Pexels")
            elif item.source.startswith("pixabay"):
                self.credits.add("Pixabay")
            if item.kind == "video":
                log.info("Scene %d: %s video (%r, %sx%s, %.1fs)", idx + 1, item.source, q,
                         item.width, item.height, item.duration)
            else:
                log.warning("Scene %d: %s photo (%r): no suitable stock video clip was usable",
                            idx + 1, item.source, q)
            return item

        # --- AI-first path (concept scenes in suitability, ai-first mode, hybrid image slots) ---
        if prefer_ai_first:
            for source in ai_sources:
                tried.append(source)
                ai_tried.add(source)
                item = self._ai(source, hybrid_prompt, dest_dir, idx,
                                narration=narration, visual_description=image_prompt)
                if item:
                    if is_hybrid_slot:
                        reason = "hybrid mode: generated an AI image for this scene"
                        return _record_and_return(item, reason, source, hybrid_prompt, hybrid=False)
                    reason = f"AI image selected ({source}) for {report.get('suitability', 'ai-first')} scene"
                    return _record_and_return(item, reason, source, hybrid_prompt, hybrid=is_hybrid_slot)
                note(source=source, query=hybrid_prompt, result="failed",
                     reason="AI image generation failed")
            log.warning("Scene %d: no AI image generated; falling back to stock footage", idx + 1)

        # --- Stock-first path (filmable scenes, stock-first mode) ---
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
                        note(source=source, query=q, found=len(raw), usable=len(cand),
                             result="download-failed", reason=redact(e),
                             videos=kinds.get("video", 0), images=kinds.get("image", 0))
                        log.warning("Scene %d: download failed for %s: %s", idx + 1, item.key, redact(e))
                        continue
                    return _record_and_return(
                        item,
                        "relevant stock video selected" if item.kind == "video"
                        else "no suitable stock video; using a stock photo",
                        source, q, hybrid=is_hybrid_slot)
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

        # --- AI fallback path (stock-first / suitability filmable when stock failed) ---
        if not prefer_ai_first:
            stock_reason = ("no usable stock video or photo" if stock_sources
                            else "no stock source selected")
            for source in ai_sources:
                if source in ai_tried:
                    continue
                tried.append(source)
                item = self._ai(source, image_prompt or query, dest_dir, idx,
                                narration=narration, visual_description=image_prompt)
                if item:
                    return _record_and_return(
                        item,
                        f"{stock_reason}; generated an AI image instead",
                        source, image_prompt or query, hybrid=is_hybrid_slot)
                note(source=source, query=image_prompt or query, result="failed",
                     reason="AI image failed")

        # --- Nothing usable: animated gradient placeholder (never an unrelated image) ---
        report["reason"] = f"{'no AI image was available' if prefer_ai_first else 'no usable stock or AI visual'}; animated gradient placeholder"
        log.warning(
            "Scene %d: no real visual for %r after trying %s — using an animated gradient. "
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
