"""The production line: topic -> script -> review -> voice -> footage -> captions -> render -> upload.

``Studio.draft`` stops after the script (so it can be read and edited first); ``Studio.make`` runs the
whole line, optionally followed by translated copies that reuse the same footage."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from . import analytics, ffmpeg, notify, research, topics, tts, youtube
from .config import MAX_TARGET_SECONDS, Settings, duration_window, short_duration_bounds
from .history import History, now_iso
from .llm import LLM, LLMConfigError, LLMNotConfigured
from .media import AI_SOURCES, MediaItem, Visuals
from .overlays import ChatMessage, build_overlays
from .quality import QualityGate
from .render import extract_frame, render_video
from .script import (
    CARD_STYLES,
    DEFAULT_CAST,
    MULTI_SPEAKER,
    SCRIPT_SCHEMA,
    Script,
    ScriptQualityError,
    load_script,
    review_script,
    translate_script,
    validate_opening_hook,
    write_script,
)
from .timing import caption_chunks, scene_timeline, srt, tokenize
from .topics import Topic, pick_topic
from .utils import redact, slugify, truncate_words

log = logging.getLogger("purffle")

END_PAD = 0.6          # seconds of breathing room after the last word
REUSABLE_SOURCES = {"generated", "local", "openai-images", "ai", "dalle", "gpt-image", "pollinations", "agnes"}

# Rough share of the work done when each stage starts (for progress bars).
STAGES = {"topic": 2, "script": 8, "review": 20, "voice": 32, "footage": 45, "captions": 62, "render": 72,
          "check": 92, "upload": 95, "translate": 97, "done": 100}

ProgressFn = Callable[[str, int, str], None]


@dataclass
class Result:
    ok: bool
    title: str = ""
    folder: Path | None = None
    video: Path | None = None
    status: str = ""
    youtube_id: str | None = None
    publish_at: str | None = None
    seconds: float = 0.0
    error: str = ""
    record_id: int | None = None
    score: int | None = None
    language: str = ""
    variants: list[Result] = field(default_factory=list)


def quota_day_start() -> str:
    """YouTube's daily quota resets at midnight Pacific time."""
    from zoneinfo import ZoneInfo
    pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    start = pt.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(timezone.utc).isoformat(timespec="seconds")


def seconds_until_quota_reset() -> float:
    from zoneinfo import ZoneInfo
    pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    nxt = (pt.replace(hour=0, minute=0, second=0, microsecond=0)).timestamp() + 86400
    return max(60.0, nxt - pt.timestamp() + 120)


def final_title(script: Script, settings: Settings | None = None) -> str:
    t = script.title
    if settings is not None and not settings.vertical:
        return truncate_words(t, 100)  # a 16:9 video is not a Short
    return f"{t} #shorts" if len(t) + 8 <= 100 else truncate_words(t, 100)


def final_description(script: Script, credits: set[str], settings: Settings) -> str:
    parts = [script.description.strip()] if script.description.strip() else []
    tags = script.hashtags if settings.vertical else [h for h in script.hashtags if h.lower() != "#shorts"]
    parts.append(" ".join(tags))
    if settings.credit_footage and credits:
        parts.append("Footage: " + ", ".join(sorted(credits)))
    return "\n\n".join(p for p in parts if p)


def _unique_folder(root: Path, name: str) -> Path:
    folder = root / name
    n = 2
    while folder.exists():
        folder = root / f"{name}-{n}"
        n += 1
    return folder


def _media_for(timeline, media_map: dict[int, MediaItem]) -> list[MediaItem]:
    """Reuse the original video's footage for a translated copy (its scenes may merge differently)."""
    keys = sorted(media_map)
    out = []
    for st in timeline:
        k = max([k for k in keys if k <= st.index], default=keys[0])
        out.append(media_map[k])
    return out


class Studio:
    def __init__(self, settings: Settings, progress: ProgressFn | None = None):
        self.s = settings
        self.history = History(settings.data_path / "history.db")
        self._llm: LLM | None = None
        self.progress = progress
        self._stats_synced = 0.0
        self._warned_no_llm = False
        self._refused_on = ""  # the quota day on which YouTube itself refused an upload

    def _step(self, stage: str, message: str = "") -> None:
        if self.progress:
            try:
                self.progress(stage, STAGES.get(stage, 0), message)
            except Exception:  # a progress display must never break a video
                log.debug("progress callback failed", exc_info=True)

    @property
    def llm(self) -> LLM | None:
        if self.s.offline:
            return None
        if self._llm is None:
            self._llm = LLM(self.s)
            log.info("LLM: %s%s", self._llm.label,
                     f" (fallbacks: {', '.join(str(p) for p in self._llm.providers[1:])})"
                     if len(self._llm.providers) > 1 else "")
        return self._llm

    def _llm_or_none(self) -> LLM | None:
        """The LLM when one is configured, else None (write a basic script offline instead of failing).

        A genuinely misconfigured provider (unknown name, custom without a URL, a named provider with no
        key) still raises so the user can fix it; only "nothing is configured at all" degrades gracefully."""
        try:
            return self.llm
        except LLMNotConfigured as e:
            if not self._warned_no_llm:
                self._warned_no_llm = True
                log.warning("No LLM is configured — writing a basic script offline. For AI scripts set one LLM "
                            "key in .env (or run Ollama). Details: %s", e)
            return None

    # ------------------------------------------------------------------ voice with fallbacks
    def _speak(self, script: Script, work: Path, s: Settings) -> tts.Speech:
        engines = [s.tts_engine]
        if s.tts_engine != "edge":
            engines.append("edge")
        else:
            engines.append("system")
        if s.offline:
            engines += ["system", "silent"]
        lines = [(sc.speaker, sc.narration) for sc in script.scenes] if (
            script.multi_speaker or script.style in MULTI_SPEAKER) else None
        last: Exception | None = None
        for eng in dict.fromkeys(engines):
            try:
                cfg = s if eng == s.tts_engine else replace(s, tts_engine=eng, tts_voice="", tts_voice_b="")
                if lines:
                    return tts.synthesize_lines(lines, work / "voice", cfg)
                return tts.synthesize(script.narration, work, cfg)
            except Exception as e:
                last = e
                log.warning("Voice engine '%s' failed: %s", eng, e)
        raise RuntimeError(f"All voice engines failed: {last}")

    # ------------------------------------------------------------------ script only
    def draft(self, topic: str | None = None, *, source: str | None = None, style: str | None = None,
              url: str | None = None, document: str | Path | None = None) -> tuple[Script, Topic, str]:
        """Pick a topic, write the script and let the Script Doctor improve it. Nothing is rendered."""
        s = self.s
        self._step("topic")
        if url:
            pick = topics.from_url(url)
        elif document:
            pick = topics.from_document(document)
        else:
            pick = pick_topic(s, self.history, explicit=topic, source=source)
        if topic and (url or document):
            pick.subject = f"{topic} (source: {pick.subject})"
        log.info("Topic [%s]: %s", pick.source, pick.subject)
        try:
            self._step("script", pick.subject)
            insights = analytics.insights(self.history) if s.learn_from_stats else ""
            llm = self._llm_or_none()
            script_started = time.perf_counter()
            script = write_script(llm, pick.subject, s, source=pick.source, avoid=self.history.recent_titles(),
                                  context=pick.context, style=style or pick.style or None, insights=insights)
            script_took = time.perf_counter() - script_started
            log.info("Initial script generation: %.2fs", script_took)

            # In score-only production mode, hook quality is advisory: never discard
            # an otherwise renderable script because a hook rewrite did not satisfy QC.
            if llm is not None:
                hook_issue = validate_opening_hook(script)
                if hook_issue and s.script_review == "rewrite":
                    log.warning("Initial script hook failed: %s", hook_issue)
                    script = self._rewrite_hook_only(llm, script, s, hook_issue)
                    hook_issue = validate_opening_hook(script)
                if hook_issue:
                    log.warning("Initial hook QC advisory; continuing to render: %s", hook_issue)

            if llm is not None and s.script_review != "off":
                self._step("review", "Script Doctor")
                review_started = time.perf_counter()
                script = review_script(llm, script, s)
                review_took = time.perf_counter() - review_started
                log.info("Script Doctor: %.2fs", review_took)
                # Revalidate hook after rewrite. In score-only mode this is advisory,
                # not a reason to stop the production run.
                hook_issue = validate_opening_hook(script)
                if hook_issue and s.script_review == "rewrite":
                    log.warning("Post-Doctor script hook failed: %s", hook_issue)
                    script = self._rewrite_hook_only(llm, script, s, hook_issue)
                    hook_issue = validate_opening_hook(script)
                if hook_issue:
                    log.warning("Post-Doctor hook QC advisory; continuing to render: %s", hook_issue)

            if llm is not None and s.research_verify:
                self._step("research", "fact verification")
                claims = research.extract_claims_from_script(script, s.language)
                report = research.verify_script_claims(claims, pick, llm)
                script.verification_report = report.to_dict()
                log.info("Fact verification: %d claims, %s", len(report.claims), report.status)
                # Unsupported claims are rewritten and verified again. Unresolved claims
                # are never silently relabeled as verified.
                if report.status == research.NEEDS_HUMAN_REVIEW:
                    unresolved = [c for c in report.claims if c.status == research.NEEDS_HUMAN_REVIEW]
                    if unresolved:
                        repaired = self._rewrite_unverified_claims(
                            llm, script, unresolved, s, pick)
                        if repaired is not None:
                            script = repaired
                            claims = research.extract_claims_from_script(script, s.language)
                            report = research.verify_script_claims(claims, pick, llm)
                            script.verification_report = report.to_dict()
                            log.info(
                                "Fact verification after claim rewrite: %d claims, %s",
                                len(report.claims), report.status,
                            )
                # In rewrite/strict mode unresolved claims or extraction failures block
                # production. In score-only production mode, preserve the report and
                # warnings but still render a reviewable draft; it is explicitly marked
                # unverified in metadata.
                if s.script_review == "rewrite":
                    if report.status == research.NEEDS_HUMAN_REVIEW or report.llm_extraction_failed:
                        raise RuntimeError(
                            "Fact verification unresolved: claims need human review before rendering. "
                            f"Checked by {report.checked_by}; "
                            f"{'LLM claim extraction failed; ' if report.llm_extraction_failed else ''}"
                            f"{len(report.claims)} claim(s) unresolved."
                        )
                else:
                    if report.status == research.NEEDS_HUMAN_REVIEW or report.llm_extraction_failed:
                        detail = (
                            f"Fact verification unresolved: {len(report.claims)} claim(s) need review; "
                            f"checked by {report.checked_by}; extraction_failed={report.llm_extraction_failed}."
                        )
                        log.warning("%s Continuing with an unverified draft for human review.", detail)
        except BaseException:
            if pick.idea_id:
                self.history.release_idea(pick.idea_id)
            raise
        return script, pick, llm.label if llm else "offline"

    # ------------------------------------------------------------------ hook-only targeted rewrite
    def _rewrite_hook_only(self, llm, script: Script, settings, hook_issue: str) -> Script:
        """Targeted rewrite that fixes only the opening hook.

        Preserves all verified factual content, target duration, scene count,
        and scene structure. Only the first scene's narration is changed.
        """
        from .script import build_prompt, normalize

        first = script.scenes[0]
        system, user = build_prompt(
            script.topic, script.style, settings, [],
            context="",
            insights=(
                f"REWRITE ONLY THE OPENING HOOK. Problem: {hook_issue}. "
                f"Current opening: '{first.narration}'. "
                "Replace ONLY the first scene's narration with a specific, "
                "scroll-stopping opening that names the subject immediately. "
                "Do not use greetings, 'Did you know?', or throat-clearing. "
                "Keep all other scenes, facts, search queries, image prompts, "
                "and structure exactly as-is. Return the complete script JSON."
            ),
        )
        data = llm.complete_json(system, user, SCRIPT_SCHEMA, max_tokens=8000)
        rewritten = normalize(data, script.topic, script.style, script.language)
        # Preserve everything except the fixed opening narration
        for i, scene in enumerate(rewritten.scenes):
            if i == 0:
                continue  # new opening from the rewrite
            if i < len(script.scenes):
                scene.narration = script.scenes[i].narration
                scene.search_query = script.scenes[i].search_query
                scene.image_prompt = script.scenes[i].image_prompt
                scene.visual_description = script.scenes[i].visual_description
                scene.visual_source = script.scenes[i].visual_source
                scene.attribution = script.scenes[i].attribution
        return rewritten

    # ------------------------------------------------------------------ unverified-claim repair
    def _rewrite_unverified_claims(self, llm, script: Script, unresolved: list,
                                   settings: Settings, pick) -> Script | None:
        """Rewrite only the scenes whose claims lack source support.

        Verified scenes and every other story element are preserved verbatim;
        only the narration of flagged scenes is replaced with wording that can
        be backed by the gathered sources (unsupported specificity is dropped,
        never invented). Returns None when no trustworthy rewrite is available,
        so the caller keeps blocking instead of rendering unresolved claims.
        """
        from .script import SCRIPT_SCHEMA, build_prompt, normalize

        scene_indices = sorted(
            {c.scene_index for c in unresolved if c.scene_index is not None})
        if not scene_indices:
            return None
        scenes = script.scenes
        flagged = "\n".join(f"  Scene {i + 1}: {scenes[i].narration}" for i in scene_indices)
        # List the source titles and URLs so the LLM can ground its rewrite
        # in actual, citable material instead of inventing claims.
        source_lines = []
        for ref in (pick.source_references or []):
            title = ref.get("title", "")
            url = ref.get("url", "")
            if title or url:
                source_lines.append(f"  - {title}: {url}")
        source_block = "\n".join(source_lines)
        system, user = build_prompt(
            script.topic, script.style, settings, [],
            context=pick.context or "",
            insights=(
                "REWRITE ONLY THE SCENES LISTED BELOW. Every other scene, the hook, "
                "the sources, scene order and length must stay unchanged.\n"
                + flagged
                + "\n\nAvailable sources (ground your claims in these only):\n"
                + source_block
                + "\n\nReplace each listed narration with wording whose factual claims "
                "are supported by the cited background sources: drop unsupported "
                "statistics, names, dates and certainty; keep what the sources confirm. "
                "Do not invent facts, studies or quotes. Return the complete script JSON."
            ),
        )
        try:
            data = llm.complete_json(system, user, SCRIPT_SCHEMA, max_tokens=8000)
        except Exception as e:
            log.warning("Unverified-claim rewrite failed: %s", redact(e))
            return None
        if not isinstance(data, dict):
            return None
        try:
            rewritten = normalize(data, script.topic, script.style, script.language)
        except ValueError as e:
            log.warning("Unverified-claim rewrite could not be normalized: %s", e)
            return None
        if len(rewritten.scenes) != len(scenes):
            log.warning("Unverified-claim rewrite changed the scene count; keeping the draft")
            return None
        # Adopt only the flagged scenes' narration; everything else is preserved.
        for i in scene_indices:
            scenes[i].narration = rewritten.scenes[i].narration
        return script

    # ------------------------------------------------------------------ one video
    def make(self, topic: str | None = None, *, source: str | None = None, upload: bool | None = None,
             style: str | None = None, script_file: str | Path | None = None, script: Script | None = None,
             url: str | None = None, document: str | Path | None = None,
             languages: list[str] | None = None) -> Result:
        t0 = time.time()
        s = self.s
        upload = s.upload if upload is None else upload
        languages = s.also_languages if languages is None else languages
        state: dict = {}
        pick: Topic | None = None
        try:
            if script is not None or script_file:
                if script is None:
                    script = load_script(script_file, s)
                pick, writer = Topic(script.topic, "script"), ("script file" if script_file else "editor")
                log.info("Script from %s (%d scenes): %s", script_file or "editor", len(script.scenes), script.title)
            else:
                script, pick, writer = self.draft(topic, source=source, style=style, url=url, document=document)
            return self.produce(script, pick, writer, upload=upload, t0=t0, languages=languages, state=state)
        except (youtube.NotAuthorized, LLMConfigError, KeyboardInterrupt):
            raise  # setup problems stop the run instead of failing every video the same way
        except Exception as e:
            if pick is not None and pick.idea_id and not state.get("record_id"):
                self.history.release_idea(pick.idea_id)
            log.error("Video failed: %s", redact(e), exc_info=log.isEnabledFor(logging.DEBUG))
            notify.send(s, "failed", title=getattr(script, "title", "") or (pick.subject if pick else ""),
                        detail=redact(e))
            return Result(False, folder=state.get("folder"), status="failed", error=redact(e),
                          seconds=time.time() - t0)

    def produce(self, script: Script, pick: Topic, writer: str, *, upload: bool, t0: float | None = None,
                languages: list[str] | None = None, media_map: dict[int, MediaItem] | None = None,
                credits: set[str] | None = None, parent_id: int | None = None, settings: Settings | None = None,
                state: dict | None = None) -> Result:
        t0 = t0 or time.time()
        state = state if state is not None else {}
        s = settings or self.s
        if script.language and script.language != s.language:
            s = replace(s, language=script.language, tts_voice="", tts_voice_b="")

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}_{slugify(script.title)}" + (f"-{s.language}" if parent_id else "")
        folder = _unique_folder(s.out_path, name).resolve()  # absolute: history outlives cwd
        state["folder"] = folder
        work = folder / "work"
        work.mkdir(parents=True, exist_ok=True)
        (folder / "script.json").write_text(json.dumps(script.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            self._step("voice", f"{s.tts_engine} voice")
            voice_started = time.perf_counter()
            speech = self._speak(script, work, s)
            voice_took = time.perf_counter() - voice_started
            total = round(speech.duration + END_PAD, 3)
            # voice_took is wall-clock synthesis time (retries/network included);
            # speech.duration is the actual audio length. total = audio + pad is the
            # single authoritative render duration used by the timeline, captions and FFmpeg.
            log.info(
                "TTS: audio %.1fs + %.1fs pad = %.1fs total (synthesis took %.2fs wall)",
                speech.duration, END_PAD, total, voice_took,
            )

            # Pre-render duration guard: if the spoken audio already exceeds the
            # QC maximum, attempt a meaning-preserving script reduction before
            # rendering anything. This catches overlong scripts early instead of
            # rendering a too-long video that QC will reject.
            max_dur = float(duration_window(s.target_seconds)[1])
            # Hard maximum for the 40-second Shorts format: the rendered video must
            # never exceed the target (40s). No tolerance is added above the cap.
            short = short_duration_bounds(s.target_seconds)
            if short is not None:
                max_dur = min(max_dur, short[1])
            if total > max_dur:
                log.warning(
                    "Narration too long (%.1fs incl. pad; max %.1fs), attempting script reduction",
                    total, max_dur,
                )
                for reduce_attempt in range(2):
                    if s.llm_provider is None or s.llm_provider == "off":
                        break
                    try:
                        llm = self._llm_or_none()
                        if llm is None:
                            break
                        script = review_script(
                            llm, script, replace(s, script_review="rewrite",
                                                 script_quality_threshold=s.script_quality_threshold),
                        )
                        speech = self._speak(script, work, s)
                        total = round(speech.duration + END_PAD, 3)
                        if total <= max_dur:
                            log.info(
                                "Script reduced to %.1fs (incl. pad); within limits",
                                total,
                            )
                            break
                    except ScriptQualityError as e:
                        log.warning("Script reduction failed: %s", e)
                        break
                else:
                    # Loop exhausted without breaking → still too long
                    pass

                if total > max_dur:
                    raise RuntimeError(
                        f"Script narration ({speech.duration:.1f}s + {END_PAD}s pad = {total:.1f}s) "
                        f"exceeds the {max_dur}s maximum for a {s.target_seconds}s target. "
                        f"Reduce script length or increase TARGET_SECONDS."
                    )
            chat = script.style == "chat"
            timeline = scene_timeline([sc.narration for sc in script.scenes], speech.words, total, s.language,
                                      starts=speech.line_starts, min_len=4.5 if chat else 1.2)
            T = s.transition_seconds if s.transition != "none" else 0.0

            self._step("footage", f"{len(timeline)} scenes")
            selection: list[dict] = [{} for _ in timeline]
            if media_map:
                media = _media_for(timeline, media_map)
                credits = set(credits or ())
            else:
                media_started = time.perf_counter()
                jobs = [(script.scenes[st.index].search_query, script.scenes[st.index].image_prompt,
                         st.duration + T, script.scenes[st.index].narration) for st in timeline]
                visuals = Visuals(s, self.history.used_media())
                media = visuals.for_scenes(jobs, script.topic, work / "media")
                credits = visuals.credits
                selection = [visuals.reports.get(i, {}) for i in range(len(media))]
                videos = sum(m.kind == "video" for m in media)
                generated = sum(m.kind == "image" and m.source in AI_SOURCES for m in media)
                photos = sum(m.kind == "image" and m.source not in AI_SOURCES for m in media)
                fallbacks = sum(m.kind == "generated" for m in media)
                media_took = time.perf_counter() - media_started
                log.info("Visuals: %d video, %d generated image%s, %d stock photo%s, %d fallback background%s "
                         "(mode: %s; sources: %s; %.2fs)",
                         videos, generated, "" if generated == 1 else "s",
                         photos, "" if photos == 1 else "s",
                         fallbacks, "" if fallbacks == 1 else "s",
                         s.visual_mode, ", ".join(s.visual_sources), media_took)
                missing = [m for m in media if m.kind == "generated"]
                if missing:
                    log.warning("%d of %d scenes have no real visual — those scenes use an animated gradient. "
                                "Add a free Pexels/Pixabay key, an image provider, or clips in MEDIA_DIR.",
                                len(missing), len(media))
            by_scene = {st.index: m for st, m in zip(timeline, media)}

            self._step("captions")
            chunks = caption_chunks(speech.words, s.caption_max_words, total=total)
            labels = ({"A": script.speaker_name("A"), "B": script.speaker_name("B")}
                      if script.style == "dialogue" and script.multi_speaker else None)
            messages = ([ChatMessage(sc.speaker, sc.narration, st) for sc, st in zip(script.scenes, speech.line_starts)]
                        if chat and speech.line_starts else None)
            card, card_until, on_screen = None, 0.0, chunks
            if script.style in CARD_STYLES:
                # The post card shows the title while it is read; captions start with the story.
                n0 = len(tokenize(script.scenes[0].narration, s.language))
                if 0 < n0 < len(speech.words):
                    card_until = min(speech.words[n0 - 1].end + 0.35, total * 0.6)
                    community, user = (script.cast + DEFAULT_CAST["reddit"][len(script.cast):])[:2]
                    card = (community, user, script.scenes[0].narration)
                    on_screen = caption_chunks(speech.words[n0:], s.caption_max_words, total=total)
            plan = build_overlays(s, on_screen, total, script.hook_text, work, labels=labels, chat=messages,
                                  contact=script.speaker_name("A"), card=card, card_until=card_until)

            self._step("render", f"{s.width}x{s.height}")
            video = folder / ("short.mp4" if s.vertical else "video.mp4")
            render_started = time.perf_counter()
            render_video(s, media, [st.duration for st in timeline], speech.audio, total, plan, video, work)
            render_took = time.perf_counter() - render_started
            log.info("Render: %.2fs", render_took)

            self._step("check")
            info = ffmpeg.probe(video)
            if (info["width"], info["height"]) != s.resolution or abs(info["duration"] - total) > 0.5:
                raise RuntimeError(f"Render check failed: got {info['width']}x{info['height']} "
                                   f"{info['duration']:.2f}s, expected {s.width}x{s.height} {total:.2f}s")
            if not info["has_audio"]:
                raise RuntimeError("Render check failed: the rendered video has no audio track")
            # Report how the narration length compares with the requested target so a drift is visible
            # instead of silent (the single timeline makes this the real duration of the video).
            deviation = round(info["duration"] - s.target_seconds, 2)
            if s.target_seconds and abs(deviation) > max(3.0, s.target_seconds * 0.25):
                log.warning("Narration is %.1fs, about %.0f%% off the requested %ds target. Adjust the "
                            "script length (TARGET_SECONDS or the scene count) if this matters.",
                            info["duration"], abs(deviation) / s.target_seconds * 100, s.target_seconds)

            extract_frame(video, min(1.2, total / 3), folder / "cover.jpg")
            (folder / "captions.srt").write_text(srt(chunks), encoding="utf-8")

            title = final_title(script, s)
            description = final_description(script, credits, s)
            research_warnings = [
                "AI-generated factual claims have not been independently verified; check each claim against the cited source."
            ]
            if not pick.source_references:
                research_warnings.append("No external source URL is recorded; verify claims manually before publication.")
            research = {
                "source": pick.source,
                "references": pick.source_references,
                "notes": pick.context[:6000],
                "verification_status": "needs_human_review",
                "warnings": research_warnings,
            }
            script_review = {
                "retention_score": script.score,
                "issues": list(script.review),
                "factual_verification": "not_performed",
                "warnings": research_warnings,
            }
            if script.verification_report:
                script_review["factual_verification"] = script.verification_report.get("status", "not_performed")
                script_review["claims"] = script.verification_report.get("claims", [])
            meta = {
                "title": title, "description": description, "tags": script.tags,
                "category_id": script.category_id, "language": s.language,
                "topic": script.topic, "source": pick.source, "style": script.style,
                "duration": round(info["duration"], 2), "resolution": f"{s.width}x{s.height}",
                "requested_duration": s.target_seconds, "actual_duration": round(info["duration"], 2),
                "duration_deviation": deviation, "assembled_duration": round(timeline[-1].end, 2),
                "llm": writer, "score": script.score, "review": script.review, "cast": script.cast,
                "research": research, "review_results": script_review, "approval_status": "pending",
                "voice": f"{speech.engine}:{speech.voice}", "timing": speech.timing,
                "visuals": [{"source": m.source, "id": m.id, "kind": m.kind, "query": m.query,
                             "credit": m.credit, "license": m.license, "source_url": m.source_url,
                             "attribution": m.attribution_line} for m in media],
                "visual_mode": s.visual_mode,
                "visual_choice": s.visual_choice,
                "visual_selection": selection,
                "verification_report": script.verification_report,
                "parent_id": parent_id, "created_at": now_iso(),
            }
            (folder / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

            # Save attribution/credits file alongside the video.
            credits_lines = []
            for m in media:
                if m.attribution_line:
                    credits_lines.append(f"{m.query}: {m.attribution_line}")
            if credits_lines:
                (folder / "credits.txt").write_text("\n".join(credits_lines) + "\n", encoding="utf-8")

            qc_report = QualityGate(s).check(Result(
                ok=True, video=video, folder=folder, status="rendered"))
            if not qc_report.passed:
                log.warning("QC gate reported %d automated failure(s): %s",
                            len(qc_report.automated_failures), qc_report.summary)
                for adv in qc_report.heuristic_advisories:
                    log.info("QC advisory (%s): %s", adv.name, adv.message)

            rid = self.history.add_video(
                subject=pick.subject, source=pick.source, topic=script.topic, style=script.style,
                language=s.language, title=title, description=description, tags=script.tags,
                folder=str(folder), video_path=str(video), duration=info["duration"],
                llm=meta["llm"], voice=meta["voice"], status="rendered", score=script.score, parent_id=parent_id,
                approval_status="pending", research_json=json.dumps(research, ensure_ascii=False),
                review_json=json.dumps(script_review, ensure_ascii=False))
            state["record_id"] = rid
            if not media_map:
                self.history.mark_media([m.key for m in media if m.source not in REUSABLE_SOURCES])
            if pick.key and not parent_id:
                self.history.mark_topic(pick.key)
            if pick.idea_id and not parent_id:
                self.history.link_idea(pick.idea_id, rid)

            took = time.time() - t0
            log.info("Rendered %s (%.1fs video) in %.0fs total -> %s", title, info["duration"], took, video)
            result = Result(True, title, folder, video, "rendered", seconds=took, record_id=rid,
                            score=script.score, language=s.language)
            if upload:
                # Prevent duplicate publishing: check YouTube for an existing upload
                # of this slot's video before uploading again.
                self._step("upload")
                self._check_duplicate_upload(s, title, video)
                self.upload_record(rid, result)
            else:
                notify.send(s, "rendered", title=title, detail=str(video))

            for lang in [x for x in dict.fromkeys(languages or []) if x != s.language]:
                self._step("translate", lang)
                result.variants.append(self._translation(script, pick, writer, lang, s, by_scene, credits,
                                                         rid, upload))
            self._step("done", title)
            return result
        finally:
            if not os.getenv("KEEP_WORK"):
                shutil.rmtree(work, ignore_errors=True)

    def _check_duplicate_upload(self, settings, title: str, video: Path) -> None:
        """Before uploading, check YouTube for an existing video matching this slot.

        Never uploads the same slot twice. Preserves existing video IDs and
        successful upload state.
        """
        try:
            yt = youtube.service(settings, interactive=False)
            marker = f"{title} #shorts"
            existing = youtube.find_existing_video(yt, marker)
            if existing:
                log.warning(
                    "Duplicate upload prevented: video %s already on YouTube for %s",
                    existing, title,
                )
                raise RuntimeError(
                    f"Refusing to re-upload: {title} already exists on YouTube (video {existing})."
                )
        except RuntimeError:
            raise
        except Exception as e:
            # Never block an upload due to a duplicate-check failure.
            log.warning("Duplicate upload check skipped: %s", e)

    def _translation(self, script: Script, pick: Topic, writer: str, lang: str, s: Settings,
                     by_scene: dict[int, MediaItem], credits: set[str], parent_id: int, upload: bool) -> Result:
        t0 = time.time()
        try:
            if self.llm is None:
                raise RuntimeError("translations need an LLM (not available in demo mode)")
            translated = translate_script(self.llm, script, lang)
            translated.score, translated.review = script.score, script.review
            log.info("Translated to %s: %s", lang, translated.title)
            return self.produce(translated, pick, writer, upload=upload, t0=t0, media_map=by_scene, credits=credits,
                                parent_id=parent_id, settings=replace(s, language=lang, tts_voice="", tts_voice_b=""))
        except youtube.NotAuthorized:
            raise
        except Exception as e:  # including LLMConfigError: the original video is already made
            log.error("Translation to %s failed: %s", lang, redact(e))
            return Result(False, status="failed", error=redact(e), language=lang, seconds=time.time() - t0)

    # ------------------------------------------------------------------ uploading
    def quota_left(self) -> int:
        day = quota_day_start()
        if self._refused_on == day:  # YouTube said no today: nothing more goes up until the reset
            return 0
        return self.s.daily_upload_limit - self.history.uploads_since(day)

    def _pre_upload_video_check(self, rid: int, rec: dict) -> bool:
        """Validate the rendered video before any upload is attempted.

        Returns True when the video is usable (exists, ffprobe-readable, has
        video and audio streams). When it is not, the record is marked failed
        with a clear, actionable error and False is returned so no upload is
        attempted and the run is never mistaken for a successful publication.
        """
        video_path = Path(rec["video_path"]) if rec.get("video_path") else None
        if not rec or not video_path or not video_path.exists():
            log.error("Video #%s has no file to upload", rid)
            return False
        try:
            info = ffmpeg.probe(video_path)
        except Exception as e:
            log.error("Video #%s is not a usable video file: %s", rid, e)
            self.history.update_video(rid, status="failed", error=f"invalid video file: {e}")
            return False
        if not info.get("has_video"):
            log.error("Video #%s has no video stream", rid)
            self.history.update_video(rid, status="failed", error="rendered video has no video stream")
            return False
        if not info.get("has_audio"):
            log.error("Video #%s has no audio stream", rid)
            self.history.update_video(rid, status="failed", error="rendered video has no audio stream")
            return False
        return True

    def upload_record(self, rid: int, result: Result | None = None) -> str:
        rec = self.history.get(rid)
        if not rec:
            log.error("Video #%s has no file to upload", rid)
            return "missing"
        if rec.get("youtube_id") or rec.get("status") in ("uploaded", "scheduled"):
            log.warning("Video #%s already has a YouTube upload; refusing to create a duplicate", rid)
            return "already-uploaded"
        if not self._pre_upload_video_check(rid, rec):
            return "missing"
        if rec.get("approval_status") != "approved" or not rec.get("approved_at"):
            log.warning("Video #%s has not been explicitly approved; refusing to upload", rid)
            return "approval-required"
        if self.quota_left() <= 0:
            log.warning("Daily upload limit (%d) reached — video #%d queued for tomorrow", self.s.daily_upload_limit, rid)
            self.history.update_video(rid, status="queued")
            if result:
                result.status = "queued"
            notify.send(self.s, "queued", title=rec["title"], detail="daily upload limit reached")
            return "queued"
        folder = Path(rec["folder"])
        meta = json.loads((folder / "metadata.json").read_text()) if (folder / "metadata.json").exists() else {}
        publish = youtube.next_publish_slot(self.s, self.history.scheduled_times())
        body = youtube.build_body(self.s, rec["title"], rec["description"] or "",
                                  json.loads(rec["tags"] or "[]"), meta.get("category_id", "27"),
                                  meta.get("language", self.s.language), publish)
        try:
            resp = youtube.upload(self.s, Path(rec["video_path"]), body)
        except youtube.QuotaExceeded:
            log.warning("YouTube refused the upload (quota or upload limit) — video #%d queued until midnight Pacific",
                        rid)
            self._refused_on = quota_day_start()
            self.history.update_video(rid, status="queued")
            if result:
                result.status = "queued"
            notify.send(self.s, "queued", title=rec["title"], detail="YouTube's upload limit reached")
            return "queued"
        except youtube.NotAuthorized:
            raise
        except Exception as e:
            log.error("Upload of #%d failed: %s", rid, e)
            self.history.update_video(rid, status="failed", error=str(e)[:500])
            if result:
                result.status, result.error = "upload-failed", str(e)
            notify.send(self.s, "failed", title=rec["title"], detail=f"upload failed: {redact(e)}")
            return "failed"
        vid = resp.get("id")
        status = "scheduled" if publish else "uploaded"
        publish_iso = youtube.to_rfc3339(publish) if publish else None
        fields = dict(status=status, youtube_id=vid, uploaded_at=now_iso(), publish_at=publish_iso, error=None)
        if not self.s.keep_videos:
            Path(rec["video_path"]).unlink(missing_ok=True)
            fields["video_path"] = None
        self.history.update_video(rid, **fields)
        when = f", goes public {publish.strftime('%a %d %b %H:%M %Z')}" if publish else ""
        log.info("%s: https://youtube.com/shorts/%s%s", status.capitalize(), vid, when)
        notify.send(self.s, status, title=rec["title"], url=f"https://youtube.com/shorts/{vid}",
                    detail=f"public {publish.strftime('%a %d %b %H:%M %Z')}" if publish else "")
        if result:
            result.status, result.youtube_id, result.publish_at = status, vid, publish_iso
        return status

    # ------------------------------------------------------------------ scheduled (non-interactive) upload
    def upload_scheduled(self, rid: int, publish_at: datetime, result: Result | None = None,
                         slot: int | None = None, run_date: str | None = None) -> str:
        """Upload video ``rid`` as a private YouTube video scheduled to go public at ``publish_at``.

        This is the dedicated non-interactive publishing path: no ``PUBLISH`` prompt, no
        public upload. ``privacyStatus`` is ``private`` and ``publishAt`` is a future
        timestamp for the assigned IST slot. Quality approval must already have happened
        (the caller runs the quality gate first).

        Returns one of: ``scheduled``, ``queued``, ``already-uploaded``, ``missing``, ``failed``.
        """
        from . import youtube as YT

        rec = self.history.get(rid)
        if not rec:
            log.error("Video #%s has no file to upload", rid)
            return "missing"
        if rec.get("youtube_id") or rec.get("status") in ("uploaded", "scheduled"):
            log.warning("Video #%s already has a YouTube upload; refusing to create a duplicate", rid)
            return "already-uploaded"
        if not self._pre_upload_video_check(rid, rec):
            return "missing"
        if self.quota_left() <= 0:
            log.warning("Daily upload limit (%d) reached — video #%d not scheduled",
                        self.s.daily_upload_limit, rid)
            self.history.update_video(rid, status="queued")
            if result:
                result.status = "queued"
            notify.send(self.s, "queued", title=rec["title"], detail="daily upload limit reached")
            return "queued"

        # Best-effort cross-runner idempotency: if this slot already has a video on the
        # channel (e.g. the history database was not carried over to this runner), do not
        # re-upload. This never blocks an upload if YouTube itself cannot be reached.
        if slot is not None and run_date is not None:
            try:
                yt = YT.service(self.s, interactive=False)
                found = YT.find_existing_video(yt, f"CurioStick Short slot {slot} {run_date}")
            except YT.NotAuthorized:
                found = None
            except Exception as e:  # network/auth hiccups: local ledger remains the authority
                log.warning("YouTube duplicate check could not run: %s", e)
                found = None
            if found:
                log.warning("Slot %s on %s already has YouTube video %s; not re-uploading",
                            slot, run_date, found)
                self.history.update_video(
                    rid, status="scheduled", youtube_id=found,
                    uploaded_at=now_iso(), publish_at=YT.to_rfc3339(publish_at))
                if result:
                    result.status, result.youtube_id, result.publish_at = (
                        "scheduled", found, YT.to_rfc3339(publish_at))
                return "already-uploaded"

        folder = Path(rec["folder"])
        meta = json.loads((folder / "metadata.json").read_text()) if (folder / "metadata.json").exists() else {}
        description = rec["description"] or ""
        if slot is not None and run_date is not None:
            description += (
                f"\n\nCurioStick Short — slot {slot} | {run_date} (Asia/Kolkata)"
                f"\nScheduled to go public at {publish_at.strftime('%H:%M IST')}."
            )
        body = YT.build_body(self.s, rec["title"], description,
                             json.loads(rec["tags"] or "[]"), meta.get("category_id", "27"),
                             meta.get("language", self.s.language), publish_at)
        try:
            resp = YT.upload(self.s, Path(rec["video_path"]), body)
        except YT.QuotaExceeded:
            log.warning("YouTube refused the upload (quota/upload limit) — video #%d queued until midnight Pacific", rid)
            self._refused_on = quota_day_start()
            self.history.update_video(rid, status="queued")
            if result:
                result.status = "queued"
            notify.send(self.s, "queued", title=rec["title"], detail="YouTube's upload limit reached")
            return "queued"
        except YT.NotAuthorized:
            raise
        except Exception as e:
            log.error("Scheduled upload of #%d failed: %s", rid, e)
            self.history.update_video(rid, status="failed", error=str(e)[:500])
            if result:
                result.status, result.error = "upload-failed", str(e)
            notify.send(self.s, "failed", title=rec["title"], detail=f"scheduled upload failed: {redact(e)}")
            return "failed"

        vid = resp.get("id")
        publish_iso = YT.to_rfc3339(publish_at)
        fields = dict(status="scheduled", youtube_id=vid, uploaded_at=now_iso(), publish_at=publish_iso)
        if not self.s.keep_videos:
            Path(rec["video_path"]).unlink(missing_ok=True)
            fields["video_path"] = None
        self.history.update_video(rid, **fields)
        log.info("Scheduled: https://youtube.com/shorts/%s, public %s", vid,
                 publish_at.strftime("%a %d %b %H:%M %Z"))
        notify.send(self.s, "scheduled", title=rec["title"],
                    url=f"https://youtube.com/shorts/{vid}",
                    detail=f"public {publish_at.strftime('%a %d %b %H:%M %Z')}")
        if result:
            result.status, result.youtube_id, result.publish_at = "scheduled", vid, publish_iso
        return "scheduled"

    def flush_queue(self, include_rendered: bool = False) -> int:
        statuses = ("queued", "rendered") if include_rendered else ("queued",)
        done = 0
        for rec in self.history.pending_uploads(statuses):
            if self.quota_left() <= 0:
                break
            if self.upload_record(rec["id"]) in ("uploaded", "scheduled"):
                done += 1
        return done

    def delete(self, rid: int) -> bool:
        """Remove a video from the library and delete its folder (never touches YouTube)."""
        rec = self.history.get(rid)
        if not rec:
            return False
        folder = Path(rec["folder"]) if rec.get("folder") else None
        if folder and folder.is_dir() and folder.resolve().is_relative_to(self.s.out_path.resolve()):
            shutil.rmtree(folder, ignore_errors=True)
        self.history.delete_video(rid)
        return True

    # ------------------------------------------------------------------ autopilot
    def autopilot(self, *, count: int | None = None, once: bool = False, upload: bool | None = None,
                  topic: str | None = None, source: str | None = None) -> int:
        s = self.s
        upload = s.upload if upload is None else upload
        produced, batch, empty_batches = 0, 0, 0
        mode = "upload" if upload else "no-upload"
        _ = self._llm_or_none()  # fail fast on real LLM problems; fall back to offline scripts otherwise
        log.info("Autopilot started (%s, batch %d, workers %d%s)", mode, s.batch_size, s.workers,
                 f", target {count}" if count else "")
        try:
            while True:
                batch += 1
                if s.learn_from_stats and time.time() - self._stats_synced > 12 * 3600:
                    analytics.sync_stats(s, self.history)
                    self._stats_synced = time.time()
                if upload:
                    self.flush_queue()
                    if self.quota_left() <= 0 and self.history.pending_uploads():
                        wait = seconds_until_quota_reset()
                        log.info("Upload limit reached and videos are queued; sleeping %.1fh until quota resets",
                                 wait / 3600)
                        time.sleep(wait)
                        continue
                n = s.batch_size if count is None else min(s.batch_size, count - produced)
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(s.workers, n)) as ex:
                    futs = [ex.submit(self.make, topic, source=source, upload=upload) for _ in range(n)]
                    results = [f.result() for f in futs]
                ok = sum(r.ok for r in results)
                produced += ok
                log.info("Batch %d: %d/%d succeeded (total %d)", batch, ok, n, produced)
                empty_batches = empty_batches + 1 if ok == 0 else 0
                if once or (count is not None and produced >= count):
                    break
                if empty_batches >= 3:
                    log.error("Three batches in a row produced nothing — stopping. Check the log above.")
                    break
                if s.batch_delay:
                    log.info("Next batch in %ds", s.batch_delay)
                    time.sleep(s.batch_delay)
        except KeyboardInterrupt:
            log.info("Stopped by user.")
        return produced
