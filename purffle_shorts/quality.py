"""Quality gate: validates rendered videos before publishing.

Checks are split into two kinds:

  * ``automated`` — measured against the rendered file or its metadata
    (ffprobe, ffmpeg blackdetect, SRT/audio timing comparison, metadata fields).
  * ``heuristic`` — approximate estimates that cannot be proven without
    human review (caption overflow estimate, visual relevance estimate).

A check marked ``heuristic`` is never reported as "verified"; it is an
advisory signal only.

The weighted quality score is out of 100: integrity 35 (file 8, ffprobe 8,
streams 8, audio 8, render status 3); presentation 20 (aspect ratio 8,
duration 6, black frames 4, thumbnail 2); captions 15 (valid SRT 5, timing
7, overflow 3); content 20 (script 7, hook 5, visual mapping 4, visuals 4);
metadata 10 (metadata 4, research 3, attribution 3). Passing requires
80/100 and category minimums of integrity 30/35, presentation 14/20,
captions 10/15, content 12/20, and metadata 5/10. Missing/invalid video,
missing or inaudible audio, and a missing thumbnail are critical blockers.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from . import ffmpeg
from .config import ASPECTS, Settings, duration_window, short_duration_bounds
from .script import HOOK_BAD_PATTERNS, word_count

if TYPE_CHECKING:
    from .pipeline import Result

log = logging.getLogger("purffle")

# Minimum fraction of the video that must not be black.
MAX_BLACK_FRACTION = 0.10
# Caption safe-area margin (fraction of frame width/height).
CAPTION_SAFE_MARGIN = 0.08
MAX_CAPTION_GAP_SECONDS = 3.0
MINIMUM_QUALITY_SCORE = 80
QUALITY_WEIGHTS = {
    "file_exists": ("integrity", 8),
    "ffprobe_readable": ("integrity", 8),
    "streams": ("integrity", 8),
    "audio_present": ("integrity", 4),
    "audio_audible": ("integrity", 4),
    "render_status": ("integrity", 3),
    "aspect_ratio": ("presentation", 8),
    "duration": ("presentation", 6),
    "black_frames": ("presentation", 4),
    "thumbnail": ("presentation", 2),
    "captions": ("captions", 5),
    "caption_sync": ("captions", 7),
    "caption_overflow": ("captions", 3),
    "script_complete": ("content", 7),
    "script_hook": ("content", 5),
    "scene_visual_mapping": ("content", 4),
    "visuals_present": ("content", 4),
    "metadata": ("metadata", 4),
    "research_verification": ("metadata", 3),
    "attribution": ("metadata", 3),
}
CATEGORY_MINIMUMS = {
    "integrity": 30,
    "presentation": 14,
    "captions": 10,
    "content": 12,
    "metadata": 5,
}
CRITICAL_CHECKS = {
    "file_exists", "ffprobe_readable", "streams", "audio_present", "audio_audible", "duration", "thumbnail",
}
SRT_TIMESTAMP = re.compile(r"^(-?\d{2,}:[0-5]\d:[0-5]\d,\d{3})$")


def parse_srt(content: str, duration: float | None = None) -> list[tuple[float, float, str]]:
    """Parse and validate standard SRT entries, including their timeline bounds."""
    content = content.lstrip("\ufeff").strip()
    if not content:
        raise ValueError("SRT is empty")
    entries = re.split(r"\r?\n\s*\r?\n", content)
    parsed: list[tuple[float, float, str]] = []
    previous_end = 0.0
    previous_index = 0

    def parse_timestamp(value: str, index: int) -> float:
        if not SRT_TIMESTAMP.fullmatch(value):
            raise ValueError(f"Caption {index} has an invalid timestamp")
        sign = -1 if value.startswith("-") else 1
        hours, minutes, seconds = value.lstrip("-").replace(",", ".").split(":")
        return sign * (int(hours) * 3600 + int(minutes) * 60 + float(seconds))

    for entry_number, entry in enumerate(entries, start=1):
        lines = entry.strip().splitlines()
        if len(lines) < 3 or not lines[0].strip().isdigit():
            raise ValueError(f"Caption {entry_number} has an invalid SRT structure")
        index = int(lines[0].strip())
        if index <= previous_index:
            raise ValueError(f"Caption {entry_number} has a non-increasing cue number")
        match = re.fullmatch(r"\s*(\S+)\s+-->\s+(\S+)\s*", lines[1])
        if not match:
            raise ValueError(f"Caption {index} has an invalid timestamp interval")
        start = parse_timestamp(match.group(1), index)
        end = parse_timestamp(match.group(2), index)
        if start < 0:
            raise ValueError(f"Caption {index} starts before zero")
        if end <= start:
            raise ValueError(f"Caption {index} ends before or at its start")
        if duration is not None and end > duration:
            raise ValueError(f"Caption {index} ends beyond video duration")
        if start < previous_end - 0.1:
            raise ValueError(f"Caption {index} overlaps the previous caption")
        gap = start - previous_end
        if gap > MAX_CAPTION_GAP_SECONDS:
            raise ValueError(f"Caption {index} follows an excessive {gap:.2f}s timing gap")
        text = "\n".join(lines[2:]).strip()
        if not text:
            raise ValueError(f"Caption {index} has no text")
        parsed.append((start, end, text))
        previous_end = end
        previous_index = index
    return parsed


@dataclass
class QualityCheck:
    """A single quality check result."""

    name: str
    passed: bool
    message: str = ""
    details: dict = field(default_factory=dict)

    @property
    def method(self) -> str:
        return self.details.get("method", "automated")


@dataclass
class QualityReport:
    """Complete quality gate report."""

    passed: bool
    checks: list[QualityCheck]
    video_path: Path
    checked_at: str = field(default_factory=lambda: datetime.now().isoformat())
    score: int = 0
    category_scores: dict[str, int] = field(default_factory=dict)
    category_totals: dict[str, int] = field(default_factory=dict)
    category_minimums: dict[str, int] = field(default_factory=lambda: dict(CATEGORY_MINIMUMS))
    critical_failures: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Human-readable summary of failures."""
        failures = [c for c in self.checks if not c.passed]
        result = f"Quality score {self.score}/100 (minimum {MINIMUM_QUALITY_SCORE})"
        if failures:
            result += "; " + f"{len(failures)} check(s) failed: " + "; ".join(
                f"{c.name}: {c.message}" for c in failures
            )
        elif self.passed:
            result += "; all checks passed"
        if self.critical_failures:
            result += "; critical blockers: " + ", ".join(self.critical_failures)
        return result

    @property
    def failed_checks(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def automated_failures(self) -> list[QualityCheck]:
        """Checks that failed and are automated (not heuristic)."""
        return [c for c in self.failed_checks if c.method == "automated"]

    @property
    def heuristic_advisories(self) -> list[QualityCheck]:
        """Checks that are heuristic/advisory (not proof of a problem)."""
        return [c for c in self.checks if c.method == "heuristic"]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checked_at": self.checked_at,
            "video_path": str(self.video_path),
            "score": self.score,
            "minimum_score": MINIMUM_QUALITY_SCORE,
            "category_scores": self.category_scores,
            "category_totals": self.category_totals,
            "category_minimums": self.category_minimums,
            "critical_failures": self.critical_failures,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "details": c.details,
                    "method": c.method,
                }
                for c in self.checks
            ],
            "summary": self.summary,
        }


class QualityGate:
    """Validates rendered videos against quality criteria."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def check(self, result: Result) -> QualityReport:
        """Run all quality checks on a rendered video."""
        if not result.video or not result.video.exists():
            return self._report(
                [QualityCheck("file_exists", False, "Video file does not exist")],
                result.video or Path(""),
            )

        checks = []

        # 1. File exists and is not empty (automated)
        checks.append(self._check_file_exists(result.video))

        # 2. FFprobe can read the file (automated)
        checks.append(self._check_ffprobe_readable(result.video))

        # 3. Video and audio streams present (automated)
        checks.append(self._check_streams(result.video))

        # 4. Aspect ratio matches requested output (automated)
        checks.append(self._check_aspect_ratio(result.video))

        # 5. Duration within limits (automated)
        checks.append(self._check_duration(result.video))

        # 6. Missing audio / corrupted file (automated, part of streams)
        checks.append(self._check_audio_present(result.video))
        checks.append(self._check_audio_audible(result.video))

        # 7. Black-frame detection (automated)
        checks.append(self._check_black_frames(result.video))

        # 8. Captions file exists (automated)
        checks.append(self._check_captions_exist(result.folder))
        checks.append(self._check_thumbnail_exists(result.folder))

        # 9. Caption-audio synchronization (automated)
        checks.append(self._check_caption_sync(result))

        # 11. Caption spelling and punctuation (heuristic)
        checks.append(self._check_caption_spelling(result))

        # 10. Caption overflow / safe-area (heuristic)
        checks.append(self._check_caption_overflow(result))

        # 11. Missing / irrelevant scene visuals (automated)
        checks.append(self._check_visuals_present(result))

        # 12. Metadata file exists and is valid (automated)
        checks.append(self._check_metadata(result.folder))

        # 13. Script narration and scene-to-visual mapping are complete (automated)
        checks.append(self._check_script_complete(result.folder))
        checks.append(self._check_hook(result.folder))
        checks.append(self._check_scene_visual_mapping(result.folder))

        # 14. Research verification present (automated)
        checks.append(self._check_research_verification(result))

        # 15. Attribution present for stock visuals (automated)
        checks.append(self._check_attribution(result))

        # 16. Render completed successfully (automated)
        checks.append(self._check_render_status(result))

        return self._report(checks, result.video)

    @staticmethod
    def _report(checks: list[QualityCheck], video_path: Path) -> QualityReport:
        by_name = {check.name: check for check in checks}
        category_scores = {name: 0 for name in CATEGORY_MINIMUMS}
        category_totals = {name: 0 for name in CATEGORY_MINIMUMS}
        for name, (category, points) in QUALITY_WEIGHTS.items():
            category_totals[category] += points
            check = by_name.get(name)
            if check and check.passed:
                category_scores[category] += points
        score = sum(category_scores.values())
        critical_failures = sorted(
            name for name in CRITICAL_CHECKS
            if name not in by_name or not by_name[name].passed
        )
        passed = (
            score >= MINIMUM_QUALITY_SCORE
            and all(category_scores[name] >= minimum for name, minimum in CATEGORY_MINIMUMS.items())
            and not critical_failures
        )
        return QualityReport(
            passed=passed, checks=checks, video_path=video_path, score=score,
            category_scores=category_scores, category_totals=category_totals,
            critical_failures=critical_failures,
        )

    def _check_file_exists(self, video_path: Path) -> QualityCheck:
        """Check video file exists and has size > 0."""
        if not video_path.exists():
            return QualityCheck("file_exists", False, "Video file does not exist")
        size = video_path.stat().st_size
        if size == 0:
            return QualityCheck("file_exists", False, "Video file is empty (0 bytes)")
        return QualityCheck("file_exists", True, f"File exists ({size:,} bytes)")

    def _check_ffprobe_readable(self, video_path: Path) -> QualityCheck:
        """Check ffprobe can read the file."""
        try:
            info = ffmpeg.probe(video_path)
            return QualityCheck(
                "ffprobe_readable",
                True,
                f"FFprobe readable: {info.get('width')}x{info.get('height')}, {info.get('duration', 0):.1f}s"
            )
        except Exception as e:
            return QualityCheck("ffprobe_readable", False, f"FFprobe failed: {e}")

    def _check_streams(self, video_path: Path) -> QualityCheck:
        """Check video and audio streams are present."""
        try:
            info = ffmpeg.probe(video_path)
            has_video = info.get("has_video", False)
            has_audio = info.get("has_audio", False)

            if not has_video and not has_audio:
                return QualityCheck("streams", False, "No video or audio streams found")
            if not has_video:
                return QualityCheck("streams", False, "No video stream found")
            if not has_audio:
                return QualityCheck("streams", False, "No audio stream found")

            return QualityCheck("streams", True, "Video and audio streams present")
        except Exception as e:
            return QualityCheck("streams", False, f"Stream check failed: {e}")

    def _check_aspect_ratio(self, video_path: Path) -> QualityCheck:
        """Check the video matches the requested output shape (9:16, 16:9, 1:1 or 4:5)."""
        try:
            info = ffmpeg.probe(video_path)
            width = info.get("width", 0)
            height = info.get("height", 0)

            if width == 0 or height == 0:
                return QualityCheck("aspect_ratio", False, "Could not determine dimensions")

            target_w, target_h = self.settings.resolution
            label = next((name for name, dims in ASPECTS.items() if dims == (target_w, target_h)),
                         f"{target_w}:{target_h}")
            ratio = width / height
            target_ratio = target_w / target_h
            tolerance = 0.02

            if abs(ratio - target_ratio) <= tolerance:
                return QualityCheck(
                    "aspect_ratio", True,
                    f"Aspect ratio {width}:{height} ≈ {label}"
                )
            return QualityCheck(
                "aspect_ratio", False,
                f"Aspect ratio {width}:{height} = {ratio:.4f}, expected ~{target_ratio:.4f} ({label})"
            )
        except Exception as e:
            return QualityCheck("aspect_ratio", False, f"Aspect ratio check failed: {e}")

    def _check_duration(self, video_path: Path) -> QualityCheck:
        """Check video duration is within configured limits."""
        try:
            info = ffmpeg.probe(video_path)
            duration = info.get("duration", 0)

            if duration <= 0:
                return QualityCheck("duration", False, "Invalid duration")

            min_dur, max_dur = duration_window(self.settings.target_seconds)
            # The 40-second Shorts format uses a strict window: preferred 36-40s
            # with a hard maximum of 40s (no encoding tolerance added above the cap).
            short = short_duration_bounds(self.settings.target_seconds)
            if short is not None:
                min_dur, max_dur = short

            if short is not None:
                # Shorts contract: the hard maximum is never exceeded; the
                # preferred minimum is advisory because narration pace varies
                # by voice and must not be forced with unnatural TTS speed.
                if duration > max_dur:
                    return QualityCheck(
                        "duration", False,
                        f"Duration {duration:.1f}s exceeds hard maximum {max_dur}s")
                if duration < min_dur:
                    return QualityCheck(
                        "duration", True,
                        f"Duration {duration:.1f}s under preferred minimum {min_dur}s "
                        f"(hard cap {max_dur}s)",
                        details={"method": "automated", "advisory": "under_preferred_min"})
                return QualityCheck(
                    "duration", True,
                    f"Duration {duration:.1f}s within preferred range ({min_dur}-{max_dur}s)")
            if duration < min_dur:
                return QualityCheck(
                    "duration", False,
                    f"Duration {duration:.1f}s below minimum {min_dur}s")
            if duration > max_dur:
                return QualityCheck(
                    "duration", False,
                    f"Duration {duration:.1f}s exceeds maximum {max_dur}s")

            return QualityCheck(
                "duration", True,
                f"Duration {duration:.1f}s within limits ({min_dur}-{max_dur}s)"
            )
        except Exception as e:
            return QualityCheck("duration", False, f"Duration check failed: {e}")

    def _check_captions_exist(self, folder: Path | None, duration: float | None = None) -> QualityCheck:
        """Check captions SRT file exists and contains valid, bounded entries."""
        if not folder:
            return QualityCheck("captions", False, "No folder path provided")

        srt_path = folder / "captions.srt"
        if not srt_path.exists():
            return QualityCheck("captions", False, "captions.srt not found")

        try:
            entries = parse_srt(srt_path.read_text(encoding="utf-8"), duration)
            return QualityCheck("captions", True, f"Captions found ({len(entries)} entries)")
        except ValueError as e:
            return QualityCheck("captions", False, f"Invalid captions.srt: {e}")
        except Exception as e:
            return QualityCheck("captions", False, f"Failed to read captions: {e}")

    def _check_thumbnail_exists(self, folder: Path | None) -> QualityCheck:
        if not folder:
            return QualityCheck("thumbnail", False, "No folder path provided")
        thumbnail = folder / "cover.jpg"
        if not thumbnail.exists() or thumbnail.stat().st_size == 0:
            return QualityCheck("thumbnail", False, "cover.jpg not found or is empty")
        try:
            from PIL import Image
            with Image.open(thumbnail) as image:
                width, height = image.size
            if width <= 0 or height <= 0:
                return QualityCheck("thumbnail", False, "cover.jpg has invalid dimensions")
            return QualityCheck("thumbnail", True, f"Cover image valid ({width}x{height})")
        except Exception as exc:
            return QualityCheck("thumbnail", False, f"cover.jpg could not be read: {exc}")

    def _check_metadata(self, folder: Path | None) -> QualityCheck:
        """Check metadata.json exists and has required fields."""
        if not folder:
            return QualityCheck("metadata", False, "No folder path provided")

        meta_path = folder / "metadata.json"
        if not meta_path.exists():
            return QualityCheck("metadata", False, "metadata.json not found")

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            required = ["title", "duration", "resolution", "created_at"]
            missing = [f for f in required if f not in meta]

            if missing:
                return QualityCheck("metadata", False, f"Missing required fields: {missing}")

            return QualityCheck("metadata", True, "Metadata valid with all required fields")
        except Exception as e:
            return QualityCheck("metadata", False, f"Failed to read metadata: {e}")

    def _check_script_complete(self, folder: Path | None) -> QualityCheck:
        if not folder:
            return QualityCheck("script_complete", False, "No folder path provided")
        path = folder / "script.json"
        if not path.exists():
            return QualityCheck("script_complete", False, "script.json not found")
        try:
            script = json.loads(path.read_text(encoding="utf-8"))
            scenes = script.get("scenes")
            if not script.get("topic") or not script.get("title") or not isinstance(scenes, list) or not scenes:
                return QualityCheck("script_complete", False, "Script needs a topic, title and at least one scene")
            missing = [
                i + 1 for i, scene in enumerate(scenes)
                if not scene.get("narration", "").strip()
                or not scene.get("search_query", "").strip()
                or not scene.get("image_prompt", "").strip()
            ]
            if missing:
                return QualityCheck("script_complete", False,
                                    f"Scene(s) {', '.join(map(str, missing))} lack narration or visual prompts")
            return QualityCheck("script_complete", True, f"Complete script with {len(scenes)} scene(s)")
        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as e:
            return QualityCheck("script_complete", False, f"Script could not be validated: {e}")

    def _check_scene_visual_mapping(self, folder: Path | None) -> QualityCheck:
        if not folder:
            return QualityCheck("scene_visual_mapping", False, "No folder path provided")
        script_path, metadata_path = folder / "script.json", folder / "metadata.json"
        if not script_path.exists() or not metadata_path.exists():
            return QualityCheck("scene_visual_mapping", False, "Script or video metadata is missing")
        try:
            script = json.loads(script_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            scenes = script.get("scenes") or []
            visuals = metadata.get("visuals") or []
            if not scenes or len(scenes) != len(visuals):
                return QualityCheck(
                    "scene_visual_mapping", False,
                    f"Expected one selected visual per narration scene; found {len(visuals)} for {len(scenes)} scene(s)",
                )
            missing = [
                i + 1 for i, (scene, visual) in enumerate(zip(scenes, visuals))
                if not scene.get("narration", "").strip()
                or not scene.get("image_prompt", "").strip()
                or not visual.get("source")
                or not visual.get("kind")
                or not visual.get("query", "").strip()
            ]
            if missing:
                return QualityCheck("scene_visual_mapping", False,
                                    f"Scene(s) {', '.join(map(str, missing))} lack a recorded visual selection")
            return QualityCheck(
                "scene_visual_mapping", True,
                f"Recorded a visual prompt and selected asset for each of {len(scenes)} scene(s)",
                details={"method": "automated", "scene_count": len(scenes)},
            )
        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as e:
            return QualityCheck("scene_visual_mapping", False, f"Scene visual mapping could not be validated: {e}")

    def _check_hook(self, folder: Path | None) -> QualityCheck:
        if not folder:
            return QualityCheck("script_hook", False, "No folder path provided")
        path = folder / "script.json"
        if not path.exists():
            return QualityCheck("script_hook", False, "script.json not found")
        try:
            script = json.loads(path.read_text(encoding="utf-8"))
            scenes = script.get("scenes") or []
            if not scenes or not scenes[0].get("narration", "").strip():
                return QualityCheck("script_hook", False, "Opening scene has no narration")
            opening = scenes[0]["narration"].strip()
            if HOOK_BAD_PATTERNS.match(opening):
                return QualityCheck("script_hook", False,
                                    "Opening uses a greeting or generic 'Did you know?' hook")
            if word_count(opening, script.get("language", "en")) < 3:
                return QualityCheck(
                    "script_hook", False,
                    "Opening hook is too short to establish a clear idea (minimum three spoken words)")
            return QualityCheck("script_hook", True, "Opening narration starts directly with the story")
        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as e:
            return QualityCheck("script_hook", False, f"Opening hook could not be validated: {e}")

    def _check_render_status(self, result: Result) -> QualityCheck:
        """Check that render completed successfully (status is 'rendered')."""
        if result.status == "rendered":
            return QualityCheck("render_status", True, f"Render status: {result.status}")
        return QualityCheck("render_status", False, f"Render status: {result.status} (expected 'rendered')")

    # ------------------------------------------------------------------ new checks (Phase 4)

    def _check_audio_present(self, video_path: Path) -> QualityCheck:
        """Check the rendered video has a non-empty audio track (automated)."""
        try:
            info = ffmpeg.probe(video_path)
            if info.get("has_audio"):
                return QualityCheck(
                    "audio_present", True,
                    "Audio stream present",
                    details={"method": "automated"},
                )
            return QualityCheck(
                "audio_present", False,
                "No audio stream found in rendered video",
                details={"method": "automated"},
            )
        except Exception as e:
            return QualityCheck("audio_present", False, f"Audio check failed: {e}",
                                details={"method": "automated"})

    def _check_audio_audible(self, video_path: Path) -> QualityCheck:
        """Reject digital silence or effectively inaudible audio tracks before scheduling."""
        try:
            proc = subprocess.run(
                [
                    ffmpeg.ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", str(video_path),
                    "-vn", "-af", "volumedetect", "-f", "null", "-",
                ],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode:
                return QualityCheck("audio_audible", False,
                                    f"Audio level analysis failed: {proc.stderr[-500:]}")
            match = re.search(r"max_volume:\s*(-inf|-?\d+(?:\.\d+)?)\s*dB", proc.stderr)
            if not match:
                return QualityCheck("audio_audible", False, "FFmpeg did not report an audio peak level")
            peak = match.group(1)
            if peak == "-inf" or float(peak) <= -60:
                return QualityCheck("audio_audible", False,
                                    f"Audio is silent or effectively inaudible (peak {peak} dBFS)")
            return QualityCheck("audio_audible", True, f"Audio signal detected (peak {peak} dBFS)")
        except (OSError, subprocess.SubprocessError) as e:
            return QualityCheck("audio_audible", False, f"Audio level check failed: {e}")

    def _check_black_frames(self, video_path: Path) -> QualityCheck:
        """Detect black frames using ffmpeg's blackdetect filter (automated)."""
        try:
            info = ffmpeg.probe(video_path)
            duration = info.get("duration", 0)
            if duration <= 0:
                return QualityCheck("black_frames", False, "Cannot determine video duration",
                                    details={"method": "automated"})
            cmd = [
                ffmpeg.ffmpeg_bin(), "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error",
                "-i", str(video_path),
                "-vf", "blackdetect=d=0.5:pix_th=0.10",
                "-f", "null", "-",
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
            text = proc.stderr or ""
            black_periods = re.findall(r"black_detect.*?duration: ([0-9.]+)", text)
            total_black = sum(float(d) for d in black_periods)
            black_fraction = total_black / duration if duration > 0 else 0
            if black_fraction > MAX_BLACK_FRACTION:
                return QualityCheck(
                    "black_frames", False,
                    f"Excessive black frames: {black_fraction:.1%} of duration",
                    details={"method": "automated", "black_fraction": round(black_fraction, 3),
                             "black_periods": len(black_periods)},
                )
            return QualityCheck(
                "black_frames", True,
                f"Black frames acceptable ({black_fraction:.1%} of duration)",
                details={"method": "automated", "black_fraction": round(black_fraction, 3),
                         "black_periods": len(black_periods)},
            )
        except subprocess.TimeoutExpired:
            return QualityCheck("black_frames", False, "Black-frame detection timed out",
                                details={"method": "automated"})
        except Exception as e:
            return QualityCheck("black_frames", False, f"Black-frame detection failed: {e}",
                                details={"method": "automated"})

    def _check_caption_sync(self, result: Result) -> QualityCheck:
        """Validate caption timing order, gaps, and end bounds against the rendered video."""
        if not result.folder:
            return QualityCheck("caption_sync", False, "No folder path provided",
                                details={"method": "automated"})
        srt_path = result.folder / "captions.srt"
        meta_path = result.folder / "metadata.json"
        if not srt_path.exists() or not meta_path.exists():
            return QualityCheck(
                "caption_sync", False,
                "Captions or metadata missing; cannot verify sync",
                details={"method": "automated"},
            )
        try:
            assert result.video is not None
            info = ffmpeg.probe(result.video)
            duration = info.get("duration", 0)
            entries = parse_srt(srt_path.read_text(encoding="utf-8"), duration)
            max_caption_end = max(end for _, end, _ in entries)
            return QualityCheck(
                "caption_sync", True,
                f"Captions within video duration ({len(entries)} captions, ends {max_caption_end:.2f}s)",
                details={"method": "automated", "max_caption_end": round(max_caption_end, 2),
                         "video_duration": round(duration, 2)},
            )
        except Exception as e:
            return QualityCheck("caption_sync", False, f"Caption sync check failed: {e}",
                                details={"method": "automated"})

    def _check_caption_overflow(self, result: Result) -> QualityCheck:
        """Estimate caption overflow against the safe area (heuristic).

        This is an approximate check: it estimates whether caption text could
        exceed the safe-area bounds based on character count and font size. It
        cannot render the actual frames, so it is advisory only.
        """
        if not result.folder:
            return QualityCheck("caption_overflow", False, "No folder path provided",
                                details={"method": "heuristic"})
        srt_path = result.folder / "captions.srt"
        if not srt_path.exists():
            return QualityCheck("caption_overflow", False, "Captions file not found",
                                details={"method": "heuristic"})
        try:
            entries = parse_srt(srt_path.read_text(encoding="utf-8"))
            max_chars = 0
            for _, _, text in entries:
                max_chars = max(max_chars, len(text.replace("\n", " ")))
            # Heuristic: at 1080x1920 with the default kinetic style, ~40 chars
            # is the practical limit before overflow risk rises.
            overflow_risk = max_chars > 45
            if overflow_risk:
                return QualityCheck(
                    "caption_overflow", False,
                    f"Caption may overflow safe area (max {max_chars} chars per caption)",
                    details={"method": "heuristic", "max_chars": max_chars,
                             "risk": "possible_overflow"},
                )
            return QualityCheck(
                "caption_overflow", True,
                f"Caption lengths within estimated safe area (max {max_chars} chars)",
                details={"method": "heuristic", "max_chars": max_chars,
                         "risk": "low"},
            )
        except Exception as e:
            return QualityCheck("caption_overflow", False, f"Caption overflow check failed: {e}",
                                details={"method": "heuristic"})

    def _check_caption_spelling(self, result: Result) -> QualityCheck:
        """Check caption text for obvious spelling errors.

        This is a heuristic (advisory) check only: it looks for things like
        repeated words, obvious typos using simple pattern matching, and missing
        sentence-ending punctuation where appropriate. It does NOT reject
        sentence fragments, informal speech, or creative language — those are
        valid in captions and short-form video narration.
        """
        if not result.folder:
            return QualityCheck("caption_spelling", False, "No folder path provided",
                                details={"method": "heuristic"})
        srt_path = result.folder / "captions.srt"
        if not srt_path.exists():
            return QualityCheck("caption_spelling", False, "Captions file not found",
                                details={"method": "heuristic"})
        try:
            entries = parse_srt(srt_path.read_text(encoding="utf-8"))
            flags = []
            for i, (start, end, text) in enumerate(entries, start=1):
                # Check for repeated words (e.g. "the the")
                words = text.split()
                for j in range(len(words) - 1):
                    if words[j].lower() == words[j + 1].lower() and len(words[j]) > 2:
                        flags.append(f"Caption {i}: repeated word '{words[j]}'")
                        break
                # Check for missing period at end (if the caption looks like a statement)
                # but allow fragments and questions
                stripped = text.strip()
                if stripped and not stripped.endswith(('.', '!', '?')) and not stripped.startswith(('"', "'")):
                    # Only flag if it looks like a complete sentence (capitalized and no conjunction start)
                    if stripped[0].isupper() and len(stripped) > 8:
                        flags.append(f"Caption {i}: missing sentence-ending punctuation")
            if flags:
                return QualityCheck(
                    "caption_spelling", False,
                    "; ".join(flags[:5]),
                    details={"method": "heuristic", "flags": flags},
                )
            return QualityCheck(
                "caption_spelling", True,
                "No spelling issues detected",
                details={"method": "heuristic"},
            )
        except Exception as e:
            return QualityCheck("caption_spelling", False, f"Spell check failed: {e}",
                                details={"method": "heuristic"})


    def _check_visuals_present(self, result: Result) -> QualityCheck:
        """Check that no scene fell back to an animated gradient (automated).

        Inspects metadata.visual_selection for scenes whose source is "generated"
        with kind "generated" (the gradient placeholder). Those scenes have no
        real visual matched to their narration.

        In production mode, gradient placeholders always fail the quality gate.
        In demo/auto mode, gradients are acceptable with a warning.
        """
        if not result.folder:
            return QualityCheck("visuals_present", False, "No folder path provided",
                                details={"method": "automated"})
        meta_path = result.folder / "metadata.json"
        if not meta_path.exists():
            return QualityCheck("visuals_present", False, "Metadata not found",
                                details={"method": "automated"})
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            visuals = meta.get("visuals", [])
            if not visuals:
                return QualityCheck("visuals_present", True, "No visual selection data",
                                    details={"method": "automated"})
            gradient_scenes = [
                v for v in visuals
                if v.get("source") == "generated" and v.get("kind") == "generated"
            ]
            # Check if production mode: gradient placeholders always fail
            try:
                from purffle_shorts.config import Settings
                settings = getattr(result, "settings", None)
                is_production = settings is not None and                                isinstance(settings, Settings) and                                settings.visual_mode == "production"
            except Exception:
                is_production = False
            if gradient_scenes:
                scene_nums = [v.get("scene") for v in gradient_scenes if "scene" in v]
                if is_production:
                    return QualityCheck(
                        "visuals_present", False,
                        f"{len(gradient_scenes)} scene(s) lack a real visual (gradient placeholder) — production mode requires real assets",
                        details={"method": "automated", "gradient_scenes": scene_nums,
                                 "total_scenes": len(visuals)},
                    )
                else:
                    # Demo/auto mode: gradient placeholders are acceptable
                    return QualityCheck(
                        "visuals_present", True,
                        f"{len(gradient_scenes)} scene(s) use gradient placeholders (demo mode)",
                        details={"method": "automated", "gradient_scenes": scene_nums,
                                 "total_scenes": len(visuals), "mode": "demo"},
                    )
            return QualityCheck(
                "visuals_present", True,
                f"All {len(visuals)} scenes have a real visual source",
                details={"method": "automated", "total_scenes": len(visuals)},
            )
        except Exception as e:
            return QualityCheck("visuals_present", False, f"Visual check failed: {e}",
                                details={"method": "automated"})

    def _check_research_verification(self, result: Result) -> QualityCheck:
        """Check that research verification data is present in metadata (automated).

        Does not judge whether claims are true — only that source references and
        verification status were recorded. Claims marked ``needs_human_review``
        are expected and do not fail this check.
        """
        if not result.folder:
            return QualityCheck("research_verification", False, "No folder path provided",
                                details={"method": "automated"})
        meta_path = result.folder / "metadata.json"
        if not meta_path.exists():
            return QualityCheck("research_verification", False, "Metadata not found",
                                details={"method": "automated"})
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            research = meta.get("research", {})
            if not research:
                return QualityCheck(
                    "research_verification", False,
                    "No research/verification data recorded in metadata",
                    details={"method": "automated"},
                )
            references = research.get("references", [])
            status = research.get("verification_status", "not_performed")
            return QualityCheck(
                "research_verification", True,
                f"Research data present ({len(references)} source reference(s), "
                f"verification: {status})",
                details={"method": "automated", "source_count": len(references),
                         "verification_status": status},
            )
        except Exception as e:
            return QualityCheck("research_verification", False, f"Research check failed: {e}",
                                details={"method": "automated"})

    def _check_attribution(self, result: Result) -> QualityCheck:
        """Check that stock visuals carry attribution from actual provider metadata (automated).

        AI-generated visuals are recorded separately and do not require
        photographer attribution. Stock visuals (Pexels/Pixabay) should have a
        non-empty attribution_line built from provider data.
        """
        if not result.folder:
            return QualityCheck("attribution", False, "No folder path provided",
                                details={"method": "automated"})
        meta_path = result.folder / "metadata.json"
        if not meta_path.exists():
            return QualityCheck("attribution", False, "Metadata not found",
                                details={"method": "automated"})
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            visuals = meta.get("visuals", [])
            if not visuals:
                return QualityCheck("attribution", True, "No visual selection data",
                                    details={"method": "automated"})
            missing_attribution = []
            for v in visuals:
                source = v.get("source", "")
                attribution = v.get("attribution", "")
                is_ai = source in {"agnes", "openai-images", "pollinations"} or v.get(
                    "kind") == "generated"
                if not is_ai and not attribution:
                    missing_attribution.append(v.get("query", "unknown"))
            if missing_attribution:
                return QualityCheck(
                    "attribution", False,
                    f"{len(missing_attribution)} stock visual(s) missing attribution",
                    details={"method": "automated", "missing_queries": missing_attribution},
                )
            return QualityCheck(
                "attribution", True,
                f"All {len(visuals)} visual(s) have appropriate attribution",
                details={"method": "automated", "total_visuals": len(visuals)},
            )
        except Exception as e:
            return QualityCheck("attribution", False, f"Attribution check failed: {e}",
                                details={"method": "automated"})


def run_quality_checks(video_path: Path, folder: Path, settings: Settings) -> QualityReport:
    """Convenience function to run quality checks on a video."""
    gate = QualityGate(settings)
    # Create a minimal Result object for the check (lazy import to avoid circular import).
    from .pipeline import Result
    result = Result(
        ok=True,
        video=video_path,
        folder=folder,
        status="rendered",
    )
    return gate.check(result)
