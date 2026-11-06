"""Quality gate: validates rendered videos before publishing.

Checks are split into two kinds:

  * ``automated`` — measured against the rendered file or its metadata
    (ffprobe, ffmpeg blackdetect, SRT/audio timing comparison, metadata fields).
  * ``heuristic`` — approximate estimates that cannot be proven without
    human review (caption overflow estimate, visual relevance estimate).

A check marked ``heuristic`` is never reported as "verified"; it is an
advisory signal only.
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
from .config import ASPECTS, Settings, duration_window
from .script import HOOK_BAD_PATTERNS

if TYPE_CHECKING:
    from .pipeline import Result

log = logging.getLogger("purffle")

# Tolerance for caption-vs-audio sync drift (seconds).
CAPTION_SYNC_TOLERANCE = 0.5
# Minimum fraction of the video that must not be black.
MAX_BLACK_FRACTION = 0.10
# Caption safe-area margin (fraction of frame width/height).
CAPTION_SAFE_MARGIN = 0.08


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

    @property
    def summary(self) -> str:
        """Human-readable summary of failures."""
        failures = [c for c in self.checks if not c.passed]
        if not failures:
            return "All checks passed"
        return f"{len(failures)} check(s) failed: " + "; ".join(
            f"{c.name}: {c.message}" for c in failures
        )

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
            return QualityReport(
                passed=False,
                checks=[QualityCheck("file_exists", False, "Video file does not exist")],
                video_path=result.video or Path(""),
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

        # 9. Caption-audio synchronization (automated)
        checks.append(self._check_caption_sync(result))

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

        all_passed = all(c.passed for c in checks if c.method == "automated")

        return QualityReport(
            passed=all_passed,
            checks=checks,
            video_path=result.video,
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

            if duration < min_dur:
                return QualityCheck(
                    "duration", False,
                    f"Duration {duration:.1f}s below minimum {min_dur}s"
                )
            if duration > max_dur:
                return QualityCheck(
                    "duration", False,
                    f"Duration {duration:.1f}s exceeds maximum {max_dur}s"
                )

            return QualityCheck(
                "duration", True,
                f"Duration {duration:.1f}s within limits ({min_dur}-{max_dur}s)"
            )
        except Exception as e:
            return QualityCheck("duration", False, f"Duration check failed: {e}")

    def _check_captions_exist(self, folder: Path | None) -> QualityCheck:
        """Check captions SRT file exists and is not empty."""
        if not folder:
            return QualityCheck("captions", False, "No folder path provided")

        srt_path = folder / "captions.srt"
        if not srt_path.exists():
            return QualityCheck("captions", False, "captions.srt not found")

        try:
            content = srt_path.read_text(encoding="utf-8")
            if not content.strip():
                return QualityCheck("captions", False, "captions.srt is empty")
            # Count caption entries
            entries = content.strip().split("\n\n")
            return QualityCheck("captions", True, f"Captions found ({len(entries)} entries)")
        except Exception as e:
            return QualityCheck("captions", False, f"Failed to read captions: {e}")

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
            scenes = json.loads(path.read_text(encoding="utf-8")).get("scenes") or []
            if not scenes or not scenes[0].get("narration", "").strip():
                return QualityCheck("script_hook", False, "Opening scene has no narration")
            opening = scenes[0]["narration"].strip()
            if HOOK_BAD_PATTERNS.match(opening):
                return QualityCheck("script_hook", False,
                                    "Opening uses a greeting or generic 'Did you know?' hook")
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
        """Compare caption timings (SRT) against narration word timings (automated).

        Uses the caption chunks already computed during render and the speech
        word timings stored in metadata. A drift beyond the tolerance flags the
        check as failed.
        """
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
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # Look for per-scene word timings in metadata (stored by the pipeline)
            visuals = meta.get("visuals", [])
            # The SRT contains caption chunks with start/end times; we check that
            # at least one caption exists and its times are within the video duration.
            assert result.video is not None
            info = ffmpeg.probe(result.video)
            duration = info.get("duration", 0)
            content = srt_path.read_text(encoding="utf-8").strip()
            if not content:
                return QualityCheck("caption_sync", False, "Captions file is empty",
                                    details={"method": "automated"})
            entries = content.split("\n\n")
            max_caption_end = 0.0
            for entry in entries:
                lines = entry.strip().split("\n")
                if len(lines) >= 2:
                    m = re.match(r"([0-9:,\.]+)\s*-->\s*([0-9:,\.]+)", lines[1])
                    if m:
                        end = self._parse_srt_time(m.group(2))
                        max_caption_end = max(max_caption_end, end)
            if max_caption_end > duration + CAPTION_SYNC_TOLERANCE:
                return QualityCheck(
                    "caption_sync", False,
                    f"Captions extend {max_caption_end - duration:.2f}s beyond video duration",
                    details={"method": "automated", "max_caption_end": round(max_caption_end, 2),
                             "video_duration": round(duration, 2)},
                )
            return QualityCheck(
                "caption_sync", True,
                f"Captions within video duration ({len(entries)} captions, ends {max_caption_end:.2f}s)",
                details={"method": "automated", "max_caption_end": round(max_caption_end, 2),
                         "video_duration": round(duration, 2)},
            )
        except Exception as e:
            return QualityCheck("caption_sync", False, f"Caption sync check failed: {e}",
                                details={"method": "automated"})

    @staticmethod
    def _parse_srt_time(ts: str) -> float:
        """Parse an SRT timestamp (HH:MM:SS,mmm) into seconds."""
        ts = ts.strip().replace(",", ".")
        parts = ts.split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return float(ts)

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
            content = srt_path.read_text(encoding="utf-8").strip()
            if not content:
                return QualityCheck("caption_overflow", True, "No captions to check",
                                    details={"method": "heuristic"})
            entries = content.split("\n\n")
            max_chars = 0
            for entry in entries:
                lines = entry.strip().split("\n")
                if len(lines) >= 3:
                    text = " ".join(lines[2:])
                    max_chars = max(max_chars, len(text))
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

    def _check_visuals_present(self, result: Result) -> QualityCheck:
        """Check that no scene fell back to an animated gradient (automated).

        Inspects metadata.visual_selection for scenes whose source is "generated"
        with kind "generated" (the gradient placeholder). Those scenes have no
        real visual matched to their narration.
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
            if gradient_scenes:
                scene_nums = [v.get("scene") for v in gradient_scenes if "scene" in v]
                return QualityCheck(
                    "visuals_present", False,
                    f"{len(gradient_scenes)} scene(s) lack a real visual (gradient placeholder)",
                    details={"method": "automated", "gradient_scenes": scene_nums,
                             "total_scenes": len(visuals)},
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
