"""Quality gate: validates rendered videos before publishing."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import ffmpeg
from .config import ASPECTS, Settings, duration_window
from .pipeline import Result

log = logging.getLogger("purffle")


@dataclass
class QualityCheck:
    """A single quality check result."""
    name: str
    passed: bool
    message: str = ""
    details: dict = field(default_factory=dict)


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
        return f"{len(failures)} check(s) failed: " + "; ".join(f"{c.name}: {c.message}" for c in failures)

    @property
    def failed_checks(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checked_at": self.checked_at,
            "video_path": str(self.video_path),
            "checks": [
                {"name": c.name, "passed": c.passed, "message": c.message, "details": c.details}
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

        # 1. File exists and is not empty
        checks.append(self._check_file_exists(result.video))

        # 2. FFprobe can read the file
        checks.append(self._check_ffprobe_readable(result.video))

        # 3. Video and audio streams present
        checks.append(self._check_streams(result.video))

        # 4. Aspect ratio is vertical 9:16
        checks.append(self._check_aspect_ratio(result.video))

        # 5. Duration within limits
        checks.append(self._check_duration(result.video))

        # 6. Captions file exists
        checks.append(self._check_captions_exist(result.folder))

        # 7. Metadata file exists and is valid
        checks.append(self._check_metadata(result.folder))

        # 8. Render completed successfully (check history status)
        checks.append(self._check_render_status(result))

        all_passed = all(c.passed for c in checks)

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

    def _check_render_status(self, result: Result) -> QualityCheck:
        """Check that render completed successfully (status is 'rendered')."""
        if result.status == "rendered":
            return QualityCheck("render_status", True, f"Render status: {result.status}")
        return QualityCheck("render_status", False, f"Render status: {result.status} (expected 'rendered')")


def run_quality_checks(video_path: Path, folder: Path, settings: Settings) -> QualityReport:
    """Convenience function to run quality checks on a video."""
    gate = QualityGate(settings)
    # Create a minimal Result object for the check
    result = Result(
        ok=True,
        video=video_path,
        folder=folder,
        status="rendered",
    )
    return gate.check(result)
