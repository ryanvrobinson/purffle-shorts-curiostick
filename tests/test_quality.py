"""Tests for the quality gate module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from purffle_shorts.config import ASPECTS, Settings
from purffle_shorts.quality import (
    CATEGORY_MINIMUMS,
    MINIMUM_QUALITY_SCORE,
    QUALITY_WEIGHTS,
    QualityCheck,
    QualityGate,
)


class TestQualityGate:
    """Tests for the QualityGate class."""

    @pytest.fixture
    def settings(self, tmp_path):
        font = "C:/Windows/Fonts/arialbd.ttf"
        return Settings(
            output_dir=str(tmp_path / "out"),
            data_dir=str(tmp_path / "data"),
            caption_font=font,
            upload=False,
            offline=True,
            tts_engine="silent",
        )

    @pytest.fixture
    def gate(self, settings):
        return QualityGate(settings)

    @pytest.fixture
    def mock_video(self, tmp_path):
        """Create a mock video file for testing."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")
        return video_path

    @pytest.fixture
    def mock_folder(self, tmp_path):
        """Create a mock folder with metadata and captions."""
        folder = tmp_path / "test_video"
        folder.mkdir()

        import json
        meta = {
            "title": "Test Video #shorts",
            "duration": 30.0,
            "resolution": "1080x1920",
            "created_at": "2024-01-15T10:00:00+00:00",
            "tags": ["test", "shorts"]
        }
        (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Test caption

2
00:00:05,000 --> 00:00:10,000
Another caption
"""
        (folder / "captions.srt").write_text(srt_content, encoding="utf-8")

        return folder

    def test_check_file_exists_pass(self, gate, mock_video):
        """Test file exists check passes for valid file."""
        check = gate._check_file_exists(mock_video)
        assert check.passed is True
        assert "bytes" in check.message

    def test_script_and_visual_mapping_checks(self, gate, tmp_path):
        import json

        folder = tmp_path / "story"
        folder.mkdir()
        (folder / "script.json").write_text(json.dumps({
            "topic": "Chicxulub",
            "title": "The crater that ended the dinosaurs",
            "scenes": [{
                "narration": "A buried crater points to the impact.",
                "search_query": "Chicxulub crater",
                "image_prompt": "A realistic aerial view of the Chicxulub crater region.",
            }],
        }), encoding="utf-8")
        (folder / "metadata.json").write_text(json.dumps({
            "visuals": [{"source": "pixabay", "kind": "video", "query": "Chicxulub crater"}],
        }), encoding="utf-8")

        assert gate._check_script_complete(folder).passed
        assert gate._check_scene_visual_mapping(folder).passed

        (folder / "metadata.json").write_text(json.dumps({"visuals": []}), encoding="utf-8")
        mapping = gate._check_scene_visual_mapping(folder)
        assert not mapping.passed and "one selected visual" in mapping.message

        (folder / "script.json").write_text(json.dumps({
            "scenes": [{"narration": "Did you know octopuses have three hearts?",
                        "search_query": "octopus hearts", "image_prompt": "octopus"}],
        }), encoding="utf-8")
        hook = gate._check_hook(folder)
        assert not hook.passed and "generic" in hook.message

    @patch("purffle_shorts.quality.subprocess.run")
    def test_audio_quality_rejects_silence_and_accepts_audible_signal(self, mock_run, gate, mock_video):
        mock_run.return_value = SimpleNamespace(returncode=0, stderr="max_volume: -inf dB")
        silent = gate._check_audio_audible(mock_video)
        assert not silent.passed and "inaudible" in silent.message

        mock_run.return_value = SimpleNamespace(returncode=0, stderr="max_volume: -18.4 dB")
        audible = gate._check_audio_audible(mock_video)
        assert audible.passed and "signal detected" in audible.message

    def test_check_file_exists_fail_missing(self, gate, mock_folder):
        """Test file exists check fails for missing file."""
        missing_path = mock_folder / "missing.mp4"
        check = gate._check_file_exists(missing_path)
        assert check.passed is False
        assert "does not exist" in check.message

    def test_check_file_exists_fail_empty(self, gate, tmp_path, mock_folder):
        """Test file exists check fails for empty file."""
        empty_path = tmp_path / "empty.mp4"
        empty_path.write_bytes(b"")
        check = gate._check_file_exists(empty_path)
        assert check.passed is False
        assert "empty" in check.message

    @patch("purffle_shorts.ffmpeg.probe")
    def test_check_ffprobe_readable_pass(self, mock_probe, gate, mock_video, mock_folder):
        """Test ffprobe readable check passes."""
        mock_probe.return_value = {
            "width": 1080, "height": 1920, "duration": 30.0,
            "has_video": True, "has_audio": True
        }
        check = gate._check_ffprobe_readable(mock_video)
        assert check.passed is True
        assert "FFprobe readable" in check.message

    @patch("purffle_shorts.ffmpeg.probe")
    def test_check_ffprobe_readable_fail(self, mock_probe, gate, mock_video, mock_folder):
        """Test ffprobe readable check fails."""
        mock_probe.side_effect = Exception("FFprobe error")
        check = gate._check_ffprobe_readable(mock_video)
        assert check.passed is False
        assert "FFprobe failed" in check.message

    @pytest.mark.parametrize("aspect,resolution", list(ASPECTS.items()))
    @patch("purffle_shorts.ffmpeg.probe")
    def test_aspect_ratio_accepts_every_supported_output_shape(self, mock_probe, settings, aspect, resolution):
        """The gate must not hardcode 9:16; the requested output shape is the contract."""
        mock_probe.return_value = {"width": resolution[0], "height": resolution[1]}
        gate = QualityGate(settings.with_overrides(resolution=resolution))
        check = gate._check_aspect_ratio(Path("video.mp4"))
        assert check.passed is True, check.message
        assert aspect in check.message

    @patch("purffle_shorts.ffmpeg.probe")
    def test_aspect_ratio_rejects_a_different_supported_shape(self, mock_probe, gate):
        """A 9:16 project must reject a 16:9 render even though 16:9 is itself supported."""
        mock_probe.return_value = {"width": 1920, "height": 1080}
        check = gate._check_aspect_ratio(Path("video.mp4"))
        assert check.passed is False
        assert "9:16" in check.message

    @patch("purffle_shorts.ffmpeg.probe")
    def test_aspect_ratio_tolerates_scaled_output(self, mock_probe, gate):
        """A 9:16 render at 360x640 is the same shape as 1080x1920."""
        mock_probe.return_value = {"width": 360, "height": 640}
        check = gate._check_aspect_ratio(Path("video.mp4"))
        assert check.passed is True

    @patch("purffle_shorts.ffmpeg.probe")
    def test_duration_uses_the_strict_short_window(self, mock_probe, gate):
        """Target 40s uses the Short window: hard max is 40.0s, no tolerance above it."""
        gate = QualityGate(gate.settings.with_overrides(target_seconds=40))
        mock_probe.return_value = {"duration": 30.0}
        check = gate._check_duration(Path("video.mp4"))
        assert check.passed is True  # under preferred minimum 36s, advisory only
        mock_probe.return_value = {"duration": 35.7}
        assert gate._check_duration(Path("video.mp4")).passed is True  # under 36s, advisory only
        mock_probe.return_value = {"duration": 39.9}
        assert gate._check_duration(Path("video.mp4")).passed is True  # just under the cap
        mock_probe.return_value = {"duration": 40.0}
        assert gate._check_duration(Path("video.mp4")).passed is True  # exactly at the cap
        mock_probe.return_value = {"duration": 40.1}
        assert gate._check_duration(Path("video.mp4")).passed is False  # exceeds hard maximum

    @patch("purffle_shorts.ffmpeg.probe")
    def test_short_cap_is_target_not_40(self, mock_probe, gate):
        """With TARGET_SECONDS=30, the QC window is 27-30s (90% target to target).
        Nothing above the target ever passes QC when short bounds are active."""
        gate = QualityGate(gate.settings.with_overrides(target_seconds=30))
        for duration in (30.0, 27.0, 27.5):
            mock_probe.return_value = {"duration": duration}
            assert gate._check_duration(Path("video.mp4")).passed is True, duration
        for duration in (30.1, 32.1, 35.7, 39.9, 40.0, 45.0):
            mock_probe.return_value = {"duration": duration}
            assert gate._check_duration(Path("video.mp4")).passed is False, duration

    def test_all_three_workflow_slots_respect_target_maximum(self):
        """Every CurioStick slot loads the same TARGET_SECONDS, and the QC hard
        cap each slot enforces is the 40s format maximum -- not the target itself."""
        import re

        from purffle_shorts.config import short_duration_bounds
        targets = {}
        tests_root = Path(__file__).resolve().parent.parent
        for slot in (1, 2, 3):
            path = tests_root / f".github/workflows/curiostick-slot-{slot}.yml"
            match = re.search(
                r'^\s*TARGET_SECONDS: "(\d+)"\s*$',
                path.read_text(encoding="utf-8"), re.MULTILINE,
            )
            assert match is not None, f"slot {slot} does not set TARGET_SECONDS"
            targets[slot] = int(match.group(1))
        assert len(set(targets.values())) == 1, targets
        target = targets[1]
        _, maximum = short_duration_bounds(target)
        assert maximum == float(target)

    @patch("purffle_shorts.ffmpeg.probe")
    def test_duration_window_is_never_impossible(self, mock_probe, settings):
        """A short request is clamped so the quality window has a valid minimum and maximum."""
        gate = QualityGate(settings.with_overrides(target_seconds=10))
        assert gate.settings.target_seconds == 15
        mock_probe.return_value = {"duration": 15.0}
        check = gate._check_duration(Path("video.mp4"))
        assert check.passed is True
        assert "15-19s" in check.message

    @patch("purffle_shorts.ffmpeg.probe")
    def test_duration_uses_the_shared_target_window(self, mock_probe, gate):
        """Targets above the Short range use the general +/-30% window."""
        gate = QualityGate(gate.settings.with_overrides(target_seconds=60))
        mock_probe.return_value = {"duration": 78.0}
        assert gate._check_duration(Path("video.mp4")).passed is True  # 60 + 30%
        mock_probe.return_value = {"duration": 79.0}
        assert gate._check_duration(Path("video.mp4")).passed is False


def test_srt_validation_accepts_valid_nonpunctuated_captions(tmp_path):
    folder = tmp_path / "captions"
    folder.mkdir()
    (folder / "captions.srt").write_bytes(
        b"1\r\n00:00:00,000 --> 00:00:01,000\r\nOctopuses have three hearts\r\n\r\n"
        b"2\r\n00:00:02,000 --> 00:00:03,000\r\nTheir blood is blue\r\n"
    )

    check = QualityGate(Settings())._check_captions_exist(folder, duration=4)

    assert check.passed
    assert "2 entries" in check.message


@pytest.mark.parametrize(("caption", "duration", "message"), [
    ("1\n-00:00:01,000 --> 00:00:02,000\nNegative start", 10, "before zero"),
    ("1\n00:00:02,000 --> 00:00:01,000\nReversed", 10, "before or at"),
    ("1\n00:00:01,000 --> 00:00:01,000\nZero interval", 10, "before or at"),
    ("1\n00:00:09,000 --> 00:00:11,000\nToo late", 10, "beyond video duration"),
    ("1\n00:00:00,000 --> 00:00:01,000\nFirst\n\n2\n00:00:05,000 --> 00:00:06,000\nGap", 10,
     "excessive"),
    ("1\n00:00:00.000 --> 00:00:01.000\nBad timestamp", 10, "invalid timestamp"),
])
def test_srt_validation_rejects_invalid_timing(tmp_path, caption, duration, message):
    folder = tmp_path / "captions"
    folder.mkdir()
    (folder / "captions.srt").write_text(caption, encoding="utf-8")

    check = QualityGate(Settings())._check_captions_exist(folder, duration=duration)

    assert not check.passed
    assert message in check.message


def test_quality_report_uses_weighted_score_category_floors_and_critical_blockers():
    checks = [QualityCheck(name, True) for name in QUALITY_WEIGHTS]
    passing = QualityGate._report(checks, Path("video.mp4"))
    assert passing.passed
    assert passing.score == 100
    assert passing.category_minimums == CATEGORY_MINIMUMS
    assert passing.to_dict()["minimum_score"] == MINIMUM_QUALITY_SCORE

    advisory_loss = [QualityCheck(c.name, c.name != "research_verification") for c in checks]
    assert QualityGate._report(advisory_loss, Path("video.mp4")).passed

    missing_audio = [QualityCheck(c.name, c.name != "audio_present") for c in checks]
    blocked = QualityGate._report(missing_audio, Path("video.mp4"))
    assert not blocked.passed
    assert "audio_present" in blocked.critical_failures
    assert blocked.score >= MINIMUM_QUALITY_SCORE
