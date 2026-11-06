"""Tests for the quality gate module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from purffle_shorts.config import ASPECTS, Settings
from purffle_shorts.quality import QualityGate


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
        gate = QualityGate(gate.settings.with_overrides(target_seconds=40))
        mock_probe.return_value = {"duration": 52.0}
        assert gate._check_duration(Path("video.mp4")).passed is True  # 40s + 30%
        mock_probe.return_value = {"duration": 53.0}
        assert gate._check_duration(Path("video.mp4")).passed is False
