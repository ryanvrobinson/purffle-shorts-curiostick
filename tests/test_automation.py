"""Tests for the daily automation controller."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from purffle_shorts.automation import DailyAutomation, DailyRunTracker, RunLock
from purffle_shorts.config import Settings
from purffle_shorts.history import History
from purffle_shorts.pipeline import Result


class TestRunLock:
    """Tests for the RunLock class."""

    def test_lock_acquire_release(self, tmp_path):
        """Test acquiring and releasing a lock."""
        lock_path = tmp_path / "test.lock"
        lock = RunLock(lock_path)

        assert lock.acquire() is True
        assert lock.acquired is True
        assert lock_path.exists()

        lock.release()
        assert lock.acquired is False
        assert not lock_path.exists()

    def test_lock_prevents_second_acquire(self, tmp_path):
        """Test that a second lock cannot be acquired while first is held."""
        lock_path = tmp_path / "test.lock"
        lock1 = RunLock(lock_path)
        lock2 = RunLock(lock_path)

        assert lock1.acquire() is True
        assert lock2.acquire() is False

        lock1.release()
        assert lock2.acquire() is True
        lock2.release()

    def test_lock_context_manager(self, tmp_path):
        """Test lock as context manager."""
        lock_path = tmp_path / "test.lock"

        with RunLock(lock_path) as lock:
            assert lock.acquired is True
            assert lock_path.exists()

        assert not lock_path.exists()


class TestDailyRunTracker:
    """Tests for the DailyRunTracker class."""

    @pytest.fixture
    def history(self, tmp_path):
        return History(tmp_path / "history.db")

    @pytest.fixture
    def tracker(self, history):
        return DailyRunTracker(history)

    def test_start_and_complete_run(self, tracker):
        """Test starting and completing a run."""
        run_date = "2024-01-15"
        run_id = tracker.start_run(run_date)

        assert run_id > 0

        tracker.complete_run(run_id, video_id=1, quality_passed=True,
                           uploaded=True, instagram_published=False)

        run = tracker.get_today_run(run_date)
        assert run is not None
        assert run["status"] == "completed"
        assert run["video_id"] == 1
        assert run["quality_gate_passed"] == 1
        assert run["uploaded"] == 1

    def test_has_completed_run_today(self, tracker):
        """Test checking for completed run."""
        run_date = "2024-01-15"

        assert tracker.has_completed_run_today(run_date) is False

        run_id = tracker.start_run(run_date)
        tracker.complete_run(run_id, video_id=1, quality_passed=True,
                           uploaded=False, instagram_published=False)

        assert tracker.has_completed_run_today(run_date) is True

    def test_skip_run(self, tracker):
        """Test skipping a run."""
        run_date = "2024-01-15"
        tracker.skip_run(run_date, "test reason")

        run = tracker.get_today_run(run_date)
        assert run is not None
        assert run["status"] == "skipped"
        assert run["error"] == "test reason"

    def test_failed_run(self, tracker):
        """Test marking a run as failed."""
        run_date = "2024-01-15"
        run_id = tracker.start_run(run_date)

        tracker.complete_run(run_id, video_id=None, quality_passed=False,
                           uploaded=False, instagram_published=False,
                           error="Generation failed")

        run = tracker.get_today_run(run_date)
        assert run["status"] == "failed"
        assert run["error"] == "Generation failed"


class TestDailyAutomation:
    """Tests for the DailyAutomation class."""

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
    def automation(self, settings):
        return DailyAutomation(settings, dry_run=True, skip_quality=True, skip_upload=True)

    def test_dry_run(self, automation):
        """Test dry run mode."""
        result = automation.run_once()

        assert result.ok is True
        assert "dry run" in result.error.lower()
        assert result.run_date != ""

    def test_duplicate_run_prevention(self, automation):
        """Test that a second run on same day is skipped."""
        result1 = automation.run_once()
        assert result1.ok is True

        result2 = automation.run_once()
        assert result2.ok is True
        assert "already completed" in result2.error

    def test_failed_generation_not_marked_completed(self, settings, tmp_path):
        """Test that failed generation is not marked as completed."""
        automation = DailyAutomation(settings, dry_run=False, skip_quality=True, skip_upload=True)

        with patch.object(automation.studio, 'make') as mock_make:
            mock_make.return_value = Result(
                ok=False,
                error="LLM not configured",
                status="failed"
            )

            result = automation.run_once()

            assert result.ok is False
            assert "LLM not configured" in result.error

            run = automation.tracker.get_today_run(result.run_date)
            assert run is not None
            assert run["status"] == "failed"

    def test_overlapping_run_prevention(self, settings, tmp_path):
        """Test that overlapping runs are prevented."""
        auto1 = DailyAutomation(settings, dry_run=False, skip_quality=True, skip_upload=True)
        auto2 = DailyAutomation(settings, dry_run=False, skip_quality=True, skip_upload=True)

        with patch.object(auto1.studio, 'make') as mock_make:
            def slow_make():
                time.sleep(0.5)
                return Result(ok=True, title="Test", status="rendered")

            mock_make.side_effect = slow_make

            import threading
            result_container = {}

            def run_auto1():
                result_container['result'] = auto1.run_once()

            thread = threading.Thread(target=run_auto1)
            thread.start()

            time.sleep(0.1)

            result2 = auto2.run_once()

            thread.join()

            assert result2.ok is False
            assert "already in progress" in result2.error

    @pytest.mark.skip(reason="Requires full pipeline setup")
    def test_quality_gate_rejection(self, settings):
        """Test that quality gate rejection prevents upload."""
        pass

    @pytest.mark.skip(reason="Requires YouTube API credentials")
    def test_upload_limit_enforcement(self, settings):
        """Test that daily upload limit is enforced."""
        pass

    @pytest.mark.skip(reason="Requires Instagram API credentials")
    def test_instagram_publishing(self, settings):
        """Test Instagram Reels publishing workflow."""
        pass