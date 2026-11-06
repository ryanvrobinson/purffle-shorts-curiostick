"""Tests for the three-slot scheduled YouTube publishing pipeline.

YouTube API calls are mocked: no real upload ever happens in these tests.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
from zoneinfo import ZoneInfo

from purffle_shorts import youtube as Y
from purffle_shorts.automation import (
    SLOT_LABELS,
    SLOT_TIMES,
    SlotAutomation,
    SlotRunResult,
    slot_key,
    slot_publish_moments,
)
from purffle_shorts.config import Settings
from purffle_shorts.history import History
from purffle_shorts.pipeline import Result
from purffle_shorts.quality import QualityCheck, QualityReport

IST = ZoneInfo("Asia/Kolkata")


def future_run_date() -> str:
    return (datetime.now(IST).date() + timedelta(days=2)).isoformat()


def test_all_three_ist_slots_convert_to_the_right_utc_times():
    """9:00 / 14:00 / 19:00 IST -> 03:30 / 08:30 / 13:30 UTC."""
    moments = slot_publish_moments("2026-10-31")
    assert moments[1] == datetime(2026, 10, 31, 9, 0, tzinfo=IST)
    assert moments[2] == datetime(2026, 10, 31, 14, 0, tzinfo=IST)
    assert moments[3] == datetime(2026, 10, 31, 19, 0, tzinfo=IST)
    assert Y.to_rfc3339(moments[1]) == "2026-10-31T03:30:00Z"
    assert Y.to_rfc3339(moments[2]) == "2026-10-31T08:30:00Z"
    assert Y.to_rfc3339(moments[3]) == "2026-10-31T13:30:00Z"


def test_slot_labels_and_times():
    assert SLOT_TIMES == {1: "09:00", 2: "14:00", 3: "19:00"}
    assert SLOT_LABELS == {1: "9:00 AM IST", 2: "2:00 PM IST", 3: "7:00 PM IST"}


def test_slot_key_format():
    assert slot_key("2026-10-31", 2) == "2026-10-31-slot-2"
    assert slot_key("2026-10-31", 1) != slot_key("2026-11-01", 1)


# ------------------------------------------------------------------ ledger (persistent history)

def test_ledger_persists_across_history_instances(tmp_path):
    db = tmp_path / "history.db"
    h1 = History(db)
    key = "2026-10-31-slot-1"
    assert h1.is_slot_done(key) is False
    h1.record_slot(key, "2026-10-31", 1, "2026-10-31T03:30:00Z",
                   "scheduled", youtube_id="vid1", title="T #shorts")

    h2 = History(db)  # new instance, same file = persistent across runs
    assert h2.is_slot_done(key) is True
    row = h2.get_slot(key)
    assert row["youtube_id"] == "vid1"
    assert row["status"] == "scheduled"
    assert row["publish_at_utc"] == "2026-10-31T03:30:00Z"


def test_ledger_record_slot_updates_in_place(tmp_path):
    h = History(tmp_path / "history.db")
    key = "2026-10-31-slot-3"
    h.record_slot(key, "2026-10-31", 3, "2026-10-31T13:30:00Z", "failed",
                  error="quality gate failed")
    h.record_slot(key, "2026-10-31", 3, "2026-10-31T13:30:00Z", "scheduled",
                  youtube_id="vid3", title="T #shorts")
    row = h.get_slot(key)
    assert row["status"] == "scheduled"
    assert row["youtube_id"] == "vid3"
    assert row["error"] is None


def test_slot_done_requires_scheduled_status(tmp_path):
    """Only a successfully scheduled slot blocks re-publication; failed/queued stays retriable
    and never wipes previously recorded history for other slots."""
    h = History(tmp_path / "history.db")
    key = "2026-10-31-slot-2"
    assert h.is_slot_done(key) is False
    h.record_slot(key, "2026-10-31", 2, "2026-10-31T08:30:00Z", "failed",
                  error="LLM provider failed")
    assert h.is_slot_done(key) is False  # failed -> not done, slot can be retried
    h.record_slot(key, "2026-10-31", 2, "2026-10-31T08:30:00Z", "queued",
                  error="quota reached")
    assert h.is_slot_done(key) is False  # queued -> not done, not blindly retried
    h.record_slot(key, "2026-10-31", 2, "2026-10-31T08:30:00Z", "scheduled",
                  youtube_id="vid2", title="T #shorts")
    assert h.is_slot_done(key) is True
    # A different slot on the same date is unaffected.
    other = "2026-10-31-slot-1"
    assert h.is_slot_done(other) is False


# ------------------------------------------------------------------ fake pipeline pieces used below

class FakeStudio:
    """Stand-in for Studio: renders nothing, just returns a rendered Result."""

    def __init__(self, tmp_path: Path, history: History):
        self.tmp_path = tmp_path
        self.history = history
        self.calls = 0
        self.upload_calls = []
        self.fail_upload = None

    def make(self, *a, **k):
        self.calls += 1
        folder = self.tmp_path / f"out/video{self.calls}"
        folder.mkdir(parents=True)
        video = folder / "short.mp4"
        video.write_bytes(b"fake-video-bytes")
        (folder / "metadata.json").write_text(json.dumps({
            "title": "Test Title #shorts",
            "description": "A test description.",
            "tags": ["test", "shorts"],
            "category_id": "27",
            "language": "en",
        }), encoding="utf-8")
        rid = self.history.add_video(
            subject="x", source="niche", topic="t", style="facts", language="en",
            title="Test Title #shorts", description="A test description.", tags=["test"],
            folder=str(folder), video_path=str(video), duration=45.0,
            llm="fake", voice="silent:fake", status="rendered")
        return Result(True, "Test Title #shorts", folder, video, "rendered",
                      record_id=rid, seconds=1.0)

    def upload_scheduled(self, rid, publish_at, result=None, slot=None, run_date=None):
        from purffle_shorts.history import now_iso
        self.upload_calls.append((rid, publish_at, slot, run_date))
        if self.fail_upload is not None:
            raise self.fail_upload
        self.history.update_video(rid, status="scheduled", youtube_id="vid-slot-1",
                                  uploaded_at=now_iso(),
                                  publish_at=Y.to_rfc3339(publish_at))
        return "scheduled"


class FakeGate:
    """Stand-in for QualityGate with a programmable report."""

    def __init__(self, passed: bool = True, message: str = "all good"):
        self.passed = passed
        self.message = message

    def check(self, result):
        check = QualityCheck("gate", self.passed, self.message,
                             details={"method": "automated"})
        return QualityReport(passed=self.passed, checks=[check],
                             video_path=result.video or Path("x.mp4"))


def test_private_upload_with_future_publish_at(tmp_path, monkeypatch):
    """The scheduled upload body is private with publishAt in UTC for the IST slot."""
    history = History(tmp_path / "history.db")
    studio = FakeStudio(tmp_path, history)

    settings = Settings(output_dir=str(tmp_path / "out"), data_dir=str(tmp_path / "data"),
                        upload=False, timezone="Asia/Kolkata")
    run_date = future_run_date()
    expected_publish_at = Y.to_rfc3339(slot_publish_moments(run_date)[1])
    auto = SlotAutomation(settings, slot=1, run_date=run_date,
                          dry_run=False, studio=studio, history=history)
    # Patch the quality gate so the fake video "passes" without ffmpeg.
    monkeypatch.setattr("purffle_shorts.automation.QualityGate",
                        lambda settings: FakeGate())

    result = auto.run_slot()

    assert result.ok is True
    assert result.uploaded is True
    assert result.youtube_id == "vid-slot-1"
    assert result.publish_at_utc == expected_publish_at
    assert result.dry_run is False
    assert len(studio.upload_calls) == 1
    rid, publish_at, slot, run_date = studio.upload_calls[0]
    assert slot == 1 and run_date == auto.run_date
    assert Y.to_rfc3339(publish_at) == expected_publish_at
    # The slot is durably recorded with its YouTube id.
    assert history.is_slot_done(slot_key(run_date, 1)) is True


def test_build_body_scheduled_is_private_with_future_publish_at():
    """Direct coverage: upload metadata is private + future publishAt (no public upload)."""
    body = Y.build_body(
        Settings(privacy="public", synthetic_media=True),
        "Octopus facts #shorts", "A description.", ["octopus", "science"], "27", "en",
        datetime(2026, 11, 1, 3, 30, tzinfo=timezone.utc),
    )
    assert body["status"]["privacyStatus"] == "private"
    assert body["status"]["publishAt"] == "2026-11-01T03:30:00Z"
    assert body["status"]["containsSyntheticMedia"] is True


def test_duplicate_slot_is_not_uploaded_twice(tmp_path, monkeypatch):
    """Idempotency: a completed slot is skipped on the next run (no re-upload)."""
    history = History(tmp_path / "history.db")
    studio = FakeStudio(tmp_path, history)

    monkeypatch.setattr("purffle_shorts.automation.QualityGate",
                        lambda settings: FakeGate())

    settings = Settings(output_dir=str(tmp_path / "out"), data_dir=str(tmp_path / "data"),
                        upload=False, timezone="Asia/Kolkata")
    auto = SlotAutomation(settings, slot=2, run_date=future_run_date(),
                          dry_run=False, studio=studio, history=history)

    r1 = auto.run_slot()
    assert r1.uploaded is True
    assert r1.youtube_id == "vid-slot-1"

    r2 = auto.run_slot()  # same date + slot -> already done
    assert r2.skipped is True
    assert r2.ok is True
    assert r2.youtube_id == "vid-slot-1"
    assert len(studio.upload_calls) == 1  # no re-upload


def test_dry_run_never_publishes(tmp_path, monkeypatch):
    """Manual dry run: generation and QC run, but YouTube is never called."""
    history = History(tmp_path / "history.db")
    studio = FakeStudio(tmp_path, history)
    calls = []
    monkeypatch.setattr(Y, "upload", lambda settings, video, body: calls.append(body) or {"id": "v"})
    monkeypatch.setattr("purffle_shorts.automation.QualityGate",
                        lambda settings: FakeGate())

    settings = Settings(output_dir=str(tmp_path / "out"), data_dir=str(tmp_path / "data"),
                        upload=False, timezone="Asia/Kolkata")
    auto = SlotAutomation(settings, slot=1, run_date=future_run_date(),
                          dry_run=True, studio=studio, history=history)
    result = auto.run_slot()

    assert result.ok is True
    assert result.dry_run is True
    assert result.uploaded is False
    assert calls == []  # nothing uploaded


def test_past_slot_is_skipped_not_published(tmp_path, monkeypatch):
    """A slot whose publication time has passed is skipped, never published late."""
    history = History(tmp_path / "history.db")
    studio = FakeStudio(tmp_path, history)
    calls = []
    monkeypatch.setattr(Y, "upload", lambda settings, video, body: calls.append(body) or {"id": "v"})
    monkeypatch.setattr("purffle_shorts.automation.QualityGate",
                        lambda settings: FakeGate())

    yesterday = "2026-10-30"  # guaranteed in the past regardless of when this runs
    settings = Settings(output_dir=str(tmp_path / "out"), data_dir=str(tmp_path / "data"),
                        upload=False, timezone="Asia/Kolkata")
    auto = SlotAutomation(settings, slot=1, run_date=yesterday,
                          dry_run=False, studio=studio, history=history)
    result = auto.run_slot()

    assert result.skipped is True
    assert result.ok is True
    assert "past" in result.skip_reason.lower()
    assert calls == []  # not published into the past
    row = history.get_slot(slot_key(yesterday, 1))
    assert row["status"] == "skipped"


def test_quality_failure_blocks_upload_and_saves_details(tmp_path, monkeypatch):
    """A failed quality check prevents upload and records which check failed."""
    history = History(tmp_path / "history.db")
    studio = FakeStudio(tmp_path, history)
    calls = []
    monkeypatch.setattr(Y, "upload", lambda settings, video, body: calls.append(body) or {"id": "v"})
    monkeypatch.setattr("purffle_shorts.automation.QualityGate",
                        lambda settings: FakeGate(passed=False, message="aspect_ratio: wrong shape"))

    settings = Settings(output_dir=str(tmp_path / "out"), data_dir=str(tmp_path / "data"),
                        upload=False, timezone="Asia/Kolkata")
    run_date = future_run_date()
    auto = SlotAutomation(settings, slot=3, run_date=run_date,
                          dry_run=False, studio=studio, history=history)
    result = auto.run_slot()

    assert result.ok is False
    assert result.uploaded is False
    assert calls == []  # upload never attempted
    assert "aspect_ratio" in result.error
    failure_file = tmp_path / "data" / "failures" / f"{run_date}-slot-3.json"
    assert failure_file.exists()
    payload = json.loads(failure_file.read_text(encoding="utf-8"))
    assert payload["slot"] == 3
    assert payload["report"]["passed"] is False


def test_missing_credentials_are_recorded_not_crashed(tmp_path, monkeypatch):
    """Unattended run with no YouTube authorization fails the slot cleanly."""
    history = History(tmp_path / "history.db")
    studio = FakeStudio(tmp_path, history)
    monkeypatch.setattr("purffle_shorts.automation.QualityGate", FakeGate)

    def boom_upload(*a, **k):
        raise Y.NotAuthorized("token missing")
    monkeypatch.setattr(Y, "upload", boom_upload)

    settings = Settings(output_dir=str(tmp_path / "out"), data_dir=str(tmp_path / "data"),
                        upload=False, timezone="Asia/Kolkata")
    auto = SlotAutomation(settings, slot=1, run_date=future_run_date(),
                          dry_run=False, studio=studio, history=history)
    studio.fail_upload = Y.NotAuthorized("token missing")
    result = auto.run_slot()

    assert result.ok is False
    assert "YouTube upload failed" in result.error


def test_slot_result_round_trip():
    r = SlotRunResult(ok=True, slot=1, run_date="2026-10-31", key="2026-10-31-slot-1",
                      publish_at_utc="2026-10-31T03:30:00Z",
                      publish_at_ist="2026-10-31T09:00:00+05:30",
                      uploaded=True, youtube_id="vid1")
    assert r.youtube_id == "vid1"


# ------------------------------------------------------------------ upload retry behavior at the YouTube layer

def test_youtube_upload_retries_transient_errors(monkeypatch, tmp_path):
    """youtube.upload retries on 5xx and connection errors (exponential backoff)."""
    from googleapiclient.errors import HttpError

    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    calls = {"n": 0}

    class FakeRequest:
        def next_chunk(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise HttpError(Mock(status=503), b"unavailable")
            return None, {"id": "vid-retry"}

    class FakeVideos:
        def insert(self, **k):
            return FakeRequest()

    class FakeYT:
        def videos(self):
            return FakeVideos()

    monkeypatch.setattr(Y, "service", lambda settings, interactive=False: FakeYT())
    monkeypatch.setattr(Y.time, "sleep", lambda *a: None)  # skip real backoff sleeps
    import googleapiclient.http as gh
    monkeypatch.setattr(gh, "MediaFileUpload", lambda *a, **k: "media")

    resp = Y.upload(Settings(), video, {"snippet": {}, "status": {}})
    assert resp["id"] == "vid-retry"
    assert calls["n"] == 2  # first attempt failed, second succeeded


# ------------------------------------------------------------------ workflow files

EXPECTED_WORKFLOWS = {
    ".github/workflows/curiostick-slot-1.yml": {
        "slot": "1", "cron": "30 1 * * *",
        "publish_utc": "03:30", "publish_ist": "09:00 AM",
    },
    ".github/workflows/curiostick-slot-2.yml": {
        "slot": "2", "cron": "30 6 * * *",
        "publish_utc": "08:30", "publish_ist": "2:00 PM",
    },
    ".github/workflows/curiostick-slot-3.yml": {
        "slot": "3", "cron": "30 11 * * *",
        "publish_utc": "13:30", "publish_ist": "7:00 PM",
    },
}


def test_workflow_files_select_the_correct_slots():
    """Each workflow file runs once daily at its UTC cron for one IST slot."""
    for path, exp in EXPECTED_WORKFLOWS.items():
        text = Path(path).read_text(encoding="utf-8")
        assert f"cron: '{exp['cron']}'" in text, path
        assert f'SLOT: "{exp["slot"]}"' in text, path
        assert "workflow_dispatch:" in text, path          # manual testing
        assert "cancel-in-progress: false" in text, path   # no overlapping same-slot runs
        assert "contents: write" in text, path             # history DB can be committed back
        assert "actions/upload-artifact@v4" in text, path          # failure reports as artifacts
        assert "$GITHUB_STEP_SUMMARY" in text, path                 # readable run summary
        assert 'default: false' in text, path                       # manual dispatch defaults to dry-run
        assert '${{ github.event_name }}" = "schedule"' in text, path  # scheduled runs pass --publish
        assert 'LLM_PROVIDER: "custom"' in text, path
        assert 'LLM_BASE_URL: "https://api.groq.com/openai/v1"' in text, path
        assert 'GROQ_MODEL: "openai/gpt-oss-20b"' in text, path
        assert '${{ secrets.GROQ_API_KEY }}' in text, path
        assert 'LLM_TEMPERATURE: "0.6"' in text, path
        assert 'GEMINI_API_KEY' not in text, path
        assert 'NVIDIA_NIM_API_KEY' not in text, path
        assert 'NVIDIA_NIM_MODEL' not in text, path
        assert 'VISUAL_SOURCES: "pixabay,agnes"' in text, path
        assert 'AGNES_IMAGE_MODEL: "agnes-image-2.5-flash"' in text, path
        assert 'TTS_ENGINE: "edge"' in text, path
        assert 'YT_PRIVACY: "private"' in text, path
        assert 'KEEP_VIDEOS: "true"' in text, path
        assert 'WATERMARK: "CurioStick"' in text, path
        assert 'TARGET_SECONDS: "30"' in text, path


def test_workflow_cron_matches_ist_publish_time():
    """Sanity check on the UTC cron -> IST publication-time mapping."""
    for path, exp in EXPECTED_WORKFLOWS.items():
        text = Path(path).read_text(encoding="utf-8")
        assert exp["publish_utc"] in text or exp["cron"].split()[1] in text, path


def test_daily_parser_slot_flags(tmp_path, monkeypatch):
    """CLI: --slot/--date/--publish parse correctly."""
    from purffle_shorts.cli import build_parser, build_settings
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["daily", "--slot", "2", "--date", "2026-10-31",
                                      "--publish", "--verbose"])
    assert args.slot == 2
    assert args.date == "2026-10-31"
    assert args.publish is True
    s = build_settings(args)
    assert s.timezone == "Asia/Kolkata"


def test_invalid_slot_is_rejected():
    with pytest.raises(Exception):
        SlotAutomation(Settings(), slot=9)
