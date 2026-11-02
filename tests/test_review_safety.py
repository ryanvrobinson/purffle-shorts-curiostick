from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from purffle_shorts import cli, config
from purffle_shorts.automation import SlotAutomation, SlotRunResult
from purffle_shorts.history import History
from purffle_shorts.pipeline import Result, Studio


def _valid_video(path: Path) -> None:
    """Write a tiny valid MP4 (video + audio streams) so upload validation passes."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=160x90:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_upload_is_off_by_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UPLOAD", raising=False)
    monkeypatch.setattr(config, "load_env", lambda _path=None: None)

    assert config.Settings.from_env().upload is False


def test_make_command_forces_review_only_even_with_legacy_upload_setting(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    captured = {}

    class FakeStudio:
        def __init__(self, settings):
            captured["upload_setting"] = settings.upload

        def make(self, *args, **kwargs):
            captured["make_kwargs"] = kwargs
            return Result(ok=False, error="test stop")

    monkeypatch.setattr("purffle_shorts.pipeline.Studio", FakeStudio)
    args = cli.build_parser().parse_args(["make", "--topic", "A test topic"])

    assert cli.cmd_make(args, config.Settings(upload=True)) == 1
    assert captured["upload_setting"] is False


def test_upload_review_persists_only_exact_confirmation(monkeypatch, tmp_path, capsys):
    folder = tmp_path / "video"
    folder.mkdir()
    video = folder / "short.mp4"
    video.write_bytes(b"video")
    (folder / "metadata.json").write_text(json.dumps({"visuals": [{"source": "Pixabay"}]}), encoding="utf-8")
    (folder / "script.json").write_text(
        json.dumps({"scenes": [{"narration": "A checkable narration."}]}), encoding="utf-8")
    (folder / "captions.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nA checkable narration.\n",
                                          encoding="utf-8")
    history = History(tmp_path / "history.db")
    record_id = history.add_video(title="A test title", description="A factual description.",
                                  tags=["#shorts"], source="manual", folder=str(folder),
                                  video_path=str(video), status="rendered")
    rec = history.get(record_id)
    report = SimpleNamespace(
        passed=True,
        summary="All checks passed",
        checks=[SimpleNamespace(name="file_exists", passed=True, message="present")],
    )
    monkeypatch.setattr("purffle_shorts.quality.run_quality_checks", lambda *_args: report)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")

    assert cli._review_upload(rec, config.Settings(), history=history) is False
    assert history.get(record_id)["approval_status"] == "pending"
    assert cli._review_upload(rec, config.Settings(), history=history, dry_run=True) is True
    assert history.get(record_id)["approval_status"] == "pending"
    output = capsys.readouterr().out
    assert "factual warning" in output.lower()
    assert "Rights warning" in output
    assert "no approval, upload, or schedule was created" in output

    monkeypatch.setattr("builtins.input", lambda _prompt: "PUBLISH")
    assert cli._review_upload(rec, config.Settings(), history=history) is True
    approved = history.get(record_id)
    assert approved["approval_status"] == "approved" and approved["approved_at"]
    assert json.loads(approved["approval_json"])["platform"] == "youtube"


def test_upload_cli_exposes_safe_dry_run():
    args = cli.build_parser().parse_args(["upload", "--id", "3", "--dry-run"])
    assert args.id == 3 and args.dry_run is True


def test_upload_cli_dry_run_never_calls_uploader(monkeypatch):
    class FakeHistory:
        def get(self, _record_id):
            return {"id": 3}

        def scheduled_times(self):
            return []

    class FakeStudio:
        def __init__(self, _settings):
            self.history = FakeHistory()

        def upload_record(self, _record_id):
            raise AssertionError("dry run must not call the uploader")

    monkeypatch.setattr("purffle_shorts.pipeline.Studio", FakeStudio)
    monkeypatch.setattr(cli, "_review_upload", lambda *args, **kwargs: True)
    args = cli.build_parser().parse_args(["upload", "--id", "3", "--dry-run"])
    assert cli.cmd_upload(args, config.Settings()) == 0


def test_upload_record_requires_saved_approval(settings, tmp_path, monkeypatch):
    video = tmp_path / "short.mp4"
    _valid_video(video)
    studio = Studio(settings)
    record_id = studio.history.add_video(
        title="Draft", folder=str(tmp_path), video_path=str(video), status="rendered")
    monkeypatch.setattr("purffle_shorts.youtube.upload",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unapproved upload")))

    assert studio.upload_record(record_id) == "approval-required"
    assert studio.history.get(record_id)["approval_status"] == "pending"


def test_upload_record_refuses_known_uploaded_video(settings, tmp_path, monkeypatch):
    video = tmp_path / "short.mp4"
    _valid_video(video)
    studio = Studio(settings)
    record_id = studio.history.add_video(
        title="Already published",
        folder=str(tmp_path),
        video_path=str(video),
        status="uploaded",
        youtube_id="existing-video-id",
    )
    monkeypatch.setattr("purffle_shorts.youtube.upload",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("duplicate upload")))

    assert studio.upload_record(record_id) == "already-uploaded"


# ------------------------------------------------------------------ upload-path regression tests
# These tests mock the YouTube API so no real upload ever happens. They cover
# the required behaviours: a successful upload records the video ID and status,
# an API failure is recorded as "failed", missing OAuth raises a clear error,
# a duplicate upload is refused, and "rendered" / "uploaded" / "scheduled" are
# distinct states (a scheduled upload is private until its publishAt time).


def _approve(studio: Studio, record_id: int) -> None:
    studio.history.approve_video(record_id, "2026-01-01T00:00:00+00:00", {"platform": "youtube"})


def _mock_upload(monkeypatch, response: dict | None = None, exc: Exception | None = None) -> None:
    def fake_upload(s, video_path, body):
        if exc is not None:
            raise exc
        return response or {"id": "vid-mock"}
    monkeypatch.setattr("purffle_shorts.youtube.upload", fake_upload)
    monkeypatch.setattr(
        "purffle_shorts.youtube.next_publish_slot", lambda s, t, n=None: None)
    monkeypatch.setattr(
        "purffle_shorts.youtube.build_body", lambda *a, **k: {"snippet": {}, "status": {"privacyStatus": "public"}})


def test_upload_record_success_records_video_id_and_status(settings, tmp_path, monkeypatch):
    """Successful upload: video ID recorded, status 'uploaded', no duplicate on retry."""
    video = tmp_path / "short.mp4"
    _valid_video(video)
    studio = Studio(settings)
    record_id = studio.history.add_video(
        title="T", folder=str(tmp_path), video_path=str(video), status="rendered")
    _approve(studio, record_id)
    _mock_upload(monkeypatch, {"id": "vid-abc-123"})

    status = studio.upload_record(record_id)
    assert status == "uploaded"
    rec = studio.history.get(record_id)
    assert rec["youtube_id"] == "vid-abc-123"
    assert rec["status"] == "uploaded"
    assert rec["error"] is None
    # Duplicate upload is refused, the original video ID is preserved.
    assert studio.upload_record(record_id) == "already-uploaded"
    assert studio.history.get(record_id)["youtube_id"] == "vid-abc-123"


def test_upload_record_api_failure_records_failed_status(settings, tmp_path, monkeypatch):
    """YouTube API failure: status 'failed', clear error recorded, no video ID."""
    video = tmp_path / "short.mp4"
    _valid_video(video)
    studio = Studio(settings)
    record_id = studio.history.add_video(
        title="T", folder=str(tmp_path), video_path=str(video), status="rendered")
    _approve(studio, record_id)
    _mock_upload(monkeypatch, exc=Exception("YouTube rejected the upload"))

    status = studio.upload_record(record_id)
    assert status == "failed"
    rec = studio.history.get(record_id)
    assert rec["status"] == "failed"
    assert rec["youtube_id"] is None
    assert "YouTube rejected the upload" in rec["error"]


def test_upload_record_missing_credentials_raises_clear_error(settings, tmp_path, monkeypatch):
    """Missing OAuth credentials: NotAuthorized propagates with an actionable message."""
    from purffle_shorts.youtube import NotAuthorized
    video = tmp_path / "short.mp4"
    _valid_video(video)
    studio = Studio(settings)
    record_id = studio.history.add_video(
        title="T", folder=str(tmp_path), video_path=str(video), status="rendered")
    _approve(studio, record_id)
    monkeypatch.setattr(
        "purffle_shorts.youtube.upload",
        lambda *a, **k: (_ for _ in ()).throw(NotAuthorized("refresh token missing")))

    with pytest.raises(NotAuthorized, match="refresh token missing"):
        studio.upload_record(record_id)


def test_upload_record_invalid_video_never_attempts_upload(settings, tmp_path, monkeypatch, caplog):
    """Unusable rendered video: upload is not attempted; record marked failed."""
    video = tmp_path / "short.mp4"
    video.write_bytes(b"not a video")
    studio = Studio(settings)
    record_id = studio.history.add_video(
        title="T", folder=str(tmp_path), video_path=str(video), status="rendered")
    _approve(studio, record_id)
    called = []
    monkeypatch.setattr("purffle_shorts.youtube.upload",
                        lambda *a, **k: called.append(True) or {"id": "x"})

    status = studio.upload_record(record_id)
    assert status == "missing"
    assert not called  # YouTube upload never invoked
    rec = studio.history.get(record_id)
    assert rec["status"] == "failed"


def test_upload_scheduled_success_is_private_and_records_video_id(settings, tmp_path, monkeypatch):
    """Scheduled upload: private + future publishAt, video ID recorded, status 'scheduled' (not 'published')."""
    from datetime import datetime, timezone
    import purffle_shorts.youtube as YT
    video = tmp_path / "short.mp4"
    _valid_video(video)
    studio = Studio(settings)
    record_id = studio.history.add_video(
        title="T", folder=str(tmp_path), video_path=str(video), status="rendered")
    publish_at = datetime(2026, 11, 2, 13, 30, tzinfo=timezone.utc)
    captured = {}

    monkeypatch.setattr(YT, "find_existing_video", lambda yt, marker: None)
    monkeypatch.setattr(YT, "service", lambda s, interactive=False: object())

    def fake_upload(s, video_path, body):
        captured["body"] = body
        return {"id": "vid-sched-999"}
    monkeypatch.setattr(YT, "upload", fake_upload)
    monkeypatch.setattr(YT, "to_rfc3339", lambda dt: "2026-11-02T13:30:00Z")
    monkeypatch.setattr(
        YT, "build_body",
        lambda s, title, description, tags, category_id, language, publish_at=None: {
            "snippet": {},
            "status": {"privacyStatus": "private", "publishAt": "2026-11-02T13:30:00Z"},
        },
    )

    status = studio.upload_scheduled(record_id, publish_at, slot=1, run_date="2026-11-02")
    assert status == "scheduled"
    rec = studio.history.get(record_id)
    assert rec["youtube_id"] == "vid-sched-999"
    assert rec["status"] == "scheduled"
    # A scheduled upload is private until publishAt: never reported as publicly published.
    assert captured["body"]["status"]["privacyStatus"] == "private"
    assert "publishAt" in captured["body"]["status"]


# ------------------------------------------------------------------ run-summary reporting tests


def test_run_summary_reports_dry_run_no_upload(settings, tmp_path):
    """Dry-run slot: the summary explicitly says no YouTube upload was attempted."""
    out = SlotRunResult(
        ok=True, slot=3, run_date="2026-11-02", key="2026-11-02-slot-3",
        publish_at_ist="2026-11-02T19:00:00+05:30",
        publish_at_utc="2026-11-02T13:30:00Z",
        uploaded=False, dry_run=True,
    )
    automation = SlotAutomation(settings, slot=3, run_date="2026-11-02", dry_run=True)
    automation._write_summary(out)
    summary = (settings.data_path / "run-summary.md").read_text(encoding="utf-8")
    assert "Dry run: True" in summary
    assert "YouTube upload not attempted (dry run)" in summary


def test_run_summary_reports_scheduled_not_published(settings, tmp_path):
    """Scheduled slot: summary says 'scheduled as private, goes public X' — never 'published'."""
    studio = Studio(settings)
    out = SlotRunResult(
        ok=True, slot=3, run_date="2026-11-02", key="2026-11-02-slot-3",
        publish_at_ist="2026-11-02T19:00:00+05:30",
        publish_at_utc="2026-11-02T13:30:00Z",
        uploaded=True, youtube_id="vid-sched-999", dry_run=False,
    )
    automation = SlotAutomation(settings, slot=3, run_date="2026-11-02", dry_run=False)
    automation._write_summary(out)
    summary = (settings.data_path / "run-summary.md").read_text(encoding="utf-8")
    assert "scheduled as private" in summary
    assert "goes public 2026-11-02T19:00:00+05:30" in summary
    assert "https://youtube.com/shorts/vid-sched-999" in summary
    assert "published" not in summary.lower()
