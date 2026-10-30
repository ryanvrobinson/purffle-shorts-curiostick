from __future__ import annotations

import json
from types import SimpleNamespace

from purffle_shorts import cli, config
from purffle_shorts.history import History
from purffle_shorts.pipeline import Result, Studio


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
    video.write_bytes(b"draft")
    studio = Studio(settings)
    record_id = studio.history.add_video(
        title="Draft", folder=str(tmp_path), video_path=str(video), status="rendered")
    monkeypatch.setattr("purffle_shorts.youtube.upload",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unapproved upload")))

    assert studio.upload_record(record_id) == "approval-required"
    assert studio.history.get(record_id)["approval_status"] == "pending"


def test_upload_record_refuses_known_uploaded_video(settings, tmp_path, monkeypatch):
    video = tmp_path / "short.mp4"
    video.write_bytes(b"already uploaded")
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
