from purffle_shorts.config import Settings, parse_resolution
from purffle_shorts.history import History
from purffle_shorts.utils import redact


def test_legacy_env_names_still_work(monkeypatch):
    monkeypatch.setenv("SHORTS_RESOLUTION", "721x1281")
    monkeypatch.setenv("SHORTS_FPS", "60")
    monkeypatch.setenv("SHORTS_BATCH_SIZE", "5")
    monkeypatch.setenv("NICHES", "space, cats ,")
    monkeypatch.setenv("END_CTA", "")
    s = Settings.from_env()
    assert s.resolution == (720, 1280) and s.fps == 60 and s.batch_size == 5
    assert s.niches == ["space", "cats"]
    assert s.end_cta == ""


def test_parse_resolution_fallback():
    assert parse_resolution("garbage") == (1080, 1920)


def test_hybrid_visual_env_settings(monkeypatch):
    monkeypatch.setenv("VISUAL_MODE", "Hybrid")
    monkeypatch.setenv("HYBRID_IMAGE_EVERY", "3")
    s = Settings.from_env()
    assert s.visual_mode == "hybrid"
    assert s.hybrid_image_every == 3
    monkeypatch.setenv("HYBRID_IMAGE_EVERY", "1")
    assert Settings.from_env().hybrid_image_every == 2


def test_target_seconds_is_clamped_to_the_supported_range(monkeypatch):
    assert Settings(target_seconds=10).target_seconds == 15
    assert Settings(target_seconds=500).target_seconds == 170
    monkeypatch.setenv("TARGET_SECONDS", "10")
    assert Settings.from_env().target_seconds == 15


def test_watermark_defaults_to_channel():
    assert Settings(channel_name="Chan").watermark_text == "Chan"
    assert Settings(watermark="none").watermark_text == ""


def test_history_roundtrip(tmp_path):
    h = History(tmp_path / "h.db")
    vid = h.add_video(title="T", status="rendered", video_path="/x.mp4", tags=["a"])
    h.update_video(vid, status="queued")
    assert [r["id"] for r in h.pending_uploads()] == [vid]
    assert h.pending_uploads(("rendered",)) == []
    h.update_video(vid, status="uploaded", uploaded_at="2099-01-01T00:00:00+00:00")
    assert h.uploads_since("2098-12-31T00:00:00+00:00") == 1
    h.mark_media(["pexels:1", "pexels:1"])
    assert h.used_media() == {"pexels:1"}
    h.mark_topic("Trend:X")
    assert h.topic_used("trend:x")
    assert h.recent_titles() == ["T"]


def test_approval_persists_across_history_reopen(tmp_path):
    path = tmp_path / "h.db"
    h = History(path)
    vid = h.add_video(title="Review me", status="rendered")
    assert h.approve_video(vid, "2026-10-21T12:00:00+00:00", {"platform": "youtube"})
    reopened = History(path).get(vid)
    assert reopened["approval_status"] == "approved"
    assert reopened["approved_at"] == "2026-10-21T12:00:00+00:00"
    assert reopened["approval_json"] == '{"platform": "youtube"}'
    assert not h.approve_video(vid, "later", {})


def test_redact_hides_keys_in_urls():
    msg = "Max retries exceeded with url: /api/videos/?key=abc123SECRET&q=octopus (api_key=zzz)"
    out = redact(msg)
    assert "abc123SECRET" not in out and "zzz" not in out and "q=octopus" in out


def test_redact_hides_ip_addresses_but_not_times_or_versions():
    msg = ('HTTP 429 — {"message":"Queue full for IP: 2409:4091:a01f:c202:f83f:1657:e36b:9830"} '
           "from 203.0.113.7 at 17:49:06 with ffmpeg 9.0.2")
    out = redact(msg)
    assert "2409:4091" not in out and "203.0.113.7" not in out and out.count("<ip>") == 2
    assert "17:49:06" in out and "9.0.2" in out


def test_env_file_is_read_from_the_working_folder(tmp_path, monkeypatch):
    # Installed with pip or uvx, the package lives far from your folder; the .env next to you must still load.
    import os

    from purffle_shorts.config import load_env
    (tmp_path / ".env").write_text("PURFFLE_TEST_FROM_CWD=yes\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PURFFLE_TEST_FROM_CWD", raising=False)
    load_env()
    assert os.environ["PURFFLE_TEST_FROM_CWD"] == "yes"


def test_agnes_key_is_loaded_from_cwd_env_without_exposing_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGNES_API_KEY", raising=False)
    (tmp_path / ".env").write_text("AGNES_API_KEY=test-only-placeholder\n", encoding="utf-8")
    assert Settings.from_env().agnes_api_key == "test-only-placeholder"


def test_nonempty_process_agnes_key_overrides_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("AGNES_API_KEY=dotenv-placeholder\n", encoding="utf-8")
    monkeypatch.setenv("AGNES_API_KEY", "process-placeholder")
    assert Settings.from_env().agnes_api_key == "process-placeholder"


def test_env_example_works_as_a_docker_env_file(monkeypatch):
    # docker run --env-file passes everything after "=" verbatim, comments included.
    from pathlib import Path
    for line in (Path(__file__).parent.parent / ".env.example").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            key, _, value = line.partition("=")
            assert "#" not in value, line
            monkeypatch.setenv(key, value)
    s = Settings.from_env()
    assert s.resolution == (1080, 1920) and s.synthetic_media and s.daily_upload_limit == 6
