import pytest

from purffle_shorts.cli import build_parser, build_settings, main
from purffle_shorts.config import ConfigError


def test_providers_command(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "anthropic" in out and "ollama" in out


def test_overrides_from_flags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["make", "--lang", "es", "--visuals", "pexels,Pollinations", "--no-upload",
                                      "--resolution", "720x1280", "--duration", "500", "--no-music"])
    s = build_settings(args)
    assert s.language == "es" and s.visual_sources == ["pexels", "pollinations"]
    assert s.upload is False and s.resolution == (720, 1280)
    assert s.target_seconds == 170 and s.music_volume == 0.0


def test_duration_flags_use_the_shared_limits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    low = build_settings(build_parser().parse_args(["make", "--duration", "10"]))
    assert low.target_seconds == 15
    high = build_settings(build_parser().parse_args(["make", "--duration", "999"]))
    assert high.target_seconds == 170


def test_agnes_visual_selection_is_preserved_in_cli_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["make", "--visuals", "agnes,pixabay"])
    s = build_settings(args)
    assert s.visual_sources == ["agnes", "pixabay"]


def test_agnes_env_and_cli_visual_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGNES_API_KEY", raising=False)
    (tmp_path / ".env").write_text("AGNES_API_KEY=fake-test-key\n", encoding="utf-8")
    args = build_parser().parse_args(["make", "--visuals", "agnes,pixabay"])

    settings = build_settings(args)

    assert settings.agnes_api_key == "fake-test-key"
    assert settings.visual_sources == ["agnes", "pixabay"]


def test_quality_threshold_cli_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["make", "--quality-threshold", "101"])
    with pytest.raises(ConfigError, match="quality-threshold"):
        build_settings(args)


def test_hybrid_visual_flags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["make", "--visuals", "pixabay,agnes", "--visual-mode", "hybrid",
                                      "--hybrid-image-every", "3"])
    s = build_settings(args)
    assert s.visual_mode == "hybrid"
    assert s.visual_sources == ["pixabay", "agnes"]
    assert s.hybrid_image_every == 3


def test_hybrid_interval_requires_two(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["make", "--hybrid-image-every", "1"])
    with pytest.raises(ConfigError, match="at least 2"):
        build_settings(args)
