from pathlib import Path

import pytest
from PIL import Image, ImageChops

from purffle_shorts import overlays
from purffle_shorts.overlays import CAPTION_STYLES, PillowCaptions, _ass_escape, resolve_font, write_ass
from purffle_shorts.timing import Chunk, Word


def chunk() -> Chunk:
    words = [
        Word("Deep", 0.00, 0.22),
        Word("ocean", 0.22, 0.48),
        Word("octopus", 0.48, 0.90),
    ]
    return Chunk(words=words, start=0.0, end=0.90)


def font(settings) -> Path | None:
    return Path(settings.caption_font) if settings.caption_font else None


def test_kinetic_uses_cream_text_and_orange_active_word(settings):
    style = CAPTION_STYLES["kinetic"]
    assert style.fill == "#FFF7E8"
    assert style.active == "#FF7A18"
    assert settings.caption_style == "kinetic"


def test_dust_exit_adds_fading_frames(settings, tmp_path):
    s = settings.with_overrides(resolution=(360, 640), fps=30, caption_exit="dust",
                                caption_exit_seconds=0.18)
    playlist = PillowCaptions(s, CAPTION_STYLES["kinetic"], font(s)).build(
        [chunk()], 1.2, tmp_path / "captions")

    text = playlist.read_text(encoding="utf-8")
    assert "_out" in text
    outs = sorted((tmp_path / "captions").glob("c0000_out*.png"))
    assert len(outs) >= 3
    assert Image.open(outs[-1]).getchannel("A").getextrema()[1] == 0
    durations = [float(line.split()[1]) for line in text.splitlines() if line.startswith("duration ")]
    assert sum(durations) == pytest.approx(1.2, abs=0.02)


def test_exit_none_has_no_transition_frames(settings, tmp_path):
    s = settings.with_overrides(resolution=(360, 640), caption_exit="none")
    playlist = PillowCaptions(s, CAPTION_STYLES["kinetic"], font(s)).build(
        [chunk()], 1.0, tmp_path / "captions")
    assert "_out" not in playlist.read_text(encoding="utf-8")


def test_ass_fades_the_last_word_state(settings, tmp_path):
    s = settings.with_overrides(caption_exit="dissolve", caption_exit_seconds=0.18)
    dest = write_ass(s, CAPTION_STYLES["kinetic"], "Arial", [chunk()], 1.0,
                     tmp_path / "captions.ass")
    lines = [ln for ln in dest.read_text(encoding="utf-8").splitlines() if ln.startswith("Dialogue: 1")]
    # The last state fades; the fade is stretched by the stagger so the outgoing caption has already
    # thinned out by the time the next caption reaches full opacity.
    assert r"\fad(0,180)" in lines[-1] or r"\fad(0,243)" in lines[-1]
    assert r"\fad(0,180)" not in lines[-2] and r"\fad(0,243)" not in lines[-2]


def test_dust_exit_frames_really_move_and_fade(settings, tmp_path):
    s = settings.with_overrides(resolution=(360, 640), fps=30, caption_exit="dust",
                                caption_exit_seconds=0.18)
    PillowCaptions(s, CAPTION_STYLES["kinetic"], font(s)).build([chunk()], 1.2, tmp_path / "captions")
    outs = sorted((tmp_path / "captions").glob("c0000_out*.png"))
    frames = [Image.open(p).convert("RGBA") for p in outs]
    assert len(frames) >= 3
    alphas = [f.getchannel("A") for f in frames]
    totals = [sum(value * count for value, count in enumerate(a.histogram())) for a in alphas]
    assert totals == sorted(totals, reverse=True) and totals[0] > 0
    assert any(ImageChops.difference(frames[i], frames[i + 1]).getbbox()
               for i in range(len(frames) - 1))


def test_dissolve_exit_frames_exist_and_fade(settings, tmp_path):
    s = settings.with_overrides(resolution=(360, 640), fps=30, caption_exit="dissolve",
                                caption_exit_seconds=0.18)
    PillowCaptions(s, CAPTION_STYLES["kinetic"], font(s)).build([chunk()], 1.2, tmp_path / "captions")
    outs = sorted((tmp_path / "captions").glob("c0000_out*.png"))
    assert len(outs) >= 2
    last = Image.open(outs[-1]).getchannel("A")
    assert last.getextrema()[1] == 0


def test_ass_escaping_keeps_text_from_breaking_dialogue():
    assert _ass_escape("A\\B{C}\nD") == "A/B(C) D"


def test_font_falls_back_when_no_download_or_system_font(settings, monkeypatch):
    """With no download, no system font and no fallback chain left, Pillow's built-in face is used."""
    monkeypatch.setattr(overlays, "ensure_font", lambda *a, **k: None)
    monkeypatch.setattr(overlays, "SYSTEM_BOLD", [])
    monkeypatch.setattr(overlays, "SYSTEM_FALLBACK", [])
    path, family = resolve_font(settings.with_overrides(caption_font=""))
    assert path is None and family == "Arial"


def test_font_prefers_the_system_fallback_chain_before_giving_up(settings, monkeypatch):
    """The chain is what makes the look survive a machine with no Inter, no Google Fonts, no net."""
    monkeypatch.setattr(overlays, "ensure_font", lambda *a, **k: None)
    monkeypatch.setattr(overlays, "SYSTEM_FALLBACK", [(__file__, __file__)])
    path, family = resolve_font(settings.with_overrides(caption_font=""))
    assert path is not None and family == Path(__file__).stem


def test_requested_family_that_cannot_download_still_gets_a_system_face(settings, monkeypatch):
    monkeypatch.setattr(overlays, "ensure_font", lambda *a, **k: None)
    monkeypatch.setattr(overlays, "SYSTEM_FALLBACK", [(__file__, __file__)])
    path, _ = resolve_font(settings.with_overrides(caption_font="", caption_font_family="inter"))
    assert path is not None
