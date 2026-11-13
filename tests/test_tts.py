import builtins
from pathlib import Path

import pytest

from purffle_shorts import tts
from purffle_shorts.config import Settings
from purffle_shorts.pipeline import Studio
from purffle_shorts.script import offline_script
from purffle_shorts.timing import Word
from purffle_shorts.utils import PermanentError


def test_edge_403_has_a_specific_diagnostic(tmp_path, monkeypatch):
    class UpstreamForbidden(Exception):
        status = 403

    def reject(*args, **kwargs):
        raise UpstreamForbidden("service rejected the request")

    monkeypatch.setattr(tts, "with_retries", reject)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        tts._edge("A short test.", tmp_path, Settings(tts_voice="en-IN-NeerjaExpressiveNeural"))


def test_edge_timeout_has_a_distinct_diagnostic(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("request timed out")

    monkeypatch.setattr(tts, "with_retries", timeout)
    with pytest.raises(RuntimeError, match="Edge TTS timed out"):
        tts._edge("A short test.", tmp_path, Settings())


def test_missing_edge_package_has_a_distinct_diagnostic(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def without_edge(name, *args, **kwargs):
        if name == "edge_tts":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_edge)
    with pytest.raises(PermanentError, match="edge-tts is not installed"):
        tts._edge("A short test.", tmp_path, Settings())


def test_pronunciations_expand_the_spoken_text_only():
    spoken, groups = tts.expand_pronunciations(
        "The GPU and VRAM matter.", {"GPU": "gee pee you", "VRAM": "vee ram"})
    assert spoken == "The gee pee you and vee ram matter."
    assert groups == [1, 3, 1, 2, 1]


def test_pronunciations_do_not_apply_to_languages_without_spaces():
    text = "GPU性能"
    spoken, groups = tts.expand_pronunciations(text, {"GPU": "gee pee you"}, "zh")
    assert spoken == text and groups == [1]


def test_pronounced_words_keep_the_original_caption_timing():
    tokens = ["The", "GPU", "is", "fast."]
    groups = [1, 3, 1, 1]
    spoken = [Word("The", 0.0, 0.1), Word("gee", 0.1, 0.2), Word("pee", 0.2, 0.3),
              Word("you", 0.3, 0.4), Word("is", 0.4, 0.5), Word("fast.", 0.5, 0.8)]
    timed = tts._group_word_timings(tokens, groups, spoken)
    assert [w.text for w in timed] == tokens
    assert (timed[1].start, timed[1].end) == (0.1, 0.4)


# ------------------------------------------------------------------ voice resilience

def _write_silence(path: Path, seconds: float = 0.5) -> None:
    """Write a real, ffmpeg-decodable silent audio file (no network, no keys)."""
    tts.ffmpeg.run(
        ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", f"{seconds:.2f}", str(path)],
        label="test-silence", )


def _valid_speech(audio: Path, engine: str = "edge", voice: str = "test") -> tts.Speech:
    return tts.Speech(audio=audio, words=[], duration=0.5, engine=engine, voice=voice,
                      timing="estimated")


def test_edge_tts_passes_audio_validation(tmp_path, monkeypatch):
    """A real engine producing decodable audio passes validation and is used."""
    audio = tmp_path / "voice.wav"
    _write_silence(audio)

    def fake_edge(text, out, config):
        _write_silence(out / "voice.mp3")
        return out / "voice.mp3", [Word("Hello", 0.0, 0.25), Word("world", 0.25, 0.5)], "en-US-AndrewMultilingualNeural"

    monkeypatch.setitem(tts._ENGINE_FUNCS, "edge", fake_edge)
    settings = Settings(tts_engine="edge")
    speech = tts.synthesize("Hello world.", tmp_path, settings)

    assert speech.audio.exists() and speech.audio.stat().st_size > 0
    assert speech.duration > 0
    assert speech.engine == "edge"


def test_undecodable_audio_is_rejected(tmp_path, monkeypatch):
    """Corrupt output from an engine is treated as an engine failure, never rendered."""

    def fake_edge(text, out, config):
        bad = out / "voice.mp3"
        bad.write_text("not audio at all", encoding="utf-8")
        return bad, [], "en-US-AndrewMultilingualNeural"

    monkeypatch.setitem(tts._ENGINE_FUNCS, "edge", fake_edge)
    settings = Settings(tts_engine="edge")
    with pytest.raises(RuntimeError, match="unreadable audio|produced no audio"):
        tts.synthesize("Hello world.", tmp_path, settings)


def test_edge_failure_falls_back_to_system_voice(tmp_path, monkeypatch):
    """Edge TTS failing transparently falls back to the system voice."""
    settings = Settings(data_dir=str(tmp_path / "data"), tts_engine="edge", offline=False)
    studio = Studio(settings)
    script = offline_script("a space fact", settings, "facts")
    work = tmp_path / "work"
    work.mkdir()
    audio = work / "voice.wav"
    _write_silence(audio)
    system_speech = _valid_speech(audio, engine="system", voice="system-default")
    calls = []

    def synthesize(text, out_dir, config):
        calls.append(config.tts_engine)
        if config.tts_engine == "edge":
            raise RuntimeError("Edge TTS was rejected with HTTP 403")
        return system_speech

    monkeypatch.setattr(tts, "synthesize", synthesize)
    result = studio._speak(script, work, settings)

    assert calls == ["edge", "system"], calls
    assert result.engine == "system"
    assert result.duration > 0


def test_all_voice_engines_fail_raises(tmp_path, monkeypatch):
    """When every engine fails, _speak raises before any rendering happens."""
    settings = Settings(data_dir=str(tmp_path / "data"), tts_engine="edge", offline=False)
    studio = Studio(settings)
    script = offline_script("a space fact", settings, "facts")

    def synthesize(text, out_dir, config):
        raise RuntimeError(f"{config.tts_engine} is down")

    monkeypatch.setattr(tts, "synthesize", synthesize)
    with pytest.raises(RuntimeError, match="All voice engines failed"):
        studio._speak(script, tmp_path / "work", settings)


def test_invalid_audio_is_not_passed_to_renderer(tmp_path, monkeypatch):
    """An engine that returns an empty audio file is skipped in favour of the next."""
    settings = Settings(data_dir=str(tmp_path / "data"), tts_engine="edge", offline=False)
    studio = Studio(settings)
    script = offline_script("a space fact", settings, "facts")
    work = tmp_path / "work"
    work.mkdir()
    good = work / "voice.wav"
    _write_silence(good)
    bad = work / "edge_bad.mp3"
    bad.write_bytes(b"")
    calls = []
    system_speech = _valid_speech(good, engine="system")

    def synthesize(text, out_dir, config):
        calls.append(config.tts_engine)
        if config.tts_engine == "edge":
            return tts.Speech(audio=bad, words=[], duration=0.0, engine="edge", voice="x",
                              timing="estimated")
        return system_speech

    monkeypatch.setattr(tts, "synthesize", synthesize)
    result = studio._speak(script, work, settings)
    assert calls == ["edge", "system"]
    assert result.engine == "system"
    assert not bad.exists()  # corrupt file cleaned up


def test_tts_retries_are_bounded(tmp_path):
    """Transient TTS failures are retried a bounded number of times, never endlessly."""
    attempts = 0

    def flaky():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("upstream hiccup")

    with pytest.raises(RuntimeError, match="upstream hiccup"):
        tts.with_retries(flaky, attempts=3, label="fake-tts")
    assert attempts == 3


def test_system_voice_empty_output_is_rejected(tmp_path, monkeypatch):
    """espeak-ng writing an empty file is treated as unavailable, not as silence."""
    settings = Settings(tts_engine="system")
    out = tmp_path / "voice"
    out.mkdir()
    raw = out / "voice.wav"
    raw.write_bytes(b"")

    def fake_run(cmd, **kwargs):
        raw.write_bytes(b"")  # espeak-ng produced nothing
        return None

    monkeypatch.setattr(tts.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="produced no audio"):
        tts._system("Hello world.", out, settings)


def test_synthesize_sends_expanded_text_but_keeps_caption_tokens(tmp_path, monkeypatch):
    calls = []

    def fake_engine(text, out, config):
        calls.append(text)
        return out / "voice.wav", [Word("The", 0.0, 0.1), Word("gee", 0.1, 0.2),
                                   Word("pee", 0.2, 0.3), Word("you", 0.3, 0.4),
                                   Word("is", 0.4, 0.5), Word("fast.", 0.5, 0.8)], "fake"

    monkeypatch.setitem(tts._ENGINE_FUNCS, "silent", fake_engine)
    monkeypatch.setattr(tts.ffmpeg, "duration", lambda path: 0.8)
    settings = Settings(tts_engine="silent", tts_pronunciations={"GPU": "gee pee you"})

    speech = tts.synthesize("The GPU is fast.", tmp_path, settings)

    assert calls == ["The gee pee you is fast."]
    assert [w.text for w in speech.words] == ["The", "GPU", "is", "fast."]
    assert (speech.words[1].start, speech.words[1].end) == (0.1, 0.4)
