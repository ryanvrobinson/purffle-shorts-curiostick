import builtins
import logging
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


def test_edge_failure_falls_back_to_system_voice(tmp_path, monkeypatch, caplog):
    settings = Settings(data_dir=str(tmp_path / "data"), tts_engine="edge", offline=False)
    studio = Studio(settings)
    script = offline_script("a space fact", settings, "facts")
    calls = []
    speech = tts.Speech(Path("voice.wav"), [], 2.0, "system", "system-default", "estimated")

    def synthesize(text, out_dir, config):
        calls.append(config.tts_engine)
        if config.tts_engine == "edge":
            raise RuntimeError("Edge TTS was rejected with HTTP 403")
        return speech

    monkeypatch.setattr(tts, "synthesize", synthesize)
    with caplog.at_level(logging.WARNING):
        result = studio._speak(script, tmp_path / "work", settings)

    assert calls == ["edge", "system"]
    assert result.engine == "system"
    assert "HTTP 403" in caplog.text


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
