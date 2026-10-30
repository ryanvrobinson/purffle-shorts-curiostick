"""Synchronization invariants: words -> captions -> scenes -> transitions -> rendered duration.

These tests pin down the contract that keeps narration, captions and scene changes on one timeline,
so a regression in any stage (chunking, scene splitting or transition padding) is caught immediately.
They are offline except the two renderer checks, which need ffmpeg.
"""

import json

import pytest
from PIL import Image

from purffle_shorts import ffmpeg
from purffle_shorts.media import MediaItem
from purffle_shorts.pipeline import Studio
from purffle_shorts.render import prepare_segment, segment_lengths
from purffle_shorts.timing import Word, caption_chunks, estimate_timings, scene_timeline

from .conftest import needs_ffmpeg

PARAGRAPH = ("An octopus has three hearts and its blood is blue. "
             "The study of the reef in the Pacific surprised the team of divers. "
             "It was one of the best days of my life, and the water was warm again.")


# ------------------------------------------------------------------ captions on the narration timeline
def test_captions_start_and_end_on_word_boundaries():
    words = estimate_timings(PARAGRAPH.split(), 14.0)
    chunks = caption_chunks(words, max_words=3, total=14.0)
    starts = {round(w.start, 3) for w in words}
    ends = {round(w.end, 3) for w in words}
    for c in chunks:
        assert round(c.words[0].start, 3) == round(c.start, 3)
        # A caption may be extended to the next caption's start, but it starts on its first word.
        assert round(c.start, 3) in starts
        assert c.end >= c.words[-1].end - 1e-6
        assert round(c.words[-1].end, 3) in ends


def test_captions_are_monotonic_and_inside_the_total():
    words = estimate_timings(PARAGRAPH.split(), 14.0)
    chunks = caption_chunks(words, max_words=3, total=14.0)
    assert chunks
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start + 1e-6
    assert chunks[0].start >= 0.0
    assert chunks[-1].end <= 14.0


def test_caption_words_are_the_script_words_in_order():
    words = estimate_timings(PARAGRAPH.split(), 14.0)
    chunks = caption_chunks(words, max_words=3, total=14.0)
    joined = [w.text for c in chunks for w in c.words]
    assert joined == PARAGRAPH.split()  # no word dropped, duplicated or reordered


# ------------------------------------------------------------------ scene boundaries
def test_scene_timeline_tiles_the_whole_video():
    scenes = [s.strip() for s in PARAGRAPH.split(".") if s.strip()]
    words = estimate_timings(PARAGRAPH.split(), 14.0)
    timeline = scene_timeline(scenes, words, total=14.0)
    assert timeline[0].start == 0.0
    assert timeline[-1].end == 14.0
    for a, b in zip(timeline, timeline[1:]):
        assert a.end == b.start  # no gap and no overlap between shots


def test_scene_starts_follow_the_first_spoken_word():
    scenes = ["One two three.", "Four five.", "Six seven eight nine."]
    words = [Word(t, i * 0.7, i * 0.7 + 0.5) for i, t in enumerate(" ".join(scenes).split())]
    timeline = scene_timeline(scenes, words, total=8.0)
    # Scene k must begin when its first word begins (here word 0, word 3, word 5).
    assert [round(s.start, 2) for s in timeline] == [0.0, round(words[3].start, 2), round(words[5].start, 2)]


# ------------------------------------------------------------------ transitions preserve the timeline
def test_transition_padding_keeps_every_cut_on_its_boundary():
    durations = [3.0, 4.5, 2.0, 5.5]
    transition = 0.35
    lengths = segment_lengths(durations, transition)
    n = len(durations)
    # After cross-fades the video is exactly as long as the narration it carries.
    assert abs(sum(lengths) - (n - 1) * transition - sum(durations)) < 1e-9


def test_no_transition_leaves_durations_untouched():
    durations = [3.0, 4.5, 2.0]
    assert segment_lengths(durations, 0.0) == durations


# ------------------------------------------------------------------ renderer fallbacks
@needs_ffmpeg
def test_prepare_segment_falls_back_when_media_is_missing(settings, tmp_path):
    settings = settings.with_overrides(resolution=(180, 320), fps=12)
    item = MediaItem(source="local", id="gone", kind="video", path=tmp_path / "does-not-exist.mp4")
    out = prepare_segment(item, 1.5, 0, settings, tmp_path)
    assert out.exists()
    info = ffmpeg.probe(out)
    assert (info["width"], info["height"]) == (180, 320) and info["duration"] > 1.0


@needs_ffmpeg
def test_prepare_segment_falls_back_when_media_is_empty(settings, tmp_path):
    settings = settings.with_overrides(resolution=(180, 320), fps=12)
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    item = MediaItem(source="local", id="empty", kind="video", path=empty)
    out = prepare_segment(item, 1.5, 1, settings, tmp_path)
    assert out.exists() and ffmpeg.probe(out)["duration"] > 1.0


@needs_ffmpeg
def test_prepare_segment_renders_a_portrait_still(settings, tmp_path):
    settings = settings.with_overrides(resolution=(360, 640), fps=12, ken_burns=True)
    still = tmp_path / "agnes.png"
    Image.new("RGB", (720, 1280), (200, 30, 30)).save(still)
    item = MediaItem(source="agnes", id="agnes-1", kind="image", path=still, width=720, height=1280)
    out = prepare_segment(item, 2.0, 2, settings, tmp_path)
    info = ffmpeg.probe(out)
    assert (info["width"], info["height"]) == (360, 640)
    assert info["duration"] >= 1.9


# ------------------------------------------------------------------ rendered output reports its timeline
@needs_ffmpeg
def test_offline_render_reports_a_consistent_duration(settings):
    s = settings.with_overrides(offline=True, tts_engine="silent", resolution=(180, 320), fps=12,
                                transition="fade", music_volume=0.0)
    r = Studio(s).make(topic="the history of tea")
    assert r.ok, r.error
    info = ffmpeg.probe(r.video)
    assert info["has_audio"]
    meta = json.loads((r.folder / "metadata.json").read_text())
    assert meta["actual_duration"] == pytest.approx(info["duration"], abs=0.01)
    assert meta["requested_duration"] == s.target_seconds
    assert abs(meta["duration_deviation"] - (info["duration"] - s.target_seconds)) < 0.02
    # The assembled scene timeline and the real file agree (the renderer did not drop or pad time).
    assert abs(meta["assembled_duration"] - info["duration"]) < 0.6
