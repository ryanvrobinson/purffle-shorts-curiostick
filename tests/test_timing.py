from purffle_shorts.timing import (
    Word,
    _dangling,
    caption_chunks,
    estimate_timings,
    scene_timeline,
    srt,
    tokenize,
    transfer_timings,
)


def test_estimate_timings_monotonic_and_bounded():
    words = estimate_timings(["Did", "you", "know?", "Octopuses", "have", "three", "hearts."], 3.0)
    assert len(words) == 7
    assert words[0].start >= 0 and words[-1].end <= 3.0
    assert all(a.start <= b.start for a, b in zip(words, words[1:]))
    assert all(w.end > w.start for w in words)


def test_transfer_keeps_script_words_and_fills_gaps():
    tokens = ["Hello,", "world!", "It", "has", "1,000", "cats."]
    timed = [Word("Hello", 0.0, 0.3), Word("world", 0.35, 0.7), Word("It", 0.9, 1.0), Word("has", 1.0, 1.2),
             Word("one", 1.25, 1.4), Word("thousand", 1.4, 1.8), Word("cats", 1.85, 2.2)]
    out = transfer_timings(tokens, timed, 2.5)
    assert [w.text for w in out] == tokens
    assert out[0].start == 0.0 and out[1].start == 0.35
    assert out[3].end <= out[4].start <= out[5].start == 1.85
    assert all(a.start <= b.start for a, b in zip(out, out[1:]))


def test_transfer_without_aligner_estimates():
    out = transfer_timings(["a", "b", "c"], [], 1.5)
    assert len(out) == 3 and out[-1].end <= 1.5


def test_scene_timeline_follows_voice():
    scenes = ["One two three.", "Four five.", "Six seven eight nine."]
    words = [Word(t, i * 0.7, i * 0.7 + 0.5) for i, t in enumerate(" ".join(scenes).split())]
    tl = scene_timeline(scenes, words, total=8.0)
    assert [round(s.start, 2) for s in tl] == [0.0, 2.1, 3.5]
    assert tl[-1].end == 8.0


def test_scene_timeline_merges_tiny_scenes():
    scenes = ["One two three four five.", "Six.", "Seven eight nine ten eleven twelve."]
    words = [Word(t, i * 0.4, i * 0.4 + 0.3) for i, t in enumerate(" ".join(scenes).split())]
    tl = scene_timeline(scenes, words, total=6.0)
    assert len(tl) == 2 and all(s.duration >= 1.2 for s in tl)


def test_caption_chunks_break_on_punctuation_and_limits():
    words = estimate_timings("An octopus has three hearts. Its blood is blue.".split(), 4.0)
    chunks = caption_chunks(words, max_words=3)
    # Sentences stay whole: a sentence-closing word may finish a phrase one word over the soft limit.
    assert [c.text for c in chunks] == ["An octopus has three hearts.", "Its blood is blue."]
    assert all(len(c.words) <= 5 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start + 1e-6


def test_caption_chunks_never_end_on_a_dangling_word():
    text = ("The study of the reef in the Pacific surprised the team of divers today. "
            "It was one of the best days of my life.")
    words = estimate_timings(text.split(), 8.0)
    chunks = caption_chunks(words, max_words=3)
    for c in chunks:
        assert not _dangling(c.words[-1].text), c.text
    assert all(len(c.words) <= 5 for c in chunks)  # never more than max_words + 2


def test_caption_chunks_never_emit_a_subframe_caption():
    words = estimate_timings("Go. Now. Wait. Here. Stop. Yes. No. Go again please.".split(), 6.0)
    chunks = caption_chunks(words, max_words=3, total=6.0)
    for c in chunks:
        assert c.end > c.start and c.end - c.start >= 0.05, c.text
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start + 1e-6


def test_tokenize_no_space_languages():
    toks = tokenize("章鱼有三颗心脏。它的血液是蓝色的！", "zh")
    assert len(toks) >= 4 and all(len(t) <= 5 for t in toks)
    assert tokenize("hello  world", "en") == ["hello", "world"]


def test_srt_format():
    text = srt(caption_chunks([Word("Hi", 0.0, 0.5), Word("there.", 0.5, 1.0)], max_words=3))
    # New timing behavior: chunks extend to next chunk start or +0.4s for last chunk
    assert "00:00:00,000 -->" in text
    assert "Hi" in text
    assert "there." in text


def test_srt_handles_non_breaking_hyphen():
    """Regression test for Windows Unicode encoding error with U+2011 (non-breaking hyphen)."""
    # U+2011 is a non-breaking hyphen that caused 'charmap' codec can't encode character '\\u2011'
    words = [Word("test\u2011hyphen", 0.0, 1.0)]
    chunks = caption_chunks(words, max_words=3)
    text = srt(chunks)
    assert "\u2011" in text  # non-breaking hyphen preserved
    assert "test\u2011hyphen" in text
