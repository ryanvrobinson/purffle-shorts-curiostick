"""Two-speaker scripts, the Script Doctor and translation — with fake LLMs, no network."""

import pytest

from purffle_shorts.config import Settings
from purffle_shorts.script import (
    Script,
    ScriptQualityError,
    build_prompt,
    normalize,
    offline_script,
    review_prompt,
    review_script,
    scene_plan,
    script_from_data,
    translate_script,
    word_count,
    write_script,
)


class FakeLLM:
    label = "fake:model"

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def complete_json(self, system, user, schema=None, max_tokens=4000):
        self.calls.append({"system": system, "user": user, "schema": schema})
        if self.error:
            raise self.error
        return self.reply(user) if callable(self.reply) else self.reply


def _scenes(n, words=8, speakers="A"):
    return [{"speaker": speakers[i % len(speakers)],
             "narration": f"s{i} " + " ".join(["word"] * (words - 1)) + ".",
             "search_query": f"q{i}", "image_prompt": f"p{i}"} for i in range(n)]


def test_dialogue_keeps_speakers_and_fills_cast():
    s = normalize({"title": "T", "cast": ["Mia"], "scenes": _scenes(4, speakers="AB")}, "x", "dialogue", "en")
    assert [sc.speaker for sc in s.scenes] == ["A", "B", "A", "B"]
    assert s.cast == ["Mia", "Sam"] and s.multi_speaker
    assert s.speaker_name("B") == "Sam"


def test_single_narrator_formats_ignore_speakers():
    s = normalize({"title": "T", "cast": ["X", "Y"], "scenes": _scenes(4, speakers="AB")}, "x", "facts", "en")
    assert {sc.speaker for sc in s.scenes} == {"A"} and s.cast == [] and not s.multi_speaker


def test_chat_allows_more_scenes_and_merges_only_same_speaker():
    s = normalize({"scenes": _scenes(26, words=2, speakers="AB")}, "x", "chat", "en")
    assert len(s.scenes) == 26  # alternating speakers can't be merged
    s = normalize({"scenes": _scenes(26, words=2, speakers="A")}, "x", "chat", "en")
    assert len(s.scenes) == 20


def test_from_dict_ignores_unknown_and_missing_fields():
    d = offline_script("", Settings()).to_dict()
    d["future_field"] = 1
    d["scenes"][0]["extra"] = "x"
    del d["cast"]
    s = Script.from_dict(d)
    assert s.title and s.cast == [] and s.scenes[0].speaker == "A"


def test_script_from_data_detects_a_conversation():
    s = script_from_data({"title": "Chat", "scenes": _scenes(4, speakers="AB")}, Settings())
    assert s.style == "dialogue" and s.multi_speaker


def test_scene_plan_by_format():
    assert scene_plan("facts", 40) == (104, 10)
    assert scene_plan("chat", 40)[1] == 14
    assert scene_plan("dialogue", 40)[1] == 11


def test_prompt_carries_format_rules_and_channel_insights():
    _, user = build_prompt("cats", "chat", Settings(), [], insights="What performed best: X")
    assert "text message" in user and '"Me"' in user and "What performed best: X" in user
    _, user = build_prompt("cats", "facts", Settings(), [])
    assert 'speaker is always "A"' in user
    assert "name the subject or its consequence immediately" in user
    _, rewrite = review_prompt(_draft(), Settings(), True)
    assert "scientifically" in rewrite and "defensible question or consequence" in rewrite


def _draft(words=10, n=4):
    return normalize({"title": "Draft", "hook_text": "HOOK", "scenes": _scenes(n, words)}, "cats", "facts", "en")


def _full_script(scenes, title="X"):
    return {
        "title": title,
        "hook_text": "HOOK",
        "cast": [],
        "scenes": scenes,
        "description": "d",
        "hashtags": ["#cats"],
        "tags": ["cats"],
        "category": "pets",
    }


def test_doctor_rewrite_is_used_when_sane():
    settings = Settings(target_seconds=16)  # 41-word target
    better = {"title": "Better", "hook_text": "WOW", "cast": [], "scenes": _scenes(5, 9), "description": "d",
              "hashtags": ["#cats"], "tags": ["cats"], "category": "pets"}
    llm = FakeLLM({"score": 91, "issues": ["Weak hook", "Scene 3 is vague"], "script": better})
    out = review_script(llm, _draft(), settings)
    assert out.title == "Better" and out.score == 91 and out.review[0] == "Weak hook"
    assert out.category == "pets" and out.style == "facts"
    assert llm.calls[0]["schema"]["required"] == ["score", "issues", "script"]


def test_doctor_rewrite_with_wrong_length_is_ignored():
    llm = FakeLLM({"score": 40, "issues": ["too long"], "script": {"title": "Bloated", "scenes": _scenes(30, 20)}})
    with pytest.raises(ScriptQualityError, match="rendering was stopped"):
        review_script(llm, _draft(), Settings(target_seconds=16))
    assert len(llm.calls) == 2


def test_doctor_stops_after_bounded_attempts_if_score_stays_low():
    replies = iter([
        {"score": 45, "issues": ["The hook is vague"], "script": None},
        {"score": 68, "issues": ["The opening still lacks a concrete consequence"], "script": None},
        {"score": 55, "issues": ["Still vague"], "script": None},
    ])
    llm = FakeLLM(lambda user: next(replies))
    with pytest.raises(ScriptQualityError, match="68.*70.*2 bounded rewrite"):
        review_script(llm, _draft(), Settings(script_quality_threshold=70))
    assert len(llm.calls) == 2


def test_final_low_score_stops_make_before_rendering(tmp_path, monkeypatch):
    from purffle_shorts import pipeline
    from purffle_shorts.script import SCRIPT_SCHEMA

    data = {
        "topic": "neutron stars",
        "title": "A neutron star question",
        "hook_text": "WHAT HAPPENS?",
        "scenes": [
            {"narration": f"Neutron stars are collapsed stars. Fact {i}.",
             "search_query": "neutron star", "image_prompt": "neutron star in space"}
            for i in range(5)
        ],
        "description": "A science explanation.",
        "hashtags": [],
        "tags": [],
        "category": "science",
    }

    class FakeLLM:
        label = "fake:model"

        def __init__(self):
            self.reviews = 0

        def complete_json(self, system, user, schema=None, max_tokens=4000):
            if schema is SCRIPT_SCHEMA:
                return data
            self.reviews += 1
            return {"score": 45, "issues": ["The hook is vague"], "script": data}

    settings = Settings(target_seconds=16, script_review="rewrite", output_dir=str(tmp_path / "out"),
                        data_dir=str(tmp_path / "data"))
    studio = pipeline.Studio(settings)
    llm = FakeLLM()
    studio._llm = llm
    monkeypatch.setattr(studio, "produce", lambda *args, **kwargs: pytest.fail("low-score script was rendered"))

    result = studio.make(topic="neutron stars", upload=False)

    assert not result.ok and "below the 90-point threshold" in result.error
    assert llm.reviews == 2


def test_low_score_gets_one_bounded_second_review():
    better = {"title": "Specific", "hook_text": "A precise hook", "scenes": _scenes(5, 9)}

    def replies(user):
        if len(llm.calls) == 1:
            return {"score": 45, "issues": ["The hook is vague"], "script": better}
        assert "same subject" in user
        return {"score": 82, "issues": [], "script": better}

    llm = FakeLLM(replies)
    out = review_script(llm, _draft(), Settings(target_seconds=16, script_quality_threshold=70))

    assert out.title == "Specific" and out.score == 82
    assert len(llm.calls) == 2


def test_undersized_rewrite_is_ignored_then_bounded_retry_succeeds():
    """An undersized rewrite is ignored; the bounded retry with an in-range, higher-score rewrite wins."""
    target = scene_plan("facts", 16)[0]  # 41-word target
    lo, hi = int(0.75 * target), int(1.3 * target)  # 30 .. 53

    def replies(user):
        if len(llm.calls) == 1:
            return {
                "score": 45,
                "issues": ["too short"],
                "script": _full_script(_scenes(4, 5)),  # 20 words, below the 30-word floor
            }
        assert f"between {lo} and {hi} words" in user
        return {"score": 82, "issues": [], "script": _full_script(_scenes(5, 9))}  # 45 words, in range

    llm = FakeLLM(replies)
    out = review_script(llm, _draft(), Settings(target_seconds=16, script_quality_threshold=70))

    assert out.score == 82 and out.review == []
    assert len(llm.calls) == 2  # one bounded retry happened


def test_in_bounds_rewrite_without_score_gain_is_not_adopted():
    """A rewrite inside the word bounds but with a lower score is rejected on the bounded retry."""
    def replies(user):
        if len(llm.calls) == 1:
            return {"score": 60, "issues": ["ok"],
                    "script": _full_script(_scenes(5, 9))}  # 45 words, score 60
        return {"score": 55, "issues": ["worse"],
                "script": _full_script(_scenes(5, 9), title="Second")}  # lower score, in bounds

    llm = FakeLLM(replies)
    with pytest.raises(ScriptQualityError, match="rendering was stopped"):
        review_script(llm, _draft(), Settings(target_seconds=16, script_quality_threshold=70))
    assert len(llm.calls) == 2  # bounded retries happened, lower-score rewrite not adopted


def test_doctor_failure_never_fails_the_video():
    draft = _draft()
    assert review_script(FakeLLM(error=RuntimeError("503")), draft, Settings()) is draft
    assert draft.score is None


def test_doctor_score_only_and_off():
    llm = FakeLLM({"score": "77", "issues": []})
    out = review_script(llm, _draft(), Settings(script_review="score"))
    assert out.title == "Draft" and out.score == 77 and "script" not in llm.calls[0]["schema"]["properties"]
    off = FakeLLM({"score": 1})
    review_script(off, _draft(), Settings(script_review="off"))
    assert off.calls == []


def test_translation_keeps_footage_and_speakers():
    src = normalize({"title": "Hi", "cast": ["A1", "B1"], "scenes": _scenes(4, speakers="AB")}, "x", "dialogue", "en")

    def reply(user):
        assert "Spanish" in user and "EXACTLY 4 scenes" in user
        return {"title": "Hola", "hook_text": "HOLA", "cast": ["Ana", "Bea"], "category": "education",
                "scenes": [{"speaker": "A", "narration": f"frase {i}.", "search_query": "cambiado",
                            "image_prompt": "cambiado"} for i in range(4)],
                "description": "d", "hashtags": ["#hola"], "tags": ["hola"]}
    out = translate_script(FakeLLM(reply), src, "es")
    assert out.language == "es" and out.title == "Hola" and out.cast == ["Ana", "Bea"]
    assert [s.search_query for s in out.scenes] == ["q0", "q1", "q2", "q3"]
    assert [s.speaker for s in out.scenes] == ["A", "B", "A", "B"]


def test_translation_with_wrong_scene_count_is_rejected():
    src = _draft(n=4)
    with pytest.raises(ValueError, match="scenes"):
        translate_script(FakeLLM({"title": "x", "scenes": _scenes(3)}), src, "fr")


@pytest.mark.parametrize("style", ["dialogue", "chat"])
def test_offline_conversations_are_complete(style):
    s = offline_script("", Settings(), style)
    assert s.style == style and s.multi_speaker and len(s.cast) == 2 and len(s.scenes) >= 8


def test_reddit_story_is_one_narrator_with_handles():
    data = {"title": "t", "cast": ["pettyrevenge", "quiet otter"], "scenes": _scenes(6, speakers="AB")}
    s = normalize(data, "x", "reddit", "en")
    assert s.cast == ["r/pettyrevenge", "u/quiet_otter"]
    assert {sc.speaker for sc in s.scenes} == {"A"} and not s.multi_speaker
    assert normalize({"title": "t", "scenes": _scenes(5)}, "x", "reddit", "en").cast == ["r/stories", "u/throwaway"]
    system, user = build_prompt("a lost dog", "reddit", Settings(), [])
    assert "post title" in user and "u/" in user


def test_reddit_translation_keeps_the_handles():
    src = normalize({"title": "t", "cast": ["r/confessions", "u/otter"], "scenes": _scenes(4)}, "x", "reddit", "en")
    reply = {"title": "Hola", "cast": ["r/confesiones", "u/nutria"], "scenes": _scenes(4)}
    assert translate_script(FakeLLM(reply), src, "es").cast == ["r/confessions", "u/otter"]


def test_offline_reddit_story():
    s = offline_script("", Settings(), "reddit")
    assert s.style == "reddit" and s.cast == ["r/confessions", "u/quiet_otter22"]
    assert s.scenes[0].narration.startswith("Am I wrong") and len(s.scenes) == 8


# ------------------------------------------------------------------ regression: the Slot 3 failure
# Reported: original 109 words, rewrites 66 and 78 words against a 117-word target,
# final score 58 below the 90-point target after up to 3 bounded rewrite attempts.
# With TARGET_SECONDS=45 the target is 117 words and the permitted window is 87..152.


def _curio_stick_draft() -> Script:
    """~110-word facts draft for a 45-second target (117-word target, 87..152 window)."""
    return normalize(_full_script(_scenes(5, 22)), "cats", "facts", "en")


def test_length_rule_is_a_permitted_range_not_an_exact_count():
    """The length target is a centre with a +/-30% permitted window:
    ~109 words is acceptable for a 117-word target; 66 and 78 are below the 87-word floor."""
    settings = Settings(target_seconds=45)  # 45s * 2.6 words/s = 117-word target
    target = scene_plan("facts", 45)[0]
    assert target == 117
    lo, hi = int(0.75 * target), int(1.3 * target)  # 87 .. 152
    assert lo == 87 and hi == 152
    original = _curio_stick_draft()
    assert lo <= word_count(original.narration) <= hi  # ~110 words: accepted for review
    assert word_count(normalize(_full_script(_scenes(3, 22)), "x", "facts", "en").narration) == 66 < lo
    assert word_count(normalize(_full_script(_scenes(3, 26)), "x", "facts", "en").narration) == 78 < lo


def test_undersized_rewrites_rejected_then_targeted_expansion_recovers():
    """66- and 78-word rewrites are rejected; a targeted expansion after feedback
    lands in range with a higher score and is accepted. The original draft is not destroyed."""
    settings = Settings(target_seconds=45, script_quality_threshold=70)
    target = scene_plan("facts", 45)[0]  # 117
    lo, hi = int(0.75 * target), int(1.3 * target)  # 87 .. 152

    def replies(user):
        if len(llm.calls) == 1:
            # First rewrite: undersized (66 words) -> must be ignored, feedback recorded.
            assert f"between {lo} and {hi} words" in user
            return {
                "score": 58, "issues": ["thin hook"],
                "script": _full_script(_scenes(3, 22)),  # 66 words, below the 87-word floor
            }
        # Second attempt receives the targeted-expansion feedback.
        assert "The last rewrite had 66 words" in user
        assert "Expand it while preserving every fact" in user
        return {
            "score": 72, "issues": [],
            "script": _full_script(_scenes(5, 24)),  # 120 words, in range, higher score
        }

    llm = FakeLLM(replies)
    out = review_script(llm, _curio_stick_draft(), settings)

    assert out.score == 72 and out.review == []
    # The accepted rewrite meets the configured length requirement (120 in 87..152).
    assert lo <= word_count(out.narration) <= hi
    # Only the accepted rewrite replaced the draft; undersized rewrites never did.
    assert len(llm.calls) == 2  # API calls bounded to one retry


def test_both_rewrites_undersized_produces_clear_error_and_bounds_calls():
    """Retry exhaustion: every rewrite undersized -> ScriptQualityError with the
    correct reason, exactly two bounded API calls."""
    settings = Settings(target_seconds=45, script_quality_threshold=70)

    def replies(user):
        return {
            "score": 50, "issues": ["thin"],
            "script": _full_script(_scenes(3, 22)),  # 66 words, always below the floor
        }

    llm = FakeLLM(replies)
    with pytest.raises(
        ScriptQualityError,
        match="below the 70-point threshold after 2 bounded rewrite",
    ):
        review_script(llm, _curio_stick_draft(), settings)
    assert len(llm.calls) == 2  # bounded, no endless retrying


def test_rewrite_without_narration_is_ignored_then_retry_restores_narration():
    """A rewrite that drops all narration is ignored; targeted feedback on the
    bounded retry restores valid narration and a higher-scoring script wins.
    The original draft is never replaced by the empty rewrite."""
    settings = Settings(target_seconds=16, script_quality_threshold=70)
    target = scene_plan("facts", 16)[0]  # 41-word target
    lo, hi = int(0.75 * target), int(1.3 * target)  # 30 .. 53

    def replies(user):
        if len(llm.calls) == 1:
            # Rewrite drops every scene's narration -> must be ignored,
            # feedback recorded, original draft kept.
            return {
                "score": 50,
                "issues": ["missing narration"],
                "script": {
                    "title": "Empty", "hook_text": "H", "cast": [], "description": "d",
                    "hashtags": ["#cats"], "tags": ["cats"], "category": "pets",
                    "scenes": [
                        {"speaker": "A", "narration": "", "search_query": "q", "image_prompt": "p"}
                        for _ in range(4)
                    ],
                },
            }
        # Retry receives the narration feedback and returns valid narration.
        return {
            "score": 80, "issues": [],
            "script": _full_script(_scenes(5, 9)),  # 45 words, in range
        }

    llm = FakeLLM(replies)
    out = review_script(llm, _draft(), settings)

    assert out.score == 80 and out.review == []
    assert lo <= word_count(out.narration) <= hi  # accepted rewrite meets length rule
    assert len(llm.calls) == 2  # one bounded retry
    # The retry received targeted feedback about the missing narration.
    assert "no usable narration" in llm.calls[1]["user"]


def test_rewrite_with_unresolved_blocking_issues_is_rejected_then_resolved():
    """A rewrite that is in range and higher-scored but still carries blocking
    factual/safety issues is rejected; the retry that resolves them is accepted."""
    settings = Settings(target_seconds=16, script_quality_threshold=70)
    target = scene_plan("facts", 16)[0]  # 41-word target
    lo, hi = int(0.75 * target), int(1.3 * target)  # 30 .. 53

    def replies(user):
        if len(llm.calls) == 1:
            return {
                "score": 75,
                "issues": ["unverified claim"],
                "blocking_issues": [
                    "The claim that the brine has been trapped for two million years "
                    "cannot be verified.",
                ],
                "script": _full_script(_scenes(5, 9)),  # in range, higher score
            }
        # Retry receives blocking feedback and returns a clean rewrite.
        return {
            "score": 85, "issues": [], "blocking_issues": [],
            "script": _full_script(_scenes(5, 9)),
        }

    llm = FakeLLM(replies)
    out = review_script(llm, _draft(), settings)

    assert out.score == 85 and out.review == [] and out.blocking == []
    assert lo <= word_count(out.narration) <= hi
    assert len(llm.calls) == 2  # one bounded retry
    # The retry received targeted feedback about the unresolved blocking issues.
    assert "unresolved blocking issues" in llm.calls[1]["user"]
    assert "cannot be verified" in llm.calls[1]["user"]


def test_rewrite_keeps_blocking_issues_produces_clear_blocking_error():
    """Every rewrite stays in range and high-scored but keeps blocking issues:
    bounded retries then raise a clear blocking-issues error, preserving the draft."""
    settings = Settings(target_seconds=16, script_quality_threshold=70)

    def replies(user):
        return {
            "score": 80, "issues": ["unverified claim"],
            "blocking_issues": [
                "The claim that the brine has been trapped for two million years "
                "cannot be verified.",
                "Describing the flow as 'bleeding' contradicts the scientific "
                "explanation of iron-rich brine.",
            ],
            "script": _full_script(_scenes(5, 9)),
        }

    llm = FakeLLM(replies)
    with pytest.raises(
        ScriptQualityError,
        match="unresolved blocking factual/safety/usability issues",
    ):
        review_script(llm, _draft(), settings)
    assert len(llm.calls) == 2  # bounded, no endless retrying
