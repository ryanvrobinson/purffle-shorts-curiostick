"""Two-speaker scripts, the Script Doctor and translation — with fake LLMs, no network."""

import logging

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
    validate_opening_hook,
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

    assert not result.ok and "below the 50-point threshold" in result.error
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


def test_rewrite_rejected_for_invalid_structure_retains_draft():
    """A rewrite whose script fails the structure check (e.g. a scene with no
    narration) is ignored and the draft is retained with its correct score."""
    script = _draft()
    script.score = 70
    invalid = {
        "title": "T", "hook_text": "H", "cast": [], "scenes": [
            {"speaker": "A", "narration": "", "search_query": "q", "image_prompt": "p"},
        ],
        "description": "d", "hashtags": [], "tags": [], "category": "pets",
    }
    llm = FakeLLM({"score": 88, "issues": ["structure"], "blocking_issues": ["structure"], "script": invalid})
    settings = Settings(script_quality_threshold=50)
    with pytest.raises(ScriptQualityError, match="unresolved blocking"):
        review_script(llm, script, settings)
    assert len(llm.calls) == 2  # bounded, no endless retrying
    # The rejected rewrite must not overwrite the retained draft's score.
    assert script.score == 70


def test_rejected_higher_scoring_rewrite_retains_correct_score():
    """A rewrite that scores higher but carries no usable script must not
    overwrite the retained draft's score (the Slot 1 regression)."""
    script = _draft()
    script.score = 50  # already passes the new 50/100 threshold
    llm = FakeLLM({"score": 95, "issues": [], "script": None})  # no script -> not adopted
    out = review_script(llm, script, Settings(script_quality_threshold=50))
    assert out is script                      # same retained object
    assert out.score == 50                    # retained draft's score, not the rejected 95
    assert len(llm.calls) == 1                # score >= threshold: no retry


# ------------------------------------------------------------------ hook-only rewrite regression tests

def _script_with_bad_hook():
    """Script whose opening uses a generic 'Did you know?' greeting."""
    return normalize({
        "title": "T", "hook_text": "HOOK", "cast": [],
        "scenes": [
            {"speaker": "A", "narration": "Did you know octopuses have three hearts?",
             "search_query": "octopus hearts", "image_prompt": "octopus"},
            {"speaker": "A", "narration": "Their blood is blue because of copper.",
             "search_query": "octopus blood", "image_prompt": "blue blood"},
        ],
        "description": "d", "hashtags": ["#cats"], "tags": ["cats"], "category": "pets",
    }, "cats", "facts", "en")


def test_hook_rewrite_fixes_opening_preserves_rest():
    """Hook-only rewrite replaces only the first scene's narration."""
    from purffle_shorts.pipeline import Studio
    settings = Settings(target_seconds=16)
    studio = Studio.__new__(Studio)
    studio.s = settings

    script = _script_with_bad_hook()
    assert validate_opening_hook(script) is not None

    class FakeLLM:
        label = "fake:model"
        def complete_json(self, system, user, schema=None, max_tokens=4000):
            return {
                "title": "T", "hook_text": "FIXED HOOK", "cast": [],
                "scenes": [
                    {"speaker": "A", "narration": "Octopuses have three hearts and blue blood.",
                     "search_query": "octopus hearts", "image_prompt": "octopus"},
                    {"speaker": "A", "narration": "Their blood is blue because of copper.",
                     "search_query": "octopus blood", "image_prompt": "blue blood"},
                ],
                "description": "d", "hashtags": ["#cats"], "tags": ["cats"], "category": "pets",
            }

    result = studio._rewrite_hook_only(FakeLLM(), script, settings, "generic greeting")
    assert validate_opening_hook(result) is None
    # Second scene preserved unchanged
    assert result.scenes[1].narration == "Their blood is blue because of copper."
    assert result.scenes[1].search_query == "octopus blood"


def test_hook_rewrite_cannot_bypass_factual_verification(tmp_path):
    """Hook rewrite does not skip fact verification; needs_human_review still blocks."""
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
        "hashtags": [], "tags": [], "category": "science",
    }

    class FakeLLM:
        label = "fake:model"
        def __init__(self):
            self.reviews = 0
        def complete_json(self, system, user, schema=None, max_tokens=4000):
            if schema is SCRIPT_SCHEMA:
                return data
            self.reviews += 1
            return {"score": 95, "issues": [], "script": data}

    settings = Settings(target_seconds=16, script_review="rewrite", research_verify=True,
                        output_dir=str(tmp_path / "out"), data_dir=str(tmp_path / "data"))
    studio = pipeline.Studio(settings)
    llm = FakeLLM()
    studio._llm = llm

    # verification_report with needs_human_review should block production
    script = normalize(data, "neutron stars", "facts", "en")
    script.verification_report = {
        "status": "needs_human_review",
        "checked_by": "heuristic",
        "source_count": 0,
        "llm_extraction_failed": False,
        "claims": [{"text": "unverified claim", "status": "needs_human_review"}],
    }

    # draft() should raise because verification is needs_human_review
    try:
        studio.draft(topic="neutron stars")
        raise AssertionError("draft() should have raised RuntimeError for needs_human_review")
    except RuntimeError as e:
        assert "human review" in str(e).lower() or "unresolved" in str(e).lower()


def test_invalid_claim_extraction_json_fails_safely():
    """LLM claim extraction returning incomplete JSON twice raises LLMError."""
    from purffle_shorts.llm import LLMError
    from purffle_shorts.research import _extract_claims_llm

    class FailingLLM:
        name = "custom"
        model = "openai/gpt-oss-20b"
        base_url = "https://api.groq.com/openai/v1"
        api_key = "test-key"

        def complete_json(self, system, user, schema=None, max_tokens=4000):
            # First call: incomplete JSON (finish_reason=length)
            raise LLMError("empty reply (finish_reason=length)")

    llm = FailingLLM()
    try:
        _extract_claims_llm(llm, "Octopuses have three hearts. Their blood is blue.", "octopuses")
        raise AssertionError("_extract_claims_llm should have raised LLMError")
    except LLMError:
        pass  # expected: failure propagates, no silent heuristic fallback

# ------------------------------------------------------------------ duplicate upload and asset reuse tests

def test_duplicate_upload_prevented(monkeypatch):
    """_check_duplicate_upload raises when YouTube already has the video."""
    from pathlib import Path

    from purffle_shorts.config import Settings
    from purffle_shorts.pipeline import Studio

    settings = Settings()
    studio = Studio.__new__(Studio)
    studio.s = settings

    class FakeYT:
        def search(self):
            return self
        def list(self, **kwargs):
            return self
        def execute(self):
            return {"items": [{"id": {"videoId": "existing123"}}]}

    class FakeYouTubeModule:
        @staticmethod
        def service(settings, interactive=False):
            return FakeYT()
        @staticmethod
        def find_existing_video(yt, marker):
            return "existing123"

    import purffle_shorts.pipeline as pip
    monkeypatch.setattr(pip, "youtube", FakeYouTubeModule)

    try:
        studio._check_duplicate_upload(settings, "My Video", Path("video.mp4"))
        raise AssertionError("Should have raised RuntimeError for duplicate")
    except RuntimeError as e:
        assert "already exists on YouTube" in str(e)


def test_duplicate_upload_check_failure_does_not_block(monkeypatch):
    """Duplicate check failure (network error) never blocks upload."""
    from pathlib import Path

    from purffle_shorts.config import Settings
    from purffle_shorts.pipeline import Studio

    settings = Settings()
    studio = Studio.__new__(Studio)
    studio.s = settings

    class FakeYouTubeModule:
        @staticmethod
        def service(settings, interactive=False):
            raise ConnectionError("network down")
        @staticmethod
        def find_existing_video(yt, marker):
            return None

    import purffle_shorts.pipeline as pip
    monkeypatch.setattr(pip, "youtube", FakeYouTubeModule)

    # Should not raise — duplicate check failure is non-blocking
    studio._check_duplicate_upload(settings, "My Video", Path("video.mp4"))


# ------------------------------------------------------------------ 50/100 production threshold tests
# The Script Doctor production threshold is now 50/100:
#   score >= 50  -> production allowed (score + findings + decision logged)
#   score < 50   -> production blocked (ScriptQualityError)
# Hard gates (invalid structure, unresolved factual claims, missing narration,
# unusable scene data) still block production regardless of score.


def _reply(score, blocking_issues=None, script=None):
    return {"score": score, "issues": [], "blocking_issues": blocking_issues or [], "script": script}


def test_score_95_allows_production(caplog):
    caplog.set_level(logging.INFO)
    llm = FakeLLM(_reply(95))
    review_script(llm, _draft(), Settings(target_seconds=16, script_review="rewrite"))
    assert len(llm.calls) == 1  # passed on the first attempt, no retry
    assert any("production_allowed=True" in r.getMessage() and "95" in r.getMessage()
               for r in caplog.records)


def test_score_82_allows_production(caplog):
    caplog.set_level(logging.INFO)
    llm = FakeLLM(_reply(82))
    review_script(llm, _draft(), Settings(target_seconds=16, script_review="rewrite"))
    assert len(llm.calls) == 1
    assert any("production_allowed=True" in r.getMessage() for r in caplog.records)


def test_score_65_allows_production(caplog):
    caplog.set_level(logging.INFO)
    llm = FakeLLM(_reply(65))
    review_script(llm, _draft(), Settings(target_seconds=16, script_review="rewrite"))
    assert len(llm.calls) == 1
    assert any("production_allowed=True" in r.getMessage() for r in caplog.records)


def test_score_50_allows_production(caplog):
    caplog.set_level(logging.INFO)
    llm = FakeLLM(_reply(50))
    review_script(llm, _draft(), Settings(target_seconds=16, script_review="rewrite"))
    assert len(llm.calls) == 1
    assert any("production_allowed=True" in r.getMessage() for r in caplog.records)


def test_score_49_blocks_production():
    llm = FakeLLM(_reply(49))
    with pytest.raises(ScriptQualityError, match="below the 50-point threshold"):
        review_script(llm, _draft(), Settings(target_seconds=16, script_review="rewrite"))
    assert len(llm.calls) == 2  # bounded retries, then stop


def test_score_0_blocks_production():
    llm = FakeLLM(_reply(0))
    with pytest.raises(ScriptQualityError, match="below the 50-point threshold"):
        review_script(llm, _draft(), Settings(target_seconds=16, script_review="rewrite"))
    assert len(llm.calls) == 2


def test_score_65_with_unresolved_factual_claims_blocks_production():
    llm = FakeLLM(_reply(
        65, blocking_issues=["The claim that the brine is two million years old cannot be verified."]))
    with pytest.raises(ScriptQualityError, match="unresolved blocking"):
        review_script(llm, _draft(), Settings(target_seconds=16, script_review="rewrite"))
    assert len(llm.calls) == 2


def test_score_65_with_invalid_scene_structure_blocks_production():
    llm = FakeLLM(_reply(
        65, blocking_issues=["Scene 2 has no narration: invalid scene structure."]))
    with pytest.raises(ScriptQualityError, match="unresolved blocking"):
        review_script(llm, _draft(), Settings(target_seconds=16, script_review="rewrite"))
    assert len(llm.calls) == 2


def test_rejected_rewrite_retains_script_and_score_at_50_threshold():
    """A rejected rewrite must not overwrite the retained script or its score."""
    script = _draft()
    script.score = 55  # passes the 50/100 threshold
    invalid = {
        "title": "T", "hook_text": "H", "cast": [], "scenes": [
            {"speaker": "A", "narration": "", "search_query": "q", "image_prompt": "p"},
        ],
        "description": "d", "hashtags": [], "tags": [], "category": "pets",
    }
    # Rewrite scores higher (88) but is structurally invalid -> must be rejected.
    llm = FakeLLM({"score": 88, "issues": ["structure"], "blocking_issues": ["structure"], "script": invalid})
    with pytest.raises(ScriptQualityError, match="unresolved blocking"):
        review_script(llm, script, Settings(script_quality_threshold=50))
    assert len(llm.calls) == 2  # bounded retries, not endless
    # Retained draft is unchanged and keeps its correct score.
    assert script.score == 55
    assert script.title == "Draft"


# ------------------------------------------------------------------ factual-claim recovery tests
# Slot 1, 2026-11-11: Script Doctor flagged "Depth claim about lake may be
# inaccurate." and "NASA probe plan is speculative." as blocking issues. The
# rewrite loop must remove or rewrite such questionable claims (never invent
# replacement facts/dates/measurements) and re-evaluate; production is allowed
# only when all blocking claims are resolved AND the score is >= 50.


def _blocking_reply(score, blocking, script):
    return {"score": score, "issues": ["Unsupported claims"], "blocking_issues": blocking, "script": script}


def _clean_reply(score, script):
    return {"score": score, "issues": [], "blocking_issues": [], "script": script}


def test_speculative_nasa_dates_removed_then_production_allowed(caplog):
    """A rewrite that removes speculative mission dates (no replacement dates
    invented) resolves the blocking issues and allows production at 50+."""
    caplog.set_level(logging.INFO)
    settings = Settings(target_seconds=16, script_quality_threshold=50)
    # 41-word target, window 30..53; better script (36 words) is in range.

    draft = _draft()
    draft.score = 55
    better = _full_script(_scenes(4, 9))  # 36 words, in range

    def replies(user):
        if len(llm.calls) == 1:
            return _blocking_reply(
                55,
                ["NASA probe plan is speculative.", "Depth claim about lake may be inaccurate."],
                better,
            )
        # The rewrite removed the speculative date claim; no replacement date invented.
        return _clean_reply(70, better)

    llm = FakeLLM(replies)
    out = review_script(llm, draft, settings)

    assert out.score == 70
    assert out.blocking == [] and out.review == []
    assert len(llm.calls) == 2  # one bounded rewrite
    assert any("production_allowed=True" in r.getMessage() and "70" in r.getMessage()
               for r in caplog.records)


def test_uncertain_measurement_removed_then_production_allowed(caplog):
    """A rewrite that drops an uncertain scientific measurement (no invented
    replacement measurement) resolves blocking and allows production at 50+."""
    caplog.set_level(logging.INFO)
    settings = Settings(target_seconds=16, script_quality_threshold=50)
    draft = _draft()
    draft.score = 60
    better = _full_script(_scenes(4, 9))

    def replies(user):
        if len(llm.calls) == 1:
            return _blocking_reply(
                60,
                ["The depth figure for the lake is uncertain and unsupported."],
                better,
            )
        return _clean_reply(68, better)

    llm = FakeLLM(replies)
    out = review_script(llm, draft, settings)

    assert out.score == 68
    assert out.blocking == []
    assert len(llm.calls) == 2
    assert any("production_allowed=True" in r.getMessage() for r in caplog.records)


def test_unsupported_claim_not_silently_accepted():
    """A high score does not bypass factual verification: unresolved blocking
    claims still block production even when the score is well above 50."""
    draft = _draft()
    draft.score = 85
    llm = FakeLLM(_blocking_reply(
        85,
        ["The claim that the probe launched in 2029 cannot be verified."],
        _full_script(_scenes(4, 9)),
    ))
    with pytest.raises(ScriptQualityError, match="unresolved blocking"):
        review_script(llm, draft, Settings(target_seconds=16, script_quality_threshold=50))
    assert len(llm.calls) == 2  # bounded retries, then stop


def test_unresolved_factual_claims_block_after_bounded_retries():
    """Every rewrite keeps the same speculative claims; bounded retries then
    block production with the exact reason recorded."""
    draft = _draft()
    draft.score = 55
    llm = FakeLLM(lambda user: _blocking_reply(
        70,
        ["NASA probe plan is speculative."],
        _full_script(_scenes(4, 9)),
    ))
    with pytest.raises(
        ScriptQualityError,
        match="unresolved blocking factual/safety/usability issues",
    ):
        review_script(llm, draft, Settings(target_seconds=16, script_quality_threshold=50))
    assert len(llm.calls) == 2  # bounded, no endless retrying
    # Draft is never replaced by the repeatedly-rejected rewrite.
    assert draft.score == 55


def test_factual_repair_score_association_after_rewrite():
    """After a successful factual repair, the adopted rewrite's score is
    associated with the new script (not the old draft's score)."""
    settings = Settings(target_seconds=16, script_quality_threshold=50)
    draft = _draft()
    draft.score = 50
    better = _full_script(_scenes(4, 9), title="Revised")

    def replies(user):
        if len(llm.calls) == 1:
            return _blocking_reply(50, ["Depth claim about lake may be inaccurate."], better)
        return _clean_reply(75, better)

    llm = FakeLLM(replies)
    out = review_script(llm, draft, settings)

    assert out is not draft                     # the rewrite was adopted
    assert out.title == "Revised"
    assert out.score == 75                      # new script's score, not the draft's 50
    assert out.blocking == []
    assert len(llm.calls) == 2


# ------------------------------------------------------------------ 95-word / length and scene-level repair tests
def test_95_word_rewrite_in_range_not_rejected_for_length():
    """A 95-word rewrite inside the permitted window is not rejected for length."""
    settings = Settings(target_seconds=40, script_quality_threshold=50)
    target = scene_plan("facts", 40)[0]  # 104-word target
    lo, hi = int(0.75 * target), int(1.3 * target)  # 78 .. 135
    assert lo <= 95 <= hi  # 95 words is inside the permitted window

    draft = _draft()
    draft.score = 40
    # 5 scenes x 19 words = 95 words, in range
    better = _full_script(_scenes(5, 19), title="Ninety-five")

    def replies(user):
        assert f"between {lo} and {hi} words" in user
        return _clean_reply(70, better)

    llm = FakeLLM(replies)
    out = review_script(llm, draft, settings)

    assert out.title == "Ninety-five"
    assert out.score == 70
    assert out.blocking == []
    # Passed on the first attempt: not rejected for length, no bounded retry.
    assert len(llm.calls) == 1


def test_incomplete_sentence_scene_7_repaired():
    """Scene 7 with an incomplete sentence is rewritten into a complete sentence."""
    settings = Settings(target_seconds=16, script_quality_threshold=50)
    scenes = _scenes(8, 9)
    scenes[6]["narration"] = "water-rich ejecta that"  # incomplete fragment
    draft = normalize(_full_script(scenes), "cats", "facts", "en")
    draft.score = 50
    draft.blocking = ["Scene 7 incomplete sentence about water-rich ejecta."]
    # 8 scenes x 6 words = 48 words, inside the permitted 30..53 window.
    better = _full_script(_scenes(8, 6), title="Repaired")

    def replies(user):
        if len(llm.calls) == 1:
            return _blocking_reply(50, ["Scene 7 incomplete sentence about water-rich ejecta."], better)
        return _clean_reply(65, better)

    llm = FakeLLM(replies)
    out = review_script(llm, draft, settings)

    assert out.score == 65
    assert out.blocking == []
    assert len(llm.calls) == 2  # one bounded rewrite repaired the incomplete sentence
    # The prompt targeted Scene 7 and its narration, and asked for a complete sentence.
    assert "Scene 7" in llm.calls[0]["user"]
    assert "water-rich ejecta that" in llm.calls[0]["user"]
    assert "complete, grammatical spoken sentence" in llm.calls[0]["user"]


def test_vague_final_payoff_repaired():
    """A vague final-scene payoff is rewritten to answer the opening mystery explicitly."""
    settings = Settings(target_seconds=16, script_quality_threshold=50)
    draft = _draft()
    draft.score = 50
    draft.blocking = ["Final scene payoff too vague."]
    better = _full_script(_scenes(4, 9), title="Payoff")

    def replies(user):
        if len(llm.calls) == 1:
            return _blocking_reply(50, ["Final scene payoff too vague."], better)
        return _clean_reply(60, better)

    llm = FakeLLM(replies)
    out = review_script(llm, draft, settings)

    assert out.score == 60
    assert out.blocking == []
    assert len(llm.calls) == 2
    # The prompt asked the final scene to answer the opening mystery explicitly.
    assert "final scene" in llm.calls[0]["user"].lower()
    assert "answer the opening mystery explicitly" in llm.calls[0]["user"]


def test_repaired_candidate_score_50_passes(caplog):
    """A repaired candidate scoring exactly 50 passes the quality gate."""
    caplog.set_level(logging.INFO)
    settings = Settings(target_seconds=16, script_quality_threshold=50)
    draft = _draft()
    draft.score = 45
    better = _full_script(_scenes(4, 9), title="Fixed")

    def replies(user):
        if len(llm.calls) == 1:
            return _blocking_reply(45, ["Some issue"], better)
        return _clean_reply(50, better)

    llm = FakeLLM(replies)
    out = review_script(llm, draft, settings)

    assert out.score == 50
    assert out.blocking == []
    assert any("production_allowed=True" in r.getMessage() and "50" in r.getMessage()
               for r in caplog.records)


def test_candidate_score_49_fails():
    """A candidate scoring 49 is below the 50/100 threshold and blocks production."""
    draft = _draft()
    draft.score = 60
    llm = FakeLLM(_clean_reply(49, _full_script(_scenes(4, 9))))
    with pytest.raises(ScriptQualityError, match="below the 50-point threshold"):
        review_script(llm, draft, Settings(target_seconds=16, script_quality_threshold=50))
    assert len(llm.calls) == 2  # bounded retries, then stop


def test_all_workflows_use_same_threshold():
    """All three slot workflows set SCRIPT_QUALITY_THRESHOLD to 50."""
    import yaml
    for slot in ("1", "2", "3"):
        with open(f".github/workflows/curiostick-slot-{slot}.yml") as fh:
            doc = yaml.safe_load(fh)
        found = []

        def walk(node, acc):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "env" and isinstance(v, dict) and "SCRIPT_QUALITY_THRESHOLD" in v:
                        acc.append(v["SCRIPT_QUALITY_THRESHOLD"])
                    walk(v, acc)
            elif isinstance(node, list):
                for item in node:
                    walk(item, acc)

        walk(doc, found)
        assert "50" in found, (
            f"slot {slot}: no SCRIPT_QUALITY_THRESHOLD=50 found; env values={found}"
        )
