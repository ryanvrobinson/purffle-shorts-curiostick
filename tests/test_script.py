import json
from pathlib import Path

import pytest

from purffle_shorts.config import Settings
from purffle_shorts.llm import LLMError, extract_json
from purffle_shorts.script import (
    SCENE_SCHEMA,
    SCRIPT_SCHEMA,
    Scene,
    build_prompt,
    load_script,
    normalize,
    offline_script,
    scene_plan,
    write_script,
)


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Sure! Here it is:\n{"a": 1}\nHope that helps.',
    '{"a": 1,}',
])
def test_extract_json_variants(text):
    assert extract_json(text) == {"a": 1}


def test_normalize_cleans_everything():
    data = {
        "topic": "Octopus", "title": '"Octopus Secrets" #shorts 🐙', "hook_text": "3 HEARTS?!",
        "scenes": [
            {"narration": "[Hook] Narrator: Octopuses have three hearts! 🐙 #facts", "search_query": "octopus",
             "image_prompt": "an octopus"},
            {"narration": "", "search_query": "x", "image_prompt": "x"},
            "Plain string scene works too.",
        ],
        "description": "Wow. #octopus", "hashtags": ["octopus", "#Ocean Life", "#shorts", "#octopus"],
        "tags": ["a" * 70] + [f"tag{i}" for i in range(200)], "category": "nonsense",
    }
    s = normalize(data, "octopus", "facts", "en")
    assert s.scenes[0].narration == "Octopuses have three hearts!"
    assert len(s.scenes) == 2
    assert s.title == "Octopus Secrets"
    assert s.hashtags[0] == "#shorts" and "#OceanLife" in s.hashtags
    assert len([h for h in s.hashtags if h.lower() == "#octopus"]) == 1
    assert sum(len(t) + 1 for t in s.tags) <= 500
    assert s.category == "education" and s.category_id == "27"
    assert "#" not in s.description


def test_normalize_merges_too_many_scenes():
    data = {"scenes": [{"narration": f"Sentence {i}.", "search_query": "q", "image_prompt": "p"} for i in range(20)]}
    s = normalize(data, "x", "facts", "en")
    assert len(s.scenes) == 12
    assert s.narration.count("Sentence") == 20


def test_offline_script_is_complete():
    s = offline_script("", Settings())
    assert s.title and len(s.scenes) >= 4 and s.hashtags[0] == "#shorts"


def test_offline_script_uses_distinct_queries_for_a_generic_topic():
    s = offline_script("the history of coffee", Settings())
    queries = [sc.search_query for sc in s.scenes]
    assert len(queries) == len(set(queries)) > 1
    assert all("coffee" in q.lower() for q in queries)


def test_prompt_mentions_language_and_length():
    system, user = build_prompt("space", "facts", Settings(language="es", target_seconds=30), ["Old title"])
    assert "Spanish" in user and "between 70 and 85 words" in user and "Old title" in user
    assert "JSON" in system
    assert set(SCRIPT_SCHEMA["required"]) >= {"title", "scenes", "hashtags"}
    assert "most surprising well-established fact" in user
    assert "No greeting, \"Did you know?\"" in user
    assert "Omit generic outros" in user
    assert "actual place, geological/biological/astronomical subject" in user


def test_retention_defaults_favor_short_visual_beats():
    settings = Settings()
    assert settings.target_seconds == 30
    assert "Chicxulub" in settings.niches[1]
    from purffle_shorts.script import scene_plan
    assert scene_plan("facts", settings.target_seconds)[1] == 9


def test_load_script_cleans_like_an_ai_script(tmp_path):
    f = tmp_path / "mine.json"
    f.write_text(json.dumps({"title": "My Own Script #viral", "style": "not-a-style", "hashtags": ["cats"],
                             "scenes": [{"narration": "Cats sleep a lot.", "search_query": "sleeping cat"}]}))
    sc = load_script(f, Settings())
    assert sc.title == "My Own Script" and sc.style == "facts"
    assert sc.hashtags[:2] == ["#shorts", "#cats"] and sc.topic == "My Own Script #viral"
    assert sc.scenes[0].image_prompt == "sleeping cat"


def test_load_script_rejects_non_objects(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("[1, 2]")
    with pytest.raises(ValueError):
        load_script(f, Settings())


@pytest.mark.parametrize("path", sorted((Path(__file__).parent.parent / "examples").glob("*.json")))
def test_shipped_examples_load(path):
    sc = load_script(path, Settings())
    assert len(sc.scenes) >= 4 and sc.hook_text and all(s.image_prompt for s in sc.scenes)


def test_scene_backward_compatible_fields_only():
    # Old-style construction with only the original fields still works.
    scene = Scene(narration="Hello world.", search_query="hello", image_prompt="a hello")
    assert scene.narration == "Hello world."
    assert scene.search_query == "hello"
    assert scene.image_prompt == "a hello"
    assert scene.speaker == "A"


def test_scene_new_optional_fields_default_to_empty():
    scene = Scene(narration="Hello world.", search_query="hello", image_prompt="a hello")
    assert scene.duration is None
    assert scene.visual_description == ""
    assert scene.visual_source == ""
    assert scene.attribution == ""


def test_normalize_passes_through_optional_scene_fields():
    data = {
        "title": "T",
        "scenes": [
            {
                "narration": "First line.",
                "search_query": "first",
                "image_prompt": "a first",
                "duration": 6.5,
                "visual_description": "A wide shot of the ocean.",
                "visual_source": "stock",
                "attribution": "Photographer via Pexels",
            }
        ],
    }
    s = normalize(data, "x", "facts", "en")
    assert len(s.scenes) == 1
    sc = s.scenes[0]
    assert sc.duration == 6.5
    assert sc.visual_description == "A wide shot of the ocean."
    assert sc.visual_source == "stock"
    assert sc.attribution == "Photographer via Pexels"


def test_settings_defaults_preserve_existing_behaviour():
    s = Settings()
    assert s.visual_choice == "stock-first"
    assert s.research_verify is False
    assert s.verify_provider == ""


def test_scene_schema_accepts_optional_new_fields():
    schema = SCENE_SCHEMA
    assert "duration" in schema["properties"]
    assert "visual_description" in schema["properties"]
    assert "visual_source" in schema["properties"]
    assert "attribution" in schema["properties"]
    # None of the new fields are required, so old LLM output still validates.
    assert "duration" not in schema["required"]
    assert "visual_description" not in schema["required"]
    assert "visual_source" not in schema["required"]
    assert "attribution" not in schema["required"]


class _WriteFakeLLM:
    """Fake LLM for testing write_script with scripted replies."""

    label = "fake:model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete_json(self, system, user, schema=None, max_tokens=4000):
        self.calls.append(
            {"system": system, "user": user, "schema": schema, "max_tokens": max_tokens}
        )
        if not self.replies:
            raise LLMError("no more replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _full_script_data(scenes, title="X"):
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


def _scenes_data(n, words=8):
    return [
        {
            "speaker": "A",
            "narration": f"s{i} " + " ".join(["word"] * (words - 1)) + ".",
            "search_query": f"q{i}",
            "image_prompt": f"p{i}",
        }
        for i in range(n)
    ]


def test_write_script_accepts_complete_script_on_first_try():
    """A complete script is accepted without retry."""
    settings = Settings(target_seconds=30)
    words, n_scenes = scene_plan("facts", 30)

    def replies():
        yield _full_script_data(_scenes_data(n_scenes, words // n_scenes))

    llm = _WriteFakeLLM(replies())
    script = write_script(llm, "cats", settings)

    assert script.title == "X"
    assert len(script.scenes) == n_scenes
    assert len(llm.calls) == 1  # no retry needed


def test_write_script_rejects_truncated_and_retries():
    """An undersized initial script is rejected; the retry with more tokens produces a complete script."""
    settings = Settings(target_seconds=30)
    words, n_scenes = scene_plan("facts", 30)

    def replies():
        # First attempt: truncated (2 scenes, few words)
        yield _full_script_data(_scenes_data(2, 9))
        # Second attempt: complete
        yield _full_script_data(_scenes_data(n_scenes, words // n_scenes))

    llm = _WriteFakeLLM(replies())
    script = write_script(llm, "cats", settings)

    assert script.title == "X"
    assert len(script.scenes) == n_scenes
    assert len(llm.calls) == 2  # first undersized, then complete


def test_write_script_rejects_undersized_without_recovery():
    """Two undersized attempts in a row raise LLMError."""
    def replies():
        yield _full_script_data(_scenes_data(2, 9))  # undersized
        yield _full_script_data(_scenes_data(3, 9))  # still undersized

    llm = _WriteFakeLLM(replies())
    settings = Settings(target_seconds=30)
    with pytest.raises(LLMError, match="incomplete script"):
        write_script(llm, "cats", settings)
    assert len(llm.calls) == 2  # both attempts used, then raised

def test_write_script_rejects_oversized_and_retries():
    """An oversized initial script is rejected; the retry with a concise prompt produces an in-range script."""
    settings = Settings(target_seconds=16)
    words, n_scenes = scene_plan("facts", 16)

    def replies():
        # First attempt: oversized script (too many words)
        yield _full_script_data(_scenes_data(n_scenes + 2, words // 2))
        # Second attempt: in-range script
        yield _full_script_data(_scenes_data(n_scenes, words // n_scenes))

    llm = _WriteFakeLLM(replies())
    script = write_script(llm, "cats", settings)

    assert script.title == "X"
    assert len(script.scenes) == n_scenes
    assert len(llm.calls) == 2  # first oversized, then in-range


def test_write_script_rejects_oversized_without_recovery():
    """Two oversized attempts in a row raise LLMError."""
    def replies():
        yield _full_script_data(_scenes_data(10, 50))  # oversized
        yield _full_script_data(_scenes_data(10, 50))  # still oversized

    llm = _WriteFakeLLM(replies())
    settings = Settings(target_seconds=16)
    with pytest.raises(LLMError, match="overlong script"):
        write_script(llm, "cats", settings)
    assert len(llm.calls) == 2

def test_validate_opening_hook_rejects_generic_greeting():
    """Generic greeting hook is rejected before TTS/images/render."""
    from purffle_shorts.script import Scene, Script, validate_opening_hook
    script = Script(
        topic="cats", title="T", hook_text="HOOK", scenes=[
            Scene(narration="Did you know cats have whiskers?", search_query="cats", image_prompt="cats"),
        ],
        description="d", hashtags=["#cats"], tags=["cats"], category="pets",
    )
    issue = validate_opening_hook(script)
    assert issue is not None
    assert "greeting" in issue or "Did you know" in issue


def test_validate_opening_hook_rejects_did_you_know():
    """'Did you know?' opening is rejected."""
    from purffle_shorts.script import Scene, Script, validate_opening_hook
    script = Script(
        topic="cats", title="T", hook_text="HOOK", scenes=[
            Scene(narration="Did you know octopuses have three hearts?", search_query="octopus", image_prompt="octopus"),
        ],
        description="d", hashtags=["#cats"], tags=["cats"], category="pets",
    )
    issue = validate_opening_hook(script)
    assert issue is not None


def test_validate_opening_hook_accepts_specific_opening():
    """Specific subject-opening hook is accepted."""
    from purffle_shorts.script import Scene, Script, validate_opening_hook
    script = Script(
        topic="cats", title="T", hook_text="HOOK", scenes=[
            Scene(narration="Octopuses have three hearts and blue blood.", search_query="octopus", image_prompt="octopus"),
        ],
        description="d", hashtags=["#cats"], tags=["cats"], category="pets",
    )
    issue = validate_opening_hook(script)
    assert issue is None


def test_validate_opening_hook_empty_narration():
    """Empty opening narration is rejected."""
    from purffle_shorts.script import Scene, Script, validate_opening_hook
    script = Script(
        topic="cats", title="T", hook_text="HOOK", scenes=[
            Scene(narration="", search_query="cats", image_prompt="cats"),
        ],
        description="d", hashtags=["#cats"], tags=["cats"], category="pets",
    )
    issue = validate_opening_hook(script)
    assert issue is not None


def test_validate_opening_hook_no_scenes():
    """Script with no scenes is rejected."""
    from purffle_shorts.script import Script, validate_opening_hook
    script = Script(
        topic="cats", title="T", hook_text="HOOK", scenes=[],
        description="d", hashtags=["#cats"], tags=["cats"], category="pets",
    )
    issue = validate_opening_hook(script)
    assert issue is not None
