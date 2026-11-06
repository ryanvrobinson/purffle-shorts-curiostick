import json
from pathlib import Path

import pytest

from purffle_shorts.config import Settings
from purffle_shorts.llm import extract_json
from purffle_shorts.script import SCENE_SCHEMA, SCRIPT_SCHEMA, Scene, build_prompt, load_script, normalize, offline_script


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
    assert "Spanish" in user and "78 words" in user and "Old title" in user
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
