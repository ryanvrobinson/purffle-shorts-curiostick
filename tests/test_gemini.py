"""Gemini provider tests — every Google SDK call is faked, no API credits are consumed."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from purffle_shorts import llm as L
from purffle_shorts.config import Settings
from purffle_shorts.script import Scene, Script


def _install_fake_genai(monkeypatch, captured=None, *, fail=False, rate_limited=False,
                        auth_error=False, empty=False, not_found=False):
    """Replace ``google.genai`` with a fake that records constructor + generate_content args."""
    captured = [] if captured is None else captured
    mod = types.ModuleType("google.genai")
    errors_mod = types.ModuleType("google.genai.errors")
    errors_mod.APIError = type("APIError", (Exception,), {})

    def _api_error(code: int, message: str) -> Exception:
        exc = errors_mod.APIError(message)
        exc.code = code
        return exc

    class FakeClient:
        def __init__(self, api_key=None, http_options=None):
            captured.append({"api_key": api_key, "http_options": http_options})
            self.models = FakeModels(captured, fail=fail, rate_limited=rate_limited,
                                     auth_error=auth_error, empty=empty, not_found=not_found)

    class FakeModels:
        def __init__(self, captured, fail, rate_limited, auth_error, empty, not_found):
            self.captured = captured
            self.fail, self.rate_limited, self.auth_error, self.empty, self.not_found = \
                fail, rate_limited, auth_error, empty, not_found
            self.calls = 0

        def generate_content(self, model, contents, config=None, **kwargs):
            self.calls += 1
            call = {"model": model, "contents": contents, "config": config, "kwargs": kwargs}
            self.captured.append(call)
            if self.not_found:
                raise _api_error(404, "This model models/gemini-2.5-flash is no longer available")
            if self.fail:
                raise _api_error(500, "server error")
            if self.rate_limited:
                raise _api_error(429, "rate limited")
            if self.auth_error:
                raise _api_error(401, "invalid key")
            if self.empty:
                return SimpleNamespace(text="")
            return SimpleNamespace(text='{"ok": true}', usage_metadata=SimpleNamespace(total_token_count=12))

    types_mod = types.ModuleType("google.genai.types")

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    types_mod.GenerateContentConfig = GenerateContentConfig
    types_mod.HttpOptions = SimpleNamespace

    genai_pkg = types.ModuleType("google")
    genai_pkg.genai = mod
    sys.modules["google"] = genai_pkg
    sys.modules["google.genai"] = mod
    sys.modules["google.genai.errors"] = errors_mod
    sys.modules["google.genai.types"] = types_mod
    mod.Client = FakeClient
    mod.errors = errors_mod
    mod.types = types_mod
    return mod


def test_gemini_provider_uses_sdk_and_caps_output_tokens(monkeypatch):
    captured = []
    _install_fake_genai(monkeypatch, captured)
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key-123")
    p = L.build_provider("gemini", Settings(llm_provider="gemini", llm_model="gemini-3.8-flash"))
    assert isinstance(p, L.GeminiProvider)
    assert p.model == "gemini-3.8-flash"
    out = p.complete("sys", "user", schema={"type": "object"}, max_tokens=12000)
    assert out == '{"ok": true}'
    cfg = captured[-1]["config"]
    assert cfg.kwargs["max_output_tokens"] == 8000  # capped from 12000
    assert cfg.kwargs["response_mime_type"] == "application/json"
    assert cfg.kwargs["response_schema"] == {"type": "object"}
    # The key never reaches a log or the message payload: constructor got it, config carries none.
    assert captured[0]["api_key"] == "gem-key-123"
    assert "gem-key-123" not in str(captured[-1])


def test_gemini_default_model_from_settings(monkeypatch):
    captured = []
    _install_fake_genai(monkeypatch, captured)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    p = L.build_provider("gemini", Settings(llm_provider="gemini"))
    assert p.model == "gemini-3.8-flash"  # preset default


def test_gemini_http_options_timeout_is_120_seconds_not_60_ms(monkeypatch):
    """google-genai's HttpOptions.timeout is in MILLISECONDS.

    Regression: _REQUEST_TIMEOUT was 60, which the SDK read as 60 milliseconds (0.06s) and
    caused the Slot 3 Gemini failures. It must be 120_000 ms (120 seconds), never 60.
    """
    captured = []
    _install_fake_genai(monkeypatch, captured)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    p = L.build_provider("gemini", Settings(llm_provider="gemini"))
    assert isinstance(p, L.GeminiProvider)
    http_options = captured[0]["http_options"]
    assert http_options.timeout == 120_000, http_options
    assert http_options.timeout != 60  # 60 ms = 0.06s request timeout
    assert L.GeminiProvider._REQUEST_TIMEOUT == 120_000


def test_gemini_missing_api_key_raises_config_error(monkeypatch):
    _install_fake_genai(monkeypatch, [])
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(L.LLMConfigError, match="GEMINI_API_KEY"):
        L.build_provider("gemini", Settings(llm_provider="gemini"))


def test_gemini_valid_json_response(monkeypatch):
    captured = []
    _install_fake_genai(monkeypatch, captured)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    p = L.build_provider("gemini", Settings(llm_provider="gemini"))
    chain = object.__new__(L.LLM)
    chain.settings, chain.providers = Settings(), [p]
    result = chain.complete_json("s", "u", schema={"type": "object"})
    assert result == {"ok": True}
    assert captured[-1]["config"].kwargs["response_mime_type"] == "application/json"


def test_gemini_rate_limit_retries_then_raises_llm_error(monkeypatch):
    captured = []
    _install_fake_genai(monkeypatch, captured, rate_limited=True)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    p = L.build_provider("gemini", Settings(llm_provider="gemini"))
    chain = object.__new__(L.LLM)
    chain.settings, chain.providers = Settings(), [p]
    with pytest.raises(L.LLMError, match="rate limited"):
        chain.complete_json("s", "u")
    assert sum(1 for c in captured if "model" in c) == 2  # primary + one bounded retry


def test_gemini_auth_error_is_not_retried(monkeypatch):
    captured = []
    _install_fake_genai(monkeypatch, captured, auth_error=True)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    p = L.build_provider("gemini", Settings(llm_provider="gemini"))
    # complete() raises PermanentError immediately (no retry) for auth failures.
    with pytest.raises(L.PermanentError, match="authentication or permission failure"):
        p.complete("sys", "user")
    assert sum(1 for c in captured if "model" in c) == 1  # no retry on auth failure
    # complete_json surfaces it as a clear LLMError rather than silently falling back.
    chain = object.__new__(L.LLM)
    chain.settings, chain.providers = Settings(), [p]
    with pytest.raises(L.LLMError, match="All LLM providers failed"):
        chain.complete_json("s", "u")


def test_gemini_404_model_not_found_is_permanent_and_not_retried(monkeypatch):
    """HTTP 404 (dead / deprecated model) is permanent: fail fast, never retry.

    Regression: a deprecated model such as gemini-2.5-flash returns HTTP 404. Retrying it
    burns the retry budget and delays the slot failure instead of surfacing "use a supported
    model." 404 must be classified permanent so ``with_retries`` does not re-attempt it.
    """
    captured = []
    _install_fake_genai(monkeypatch, captured, not_found=True)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    p = L.build_provider("gemini", Settings(llm_provider="gemini", llm_model="gemini-2.5-flash"))
    with pytest.raises(L.PermanentError, match="no longer available"):
        p.complete("sys", "user")
    # Exactly one attempt: a permanent error is never retried.
    assert sum(1 for c in captured if "model" in c) == 1
    # complete_json surfaces it as a clear failure instead of silently falling through retries.
    chain = object.__new__(L.LLM)
    chain.settings, chain.providers = Settings(), [p]
    with pytest.raises(L.LLMError, match="All LLM providers failed"):
        chain.complete_json("s", "u")


def test_gemini_never_logs_the_key(monkeypatch, caplog):
    captured = []
    _install_fake_genai(monkeypatch, captured)
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-gemini-key")
    p = L.build_provider("gemini", Settings(llm_provider="gemini"))
    with caplog.at_level("DEBUG"):
        p.complete("sys", "user")
    joined = caplog.text
    assert "super-secret-gemini-key" not in joined
    assert "GEMINI_API_KEY" not in joined


def test_gemini_available_in_auto_order(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert "gemini" in L.available_providers(Settings())
    assert L.resolve_provider_name(Settings(llm_provider="gemini")) == "gemini"


def test_build_prompt_requests_three_hooks_and_angle(monkeypatch):
    from purffle_shorts.script import build_prompt
    system, user = build_prompt("octopuses", "facts", Settings(), [], context="marine biology text")
    assert "hooks" in user and "selected_hook" in user and "angle" in user
    assert "THREE substantially different hooks" in user
    assert "ground claims" in user.lower() or "ground claims" in user


def test_script_carries_angle_hooks_sources():
    s = Script(
        topic="t", title="T", hook_text="H", scenes=[Scene("narration", "q", "p")],
        description="d", hashtags=["#t"], tags=["t"], category="education",
        angle="a specific angle", hooks=["hook one", "hook two", "hook three"],
        selected_hook="hook one", sources=["https://example.com/src"],
    )
    assert s.angle == "a specific angle"
    assert len(s.hooks) == 3
    assert s.selected_hook == "hook one"
    assert s.sources == ["https://example.com/src"]


def _walk_forbidden(schema):
    found = []
    if isinstance(schema, dict):
        for key in ("additionalProperties", "additional_properties"):
            if key in schema:
                found.append(key)
        for value in schema.values():
            found += _walk_forbidden(value)
    elif isinstance(schema, list):
        for value in schema:
            found += _walk_forbidden(value)
    return found


def test_gemini_sanitization_drops_additional_properties(monkeypatch):
    """Regression: the exact reported 400 INVALID_ARGUMENT on additional_properties,
    including nested inside array items, must not reach the Gemini request config."""
    captured = []
    _install_fake_genai(monkeypatch, captured)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    p = L.build_provider("gemini", L.Settings(llm_provider="gemini", llm_model="gemini-3.8-flash"))
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "narration": {"type": "string"},
                        "image_prompt": {"type": "string"},
                    },
                },
            },
        },
    }
    p.complete("sys", "user", schema=schema)
    cfg = captured[-1]["config"]
    assert cfg.kwargs["response_mime_type"] == "application/json"
    sanitized = cfg.kwargs["response_schema"]
    assert _walk_forbidden(sanitized) == []
    assert sanitized["type"] == "object"
    assert sanitized["properties"]["scenes"]["items"]["type"] == "object"
    assert "narration" in sanitized["properties"]["scenes"]["items"]["properties"]
    assert "image_prompt" in sanitized["properties"]["scenes"]["items"]["properties"]


def test_gemini_sanitization_nullable_union():
    """Nullable unions become type + nullable: true instead of a rejected type list."""
    assert L._sanitize_gemini_schema({"type": ["string", "null"]}) == {
        "type": "string",
        "nullable": True,
    }
    assert L._sanitize_gemini_schema({"type": ["integer", "null"]}) == {
        "type": "integer",
        "nullable": True,
    }
    # A union with no concrete type degrades safely, never to a list.
    assert L._sanitize_gemini_schema({"type": ["null"]}) == {
        "type": "string",
        "nullable": True,
    }


def test_gemini_sanitization_preserves_output_structure():
    """Dropping additionalProperties must not drop properties/required/items/enum."""
    schema = {
        "type": "object",
        "required": ["title"],
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "count": {"type": ["integer", "null"]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "speaker": {"type": "string", "enum": ["A", "B"]},
        },
    }
    clean = L._sanitize_gemini_schema(schema)
    assert "additionalProperties" not in _walk_forbidden(clean)
    assert clean["required"] == ["title"]
    assert clean["properties"]["title"] == {"type": "string"}
    assert clean["properties"]["count"] == {"type": "integer", "nullable": True}
    assert clean["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert clean["properties"]["speaker"] == {"type": "string", "enum": ["A", "B"]}


def test_gemini_request_configuration_sends_json_mime_and_sanitized_schema(monkeypatch):
    """The actual SDK GenerateContentConfig carries response_mime_type plus a
    sanitized response_schema (no additional_properties) for gemini-3.8-flash."""
    captured = []
    _install_fake_genai(monkeypatch, captured)
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key-123")
    p = L.build_provider("gemini", L.Settings(llm_provider="gemini", llm_model="gemini-3.8-flash"))
    p.complete("sys", "user", schema={"type": "object", "additionalProperties": False}, max_tokens=4000)
    cfg = captured[-1]["config"]
    assert cfg.kwargs["response_mime_type"] == "application/json"
    assert cfg.kwargs["response_schema"] == {"type": "object"}
    assert captured[0]["api_key"] == "gem-key-123"
    # The key never reaches the config payload sent to the SDK.
    assert "gem-key-123" not in str(cfg.kwargs)
    assert "gem-key-123" not in str(captured[-1]["config"].kwargs)
