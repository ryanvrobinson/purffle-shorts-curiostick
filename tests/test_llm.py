import sys
import types

import pytest
import requests

from purffle_shorts import llm as L
from purffle_shorts.config import Settings
from purffle_shorts.utils import PermanentError


class FakeResp:
    def __init__(self, status=200, data=None):
        self.status_code, self._data = status, data or {}
        self.text = str(self._data)

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return self.resp


def _ok(content='{"x": 1}'):
    return FakeResp(200, {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})


def test_auto_picks_first_configured(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("GEMINI_API_KEY", "m")
    assert L.resolve_provider_name(Settings()) == "gemini"


def test_no_provider_gives_helpful_error(monkeypatch):
    monkeypatch.setattr(L, "_ollama_up", lambda url: False)
    with pytest.raises(L.LLMError, match="demo"):
        L.resolve_provider_name(Settings())


def test_no_provider_raises_not_configured(monkeypatch):
    """Nothing configured is a graceful-fallback case, not a hard config error."""
    monkeypatch.setattr(L, "_ollama_up", lambda url: False)
    with pytest.raises(L.LLMNotConfigured):
        L.resolve_provider_name(Settings())
    assert issubclass(L.LLMNotConfigured, L.LLMConfigError)


def test_unknown_provider_still_hard_fails():
    with pytest.raises(L.LLMConfigError) as exc:
        L.resolve_provider_name(Settings(llm_provider="not-a-provider"))
    assert not isinstance(exc.value, L.LLMNotConfigured)


def test_openai_payload_for_chat_model(monkeypatch):
    sess = FakeSession(_ok())
    monkeypatch.setattr(L, "http", lambda: sess)
    p = L.OpenAICompatible(L.PRESETS["openai"], "gpt-4o-mini", "k", L.PRESETS["openai"].base_url)
    assert p.complete("sys", "user", schema={}) == '{"x": 1}'
    body = sess.calls[0]["json"]
    assert body["temperature"] == 0.9 and body["max_tokens"] == 4000
    assert body["response_format"] == {"type": "json_object"}
    assert sess.calls[0]["timeout"] == (10, 120)
    assert sess.calls[0]["headers"]["Authorization"] == "Bearer k"


def test_openai_uses_final_content_not_reasoning_content(monkeypatch):
    session = FakeSession(FakeResp(200, {
        "choices": [{"message": {
            "content": '{"answer": "final"}',
            "reasoning_content": "private reasoning must not be returned",
        }, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }))
    monkeypatch.setattr(L, "http", lambda: session)
    provider = L.OpenAICompatible(L.PRESETS["custom"], "reasoning-model", "fake-key", "https://x/v1")

    assert provider.complete("s", "u") == '{"answer": "final"}'


def test_openai_does_not_use_reasoning_when_final_content_is_empty(monkeypatch):
    session = FakeSession(FakeResp(200, {
        "choices": [{"message": {
            "content": None,
            "reasoning_content": "private reasoning must not be returned",
        }, "finish_reason": "length"}],
    }))
    monkeypatch.setattr(L, "http", lambda: session)
    provider = L.OpenAICompatible(L.PRESETS["custom"], "reasoning-model", "fake-key", "https://x/v1")

    with pytest.raises(L.LLMError, match="empty reply.*reasoning_content is not final output"):
        provider.complete("s", "u")


def test_openai_timeout_is_reported_and_not_retried(monkeypatch):
    class TimeoutSession:
        def __init__(self):
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            assert kwargs["timeout"] == (10, 120)
            raise requests.exceptions.ReadTimeout("socket stalled")

    session = TimeoutSession()
    monkeypatch.setattr(L, "http", lambda: session)
    provider = L.OpenAICompatible(L.PRESETS["custom"], "model", "fake-key", "https://x/v1")
    chain = object.__new__(L.LLM)
    chain.settings, chain.providers = Settings(), [provider]

    with pytest.raises(L.LLMError, match="request timed out.*connect/read limits 10s/120s"):
        chain.complete_json("s", "u")
    assert session.calls == 1


def test_malformed_json_error_does_not_echo_model_output():
    with pytest.raises(L.LLMError, match="not a complete JSON object") as exc:
        L.extract_json('{"sensitive": "generated text",')
    assert "generated text" not in str(exc.value)


def test_reasoning_models_skip_temperature(monkeypatch):
    sess = FakeSession(_ok())
    monkeypatch.setattr(L, "http", lambda: sess)
    L.OpenAICompatible(L.PRESETS["openai"], "gpt-5-mini", "k", "https://x/v1").complete("s", "u", schema={})
    body = sess.calls[0]["json"]
    assert "temperature" not in body and body["max_completion_tokens"] >= 8000


def test_json_mode_is_dropped_when_unsupported(monkeypatch):
    calls = []

    class S:
        def post(self, url, json=None, headers=None, timeout=None):
            calls.append(dict(json))
            if "response_format" in json:
                return FakeResp(400, {"error": "response_format not supported"})
            return _ok()
    monkeypatch.setattr(L, "http", lambda: S())
    p = L.OpenAICompatible(L.PRESETS["groq"], "m", "k", "https://x/v1")
    assert p.complete("s", "u", schema={}) == '{"x": 1}'
    assert "response_format" in calls[0] and "response_format" not in calls[1]


def test_fallback_chain_uses_next_provider():
    class Bad(L.Provider):
        name, model = "bad", "m"

        def complete(self, *a, **k):
            raise PermanentError("401 bad key")

    class Good(L.Provider):
        name, model = "good", "m"

        def complete(self, *a, **k):
            return '```json\n{"ok": true}\n```'

    chain = object.__new__(L.LLM)
    chain.settings, chain.providers = Settings(), [Bad(), Good()]
    assert chain.complete_json("s", "u") == {"ok": True}





def test_invalid_json_gets_one_higher_token_retry():
    class Truncated(L.Provider):
        name, model = "truncated", "m"

        def __init__(self):
            self.limits = []

        def complete(self, system, user, *, schema=None, temperature=0.9, max_tokens=4000):
            self.limits.append(max_tokens)
            return "{" if len(self.limits) == 1 else '{"ok": true}'

    provider = Truncated()
    chain = object.__new__(L.LLM)
    chain.settings, chain.providers = Settings(), [provider]

    assert chain.complete_json("s", "u") == {"ok": True}
    assert provider.limits == [4000, 8000]


def test_empty_length_response_gets_bounded_higher_token_retry():
    class EmptyOnce(L.Provider):
        name, model = "empty", "m"

        def __init__(self):
            self.limits = []

        def complete(self, system, user, *, schema=None, temperature=0.9, max_tokens=4000):
            self.limits.append(max_tokens)
            if len(self.limits) == 1:
                raise L.LLMError("empty reply (finish_reason=length)")
            return '{"ok": true}'

    provider = EmptyOnce()
    chain = object.__new__(L.LLM)
    chain.settings, chain.providers = Settings(), [provider]

    assert chain.complete_json("s", "u") == {"ok": True}
    assert provider.limits == [4000, 8000]


def test_valid_json_is_preserved_when_provider_reports_length(monkeypatch):
    class LengthResponse:
        def post(self, *args, **kwargs):
            return FakeResp(200, {"choices": [{"message": {"content": '{"ok": true}'},
                                               "finish_reason": "length"}]})

    monkeypatch.setattr(L, "http", lambda: LengthResponse())
    provider = L.OpenAICompatible(L.PRESETS["openai"], "gpt-4o-mini", "fake-key",
                                  L.PRESETS["openai"].base_url)
    chain = object.__new__(L.LLM)
    chain.settings, chain.providers = Settings(), [provider]
    assert chain.complete_json("s", "u") == {"ok": True}


def _fake_anthropic(monkeypatch, captured):
    mod = types.ModuleType("anthropic")

    class Err(Exception):
        pass
    for name in ("BadRequestError", "AuthenticationError", "PermissionDeniedError", "NotFoundError",
                 "RateLimitError", "APIStatusError", "APIConnectionError"):
        setattr(mod, name, type(name, (Err,), {}))

    reply = types.SimpleNamespace(stop_reason="end_turn",
                                  content=[types.SimpleNamespace(type="text", text='{"a": 1}')])

    class Messages:
        def __init__(self, kind):
            self.kind = kind

        def create(self, **kw):
            captured.append((self.kind, kw))
            return reply

    class Client:
        def __init__(self, api_key=None):
            self.messages = Messages("messages")
            self.beta = types.SimpleNamespace(messages=Messages("beta"))
    mod.Anthropic = Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)


def test_anthropic_opus_uses_structured_output_and_fallbacks(monkeypatch):
    captured = []
    _fake_anthropic(monkeypatch, captured)
    p = L.AnthropicProvider("claude-opus-5", "key")
    assert p.complete("sys", "user", schema={"type": "object"}) == '{"a": 1}'
    kind, kw = captured[0]
    assert kind == "beta" and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in kw and kw["system"] == "sys"


def test_anthropic_other_models_use_plain_messages(monkeypatch):
    captured = []
    _fake_anthropic(monkeypatch, captured)
    L.AnthropicProvider("claude-haiku-4-5", "key").complete("s", "u", schema={})
    kind, kw = captured[0]
    assert kind == "messages" and "fallbacks" not in kw and "output_config" not in kw


@pytest.mark.parametrize("host,url", [
    ("", "http://localhost:11434/v1"),
    ("127.0.0.1:11434", "http://127.0.0.1:11434/v1"),
    ("http://gpu-box:11434/", "http://gpu-box:11434/v1"),
])
def test_ollama_url_accepts_bare_host(monkeypatch, host, url):
    monkeypatch.setenv("OLLAMA_HOST", host)
    assert L.ollama_url() == url


def test_ollama_uses_an_installed_model_when_default_missing(monkeypatch):
    monkeypatch.setattr(L, "ollama_models", lambda url: ["granite4.1:3b", "qwen3:8b"])
    assert L.build_provider("ollama", Settings()).model == "granite4.1:3b"
    assert L.build_provider("ollama", Settings(), "qwen3:8b").model == "qwen3:8b"
    monkeypatch.setattr(L, "ollama_models", lambda url: ["llama3.1:latest"])
    assert L.build_provider("ollama", Settings()).model == "llama3.1"
    assert L.ollama_has(["llama3.1:latest"], "llama3.1") and not L.ollama_has(["llama3.1:8b"], "llama3.1")


def test_nvidia_gpt_oss_uses_low_reasoning_effort(monkeypatch):
    session = FakeSession(_ok())
    monkeypatch.setattr(L, "http", lambda: session)
    provider = L.OpenAICompatible(
        L.PRESETS["custom"], "openai/gpt-oss-20b", "fake-key",
        "https://integrate.api.nvidia.com/v1",
    )
    provider.complete("system", "user", schema={})
    body = session.calls[0]["json"]
    assert body["reasoning_effort"] == "low"


def test_nvidia_gpt_oss_retry_is_capped_at_4096_tokens():
    class Truncated(L.Provider):
        name = "custom"
        model = "openai/gpt-oss-20b"
        base_url = "https://integrate.api.nvidia.com/v1"

        def __init__(self):
            self.limits = []

        def complete(self, system, user, *, schema=None, temperature=0.9, max_tokens=4000):
            self.limits.append(max_tokens)
            return "{" if len(self.limits) == 1 else '{"ok": true}'

    provider = Truncated()
    chain = object.__new__(L.LLM)
    chain.settings, chain.providers = Settings(), [provider]
    assert chain.complete_json("system", "user") == {"ok": True}
    assert provider.limits == [4000, 4096]


def test_nemotron_reasoning_model_payload_disables_thinking_text_only(monkeypatch):
    """NVIDIA nemotron on NVIDIA NIM: thinking off, text-only, no image inputs, adequate budget."""
    session = FakeSession(_ok())
    monkeypatch.setattr(L, "http", lambda: session)
    provider = L.OpenAICompatible(
        L.PRESETS["custom"],
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "fake-key",
        "https://integrate.api.nvidia.com/v1",
    )
    provider.complete("system", "user", schema={})
    body = session.calls[0]["json"]
    assert body["model"] == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] >= 4096
    assert "reasoning_effort" not in body   # that's the gpt-oss knob, not nemotron's
    # Nemotron is handled like a reasoning model: temperature is not sent (thinking is off instead).
    assert "temperature" not in body
    # text-only: every message content is a plain string, no image_url parts
    assert all(isinstance(m["content"], str) for m in body["messages"])


def test_nemotron_empty_reply_with_reasoning_content_errors_without_printing_it(monkeypatch):
    """Nemotron burns its budget on reasoning -> empty content. Error must not echo reasoning."""
    session = FakeSession(FakeResp(200, {
        "choices": [{"message": {
            "content": None,
            "reasoning_content": "hidden chain of thought that must not leak",
        }, "finish_reason": "length"}],
    }))
    monkeypatch.setattr(L, "http", lambda: session)
    provider = L.OpenAICompatible(
        L.PRESETS["custom"],
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "fake-key",
        "https://integrate.api.nvidia.com/v1",
    )
    with pytest.raises(L.LLMError, match="empty reply.*reasoning_content is not final output") as exc:
        provider.complete("system", "user")
    assert "hidden chain of thought" not in str(exc.value)


def test_complete_json_retries_without_forced_json_mode(monkeypatch):
    """On an empty/invalid first reply the retry drops response_format and raises the token budget."""
    calls = []

    class S:
        def post(self, url, json=None, headers=None, timeout=None):
            calls.append(dict(json))
            if len(calls) == 1:
                return FakeResp(200, {"choices": [{"message": {"content": None},
                                                    "finish_reason": "length"}]})
            return FakeResp(200, {"choices": [{"message": {"content": '{"ok": true}'},
                                                "finish_reason": "stop"}]})

    monkeypatch.setattr(L, "http", lambda: S())
    p = L.OpenAICompatible(L.PRESETS["custom"], "some-model", "k", "https://example.com/v1")
    chain = object.__new__(L.LLM)
    chain.settings, chain.providers = Settings(), [p]
    assert chain.complete_json("s", "u") == {"ok": True}
    assert len(calls) == 2, calls
    assert "response_format" in calls[0], calls[0]          # first attempt used JSON mode
    assert "response_format" not in calls[1], calls[1]      # retry dropped forced JSON mode
    assert calls[1]["max_tokens"] == 8000, calls[1]         # larger budget on retry


def test_nemotron_empty_reply_retries_without_json_mode_and_more_tokens(monkeypatch):
    """Nemotron retry is a genuinely different request, not the same failed call with more tokens."""
    calls = []

    class S:
        def post(self, url, json=None, headers=None, timeout=None):
            calls.append(dict(json))
            if len(calls) == 1:
                return FakeResp(200, {"choices": [{"message": {"content": None},
                                                    "finish_reason": "length"}]})
            return FakeResp(200, {"choices": [{"message": {"content": '{"ok": true}'},
                                                "finish_reason": "stop"}]})

    monkeypatch.setattr(L, "http", lambda: S())
    p = L.OpenAICompatible(
        L.PRESETS["custom"],
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "fake-key",
        "https://integrate.api.nvidia.com/v1",
    )
    chain = object.__new__(L.LLM)
    chain.settings, chain.providers = Settings(), [p]
    assert chain.complete_json("s", "u") == {"ok": True}
    assert len(calls) == 2, calls
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]
    assert calls[1]["max_tokens"] > calls[0]["max_tokens"]
