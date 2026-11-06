"""Pluggable LLMs: any OpenAI-compatible endpoint, native Anthropic Claude and native Google Gemini,
with a fallback chain.

Pick one with LLM_PROVIDER (or --provider) and LLM_MODEL (or --model). With LLM_PROVIDER=auto the
first provider whose API key is present is used. LLM_FALLBACKS=anthropic,gemini adds backups that
are tried in order when the primary fails.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from .config import Settings
from .utils import PermanentError, http, raise_for_status, redact, with_retries

log = logging.getLogger("purffle")
_LLM_REQUEST_TIMEOUT = (10, 120)


class LLMError(RuntimeError):
    pass


class LLMConfigError(LLMError):
    """Missing key, unknown provider, etc. — stop instead of retrying every video."""


class LLMNotConfigured(LLMConfigError):
    """No LLM provider has credentials (auto mode found nothing, Ollama is not running).

    Unlike other configuration errors this is not the user's mistake to fix before continuing: the pipeline
    can still write a simple script offline, so callers may catch it and fall back instead of failing."""


@dataclass(frozen=True)
class Preset:
    name: str
    label: str
    base_url: str
    key_envs: tuple[str, ...]
    default_model: str
    kind: str = "openai"          # openai (chat/completions compatible) | anthropic
    json_mode: bool = True        # supports response_format={"type":"json_object"}
    local: bool = False


PRESETS: dict[str, Preset] = {p.name: p for p in [
    Preset("openai", "OpenAI (GPT)", "https://api.openai.com/v1", ("OPENAI_API_KEY",), "gpt-4o-mini"),
    Preset("anthropic", "Anthropic (Claude)", "", ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
           "claude-opus-5", kind="anthropic"),
    Preset("gemini", "Google Gemini", "", ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
           "gemini-3.8-flash", kind="gemini"),
    Preset("groq", "Groq (Llama, fast)", "https://api.groq.com/openai/v1", ("GROQ_API_KEY",),
           "llama-3.3-70b-versatile"),
    Preset("openrouter", "OpenRouter (hundreds of models)", "https://openrouter.ai/api/v1",
           ("OPENROUTER_API_KEY",), "openai/gpt-4o-mini"),
    Preset("deepseek", "DeepSeek", "https://api.deepseek.com/v1", ("DEEPSEEK_API_KEY",), "deepseek-chat"),
    Preset("mistral", "Mistral", "https://api.mistral.ai/v1", ("MISTRAL_API_KEY",), "mistral-small-latest"),
    Preset("together", "Together AI", "https://api.together.xyz/v1", ("TOGETHER_API_KEY",),
           "meta-llama/Llama-3.3-70B-Instruct-Turbo", json_mode=False),
    Preset("xai", "xAI (Grok)", "https://api.x.ai/v1", ("XAI_API_KEY",), "grok-3-mini"),
    Preset("ollama", "Ollama (local, free)", "http://localhost:11434/v1", (), "llama3.1", local=True),
    Preset("lmstudio", "LM Studio (local, free)", "http://localhost:1234/v1", (), "local-model",
           json_mode=False, local=True),
    Preset("custom", "Any OpenAI-compatible URL (LLM_BASE_URL)", "", ("LLM_API_KEY",), ""),
]}

AUTO_ORDER = ["openai", "anthropic", "gemini", "groq", "openrouter", "deepseek", "mistral", "together", "xai"]

# Claude models whose requests get server-side refusal fallbacks by default.
_CLAUDE_FALLBACK_MODELS = re.compile(r"^claude-(opus-5|fable-5)")


def _key_for(preset: Preset, settings: Settings) -> str:
    if preset.name == "custom":
        return settings.llm_api_key
    if preset.name == "gemini":
        # GEMINI_API_KEY (GOOGLE_API_KEY as fallback) from the environment first, then the Settings field.
        for env in preset.key_envs:
            value = os.getenv(env)
            if value:
                return value
        return settings.gemini_api_key
    for env in preset.key_envs:
        if os.getenv(env):
            return os.environ[env]
    return settings.llm_api_key if preset.name == settings.llm_provider else ""


def _ollama_up(base_url: str) -> bool:
    try:
        root = base_url.rsplit("/v1", 1)[0]
        return http().get(root + "/api/tags", timeout=1.5).ok
    except Exception:
        return False


def ollama_url() -> str:
    host = os.getenv("OLLAMA_HOST", "").rstrip("/") or PRESETS["ollama"].base_url
    if not host.startswith("http"):
        host = "http://" + host
    return host if host.endswith("/v1") else host + "/v1"


def ollama_models(base_url: str) -> list[str] | None:
    """Models pulled into Ollama, or None when Ollama is not reachable."""
    try:
        r = http().get(base_url.rsplit("/v1", 1)[0] + "/api/tags", timeout=1.5)
        return [m.get("name", "") for m in r.json().get("models", [])] if r.ok else None
    except Exception:
        return None


def ollama_has(installed: list[str], model: str) -> bool:
    return model in installed or (":" not in model and f"{model}:latest" in installed)


def available_providers(settings: Settings) -> list[str]:
    """Providers that have credentials configured (in auto order)."""
    return [n for n in AUTO_ORDER if _key_for(PRESETS[n], settings)]


def resolve_provider_name(settings: Settings) -> str:
    name = settings.llm_provider or "auto"
    if name != "auto":
        if name not in PRESETS:
            raise LLMConfigError(f"Unknown LLM_PROVIDER '{name}'. Choose one of: {', '.join(PRESETS)}")
        return name
    if settings.llm_base_url:
        return "custom"
    found = available_providers(settings)
    if found:
        return found[0]
    if _ollama_up(ollama_url()):
        return "ollama"
    raise LLMNotConfigured(
        "No LLM is configured. Set one API key in .env (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, "
        "GROQ_API_KEY, OPENROUTER_API_KEY, DEEPSEEK_API_KEY, MISTRAL_API_KEY, ...), run Ollama locally, "
        "or try the no-key demo:  python -m purffle_shorts demo"
    )


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model reply that may include code fences or chatter.

    Handles:
    - Code fences (```json ... ```)
    - Trailing commas in objects/arrays
    - Multiple JSON objects (takes the first complete one)
    - JSON with comments (strips // and /* */ comments)
    - Unescaped newlines in strings
    """
    t = text.strip()
    # Remove code fences
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.S).strip()

    # Try direct parse first
    try:
        val = json.loads(t)
        if isinstance(val, dict):
            return val
    except json.JSONDecodeError:
        pass

    # Try to find and extract a JSON object
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        chunk = t[start:end + 1]
        # Try parsing with various fixes
        for fix_func in [
            lambda x: x,  # No fix
            lambda x: re.sub(r",\s*([}\]])", r"\1", x),  # Fix trailing commas
            lambda x: re.sub(r"//.*", "", x),  # Remove // comments
            lambda x: re.sub(r"/\*.*?\*/", "", x, flags=re.DOTALL),  # Remove /* */ comments
            lambda x: re.sub(r"(?<!\\)\n", "\\n", x),  # Escape unescaped newlines in strings
            lambda x: re.sub(r"(?<!\\)\t", "\\t", x),  # Escape unescaped tabs
        ]:
            try:
                fixed = fix_func(chunk)
                val = json.loads(fixed)
                if isinstance(val, dict):
                    return val
            except json.JSONDecodeError:
                continue

    # As a last resort, try to find any valid JSON object in the text
    # Look for balanced braces
    brace_count = 0
    potential_starts = []
    for i, ch in enumerate(t):
        if ch == '{':
            if brace_count == 0:
                potential_starts.append(i)
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0 and potential_starts:
                for start_idx in potential_starts:
                    candidate = t[start_idx:i+1]
                    try:
                        val = json.loads(candidate)
                        if isinstance(val, dict):
                            return val
                    except json.JSONDecodeError:
                        # Try with fixes
                        for fix_func in [
                            lambda x: re.sub(r",\s*([}\]])", r"\1", x),
                            lambda x: re.sub(r"//.*", "", x),
                            lambda x: re.sub(r"/\*.*?\*/", "", x, flags=re.DOTALL),
                        ]:
                            try:
                                val = json.loads(fix_func(candidate))
                                if isinstance(val, dict):
                                    return val
                            except json.JSONDecodeError:
                                continue
                potential_starts = []

    raise LLMError("Model did not return valid JSON (response content was not a complete JSON object)")


class Provider:
    name: str = ""
    model: str = ""

    def complete(self, system: str, user: str, *, schema: dict | None = None,
                 temperature: float = 0.9, max_tokens: int = 4000) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name}:{self.model}"


class OpenAICompatible(Provider):
    def __init__(self, preset: Preset, model: str, api_key: str, base_url: str):
        self.name, self.preset, self.model = preset.name, preset, model
        self.api_key, self.base_url = api_key, base_url.rstrip("/")
        self._json_mode = preset.json_mode

    def _is_reasoning_model(self) -> bool:
        return bool(re.match(r"^(o\d|gpt-5)", self.model.split("/")[-1]))

    def _is_nvidia_reasoning_model(self) -> bool:
        """NVIDIA Nemotron reasoning models (e.g. nemotron-3.5-super-vl-120b-a12b).

        Reasoning is on by default and its tokens count against ``max_tokens``, so a small budget
        ends inside the chain of thought and leaves ``content`` empty. CurioStick script generation is
        text-only and does not need reasoning, so we switch thinking off and never print it.
        """
        return ("integrate.api.nvidia.com" in self.base_url.lower()
                and "nemotron" in self.model.lower())

    def complete(self, system, user, *, schema=None, temperature=0.9, max_tokens=4000) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self._is_reasoning_model():
            payload["max_completion_tokens"] = max(max_tokens, 8000)  # room for hidden reasoning tokens
        elif self._is_nvidia_reasoning_model():
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["max_tokens"] = max(max_tokens, 4096)
        else:
            payload["temperature"] = temperature
            payload["max_tokens"] = max_tokens
        if schema is not None and self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        if (self.name == "custom"
                and "integrate.api.nvidia.com" in self.base_url.lower()
                and self.model.lower().startswith("openai/gpt-oss-")):
            payload["reasoning_effort"] = "low"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        if self.name == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/Chamanrajragu/purffle-shorts"
            headers["X-Title"] = "PurffleShorts"
        try:
            started = time.perf_counter()
            r = http().post(f"{self.base_url}/chat/completions", json=payload, headers=headers,
                            timeout=_LLM_REQUEST_TIMEOUT)
            elapsed = time.perf_counter() - started
            log.debug("%s HTTP %d in %.2fs", self, r.status_code, elapsed)
            raise_for_status(r, f"{self}")
        except requests.Timeout as e:
            elapsed = time.perf_counter() - started
            raise LLMError(
                f"{self}: request timed out after {elapsed:.1f}s "
                f"(connect/read limits {_LLM_REQUEST_TIMEOUT[0]}s/{_LLM_REQUEST_TIMEOUT[1]}s)"
            ) from e
        except PermanentError as e:
            if "response_format" in payload and "response_format" in str(e):
                self._json_mode = False  # endpoint doesn't support JSON mode; rely on prompt + parser
                return self.complete(system, user, schema=schema, temperature=temperature, max_tokens=max_tokens)
            raise
        data = r.json()
        try:
            choice = data["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason")
            content = message.get("content")
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"{self}: unexpected response shape (HTTP {r.status_code})") from e
        if not isinstance(choice, dict) or not isinstance(message, dict):
            raise LLMError(f"{self}: unexpected response shape (HTTP {r.status_code})")
        if isinstance(content, list):
            content = "".join(
                part["text"] for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        if not isinstance(content, str) or not content.strip():
            reasoning_note = "; reasoning_content is not final output" if message.get("reasoning_content") else ""
            raise LLMError(f"{self}: empty reply (finish_reason={finish_reason}{reasoning_note})")
        log.debug("%s finish_reason=%s usage=%s content_length=%d",
                  self, finish_reason, data.get("usage"), len(content))
        return content


class AnthropicProvider(Provider):
    """Native Claude via the official Anthropic SDK (structured JSON output + refusal fallbacks)."""

    def __init__(self, model: str, api_key: str, effort: str = ""):
        try:
            import anthropic
        except ImportError as e:
            raise LLMConfigError("The Anthropic provider needs the SDK:  pip install anthropic") from e
        self.name, self.model, self.effort = "anthropic", model, effort
        self._anthropic = anthropic
        # With no explicit key the SDK resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login`.
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(self, system, user, *, schema=None, temperature=0.9, max_tokens=4000) -> str:
        a = self._anthropic
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,  # thinking is on by default on current Claude models; leave room
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        output_config: dict[str, Any] = {}
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if self.effort:
            output_config["effort"] = self.effort
        if output_config:
            kwargs["output_config"] = output_config
        # Sampling params (temperature) are rejected by current Claude models, so none are sent.
        try:
            if _CLAUDE_FALLBACK_MODELS.match(self.model):
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        except (a.BadRequestError, a.AuthenticationError, a.PermissionDeniedError, a.NotFoundError) as e:
            raise PermanentError(f"{self}: {e}") from e
        except (a.RateLimitError, a.APIStatusError, a.APIConnectionError) as e:
            raise LLMError(f"{self}: {e}") from e
        if resp.stop_reason == "refusal":
            raise PermanentError(f"{self}: request declined by the model")
        text = "".join(b.text for b in resp.content if b.type == "text")
        if not text.strip():
            raise LLMError(f"{self}: empty reply (stop_reason={resp.stop_reason})")
        return text


# JSON Schema keys the Gemini API rejects for response_schema. They are validation
# constraints, not structural fields, and the pipeline's normalize() already whitelists
# the output fields, so dropping them cannot change the expected output structure.
_GEMINI_UNSUPPORTED_SCHEMA_KEYS = {"additionalProperties", "additional_properties"}

def _sanitize_gemini_schema(value: Any) -> Any:
    """Convert JSON Schema constructs the Gemini API rejects for structured output.

    - additionalProperties / additional_properties are not accepted by the Gemini API
      for response_schema and are dropped. They are validation constraints only, so the
      expected output structure is preserved.
    - A JSON Schema nullable union such as type: ["string", "null"] becomes
      type: "string" plus nullable: true (Gemini-native). Optional properties are already
      optional unless listed in 'required', so the non-null type is retained.
    - Recurses into properties, items, any_of/one_of and defs so nested schemas are
      sanitized before the request reaches the API.
    """
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key in _GEMINI_UNSUPPORTED_SCHEMA_KEYS:
                continue
            if key == "type" and isinstance(item, list):
                non_null = [part for part in item if isinstance(part, str) and part.lower() != "null"]
                cleaned["type"] = non_null[0] if non_null else "string"
                if len(item) != len(non_null):
                    cleaned["nullable"] = True
            else:
                cleaned[key] = _sanitize_gemini_schema(item)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_gemini_schema(item) for item in value]
    return value


class GeminiProvider(Provider):
    """Native Google Gemini via the officially supported google-genai SDK.

    Reads GEMINI_API_KEY (GOOGLE_API_KEY as fallback) from the environment. The model is configured
    through GEMINI_MODEL / LLM_MODEL and defaults to gemini-3.8-flash, a currently supported model
    that supports native structured JSON output. Output tokens are capped for cost control, JSON mode
    uses the SDK's response_mime_type/response_schema (not an OpenAI-compatible shim), and API
    failures raise LLMError with a clear, key-free message. This provider never falls back to another
    provider on its own and never logs or echoes credentials.
    """

    _MAX_OUTPUT_TOKENS = 8000
    # google-genai's HttpOptions.timeout is in MILLISECONDS (the SDK docs state the unit
    # explicitly). A value of 60 therefore meant a 0.06s request timeout, which caused the
    # Slot 3 Gemini failures. 120_000 ms = 120 seconds, matching the intended bound.
    _REQUEST_TIMEOUT = 120_000

    def __init__(self, model: str, api_key: str):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise LLMConfigError(
                "Gemini provider needs the google-genai SDK:  pip install google-genai"
            ) from e
        if not api_key:
            raise LLMConfigError("Gemini: set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")
        self.name, self.model = "gemini", model
        self._genai = genai
        self._types = types
        # Construction is local: no API call happens here. The timeout bounds every request.
        self.client = genai.Client(api_key=api_key,
                                   http_options=types.HttpOptions(timeout=self._REQUEST_TIMEOUT))

    def _classify_error(self, exc: Exception) -> tuple[bool, str]:
        """Classify a Gemini SDK error into (permanent, key-free message)."""
        code = getattr(exc, "code", None)
        message = redact(str(exc))
        if code == 429 or "rate limit" in message.lower() or "quota" in message.lower():
            return False, f"rate limited or quota exhausted ({message})"
        if code in (401, 403):
            return True, f"authentication or permission failure ({message})"
        if code == 400:
            return True, f"request rejected ({message})"
        if isinstance(exc, self._genai.errors.ServerError):
            return False, f"Gemini API error ({message})"
        return False, f"request failed ({message})"

    def complete(self, system, user, *, schema=None, temperature=0.9, max_tokens=4000) -> str:
        kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": temperature,
            "max_output_tokens": min(max_tokens, self._MAX_OUTPUT_TOKENS),
        }
        if schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = _sanitize_gemini_schema(schema)
        config = self._types.GenerateContentConfig(**kwargs)

        def call() -> str:
            try:
                started = time.perf_counter()
                resp = self.client.models.generate_content(
                    model=self.model, contents=user, config=config,
                )
                elapsed = time.perf_counter() - started
            except PermanentError:
                raise
            except Exception as exc:
                permanent, message = self._classify_error(exc)
                raised = PermanentError(message) if permanent else LLMError(message)
                raise raised from exc
            text = getattr(resp, "text", None) or ""
            usage = getattr(resp, "usage_metadata", None)
            if usage:
                log.debug("%s finished in %.2fs (total tokens %s)",
                          self, elapsed, getattr(usage, "total_token_count", None))
            if not text.strip():
                raise LLMError(f"{self}: empty reply")
            return text

        # One bounded retry covers transient failures (5xx, rate limits, timeouts).
        # Auth failures and bad requests are PermanentError and are not retried.
        return with_retries(call, attempts=2, label=f"LLM {self}")


def build_provider(name: str, settings: Settings, model: str = "") -> Provider:
    preset = PRESETS[name]
    chosen = model
    model = model or preset.default_model
    key = _key_for(preset, settings)
    if preset.kind == "anthropic":
        return AnthropicProvider(model, key, settings.anthropic_effort)
    if preset.kind == "gemini":
        return GeminiProvider(model, key)
    base_url = preset.base_url
    if name == "custom":
        base_url = settings.llm_base_url
        if not base_url or not model:
            raise LLMConfigError("LLM_PROVIDER=custom needs LLM_BASE_URL and LLM_MODEL")
    elif name == "ollama":
        base_url = ollama_url()
        installed = None if chosen else ollama_models(base_url)
        if installed and not ollama_has(installed, model):
            # No model chosen and the default is not pulled: use one that is, instead of failing every video.
            log.info("Ollama has no '%s'; using installed model '%s' (set LLM_MODEL to choose)", model, installed[0])
            model = installed[0]
    if settings.llm_base_url and name == settings.llm_provider:
        base_url = settings.llm_base_url
    if not key and not preset.local and name != "custom":
        raise LLMConfigError(f"{preset.label}: set {' or '.join(preset.key_envs)} in .env")
    return OpenAICompatible(preset, model, key, base_url)


class LLM:
    """Primary provider + ordered fallbacks. ``complete_json`` returns a parsed dict."""

    def __init__(self, settings: Settings):
        self.settings = settings
        primary = resolve_provider_name(settings)
        names = [primary] + [n for n in settings.llm_fallbacks if n != primary]
        self.providers: list[Provider] = []
        for n in names:
            if n not in PRESETS:
                log.warning("Ignoring unknown LLM fallback '%s'", n)
                continue
            try:
                self.providers.append(build_provider(n, settings, settings.llm_model if n == primary else ""))
            except LLMError as e:
                if n == primary:
                    raise
                log.warning("Skipping fallback %s: %s", n, e)

    @property
    def label(self) -> str:
        return str(self.providers[0])

    def complete_json(self, system: str, user: str, schema: dict | None = None,
                      max_tokens: int = 4000) -> dict[str, Any]:
        errors = []
        for p in self.providers:
            nvidia_gpt_oss = (
                getattr(p, "name", "") == "custom"
                and "integrate.api.nvidia.com" in getattr(p, "base_url", "").lower()
                and getattr(p, "model", "").lower().startswith("openai/gpt-oss-")
            )
            if nvidia_gpt_oss:
                first_limit = min(max_tokens, 4096)
                limits = [first_limit]
                if first_limit < 4096:
                    limits.append(4096)
            else:
                limits = [max_tokens]
                if max_tokens < 16000:
                    limits.append(min(max_tokens * 2, 16000))
            for attempt, token_limit in enumerate(limits):
                # The retry is a genuinely different request: drop forced JSON mode (some endpoints
                # return an empty body when JSON mode is required) and rely on the extract_json
                # parser instead, on top of the larger token budget. Never the same call twice.
                attempt_schema = None if attempt > 0 else (schema if schema is not None else {})
                def _call(p=p, token_limit=token_limit, attempt_schema=attempt_schema):
                    try:
                        return p.complete(system, user, schema=attempt_schema,
                                          max_tokens=token_limit, temperature=self.settings.llm_temperature)
                    except LLMError as e:
                        if "empty reply" in str(e).lower():
                            raise PermanentError(str(e)) from e
                        raise
                try:
                    text = with_retries(_call, attempts=1, label=f"LLM {p}")
                    result = extract_json(text)
                    if p is not self.providers[0]:
                        log.info("Used fallback LLM %s", p)
                    return result
                except Exception as e:
                    message = str(e).lower()
                    is_incomplete = ("empty reply" in message or "model did not return valid json" in message)
                    if attempt == 0 and is_incomplete and len(limits) > 1:
                        log.warning("LLM %s returned an empty or invalid JSON response; retrying once without "
                                    "forced JSON mode and with %d output tokens", p, limits[1])
                        continue
                    errors.append(f"{p}: {e}")
                    log.warning("LLM %s failed: %s", p, e)
                    break
        raise LLMError("All LLM providers failed:\n  " + "\n  ".join(errors))
