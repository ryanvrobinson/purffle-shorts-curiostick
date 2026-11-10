"""Claim extraction and fact verification regression tests.

No real API credits are consumed: every LLM call is mocked.
"""

import pytest

from purffle_shorts.llm import LLMError, PermanentError
from purffle_shorts.research import (
    _CLAIM_EXTRACTION_SCHEMA,
    NEEDS_HUMAN_REVIEW,
    VERIFIED,
    _extract_claims_llm,
    verify_claims,
    verify_script_claims,
)
from purffle_shorts.topics import Topic


class FakeLLM:
    """Stand-in for LLM whose ``complete_json`` returns a parsed dict or raises."""

    def __init__(self, reply=None, error=None, call_limit=None):
        self.reply = reply
        self.error = error
        self.call_limit = call_limit
        self.calls = []

    def complete_json(self, system, user, schema=None, max_tokens=4000):
        self.calls.append((system, user, schema, max_tokens))
        if self.call_limit is not None and len(self.calls) > self.call_limit:
            raise LLMError("call limit exceeded")
        if self.error is not None:
            raise self.error
        return self.reply


# ------------------------------------------------------------------ extraction


def test_valid_object_extraction():
    """A well-formed {"claims": [...]} object is returned as a claim list."""
    llm = FakeLLM(reply={"claims": ["Octopuses have three hearts."]})
    claims = _extract_claims_llm(llm, "Octopuses have three hearts.", "octopuses")
    assert claims == ["Octopuses have three hearts."]
    assert llm.calls[0][2] is _CLAIM_EXTRACTION_SCHEMA
    assert llm.calls[0][3] == 2000


def test_empty_claims_array_is_valid():
    llm = FakeLLM(reply={"claims": []})
    assert _extract_claims_llm(llm, "Hello world.", "topic") == []


def test_missing_claims_field_rejected():
    llm = FakeLLM(reply={})
    with pytest.raises(LLMError, match="claims' field is missing"):
        _extract_claims_llm(llm, "Octopuses have three hearts.", "octopuses")


def test_claims_not_array_rejected():
    llm = FakeLLM(reply={"claims": "not a list"})
    with pytest.raises(LLMError, match="not an array"):
        _extract_claims_llm(llm, "Octopuses have three hearts.", "octopuses")


def test_invalid_claim_type_rejected():
    llm = FakeLLM(reply={"claims": ["ok", 5]})
    with pytest.raises(LLMError, match="is not a string"):
        _extract_claims_llm(llm, "Octopuses have three hearts.", "octopuses")


def test_non_object_result_rejected():
    """complete_json must return an object; a bare array is rejected."""
    llm = FakeLLM(reply=["Octopuses have three hearts."])
    with pytest.raises(LLMError, match="non-object result"):
        _extract_claims_llm(llm, "Octopuses have three hearts.", "octopuses")


def test_extraction_failure_propagates_no_silent_fallback():
    """A failed extraction raises; _extract_claims_llm never fabricates claims."""
    llm = FakeLLM(error=LLMError("empty reply (finish_reason=length)"))
    with pytest.raises(LLMError, match="empty reply"):
        _extract_claims_llm(llm, "Octopuses have three hearts.", "octopuses")
    assert len(llm.calls) == 1


# ------------------------------------------------------------------ verification


def test_heuristic_claims_with_no_sources_are_unverified():
    """Without supporting evidence a claim is needs_human_review, never verified."""
    result = verify_claims(["Octopuses have three hearts."], [])
    assert result[0].status == NEEDS_HUMAN_REVIEW
    assert result[0].source_urls == []


def test_claim_matched_to_source_is_verified():
    sources = [{"title": "Chicxulub crater", "url": "https://example.com/chicxulub"}]
    result = verify_claims(["The Chicxulub crater is in Mexico."], sources)
    assert result[0].status == VERIFIED
    assert result[0].source_urls == ["https://example.com/chicxulub"]


def test_unrelated_claim_is_unverified():
    sources = [{"title": "Chicxulub crater", "url": "https://example.com/chicxulub"}]
    result = verify_claims(["Blood Falls is red."], sources)
    assert result[0].status == NEEDS_HUMAN_REVIEW


def test_claim_relevance_is_word_overlap_based():
    sources = [{"title": "octopuses hearts anatomy", "url": "https://example.com/octopus"}]
    result = verify_claims(["Octopuses have three hearts."], sources)
    assert result[0].status == VERIFIED


# ------------------------------------------------------------------ script-level verification


def _topic(sources=None):
    return Topic(
        subject="Chicxulub crater",
        source="niche",
        source_references=sources or [],
    )


def test_genuine_unresolved_claims_block_rendering():
    """verify_script_claims returns needs_human_review when claims lack sources."""
    report = verify_script_claims([(0, "The crater is 66 million years old.")], _topic())
    assert report.status == NEEDS_HUMAN_REVIEW
    assert report.review_count == 1
    assert report.checked_by == "heuristic"
    assert report.llm_extraction_failed is False


def test_fully_verified_claims_allow_continue():
    sources = [{"title": "Chicxulub crater", "url": "https://example.com/chicxulub"}]
    report = verify_script_claims(
        [(0, "The Chicxulub crater is in Mexico.")], _topic(sources)
    )
    assert report.status == VERIFIED
    assert report.verified_count == 1
    assert report.review_count == 0


def test_llm_extraction_failure_is_distinct_from_verification_failure():
    """Extraction failure is flagged separately and blocks with llm_extraction_failed=True."""
    llm = FakeLLM(error=LLMError("rate limited"))
    report = verify_script_claims(
        [(0, "The crater is 66 million years old.")], _topic(), llm=llm
    )
    assert report.llm_extraction_failed is True
    # Heuristic fallback claims still need review; nothing is marked verified.
    assert report.status == NEEDS_HUMAN_REVIEW
    assert report.checked_by == "heuristic"


def test_extraction_failure_does_not_mark_claims_verified():
    """A failed extraction never upgrades a heuristic claim to verified.

    The original script claims are verified against source evidence only; a
    claim with no supporting source stays needs_human_review even though the
    extraction itself failed (flagged separately via llm_extraction_failed).
    """
    sources = [{"title": "Chicxulub crater", "url": "https://example.com/chicxulub"}]
    llm = FakeLLM(error=LLMError("rate limited"))
    report = verify_script_claims(
        [(0, "Blood Falls runs red from iron-rich brine.")], _topic(sources), llm=llm
    )
    assert report.llm_extraction_failed is True
    assert report.status == NEEDS_HUMAN_REVIEW
    assert report.claims[0].status == NEEDS_HUMAN_REVIEW
    assert report.claims[0].source_urls == []


# ------------------------------------------------------------------ rate limit / permanent error at the provider level


def _fake_session(responses):
    """Build a fake requests session returning responses in order."""
    calls = []

    class S:
        def post(self, url, json=None, headers=None, timeout=None):
            calls.append({"url": url, "json": json, "timeout": timeout})
            return responses[len(calls) - 1]

    return S(), calls


class FakeResp:
    def __init__(self, status=200, data=None, headers=None):
        self.status_code = status
        self._data = data or {}
        self.text = str(self._data)
        self.headers = headers or {}

    def json(self):
        return self._data


def test_groq_429_respects_retry_after(monkeypatch):
    """HTTP 429 with Retry-After: retry after the suggested wait, then succeed."""
    from purffle_shorts import llm as L

    session, calls = _fake_session([
        FakeResp(429, {"error": {"message": "rate limit"}},
                 headers={"Retry-After": "1"}),
        FakeResp(200, {"choices": [{"message": {"content": "{\"x\": 1}"},
                                   "finish_reason": "stop"}]}),
    ])
    monkeypatch.setattr(L, "http", lambda: session)
    p = L.OpenAICompatible(L.PRESETS["custom"], "openai/gpt-oss-20b", "k",
                           "https://api.groq.com/openai/v1")
    assert p.complete("system", "user") == '{"x": 1}'
    assert len(calls) == 2


def test_http_400_is_permanent_not_retried(monkeypatch):
    """HTTP 400 raises PermanentError immediately, not retried as 429."""
    from purffle_shorts import llm as L

    session, calls = _fake_session([
        FakeResp(400, {"error": {"message": "bad request"}}),
    ])
    monkeypatch.setattr(L, "http", lambda: session)
    p = L.OpenAICompatible(L.PRESETS["custom"], "openai/gpt-oss-20b", "k",
                           "https://api.groq.com/openai/v1")
    with pytest.raises(PermanentError, match="HTTP 400"):
        p.complete("system", "user")
    assert len(calls) == 1  # no retry on a permanent 400
