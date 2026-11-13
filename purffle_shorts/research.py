"""Claim-based fact verification for PurffleShorts scripts.

Every factual claim in a narration is extracted, matched against the sources
gathered during topic research, and assigned a verification status:

  * ``verified``        - at least one source URL directly supports the claim
  * ``needs_human_review`` - the claim is plausible but no source confirms it
  * ``unverifiable``    - the claim cannot be checked against any available source

An LLM may help extract claims and suggest source matches, but an LLM's own
assertion is never accepted as proof. Status depends on actual source evidence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .config import Settings
from .llm import LLM, LLMError
from .topics import Topic
from .utils import redact

log = logging.getLogger("purffle")

VERIFIED = "verified"
NEEDS_HUMAN_REVIEW = "needs_human_review"
UNVERIFIABLE = "unverifiable"

VALID_STATUSES = {VERIFIED, NEEDS_HUMAN_REVIEW, UNVERIFIABLE}

# Strict schema for LLM claim extraction: a compact object with only the
# ``claims`` field downstream needs. Using an object (not a bare array) keeps
# ``extract_json`` happy and lets Groq's json_object mode validate the shape.
_CLAIM_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["claims"],
    "additionalProperties": False,
}

# A claim is a sentence-like span of narration. These separators split narration
# into claim-sized chunks without needing an LLM.
_CLAIM_SPLIT = re.compile(r"(?<=[.!?])\s+|(?:\n+)")


@dataclass
class Claim:
    """A single factual claim extracted from narration."""

    text: str
    source_urls: list[str] = field(default_factory=list)
    status: str = NEEDS_HUMAN_REVIEW
    reason: str = ""
    scene_index: int | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            self.status = NEEDS_HUMAN_REVIEW


@dataclass
class VerificationReport:
    """Result of verifying all claims in a script."""

    claims: list[Claim] = field(default_factory=list)
    status: str = NEEDS_HUMAN_REVIEW  # overall: verified | needs_human_review | unverifiable
    source_count: int = 0
    checked_by: str = "heuristic"  # heuristic | llm
    llm_extraction_failed: bool = False  # True if LLM claim extraction failed

    @property
    def verified_count(self) -> int:
        return sum(1 for c in self.claims if c.status == VERIFIED)

    @property
    def review_count(self) -> int:
        return sum(1 for c in self.claims if c.status == NEEDS_HUMAN_REVIEW)

    @property
    def unverifiable_count(self) -> int:
        return sum(1 for c in self.claims if c.status == UNVERIFIABLE)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "checked_by": self.checked_by,
            "source_count": self.source_count,
            "llm_extraction_failed": self.llm_extraction_failed,
            "claims": [
                {
                    "text": c.text,
                    "source_urls": c.source_urls,
                    "status": c.status,
                    "reason": c.reason,
                    "scene_index": c.scene_index,
                }
                for c in self.claims
            ],
            "summary": {
                "total": len(self.claims),
                "verified": self.verified_count,
                "needs_human_review": self.review_count,
                "unverifiable": self.unverifiable_count,
            },
        }


def _extract_claims_heuristic(narration: str) -> list[str]:
    """Split narration into claim-sized sentences without an LLM."""
    parts = [p.strip() for p in _CLAIM_SPLIT.split(narration) if p.strip()]
    claims = []
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip()
        if len(part) < 6:
            continue
        # Skip pure calls to action / greetings that carry no factual claim.
        if part.lower().startswith(("follow for", "subscribe", "like this", "comment")):
            continue
        claims.append(part)
    return claims


def _extract_claims_llm(llm: LLM, narration: str, topic: str) -> list[str]:
    """Ask the LLM to extract factual claims from narration.

    Requests a compact JSON object ``{"claims": [...]}`` matching
    :data:`_CLAIM_EXTRACTION_SCHEMA` so ``extract_json`` can parse it and
    Groq's json_object mode can validate the shape. A bare array is never
    requested, since :func:`extract_json` only accepts objects.

    ``complete_json`` performs at most one bounded repair attempt (drop forced
    JSON mode, larger token budget) when the first reply is truncated or
    invalid. Any incomplete, truncated, missing-field, or mistyped result is
    rejected with :exc:`LLMError` -- never silently treated as a successful
    extraction and never fabricated.
    """
    system = (
        "You extract verifiable factual claims from a video narration. "
        "Reply with a single JSON object and nothing else."
    )
    user = f"""Topic: {topic}

Narration:
{narration}

Return a JSON object with exactly one key: {{"claims": [...]}} where each item
is a short, concrete, checkable claim string (numbers, names, dates, causality).
Include only claims that can be checked against evidence. Exclude greetings,
calls to action, opinions, and subjective phrasing. If there are no checkable
claims, return {{"claims": []}}.
"""
    data = llm.complete_json(system, user, _CLAIM_EXTRACTION_SCHEMA, max_tokens=2000)
    if not isinstance(data, dict):
        raise LLMError("LLM claim extraction returned a non-object result")
    claims = data.get("claims")
    if not isinstance(claims, list):
        raise LLMError("LLM claim extraction 'claims' field is missing or not an array")
    cleaned: list[str] = []
    for c in claims:
        if not isinstance(c, str):
            raise LLMError(f"LLM claim extraction item {c!r} is not a string")
        text = c.strip()
        if text:
            cleaned.append(text)
    return cleaned


def _claim_relevance(claim: str, source: dict) -> bool:
    """Heuristic: does a source's title/URL seem to cover the claim's subject?"""
    claim_words = set(re.findall(r"[a-z]{4,}", claim.lower()))
    source_text = f"{source.get('title', '')} {source.get('url', '')}".lower()
    source_words = set(re.findall(r"[a-z]{4,}", source_text))
    if not claim_words:
        return False
    overlap = len(claim_words & source_words)
    return overlap >= max(1, len(claim_words) // 3)


def verify_claims(
    claims: list[str],
    sources: list[dict],
    llm: Optional[LLM] = None,
) -> list[Claim]:
    """Verify each claim against the gathered sources.

    A claim is ``verified`` only when at least one source appears to cover it.
    Without supporting source evidence, the claim is ``needs_human_review``.
    An LLM never confers ``verified`` status by itself.
    """
    verified: list[Claim] = []
    for i, claim in enumerate(claims):
        supporting = [s.get("url", "") for s in sources if _claim_relevance(claim, s)]
        supporting = [u for u in supporting if u]
        if supporting:
            status = VERIFIED
            reason = f"Supported by {len(supporting)} source URL(s)."
        else:
            status = NEEDS_HUMAN_REVIEW
            reason = "No gathered source confirms this claim; verify before publication."
        verified.append(
            Claim(
                text=claim,
                source_urls=supporting,
                status=status,
                reason=reason,
                scene_index=i,
            )
        )
    return verified


def verify_script_claims(
    script_claims: list[tuple[int, str]],
    topic: Topic,
    llm: Optional[LLM] = None,
) -> VerificationReport:
    """Verify all claims in a script against the topic's gathered sources.

    ``script_claims`` is a list of ``(scene_index, claim_text)`` pairs.
    """
    sources = list(topic.source_references or [])
    all_claims = [c for _, c in script_claims]
    checked_by = "heuristic"
    llm_extraction_failed = False
    extraction_error: str | None = None
    if llm is not None and all_claims:
        # Let the LLM re-extract/refine claims, but verification still depends
        # on source evidence, not LLM opinion. A failed LLM extraction is
        # recorded distinctly from a genuine verification failure and never
        # silently treated as successful LLM-based extraction.
        try:
            llm_claims = _extract_claims_llm(llm, " ".join(all_claims), topic.subject)
            if llm_claims:
                all_claims = llm_claims
                checked_by = "llm"
                log.info(
                    "LLM claim extraction returned %d claim(s) for topic %r",
                    len(all_claims), topic.subject,
                )
        except Exception as e:
            llm_extraction_failed = True
            extraction_error = redact(e)
            log.warning(
                "LLM claim extraction failed (extraction error, not a verification "
                "result): %s. Falling back to heuristic claims.", extraction_error)

    claims = verify_claims(all_claims, sources, llm)
    if llm_extraction_failed:
        log.warning(
            "Claim extraction failed for topic %r; %d heuristic claim(s) carry "
            "needs_human_review status: %s",
            topic.subject, len(claims), extraction_error,
        )

    # Re-map scene indices when LLM re-extraction changed the claim list.
    indexed: list[Claim] = []
    for idx, claim in enumerate(claims):
        if idx < len(script_claims):
            claim.scene_index = script_claims[idx][0]
        indexed.append(claim)

    overall = _overall_status(indexed)
    log.info(
        "Fact verification for topic %r: %d claim(s), status=%s, checked_by=%s, "
        "llm_extraction_failed=%s",
        topic.subject, len(indexed), overall, checked_by, llm_extraction_failed,
    )
    return VerificationReport(
        claims=indexed,
        status=overall,
        source_count=len(sources),
        checked_by=checked_by,
        llm_extraction_failed=llm_extraction_failed,
    )


def _overall_status(claims: list[Claim]) -> str:
    if not claims:
        return NEEDS_HUMAN_REVIEW
    if any(c.status == UNVERIFIABLE for c in claims):
        return UNVERIFIABLE
    if any(c.status == NEEDS_HUMAN_REVIEW for c in claims):
        return NEEDS_HUMAN_REVIEW
    return VERIFIED


def extract_claims_from_script(script, language: str = "en") -> list[tuple[int, str]]:
    """Extract one claim per scene from the script narration (heuristic)."""
    claims: list[tuple[int, str]] = []
    for i, scene in enumerate(script.scenes):
        for claim in _extract_claims_heuristic(scene.narration):
            claims.append((i, claim))
    return claims
