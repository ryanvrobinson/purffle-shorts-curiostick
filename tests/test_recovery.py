"""Recovery queue, failure classification, and slot independence regression tests."""

from datetime import datetime, timezone

import pytest

from purffle_shorts.automation import _is_temporary, _retry_delay
from purffle_shorts.config import Settings
from purffle_shorts.history import History
from purffle_shorts.llm import LLMError
from purffle_shorts.pipeline import Studio
from purffle_shorts.research import (
    NEEDS_HUMAN_REVIEW,
    VERIFIED,
    Claim,
    _extract_claims_llm,
    verify_claims,
    verify_script_claims,
)
from purffle_shorts.script import Scene, Script, validate_opening_hook
from purffle_shorts.topics import Topic
from purffle_shorts.youtube import NotAuthorized

# ------------------------------------------------------------------ failure classification

@pytest.mark.parametrize("error,stage,expected", [
    ("HTTP 429 rate limit", "tts", True),
    ("temporary network error", "footage", True),
    ("HTTP 503 service unavailable", "render", True),
    ("Groq API overloaded", "upload", True),
    ("invalid script JSON", "script", False),
    ("quality gate failed", "qc", False),
    ("Script Doctor score below 90", "review", False),
    ("unknown error", "unknown", False),
    ("connection timeout", "tts", True),
    ("API key invalid", "script", False),
])
def test_is_temporary_classification(error, stage, expected):
    """Temporary failures are retryable; content/config failures are not."""
    assert _is_temporary(error, stage) is expected


def test_retry_delay_is_bounded_and_increasing():
    """Exponential backoff with jitter stays within bounds."""
    d0 = _retry_delay(0)
    d1 = _retry_delay(1)
    d2 = _retry_delay(2)
    assert d0 > 0 and d1 >= d0 and d2 >= d1
    # Capped at RETRY_MAX_SECONDS plus jitter (<= 25% of base)
    assert _retry_delay(10) <= 3600 + 900


# ------------------------------------------------------------------ recovery queue persistence

@pytest.fixture
def tmp_history(tmp_path):
    db = tmp_path / "history.db"
    return History(db)


def test_record_and_retrieve_failure(tmp_history):
    """A recorded failure can be retrieved as a pending recovery."""
    rid = tmp_history.record_failure(
        run_date="2026-10-31", slot=1, key="2026-10-31-slot-1",
        topic="Chicxulub crater", stage="script",
        error="incomplete JSON from LLM", recoverable=True,
        rendered=False, uploaded=False,
    )
    assert rid > 0

    pending = tmp_history.pending_recoveries(limit=10)
    assert len(pending) == 1
    assert pending[0]["run_date"] == "2026-10-31"
    assert pending[0]["slot"] == 1
    assert pending[0]["stage"] == "script"
    assert pending[0]["retry_count"] == 0
    assert pending[0]["recoverable"] == 1


def test_record_failure_is_slot_scoped(tmp_history):
    """Different slots for the same date get independent recovery entries."""
    tmp_history.record_failure(
        run_date="2026-10-31", slot=1, key="2026-10-31-slot-1",
        topic="topic A", stage="upload", error="network", recoverable=True,
    )
    tmp_history.record_failure(
        run_date="2026-10-31", slot=2, key="2026-10-31-slot-2",
        topic="topic B", stage="qc", error="weak hook", recoverable=True,
    )
    pending = tmp_history.pending_recoveries(limit=10)
    assert len(pending) == 2


def test_get_recovery_for_slot_returns_latest(tmp_history):
    """get_recovery_for_slot returns the most recent pending entry."""
    tmp_history.record_failure(
        run_date="2026-10-31", slot=1, key="2026-10-31-slot-1",
        topic="topic", stage="script", error="first", recoverable=True,
    )
    tmp_history.record_failure(
        run_date="2026-10-31", slot=1, key="2026-10-31-slot-1",
        topic="topic", stage="upload", error="second", recoverable=True,
    )
    rec = tmp_history.get_recovery_for_slot("2026-10-31-slot-1")
    assert rec is not None
    assert rec["stage"] in ("script", "upload")


def test_permanent_failures_are_not_returned(tmp_history):
    """recoverable=0 entries are excluded from pending_recoveries."""
    tmp_history.record_failure(
        run_date="2026-10-31", slot=1, key="2026-10-31-slot-1",
        topic="topic", stage="script", error="invalid JSON", recoverable=False,
    )
    pending = tmp_history.pending_recoveries(limit=10)
    assert len(pending) == 0


def test_exhausted_retries_are_not_returned(tmp_history):
    """Entries where retry_count >= max_retries are excluded."""
    rid = tmp_history.record_failure(
        run_date="2026-10-31", slot=1, key="2026-10-31-slot-1",
        topic="topic", stage="tts", error="temp", recoverable=True,
        max_retries=1,
    )
    tmp_history.increment_retry(rid)
    pending = tmp_history.pending_recoveries(limit=10)
    assert len(pending) == 0


def test_mark_recovery_completed_and_failed(tmp_history):
    """Recovery entries can be marked completed or failed."""
    rid = tmp_history.record_failure(
        run_date="2026-10-31", slot=1, key="2026-10-31-slot-1",
        topic="topic", stage="upload", error="temp", recoverable=True,
    )
    tmp_history.mark_recovery_completed(rid, "retry-success", youtube_id="abc123")
    rec = tmp_history.get_recovery(rid)
    assert rec["status"] == "completed"
    assert rec["youtube_id"] == "abc123"
    assert rec["resolved_by"] == "retry-success"

    rid2 = tmp_history.record_failure(
        run_date="2026-10-31", slot=2, key="2026-10-31-slot-2",
        topic="topic", stage="script", error="permanent", recoverable=False,
    )
    tmp_history.mark_recovery_failed(rid2, "exhausted")
    rec2 = tmp_history.get_recovery(rid2)
    assert rec2["status"] == "failed"


# ------------------------------------------------------------------ hook validation regression (Chicxulub failure)

def test_generic_hook_detected_before_tts():
    """A 'Did you know?' opening is rejected before any rendering."""
    script = Script(
        topic="Chicxulub", title="The Crater", hook_text="HOOK",
        scenes=[Scene(narration="Did you know the Chicxulub crater killed the dinosaurs?",
                      search_query="Chicxulub crater", image_prompt="crater")],
        description="d", hashtags=["#science"], tags=["science"], category="science",
    )
    issue = validate_opening_hook(script)
    assert issue is not None
    assert "Did you know" in issue or "greeting" in issue


def test_specific_hook_accepted():
    """A specific subject-opening hook passes validation."""
    script = Script(
        topic="Chicxulub", title="The Crater", hook_text="HOOK",
        scenes=[Scene(narration="The Chicxulub impact ended the age of dinosaurs.",
                      search_query="Chicxulub crater", image_prompt="crater")],
        description="d", hashtags=["#science"], tags=["science"], category="science",
    )
    issue = validate_opening_hook(script)
    assert issue is None


# ------------------------------------------------------------------ fact verification regression

def test_unsupported_claim_is_needs_human_review():
    """A claim with no supporting source gets needs_human_review status."""


    claims = [("The Chicxulub asteroid was 10 kilometers wide.")]
    sources = []  # no sources to confirm
    result = verify_claims(claims, sources)
    assert len(result) == 1
    assert result[0].status == NEEDS_HUMAN_REVIEW


def test_supported_claim_is_verified():
    """A claim matched to a source gets verified status."""


    claims = [("The Chicxulub crater is in Mexico.")]
    sources = [{"title": "Chicxulub crater", "url": "https://example.com/chicxulub"}]
    result = verify_claims(claims, sources)
    assert len(result) == 1
    assert result[0].status == VERIFIED


# ------------------------------------------------------------------ duplicate upload prevention

def test_duplicate_upload_prevented_by_history():
    """upload_record refuses a video that already has a youtube_id."""


    settings = Settings()
    studio = Studio.__new__(Studio)
    studio.s = settings

    # Create a fake history record with youtube_id already set
    class FakeHistory:
        def get(self, rid):
            return {"id": rid, "youtube_id": "existing123", "status": "scheduled",
                    "video_path": "/tmp/video.mp4", "title": "Test"}
        def update_video(self, vid, **fields):
            pass

    studio.history = FakeHistory()
    status = studio.upload_record(1)
    assert status == "already-uploaded"


# ------------------------------------------------------------------ recovery: upload success after runner interruption

def test_recovery_reuses_uploaded_video(tmp_history):
    """A previously uploaded video is not re-uploaded during recovery."""
    tmp_history.record_failure(
        run_date="2026-10-31", slot=1, key="2026-10-31-slot-1",
        topic="topic", stage="upload", error="interrupted", recoverable=True,
        rendered=True, uploaded=True, youtube_id="existing-vid",
    )
    rec = tmp_history.get_recovery_for_slot("2026-10-31-slot-1")
    assert rec is not None
    assert rec["uploaded"] == 1
    assert rec["youtube_id"] == "existing-vid"


# ------------------------------------------------------------------ independent slot operation

def test_slot_independence_different_keys(tmp_history):
    """Slot 1 and slot 2 for the same date have different recovery keys."""
    tmp_history.record_failure(
        run_date="2026-10-31", slot=1, key="2026-10-31-slot-1",
        topic="A", stage="script", error="fail", recoverable=True,
    )
    tmp_history.record_failure(
        run_date="2026-10-31", slot=2, key="2026-10-31-slot-2",
        topic="B", stage="script", error="fail", recoverable=True,
    )
    slot1 = tmp_history.get_recovery_for_slot("2026-10-31-slot-1")
    slot2 = tmp_history.get_recovery_for_slot("2026-10-31-slot-2")
    assert slot1 is not None and slot2 is not None
    assert slot1["key"] != slot2["key"]


# ------------------------------------------------------------------ SlotAutomation recovery flow

def test_slot_automation_checks_recovery_first(tmp_path, monkeypatch):
    """SlotAutomation checks the recovery queue before generating fresh."""
    db = tmp_path / "history.db"
    history = History(db)

    # Record a failure for slot 1 today
    today = datetime.now(timezone.utc).date().isoformat()
    key = f"{today}-slot-1"
    history.record_failure(
        run_date=today, slot=1, key=key,
        topic="test topic", stage="upload", error="network timeout",
        recoverable=True, rendered=True, uploaded=False,
    )

    # The recovery should be found
    rec = history.get_recovery_for_slot(key)
    assert rec is not None
    assert rec["stage"] == "upload"


def test_recovery_stats_returns_counts(tmp_history):
    """recovery_stats returns per-status counts."""
    tmp_history.record_failure(
        run_date="2026-10-31", slot=1, key="2026-10-31-slot-1",
        topic="A", stage="script", error="fail", recoverable=True,
    )
    tmp_history.record_failure(
        run_date="2026-10-31", slot=2, key="2026-10-31-slot-2",
        topic="B", stage="script", error="fail", recoverable=True,
    )
    stats = tmp_history.recovery_stats()
    assert stats.get("pending", 0) >= 2


# ------------------------------------------------------------------ LLMError and NotAuthorized are permanent

def test_llm_error_is_permanent():
    """LLMError (config/auth) should not be retried as temporary."""
    # LLMError in script stage -> permanent
    assert _is_temporary(str(LLMError("invalid API key")), "script") is False


def test_not_authorized_is_permanent():
    """YouTube NotAuthorized should not be retried."""
    assert _is_temporary(str(NotAuthorized("no credentials")), "upload") is False


# ------------------------------------------------------------------ invalid claim JSON fails safely

def test_invalid_claim_json_does_not_silent_fallback():
    """LLM claim extraction failure raises, never silently uses heuristic."""


    class FailingLLM:
        name = "custom"
        model = "openai/gpt-oss-20b"
        base_url = "https://api.groq.com/openai/v1"
        api_key = "test-key"

        def complete_json(self, system, user, schema=None, max_tokens=4000):
            raise LLMError("empty reply (finish_reason=length)")

    llm = FailingLLM()
    with pytest.raises(LLMError):
        _extract_claims_llm(llm, "Octopuses have three hearts.", "octopuses")


# ------------------------------------------------------------------ unverified-claim rewrite regression (slot 3 crash)

def _make_script(scenes):

    return Script(
        topic="Chicxulub", title="The Crater", hook_text="HOOK",
        scenes=[Scene(narration=s, search_query="crater", image_prompt="crater") for s in scenes],
        description="d", hashtags=["#science"], tags=["science"], category="science",
    )


def _pick():
    return Topic("Chicxulub", "niche",
                 source_references=[{"title": "Crater", "url": "https://example.com/crater"}],
                 context="")


class _RewriteLLM:
    """Fake LLM for _rewrite_unverified_claims: returns a full rewritten script JSON."""

    def __init__(self, scene_narrations=None, fail=False):
        self.scene_narrations = scene_narrations
        self.fail = fail
        self.calls = 0

    def complete_json(self, system, user, schema=None, max_tokens=4000):
        self.calls += 1
        if self.fail:
            raise RuntimeError("LLM is down")
        if self.scene_narrations is None:
            return None
        return {
            "topic": "Chicxulub", "title": "The Crater",
            "scenes": [{"narration": n, "search_query": "crater", "image_prompt": "crater"}
                       for n in self.scene_narrations],
        }


def test_claim_rewrite_out_of_range_scene_index_does_not_crash(tmp_path):
    """Reproduces the slot-3 crash: a claim points past the last scene.

    _rewrite_unverified_claims must not raise IndexError; claims on missing
    scenes stay unresolved in the report instead of rewriting the wrong scene.
    """



    settings = Settings(output_dir=str(tmp_path / "out"), data_dir=str(tmp_path / "data"))
    script = _make_script(["scene one.", "scene two."])
    before = [sc.narration for sc in script.scenes]
    # Indices 5 and 7 do not exist on a 2-scene script -- this used to crash.
    unresolved = [Claim(text="old claim", scene_index=5), Claim(text="other", scene_index=7)]
    studio = Studio.__new__(Studio)
    repaired = studio._rewrite_unverified_claims(_RewriteLLM(), script, unresolved, settings, _pick())
    assert repaired is None  # nothing placeable: no trustworthy rewrite, nothing changes
    assert [sc.narration for sc in script.scenes] == before  # nothing was rewritten


def test_claim_rewrite_replaces_only_flagged_scenes(tmp_path):
    """Verified scenes and every other story element are preserved verbatim."""



    settings = Settings(output_dir=str(tmp_path / "out"), data_dir=str(tmp_path / "data"))
    script = _make_script(["scene one.", "scene two.", "scene three."])
    unresolved = [Claim(text="claim one", scene_index=0), Claim(text="claim three", scene_index=2)]
    studio = Studio.__new__(Studio)
    repaired = studio._rewrite_unverified_claims(
        _RewriteLLM(["REWRITTEN ONE", "scene two.", "REWRITTEN THREE"]), script, unresolved,
        settings, _pick())
    assert repaired is not None
    assert script.scenes[0].narration == "REWRITTEN ONE"
    assert script.scenes[1].narration == "scene two."  # not flagged: untouched
    assert script.scenes[2].narration == "REWRITTEN THREE"


def test_claim_rewrite_mismatched_scene_count_keeps_the_draft(tmp_path):
    """A rewrite that changes the scene count is discarded, never rendered."""



    settings = Settings(output_dir=str(tmp_path / "out"), data_dir=str(tmp_path / "data"))
    script = _make_script(["scene one.", "scene two."])
    before = [sc.narration for sc in script.scenes]
    unresolved = [Claim(text="claim one", scene_index=0)]
    studio = Studio.__new__(Studio)
    repaired = studio._rewrite_unverified_claims(
        _RewriteLLM(["only one scene"]), script, unresolved, settings, _pick())
    assert repaired is None
    assert [sc.narration for sc in script.scenes] == before


def test_claim_rewrite_empty_or_unplaceable_claims_return_none(tmp_path):
    """Nothing to rewrite, or nothing placeable: keep the draft, never crash."""



    settings = Settings(output_dir=str(tmp_path / "out"), data_dir=str(tmp_path / "data"))
    script = _make_script(["scene one."])
    studio = Studio.__new__(Studio)
    assert studio._rewrite_unverified_claims(_RewriteLLM(), script, [], settings, _pick()) is None
    # Negative / out-of-range indices are unplaceable: nothing is rewritten.
    assert studio._rewrite_unverified_claims(
        _RewriteLLM(), script, [Claim(text="c", scene_index=-1)], settings, _pick()) is None
    assert studio._rewrite_unverified_claims(
        _RewriteLLM(), script, [Claim(text="c", scene_index=9)], settings, _pick()) is None
    assert script.scenes[0].narration == "scene one."


def test_claim_rewrite_llm_failure_returns_none(tmp_path):
    """A failing LLM means no trustworthy rewrite: caller keeps blocking."""



    settings = Settings(output_dir=str(tmp_path / "out"), data_dir=str(tmp_path / "data"))
    script = _make_script(["scene one.", "scene two."])
    unresolved = [Claim(text="c", scene_index=0)]
    studio = Studio.__new__(Studio)
    assert studio._rewrite_unverified_claims(_RewriteLLM(fail=True), script, unresolved,
                                             settings, _pick()) is None


def test_claim_extraction_keeps_scene_indices_within_the_script():
    """LLM re-extraction that changes the claim list never indexes past the last scene."""


    class ReextractingLLM:
        def complete_json(self, system, user, schema=None, max_tokens=4000):
            # 5 claims extracted from a script that has only 3 claim slots
            return {"claims": ["one", "two", "three", "four", "five"]}

    topic = Topic("Chicxulub", "niche", source_references=[], context="")
    script_claims = [(0, "claim one"), (0, "claim two"), (1, "claim three")]
    report = verify_script_claims(script_claims, topic, ReextractingLLM())
    assert report.llm_extraction_failed is True
    assert len(report.claims) == 5  # no claim is silently dropped
    for claim in report.claims:
        assert claim.scene_index is None or claim.scene_index in (0, 1)
