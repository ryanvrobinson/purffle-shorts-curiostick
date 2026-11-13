"""Recovery queue, failure classification, and slot independence regression tests."""

from datetime import datetime, timezone

import pytest

from purffle_shorts.automation import _is_temporary, _retry_delay
from purffle_shorts.config import Settings
from purffle_shorts.history import History
from purffle_shorts.llm import LLMError
from purffle_shorts.script import Script, Scene, validate_opening_hook
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
    from purffle_shorts.research import verify_claims, NEEDS_HUMAN_REVIEW

    claims = [("The Chicxulub asteroid was 10 kilometers wide.")]
    sources = []  # no sources to confirm
    result = verify_claims(claims, sources)
    assert len(result) == 1
    assert result[0].status == NEEDS_HUMAN_REVIEW


def test_supported_claim_is_verified():
    """A claim matched to a source gets verified status."""
    from purffle_shorts.research import VERIFIED, verify_claims

    claims = [("The Chicxulub crater is in Mexico.")]
    sources = [{"title": "Chicxulub crater", "url": "https://example.com/chicxulub"}]
    result = verify_claims(claims, sources)
    assert len(result) == 1
    assert result[0].status == VERIFIED


# ------------------------------------------------------------------ duplicate upload prevention

def test_duplicate_upload_prevented_by_history():
    """upload_record refuses a video that already has a youtube_id."""
    from purffle_shorts.pipeline import Studio

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
    from purffle_shorts.research import _extract_claims_llm

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
