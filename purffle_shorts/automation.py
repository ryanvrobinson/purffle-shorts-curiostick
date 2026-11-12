"""Daily automation controller: runs at most one video per day with proper locking, quality gates,
and upload safeguards. Designed for Windows Task Scheduler."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from .config import ConfigError, Settings
from .history import History, now_iso
from . import youtube
from .pipeline import Result, Studio
from .quality import QualityGate, QualityReport
from .llm import LLMError

log = logging.getLogger("purffle")

# ------------------------------------------------------------------ failure classification
TEMPORARY_FAILURE_STAGES = {"tts", "footage", "render", "upload", "verify"}
TEMPORARY_ERROR_HINTS = (
    "429", "rate limit", "temporary", "network", "timeout",
    "connection", "502", "503", "504", "unavailable", "overloaded",
)

PERMANENT_FAILURE_STAGES = {"script", "review", "qc"}

# Bounded retry configuration
MAX_RECOVERY_RETRIES = 3
RETRY_BASE_SECONDS = 60
RETRY_MAX_SECONDS = 3600


def _is_temporary(error: str, stage: str) -> bool:
    """Classify a failure as temporary (retryable) or permanent (content/config)."""
    err = (error or "").lower()
    if stage in TEMPORARY_FAILURE_STAGES:
        return any(h in err for h in TEMPORARY_ERROR_HINTS) or stage in {"tts", "footage", "render"}
    if stage in PERMANENT_FAILURE_STAGES:
        return False
    # Unknown stage: treat error-message hints as the decider
    return any(h in err for h in TEMPORARY_ERROR_HINTS)


def _retry_delay(retry_count: int) -> float:
    """Exponential backoff with jitter, capped at RETRY_MAX_SECONDS."""
    import random
    delay = min(RETRY_BASE_SECONDS * (2 ** retry_count), RETRY_MAX_SECONDS)
    return delay + random.uniform(0, delay * 0.25)

# The three CurioStick publishing slots, Asia/Kolkata (UTC+5:30).
SLOT_TIMES: dict[int, str] = {1: "09:00", 2: "14:00", 3: "19:00"}
SLOT_LABELS: dict[int, str] = {1: "9:00 AM IST", 2: "2:00 PM IST", 3: "7:00 PM IST"}


@dataclass
class DailyRunResult:
    """Result of a daily automation run."""
    ok: bool
    video_result: Optional[Result] = None
    quality_report: Optional[QualityReport] = None
    uploaded: bool = False
    instagram_published: bool = False
    error: str = ""
    run_date: str = ""
    run_id: int = 0


class RunLock:
    """File-based lock to prevent overlapping daily runs."""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.lock_fd: Optional[int] = None
        self.acquired = False

    def acquire(self, timeout: float = 0) -> bool:
        """Try to acquire the lock. Returns True if successful."""
        try:
            # Create lock file with exclusive creation
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            if sys.platform == "win32":
                import msvcrt
                # On Windows, use msvcrt locking
                self.lock_fd = os.open(self.lock_path, flags)
                msvcrt.locking(self.lock_fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                self.lock_fd = os.open(self.lock_path, flags)
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write our PID and timestamp
            pid = os.getpid()
            timestamp = now_iso()
            os.write(self.lock_fd, f"{pid}\n{timestamp}\n".encode())
            os.fsync(self.lock_fd)
            self.acquired = True
            return True
        except (OSError, IOError, BlockingIOError):
            if self.lock_fd is not None:
                try:
                    os.close(self.lock_fd)
                except Exception:
                    pass
                self.lock_fd = None
            return False

    def release(self) -> None:
        """Release the lock."""
        if self.acquired and self.lock_fd is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    msvcrt.locking(self.lock_fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                os.close(self.lock_fd)
            except Exception:
                pass
            self.lock_fd = None
            self.acquired = False
        # Remove lock file (after closing the file descriptor)
        try:
            self.lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Could not acquire run lock; another instance may be running")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class DailyRunTracker:
    """Tracks daily run status in the history database."""

    def __init__(self, history: History):
        self.history = history
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure the daily_runs table exists."""
        with self.history._db() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS daily_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT NOT NULL UNIQUE,  -- YYYY-MM-DD in configured timezone
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,  -- running | completed | failed | skipped
                    video_id INTEGER,
                    error TEXT,
                    quality_gate_passed BOOLEAN DEFAULT 0,
                    uploaded BOOLEAN DEFAULT 0,
                    instagram_published BOOLEAN DEFAULT 0
                )
            """)

    def get_today_run(self, run_date: str) -> Optional[dict]:
        """Get today's run record if it exists."""
        rows = self.history._all(
            "SELECT * FROM daily_runs WHERE run_date = ?", (run_date,)
        )
        return rows[0] if rows else None

    def start_run(self, run_date: str) -> int:
        """Record the start of a daily run. Returns the run ID."""
        # Check if a run for this date already exists
        existing = self.get_today_run(run_date)
        if existing:
            # Update existing record to 'running'
            self.history._exec(
                "UPDATE daily_runs SET started_at = ?, status = 'running' WHERE id = ?",
                (now_iso(), existing["id"])
            )
            return existing["id"]

        run_id = self.history._exec(
            "INSERT INTO daily_runs (run_date, started_at, status) VALUES (?, ?, 'running')",
            (run_date, now_iso())
        )
        return run_id

    def complete_run(self, run_id: int, video_id: Optional[int], quality_passed: bool,
                     uploaded: bool, instagram_published: bool, error: str = "") -> None:
        """Mark the run as completed or failed."""
        status = "completed" if not error else "failed"
        self.history._exec(
            """UPDATE daily_runs SET completed_at = ?, status = ?, video_id = ?,
               quality_gate_passed = ?, uploaded = ?, instagram_published = ?, error = ?
               WHERE id = ?""",
            (now_iso(), status, video_id, quality_passed, uploaded, instagram_published, error, run_id)
        )

    def skip_run(self, run_date: str, reason: str) -> None:
        """Record that a run was skipped."""
        self.history._exec(
            "INSERT OR IGNORE INTO daily_runs (run_date, started_at, status, error) VALUES (?, ?, 'skipped', ?)",
            (run_date, now_iso(), reason)
        )

    def has_completed_run_today(self, run_date: str) -> bool:
        """Check if a run for this date has already completed successfully."""
        rows = self.history._all(
            "SELECT 1 FROM daily_runs WHERE run_date = ? AND status = 'completed'", (run_date,)
        )
        return len(rows) > 0


class DailyAutomation:
    """Main daily automation controller."""

    def __init__(self, settings: Settings, dry_run: bool = False, skip_quality: bool = False,
                 skip_upload: bool = True, skip_instagram: bool = True):
        self.settings = settings.with_overrides(upload=False)
        self.dry_run = dry_run
        self.skip_quality = skip_quality
        self.skip_upload = skip_upload
        self.skip_instagram = skip_instagram
        self.history = History(settings.data_path / "history.db")
        self.tracker = DailyRunTracker(self.history)
        self.studio = Studio(self.settings)
        self._lock: Optional[RunLock] = None

    def _get_run_date(self) -> str:
        """Get today's date in the configured timezone."""
        if self.settings.timezone:
            from zoneinfo import ZoneInfo
            try:
                tz = ZoneInfo(self.settings.timezone)
                return datetime.now(tz).date().isoformat()
            except Exception:
                pass
        return datetime.now(timezone.utc).date().isoformat()

    def _get_lock_path(self) -> Path:
        """Get the lock file path."""
        return self.settings.data_path / "daily_run.lock"

    def run_once(self) -> DailyRunResult:
        """Execute one daily automation cycle."""
        run_date = self._get_run_date()
        lock_path = self._get_lock_path()

        # Check if already completed today
        if self.tracker.has_completed_run_today(run_date):
            msg = f"Daily run for {run_date} already completed"
            log.info(msg)
            return DailyRunResult(
                ok=True, error=msg, run_date=run_date
            )

        # Try to acquire lock
        self._lock = RunLock(lock_path)
        if not self._lock.acquire():
            msg = "Another daily run is already in progress"
            log.warning(msg)
            return DailyRunResult(ok=False, error=msg, run_date=run_date)

        # Record run start
        run_id = self.tracker.start_run(run_date)

        try:
            if self.dry_run:
                log.info("DRY RUN MODE - no video will be generated or uploaded")
                # Just verify configuration
                self._verify_config()
                self.tracker.complete_run(run_id, None, True, False, False, "")
                return DailyRunResult(ok=True, run_date=run_date, run_id=run_id,
                                      error="dry run completed")

            # Generate video
            log.info("Starting video generation for %s", run_date)
            result = self.studio.make()

            if not result.ok:
                error = f"Video generation failed: {result.error}"
                log.error(error)
                self.tracker.complete_run(run_id, None, False, False, False, error)
                return DailyRunResult(ok=False, error=error, run_date=run_date, run_id=run_id)

            # Quality gate
            quality_passed = True
            quality_report = None
            if not self.skip_quality:
                log.info("Running quality gate checks...")
                self.quality_gate = QualityGate(self.settings)
                quality_report = self.quality_gate.check(result)
                quality_passed = quality_report.passed
                if not quality_passed:
                    error = f"Quality gate failed: {quality_report.summary}"
                    log.error(error)
                    self.tracker.complete_run(run_id, result.record_id, False, False, False, error)
                    return DailyRunResult(
                        ok=False, video_result=result, quality_report=quality_report,
                        error=error, run_date=run_date, run_id=run_id
                    )
                log.info("Quality gate passed")

            # Upload to YouTube
            uploaded = False
            if not self.skip_upload and self.settings.upload:
                log.info("Uploading to YouTube...")
                try:
                    upload_result = self.studio.upload_record(result.record_id, result)
                    uploaded = upload_result in ("uploaded", "scheduled")
                    if not uploaded:
                        error = "YouTube upload failed"
                        log.error(error)
                        self.tracker.complete_run(run_id, result.record_id, quality_passed, False, False, error)
                        return DailyRunResult(
                            ok=False, video_result=result, quality_report=quality_report,
                            error=error, run_date=run_date, run_id=run_id
                        )
                    log.info("YouTube upload successful")
                except Exception as e:
                    error = f"YouTube upload error: {e}"
                    log.error(error)
                    self.tracker.complete_run(run_id, result.record_id, quality_passed, False, False, error)
                    return DailyRunResult(
                        ok=False, video_result=result, quality_report=quality_report,
                        error=error, run_date=run_date, run_id=run_id
                    )
            else:
                log.info("Skipping YouTube upload (skip_upload=%s, settings.upload=%s)",
                         self.skip_upload, self.settings.upload)

            # Instagram Reels publishing (if configured and not skipped)
            instagram_published = False
            if not self.skip_instagram and self._instagram_configured():
                log.info("Publishing to Instagram Reels...")
                try:
                    from .instagram import publish_reel
                    instagram_published = publish_reel(self.settings, result)
                    if instagram_published:
                        log.info("Instagram Reels publish successful")
                    else:
                        log.warning("Instagram Reels publish returned false")
                except Exception as e:
                    log.warning("Instagram Reels publish failed (non-fatal): %s", e)
            else:
                log.info("Skipping Instagram Reels (skip_instagram=%s, configured=%s)",
                         self.skip_instagram, self._instagram_configured())

            # Record successful completion
            self.tracker.complete_run(
                run_id, result.record_id, quality_passed, uploaded, instagram_published
            )

            return DailyRunResult(
                ok=True, video_result=result, quality_report=quality_report,
                uploaded=uploaded, instagram_published=instagram_published,
                run_date=run_date, run_id=run_id
            )

        finally:
            if self._lock:
                self._lock.release()

    def _verify_config(self) -> None:
        """Verify configuration is valid for automation."""
        issues = []
        if not self.settings.llm_provider and not self.settings.offline:
            issues.append("No LLM provider configured (set LLM_PROVIDER or use offline mode)")
        if not self.settings.tts_engine:
            issues.append("No TTS engine configured")
        if not self.settings.pexels_api_key and not self.settings.pixabay_api_key and \
           not self.settings.pollinations_api_key and not self.settings.openai_api_key:
            issues.append("No visual sources configured (PEXELS_API_KEY, PIXABAY_API_KEY, etc.)")
        if self.settings.upload and not Path(self.settings.client_secrets).exists():
            issues.append("YouTube upload enabled but credentials.json not found")
        if issues:
            log.warning("Configuration issues: %s", "; ".join(issues))

    def _instagram_configured(self) -> bool:
        """Check if Instagram publishing is configured."""
        from .instagram import is_instagram_configured
        return is_instagram_configured()


# ------------------------------------------------------------------ slot scheduling helpers

def slot_key(run_date: str, slot: int) -> str:
    """Stable idempotency key for one date + slot, e.g. 2026-10-31-slot-1."""
    return History.slot_key(run_date, slot)


def slot_publish_moments(run_date: str | None = None,
                         timezone_name: str = "Asia/Kolkata") -> dict[int, datetime]:
    """Return {slot: aware IST datetime} for the three daily slots on ``run_date``.

    Slot 1 = 09:00 IST, slot 2 = 14:00 IST, slot 3 = 19:00 IST.
    Defaults to today in the project timezone.
    """
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(timezone_name)
    if run_date is None:
        day = datetime.now(tz).date()
    else:
        day = datetime.strptime(run_date, "%Y-%m-%d").date()
    moments = {}
    for slot, hhmm in SLOT_TIMES.items():
        hh, mm = (int(x) for x in hhmm.split(":"))
        moments[slot] = datetime.combine(day, dtime(hh, mm), tzinfo=tz)
    return moments


def next_available_moment(now: datetime | None = None,
                          timezone_name: str = "Asia/Kolkata") -> datetime:
    """The next publishing slot moment at or after ``now`` (09:00 / 14:00 / 19:00 IST).

    Implements the CurioStick catch-up policy: when a slot's nominal publication
    time has passed, the video is published at the next available future slot
    instead of being dropped or published into the past. A small lead time keeps
    the schedule realistic for unattended runs.
    """
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(timezone_name)
    now = (now or datetime.now(timezone.utc)).astimezone(tz)
    lead = timedelta(minutes=5)
    day = now.date()
    for offset in range(0, 3):
        d = day + timedelta(days=offset)
        for slot in (1, 2, 3):
            hh, mm = (int(x) for x in SLOT_TIMES[slot].split(":"))
            cand = datetime.combine(d, dtime(hh, mm), tzinfo=tz)
            if cand >= now + lead:
                return cand
    raise RuntimeError("No future publishing slot found")


@dataclass
class SlotRunResult:
    """Result of running one publishing slot."""

    ok: bool
    slot: int
    run_date: str
    key: str
    publish_at_utc: str | None = None
    publish_at_ist: str | None = None
    video_result: Optional[Result] = None
    quality_report: Optional[QualityReport] = None
    uploaded: bool = False
    youtube_id: str | None = None
    skipped: bool = False
    skip_reason: str = ""
    dry_run: bool = False
    error: str = ""


class SlotAutomation:
    """One Short for one publishing slot: generate -> quality gate -> schedule on YouTube.

    Designed for unattended GitHub Actions runs. There is no interactive PUBLISH prompt:
    a scheduled run uploads the video as ``private`` with a future ``publishAt`` for the
    assigned IST slot, and a dry run never touches YouTube.
    """

    def __init__(self, settings: Settings, slot: int, run_date: str | None = None,
                 dry_run: bool = False, skip_quality: bool = False,
                 studio: Studio | None = None, history: History | None = None,
                 recover: bool = True):
        if slot not in SLOT_TIMES:
            raise ConfigError(f"Invalid slot {slot}: choose 1 (09:00 IST), 2 (14:00 IST) or 3 (19:00 IST)")
        self.s = settings.with_overrides(upload=False)
        self.slot = int(slot)
        self.run_date = run_date or self._today_iso()
        self.key = slot_key(self.run_date, self.slot)
        self.dry_run = dry_run
        self.skip_quality = skip_quality
        self.recover = recover
        self.history = history or History(self.s.data_path / "history.db")
        self.studio = studio or Studio(self.s)
        self._failures_dir = self.s.data_path / "failures"

    def _today_iso(self) -> str:
        tz_name = self.s.timezone or "Asia/Kolkata"
        try:
            return datetime.now(ZoneInfo(tz_name)).date().isoformat()
        except Exception:
            return datetime.now(timezone.utc).date().isoformat()

    def run_slot(self) -> SlotRunResult:
        """Generate one Short and schedule it for this slot's IST publication time.

        If a previous run for this slot+date failed and is recoverable, attempt
        recovery first (reusing rendered assets when safe) before generating fresh.
        """
        moments = slot_publish_moments(self.run_date, self.s.timezone)
        publish_at = moments[self.slot]                       # aware IST datetime
        publish_at_utc = youtube.to_rfc3339(publish_at)
        now = datetime.now(publish_at.tzinfo)

        log.info("Slot %s (%s) for %s: publish target %s = %s UTC",
                 self.slot, SLOT_TIMES[self.slot], self.run_date,
                 publish_at.isoformat(), publish_at_utc)

        # Catch-up policy: if this slot's nominal publication time has passed and the
        # video was not already produced, publish at the next available future slot
        # instead of dropping it or publishing into the past. Dry runs keep the
        # original target (nothing is scheduled) so a manual dry-run still works
        # after the nominal slot time.
        if (not self.dry_run and publish_at <= now
                and not self.history.is_slot_done(self.key)):
            catch_up = next_available_moment(now, self.s.timezone)
            log.warning(
                "Slot %s (%s) for %s: publish target %s is in the past; "
                "catching up at %s",
                self.slot, SLOT_TIMES[self.slot], self.run_date,
                publish_at.isoformat(), catch_up.isoformat())
            publish_at = catch_up
            publish_at_utc = youtube.to_rfc3339(catch_up)

        # Recovery-first: if this slot previously failed and is retryable, try recovery.
        if self.recover:
            recovery = self.history.get_recovery_for_slot(self.key)
            if recovery:
                log.info("Found pending recovery for %s (attempt %d/%d, stage %s)",
                         self.key, recovery["retry_count"], recovery["max_retries"],
                         recovery["stage"])
                out = self._attempt_recovery(recovery, publish_at, publish_at_utc, now)
                self._write_summary(out)
                return out

        out = self._run_slot(publish_at, publish_at_utc, now)
        self._write_summary(out)
        return out

    def _run_slot(self, publish_at: datetime, publish_at_utc: str, now: datetime) -> SlotRunResult:
        # 1. Idempotency: this slot was already completed on YouTube.
        if self.history.is_slot_done(self.key):
            existing = self.history.get_slot(self.key)
            msg = f"Slot {self.slot} for {self.run_date} already scheduled (video {existing.get('youtube_id')})"
            log.info(msg)
            return SlotRunResult(
                ok=True, slot=self.slot, run_date=self.run_date, key=self.key,
                publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                skipped=True, skip_reason=msg, youtube_id=existing.get("youtube_id"))

        # 2. Never publish into the past: if the slot time passed, never publish late.
        if publish_at <= now:
            if self.history.is_slot_done(self.key):
                # Already produced (e.g. by an earlier catch-up run).
                existing = self.history.get_slot(self.key)
                if existing is None:
                    existing = {}
                msg = f"Slot {self.slot} for {self.run_date} already scheduled (video {existing.get('youtube_id')})"
                log.info(msg)
                return SlotRunResult(
                    ok=True, slot=self.slot, run_date=self.run_date, key=self.key,
                    publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                    skipped=True, skip_reason=msg, youtube_id=existing.get("youtube_id"))
            if self.dry_run:
                # Manual dry-runs still work after the nominal slot time: generate and
                # quality-check, but never schedule into the past.
                log.warning(
                    "Slot %s (%s) for %s: publish target %s is in the past; "
                    "dry run only, nothing will be scheduled",
                    self.slot, SLOT_TIMES[self.slot], self.run_date, publish_at.isoformat())
            else:
                # Real run: the slot was missed. Record it accurately and queue a
                # bounded catch-up recovery that publishes at the next available slot.
                msg = f"Slot {self.slot} publish time {publish_at.isoformat()} is in the past; missed slot, not publishing late"
                log.warning(msg)
                self.history.record_slot(self.key, self.run_date, self.slot, publish_at_utc,
                                         "missed", error="publish time passed")
                self.history.record_failure(
                    self.run_date, self.slot, self.key, None, "schedule", msg,
                    recoverable=True, rendered=False, uploaded=False,
                )
                return SlotRunResult(
                    ok=False, slot=self.slot, run_date=self.run_date, key=self.key,
                    publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                    skipped=True, skip_reason=msg, error=msg)

        # 3. Generate the video (topic, script, voice, footage, render).
        try:
            result = self.studio.make()
        except Exception as e:  # setup problems (no LLM, no auth) stop this slot cleanly
            error = f"Video generation failed: {e}"
            log.error(error)
            recoverable = not isinstance(e, (LLMError, youtube.NotAuthorized))
            self.history.record_slot(self.key, self.run_date, self.slot, publish_at_utc,
                                     "failed", error=error)
            self.history.record_failure(
                self.run_date, self.slot, self.key, None, "script", error,
                recoverable=recoverable,
            )
            return SlotRunResult(ok=False, slot=self.slot, run_date=self.run_date, key=self.key,
                                 publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                                 error=error)
        if not result.ok:
            error = f"Video generation failed: {result.error}"
            log.error(error)
            self.history.record_slot(self.key, self.run_date, self.slot, publish_at_utc,
                                     "failed", error=error)
            self.history.record_failure(
                self.run_date, self.slot, self.key, result.title, "script", error,
                recoverable=True,
            )
            return SlotRunResult(
                ok=False, slot=self.slot, run_date=self.run_date, key=self.key,
                publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                video_result=result, error=error)

        # 4. Quality gate: a failed gate blocks publishing and is recorded.
        quality_report = None
        if not self.skip_quality:
            quality_report = QualityGate(self.s).check(result)
            if not quality_report.passed:
                error = f"Quality gate failed: {quality_report.summary}"
                log.error(error)
                self._save_failure_details(result, quality_report)
                self.history.record_slot(self.key, self.run_date, self.slot, publish_at_utc,
                                         "failed", error=error)
                self.history.record_failure(
                    self.run_date, self.slot, self.key, result.title, "qc", error,
                    recoverable=True, rendered=True,
                    video_id=result.record_id,
                )
                return SlotRunResult(
                    ok=False, slot=self.slot, run_date=self.run_date, key=self.key,
                    publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                    video_result=result, quality_report=quality_report, error=error)
            log.info("Quality gate passed for slot %s", self.slot)

        # 5. Dry run: everything above ran, but nothing is uploaded or scheduled.
        if self.dry_run:
            return SlotRunResult(
                ok=True, slot=self.slot, run_date=self.run_date, key=self.key,
                publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                video_result=result, quality_report=quality_report, dry_run=True)

        # 6. Non-interactive scheduled upload: private + future publishAt.
        try:
            status = self.studio.upload_scheduled(
                result.record_id, publish_at, result, slot=self.slot, run_date=self.run_date)
        except Exception as e:  # NotAuthorized, network, quota: record, never blindly retry
            error = f"YouTube upload failed: {e}"
            log.error(error)
            recoverable = not isinstance(e, youtube.NotAuthorized)
            self.history.record_slot(self.key, self.run_date, self.slot, publish_at_utc,
                                     "failed", error=error)
            self.history.record_failure(
                self.run_date, self.slot, self.key, result.title, "upload", error,
                recoverable=recoverable, rendered=True, uploaded=False,
                video_id=result.record_id,
            )
            return SlotRunResult(
                ok=False, slot=self.slot, run_date=self.run_date, key=self.key,
                publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                video_result=result, quality_report=quality_report, error=error)

        rec = self.history.get(result.record_id)
        if status == "scheduled":
            self.history.record_slot(self.key, self.run_date, self.slot, publish_at_utc,
                                     "scheduled", youtube_id=rec.get("youtube_id"),
                                     title=rec.get("title"))
            log.info("Slot %s scheduled on YouTube: %s", self.slot, rec.get("youtube_id"))
            return SlotRunResult(
                ok=True, slot=self.slot, run_date=self.run_date, key=self.key,
                publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                video_result=result, quality_report=quality_report,
                uploaded=True, youtube_id=rec.get("youtube_id"))
        if status == "queued":
            error = "YouTube upload quota reached; slot not scheduled (not blindly retried)"
            log.warning(error)
            self.history.record_slot(self.key, self.run_date, self.slot, publish_at_utc,
                                     "queued", error=error)
            return SlotRunResult(
                ok=False, slot=self.slot, run_date=self.run_date, key=self.key,
                publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                video_result=result, quality_report=quality_report, error=error)
        error = f"YouTube upload did not schedule the video (status: {status})"
        log.error(error)
        self.history.record_slot(self.key, self.run_date, self.slot, publish_at_utc,
                                 "failed", error=error)
        return SlotRunResult(
            ok=False, slot=self.slot, run_date=self.run_date, key=self.key,
            publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
            video_result=result, quality_report=quality_report, error=error)

    def _attempt_recovery(self, recovery: dict, publish_at: datetime,
                          publish_at_utc: str, now: datetime) -> SlotRunResult:
        """Attempt to recover a previously failed slot.

        Reuses valid rendered videos when safe; otherwise regenerates.
        Never publishes an already-published video twice.
        """
        recovery_id = recovery["id"]
        error = recovery["error"]

        # If already uploaded/scheduled, mark completed and skip.
        if recovery.get("uploaded") and recovery.get("youtube_id"):
            log.info("Slot %s already uploaded (video %s); marking recovery complete",
                     self.key, recovery["youtube_id"])
            self.history.mark_recovery_completed(recovery_id, "already-uploaded",
                                                 recovery["youtube_id"])
            self.history.record_slot(self.key, self.run_date, self.slot, publish_at_utc,
                                     "scheduled", youtube_id=recovery["youtube_id"],
                                     title=recovery.get("topic"))
            return SlotRunResult(
                ok=True, slot=self.slot, run_date=self.run_date, key=self.key,
                publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                uploaded=True, youtube_id=recovery["youtube_id"],
                skipped=True, skip_reason="already uploaded from earlier attempt")

        # Increment retry count; check if we've exhausted the budget.
        self.history.increment_retry(recovery_id)
        rec = self.history.get_recovery(recovery_id)
        if rec and rec["retry_count"] >= rec["max_retries"]:
            log.error("Slot %s recovery exhausted (%d attempts); marking failed",
                      self.key, rec["max_retries"])
            self.history.mark_recovery_failed(recovery_id, "exhausted")
            self.history.record_slot(self.key, self.run_date, self.slot, publish_at_utc,
                                     "failed", error=f"recovery exhausted: {error}")
            return SlotRunResult(
                ok=False, slot=self.slot, run_date=self.run_date, key=self.key,
                publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                error=f"recovery exhausted after {rec['max_retries']} attempts: {error}")

        # If a video was already rendered, reuse it (QC + upload only).
        if recovery.get("rendered") and recovery.get("video_id"):
            rec_row = self.history.get(recovery["video_id"])
            if rec_row and rec_row.get("video_path") and Path(rec_row["video_path"]).exists():
                log.info("Slot %s recovery: reusing rendered video #%d", self.key, recovery["video_id"])
                # Fetch the previously rendered record
                # Re-run quality gate on the existing video
                from .pipeline import Result as _Result
                existing_result = _Result(
                    ok=True, title=rec_row.get("title", ""),
                    folder=Path(rec_row["folder"]) if rec_row.get("folder") else None,
                    video=Path(rec_row["video_path"]),
                    status="rendered", record_id=recovery["video_id"],
                    score=rec_row.get("score"), language=rec_row.get("language", ""),
                )
                if not self.skip_quality:
                    quality_report = QualityGate(self.s).check(existing_result)
                    if not quality_report.passed:
                        err = f"quality gate failed on recovered video: {quality_report.summary}"
                        log.error(err)
                        self.history.record_slot(self.key, self.run_date, self.slot,
                                                 publish_at_utc, "failed", error=err)
                        return SlotRunResult(
                            ok=False, slot=self.slot, run_date=self.run_date, key=self.key,
                            publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                            error=err)
                # Upload the existing video
                if not self.dry_run:
                    try:
                        status = self.studio.upload_scheduled(
                            recovery["video_id"], publish_at, existing_result,
                            slot=self.slot, run_date=self.run_date)
                    except Exception as e:
                        err = f"recovery upload failed: {e}"
                        log.error(err)
                        self.history.record_slot(self.key, self.run_date, self.slot,
                                                 publish_at_utc, "failed", error=err)
                        return SlotRunResult(
                            ok=False, slot=self.slot, run_date=self.run_date, key=self.key,
                            publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                            error=err)
                    rec2 = self.history.get(recovery["video_id"])
                    if status == "scheduled":
                        self.history.record_slot(self.key, self.run_date, self.slot,
                                                 publish_at_utc, "scheduled",
                                                 youtube_id=rec2.get("youtube_id"),
                                                 title=rec2.get("title"))
                        self.history.mark_recovery_completed(recovery_id, "retry-upload",
                                                             rec2.get("youtube_id"))
                        return SlotRunResult(
                            ok=True, slot=self.slot, run_date=self.run_date, key=self.key,
                            publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                            uploaded=True, youtube_id=rec2.get("youtube_id"))
                # Dry run: recovery succeeded up to upload point
                self.history.mark_recovery_completed(recovery_id, "retry-dry-run")
                self.history.record_slot(self.key, self.run_date, self.slot, publish_at_utc,
                                         "scheduled", title=rec_row.get("title"))
                return SlotRunResult(
                    ok=True, slot=self.slot, run_date=self.run_date, key=self.key,
                    publish_at_utc=publish_at_utc, publish_at_ist=publish_at.isoformat(),
                    dry_run=True)
            else:
                missing_path = rec_row.get("video_path") if rec_row else "(no record)"
                log.warning(
                    "Slot %s recovery: rendered video file not found at %s; generating fresh",
                    self.key, missing_path,
                )
        # No usable rendered video: generate fresh, but record this as a new attempt.
        log.info("Slot %s recovery: no reusable video; generating fresh", self.key)
        out = self._run_slot(publish_at, publish_at_utc, now)
        # If the fresh generation also failed, the failure is already recorded by _run_slot.
        # Mark this recovery attempt as completed (the new slot record supersedes it).
        if not out.ok:
            self.history.mark_recovery_failed(recovery_id, "regeneration-failed")
        else:
            self.history.mark_recovery_completed(recovery_id, "regeneration-success",
                                                 out.youtube_id)
        return out

    def _save_failure_details(self, result: Result, report: QualityReport) -> None:
        """Persist which check(s) failed so the failure can be reviewed later."""
        self._failures_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "slot": self.slot, "run_date": self.run_date, "key": self.key,
            "video": str(result.video) if result.video else None,
            "folder": str(result.folder) if result.folder else None,
            "title": result.title,
            "report": report.to_dict(),
        }
        (self._failures_dir / f"{self.key}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Failure details saved to %s/%s.json", self._failures_dir, self.key)

    def _write_summary(self, out: SlotRunResult) -> None:
        """Markdown summary for the workflow run and $GITHUB_STEP_SUMMARY."""
        lines = [
            "# CurioStick Shorts — scheduled slot run",
            "",
            f"- Slot: {out.slot} ({SLOT_TIMES[out.slot]} IST)",
            f"- Date: {out.run_date}",
            f"- Publish target: {out.publish_at_ist} ({out.publish_at_utc} UTC)",
            f"- Idempotency key: {out.key}",
            f"- Dry run: {out.dry_run}",
            f"- Skipped: {out.skipped}" + (f" — {out.skip_reason}" if out.skip_reason else ""),
        ]
        uploaded = f"- Uploaded: {out.uploaded}"
        if out.youtube_id:
            # A scheduled upload is private until its publishAt time: it is not yet
            # public on the channel. Never describe it as "published."
            when = f", goes public {out.publish_at_ist}" if out.publish_at_ist else ""
            uploaded = (
                f"- Uploaded: {out.uploaded} — scheduled as private{when} — "
                f"https://youtube.com/shorts/{out.youtube_id}"
            )
        elif out.dry_run:
            uploaded += " — YouTube upload not attempted (dry run)"
        lines.append(uploaded)
        if out.quality_report is None:
            lines.append("- Quality gate: not run (--skip-quality)")
        else:
            lines.append(f"- Quality gate: {'passed' if out.quality_report.passed else 'failed'}")
            if not out.quality_report.passed:
                lines.append("")
                lines.append("Failed checks: " + "; ".join(
                    f"{c.name}: {c.message}" for c in out.quality_report.failed_checks))
        if out.error:
            lines.append("")
            lines.append(f"Error: {out.error}")
        try:
            (self.s.data_path / "run-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as e:  # a summary file must never fail the run
            log.warning("Could not write run summary: %s", e)
