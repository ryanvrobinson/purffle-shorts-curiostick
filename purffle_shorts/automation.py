"""Daily automation controller: runs at most one video per day with proper locking, quality gates,
and upload safeguards. Designed for Windows Task Scheduler."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import Settings
from .history import History, now_iso
from .pipeline import Result, Studio
from .quality import QualityGate, QualityReport

log = logging.getLogger("purffle")


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