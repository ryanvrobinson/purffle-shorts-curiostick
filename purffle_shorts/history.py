"""SQLite history: every video made, what was uploaded or scheduled, which stock clips and topics
were used (so nothing repeats), today's upload count (YouTube quota), the idea queue filled by the
planner, and view counts for the videos that are live."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    subject TEXT, source TEXT, topic TEXT, style TEXT, language TEXT,
    title TEXT, description TEXT, tags TEXT,
    folder TEXT, video_path TEXT, duration REAL,
    llm TEXT, voice TEXT,
    status TEXT NOT NULL,          -- rendered | queued | uploaded | scheduled | failed
    youtube_id TEXT, publish_at TEXT, uploaded_at TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS media_used (key TEXT PRIMARY KEY, used_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS topics_used (key TEXT PRIMARY KEY, used_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    subject TEXT NOT NULL,
    style TEXT, notes TEXT,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | used
    used_at TEXT, video_id INTEGER
);
CREATE TABLE IF NOT EXISTS slot_uploads (
    key TEXT PRIMARY KEY,            -- YYYY-MM-DD-slot-N (Asia/Kolkata date + slot 1-3)
    run_date TEXT NOT NULL,          -- YYYY-MM-DD in the project timezone
    slot INTEGER NOT NULL,           -- 1 = 09:00, 2 = 14:00, 3 = 19:00 IST
    publish_at_utc TEXT NOT NULL,    -- RFC3339 UTC timestamp the video goes public
    youtube_id TEXT,                 -- YouTube video ID once scheduled/uploaded
    title TEXT,
    status TEXT NOT NULL,            -- scheduled | queued | failed | skipped
    created_at TEXT NOT NULL,
    updated_at TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS analytics_metrics (
    youtube_id TEXT PRIMARY KEY,
    views INTEGER, likes INTEGER, subscribers_gained INTEGER,
    stayed_to_watch REAL, average_view_duration REAL, average_percentage_viewed REAL,
    topic_category TEXT, video_duration REAL,
    imported_at TEXT NOT NULL, source_file TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    run_date TEXT NOT NULL,           -- YYYY-MM-DD in Asia/Kolkata
    slot INTEGER NOT NULL,            -- 1 = 09:00, 2 = 14:00, 3 = 19:00 IST
    key TEXT NOT NULL,                -- YYYY-MM-DD-slot-N idempotency key
    topic TEXT,
    stage TEXT NOT NULL,              -- script | review | verify | tts | footage | render | qc | upload
    error TEXT NOT NULL,              -- failure diagnostic
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    recoverable INTEGER NOT NULL DEFAULT 1,   -- 1 = retryable, 0 = permanent
    rendered INTEGER NOT NULL DEFAULT 0,      -- 1 = video rendered, 0 = not
    uploaded INTEGER NOT NULL DEFAULT 0,      -- 1 = uploaded/scheduled, 0 = not
    video_id INTEGER,                   -- videos.id if a video was rendered
    youtube_id TEXT,                    -- YouTube video ID if uploaded
    status TEXT NOT NULL DEFAULT 'pending', -- pending | retrying | completed | failed | skipped
    last_attempt_at TEXT,
    next_retry_at TEXT,                 -- when to retry next (ISO format)
    resolved_at TEXT,
    resolved_by TEXT                    -- how it was resolved: retry | manual | skipped
);
"""

# Columns added after 2.0; existing databases get them on open.
VIDEO_COLUMNS = {
    "score": "INTEGER",            # Script Doctor retention score, 0-100
    "parent_id": "INTEGER",        # the original video when this one is a translated copy
    "views": "INTEGER", "likes": "INTEGER", "comments": "INTEGER", "stats_at": "TEXT",
    "approval_status": "TEXT NOT NULL DEFAULT 'pending'",
    "approved_at": "TEXT",
    "approval_json": "TEXT",
    "research_json": "TEXT",
    "review_json": "TEXT",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class History:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._db() as c:
            c.executescript(SCHEMA)
            have = {r["name"] for r in c.execute("PRAGMA table_info(videos)")}
            for col, kind in VIDEO_COLUMNS.items():
                if col not in have:
                    c.execute(f"ALTER TABLE videos ADD COLUMN {col} {kind}")

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _exec(self, sql: str, args: tuple = ()) -> int:
        with self._db() as c:
            return c.execute(sql, args).lastrowid or 0

    def _all(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._db() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    # -- videos --
    def add_video(self, **fields) -> int:
        fields.setdefault("created_at", now_iso())
        if isinstance(fields.get("tags"), list):
            fields["tags"] = json.dumps(fields["tags"])
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        return self._exec(f"INSERT INTO videos ({cols}) VALUES ({marks})", tuple(fields.values()))

    def update_video(self, vid: int, **fields) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE videos SET {sets} WHERE id = ?", (*fields.values(), vid))

    def approve_video(self, vid: int, approved_at: str, approval: dict) -> bool:
        """Persist explicit per-video publication approval without changing render state."""
        with self._db() as c:
            row = c.execute(
                "SELECT status, youtube_id, approval_status FROM videos WHERE id = ?", (vid,)
            ).fetchone()
            if (not row or row["youtube_id"] or row["status"] in ("uploaded", "scheduled")
                    or row["approval_status"] == "approved"):
                return False
            c.execute(
                "UPDATE videos SET approval_status = 'approved', approved_at = ?, approval_json = ? WHERE id = ?",
                (approved_at, json.dumps(approval, ensure_ascii=False), vid),
            )
            return True

    def get(self, vid: int) -> dict | None:
        rows = self._all("SELECT * FROM videos WHERE id = ?", (vid,))
        return rows[0] if rows else None

    def recent(self, limit: int = 20) -> list[dict]:
        return self._all("SELECT * FROM videos ORDER BY id DESC LIMIT ?", (limit,))

    def delete_video(self, vid: int) -> None:
        self._exec("DELETE FROM videos WHERE id = ?", (vid,))

    def recent_titles(self, limit: int = 40) -> list[str]:
        return [r["title"] for r in self._all(
            "SELECT title FROM videos WHERE title IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,))]

    def pending_uploads(self, statuses: tuple[str, ...] = ("queued",)) -> list[dict]:
        """queued = deferred by the daily quota; rendered = made with --no-upload."""
        marks = ", ".join("?" for _ in statuses)
        return self._all(f"SELECT * FROM videos WHERE status IN ({marks}) AND video_path IS NOT NULL ORDER BY id",
                         statuses)

    def uploads_since(self, since_iso: str) -> int:
        return self._all("SELECT COUNT(*) AS n FROM videos WHERE uploaded_at >= ?", (since_iso,))[0]["n"]

    def scheduled_times(self) -> list[str]:
        return [r["publish_at"] for r in self._all(
            "SELECT publish_at FROM videos WHERE publish_at IS NOT NULL AND status = 'scheduled'")]

    def stats(self) -> dict:
        rows = self._all("SELECT status, COUNT(*) AS n FROM videos GROUP BY status")
        return {r["status"]: r["n"] for r in rows}

    # -- channel performance --
    def live_video_ids(self, limit: int = 500) -> list[str]:
        return [r["youtube_id"] for r in self._all(
            "SELECT youtube_id FROM videos WHERE youtube_id IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,))]

    def set_stats(self, youtube_id: str, views: int, likes: int, comments: int) -> None:
        self._exec("UPDATE videos SET views = ?, likes = ?, comments = ?, stats_at = ? WHERE youtube_id = ?",
                   (views, likes, comments, now_iso(), youtube_id))

    def performers(self, n: int = 5, *, best: bool = True, min_age_hours: int = 48) -> list[dict]:
        """Most (or least) viewed videos that have been public long enough for the numbers to mean something."""
        cutoff = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - min_age_hours * 3600,
                                        timezone.utc).isoformat(timespec="seconds")
        order = "DESC" if best else "ASC"
        return self._all(
            "SELECT id, title, style, views, likes, comments FROM videos WHERE views IS NOT NULL "
            f"AND COALESCE(publish_at, uploaded_at) <= ? ORDER BY views {order} LIMIT ?", (cutoff, n))

    def set_analytics_metrics(self, youtube_id: str, metrics: dict, source_file: str) -> None:
        columns = (
            "views", "likes", "subscribers_gained", "stayed_to_watch", "average_view_duration",
            "average_percentage_viewed", "topic_category", "video_duration",
        )
        values = [metrics.get(col) for col in columns]
        assignments = ", ".join(f"{col}=COALESCE(excluded.{col}, analytics_metrics.{col})" for col in columns)
        self._exec(
            "INSERT INTO analytics_metrics (youtube_id, views, likes, subscribers_gained, stayed_to_watch, "
            "average_view_duration, average_percentage_viewed, topic_category, video_duration, imported_at, "
            "source_file) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            f"ON CONFLICT(youtube_id) DO UPDATE SET {assignments}, "
            "imported_at=excluded.imported_at, source_file=excluded.source_file",
            (youtube_id, *values, now_iso(), source_file),
        )

    def analytics_metrics(self) -> list[dict]:
        return self._all(
            "SELECT am.*, v.title, v.topic, v.style, v.duration AS recorded_duration "
            "FROM analytics_metrics am LEFT JOIN videos v ON v.youtube_id = am.youtube_id "
            "ORDER BY am.imported_at DESC")

    # -- idea queue (planner) --
    def add_idea(self, subject: str, style: str = "", notes: str = "") -> int:
        return self._exec("INSERT INTO ideas (created_at, subject, style, notes) VALUES (?, ?, ?, ?)",
                          (now_iso(), subject.strip(), style or None, notes or None))

    def ideas(self, status: str | None = "pending", limit: int = 200) -> list[dict]:
        if status:
            return self._all("SELECT * FROM ideas WHERE status = ? ORDER BY id LIMIT ?", (status, limit))
        return self._all("SELECT * FROM ideas ORDER BY id DESC LIMIT ?", (limit,))

    def claim_idea(self) -> dict | None:
        """Take the oldest pending idea (atomically, so parallel workers never get the same one)."""
        with self._db() as c:
            row = c.execute("SELECT * FROM ideas WHERE status = 'pending' ORDER BY id LIMIT 1").fetchone()
            if not row:
                return None
            c.execute("UPDATE ideas SET status = 'used', used_at = ? WHERE id = ?", (now_iso(), row["id"]))
            return dict(row)

    def link_idea(self, idea_id: int, video_id: int) -> None:
        self._exec("UPDATE ideas SET video_id = ? WHERE id = ?", (video_id, idea_id))

    def release_idea(self, idea_id: int) -> None:
        """Put an idea back in the queue after its video failed."""
        self._exec("UPDATE ideas SET status = 'pending', used_at = NULL WHERE id = ?", (idea_id,))

    def delete_idea(self, idea_id: int) -> None:
        self._exec("DELETE FROM ideas WHERE id = ?", (idea_id,))

    # -- de-duplication --
    def used_media(self) -> set[str]:
        return {r["key"] for r in self._all("SELECT key FROM media_used")}

    def mark_media(self, keys: list[str]) -> None:
        ts = now_iso()
        with self._db() as c:
            c.executemany("INSERT OR IGNORE INTO media_used (key, used_at) VALUES (?, ?)", [(k, ts) for k in keys])

    def topic_used(self, key: str) -> bool:
        return bool(self._all("SELECT 1 FROM topics_used WHERE key = ?", (key.lower(),)))

    def mark_topic(self, key: str) -> None:
        self._exec("INSERT OR IGNORE INTO topics_used (key, used_at) VALUES (?, ?)", (key.lower(), now_iso()))

    # ------------------------------------------------------------------ slot publishing ledger
    @staticmethod
    def slot_key(run_date: str, slot: int) -> str:
        """Stable idempotency key for one date + slot, e.g. 2026-10-31-slot-1."""
        return f"{run_date}-slot-{int(slot)}"

    def is_slot_done(self, key: str) -> bool:
        """True if this slot was already successfully scheduled on YouTube."""
        rows = self._all(
            "SELECT 1 FROM slot_uploads WHERE key = ? AND status = 'scheduled'", (key,))
        return len(rows) > 0

    def get_slot(self, key: str) -> dict | None:
        rows = self._all("SELECT * FROM slot_uploads WHERE key = ?", (key,))
        return rows[0] if rows else None

    def record_slot(self, key: str, run_date: str, slot: int, publish_at_utc: str,
                    status: str, youtube_id: str | None = None, title: str | None = None,
                    error: str | None = None) -> None:
        """Insert or refresh a slot record. `status` is one of scheduled/queued/failed/skipped."""
        now = now_iso()
        existing = self.get_slot(key)
        if existing:
            self._exec(
                "UPDATE slot_uploads SET status = ?, updated_at = ?, youtube_id = ?, "
                "title = ?, error = ? WHERE key = ?",
                (status, now, youtube_id, title, error, key),
            )
            return
        self._exec(
            "INSERT INTO slot_uploads "
            "(key, run_date, slot, publish_at_utc, status, created_at, youtube_id, title, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, run_date, slot, publish_at_utc, status, now, youtube_id, title, error),
        )

    def list_slots_for_date(self, run_date: str) -> list[dict]:
        return self._all(
            "SELECT * FROM slot_uploads WHERE run_date = ? ORDER BY slot", (run_date,))

    # ------------------------------------------------------------------ recovery queue
    def record_failure(self, run_date: str, slot: int, key: str, topic: str | None,
                       stage: str, error: str, recoverable: bool = True,
                       rendered: bool = False, uploaded: bool = False,
                       video_id: int | None = None, youtube_id: str | None = None,
                       max_retries: int = 3) -> int:
        """Record a failed slot attempt in the recovery queue.

        Returns the recovery queue row id.
        """
        now = now_iso()
        return self._exec(
            """INSERT INTO recovery_queue
               (created_at, run_date, slot, key, topic, stage, error,
                retry_count, max_retries, recoverable, rendered, uploaded,
                video_id, youtube_id, status, next_retry_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (now, run_date, int(slot), key, topic, stage, error,
             0, max_retries, 1 if recoverable else 0,
             1 if rendered else 0, 1 if uploaded else 0,
             video_id, youtube_id, now),
        )

    def increment_retry(self, recovery_id: int) -> None:
        """Bump the retry count and schedule the next retry with backoff."""
        now = now_iso()
        self._exec(
            """UPDATE recovery_queue
               SET retry_count = retry_count + 1,
                   last_attempt_at = ?,
                   next_retry_at = ?,
                   status = 'retrying'
               WHERE id = ?""",
            (now, now, recovery_id),
        )

    def mark_recovery_completed(self, recovery_id: int, resolved_by: str,
                                youtube_id: str | None = None) -> None:
        """Mark a recovery queue entry as completed."""
        now = now_iso()
        fields = dict(status="completed", resolved_at=now, resolved_by=resolved_by)
        if youtube_id:
            fields["youtube_id"] = youtube_id
        sets = ", ".join(f"{k} = ?" for k in fields)
        self._exec(
            f"UPDATE recovery_queue SET {sets} WHERE id = ?",
            (*fields.values(), recovery_id),
        )

    def mark_recovery_failed(self, recovery_id: int, resolved_by: str = "exhausted") -> None:
        """Mark a recovery queue entry as permanently failed after exhausting retries."""
        now = now_iso()
        self._exec(
            "UPDATE recovery_queue SET status = 'failed', resolved_at = ?, resolved_by = ? WHERE id = ?",
            (now, resolved_by, recovery_id),
        )

    def pending_recoveries(self, limit: int = 50) -> list[dict]:
        """Return recovery queue entries that are due for retry.

        Only pending or retrying entries with next_retry_at <= now are returned.
        Permanently failed or completed entries are excluded.
        """
        now = now_iso()
        return self._all(
            """SELECT * FROM recovery_queue
               WHERE status IN ('pending', 'retrying')
                 AND recoverable = 1
                 AND retry_count < max_retries
                 AND (next_retry_at IS NULL OR next_retry_at <= ?)
               ORDER BY created_at ASC LIMIT ?""",
            (now, limit),
        )

    def get_recovery(self, recovery_id: int) -> dict | None:
        rows = self._all("SELECT * FROM recovery_queue WHERE id = ?", (recovery_id,))
        return rows[0] if rows else None

    def get_recovery_for_slot(self, key: str) -> dict | None:
        """Return the most recent pending recovery entry for a slot+date key."""
        rows = self._all(
            """SELECT * FROM recovery_queue
               WHERE key = ? AND status IN ('pending', 'retrying')
                 AND recoverable = 1
                 AND retry_count < max_retries
               ORDER BY created_at DESC LIMIT 1""",
            (key,),
        )
        return rows[0] if rows else None

    def recovery_stats(self) -> dict:
        rows = self._all(
            "SELECT status, COUNT(*) AS n FROM recovery_queue GROUP BY status")
        return {r["status"]: r["n"] for r in rows}
