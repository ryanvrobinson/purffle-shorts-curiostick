"""Merge remote CurioStick slot history into the local SQLite database.

Used by the three slot GitHub Actions workflows when pushing publishing
history to ``main``. Remote slot runs only append to ``data/history.db``
(idempotent by ``date + slot`` key; each slot owns its key), so this script
merges remote ``slot_uploads`` rows into the local database without
overwriting the SQLite file wholesale.

Usage:
    python3 scripts/merge_slot_history.py <remote_db_path> <local_db_path>
"""

from __future__ import annotations

import sqlite3
import sys

COLUMNS = (
    "key, run_date, slot, publish_at_utc, youtube_id, title, status, "
    "created_at, updated_at, error"
)


def merge(remote_path: str, local_path: str) -> int:
    """Insert or replace remote ``slot_uploads`` rows into the local DB.

    Returns the number of rows merged.
    """
    rcon = sqlite3.connect(remote_path)
    lcon = sqlite3.connect(local_path)
    try:
        rows = rcon.execute(f"SELECT {COLUMNS} FROM slot_uploads").fetchall()
        merged = 0
        for row in rows:
            lcon.execute(
                f"""
                INSERT INTO slot_uploads ({COLUMNS})
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    run_date = excluded.run_date,
                    slot = excluded.slot,
                    publish_at_utc = excluded.publish_at_utc,
                    youtube_id = excluded.youtube_id,
                    title = excluded.title,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    error = excluded.error
                """,
                row,
            )
            merged += 1
        lcon.commit()
        return merged
    finally:
        rcon.close()
        lcon.close()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: merge_slot_history.py <remote_db> <local_db>", file=sys.stderr
        )
        return 2
    remote_path, local_path = argv[1], argv[2]
    try:
        merged = merge(remote_path, local_path)
        print(f"merged {merged} remote slot history rows into {local_path}")
        return 0
    except sqlite3.Error as exc:
        print(f"error: sqlite merge failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
