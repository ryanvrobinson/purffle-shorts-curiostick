# CurioStick artifact retention

The three CurioStick GitHub Actions workflows (slots 1/2/3) upload render outputs —
MP4s, scripts, subtitles, attribution/credits and failure reports — as workflow
artifacts with a **30-day retention** (`retention-days: 30`).

## Why 30 days

`KEEP_VIDEOS=true` means each slot run keeps its rendered MP4 alongside the script,
captions and metadata so a failed or missed slot can be recovered later without
re-rendering from scratch. Thirty days covers a full recovery window (a slot missed
on Monday can still be caught up the following week) while staying within the free
GitHub Actions storage budget.

## Limitation

Artifact retention is bounded by GitHub Actions storage limits for the repository.
Once a workflow run passes its 30-day retention, its artifacts (including the MP4)
are automatically deleted by GitHub and cannot be recovered from Actions. This is
a hard platform limit; no paid storage is introduced by this project.

## What is NOT kept in Git

Rendered MP4s are **never committed into Git history**. Only the history database
(`data/history.db`, git-ignored), small text reports and source code are versioned.
The publishing ledger (`slot_uploads`) and recovery queue live in `data/history.db`
and are pushed back to `main` by each slot run so every slot's history survives
without force-pushing or overwriting another slot's records.
