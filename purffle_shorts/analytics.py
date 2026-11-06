"""Channel learning loop: read the view, like and comment counts of the videos PurffleShorts uploaded,
and show the LLM which ones worked and which flopped, so new scripts lean toward what your audience
actually watches.

    LEARN_FROM_STATS=true
    YOUTUBE_API_KEY=...     optional: a plain API key reads public stats without touching your OAuth token.
                            Without it, run `purffle-shorts auth` once more to grant read access.
"""

from __future__ import annotations

import csv
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from .config import Settings
from .history import History
from .utils import http, raise_for_status, redact

log = logging.getLogger("purffle")

API = "https://www.googleapis.com/youtube/v3/videos"

CSV_FIELDS = {
    "views": ("views",),
    "likes": ("likes",),
    "subscribers_gained": ("subscribersgained",),
    "stayed_to_watch": ("stayedtowatch", "stayedtowatchpercent", "viewedvswipedaway",
                        "viewedvswipedawaypercent"),
    "average_view_duration": ("averageviewduration", "averageviewdurationseconds"),
    "average_percentage_viewed": ("averagepercentageviewed", "averageviewedpercentage"),
    "topic_category": ("topiccategory", "category"),
    "video_duration": ("videoduration", "duration"),
}
ID_FIELDS = ("videoid", "youtubeid", "contentid")


@dataclass(frozen=True)
class AnalyticsImportResult:
    imported: int
    skipped: int


def _column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _duration_seconds(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    if ":" in value:
        try:
            parts = [float(part) for part in value.split(":")]
        except ValueError:
            raise ValueError(f"invalid duration {value!r}") from None
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + part
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"expected a finite non-negative duration, got {value!r}")
        return seconds
    return _number(value)


def _number(value: str, *, integer: bool = False) -> int | float | None:
    value = value.strip().replace(",", "").replace("%", "")
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"invalid number {value!r}") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"expected a finite non-negative number, got {value!r}")
    if integer and not number.is_integer():
        raise ValueError(f"expected a whole number, got {value!r}")
    return int(number) if integer else number


def import_csv(path: str | Path, history: History) -> AnalyticsImportResult:
    """Import an optional YouTube Studio CSV export; no Analytics API scope is assumed."""
    source = Path(path)
    skipped = 0
    rows = []
    with source.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{source}: CSV is empty or has no header row")
        columns = {_column_name(name): name for name in reader.fieldnames if name}
        id_column = next((columns[name] for name in ID_FIELDS if name in columns), None)
        if not id_column:
            raise ValueError(f"{source}: include a Video ID column so metrics map unambiguously")
        metric_columns = {
            field: next((columns[name] for name in aliases if name in columns), None)
            for field, aliases in CSV_FIELDS.items()
        }
        for line, row in enumerate(reader, start=2):
            youtube_id = (row.get(id_column) or "").strip()
            if not youtube_id:
                skipped += 1
                continue
            metrics = {}
            for field_name, column in metric_columns.items():
                if not column:
                    continue
                raw = row.get(column) or ""
                try:
                    if field_name == "topic_category":
                        metrics[field_name] = raw.strip() or None
                    elif field_name in {"average_view_duration", "video_duration"}:
                        metrics[field_name] = _duration_seconds(raw)
                    else:
                        value = _number(raw, integer=field_name in {"views", "likes", "subscribers_gained"})
                        if field_name == "stayed_to_watch" and value is not None and value > 100:
                            raise ValueError(f"percentage must be between 0 and 100, got {raw!r}")
                        metrics[field_name] = value
                except ValueError as exc:
                    raise ValueError(f"{source}:{line}: {field_name}: {exc}") from None
            rows.append((youtube_id, metrics))
    for youtube_id, metrics in rows:
        history.set_analytics_metrics(youtube_id, metrics, source.name)
    imported = len(rows)
    if imported:
        log.info("Imported analytics for %d video(s) from %s", imported, source.name)
    else:
        log.warning("No analytics rows with a Video ID were found in %s", source.name)
    return AnalyticsImportResult(imported=imported, skipped=skipped)


def _fetch_with_key(ids: list[str], key: str) -> list[dict]:
    r = http().get(API, params={"part": "statistics", "id": ",".join(ids), "key": key}, timeout=30)
    raise_for_status(r, "YouTube stats")
    return r.json().get("items", [])


def _fetch_with_oauth(ids: list[str], settings: Settings) -> list[dict]:
    from . import youtube
    yt = youtube.service(settings, interactive=False, extra_scopes=[youtube.SCOPE_READONLY])
    return yt.videos().list(part="statistics", id=",".join(ids), maxResults=50).execute().get("items", [])


def sync_stats(settings: Settings, history: History) -> int:
    """Refresh stats for every uploaded video (1 quota unit per 50 videos). Returns how many were updated."""
    ids = history.live_video_ids()
    updated = 0
    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50]
        try:
            items = (_fetch_with_key(batch, settings.youtube_api_key) if settings.youtube_api_key
                     else _fetch_with_oauth(batch, settings))
        except Exception as e:
            log.warning("Could not read channel stats: %s", redact(e))
            break
        for it in items:
            st = it.get("statistics") or {}
            history.set_stats(it.get("id", ""), int(st.get("viewCount") or 0), int(st.get("likeCount") or 0),
                              int(st.get("commentCount") or 0))
            updated += 1
    if updated:
        log.info("Channel stats refreshed for %d video(s)", updated)
    return updated


def insights(history: History, n: int = 5) -> str:
    """A prompt block with available channel results and optional imported retention baselines."""
    best = history.performers(n, best=True)
    imported = history.analytics_metrics()
    baseline_fields = (
        ("views", "Views"),
        ("subscribers_gained", "Subscribers gained"),
        ("stayed_to_watch", "Stayed to watch"),
        ("average_view_duration", "Average view duration"),
        ("average_percentage_viewed", "Average percentage viewed"),
    )
    baselines = []
    for key, label in baseline_fields:
        values = [row[key] for row in imported if row.get(key) is not None]
        if len(values) >= 3:
            value = median(values)
            formatted = (f"{value:.1f}s" if key == "average_view_duration"
                         else f"{value:.1f}%" if key in {"stayed_to_watch", "average_percentage_viewed"}
                         else f"{value:,.0f}")
            baselines.append(f"{label}: {formatted} (n={len(values)})")
    if len(best) < 3 and not baselines:
        return ""

    def row(v: dict) -> str:
        return f'- "{v["title"]}" ({v["style"] or "?"}): {v["views"]:,} views, {v["likes"] or 0:,} likes'
    sections = []
    if len(best) >= 3:
        worst = [w for w in history.performers(n, best=False) if w["id"] not in {b["id"] for b in best}]
        sections.append("What performed best on this channel so far:\n" + "\n".join(row(v) for v in best))
        if worst:
            sections.append("What performed worst:\n" + "\n".join(row(v) for v in worst[:3]))
        sections.append("Use these as directional evidence for hooks and subjects; don't copy their titles.")
    if baselines:
        sections.append("Channel's own historical retention baseline (median; descriptive, not a target):\n"
                        + "\n".join(f"- {value}" for value in baselines))
        categories: dict[str, list[float]] = {}
        for item in imported:
            category = item.get("topic_category") or item.get("topic") or item.get("style")
            value = item.get("average_percentage_viewed")
            if category and value is not None:
                categories.setdefault(category, []).append(value)
        category_medians = [
            (median(values), category, len(values))
            for category, values in categories.items() if len(values) >= 3
        ]
        if category_medians:
            value, category, count = max(category_medians)
            sections.append(f"Highest observed topic category by median average percentage viewed: "
                            f"{category} ({value:.1f}%, n={count}); treat as a small-sample pattern, not causation.")
        sections.append("Use observed patterns cautiously; the metrics do not prove a topic or script caused results.")
    text = "\n".join(sections)
    return text
