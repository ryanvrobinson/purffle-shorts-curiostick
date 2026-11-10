"""Content planner: the LLM brainstorms a batch of specific, non-repeating video ideas for your niches
and puts them in the idea queue. The autopilot always makes queued ideas first, so a week of videos can
be planned (and pruned) in one go:

    purffle-shorts plan --count 14 --niche "space and the universe"
"""

from __future__ import annotations

import logging
import random

from .config import Settings
from .history import History
from .script import LANGUAGES, STYLES
from .utils import slugify

log = logging.getLogger("purffle")

# A CurioStick week: 3 slots per day x 7 days = 21 Shorts, arranged as interleaved
# multi-part series so no series publishes out of order and no episode is skipped.
SEVEN_DAY_DAYS = 7
SLOTS_PER_DAY = 3


IDEAS_SCHEMA = {
    "type": "object",
    "properties": {
        "ideas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "style": {"type": "string", "enum": list(STYLES)},
                    "angle": {"type": "string"},
                },
                "required": ["subject", "style", "angle"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["ideas"],
    "additionalProperties": False,
}


def plan_prompt(settings: Settings, count: int, niches: list[str], avoid: list[str], insights: str = "") -> tuple[str, str]:
    lang = LANGUAGES.get(settings.language, settings.language)
    formats = ", ".join(f"{k} ({v.split(':')[0].split('.')[0].lower()})" for k, v in STYLES.items())
    system = ("You are the content strategist of a fast-growing faceless YouTube Shorts channel. You reply with "
              "a single JSON object and nothing else.")
    avoid_block = ("Already made or planned (never repeat these or near-duplicates):\n"
                   + "\n".join(f"- {a}" for a in avoid[:80]) + "\n\n") if avoid else ""
    ins = f"{insights}\n\n" if insights else ""
    user = f"""Plan {count} YouTube Shorts for a channel about: {"; ".join(niches)}.
Audience: {settings.audience}. Language of the videos: {lang} (write subject and angle in English).

{ins}{avoid_block}Each idea:
- subject: one specific, surprising, searchable topic (not a broad theme), e.g. "why octopuses have blue blood".
- style: the format that fits it best, one of: {formats}. Mix formats; use each at most {max(2, count // 3)} times.
- angle: one sentence with the hook or twist that makes people stop scrolling.
Prefer evergreen topics with real facts behind them. Return JSON: {{"ideas": [{{subject, style, angle}}, ...]}}."""
    return system, user


def plan_ideas(llm, settings: Settings, history: History, count: int = 10, niches: list[str] | None = None,
               insights: str = "") -> list[dict]:
    count = max(1, min(50, count))
    niches = niches or random.sample(settings.niches, min(3, len(settings.niches))) or ["interesting facts"]
    avoid = history.recent_titles(60) + [i["subject"] for i in history.ideas("pending")]
    system, user = plan_prompt(settings, count, niches, avoid, insights)
    data = llm.complete_json(system, user, IDEAS_SCHEMA, max_tokens=6000)
    seen = {a.lower() for a in avoid}
    saved = []
    for idea in data.get("ideas") or []:
        if len(saved) >= count:
            break
        if not isinstance(idea, dict):
            continue
        subject = str(idea.get("subject") or "").strip()
        if len(subject) < 4 or subject.lower() in seen:
            continue
        seen.add(subject.lower())
        style = idea.get("style") if idea.get("style") in STYLES else ""
        angle = str(idea.get("angle") or "").strip()
        iid = history.add_idea(subject, style, angle)
        saved.append({"id": iid, "subject": subject, "style": style, "notes": angle})
    log.info("Planner added %d idea(s) to the queue", len(saved))
    return saved


SEVEN_DAY_SCHEMA = {
    "type": "object",
    "properties": {
        "series": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "series_id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "episodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "episode_number": {"type": "integer"},
                                "headline": {"type": "string"},
                                "angle": {"type": "string"},
                            },
                            "required": ["episode_number", "headline", "angle"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["series_id", "title", "episodes"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["series"],
    "additionalProperties": False,
}


def _seven_day_prompt(settings: Settings, days: int, n_series: int,
                      episodes_per_series: int, avoid: list[str],
                      existing_series: list[str]) -> tuple[str, str]:
    lang = LANGUAGES.get(settings.language, settings.language)
    niches = "; ".join(settings.niches or [])
    total = days * SLOTS_PER_DAY
    avoid_block = ("\nAlready used or planned (never repeat or near-duplicate):\n"
                   + "\n".join(f"- {a}" for a in avoid[:70]) + "\n") if avoid else ""
    existing_block = ("\nSeries ids already in use (do not reuse):\n"
                      + "\n".join(f"- {s}" for s in existing_series) + "\n") if existing_series else ""
    user = f"""Plan a seven-day CurioStick Shorts schedule: {days} days x {SLOTS_PER_DAY} slots = {total} Shorts.
Audience: {settings.audience}. Language of the videos: {lang}.
CurioStick niches: {niches}.

{avoid_block}{existing_block}Produce exactly {n_series} multi-part series, each with {episodes_per_series} parts, for {total} Shorts total.
Every Short is one episode of a series and must make sense on its own.

Rules:
- Each headline is specific, surprising, researchable, and complete without the other parts.
- Parts of a series form a progression: Part 1 opens the mystery, Part 2 deepens the evidence, Part 3 delivers the reveal.
- No invented numbers, statistics, studies, or quotes. Frame uncertain material as a question or hypothesis, never as fact.
- Each series_id is a short slug (e.g. "chicxulub-crater", "blood-falls"); all distinct and not in the "already in use" list.
- Return exactly {n_series} series, each with exactly {episodes_per_series} episodes numbered 1..{episodes_per_series}.
Return JSON: {{"series": [{{"series_id", "title", "description", "episodes": [{{"episode_number", "headline", "angle"}}]}}]}}."""
    system = ("You are a YouTube Shorts content strategist for a mystery/science channel. "
              "Reply with a single JSON object and nothing else.")
    return system, user


def plan_seven_day(llm, settings: Settings, history: History,
                   days: int = SEVEN_DAY_DAYS,
                   episodes_per_series: int = 3) -> list[dict]:
    """Plan a full week: ``days`` x 3 slots = 21 Shorts as interleaved multi-part series.

    Stores each series and episode in the history database and queues every episode
    as a pending idea, ordered so ``claim_idea`` never publishes Part N+1 before Part N.
    Returns the planned episodes with day/slot assignment.
    """
    total = days * SLOTS_PER_DAY
    n_series = total // episodes_per_series
    avoid = history.recent_titles(80) + [i["subject"] for i in history.ideas("pending")]
    existing = sorted(s["series_id"] for s in history.all_series())
    system, user = _seven_day_prompt(settings, days, n_series, episodes_per_series,
                                     avoid, existing)
    data = llm.complete_json(system, user, SEVEN_DAY_SCHEMA, max_tokens=9000)

    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(settings.timezone or "Asia/Kolkata")
    base = datetime.now(tz).date()

    series_list = []
    seen_ids = set(existing)
    for s in (data.get("series") or []):
        if not isinstance(s, dict):
            continue
        sid = str(s.get("series_id") or "").strip()
        if not sid or sid in seen_ids:
            continue
        seen_ids.add(sid)
        eps_by_num: dict[int, dict] = {}
        for ep in (s.get("episodes") or []):
            if isinstance(ep, dict) and ep.get("episode_number"):
                eps_by_num[int(ep["episode_number"])] = ep
        series_list.append((sid, str(s.get("title") or sid),
                            str(s.get("description") or ""), eps_by_num))
        if len(series_list) >= n_series:
            break

    # Fill any shortfall from default niches so the week is always complete.
    filler = 0
    while len(series_list) < n_series:
        sid = f"curios-{slugify(settings.niches[filler % max(1, len(settings.niches))])}-{filler}"
        if sid in seen_ids:
            filler += 1
            continue
        seen_ids.add(sid)
        series_list.append((sid, settings.niches[filler % max(1, len(settings.niches))], "", {}))
        filler += 1

    saved = []
    for g in range(total):
        day = g // SLOTS_PER_DAY + 1
        slot = g % SLOTS_PER_DAY + 1
        run_date = (base + timedelta(days=day - 1)).isoformat()
        sid, title, desc, eps_by_num = series_list[g % n_series]
        ep_num = g // n_series + 1
        if ep_num > episodes_per_series:
            continue
        ep = eps_by_num.get(ep_num)
        headline = str(ep.get("headline") or title).strip() if ep else title
        angle = str(ep.get("angle") or "").strip() if ep else ""
        history.add_series(sid, title, desc, episodes_per_series)
        history.add_series_episode(sid, ep_num, headline, slot=slot, run_date=run_date)
        iid = history.add_idea(headline, style="", notes=angle,
                               series_id=sid, episode_number=ep_num)
        saved.append({
            "id": iid, "series_id": sid, "series_title": title,
            "episode_number": ep_num, "day": day, "slot": slot,
            "run_date": run_date, "headline": headline, "angle": angle,
        })
    history._exec("UPDATE series SET status = 'active' WHERE status = 'planned'")
    log.info("Seven-day plan: %d series, %d episode(s)", len(series_list), len(saved))
    return saved
