"""Seven-day, 21-Headline series plan tests. No network, fake LLM."""

from datetime import date, datetime, timedelta

import pytest

from purffle_shorts.config import Settings
from purffle_shorts.history import History
from purffle_shorts.planner import SEVEN_DAY_DAYS, SLOTS_PER_DAY, plan_seven_day


class FakeLLM:
    def __init__(self, series):
        self.series = series
        self.calls = 0

    def complete_json(self, system, user, schema=None, max_tokens=4000):
        self.calls += 1
        assert schema is not None
        return {"series": self.series}


def _series(sid, title, episodes=3):
    return {
        "series_id": sid,
        "title": title,
        "description": f"{title} series",
        "episodes": [
            {"episode_number": n,
             "headline": f"{title} part {n}",
             "angle": f"angle for part {n}"}
            for n in range(1, episodes + 1)
        ],
    }


def _seven_series():
    return [_series(f"series-{i}", f"Series {i}") for i in range(SEVEN_DAY_DAYS)]


def test_seven_day_plan_has_21_distinct_headlines(tmp_path, monkeypatch):
    history = History(tmp_path / "history.db")
    llm = FakeLLM(_seven_series())
    saved = plan_seven_day(llm, Settings(), history)

    assert llm.calls == 1
    assert len(saved) == 21, len(saved)
    headlines = [e["headline"] for e in saved]
    assert len(set(headlines)) == 21  # all distinct
    # 7 series x 3 parts
    series_ids = {e["series_id"] for e in saved}
    assert len(series_ids) == 7
    # every slot/day covered exactly once
    days = [(e["day"], e["slot"]) for e in saved]
    assert len(set(days)) == 21
    assert all(1 <= e["day"] <= 7 for e in saved)
    assert all(1 <= e["slot"] <= 3 for e in saved)
    # each series has episodes 1,2,3 in order across the week
    for sid in series_ids:
        eps = sorted([e for e in saved if e["series_id"] == sid], key=lambda e: e["episode_number"])
        assert [e["episode_number"] for e in eps] == [1, 2, 3]
        assert eps[0]["day"] < eps[1]["day"] < eps[2]["day"]


def test_claim_idea_respects_series_order(tmp_path):
    history = History(tmp_path / "history.db")
    history.add_series("s1", "Series One")
    history.add_series_episode("s1", 1, "Part one headline", slot=1, run_date="2026-11-01")
    history.add_series_episode("s1", 2, "Part two headline", slot=2, run_date="2026-11-01")
    history.add_series_episode("s1", 3, "Part three headline", slot=3, run_date="2026-11-01")
    # queue all three as pending ideas
    history.add_idea("Part one headline", series_id="s1", episode_number=1)
    history.add_idea("Part two headline", series_id="s1", episode_number=2)
    history.add_idea("Part three headline", series_id="s1", episode_number=3)

    first = history.claim_idea()
    assert first["episode_number"] == 1
    second = history.claim_idea()
    assert second["episode_number"] == 2
    third = history.claim_idea()
    assert third["episode_number"] == 3
    assert history.claim_idea() is None


def test_next_due_episode_never_skips_or_reorders(tmp_path):
    history = History(tmp_path / "history.db")
    history.add_series("s2", "Series Two")
    for n in (1, 2, 3):
        history.add_series_episode("s2", n, f"ep {n}")
    # mark part 2 produced, part 1 still pending -> next due is part 1 (predecessor of 2 is 1, pending)
    history.mark_episode_status("s2", 2, "produced")
    assert history.next_due_episode("s2")["episode_number"] == 1
    # mark part 1 produced too -> part 3 now due (parts 1 and 2 complete)
    history.mark_episode_status("s2", 1, "produced")
    assert history.next_due_episode("s2")["episode_number"] == 3
    # nothing produced -> part 1 due
    history2 = History(tmp_path / "history2.db")
    history2.add_series("s3", "Series Three")
    for n in (1, 2, 3):
        history2.add_series_episode("s3", n, f"ep {n}")
    assert history2.next_due_episode("s3")["episode_number"] == 1


def test_plan_overview_shape(tmp_path):
    history = History(tmp_path / "history.db")
    llm = FakeLLM(_seven_series())
    plan_seven_day(llm, Settings(), history)
    overview = history.plan_overview()
    assert overview["total_series"] == 7
    assert all(s["episodes"] == 3 for s in overview["series"])
    assert all(s["produced"] == 0 for s in overview["series"])


def test_plan_does_not_duplicate_existing_series(tmp_path, monkeypatch):
    history = History(tmp_path / "history.db")
    history.add_series("series-0", "Already Planned")
    llm = FakeLLM(_seven_series())  # includes series-0 again
    saved = plan_seven_day(llm, Settings(), history)
    series_ids = {e["series_id"] for e in saved}
    assert "series-0" not in series_ids  # not re-added
    assert len(saved) == 21  # filler series completed the week


def test_plan_dates_are_consecutive_days(tmp_path):
    history = History(tmp_path / "history.db")
    llm = FakeLLM(_seven_series())
    saved = plan_seven_day(llm, Settings(), history)
    from zoneinfo import ZoneInfo
    base = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    expected = {(base + timedelta(days=d)).isoformat(): 3 for d in range(SEVEN_DAY_DAYS)}
    actual = {}
    for e in saved:
        actual[e["run_date"]] = actual.get(e["run_date"], 0) + 1
    assert actual == expected
