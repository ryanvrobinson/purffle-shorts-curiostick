import pytest

from purffle_shorts import analytics
from purffle_shorts.history import History


HEADER = (
    "Video ID,Views,Likes,Subscribers gained,Stayed to watch (%),Average view duration,"
    "Average percentage viewed (%),Topic category,Video duration\n"
)


def test_import_csv_retention_metrics_and_relative_baselines(tmp_path):
    history = History(tmp_path / "history.db")
    rows = [
        "video-a,1200,80,5,62.5%,0:00:24,82.0%,Earth mysteries,0:00:30",
        "video-b,900,60,3,51%,0:00:20,70%,Earth mysteries,0:00:28",
        "video-c,2400,190,9,75%,0:00:31,95%,Earth mysteries,0:00:32",
        ",100,4,0,50%,0:00:10,40%,,0:00:15",
    ]
    path = tmp_path / "studio.csv"
    path.write_text(HEADER + "\n".join(rows), encoding="utf-8")

    result = analytics.import_csv(path, history)

    assert (result.imported, result.skipped) == (3, 1)
    imported = {row["youtube_id"]: row for row in history.analytics_metrics()}
    assert imported["video-a"]["average_view_duration"] == 24
    assert imported["video-a"]["average_percentage_viewed"] == 82
    assert imported["video-a"]["stayed_to_watch"] == 62.5
    assert imported["video-a"]["subscribers_gained"] == 5
    insights = analytics.insights(history)
    assert "Channel's own historical retention baseline" in insights
    assert "82.0%" in insights
    assert "Earth mysteries" in insights
    assert "not a target" in insights


def test_import_csv_updates_only_metrics_present_in_export(tmp_path):
    history = History(tmp_path / "history.db")
    full = tmp_path / "full.csv"
    full.write_text(HEADER + "video-a,1200,80,5,62.5%,0:00:24,82%,Earth,0:00:30\n", encoding="utf-8")
    analytics.import_csv(full, history)

    partial = tmp_path / "partial.csv"
    partial.write_text("Video ID,Average view duration\nvideo-a,0:00:27\n", encoding="utf-8")
    analytics.import_csv(partial, history)

    row = history.analytics_metrics()[0]
    assert row["average_view_duration"] == 27
    assert row["views"] == 1200
    assert row["average_percentage_viewed"] == 82


def test_import_csv_requires_video_ids_and_does_not_partially_import(tmp_path):
    history = History(tmp_path / "history.db")
    no_ids = tmp_path / "no-ids.csv"
    no_ids.write_text("Title,Views\nExample,100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Video ID"):
        analytics.import_csv(no_ids, history)

    invalid = tmp_path / "invalid.csv"
    invalid.write_text("Video ID,Views\nvalid-id,100\nbad-id,not-a-number\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid.csv:3: views"):
        analytics.import_csv(invalid, history)
    assert history.analytics_metrics() == []
