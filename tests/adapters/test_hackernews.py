from datetime import UTC, datetime

from community_shorts.adapters.hackernews import parse_items


def test_hackernews_parser_keeps_recent_stories_and_marks_english(fixture_json) -> None:
    items = parse_items(
        fixture_json("hackernews_items.json"),
        since=datetime(2026, 8, 14, tzinfo=UTC),
        fetched_at=datetime(2026, 8, 14, 3, tzinfo=UTC),
    )

    assert [item.item_id for item in items] == ["hackernews:100"]
    assert items[0].metrics.likes == 12
    assert items[0].metrics.comments == 4
    assert items[0].source_language == "en"
