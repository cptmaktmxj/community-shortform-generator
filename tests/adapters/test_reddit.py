from datetime import UTC, datetime

from community_shorts.adapters.reddit import parse_listing
from community_shorts.config import SourceConfig


SOURCE = SourceConfig(
    source_id="reddit_ai_agents",
    adapter="reddit",
    url="https://www.reddit.com/r/AI_Agents/",
    language="en",
    subreddit="AI_Agents",
)


def test_reddit_parser_marks_english_and_ignores_stickied_posts(fixture_json) -> None:
    items = parse_listing(
        fixture_json("reddit_listing.json"),
        source=SOURCE,
        since=datetime(2026, 8, 14, tzinfo=UTC),
        fetched_at=datetime(2026, 8, 14, 3, tzinfo=UTC),
    )

    assert [item.item_id for item in items] == ["reddit_ai_agents:abc"]
    assert items[0].metrics.likes == 21
    assert items[0].metrics.comments == 8
    assert items[0].source_language == "en"
