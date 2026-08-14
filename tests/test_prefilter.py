from collections import Counter
from datetime import UTC, datetime

from community_shorts.models import Metrics, RawItem
from community_shorts.prefilter import prefilter


def item(
    item_id: str,
    source: str,
    likes: int,
    *,
    url: str | None = None,
    title: str | None = None,
    body: str = "useful discussion",
) -> RawItem:
    return RawItem(
        item_id=item_id,
        source_id=source,
        url=url or f"https://example.com/{item_id}",
        title=title or item_id,
        body=body,
        metrics=Metrics(likes=likes, comments=likes),
        source_language="en",
        fetched_at=datetime(2026, 8, 14, tzinfo=UTC),
    )


def test_prefilter_enforces_source_diversity() -> None:
    mixed = [
        item("a:1", "a", 100),
        item("a:2", "a", 90),
        item("a:3", "a", 80),
        item("b:1", "b", 30),
        item("b:2", "b", 20),
        item("b:3", "b", 10),
    ]

    selected = prefilter(mixed, global_limit=4, per_source_limit=2)

    assert Counter(entry.raw.source_id for entry in selected) == {"a": 2, "b": 2}


def test_prefilter_drops_tracking_duplicates_and_spam() -> None:
    candidates = [
        item("a:1", "a", 10, url="https://example.com/story?utm_source=x"),
        item("b:1", "b", 9, url="https://example.com/story#comments"),
        item("a:2", "a", 100, title="광고 도배", body="무료 코인 추천인 링크"),
    ]

    selected = prefilter(candidates, global_limit=10, per_source_limit=10)

    assert len(selected) == 1
    assert selected[0].raw.item_id in {"a:1", "b:1"}
