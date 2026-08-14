from datetime import UTC, datetime

from community_shorts.adapters.geeknews import parse_comments, parse_listing


NOW = datetime(2026, 8, 14, 3, 0, tzinfo=UTC)


def test_geeknews_parser_maps_points_comments_and_topic_id(fixture_text) -> None:
    items = parse_listing(fixture_text("geeknews_new.html"), fetched_at=NOW)

    assert len(items) == 1
    assert items[0].item_id == "geeknews:12345"
    assert items[0].metrics.likes == 42
    assert items[0].metrics.comments == 7
    assert str(items[0].url) == "https://news.hada.io/topic?id=12345"
    assert items[0].title == "테스트 기술 소식"
    assert items[0].source_language == "ko"


def test_geeknews_comment_parser_keeps_top_three_by_likes(fixture_text) -> None:
    comments = parse_comments(fixture_text("geeknews_topic.html"), limit=3)

    assert [(comment.text, comment.likes) for comment in comments] == [
        ("첫 댓글", 9),
        ("두 번째 댓글", 3),
        ("세 번째 댓글", 1),
    ]
