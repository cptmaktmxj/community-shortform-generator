from datetime import UTC, datetime

from community_shorts.adapters.dcinside import parse_listing
from community_shorts.config import SourceConfig


SOURCE = SourceConfig(
    source_id="dcinside_singularity",
    adapter="dcinside",
    url="https://gall.dcinside.com/mgallery/board/lists/?id=thesingularity",
    language="ko",
    gallery_id="thesingularity",
)


def test_dcinside_parser_maps_recommendations_and_comments(fixture_text) -> None:
    items = parse_listing(
        fixture_text("dcinside_list.html"),
        source=SOURCE,
        fetched_at=datetime(2026, 8, 14, 3, tzinfo=UTC),
    )

    assert [item.item_id for item in items] == ["dcinside_singularity:777"]
    assert items[0].metrics.likes == 31
    assert items[0].metrics.comments == 12
    assert str(items[0].url).endswith("id=thesingularity&no=777")
