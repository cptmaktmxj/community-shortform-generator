from datetime import UTC, date, datetime
from pathlib import Path

from community_shorts.state import StateStore
from tests.test_storage import make_item


def test_state_store_records_ingested_and_curated_items(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.sqlite")
    item = make_item("geeknews:1")

    state.mark_ingested([item], at=datetime(2026, 8, 14, 1, tzinfo=UTC))
    state.mark_curated([item.item_id], at=datetime(2026, 8, 14, 2, tzinfo=UTC))

    assert state.seen_ids("geeknews") == {"geeknews:1"}
    assert state.count_curated_on(date(2026, 8, 14)) == 1
