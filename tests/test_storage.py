import json
from datetime import UTC, datetime
from pathlib import Path

from community_shorts.models import RawItem
from community_shorts.storage import ArtifactStore


def make_item(item_id: str) -> RawItem:
    return RawItem(
        item_id=item_id,
        source_id="geeknews",
        url=f"https://news.hada.io/topic?id={item_id.split(':')[-1]}",
        title=f"제목 {item_id}",
        body="본문",
        source_language="ko",
        fetched_at=datetime(2026, 8, 14, tzinfo=UTC),
    )


def test_write_items_merges_by_id_and_preserves_korean(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.write_items([make_item("geeknews:1")])
    store.write_items([make_item("geeknews:2"), make_item("geeknews:1")])

    items = store.read_items()

    assert [item.item_id for item in items] == ["geeknews:1", "geeknews:2"]
    raw = (tmp_path / "items.json").read_text(encoding="utf-8")
    assert "제목" in raw
    assert not list(tmp_path.glob("*.tmp"))
    assert len(json.loads(raw)) == 2
