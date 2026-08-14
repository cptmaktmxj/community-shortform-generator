import json
from datetime import UTC, datetime
from pathlib import Path

from community_shorts.models import CuratedItem, RawItem
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


def make_curated(item_id: str) -> CuratedItem:
    return CuratedItem(
        item_id=item_id,
        source_id="geeknews",
        url=f"https://news.hada.io/topic?id={item_id.split(':')[-1]}",
        source_language="ko",
        output_language="ko",
        summary="한국어 요약입니다.",
        key_claim="검증된 핵심 주장입니다.",
        hook_points=["대중이 이해할 수 있는 변화"],
        tone="정보형",
        pass_=True,
        reaction_score=0.8,
        provocation_score=0.7,
        provocation_band="clear_disruption",
        provocation_reason="업무 방식에 뚜렷한 변화를 만듭니다.",
        mass_appeal_score=0.7,
        mass_appeal_band="direct_impact",
        mass_appeal_reason="직장인의 시간과 업무에 직접 영향을 줍니다.",
        curation_score=0.72,
        fidelity_score=0.9,
        safety_ok=True,
        safety_reason="안전한 기술 뉴스입니다.",
        safety_categories=[],
        curation_reason="새 타깃 기준을 충족합니다.",
        model="fixture",
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


def test_replace_curated_does_not_merge_old_results(tmp_path: Path) -> None:
    """Catch explicit rebuilds retaining selections from the old score formula."""

    store = ArtifactStore(tmp_path)
    store.write_curated([make_curated("geeknews:1")])

    store.replace_curated([make_curated("geeknews:2")])

    assert [item.item_id for item in store.read_curated()] == ["geeknews:2"]
    assert not list(tmp_path.glob("*.tmp"))
