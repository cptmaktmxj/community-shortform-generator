import json
from datetime import UTC, datetime
from pathlib import Path

from community_shorts.generation_models import ContentAnalysis, GeneratedScript
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


def make_generated(item_id: str) -> GeneratedScript:
    """Build a complete public Stage 3 artifact for persistence tests."""

    return GeneratedScript(
        item_id=item_id,
        source_id="geeknews",
        source_url=f"https://news.hada.io/topic?id={item_id.split(':')[-1]}",
        analysis=ContentAnalysis(
            topic_category="work_productivity",
            audience_relevance="일반 직장인의 반복 업무에 직접 관련됩니다.",
            core_facts=["새 기능은 반복 업무 단계를 자동화합니다."],
            angle="일상 업무에서 달라지는 시간을 설명합니다.",
            hook_strategy="기존 방식과 달라진 점을 먼저 제시합니다.",
            claims_to_avoid=["확인되지 않은 성과 수치"],
            recommended_tone="빠르고 명료한 정보형",
        ),
        script="최종 검증된 한국어 대본입니다.",
        estimated_duration_seconds=45.0,
        duration_class="ideal",
        playback_speed=1.2,
        title_candidates=[
            "직장인의 반복 업무를 바꾸는 AI",
            "AI가 반복 업무를 정말 줄여줄까",
            "AI는 답변보다 반복 업무부터 바꾼다",
        ],
        selected_title="직장인의 반복 업무를 바꾸는 AI",
        model="gpt-5.4-mini",
        generated_at=datetime(2026, 8, 14, tzinfo=UTC),
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


def test_scripts_artifact_merges_normally_and_replaces_explicitly(
    tmp_path: Path,
) -> None:
    """Catch normal resumes losing output or rebuilds retaining stale scripts."""

    store = ArtifactStore(tmp_path)
    store.write_scripts([make_generated("geeknews:1")])
    store.write_scripts([make_generated("geeknews:2")])
    assert {item.item_id for item in store.read_scripts()} == {
        "geeknews:1",
        "geeknews:2",
    }

    store.replace_scripts([make_generated("geeknews:3")])

    assert [item.item_id for item in store.read_scripts()] == ["geeknews:3"]
    assert "body" not in (tmp_path / "scripts.json").read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))
