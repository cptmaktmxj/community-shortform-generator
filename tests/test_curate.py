from datetime import UTC, datetime
from pathlib import Path

import pytest

from community_shorts.curate import CurateService
from community_shorts.llm import FixtureLlmClient
from community_shorts.models import LlmAssessment, Metrics, RawItem
from community_shorts.state import StateStore
from community_shorts.storage import ArtifactStore


NOW = datetime(2026, 8, 14, 9, tzinfo=UTC)


def make_candidate(number: int, *, source: str = "geeknews") -> RawItem:
    return RawItem(
        item_id=f"{source}:{number}",
        source_id=source,
        url=f"https://example.com/{source}/{number}",
        title=f"후보 {number}",
        body="충분히 유용한 커뮤니티 논의",
        metrics=Metrics(likes=number + 1, comments=number + 1),
        source_language="ko" if source == "geeknews" else "en",
        fetched_at=NOW,
    )


def prepare_state(tmp_path: Path, current: list[RawItem], already_curated: int = 0):
    store = ArtifactStore(tmp_path)
    state = StateStore(tmp_path / "state.sqlite")
    prior = [make_candidate(100 + index, source="prior") for index in range(already_curated)]
    all_items = prior + current
    store.write_items(all_items)
    state.mark_ingested(all_items, at=NOW)
    state.mark_curated([item.item_id for item in prior], at=NOW)
    return store, state


@pytest.mark.asyncio
async def test_curate_never_exceeds_eight_per_day(tmp_path: Path) -> None:
    candidates = [make_candidate(index) for index in range(12)]
    store, state = prepare_state(tmp_path, candidates, already_curated=7)
    service = CurateService(store, state, FixtureLlmClient())

    report = await service.run(NOW)

    assert report.passed == 1
    assert len(store.read_curated()) == 1
    assert state.count_curated_on(NOW.date()) == 8


@pytest.mark.asyncio
async def test_curate_keeps_only_top_two_per_cycle(tmp_path: Path) -> None:
    candidates = [make_candidate(index) for index in range(6)]
    store, state = prepare_state(tmp_path, candidates)
    service = CurateService(store, state, FixtureLlmClient())

    report = await service.run(NOW)

    assert report.evaluated == 6
    assert report.passed == 2
    assert all(item.pass_ for item in store.read_curated())


class OneFailureLlm(FixtureLlmClient):
    async def assess(self, item):
        if item.raw.item_id.endswith(":5"):
            raise ValueError("invalid model output")
        return await super().assess(item)


class AllFailureLlm(FixtureLlmClient):
    async def assess(self, item):
        raise ConnectionError("LLM endpoint unavailable")


@pytest.mark.asyncio
async def test_curate_isolates_one_llm_failure(tmp_path: Path) -> None:
    candidates = [make_candidate(index) for index in range(6)]
    store, state = prepare_state(tmp_path, candidates)
    service = CurateService(store, state, OneFailureLlm())

    report = await service.run(NOW)

    assert report.failed_item_ids == ["geeknews:5"]
    assert report.passed == 2
    assert len(store.read_curated()) == 2


@pytest.mark.asyncio
async def test_curate_fails_stage_when_every_llm_call_fails(tmp_path: Path) -> None:
    candidates = [make_candidate(index) for index in range(3)]
    store, state = prepare_state(tmp_path, candidates)
    service = CurateService(store, state, AllFailureLlm())

    with pytest.raises(RuntimeError, match="All LLM assessments failed"):
        await service.run(NOW)

    assert store.read_curated() == []


class RefusalLanguageLlm(FixtureLlmClient):
    async def assess(self, item):
        assessment = await super().assess(item)
        return assessment.model_copy(
            update={"summary": "죄송하지만 이 요청은 도와드릴 수 없습니다."}
        )


class UnsafeLlm(FixtureLlmClient):
    async def assess(self, item):
        return LlmAssessment(
            provocation_score=0.9,
            provocation_band="broad_shock",
            provocation_reason="광범위하게 악용될 수 있는 위험한 요청입니다.",
            mass_appeal_score=0.7,
            mass_appeal_band="direct_impact",
            mass_appeal_reason="다수의 계정과 개인정보에 직접 피해를 줄 수 있습니다.",
            fidelity_score=0.95,
            safety_ok=False,
            safety_reason="실행 가능한 공격 절차를 제공하도록 유도합니다.",
            safety_categories=["actionable_cyber_abuse"],
            reason="안전 기준을 통과하지 못했습니다.",
            summary="",
            key_claim="",
            hook_points=[],
            tone="",
            output_language="ko",
        )


class PositivePolicyLanguageLlm(FixtureLlmClient):
    async def assess(self, item):
        assessment = await super().assess(item)
        return assessment.model_copy(
            update={"safety_reason": "정책상 거절 없이 안전하게 생성 가능합니다."}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("llm", [UnsafeLlm(), RefusalLanguageLlm()])
async def test_curate_persists_safety_rejection_and_does_not_retry(
    tmp_path: Path, llm
) -> None:
    """Catch explicit or disguised safety failures reaching the next stage."""

    candidate = make_candidate(1)
    store, state = prepare_state(tmp_path, [candidate])
    service = CurateService(store, state, llm)

    first = await service.run(NOW)
    second = await service.run(NOW)

    assert first.evaluated == 1
    assert first.passed == 0
    assert first.safety_rejected == 1
    assert second.evaluated == 0
    assert store.read_curated() == []
    assert state.stage2_terminal_ids() == {candidate.item_id}


@pytest.mark.asyncio
async def test_curate_does_not_reject_positive_policy_language(tmp_path: Path) -> None:
    """Catch broad keyword matching that rejects an explicit no-refusal decision."""

    candidate = make_candidate(1)
    store, state = prepare_state(tmp_path, [candidate])
    service = CurateService(store, state, PositivePolicyLanguageLlm())

    report = await service.run(NOW)

    assert report.passed == 1
    assert report.safety_rejected == 0
