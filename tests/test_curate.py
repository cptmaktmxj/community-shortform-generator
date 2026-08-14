from datetime import UTC, datetime
from pathlib import Path

import pytest

from community_shorts.curate import CurateService
from community_shorts.llm import FixtureLlmClient
from community_shorts.models import Metrics, RawItem
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


@pytest.mark.asyncio
async def test_curate_isolates_one_llm_failure(tmp_path: Path) -> None:
    candidates = [make_candidate(index) for index in range(6)]
    store, state = prepare_state(tmp_path, candidates)
    service = CurateService(store, state, OneFailureLlm())

    report = await service.run(NOW)

    assert report.failed_item_ids == ["geeknews:5"]
    assert report.passed == 2
    assert len(store.read_curated()) == 2
