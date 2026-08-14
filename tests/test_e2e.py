from datetime import UTC, datetime
from pathlib import Path

import pytest

from community_shorts.curate import CurateService
from community_shorts.ingest import IngestService
from community_shorts.llm import FixtureLlmClient
from community_shorts.models import Metrics, RawItem
from community_shorts.state import StateStore
from community_shorts.storage import ArtifactStore


NOW = datetime(2026, 8, 14, 9, tzinfo=UTC)


class FakeAdapter:
    source_id = "hackernews"

    async def fetch_new(self, since: datetime) -> list[RawItem]:
        return [
            RawItem(
                item_id="hackernews:1",
                source_id="hackernews",
                url="https://example.com/agent-story",
                title="Why agents fail at the last mile",
                body="Developers discuss unreliable multi-step automation.",
                metrics=Metrics(likes=50, comments=20),
                source_language="en",
                fetched_at=NOW,
            )
        ]


@pytest.mark.asyncio
async def test_fixture_run_writes_only_stage_one_and_two_artifacts(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    state = StateStore(tmp_path / "state.sqlite")
    ingest = IngestService([FakeAdapter()], store, state)
    await ingest.run(NOW)
    curate = CurateService(store, state, FixtureLlmClient())
    await curate.run(NOW)

    assert (tmp_path / "items.json").exists()
    assert (tmp_path / "curated.json").exists()
    assert not (tmp_path / "scripts").exists()
    curated = store.read_curated()
    assert len(curated) == 1
    assert curated[0].output_language == "ko"
    assert not hasattr(curated[0], "body")
