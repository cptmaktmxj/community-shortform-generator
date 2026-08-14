from datetime import UTC, datetime
from pathlib import Path

import pytest

from community_shorts.ingest import IngestService
from community_shorts.state import StateStore
from community_shorts.storage import ArtifactStore
from tests.test_storage import make_item


SINCE = datetime(2026, 8, 13, tzinfo=UTC)


class FakeAdapter:
    def __init__(self, source_id: str, items=None, error: Exception | None = None) -> None:
        self.source_id = source_id
        self._items = items or []
        self._error = error

    async def fetch_new(self, since: datetime):
        if self._error:
            raise self._error
        return self._items


class FailingStore(ArtifactStore):
    def write_items(self, items):
        raise OSError("disk full")


@pytest.mark.asyncio
async def test_ingest_marks_seen_only_after_artifact_write(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.sqlite")
    service = IngestService(
        [FakeAdapter("geeknews", [make_item("geeknews:1")])],
        FailingStore(tmp_path),
        state,
    )

    with pytest.raises(OSError, match="disk full"):
        await service.run(SINCE)

    assert state.seen_ids("geeknews") == set()


@pytest.mark.asyncio
async def test_one_source_failure_keeps_other_source_results(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.sqlite")
    store = ArtifactStore(tmp_path)
    service = IngestService(
        [
            FakeAdapter("geeknews", [make_item("geeknews:1")]),
            FakeAdapter("broken", error=RuntimeError("bad source")),
        ],
        store,
        state,
    )

    report = await service.run(SINCE)

    assert report.collected == 1
    assert report.failed_sources == ["broken"]
    assert [item.item_id for item in store.read_items()] == ["geeknews:1"]


@pytest.mark.asyncio
async def test_ingest_filters_ids_already_seen(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.sqlite")
    existing = make_item("geeknews:1")
    state.mark_ingested([existing], at=datetime(2026, 8, 14, tzinfo=UTC))
    service = IngestService(
        [FakeAdapter("geeknews", [existing, make_item("geeknews:2")])],
        ArtifactStore(tmp_path),
        state,
    )

    report = await service.run(SINCE)

    assert report.collected == 1
    assert state.seen_ids("geeknews") == {"geeknews:1", "geeknews:2"}
