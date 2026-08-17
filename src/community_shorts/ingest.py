"""Stage 1 orchestration for isolated source ingestion."""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping, Sequence

from community_shorts.adapters.base import SourceAdapter
from community_shorts.models import RawItem
from community_shorts.progress import NullProgress, ProgressEvent, ProgressSink
from community_shorts.state import StateStore
from community_shorts.storage import ArtifactStore


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestReport:
    """Observable outcome of one Stage 1 run."""

    collected: int
    skipped_seen: int
    failed_sources: list[str]


class IngestService:
    """Collect enabled sources without allowing one failure to erase others."""

    def __init__(
        self,
        adapters: Sequence[SourceAdapter],
        storage: ArtifactStore,
        state: StateStore,
        *,
        progress: ProgressSink | None = None,
        source_names: Mapping[str, str] | None = None,
    ) -> None:
        self._adapters = adapters
        self._storage = storage
        self._state = state
        self._progress = progress or NullProgress()
        self._source_names = dict(source_names or {})

    async def run(self, since: datetime) -> IngestReport:
        """Fetch unseen items, write JSON, and then commit seen state."""

        run_id = uuid.uuid4().hex
        started_at = datetime.now(UTC)
        failed_sources: list[str] = []
        skipped_seen = 0
        collected: dict[str, RawItem] = {}
        successful_sources = 0

        total_sources = len(self._adapters)
        for position, adapter in enumerate(self._adapters, start=1):
            source_name = self._source_names.get(adapter.source_id, adapter.source_id)
            self._progress.emit(
                ProgressEvent(
                    stage="ingest",
                    status="running",
                    message=f"{source_name} 글 수집 중",
                    current=position,
                    total=total_sources,
                )
            )
            try:
                fetched = await adapter.fetch_new(since)
                successful_sources += 1
                seen = self._state.seen_ids(adapter.source_id)
                source_skipped = sum(item.item_id in seen for item in fetched)
                source_new = sum(item.item_id not in seen for item in fetched)
                skipped_seen += source_skipped
                for item in fetched:
                    if item.item_id not in seen:
                        collected[item.item_id] = item
                self._progress.emit(
                    ProgressEvent(
                        stage="ingest",
                        status="completed",
                        message=f"{source_name} 수집 완료",
                        current=position,
                        total=total_sources,
                        detail=(
                            f"확인 {len(fetched)}개 · 신규 {source_new}개 · "
                            f"중복 {source_skipped}개"
                        ),
                    )
                )
            except Exception as exc:
                failed_sources.append(adapter.source_id)
                LOGGER.exception("source ingestion failed", extra={"source_id": adapter.source_id})
                self._progress.emit(
                    ProgressEvent(
                        stage="ingest",
                        status="failed",
                        message=f"{source_name} 수집 실패",
                        current=position,
                        total=total_sources,
                        detail=type(exc).__name__,
                    )
                )
                continue

        if self._adapters and successful_sources == 0:
            self._progress.emit(
                ProgressEvent(
                    stage="ingest",
                    status="failed",
                    message="전체 수집 실패",
                    detail=f"실패 소스 {len(failed_sources)}개",
                )
            )
            raise RuntimeError(f"All enabled sources failed: {', '.join(failed_sources)}")

        new_items = [collected[key] for key in sorted(collected)]
        if new_items:
            self._storage.write_items(new_items)
            self._state.mark_ingested(new_items, at=datetime.now(UTC))

        finished_at = datetime.now(UTC)
        report = IngestReport(
            collected=len(new_items),
            skipped_seen=skipped_seen,
            failed_sources=failed_sources,
        )
        self._state.record_run(
            run_id=run_id,
            stage="ingest",
            started_at=started_at,
            finished_at=finished_at,
            status="partial" if failed_sources else "success",
            details={
                "collected": report.collected,
                "skipped_seen": report.skipped_seen,
                "failed_sources": report.failed_sources,
            },
        )
        self._progress.emit(
            ProgressEvent(
                stage="ingest",
                status="completed",
                message="전체 수집 완료",
                detail=(
                    f"신규 {report.collected}개 · 중복 {report.skipped_seen}개 · "
                    f"실패 소스 {len(report.failed_sources)}개"
                ),
            )
        )
        return report
