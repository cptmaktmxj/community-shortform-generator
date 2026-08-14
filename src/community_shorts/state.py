"""SQLite state for deduplication, quotas, and run recovery."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Literal, Mapping, Sequence

from community_shorts.generation_models import (
    ContentAnalysis,
    GenerationJob,
    TitlePackage,
)
from community_shorts.models import RawItem


class StateStore:
    """Persist item stage transitions in small SQLite transactions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open, transact on, and always close a SQLite connection."""

        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        """Create state tables when the database is first opened."""

        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS items (
                    item_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    curated_at TEXT,
                    status TEXT NOT NULL,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS generation_jobs (
                    item_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    analysis_json TEXT,
                    script_text TEXT,
                    estimated_duration REAL,
                    revision_count INTEGER NOT NULL DEFAULT 0,
                    title_json TEXT,
                    model TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT
                );
                """
            )

    def seen_ids(self, source_id: str) -> set[str]:
        """Return IDs already persisted for a source."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT item_id FROM items WHERE source_id = ?",
                (source_id,),
            ).fetchall()
        return {str(row["item_id"]) for row in rows}

    def mark_ingested(self, items: Sequence[RawItem], *, at: datetime) -> None:
        """Record items only after their Stage 1 artifact has been written."""

        rows = [(item.item_id, item.source_id, at.isoformat(), "ingested") for item in items]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO items(item_id, source_id, ingested_at, status)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    source_id = excluded.source_id,
                    ingested_at = excluded.ingested_at
                """,
                rows,
            )

    def mark_curated(self, item_ids: Sequence[str], *, at: datetime) -> None:
        """Mark selected items after `curated.json` has been written."""

        with self._connect() as connection:
            connection.executemany(
                "UPDATE items SET curated_at = ?, status = 'curated', error = NULL WHERE item_id = ?",
                [(at.isoformat(), item_id) for item_id in item_ids],
            )

    def mark_safety_rejected(self, item_id: str, *, at: datetime, reason: str) -> None:
        """Persist a terminal Stage 2 safety decision without counting it as curated."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE items
                SET curated_at = NULL, status = 'safety_rejected', error = ?
                WHERE item_id = ?
                """,
                (reason, item_id),
            )

    def replace_stage2_results(
        self,
        *,
        curated_ids: Sequence[str],
        safety_rejections: Mapping[str, str],
        at: datetime,
    ) -> None:
        """Replace terminal Stage 2 states after a successful rebuild write."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE items
                SET curated_at = NULL, status = 'ingested', error = NULL
                WHERE curated_at IS NOT NULL OR status = 'safety_rejected'
                """
            )
            connection.executemany(
                """
                UPDATE items
                SET status = 'safety_rejected', error = ?
                WHERE item_id = ?
                """,
                [(reason, item_id) for item_id, reason in safety_rejections.items()],
            )
            connection.executemany(
                """
                UPDATE items
                SET curated_at = ?, status = 'curated', error = NULL
                WHERE item_id = ?
                """,
                [(at.isoformat(), item_id) for item_id in curated_ids],
            )

    def count_curated_on(self, day: date) -> int:
        """Count selected records whose UTC timestamp falls on a date."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM items WHERE substr(curated_at, 1, 10) = ?",
                (day.isoformat(),),
            ).fetchone()
        return int(row["count"] if row else 0)

    def curated_ids(self) -> set[str]:
        """Return item IDs already selected into the Stage 2 artifact."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT item_id FROM items WHERE curated_at IS NOT NULL"
            ).fetchall()
        return {str(row["item_id"]) for row in rows}

    def stage2_terminal_ids(self) -> set[str]:
        """Return IDs already curated or permanently rejected by the safety gate."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT item_id FROM items
                WHERE curated_at IS NOT NULL OR status = 'safety_rejected'
                """
            ).fetchall()
        return {str(row["item_id"]) for row in rows}

    def load_generation_job(self, item_id: str) -> GenerationJob | None:
        """Load and validate one resumable Stage 3 checkpoint."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM generation_jobs WHERE item_id = ?", (item_id,)
            ).fetchone()
        if row is None:
            return None
        analysis = (
            ContentAnalysis.model_validate_json(row["analysis_json"])
            if row["analysis_json"] is not None
            else None
        )
        title_package = (
            TitlePackage.model_validate_json(row["title_json"])
            if row["title_json"] is not None
            else None
        )
        return GenerationJob(
            item_id=str(row["item_id"]),
            status=str(row["status"]),
            analysis=analysis,
            script=row["script_text"],
            estimated_duration_seconds=row["estimated_duration"],
            revision_count=int(row["revision_count"]),
            title_package=title_package,
            model=str(row["model"]),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            error=row["error"],
        )

    def save_generation_analysis(
        self,
        item_id: str,
        analysis: ContentAnalysis,
        *,
        model: str,
        at: datetime,
    ) -> None:
        """Upsert validated analysis and clear all later Stage 3 substages."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO generation_jobs(
                    item_id, status, analysis_json, revision_count, model,
                    updated_at, error
                ) VALUES (?, 'analyzed', ?, 0, ?, ?, NULL)
                ON CONFLICT(item_id) DO UPDATE SET
                    status = 'analyzed',
                    analysis_json = excluded.analysis_json,
                    script_text = NULL,
                    estimated_duration = NULL,
                    revision_count = 0,
                    title_json = NULL,
                    model = excluded.model,
                    updated_at = excluded.updated_at,
                    error = NULL
                """,
                (item_id, analysis.model_dump_json(), model, at.isoformat()),
            )

    def save_generation_script(
        self,
        item_id: str,
        *,
        script: str,
        estimated_duration: float,
        revision_count: int,
        at: datetime,
    ) -> None:
        """Save a validated script and duration while preserving its analysis."""

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE generation_jobs
                SET status = 'scripted', script_text = ?, estimated_duration = ?,
                    revision_count = ?, title_json = NULL, updated_at = ?, error = NULL
                WHERE item_id = ?
                """,
                (
                    script,
                    estimated_duration,
                    revision_count,
                    at.isoformat(),
                    item_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"generation analysis is missing for {item_id}")

    def save_generation_failure(
        self,
        item_id: str,
        *,
        status: Literal["duration_failed", "title_failed", "failed"] | str,
        error: str,
        model: str,
        at: datetime,
    ) -> None:
        """Persist an allowed terminal Stage 3 failure without losing checkpoints."""

        allowed = {"duration_failed", "title_failed", "failed"}
        if status not in allowed:
            raise ValueError(f"invalid generation failure status: {status}")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO generation_jobs(
                    item_id, status, revision_count, model, updated_at, error
                ) VALUES (?, ?, 0, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    status = excluded.status,
                    model = excluded.model,
                    updated_at = excluded.updated_at,
                    error = excluded.error
                """,
                (item_id, status, model, at.isoformat(), error),
            )

    def mark_generation_completed(
        self, item_id: str, title_package: TitlePackage, *, at: datetime
    ) -> None:
        """Store validated titles and mark a scripted checkpoint complete."""

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE generation_jobs
                SET status = 'completed', title_json = ?, updated_at = ?, error = NULL
                WHERE item_id = ?
                """,
                (title_package.model_dump_json(), at.isoformat(), item_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"generation job is missing for {item_id}")

    def reset_generation_jobs(self, item_ids: Sequence[str]) -> None:
        """Delete Stage 3 checkpoints only for the explicitly supplied IDs."""

        if not item_ids:
            return
        placeholders = ",".join("?" for _ in item_ids)
        with self._connect() as connection:
            connection.execute(
                f"DELETE FROM generation_jobs WHERE item_id IN ({placeholders})",
                tuple(item_ids),
            )

    def record_run(
        self,
        *,
        run_id: str,
        stage: str,
        started_at: datetime,
        finished_at: datetime,
        status: str,
        details: dict[str, object],
    ) -> None:
        """Insert or replace a structured pipeline run summary."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, stage, started_at, finished_at, status, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    stage,
                    started_at.isoformat(),
                    finished_at.isoformat(),
                    status,
                    json.dumps(details, ensure_ascii=False),
                ),
            )
