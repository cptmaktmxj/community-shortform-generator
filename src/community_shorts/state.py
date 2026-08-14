"""SQLite state for deduplication, quotas, and run recovery."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Sequence

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
