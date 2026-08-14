import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from community_shorts.state import StateStore
from tests.test_storage import make_item


def test_state_store_records_ingested_and_curated_items(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.sqlite")
    item = make_item("geeknews:1")

    state.mark_ingested([item], at=datetime(2026, 8, 14, 1, tzinfo=UTC))
    state.mark_curated([item.item_id], at=datetime(2026, 8, 14, 2, tzinfo=UTC))

    assert state.seen_ids("geeknews") == {"geeknews:1"}
    assert state.count_curated_on(date(2026, 8, 14)) == 1


def test_state_connection_context_closes_database(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.sqlite")

    with state._connect() as connection:
        connection.execute("SELECT 1")

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_state_store_marks_safety_rejection_as_terminal(tmp_path: Path) -> None:
    """Catch rejected items being sent back to the Stage 2 model on every run."""

    state = StateStore(tmp_path / "state.sqlite")
    item = make_item("geeknews:unsafe")
    now = datetime(2026, 8, 14, 1, tzinfo=UTC)
    state.mark_ingested([item], at=now)

    state.mark_safety_rejected(
        item.item_id,
        at=now,
        reason="actionable_cyber_abuse: 실행 가능한 공격 절차",
    )

    assert state.stage2_terminal_ids() == {item.item_id}
    with state._connect() as connection:
        row = connection.execute(
            "SELECT status, error, curated_at FROM items WHERE item_id = ?",
            (item.item_id,),
        ).fetchone()
    assert dict(row) == {
        "status": "safety_rejected",
        "error": "actionable_cyber_abuse: 실행 가능한 공격 절차",
        "curated_at": None,
    }
