import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from community_shorts.generation_models import ContentAnalysis, TitleCandidate, TitlePackage
from community_shorts.state import StateStore
from tests.test_storage import make_item


NOW = datetime(2026, 8, 14, 3, tzinfo=UTC)


def valid_analysis() -> ContentAnalysis:
    """Build deterministic analysis for SQLite round-trip tests."""

    return ContentAnalysis(
        topic_category="work_productivity",
        audience_relevance="일반 직장인의 반복 업무에 직접 관련됩니다.",
        core_facts=["새 기능은 반복 업무 단계를 자동화합니다."],
        angle="일상 업무에서 달라지는 시간을 설명합니다.",
        hook_strategy="기존 방식과 달라진 점을 먼저 제시합니다.",
        claims_to_avoid=["확인되지 않은 성과 수치"],
        recommended_tone="빠르고 명료한 정보형",
    )


def valid_titles() -> TitlePackage:
    """Build deterministic supported title state."""

    evidence = "검증할 한국어 대본입니다."
    return TitlePackage(
        candidates=[
            TitleCandidate(
                style="direct_impact",
                title="직장인의 반복 업무를 바꾸는 AI",
                supporting_script_excerpt=evidence,
            ),
            TitleCandidate(
                style="question",
                title="AI가 반복 업무를 정말 줄여줄까",
                supporting_script_excerpt=evidence,
            ),
            TitleCandidate(
                style="conventional_wisdom_reversal",
                title="AI는 답변보다 반복 업무부터 바꾼다",
                supporting_script_excerpt=evidence,
            ),
        ],
        selected_title="직장인의 반복 업무를 바꾸는 AI",
    )


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


def test_replace_stage2_results_resets_old_terminal_states(tmp_path: Path) -> None:
    """Catch a rebuilt artifact disagreeing with stale terminal SQLite states."""

    state = StateStore(tmp_path / "state.sqlite")
    items = [
        make_item("geeknews:old"),
        make_item("geeknews:new"),
        make_item("geeknews:unsafe"),
    ]
    now = datetime(2026, 8, 14, 1, tzinfo=UTC)
    state.mark_ingested(items, at=now)
    state.mark_curated(["geeknews:old"], at=now)
    state.mark_safety_rejected("geeknews:unsafe", at=now, reason="old")

    state.replace_stage2_results(
        curated_ids=["geeknews:new"],
        safety_rejections={"geeknews:unsafe": "actionable_cyber_abuse: 새 판정"},
        at=now,
    )

    assert state.curated_ids() == {"geeknews:new"}
    assert state.stage2_terminal_ids() == {"geeknews:new", "geeknews:unsafe"}


def test_generation_job_round_trips_analysis_and_script(tmp_path: Path) -> None:
    """Catch partial Stage 3 progress being lost between executions."""

    state = StateStore(tmp_path / "state.sqlite")
    state.save_generation_analysis(
        "geeknews:1", valid_analysis(), model="gpt-5.4-mini", at=NOW
    )
    state.save_generation_script(
        "geeknews:1",
        script="검증할 한국어 대본입니다.",
        estimated_duration=44.5,
        revision_count=1,
        at=NOW,
    )

    job = state.load_generation_job("geeknews:1")

    assert job is not None
    assert job.status == "scripted"
    assert job.analysis == valid_analysis()
    assert job.script == "검증할 한국어 대본입니다."
    assert job.estimated_duration_seconds == 44.5
    assert job.revision_count == 1


def test_reset_generation_jobs_only_targets_current_curated_ids(
    tmp_path: Path,
) -> None:
    """Catch a rebuild deleting checkpoints for items outside its current scope."""

    state = StateStore(tmp_path / "state.sqlite")
    for item_id in ("geeknews:1", "geeknews:2"):
        state.save_generation_analysis(
            item_id, valid_analysis(), model="gpt-5.4-mini", at=NOW
        )
        state.save_generation_script(
            item_id,
            script="검증할 한국어 대본입니다.",
            estimated_duration=44.5,
            revision_count=0,
            at=NOW,
        )
        state.mark_generation_completed(item_id, valid_titles(), at=NOW)

    state.reset_generation_jobs(["geeknews:2"])

    first = state.load_generation_job("geeknews:1")
    assert first is not None and first.status == "completed"
    assert state.load_generation_job("geeknews:2") is None


def test_generation_failure_rejects_non_terminal_status(tmp_path: Path) -> None:
    """Catch a recoverable substage being mislabeled as a permanent failure."""

    state = StateStore(tmp_path / "state.sqlite")

    with pytest.raises(ValueError, match="failure status"):
        state.save_generation_failure(
            "geeknews:1",
            status="scripted",
            error="잘못된 상태",
            model="gpt-5.4-mini",
            at=NOW,
        )
