from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from community_shorts.config import ScriptTimingConfig
from community_shorts.generate import (
    GenerateService,
    TitleValidationError,
    validate_title_package,
)
from community_shorts.generation_models import (
    ContentAnalysis,
    ScriptDraft,
    TitleCandidate,
    TitlePackage,
)
from community_shorts.models import CuratedItem
from community_shorts.state import StateStore
from community_shorts.storage import ArtifactStore
from tests.test_storage import make_curated, make_generated


NOW = datetime(2026, 8, 14, 4, tzinfo=UTC)


def timing(**overrides: float | int) -> ScriptTimingConfig:
    """Build the approved Stage 3 timing configuration."""

    values: dict[str, float | int] = {
        "target_seconds": 45,
        "min_seconds": 30,
        "max_seconds": 60,
        "playback_speed": 1.2,
        "base_spoken_units_per_second": 4.3,
        "max_revisions": 2,
    }
    return ScriptTimingConfig(**(values | overrides))


def analysis() -> ContentAnalysis:
    """Return a deterministic validated Korean analysis."""

    return ContentAnalysis(
        topic_category="work_productivity",
        audience_relevance="일반 직장인의 반복 업무에 직접 관련됩니다.",
        core_facts=["새 기능은 반복 업무 단계를 자동화합니다."],
        angle="일상 업무에서 달라지는 시간을 설명합니다.",
        hook_strategy="기존 방식과 달라진 점을 먼저 제시합니다.",
        claims_to_avoid=["확인되지 않은 성과 수치"],
        recommended_tone="빠르고 명료한 정보형",
    )


def ideal_length_script() -> str:
    """Return a deterministic script estimated inside the 40-50 second band."""

    return "업무 변화입니다. " * 25


def title_package(
    *,
    script: str,
    title: str = "직장인에게 충격, AI 업무 변화",
    excerpt: str = "업무 변화입니다.",
) -> TitlePackage:
    """Build three distinct title styles with configurable first evidence."""

    return TitlePackage(
        candidates=[
            TitleCandidate(
                style="direct_impact",
                title=title,
                supporting_script_excerpt=excerpt,
            ),
            TitleCandidate(
                style="question",
                title="AI가 반복 업무를 정말 줄여줄 수 있을까",
                supporting_script_excerpt=script[:18],
            ),
            TitleCandidate(
                style="conventional_wisdom_reversal",
                title="AI는 답변보다 반복 업무부터 바꾸고 있다",
                supporting_script_excerpt=script[:18],
            ),
        ],
        selected_title=title,
    )


class RecordingGenerationLlm:
    """Record Stage 3 method order and return configured narration revisions."""

    model_name = "gpt-5.4-mini"

    def __init__(self, *, script: str, revisions: list[str] | None = None) -> None:
        self.script = script
        self.revisions = list(revisions or [])
        self.calls: list[str] = []

    async def analyze(self, item: CuratedItem) -> ContentAnalysis:
        """Record and return valid analysis."""

        del item
        self.calls.append("analyze")
        return analysis()

    async def draft(
        self, item: CuratedItem, content_analysis: ContentAnalysis
    ) -> ScriptDraft:
        """Record and return the configured initial script."""

        del item, content_analysis
        self.calls.append("draft")
        return ScriptDraft(script=self.script)

    async def revise(
        self,
        item: CuratedItem,
        content_analysis: ContentAnalysis,
        script: str,
        *,
        direction: Literal["expand", "shorten"],
        current_duration_seconds: float,
        target_duration_seconds: float,
    ) -> ScriptDraft:
        """Record direction and return the next configured revision."""

        del item, content_analysis, script, current_duration_seconds
        del target_duration_seconds
        self.calls.append(f"revise:{direction}")
        return ScriptDraft(script=self.revisions.pop(0))

    async def title(
        self,
        item: CuratedItem,
        content_analysis: ContentAnalysis,
        final_script: str,
    ) -> TitlePackage:
        """Record and return a package supported by the final script."""

        del item, content_analysis
        self.calls.append("title")
        return title_package(script=final_script)


class FailingGenerationLlm(RecordingGenerationLlm):
    """Fail during analysis to exercise atomic rebuild preservation."""

    async def analyze(self, item: CuratedItem) -> ContentAnalysis:
        """Raise a deterministic model failure."""

        del item
        self.calls.append("analyze")
        raise RuntimeError("provider unavailable")


def prepared_service(
    tmp_path: Path, llm: RecordingGenerationLlm
) -> GenerateService:
    """Create a service with one current curated item."""

    store = ArtifactStore(tmp_path)
    store.write_curated([make_curated("geeknews:1")])
    return GenerateService(
        store,
        StateStore(tmp_path / "state.sqlite"),
        llm,
        timing(),
    )


@pytest.mark.asyncio
async def test_generation_orders_analysis_script_duration_then_title(
    tmp_path: Path,
) -> None:
    """Catch a title being generated before analysis and duration validation."""

    llm = RecordingGenerationLlm(script=ideal_length_script())
    service = prepared_service(tmp_path, llm)

    report = await service.run(NOW)

    assert llm.calls == ["analyze", "draft", "title"]
    assert report.completed == 1
    assert service.storage.read_scripts()[0].estimated_duration_seconds >= 40


@pytest.mark.asyncio
async def test_short_script_expands_at_most_twice_before_title(
    tmp_path: Path,
) -> None:
    """Catch an out-of-range script bypassing the bounded revision loop."""

    llm = RecordingGenerationLlm(
        script="짧은 대본.",
        revisions=["여전히 짧은 대본.", ideal_length_script()],
    )
    service = prepared_service(tmp_path, llm)

    report = await service.run(NOW)

    assert llm.calls == [
        "analyze",
        "draft",
        "revise:expand",
        "revise:expand",
        "title",
    ]
    assert report.completed == 1


@pytest.mark.asyncio
async def test_duration_failure_never_calls_title(tmp_path: Path) -> None:
    """Catch title spend and output occurring for an invalid narration length."""

    llm = RecordingGenerationLlm(
        script="짧음.", revisions=["짧음.", "짧음."]
    )
    service = prepared_service(tmp_path, llm)

    report = await service.run(NOW)

    assert "title" not in llm.calls
    assert report.duration_failed == 1
    job = service.state.load_generation_job("geeknews:1")
    assert job is not None and job.status == "duration_failed"
    assert job.script == "짧음."
    assert job.revision_count == 2
    assert job.estimated_duration_seconds is not None


@pytest.mark.asyncio
async def test_title_retry_reuses_saved_analysis_and_script(tmp_path: Path) -> None:
    """Catch a title retry paying again for completed analysis and script calls."""

    llm = RecordingGenerationLlm(script="사용되지 않을 초안입니다.")
    service = prepared_service(tmp_path, llm)
    service.state.save_generation_analysis(
        "geeknews:1", analysis(), model=llm.model_name, at=NOW
    )
    service.state.save_generation_script(
        "geeknews:1",
        script=ideal_length_script(),
        estimated_duration=45.0,
        revision_count=1,
        at=NOW,
    )

    await service.run(NOW)

    assert llm.calls == ["title"]


def test_title_validation_allows_strong_words_with_exact_script_evidence() -> None:
    """Keep approved strong wording from being rejected by keyword alone."""

    script = ideal_length_script()
    package = title_package(script=script)

    assert validate_title_package(package, script).candidates


def test_title_validation_rejects_missing_evidence_and_investment_instruction() -> None:
    """Catch unsupported evidence and direct investment commands."""

    script = ideal_length_script()
    package = title_package(
        script=script,
        title="무조건 지금 이 주식을 매수하세요",
        excerpt="없는 근거",
    )

    with pytest.raises(TitleValidationError):
        validate_title_package(package, script)


@pytest.mark.asyncio
async def test_failed_rebuild_preserves_previous_scripts(tmp_path: Path) -> None:
    """Catch a total provider outage erasing the last successful artifact."""

    store = ArtifactStore(tmp_path)
    store.write_curated([make_curated("geeknews:1")])
    store.write_scripts([make_generated("geeknews:old")])
    service = GenerateService(
        store,
        StateStore(tmp_path / "state.sqlite"),
        FailingGenerationLlm(script="사용되지 않습니다."),
        timing(),
    )

    with pytest.raises(RuntimeError, match="All Stage 3 generations failed"):
        await service.run(NOW, rebuild=True)

    assert [item.item_id for item in store.read_scripts()] == ["geeknews:old"]
