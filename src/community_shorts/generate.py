"""Resumable Stage 3 orchestration for analysis, narration, and titles."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from community_shorts.clickbait.ranking import JudgedTitle, select_judged_titles

from community_shorts.config import ScriptTimingConfig
from community_shorts.duration import DurationEstimate, estimate_duration
from community_shorts.generation_llm import (
    GenerationLlmClient,
    GenerationResponseError,
)
from community_shorts.generation_models import (
    ContentAnalysis,
    GeneratedScript,
    TitleCandidate,
    TitleCandidatePool,
    TitlePackage,
    TitleRankingScore,
)
from community_shorts.models import CuratedItem
from community_shorts.state import StateStore
from community_shorts.storage import ArtifactStore


LOGGER = logging.getLogger(__name__)
_INVESTMENT_INSTRUCTION_MARKERS = (
    "매수하세요",
    "매도하세요",
    "전액 투자",
    "수익 보장",
)


class TitleValidationError(ValueError):
    """Raised when generated titles are not supported or locally safe."""


class ItemGenerationError(RuntimeError):
    """Wrap an unexpected item failure with a sanitized substage name."""

    def __init__(self, substage: str, cause_type: str) -> None:
        super().__init__(f"{substage}: {cause_type}")
        self.substage = substage
        self.cause_type = cause_type


@dataclass(frozen=True, slots=True)
class GenerateReport:
    """Counts from one Stage 3 execution."""

    attempted: int
    completed: int
    duration_failed: int
    title_failed: int
    failed_item_ids: tuple[str, ...]
    rebuilt: bool


@dataclass(frozen=True, slots=True)
class _ItemOutcome:
    """Internal result from one controlled item workflow."""

    status: Literal["completed", "duration_failed", "title_failed"]
    generated: GeneratedScript | None = None
    titles: TitlePackage | None = None


def validate_title_package(package: TitlePackage, script: str) -> TitlePackage:
    """Require bounded, supported titles without direct investment instructions."""

    required_styles = {
        "direct_impact",
        "question",
        "conventional_wisdom_reversal",
    }
    _validate_title_candidates(
        package.candidates, script, required_styles=required_styles
    )
    return package


def validate_title_pool(pool: TitleCandidatePool, script: str) -> TitleCandidatePool:
    """Apply the same hard evidence and safety gates before model ranking."""

    required_styles = {
        "direct_impact",
        "question",
        "conventional_wisdom_reversal",
        "curiosity_gap",
        "strong_factual_statement",
    }
    _validate_title_candidates(pool.candidates, script, required_styles=required_styles)
    return pool


def _validate_title_candidates(
    candidates: list[TitleCandidate], script: str, *, required_styles: set[str]
) -> None:
    """Validate distinct styles, length, exact evidence, URLs, and investment commands."""

    styles = {candidate.style for candidate in candidates}
    titles = [candidate.title.strip() for candidate in candidates]
    if styles != required_styles or len(set(titles)) != len(required_styles):
        raise TitleValidationError("distinct required title styles are missing")

    for candidate, title in zip(candidates, titles, strict=True):
        if not 18 <= len(title) <= 34:
            raise TitleValidationError("title length must be between 18 and 34 characters")
        excerpt = candidate.supporting_script_excerpt.strip()
        if not excerpt or excerpt not in script:
            raise TitleValidationError("title evidence is absent from the final script")
        lowered = f"{title} {excerpt}".lower()
        if "http://" in lowered or "https://" in lowered:
            raise TitleValidationError("titles and evidence must not contain URLs")
        if any(marker in title for marker in _INVESTMENT_INSTRUCTION_MARKERS):
            raise TitleValidationError("direct investment instructions are not allowed")


class GenerateService:
    """Execute and resume Stage 3 in the approved substage order."""

    def __init__(
        self,
        storage: ArtifactStore,
        state: StateStore,
        llm: GenerationLlmClient,
        timing: ScriptTimingConfig,
        *,
        gpt_title_ranking: bool = False,
    ) -> None:
        self.storage = storage
        self.state = state
        self.llm = llm
        self.timing = timing
        self.gpt_title_ranking = gpt_title_ranking

    async def run(self, now: datetime, *, rebuild: bool = False) -> GenerateReport:
        """Generate all current curated items, isolating failures by item."""

        curated = self.storage.read_curated()
        current_ids = [item.item_id for item in curated]
        if rebuild:
            self.state.reset_generation_jobs(current_ids)

        attempted = 0
        duration_failed = 0
        title_failed = 0
        failed_ids: list[str] = []
        completed: list[tuple[GeneratedScript, TitlePackage]] = []

        for item in curated:
            job = self.state.load_generation_job(item.item_id)
            if not rebuild and job is not None and job.status == "completed":
                continue
            attempted += 1
            try:
                outcome = await self._generate_item(item, now)
                if outcome.status == "duration_failed":
                    duration_failed += 1
                    continue
                if outcome.status == "title_failed":
                    title_failed += 1
                    continue
                assert outcome.generated is not None and outcome.titles is not None
                if rebuild:
                    completed.append((outcome.generated, outcome.titles))
                else:
                    self.storage.write_scripts([outcome.generated])
                    self.state.mark_generation_completed(
                        item.item_id, outcome.titles, at=now
                    )
                    completed.append((outcome.generated, outcome.titles))
            except ItemGenerationError as exc:
                LOGGER.error(
                    "Stage 3 item %s failed during %s (%s)",
                    item.item_id,
                    exc.substage,
                    exc.cause_type,
                )
                self.state.save_generation_failure(
                    item.item_id,
                    status="failed",
                    error=str(exc),
                    model=self.llm.model_name,
                    at=now,
                )
                failed_ids.append(item.item_id)

        if rebuild:
            if completed or not curated:
                self.storage.replace_scripts([result[0] for result in completed])
                for generated, titles in completed:
                    self.state.mark_generation_completed(
                        generated.item_id, titles, at=now
                    )

        if attempted > 0 and len(failed_ids) == attempted:
            raise RuntimeError("All Stage 3 generations failed")

        return GenerateReport(
            attempted=attempted,
            completed=len(completed),
            duration_failed=duration_failed,
            title_failed=title_failed,
            failed_item_ids=tuple(failed_ids),
            rebuilt=rebuild,
        )

    async def _generate_item(
        self, item: CuratedItem, now: datetime
    ) -> _ItemOutcome:
        """Resume or execute every substage for one curated item."""

        job = self.state.load_generation_job(item.item_id)
        analysis = job.analysis if job is not None else None
        if analysis is None:
            try:
                analysis = await self.llm.analyze(item)
            except Exception as exc:
                raise ItemGenerationError("analysis", type(exc).__name__) from exc
            self.state.save_generation_analysis(
                item.item_id, analysis, model=self.llm.model_name, at=now
            )

        script: str | None = None
        estimate: DurationEstimate | None = None
        revision_count = 0
        if job is not None and job.status in {"scripted", "title_failed"} and job.script:
            saved_estimate = estimate_duration(job.script, self.timing)
            if saved_estimate.classification in {"ideal", "acceptable"}:
                script = job.script
                estimate = saved_estimate
                revision_count = job.revision_count

        if script is None:
            try:
                draft = await self.llm.draft(item, analysis)
            except Exception as exc:
                raise ItemGenerationError("script", type(exc).__name__) from exc
            script = draft.script
            estimate = estimate_duration(script, self.timing)

            while (
                estimate.classification not in {"ideal", "acceptable"}
                and revision_count < self.timing.max_revisions
            ):
                direction: Literal["expand", "shorten"] = (
                    "expand" if estimate.classification == "short" else "shorten"
                )
                try:
                    revision = await self.llm.revise(
                        item,
                        analysis,
                        script,
                        direction=direction,
                        current_duration_seconds=estimate.seconds,
                        target_duration_seconds=self.timing.target_seconds,
                    )
                except Exception as exc:
                    raise ItemGenerationError("revision", type(exc).__name__) from exc
                script = revision.script
                revision_count += 1
                estimate = estimate_duration(script, self.timing)

        assert estimate is not None
        if estimate.classification not in {"ideal", "acceptable"}:
            self.state.save_generation_script(
                item.item_id,
                script=script,
                estimated_duration=estimate.seconds,
                revision_count=revision_count,
                at=now,
            )
            self.state.save_generation_failure(
                item.item_id,
                status="duration_failed",
                error=f"duration_{estimate.classification}",
                model=self.llm.model_name,
                at=now,
            )
            return _ItemOutcome(status="duration_failed")

        self.state.save_generation_script(
            item.item_id,
            script=script,
            estimated_duration=estimate.seconds,
            revision_count=revision_count,
            at=now,
        )

        ranked_titles: list[JudgedTitle] | None = None
        try:
            if not self.gpt_title_ranking:
                titles = await self.llm.title(item, analysis, script)
                titles = validate_title_package(titles, script)
            else:
                pool = await self.llm.title_pool(item, analysis, script)
                pool = validate_title_pool(pool, script)
                judgment = await self.llm.judge_titles(
                    item, analysis, script, pool
                )
                ranked_titles = select_judged_titles(
                    [candidate.title for candidate in pool.candidates], judgment
                )
                by_title = {candidate.title: candidate for candidate in pool.candidates}
                selected_candidates = [by_title[item.title] for item in ranked_titles]
                titles = TitlePackage(
                    candidates=selected_candidates,
                    selected_title=selected_candidates[0].title,
                )
        except (GenerationResponseError, TitleValidationError, ValueError, KeyError) as exc:
            self.state.save_generation_failure(
                item.item_id,
                status="title_failed",
                error=type(exc).__name__,
                model=self.llm.model_name,
                at=now,
            )
            return _ItemOutcome(status="title_failed")
        except Exception as exc:
            raise ItemGenerationError("title", type(exc).__name__) from exc

        generated = GeneratedScript(
            item_id=item.item_id,
            source_id=item.source_id,
            source_url=item.url,
            analysis=analysis,
            script=script,
            estimated_duration_seconds=estimate.seconds,
            duration_class=estimate.classification,
            playback_speed=self.timing.playback_speed,
            title_candidates=[candidate.title for candidate in titles.candidates],
            selected_title=titles.selected_title,
            title_ranking=(
                [
                    TitleRankingScore(
                        title=item.title,
                        evidence_support=item.evidence_support,
                        clickbait_strength=item.clickbait_strength,
                        mass_appeal=item.mass_appeal,
                        safety_ok=item.safety_ok,
                        combined_score=item.score,
                        reasoning=item.reasoning,
                    )
                    for item in ranked_titles
                ]
                if ranked_titles is not None
                else None
            ),
            model=self.llm.model_name,
            generated_at=now,
        )
        return _ItemOutcome(status="completed", generated=generated, titles=titles)
