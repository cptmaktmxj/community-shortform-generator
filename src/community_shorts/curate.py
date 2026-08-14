"""Stage 2 orchestration for selection and Korean summarization."""

import logging
from dataclasses import dataclass
from datetime import datetime

from community_shorts.llm import LlmClient
from community_shorts.models import CuratedItem, LlmAssessment
from community_shorts.prefilter import ScoredRawItem, prefilter
from community_shorts.scoring import curation_score, passes_gates
from community_shorts.state import StateStore
from community_shorts.storage import ArtifactStore


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CurateReport:
    """Observable outcome of one Stage 2 run."""

    evaluated: int
    passed: int
    failed_item_ids: list[str]


class CurateService:
    """Prefilter, assess, gate, and persist a bounded set of candidates."""

    def __init__(
        self,
        storage: ArtifactStore,
        state: StateStore,
        llm: LlmClient,
        *,
        global_limit: int = 20,
        per_source_limit: int = 5,
        cycle_limit: int = 2,
        daily_cap: int = 8,
    ) -> None:
        self._storage = storage
        self._state = state
        self._llm = llm
        self._global_limit = global_limit
        self._per_source_limit = per_source_limit
        self._cycle_limit = cycle_limit
        self._daily_cap = daily_cap

    async def run(self, now: datetime) -> CurateReport:
        """Select new candidates and write only transformed Korean output."""

        curated_ids = self._state.curated_ids()
        raw_items = [item for item in self._storage.read_items() if item.item_id not in curated_ids]
        source_ids = {item.source_id for item in raw_items}
        geeknews_only = source_ids == {"geeknews"}
        candidates = prefilter(
            raw_items,
            global_limit=10 if geeknews_only else self._global_limit,
            per_source_limit=10 if geeknews_only else self._per_source_limit,
        )

        assessed: list[tuple[ScoredRawItem, LlmAssessment, float]] = []
        failed_item_ids: list[str] = []
        for candidate in candidates:
            try:
                assessment = await self._llm.assess(candidate)
                score = curation_score(
                    candidate.reaction_score,
                    assessment.provocation_score,
                    assessment.mass_appeal_score,
                )
                if passes_gates(
                    score=score,
                    fidelity=assessment.fidelity_score,
                    safe=assessment.safe,
                ):
                    assessed.append((candidate, assessment, score))
            except Exception:
                failed_item_ids.append(candidate.raw.item_id)
                LOGGER.exception("LLM assessment failed", extra={"item_id": candidate.raw.item_id})

        assessed.sort(
            key=lambda entry: (-entry[2], -entry[0].reaction_score, entry[0].raw.item_id)
        )
        remaining_today = max(self._daily_cap - self._state.count_curated_on(now.date()), 0)
        selected = assessed[: min(self._cycle_limit, remaining_today)]
        curated = [self._to_curated(*entry) for entry in selected]
        if curated:
            self._storage.write_curated(curated)
            self._state.mark_curated([item.item_id for item in curated], at=now)
        return CurateReport(
            evaluated=len(candidates),
            passed=len(curated),
            failed_item_ids=failed_item_ids,
        )

    def _to_curated(
        self,
        candidate: ScoredRawItem,
        assessment: LlmAssessment,
        score: float,
    ) -> CuratedItem:
        """Transform an assessment without carrying source body or comments forward."""

        raw = candidate.raw
        return CuratedItem(
            item_id=raw.item_id,
            source_id=raw.source_id,
            url=raw.url,
            source_language=raw.source_language,
            output_language="ko",
            summary=assessment.summary,
            key_claim=assessment.key_claim,
            hook_points=assessment.hook_points,
            tone=assessment.tone,
            pass_=True,
            reaction_score=candidate.reaction_score,
            provocation_score=assessment.provocation_score,
            mass_appeal_score=assessment.mass_appeal_score,
            curation_score=round(score, 6),
            fidelity_score=assessment.fidelity_score,
            curation_reason=assessment.reason,
            model=self._llm.model_name,
        )
