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
    safety_rejected: int
    failed_item_ids: list[str]
    rebuilt: bool = False


class RebuildIncompleteError(RuntimeError):
    """Raised when a full rebuild cannot evaluate every selected candidate."""


REFUSAL_MARKERS = (
    "도와드릴 수 없습니다",
    "도와줄 수 없습니다",
    "제공할 수 없습니다",
    "지원할 수 없습니다",
    "정책상 도와드릴 수",
    "정책상 제공할 수",
    "안전 정책에 따라 거절",
    "안전 정책으로 인해",
    "cannot assist",
    "can't assist",
    "unable to assist",
    "cannot help",
)


def _contains_refusal_language(assessment: LlmAssessment) -> bool:
    """Detect refusal or policy-warning language that must hard-fail safety."""

    text = " ".join(
        [
            assessment.reason,
            assessment.safety_reason,
            assessment.summary,
            assessment.key_claim,
            *assessment.hook_points,
            assessment.tone,
        ]
    ).casefold()
    return any(marker.casefold() in text for marker in REFUSAL_MARKERS)


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

    async def run(self, now: datetime, *, rebuild: bool = False) -> CurateReport:
        """Select new candidates and write only transformed Korean output."""

        terminal_ids = set() if rebuild else self._state.stage2_terminal_ids()
        raw_items = [item for item in self._storage.read_items() if item.item_id not in terminal_ids]
        source_ids = {item.source_id for item in raw_items}
        geeknews_only = source_ids == {"geeknews"}
        candidates = prefilter(
            raw_items,
            global_limit=10 if geeknews_only else self._global_limit,
            per_source_limit=10 if geeknews_only else self._per_source_limit,
        )

        assessed: list[tuple[ScoredRawItem, LlmAssessment, float]] = []
        failed_item_ids: list[str] = []
        safety_rejected = 0
        safety_rejections: dict[str, str] = {}
        for candidate in candidates:
            try:
                assessment = await self._llm.assess(candidate)
                refusal_detected = _contains_refusal_language(assessment)
                if not assessment.safety_ok or refusal_detected:
                    categories = ",".join(assessment.safety_categories)
                    prefix = categories or "refusal_language"
                    rejection_reason = f"{prefix}: {assessment.safety_reason}"
                    if rebuild:
                        safety_rejections[candidate.raw.item_id] = rejection_reason
                    else:
                        self._state.mark_safety_rejected(
                            candidate.raw.item_id,
                            at=now,
                            reason=rejection_reason,
                        )
                    safety_rejected += 1
                    continue
                score = curation_score(
                    candidate.reaction_score,
                    assessment.provocation_score,
                    assessment.mass_appeal_score,
                )
                if passes_gates(
                    score=score,
                    provocation=assessment.provocation_score,
                    mass_appeal=assessment.mass_appeal_score,
                    fidelity=assessment.fidelity_score,
                    safety_ok=assessment.safety_ok,
                ):
                    assessed.append((candidate, assessment, score))
            except Exception:
                failed_item_ids.append(candidate.raw.item_id)
                LOGGER.exception("LLM assessment failed", extra={"item_id": candidate.raw.item_id})

        if rebuild and failed_item_ids:
            raise RebuildIncompleteError(
                f"Stage 2 rebuild failed for {len(failed_item_ids)} candidate(s)"
            )
        if candidates and len(failed_item_ids) == len(candidates):
            raise RuntimeError("All LLM assessments failed")

        assessed.sort(
            key=lambda entry: (-entry[2], -entry[0].reaction_score, entry[0].raw.item_id)
        )
        remaining_today = (
            self._cycle_limit
            if rebuild
            else max(self._daily_cap - self._state.count_curated_on(now.date()), 0)
        )
        selected = assessed[: min(self._cycle_limit, remaining_today)]
        curated = [self._to_curated(*entry) for entry in selected]
        if rebuild:
            self._storage.replace_curated(curated)
            self._state.replace_stage2_results(
                curated_ids=[item.item_id for item in curated],
                safety_rejections=safety_rejections,
                at=now,
            )
        elif curated:
            self._storage.write_curated(curated)
            self._state.mark_curated([item.item_id for item in curated], at=now)
        return CurateReport(
            evaluated=len(candidates),
            passed=len(curated),
            safety_rejected=safety_rejected,
            failed_item_ids=failed_item_ids,
            rebuilt=rebuild,
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
            provocation_band=assessment.provocation_band,
            provocation_reason=assessment.provocation_reason,
            mass_appeal_score=assessment.mass_appeal_score,
            mass_appeal_band=assessment.mass_appeal_band,
            mass_appeal_reason=assessment.mass_appeal_reason,
            curation_score=round(score, 6),
            fidelity_score=assessment.fidelity_score,
            safety_ok=True,
            safety_reason=assessment.safety_reason,
            safety_categories=assessment.safety_categories,
            curation_reason=assessment.reason,
            model=self._llm.model_name,
        )
