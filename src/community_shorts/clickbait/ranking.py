"""Deterministic hard gates and ranking for structured GPT title reviews."""

from dataclasses import dataclass

from community_shorts.generation_models import TitleJudgeResult


@dataclass(frozen=True, slots=True)
class JudgedTitle:
    """One GPT-reviewed title with a locally derived combined score."""

    title: str
    evidence_support: float
    clickbait_strength: float
    mass_appeal: float
    safety_ok: bool
    reasoning: str

    @property
    def score(self) -> float:
        """Favor clickbait strength while retaining evidence and broad relevance."""

        return (
            0.45 * self.clickbait_strength
            + 0.35 * self.mass_appeal
            + 0.20 * self.evidence_support
        )


def select_judged_titles(
    candidates: list[str],
    judgment: TitleJudgeResult,
    *,
    evidence_threshold: float = 0.80,
) -> list[JudgedTitle]:
    """Validate review coverage, hard-filter unsafe claims, and return the top three."""

    if len(candidates) != 5 or len(set(candidates)) != 5:
        raise ValueError("exactly five distinct generated candidates are required")
    evaluations = {evaluation.title: evaluation for evaluation in judgment.evaluations}
    if set(evaluations) != set(candidates):
        raise ValueError("evaluated titles must exactly match generated candidates")
    passing = [
        JudgedTitle(
            title=title,
            evidence_support=evaluations[title].evidence_support,
            clickbait_strength=evaluations[title].clickbait_strength,
            mass_appeal=evaluations[title].mass_appeal,
            safety_ok=evaluations[title].safety_ok,
            reasoning=evaluations[title].reasoning,
        )
        for title in candidates
        if evaluations[title].safety_ok
        and evaluations[title].evidence_support >= evidence_threshold
    ]
    if len(passing) < 3:
        raise ValueError("fewer than three title candidates passed GPT safety and evidence gates")
    order = {title: index for index, title in enumerate(candidates)}
    return sorted(passing, key=lambda item: (-item.score, order[item.title]))[:3]
