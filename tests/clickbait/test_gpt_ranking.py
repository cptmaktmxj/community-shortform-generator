import pytest

from community_shorts.clickbait.ranking import select_judged_titles
from community_shorts.models import (
    TitleCandidateEvaluation,
    TitleJudgeResult,
)


def evaluation(
    title: str,
    *,
    evidence: float,
    clickbait: float,
    mass_appeal: float,
    safety_ok: bool = True,
) -> TitleCandidateEvaluation:
    """Build one Korean GPT title evaluation for deterministic ranking tests."""

    return TitleCandidateEvaluation(
        title=title,
        evidence_support=evidence,
        clickbait_strength=clickbait,
        mass_appeal=mass_appeal,
        safety_ok=safety_ok,
        reasoning="대본 근거와 대중적인 흥미를 함께 평가했습니다.",
    )


def test_select_judged_titles_hard_filters_safety_and_evidence_then_ranks() -> None:
    """Catch a provocative but unsafe or unsupported title surviving GPT review."""

    candidates = ["제목1", "제목2", "제목3", "제목4", "제목5"]
    judgment = TitleJudgeResult(
        evaluations=[
            evaluation("제목1", evidence=0.95, clickbait=0.90, mass_appeal=0.90),
            evaluation("제목2", evidence=0.90, clickbait=0.80, mass_appeal=0.80),
            evaluation("제목3", evidence=0.85, clickbait=0.70, mass_appeal=0.70),
            evaluation("제목4", evidence=0.99, clickbait=1.00, mass_appeal=1.00, safety_ok=False),
            evaluation("제목5", evidence=0.79, clickbait=1.00, mass_appeal=1.00),
        ]
    )

    selected = select_judged_titles(candidates, judgment, evidence_threshold=0.80)

    assert [item.title for item in selected] == ["제목1", "제목2", "제목3"]
    assert selected[0].score == pytest.approx(0.91)


def test_select_judged_titles_rejects_missing_or_extra_candidate_reviews() -> None:
    """Catch GPT omitting a candidate or evaluating an ungenerated title."""

    judgment = TitleJudgeResult(
        evaluations=[
            evaluation(f"제목{index}", evidence=0.9, clickbait=0.8, mass_appeal=0.8)
            for index in range(1, 6)
        ]
    )

    with pytest.raises(ValueError, match="exactly match"):
        select_judged_titles(
            ["제목1", "제목2", "제목3", "제목4", "다른제목"], judgment
        )


def test_select_judged_titles_requires_three_passing_candidates() -> None:
    """Catch an underfilled title package being marked completed."""

    judgment = TitleJudgeResult(
        evaluations=[
            evaluation(
                f"제목{index}",
                evidence=0.9,
                clickbait=0.8,
                mass_appeal=0.8,
                safety_ok=index <= 2,
            )
            for index in range(1, 6)
        ]
    )

    with pytest.raises(ValueError, match="three"):
        select_judged_titles([f"제목{index}" for index in range(1, 6)], judgment)
