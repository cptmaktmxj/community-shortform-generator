from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from community_shorts.generation_models import (
    ContentAnalysis,
    GeneratedScript,
    TitleCandidate,
    TitleCandidatePool,
    TitlePackage,
)


def valid_analysis() -> ContentAnalysis:
    return ContentAnalysis(
        topic_category="ai_impact",
        audience_relevance="일반 직장인의 업무 방식에 영향을 줍니다.",
        core_facts=["AI 도구가 새로운 업무 기능을 공개했습니다."],
        angle="이 변화가 실제 업무시간에 미치는 영향을 설명합니다.",
        hook_strategy="기존 업무 방식과 달라진 점을 먼저 제시합니다.",
        claims_to_avoid=["확인되지 않은 생산성 수치"],
        recommended_tone="빠르고 명료한 정보형",
    )


def valid_titles() -> TitlePackage:
    script = "AI 도구가 직장인의 반복 업무를 줄이는 새로운 기능을 공개했습니다."
    return TitlePackage(
        candidates=[
            TitleCandidate(
                style="direct_impact",
                title="직장인 반복 업무를 줄이는 AI 신기능",
                supporting_script_excerpt=script,
            ),
            TitleCandidate(
                style="question",
                title="AI가 반복 업무를 정말 줄여줄 수 있을까",
                supporting_script_excerpt=script,
            ),
            TitleCandidate(
                style="conventional_wisdom_reversal",
                title="AI는 답변보다 반복 업무부터 바꾸고 있다",
                supporting_script_excerpt=script,
            ),
        ],
        selected_title="직장인 반복 업무를 줄이는 AI 신기능",
    )


def valid_generated_payload() -> dict[str, object]:
    titles = valid_titles()
    return {
        "item_id": "geeknews:1",
        "source_id": "geeknews",
        "source_url": "https://news.hada.io/topic?id=1",
        "analysis": valid_analysis().model_dump(),
        "script": "AI 도구가 직장인의 반복 업무를 줄이는 새로운 기능을 공개했습니다.",
        "estimated_duration_seconds": 45.0,
        "duration_class": "ideal",
        "playback_speed": 1.2,
        "title_candidates": [candidate.title for candidate in titles.candidates],
        "selected_title": titles.selected_title,
        "model": "gpt-5.4-mini",
        "generated_at": datetime(2026, 8, 14, tzinfo=UTC),
    }


def test_analysis_requires_korean_and_core_facts() -> None:
    """Catch an English or fact-free analysis reaching the script prompt."""

    with pytest.raises(ValidationError):
        ContentAnalysis(
            topic_category="ai_impact",
            audience_relevance="work impact",
            core_facts=[],
            angle="angle",
            hook_strategy="hook",
            claims_to_avoid=[],
            recommended_tone="fast",
        )


def test_title_package_requires_three_distinct_styles() -> None:
    """Catch three copies of one title masquerading as diverse candidates."""

    repeated = TitleCandidate(
        style="direct_impact",
        title="직장인 반복 업무를 줄이는 AI 신기능",
        supporting_script_excerpt="AI 도구가 반복 업무를 줄입니다.",
    )

    with pytest.raises(ValidationError, match="three title styles"):
        TitlePackage(candidates=[repeated, repeated, repeated], selected_title=repeated.title)


def test_title_candidate_pool_requires_five_distinct_editorial_angles() -> None:
    """Catch a ranker pool that omits one requested title angle."""

    script = "AI 도구가 반복 업무를 바꾸는 사실을 설명합니다."
    base = valid_titles().candidates
    pool = TitleCandidatePool(
        candidates=base
        + [
            TitleCandidate(
                style="curiosity_gap",
                title="반복 업무에서 먼저 사라지는 한 가지",
                supporting_script_excerpt=script,
            ),
            TitleCandidate(
                style="strong_factual_statement",
                title="AI가 반복 업무 단계를 실제로 바꾼다",
                supporting_script_excerpt=script,
            ),
        ]
    )

    assert len(pool.candidates) == 5


def test_generated_script_rejects_original_body() -> None:
    """Catch Stage 1 source text leaking into the public Stage 3 artifact."""

    with pytest.raises(ValidationError):
        GeneratedScript.model_validate(valid_generated_payload() | {"body": "원문 전문"})
