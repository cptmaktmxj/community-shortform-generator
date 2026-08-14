from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from community_shorts.models import CuratedItem, LlmAssessment, RawItem


def raw_item_data() -> dict[str, object]:
    return {
        "item_id": "geeknews:1",
        "source_id": "geeknews",
        "url": "https://news.hada.io/topic?id=1",
        "title": "제목",
        "body": "본문",
        "source_language": "ko",
        "fetched_at": datetime(2026, 8, 14, tzinfo=UTC),
    }


def test_raw_item_rejects_naive_fetched_at() -> None:
    data = raw_item_data()
    data["fetched_at"] = datetime(2026, 8, 14)

    with pytest.raises(ValidationError, match="timezone-aware"):
        RawItem(**data)


def test_raw_item_rejects_negative_metrics() -> None:
    data = raw_item_data()
    data["metrics"] = {"likes": -1, "comments": 0}

    with pytest.raises(ValidationError):
        RawItem(**data)


def test_llm_assessment_rejects_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        LlmAssessment(
            provocation_score=1.1,
            mass_appeal_score=0.5,
            fidelity_score=0.9,
            safety_ok=True,
            safety_reason="일반적인 요약이 가능한 소재입니다.",
            safety_categories=[],
            reason="근거",
            summary="요약",
            key_claim="주장",
            hook_points=["지점"],
            tone="정보형",
            output_language="ko",
        )


def test_unsafe_assessment_requires_categories_and_empty_output() -> None:
    """Catch unsafe material leaking generated content into the next stage."""

    assessment = LlmAssessment(
        provocation_score=0.9,
        mass_appeal_score=0.8,
        fidelity_score=0.9,
        safety_ok=False,
        safety_reason="실행 가능한 공격 절차를 포함합니다.",
        safety_categories=["actionable_cyber_abuse"],
        reason="위험한 소재입니다.",
        summary="",
        key_claim="",
        hook_points=[],
        tone="",
        output_language="ko",
    )

    assert assessment.safety_ok is False
    assert assessment.summary == ""

    with pytest.raises(ValidationError):
        LlmAssessment.model_validate(
            assessment.model_dump()
            | {
                "summary": "공격 절차 요약",
                "key_claim": "공격이 가능합니다.",
                "hook_points": ["구체적인 공격법"],
            }
        )


def test_assessment_defaults_to_korean_but_rejects_non_korean_safe_output() -> None:
    """Catch an omitted language marker or an English result passing as Korean output."""

    payload = {
        "provocation_score": 0.7,
        "mass_appeal_score": 0.8,
        "fidelity_score": 0.9,
        "safety_ok": True,
        "safety_reason": "일반적인 기술 뉴스로 생성할 수 있습니다.",
        "safety_categories": [],
        "reason": "대중적인 기술 주제입니다.",
        "summary": "핵심 내용을 한국어로 요약했습니다.",
        "key_claim": "중요한 변화가 발생했습니다.",
        "hook_points": ["예상 밖의 결과"],
        "tone": "정보형",
    }

    assessment = LlmAssessment.model_validate(payload)

    assert assessment.output_language == "ko"

    with pytest.raises(ValidationError, match="Korean"):
        LlmAssessment.model_validate(
            payload
            | {
                "safety_reason": "Suitable for a general audience.",
                "reason": "Broad technology topic.",
                "summary": "This is an English summary.",
                "key_claim": "A relevant change occurred.",
                "hook_points": ["Unexpected result"],
                "tone": "informative",
            }
        )


def test_assessment_requires_korean_explanations() -> None:
    """Keep persisted safety and curation reasons in the configured output language."""

    payload = {
        "provocation_score": 0.7,
        "mass_appeal_score": 0.8,
        "fidelity_score": 0.9,
        "safety_ok": True,
        "safety_reason": "Safe technical news.",
        "safety_categories": [],
        "reason": "Popular and relevant.",
        "summary": "핵심 내용을 한국어로 요약했습니다.",
        "key_claim": "중요한 변화가 발생했습니다.",
        "hook_points": ["예상 밖의 결과"],
        "tone": "정보형",
        "output_language": "ko",
    }

    with pytest.raises(ValidationError, match="explanations must contain Korean"):
        LlmAssessment.model_validate(payload)


def test_curated_item_does_not_accept_original_body() -> None:
    payload = {
        "item_id": "geeknews:1",
        "source_id": "geeknews",
        "url": "https://news.hada.io/topic?id=1",
        "source_language": "ko",
        "output_language": "ko",
        "summary": "요약",
        "key_claim": "주장",
        "hook_points": ["지점"],
        "tone": "정보형",
        "pass": True,
        "reaction_score": 0.8,
        "provocation_score": 0.7,
        "mass_appeal_score": 0.9,
        "curation_score": 0.81,
        "fidelity_score": 0.9,
        "curation_reason": "반응과 대중성이 높음",
        "model": "fixture",
        "body": "원문 전문",
    }

    with pytest.raises(ValidationError):
        CuratedItem(**payload)
