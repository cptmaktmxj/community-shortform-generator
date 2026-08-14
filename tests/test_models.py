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
            safe=True,
            reason="근거",
            summary="요약",
            key_claim="주장",
            hook_points=["지점"],
            tone="정보형",
            output_language="ko",
        )


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
