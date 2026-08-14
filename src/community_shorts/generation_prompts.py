"""Prompt builders for the four isolated Stage 3 generation calls."""

import json
from typing import Literal

from community_shorts.generation_models import ContentAnalysis
from community_shorts.models import CuratedItem


FACT_AND_STYLE_POLICY = (
    "‘충격’, ‘무조건’, ‘드디어 밝혀졌다’ 같은 강한 표현은 허용하지만 "
    "입력에 없는 사건·수치·인과관계·성과·확실성을 만들어내면 안 됩니다."
)

_SHARED_SYSTEM_POLICY = f"""
당신은 한국어 테크 숏폼의 편집자입니다.
대상은 기술 종사자가 아니지만 일상·업무 프로그램, AI, 테크주에 관심이 많은 대중입니다.
모든 서술형 출력은 자연스러운 한국어로 작성하세요.
원문에 없는 투자 수익 보장이나 직접적인 투자 지시를 만들지 마세요.
{FACT_AND_STYLE_POLICY}
""".strip()


def build_analysis_input(item: CuratedItem) -> list[dict[str, str]]:
    """Build an analysis request from transformed Stage 2 fields only."""

    transformed = {
        "source_language": item.source_language,
        "summary": item.summary,
        "key_claim": item.key_claim,
        "hook_points": item.hook_points,
        "tone": item.tone,
        "provocation_band": item.provocation_band,
        "provocation_reason": item.provocation_reason,
        "mass_appeal_band": item.mass_appeal_band,
        "mass_appeal_reason": item.mass_appeal_reason,
        "curation_reason": item.curation_reason,
    }
    return [
        {"role": "system", "content": _SHARED_SYSTEM_POLICY},
        {
            "role": "user",
            "content": (
                "다음 2단계 선별 결과를 분석하세요. 대중의 일상·업무와 연결되는 이유, "
                "입력으로 확인되는 핵심 사실, 숏폼 각도, 첫 훅, 피해야 할 주장을 구분하세요.\n"
                + json.dumps(transformed, ensure_ascii=False)
            ),
        },
    ]


def build_draft_input(
    item: CuratedItem, analysis: ContentAnalysis
) -> list[dict[str, str]]:
    """Build a 45-second Korean narration request from validated analysis."""

    del item
    return [
        {"role": "system", "content": _SHARED_SYSTEM_POLICY},
        {
            "role": "user",
            "content": (
                "검증된 분석만 사용해 1.2배속 기준 약 45초 한국어 낭독 대본을 작성하세요. "
                "공백 포함 260~320자를 목표로 하세요. "
                "구조는 강한 훅, 무슨 일이 생겼는지, 대중에게 왜 중요한지, "
                "현실적인 영향, 짧은 마무리의 다섯 부분으로 자연스럽게 이어가세요. "
                "제목, URL, 제작 지시문은 넣지 마세요.\n"
                + analysis.model_dump_json()
            ),
        },
    ]


def build_revision_input(
    item: CuratedItem,
    analysis: ContentAnalysis,
    script: str,
    *,
    direction: Literal["expand", "shorten"],
    current_duration_seconds: float,
    target_duration_seconds: float,
) -> list[dict[str, str]]:
    """Build a bounded length revision request without changing factual claims."""

    del item
    duration_ratio = target_duration_seconds / max(current_duration_seconds, 0.001)
    target_character_count = max(1, round(len(script) * duration_ratio))
    revision = {
        "direction": direction,
        "current_duration_seconds": current_duration_seconds,
        "target_duration_seconds": target_duration_seconds,
        "target_character_count": target_character_count,
        "analysis": analysis.model_dump(mode="json"),
        "current_script": script,
    }
    return [
        {"role": "system", "content": _SHARED_SYSTEM_POLICY},
        {
            "role": "user",
            "content": (
                "핵심 사실과 다섯 부분 흐름을 유지하면서 대본 길이만 조정하세요. "
                "expand면 설명을 보강하고 shorten이면 중복·수식을 줄이세요. "
                "공백 포함 목표 글자 수의 ±10% 범위에 반드시 맞추세요. "
                "제목이나 URL을 추가하지 마세요.\n"
                + json.dumps(revision, ensure_ascii=False)
            ),
        },
    ]


def build_title_input(
    item: CuratedItem, analysis: ContentAnalysis, final_script: str
) -> list[dict[str, str]]:
    """Build a title request only after the final narration passes validation."""

    del item
    title_context = {
        "analysis": analysis.model_dump(mode="json"),
        "final_script": final_script,
    }
    return [
        {"role": "system", "content": _SHARED_SYSTEM_POLICY},
        {
            "role": "user",
            "content": (
                "최종 대본에 실제로 들어 있는 주장만 사용해 제목 세 개를 만드세요. "
                "각 제목은 공백 포함 18~34자로 작성하세요. "
                "스타일은 direct_impact, question, conventional_wisdom_reversal을 "
                "각각 한 번 사용하고, 근거가 되는 대본의 정확한 부분 문자열을 제시하세요.\n"
                + json.dumps(title_context, ensure_ascii=False)
            ),
        },
    ]
