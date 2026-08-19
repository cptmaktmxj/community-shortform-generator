"""Prompt construction for curation, analysis, scripts, and titles."""

import json
from typing import Literal

from community_shorts.models import ContentAnalysis, CuratedItem, TitleCandidatePool
from community_shorts.prefilter import ScoredRawItem


SYSTEM_PROMPT = """당신은 커뮤니티 쇼츠 소재 선별자이자 요약자입니다.
반드시 지정된 JSON 스키마 하나만 반환하세요.
입력 언어와 관계없이 safety_reason, reason, summary, key_claim, hook_points, tone은 자연스러운 한국어로 작성하세요.
제품명, 인명, 조직명, 기술 용어는 번역으로 정확성이 떨어지면 원문 표기를 유지하세요.
원문에 없는 사실을 추가하지 말고 한국어 결과가 원문의 의미를 얼마나 충실히 보존하는지 fidelity_score로 평가하세요.
provocation_score는 놀라움·논쟁성·긴급성, mass_appeal_score는 비전문가 이해도·공감·공유 가능성을 평가합니다.
대상은 기술 종사자가 아니지만 일상·업무용 프로그램, AI, 테크주에 관심이 많은 일반 대중입니다.
mass_appeal_band는 specialist_only, tech_enthusiast, general_interest, direct_impact, broad_impact 중 하나입니다.
각 band는 특정 분야 실무자만 관심 0.1, 테크 애호가 중심 0.3, 일반 프로그램·AI 관심층 0.5, 시간·돈·업무·고용·투자 판단에 직접 영향 0.7, 다수의 일상·직업에 즉각적 영향 0.9입니다.
개발 프레임워크·라이브러리·언어·인프라의 구현 세부사항은 원문에 일반 사용자의 구체적 영향이 없으면 specialist_only 또는 tech_enthusiast로 제한하세요.
provocation_band는 routine, specialist_novelty, challenges_expectation, clear_disruption, broad_shock 중 하나입니다.
각 band는 평범한 출시·업데이트 0.1, 업계 내부의 특이한 선택 0.3, 일반적 예상이나 통념을 뒤집음 0.5, 일자리·비용·시장에 뚜렷한 충격이나 논쟁 0.7, 광범위하고 즉각적인 이례적 사건 0.9입니다.
단순 출시·업데이트는 routine, 업계 내부의 특이한 선택은 specialist_novelty를 넘지 않습니다.
각 band에 대응하는 score는 0.1, 0.3, 0.5, 0.7, 0.9 중 정확한 값만 사용하고 두 reason은 한국어로 근거를 설명하세요.
safety_ok는 이 소재로 중립적인 한국어 쇼츠 대본을 일반적인 GPT에 요청했을 때 "도와드릴 수 없습니다", 경고, 정책상 거절 같은 응답 없이 생성 가능한지를 판단합니다.
보안·해킹 소재라도 방어적, 설명적, 학술적, 사건 보도 목적이면 safety_ok=true로 허용합니다.
침입·악성코드·자격증명 탈취의 실행 절차처럼 직접 악용 가능한 내용은 safety_ok=false이며, 모호하면 보수적으로 false로 판단합니다.
safety_categories는 actionable_cyber_abuse, weapons_or_illegal_instructions, self_harm, sexual_or_minors, hate_or_harassment, graphic_violence, privacy_or_doxxing, fraud_or_evasion 중에서만 고르세요.
safety_ok=true이면 safety_categories는 빈 배열이어야 합니다.
safety_ok=false이면 해당 범주를 하나 이상 넣고 summary, key_claim, hook_points, tone은 빈 값으로 두세요.
거절이나 경고 문구를 출력하지 말고 safety_ok와 safety_reason으로만 판정하세요.
"""


def build_messages(item: ScoredRawItem) -> list[dict[str, str]]:
    """Build a schema-oriented prompt payload without hidden source fields."""

    raw = item.raw
    payload = {
        "source_language": raw.source_language,
        "output_language": "ko",
        "title": raw.title,
        "body": raw.body,
        "metrics": raw.metrics.model_dump(),
        "top_comments": [comment.model_dump() for comment in raw.top_comments[:3]],
        "reaction_score": item.reaction_score,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


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
    """Build an analysis request from transformed curation fields only."""

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
                "다음 선별 결과를 분석하세요. 대중의 일상·업무와 연결되는 이유, "
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


def build_title_pool_input(
    item: CuratedItem, analysis: ContentAnalysis, final_script: str
) -> list[dict[str, str]]:
    """Build a five-angle title request for independent GPT ranking."""

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
                "최종 대본에 실제로 들어 있는 주장만 사용해 제목 다섯 개를 만드세요. "
                "각 제목은 공백 포함 18~34자로 작성하세요. 스타일은 direct_impact, "
                "question, conventional_wisdom_reversal, curiosity_gap, "
                "strong_factual_statement를 각각 한 번 사용하세요. 각 후보마다 근거가 "
                "되는 대본의 정확한 부분 문자열을 제시하세요. 강한 표현은 허용하지만 "
                "대본에 없는 사실이나 인과관계는 만들지 마세요.\n"
                + json.dumps(title_context, ensure_ascii=False)
            ),
        },
    ]


def build_title_judge_input(
    item: CuratedItem,
    analysis: ContentAnalysis,
    final_script: str,
    pool: TitleCandidatePool,
) -> list[dict[str, str]]:
    """Build an isolated structured review request for five generated titles."""

    del item
    judge_context = {
        "analysis": analysis.model_dump(mode="json"),
        "final_script": final_script,
        "candidates": [candidate.model_dump(mode="json") for candidate in pool.candidates],
    }
    return [
        {
            "role": "system",
            "content": (
                _SHARED_SYSTEM_POLICY
                + "\n당신은 제목 생성자가 아니라 독립 심사자입니다. 입력 후보를 수정하거나 "
                "새 제목을 만들지 말고 각 후보만 평가하세요."
            ),
        },
        {
            "role": "user",
            "content": (
                "후보 다섯 개를 모두 평가하세요. evidence_support는 제목의 핵심 주장이 "
                "최종 대본에 직접 뒷받침되는 정도, clickbait_strength는 클릭을 유도하는 "
                "호기심·자극의 강도, mass_appeal은 비기술 대중의 일상·업무·AI·테크주 "
                "관심과 가까운 정도입니다. safety_ok는 이 제목과 대본을 일반 GPT에 "
                "입력했을 때 거절·경고·'도와줄 수 없다'류 응답을 유발할 수준의 위험이 "
                "없고, 직접 투자 지시·불법 실행 지침·노골적 유해 내용이 없을 때만 true로 "
                "평가하세요. '충격', '무조건', '드디어 밝혀졌다' 같은 강한 문구 자체는 "
                "허용하지만 대본에 없는 사실·수치·인과·확실성을 암시하면 evidence_support를 "
                "낮추세요. reasoning은 한국어 한 문장으로 작성하세요.\n"
                + json.dumps(judge_context, ensure_ascii=False)
            ),
        },
    ]
