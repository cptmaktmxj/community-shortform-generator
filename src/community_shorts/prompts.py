"""Prompt construction for one-pass selection and Korean summarization."""

import json

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
