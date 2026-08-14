"""Prompt construction for one-pass selection and Korean summarization."""

import json

from community_shorts.prefilter import ScoredRawItem


SYSTEM_PROMPT = """당신은 커뮤니티 쇼츠 소재 선별자이자 요약자입니다.
반드시 지정된 JSON 스키마 하나만 반환하세요.
입력 언어와 관계없이 reason, summary, key_claim, hook_points, tone은 자연스러운 한국어로 작성하세요.
제품명, 인명, 조직명, 기술 용어는 번역으로 정확성이 떨어지면 원문 표기를 유지하세요.
원문에 없는 사실을 추가하지 말고 한국어 결과가 원문의 의미를 얼마나 충실히 보존하는지 fidelity_score로 평가하세요.
provocation_score는 놀라움·논쟁성·긴급성, mass_appeal_score는 비전문가 이해도·공감·공유 가능성을 평가합니다.
스팸, 혐오, 불법 조장 또는 근거 없는 내용이면 safe=false로 반환하고 요약 필드는 빈 값으로 두세요.
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
