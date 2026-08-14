import json
from datetime import UTC, datetime

import pytest

from community_shorts.llm import FixtureLlmClient, OpenAiLlmClient
from community_shorts.models import RawItem
from community_shorts.prefilter import ScoredRawItem


VALID_ASSESSMENT = {
    "provocation_score": 0.8,
    "mass_appeal_score": 0.9,
    "fidelity_score": 0.95,
    "safe": True,
    "reason": "대중적 관심과 실제 반응이 충분합니다.",
    "summary": "핵심 내용을 한국어로 요약했습니다.",
    "key_claim": "검증 가능한 핵심 주장입니다.",
    "hook_points": ["예상 밖의 결과"],
    "tone": "정보형",
    "output_language": "ko",
}


def english_item() -> ScoredRawItem:
    raw = RawItem(
        item_id="hackernews:1",
        source_id="hackernews",
        url="https://example.com/story",
        title="Agents fail at the last mile",
        body="A discussion about reliability.",
        source_language="en",
        fetched_at=datetime(2026, 8, 14, tzinfo=UTC),
    )
    return ScoredRawItem(raw=raw, reaction_score=0.8)


class SequencedChat:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0
        self.messages = []

    async def complete(self, *, messages, model, response_schema):
        self.messages.append(messages)
        response = self.responses[self.calls]
        self.calls += 1
        return response


@pytest.mark.asyncio
async def test_client_retries_invalid_json_once() -> None:
    transport = SequencedChat(["not-json", json.dumps(VALID_ASSESSMENT, ensure_ascii=False)])
    result = await OpenAiLlmClient(transport=transport, model="test").assess(english_item())

    assert result.output_language == "ko"
    assert transport.calls == 2
    assert "JSON" in transport.messages[1][-1]["content"]


@pytest.mark.asyncio
async def test_english_item_prompt_requires_korean_output() -> None:
    transport = SequencedChat([json.dumps(VALID_ASSESSMENT, ensure_ascii=False)])
    await OpenAiLlmClient(transport=transport, model="test").assess(english_item())

    payload = transport.messages[0][-1]["content"]
    assert '"source_language": "en"' in payload
    assert '"output_language": "ko"' in payload
    assert "한국어" in transport.messages[0][0]["content"]


@pytest.mark.asyncio
async def test_fixture_client_returns_korean_fields_for_english_input() -> None:
    result = await FixtureLlmClient().assess(english_item())

    assert result.output_language == "ko"
    assert any("가" <= character <= "힣" for character in result.summary)
