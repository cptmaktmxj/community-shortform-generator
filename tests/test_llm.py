import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import community_shorts.llm as llm_module
from community_shorts.llm import FixtureLlmClient, OpenAiLlmClient
from community_shorts.models import RawItem
from community_shorts.prefilter import ScoredRawItem


VALID_ASSESSMENT = {
    "provocation_score": 0.7,
    "provocation_band": "clear_disruption",
    "provocation_reason": "업무 방식에 뚜렷한 변화를 만드는 소재입니다.",
    "mass_appeal_score": 0.9,
    "mass_appeal_band": "broad_impact",
    "mass_appeal_reason": "많은 사람의 일상과 업무에 직접 영향을 줍니다.",
    "fidelity_score": 0.95,
    "safety_ok": True,
    "safety_reason": "일반적인 기술 뉴스 요약으로 생성할 수 있습니다.",
    "safety_categories": [],
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


def test_openai_transport_disables_sdk_retries_and_bounds_connect_timeout(monkeypatch) -> None:
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "AsyncOpenAI", FakeAsyncOpenAI)

    llm_module.OpenAiChatTransport(
        base_url="http://127.0.0.1:1/v1",
        api_key="EMPTY",
        timeout_seconds=120,
    )

    assert captured["max_retries"] == 0
    assert captured["timeout"].connect == 5.0


@pytest.mark.asyncio
async def test_openai_transport_caps_completion_tokens(monkeypatch) -> None:
    """Catch local models generating indefinitely when no output limit is sent."""

    captured = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
            )

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(llm_module, "AsyncOpenAI", FakeAsyncOpenAI)
    transport = llm_module.OpenAiChatTransport(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        timeout_seconds=120,
    )

    await transport.complete(
        messages=[{"role": "user", "content": "test"}],
        model="test-model",
        response_schema={"type": "object"},
    )

    assert captured["max_completion_tokens"] == 1024
    assert captured["temperature"] == 0


@pytest.mark.asyncio
async def test_client_retries_invalid_json_once() -> None:
    transport = SequencedChat(["not-json", json.dumps(VALID_ASSESSMENT, ensure_ascii=False)])
    result = await OpenAiLlmClient(transport=transport, model="test").assess(english_item())

    assert result.output_language == "ko"
    assert transport.calls == 2
    assert "JSON" in transport.messages[1][-1]["content"]
    assert "safety_reason" in transport.messages[1][-1]["content"]
    assert "한국어" in transport.messages[1][-1]["content"]


@pytest.mark.asyncio
async def test_client_repairs_only_a_single_missing_object_closer() -> None:
    """Catch constrained decoding that stalls on whitespace before the final brace."""

    valid_json = json.dumps(VALID_ASSESSMENT, ensure_ascii=False)
    truncated_with_whitespace = valid_json[:-1] + ("\n " * 200)
    transport = SequencedChat([truncated_with_whitespace])

    result = await OpenAiLlmClient(transport=transport, model="test").assess(english_item())

    assert result.safety_ok is True
    assert result.summary == "핵심 내용을 한국어로 요약했습니다."
    assert transport.calls == 1


@pytest.mark.asyncio
async def test_english_item_prompt_requires_korean_output() -> None:
    transport = SequencedChat([json.dumps(VALID_ASSESSMENT, ensure_ascii=False)])
    await OpenAiLlmClient(transport=transport, model="test").assess(english_item())

    payload = transport.messages[0][-1]["content"]
    assert '"source_language": "en"' in payload
    assert '"output_language": "ko"' in payload
    assert "한국어" in transport.messages[0][0]["content"]
    assert "safety_ok" in transport.messages[0][0]["content"]
    assert "도와드릴 수 없습니다" in transport.messages[0][0]["content"]
    assert "기술 종사자가 아니지만" in transport.messages[0][0]["content"]
    assert "specialist_only" in transport.messages[0][0]["content"]


@pytest.mark.asyncio
async def test_fixture_client_returns_korean_fields_for_english_input() -> None:
    result = await FixtureLlmClient().assess(english_item())

    assert result.output_language == "ko"
    assert result.safety_ok is True
    assert any("가" <= character <= "힣" for character in result.summary)
