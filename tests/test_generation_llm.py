import json
from types import SimpleNamespace
from typing import Any

import pytest

import community_shorts.generation_llm as generation_llm_module
from community_shorts.generation_llm import (
    FixtureGenerationLlmClient,
    GenerationResponseError,
    OpenAiGenerationLlmClient,
    OpenAiResponsesTransport,
)
from community_shorts.generation_models import (
    ContentAnalysis,
    ScriptDraft,
    TitleCandidate,
    TitlePackage,
)
from community_shorts.models import CuratedItem


def curated_item() -> CuratedItem:
    """Build a safe Stage 2 artifact with no original source body."""

    return CuratedItem(
        item_id="geeknews:1",
        source_id="geeknews",
        url="https://news.hada.io/topic?id=1",
        source_language="en",
        output_language="ko",
        summary="AI 도구가 반복 업무를 자동화하는 기능을 공개했습니다.",
        key_claim="비개발자도 일상 업무의 반복 단계를 줄일 수 있습니다.",
        hook_points=["직장인의 시간 절약", "기존 업무 방식의 변화"],
        tone="빠르고 명료한 정보형",
        pass_=True,
        reaction_score=0.4,
        provocation_score=0.7,
        provocation_band="clear_disruption",
        provocation_reason="기존 업무 절차를 뚜렷하게 바꿀 가능성이 있습니다.",
        mass_appeal_score=0.7,
        mass_appeal_band="direct_impact",
        mass_appeal_reason="일반 직장인의 업무와 직접 연결됩니다.",
        curation_score=0.64,
        fidelity_score=0.9,
        safety_ok=True,
        safety_reason="안전한 업무 생산성 정보입니다.",
        safety_categories=[],
        curation_reason="대중의 일상 업무에 직접적인 변화가 있습니다.",
        model="k-exaone",
    )


def valid_analysis() -> ContentAnalysis:
    """Return deterministic Korean analysis for transport tests."""

    return ContentAnalysis(
        topic_category="work_productivity",
        audience_relevance="일반 직장인의 반복 업무와 직접 관련됩니다.",
        core_facts=["새 기능은 반복 업무 단계를 자동화합니다."],
        angle="비개발자의 실제 업무시간 변화에 초점을 맞춥니다.",
        hook_strategy="달라지는 일상 업무를 첫 문장에 제시합니다.",
        claims_to_avoid=["입력에 없는 생산성 수치"],
        recommended_tone="빠르고 명료한 정보형",
    )


def valid_titles(script: str = "최종 검증된 한국어 대본입니다.") -> TitlePackage:
    """Return three supported title styles for client tests."""

    return TitlePackage(
        candidates=[
            TitleCandidate(
                style="direct_impact",
                title="직장인의 반복 업무를 바꾸는 AI",
                supporting_script_excerpt=script,
            ),
            TitleCandidate(
                style="question",
                title="AI가 반복 업무를 정말 줄여줄까",
                supporting_script_excerpt=script,
            ),
            TitleCandidate(
                style="conventional_wisdom_reversal",
                title="AI는 답변보다 반복 업무부터 바꾼다",
                supporting_script_excerpt=script,
            ),
        ],
        selected_title="직장인의 반복 업무를 바꾸는 AI",
    )


class CapturingResponsesTransport:
    """Record parse calls and return a prevalidated response."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[SimpleNamespace] = []

    async def parse(self, **kwargs: Any) -> Any:
        """Capture keyword arguments passed across the transport boundary."""

        self.calls.append(SimpleNamespace(**kwargs))
        return self.result


class SequencedResponsesTransport(CapturingResponsesTransport):
    """Raise or return each configured outcome in order."""

    def __init__(self, outcomes: list[Any]) -> None:
        super().__init__(None)
        self.outcomes = outcomes

    async def parse(self, **kwargs: Any) -> Any:
        """Return the next outcome so retry behavior can be verified."""

        self.calls.append(SimpleNamespace(**kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_analysis_payload_uses_only_transformed_stage2_fields() -> None:
    """Catch Stage 1 original text being added to the analysis request."""

    transport = CapturingResponsesTransport(valid_analysis())
    client = OpenAiGenerationLlmClient(transport=transport, model="gpt-5.4-mini")

    await client.analyze(curated_item())

    payload = json.dumps(transport.calls[0].input, ensure_ascii=False)
    assert curated_item().summary in payload
    assert "body" not in payload
    assert "top_comments" not in payload
    assert transport.calls[0].output_type is ContentAnalysis
    assert transport.calls[0].max_output_tokens == 800


@pytest.mark.asyncio
async def test_title_payload_contains_final_script_and_not_original_summary() -> None:
    """Catch titles being produced before the final validated script exists."""

    script = "최종 검증된 한국어 대본입니다."
    transport = CapturingResponsesTransport(valid_titles(script))
    client = OpenAiGenerationLlmClient(transport=transport, model="gpt-5.4-mini")

    await client.title(curated_item(), valid_analysis(), script)

    payload = json.dumps(transport.calls[0].input, ensure_ascii=False)
    assert script in payload
    assert curated_item().summary not in payload
    assert transport.calls[0].output_type is TitlePackage
    assert transport.calls[0].max_output_tokens == 600


@pytest.mark.asyncio
async def test_openai_transport_uses_responses_parse_and_pydantic_schema(
    monkeypatch,
) -> None:
    """Catch regression to chat completions or manual JSON schema handling."""

    captured: dict[str, Any] = {}

    class FakeResponses:
        async def parse(self, **kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=valid_analysis())

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_options"] = kwargs
            self.responses = FakeResponses()

    monkeypatch.setattr(generation_llm_module, "AsyncOpenAI", FakeOpenAI)
    transport = OpenAiResponsesTransport(
        base_url="https://api.openai.com/v1",
        api_key="test",
        timeout_seconds=120,
    )

    result = await transport.parse(
        model="gpt-5.4-mini",
        input=[],
        output_type=ContentAnalysis,
        max_output_tokens=800,
    )

    assert result == valid_analysis()
    assert captured["model"] == "gpt-5.4-mini"
    assert captured["text_format"] is ContentAnalysis
    assert captured["max_output_tokens"] == 800
    assert captured["client_options"]["max_retries"] == 0


@pytest.mark.asyncio
async def test_schema_failure_retries_once_with_korean_correction() -> None:
    """Catch a transient refusal or schema failure aborting the entire item."""

    transport = SequencedResponsesTransport(
        [GenerationResponseError("invalid schema"), valid_analysis()]
    )

    result = await OpenAiGenerationLlmClient(
        transport=transport, model="gpt-5.4-mini"
    ).analyze(curated_item())

    assert result.core_facts
    assert len(transport.calls) == 2
    assert "한국어" in transport.calls[1].input[-1]["content"]


@pytest.mark.asyncio
async def test_unsupported_title_evidence_retries_once() -> None:
    """Catch a plausible title whose claimed evidence is absent from the final script."""

    script = "최종 검증된 한국어 대본입니다."
    unsupported = valid_titles("대본에는 존재하지 않는 근거입니다.")
    transport = SequencedResponsesTransport([unsupported, valid_titles(script)])
    client = OpenAiGenerationLlmClient(transport=transport, model="gpt-5.4-mini")

    result = await client.title(curated_item(), valid_analysis(), script)

    assert result.candidates[0].supporting_script_excerpt == script
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_fixture_client_returns_supported_korean_outputs() -> None:
    """Keep offline E2E tests independent from OpenAI availability."""

    client = FixtureGenerationLlmClient()
    item = curated_item()
    analysis = await client.analyze(item)
    draft = await client.draft(item, analysis)
    titles = await client.title(item, analysis, draft.script)

    assert analysis.core_facts
    assert isinstance(draft, ScriptDraft)
    assert len(titles.candidates) == 3
    assert all(
        candidate.supporting_script_excerpt in draft.script
        for candidate in titles.candidates
    )
