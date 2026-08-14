"""OpenAI-compatible and deterministic fixture clients for Stage 2."""

import json
from typing import Any, Protocol, Sequence

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError

from community_shorts.models import LlmAssessment
from community_shorts.prefilter import ScoredRawItem
from community_shorts.prompts import build_messages


class LlmResponseError(ValueError):
    """Raised after both schema parsing attempts fail."""


class ChatTransport(Protocol):
    """Minimal structured chat completion boundary used by the model client."""

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, str]],
        model: str,
        response_schema: dict[str, Any],
    ) -> str:
        """Return the assistant message content as text."""


class LlmClient(Protocol):
    """Assess and summarize one prefiltered candidate."""

    model_name: str

    async def assess(self, item: ScoredRawItem) -> LlmAssessment:
        """Return a validated Korean assessment."""


class OpenAiChatTransport:
    """Call an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 5.0)),
            max_retries=0,
        )

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, str]],
        model: str,
        response_schema: dict[str, Any],
    ) -> str:
        """Request strict JSON schema output from the configured endpoint."""

        response = await self._client.chat.completions.create(
            model=model,
            messages=list(messages),  # type: ignore[arg-type]
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "curation_assessment",
                    "strict": True,
                    "schema": response_schema,
                },
            },
            temperature=0.1,
        )
        content = response.choices[0].message.content
        if not content:
            raise LlmResponseError("LLM returned empty content")
        return content


class OpenAiLlmClient:
    """Validate one-pass selection and summary JSON, retrying once if malformed."""

    def __init__(self, *, transport: ChatTransport, model: str) -> None:
        self._transport = transport
        self.model_name = model

    async def assess(self, item: ScoredRawItem) -> LlmAssessment:
        """Return a validated assessment or fail after one correction retry."""

        messages = build_messages(item)
        last_error: Exception | None = None
        for attempt in range(2):
            current_messages = list(messages)
            if attempt:
                current_messages.append(
                    {
                        "role": "user",
                        "content": "이전 응답은 유효한 JSON 스키마가 아닙니다. 설명 없이 올바른 JSON 객체만 다시 반환하세요.",
                    }
                )
            content = await self._transport.complete(
                messages=current_messages,
                model=self.model_name,
                response_schema=LlmAssessment.model_json_schema(),
            )
            try:
                return LlmAssessment.model_validate(json.loads(content))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
        raise LlmResponseError(f"Invalid LLM JSON after one retry: {last_error}") from last_error


class FixtureLlmClient:
    """Return deterministic Korean assessments for offline pipeline tests."""

    model_name = "fixture-korean-curator"

    async def assess(self, item: ScoredRawItem) -> LlmAssessment:
        """Create stable Korean output without contacting a model endpoint."""

        return LlmAssessment(
            provocation_score=0.95,
            mass_appeal_score=0.95,
            fidelity_score=0.95,
            safe=True,
            reason="실제 반응과 대중적 관심 가능성이 충분한 시험 항목입니다.",
            summary=f"'{item.raw.title}'에 관한 핵심 내용을 한국어로 요약한 시험 결과입니다.",
            key_claim="커뮤니티에서 주목할 만한 변화나 논점이 제기됐습니다.",
            hook_points=["예상 밖의 반응", "쉽게 설명할 수 있는 핵심 논점"],
            tone="정보형",
            output_language="ko",
        )
