"""Official OpenAI Responses and deterministic fixture clients for Stage 3."""

from typing import Callable, Literal, Protocol, Sequence, TypeVar

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from community_shorts.generation_models import (
    ContentAnalysis,
    ScriptDraft,
    TitleCandidate,
    TitlePackage,
)
from community_shorts.generation_prompts import (
    build_analysis_input,
    build_draft_input,
    build_revision_input,
    build_title_input,
)
from community_shorts.models import CuratedItem


ModelT = TypeVar("ModelT", bound=BaseModel)


class GenerationResponseError(ValueError):
    """Raised when a Stage 3 response remains invalid after one correction."""


class ResponsesTransport(Protocol):
    """Minimal boundary around the official Responses structured parse helper."""

    async def parse(
        self,
        *,
        model: str,
        input: Sequence[dict[str, str]],
        output_type: type[ModelT],
        max_output_tokens: int,
    ) -> ModelT:
        """Return one response validated against the requested Pydantic model."""


class GenerationLlmClient(Protocol):
    """Generate analysis, narration revisions, and titles in strict order."""

    model_name: str

    async def analyze(self, item: CuratedItem) -> ContentAnalysis:
        """Analyze a transformed Stage 2 artifact in Korean."""

    async def draft(
        self, item: CuratedItem, analysis: ContentAnalysis
    ) -> ScriptDraft:
        """Draft the initial Korean narration."""

    async def revise(
        self,
        item: CuratedItem,
        analysis: ContentAnalysis,
        script: str,
        *,
        direction: Literal["expand", "shorten"],
        current_duration_seconds: float,
        target_duration_seconds: float,
    ) -> ScriptDraft:
        """Expand or shorten a narration without changing its facts."""

    async def title(
        self, item: CuratedItem, analysis: ContentAnalysis, final_script: str
    ) -> TitlePackage:
        """Create titles supported by the final validated narration."""


class OpenAiResponsesTransport:
    """Call the official OpenAI Responses API with Pydantic Structured Outputs."""

    def __init__(
        self, *, base_url: str, api_key: str, timeout_seconds: float
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 5.0)),
            max_retries=0,
        )

    async def parse(
        self,
        *,
        model: str,
        input: Sequence[dict[str, str]],
        output_type: type[ModelT],
        max_output_tokens: int,
    ) -> ModelT:
        """Parse one official Responses API result into the requested schema."""

        response = await self._client.responses.parse(
            model=model,
            input=list(input),  # type: ignore[arg-type]
            text_format=output_type,
            max_output_tokens=max_output_tokens,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise GenerationResponseError(
                "OpenAI response did not contain parsed output"
            )
        return output_type.model_validate(parsed)


class OpenAiGenerationLlmClient:
    """Run four schema-bound GPT calls with one corrective retry per call."""

    def __init__(self, *, transport: ResponsesTransport, model: str) -> None:
        self._transport = transport
        self.model_name = model

    async def analyze(self, item: CuratedItem) -> ContentAnalysis:
        """Analyze a Stage 2 item before any narration or title is produced."""

        return await self._request(
            input=build_analysis_input(item),
            output_type=ContentAnalysis,
            max_output_tokens=800,
        )

    async def draft(
        self, item: CuratedItem, analysis: ContentAnalysis
    ) -> ScriptDraft:
        """Draft Korean narration from validated analysis."""

        return await self._request(
            input=build_draft_input(item, analysis),
            output_type=ScriptDraft,
            max_output_tokens=1200,
        )

    async def revise(
        self,
        item: CuratedItem,
        analysis: ContentAnalysis,
        script: str,
        *,
        direction: Literal["expand", "shorten"],
        current_duration_seconds: float,
        target_duration_seconds: float,
    ) -> ScriptDraft:
        """Request one fact-preserving narration length correction."""

        return await self._request(
            input=build_revision_input(
                item,
                analysis,
                script,
                direction=direction,
                current_duration_seconds=current_duration_seconds,
                target_duration_seconds=target_duration_seconds,
            ),
            output_type=ScriptDraft,
            max_output_tokens=1200,
        )

    async def title(
        self, item: CuratedItem, analysis: ContentAnalysis, final_script: str
    ) -> TitlePackage:
        """Generate titles only from analysis and the final accepted narration."""

        return await self._request(
            input=build_title_input(item, analysis, final_script),
            output_type=TitlePackage,
            max_output_tokens=600,
            response_validator=lambda package: _validate_title_evidence(
                package, final_script
            ),
        )

    async def _request(
        self,
        *,
        input: Sequence[dict[str, str]],
        output_type: type[ModelT],
        max_output_tokens: int,
        response_validator: Callable[[ModelT], None] | None = None,
    ) -> ModelT:
        """Retry only invalid, refused, or empty structured output once."""

        last_error: Exception | None = None
        for attempt in range(2):
            request_input = list(input)
            if attempt:
                request_input.append(
                    {
                        "role": "user",
                        "content": (
                            "이전 응답은 스키마를 충족하지 못했습니다. 모든 서술형 필드는 "
                            "한국어로 작성하고, 입력에 없는 사실을 추가하지 말고, "
                            "요청된 JSON 스키마에 정확히 맞춰 다시 반환하세요."
                        ),
                    }
                )
            try:
                parsed = await self._transport.parse(
                    model=self.model_name,
                    input=request_input,
                    output_type=output_type,
                    max_output_tokens=max_output_tokens,
                )
                validated = output_type.model_validate(parsed)
                if response_validator is not None:
                    response_validator(validated)
                return validated
            except (GenerationResponseError, ValidationError) as exc:
                last_error = exc
        raise GenerationResponseError(
            f"Invalid Stage 3 response after one retry: {last_error}"
        ) from last_error


class FixtureGenerationLlmClient:
    """Return deterministic Korean generation output for offline tests."""

    model_name = "fixture-korean-generator"

    def __init__(self, script: str | None = None) -> None:
        self._script = script or (
            "충격적이게도 AI의 다음 변화는 답변이 아니라 반복 업무에서 시작됐습니다. "
            "새 기능은 여러 단계로 나뉜 일상 업무를 한 흐름으로 연결하고, "
            "사용자가 매번 같은 내용을 옮겨 적는 시간을 줄이는 데 초점을 맞춥니다. "
            "개발자가 아니어도 문서 정리와 자료 확인처럼 자주 반복하는 작업에 적용할 수 있습니다. "
            "다만 실제 절약 시간이나 성과는 업무 환경에 따라 달라지므로 확정할 수 없습니다. "
            "중요한 건 AI가 화려한 답변을 넘어 우리가 매일 쓰는 프로그램의 사용 방식을 바꾸기 시작했다는 점입니다. "
            "앞으로는 새 기능의 이름보다 내 업무에서 어떤 반복 단계가 사라지는지를 먼저 확인해 보세요."
        )

    async def analyze(self, item: CuratedItem) -> ContentAnalysis:
        """Return analysis grounded in the transformed Stage 2 claims."""

        return ContentAnalysis(
            topic_category="work_productivity",
            audience_relevance="일반 직장인의 반복 업무와 프로그램 사용에 직접 연결됩니다.",
            core_facts=[item.summary, item.key_claim],
            angle="비개발자의 일상 업무에서 달라지는 반복 단계에 초점을 맞춥니다.",
            hook_strategy="AI가 답변보다 반복 업무부터 바꾼다는 반전을 제시합니다.",
            claims_to_avoid=["입력에 없는 절약 시간과 성과 수치"],
            recommended_tone="빠르고 명료한 정보형",
        )

    async def draft(
        self, item: CuratedItem, analysis: ContentAnalysis
    ) -> ScriptDraft:
        """Return the configured deterministic narration."""

        del item, analysis
        return ScriptDraft(script=self._script)

    async def revise(
        self,
        item: CuratedItem,
        analysis: ContentAnalysis,
        script: str,
        *,
        direction: Literal["expand", "shorten"],
        current_duration_seconds: float,
        target_duration_seconds: float,
    ) -> ScriptDraft:
        """Expand or shorten predictably for orchestration tests."""

        del item, analysis, current_duration_seconds, target_duration_seconds
        if direction == "expand":
            return ScriptDraft(
                script=script
                + " 특히 자주 반복하는 한 가지 업무부터 적용 범위를 확인하는 것이 현실적입니다."
            )
        shortened = script[: max(1, int(len(script) * 0.8))].rstrip(" ,")
        return ScriptDraft(script=shortened + ".")

    async def title(
        self, item: CuratedItem, analysis: ContentAnalysis, final_script: str
    ) -> TitlePackage:
        """Return three styles supported by an exact final-script substring."""

        del item, analysis
        evidence = _first_evidence_excerpt(final_script)
        return TitlePackage(
            candidates=[
                TitleCandidate(
                    style="direct_impact",
                    title="직장인의 반복 업무를 바꾸는 AI",
                    supporting_script_excerpt=evidence,
                ),
                TitleCandidate(
                    style="question",
                    title="AI가 반복 업무를 정말 줄여줄까",
                    supporting_script_excerpt=evidence,
                ),
                TitleCandidate(
                    style="conventional_wisdom_reversal",
                    title="AI는 답변보다 반복 업무부터 바꾼다",
                    supporting_script_excerpt=evidence,
                ),
            ],
            selected_title="직장인의 반복 업무를 바꾸는 AI",
        )


def _first_evidence_excerpt(script: str) -> str:
    """Return a nonempty Korean prefix that is an exact script substring."""

    first_sentence, separator, _ = script.partition(".")
    if separator:
        return first_sentence + separator
    return script


def _validate_title_evidence(package: TitlePackage, final_script: str) -> None:
    """Reject title evidence that is not an exact final-script substring."""

    if any(
        candidate.supporting_script_excerpt not in final_script
        for candidate in package.candidates
    ):
        raise GenerationResponseError(
            "title evidence must be an exact final-script substring"
        )
