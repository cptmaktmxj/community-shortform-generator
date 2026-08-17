"""Strict data contracts for Stage 3 analysis and script generation."""

from datetime import datetime
from typing import Literal

from pydantic import Field, HttpUrl, field_validator, model_validator

from community_shorts.models import StrictModel


TopicCategory = Literal[
    "ai_impact",
    "consumer_software",
    "work_productivity",
    "tech_market",
    "technology_trend",
]
TitleStyle = Literal[
    "direct_impact",
    "question",
    "conventional_wisdom_reversal",
    "curiosity_gap",
    "strong_factual_statement",
]
DurationClass = Literal["ideal", "acceptable"]
GenerationStatus = Literal[
    "pending",
    "analyzed",
    "scripted",
    "duration_failed",
    "title_failed",
    "completed",
    "failed",
]


def _contains_korean(text: str) -> bool:
    """Return whether text contains at least one modern Hangul syllable."""

    return any("가" <= character <= "힣" for character in text)


class ContentAnalysis(StrictModel):
    """Korean editorial analysis created before any narration is drafted."""

    topic_category: TopicCategory
    audience_relevance: str = Field(min_length=1)
    core_facts: list[str] = Field(min_length=1)
    angle: str = Field(min_length=1)
    hook_strategy: str = Field(min_length=1)
    claims_to_avoid: list[str] = Field(default_factory=list)
    recommended_tone: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_korean_analysis(self) -> "ContentAnalysis":
        """Reject analysis prose that is not usable as Korean output guidance."""

        required = [
            self.audience_relevance,
            *self.core_facts,
            self.angle,
            self.hook_strategy,
            self.recommended_tone,
        ]
        if any(not _contains_korean(text) for text in required):
            raise ValueError("analysis narrative fields must contain Korean text")
        if any(not claim.strip() for claim in self.claims_to_avoid):
            raise ValueError("claims_to_avoid entries must not be blank")
        return self


class ScriptDraft(StrictModel):
    """A Korean narration draft returned by the generation model."""

    script: str = Field(min_length=1)

    @field_validator("script")
    @classmethod
    def validate_script(cls, value: str) -> str:
        """Require Korean narration and keep spoken output free of raw URLs."""

        if not _contains_korean(value):
            raise ValueError("script must contain Korean text")
        if "http://" in value.lower() or "https://" in value.lower():
            raise ValueError("script must not contain URLs")
        return value


class TitleCandidate(StrictModel):
    """One title whose claim is supported by an excerpt of the final script."""

    style: TitleStyle
    title: str = Field(min_length=1)
    supporting_script_excerpt: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_korean_text(self) -> "TitleCandidate":
        """Keep both the public title and its evidence excerpt in Korean."""

        if not _contains_korean(self.title) or not _contains_korean(
            self.supporting_script_excerpt
        ):
            raise ValueError("title fields must contain Korean text")
        return self


class TitlePackage(StrictModel):
    """Three distinct title angles plus the selected title."""

    candidates: list[TitleCandidate] = Field(min_length=3, max_length=3)
    selected_title: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_three_distinct_styles(self) -> "TitlePackage":
        """Require three distinct approved styles and a valid selection."""

        styles = {candidate.style for candidate in self.candidates}
        titles = [candidate.title for candidate in self.candidates]
        if len(styles) != 3 or len(set(titles)) != 3:
            raise ValueError("three title styles and distinct titles are required")
        if self.selected_title not in titles:
            raise ValueError("selected_title must match one candidate")
        return self


class TitleCandidatePool(StrictModel):
    """Five distinct editorial angles generated for independent GPT review."""

    candidates: list[TitleCandidate] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def require_five_distinct_styles(self) -> "TitleCandidatePool":
        """Require every ranked-title style exactly once."""

        required_styles = {
            "direct_impact",
            "question",
            "conventional_wisdom_reversal",
            "curiosity_gap",
            "strong_factual_statement",
        }
        styles = {candidate.style for candidate in self.candidates}
        titles = [candidate.title for candidate in self.candidates]
        if styles != required_styles or len(set(titles)) != 5:
            raise ValueError("five title styles and distinct titles are required")
        return self


class TitleCandidateEvaluation(StrictModel):
    """GPT judge scores for one generated title candidate."""

    title: str = Field(min_length=1)
    evidence_support: float = Field(ge=0, le=1)
    clickbait_strength: float = Field(ge=0, le=1)
    mass_appeal: float = Field(ge=0, le=1)
    safety_ok: bool
    reasoning: str = Field(min_length=1)

    @field_validator("reasoning")
    @classmethod
    def require_korean_reasoning(cls, value: str) -> str:
        """Keep the persisted editorial audit understandable in Korean."""

        if not _contains_korean(value):
            raise ValueError("title evaluation reasoning must contain Korean text")
        return value


class TitleJudgeResult(StrictModel):
    """Exactly five distinct structured evaluations from the GPT judge."""

    evaluations: list[TitleCandidateEvaluation] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def require_distinct_evaluated_titles(self) -> "TitleJudgeResult":
        """Reject duplicate reviews that leave a generated candidate unevaluated."""

        titles = [evaluation.title for evaluation in self.evaluations]
        if len(set(titles)) != 5:
            raise ValueError("five distinct title evaluations are required")
        return self


class TitleRankingScore(StrictModel):
    """Auditable GPT judge scores for one public ranked title."""

    title: str = Field(min_length=1)
    evidence_support: float = Field(ge=0, le=1)
    clickbait_strength: float = Field(ge=0, le=1)
    mass_appeal: float = Field(ge=0, le=1)
    safety_ok: bool
    combined_score: float = Field(ge=0, le=1)
    reasoning: str = Field(min_length=1)


class GeneratedScript(StrictModel):
    """Public Stage 3 artifact containing no copied Stage 1 source body."""

    item_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_url: HttpUrl
    analysis: ContentAnalysis
    script: str = Field(min_length=1)
    estimated_duration_seconds: float = Field(ge=0)
    duration_class: DurationClass
    playback_speed: float = Field(gt=0)
    title_candidates: list[str] = Field(min_length=3, max_length=3)
    selected_title: str = Field(min_length=1)
    title_ranking: list[TitleRankingScore] | None = Field(
        default=None, min_length=3, max_length=3
    )
    model: str = Field(min_length=1)
    generated_at: datetime

    @field_validator("script")
    @classmethod
    def validate_final_script(cls, value: str) -> str:
        """Require Korean narration in the persisted artifact."""

        return ScriptDraft(script=value).script

    @field_validator("title_candidates")
    @classmethod
    def validate_title_candidates(cls, value: list[str]) -> list[str]:
        """Require three distinct Korean titles in the public artifact."""

        if len(set(value)) != 3 or any(not _contains_korean(title) for title in value):
            raise ValueError("title_candidates must be three distinct Korean titles")
        return value

    @model_validator(mode="after")
    def validate_generated_artifact(self) -> "GeneratedScript":
        """Keep the selected title and timestamp internally consistent."""

        if self.selected_title not in self.title_candidates:
            raise ValueError("selected_title must match one title candidate")
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return self


class GenerationJob(StrictModel):
    """Durable per-item checkpoint for resumable Stage 3 execution."""

    item_id: str = Field(min_length=1)
    status: GenerationStatus = "pending"
    analysis: ContentAnalysis | None = None
    script: str | None = None
    estimated_duration_seconds: float | None = Field(default=None, ge=0)
    duration_class: Literal["short", "ideal", "acceptable", "long"] | None = None
    title_package: TitlePackage | None = None
    revision_count: int = Field(default=0, ge=0)
    model: str = Field(min_length=1)
    updated_at: datetime
    error: str | None = None

    @field_validator("updated_at")
    @classmethod
    def updated_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        """Reject ambiguous checkpoint timestamps."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")
        return value
