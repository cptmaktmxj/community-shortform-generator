"""Typed data contracts shared across the complete content pipeline."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


SafetyCategory = Literal[
    "actionable_cyber_abuse",
    "weapons_or_illegal_instructions",
    "self_harm",
    "sexual_or_minors",
    "hate_or_harassment",
    "graphic_violence",
    "privacy_or_doxxing",
    "fraud_or_evasion",
]
MassAppealBand = Literal[
    "specialist_only",
    "tech_enthusiast",
    "general_interest",
    "direct_impact",
    "broad_impact",
]
ProvocationBand = Literal[
    "routine",
    "specialist_novelty",
    "challenges_expectation",
    "clear_disruption",
    "broad_shock",
]
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

MASS_APPEAL_SCORES: dict[str, float] = {
    "specialist_only": 0.1,
    "tech_enthusiast": 0.3,
    "general_interest": 0.5,
    "direct_impact": 0.7,
    "broad_impact": 0.9,
}
PROVOCATION_SCORES: dict[str, float] = {
    "routine": 0.1,
    "specialist_novelty": 0.3,
    "challenges_expectation": 0.5,
    "clear_disruption": 0.7,
    "broad_shock": 0.9,
}


def _contains_korean(text: str) -> bool:
    """Return whether text contains at least one modern Hangul syllable."""

    return any("가" <= character <= "힣" for character in text)


class StrictModel(BaseModel):
    """Base contract that rejects undeclared fields."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Metrics(StrictModel):
    """Community reaction counters for an item."""

    likes: int = Field(default=0, ge=0)
    comments: int = Field(default=0, ge=0)


class Comment(StrictModel):
    """A selected community comment used only as curation input."""

    text: str = Field(min_length=1)
    likes: int = Field(default=0, ge=0)


class RawItem(StrictModel):
    """Internal collection artifact containing source material."""

    item_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    url: HttpUrl
    title: str = Field(min_length=1)
    body: str = ""
    metrics: Metrics = Field(default_factory=Metrics)
    top_comments: list[Comment] = Field(default_factory=list)
    source_language: Literal["ko", "en"]
    fetched_at: datetime

    @field_validator("fetched_at")
    @classmethod
    def fetched_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        """Reject ambiguous timestamps that cannot be compared across sources."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("fetched_at must be timezone-aware")
        return value


class LlmAssessment(StrictModel):
    """Schema-constrained response produced by the curation model."""

    provocation_score: float = Field(ge=0.0, le=1.0)
    provocation_band: ProvocationBand
    provocation_reason: str = Field(min_length=1)
    mass_appeal_score: float = Field(ge=0.0, le=1.0)
    mass_appeal_band: MassAppealBand
    mass_appeal_reason: str = Field(min_length=1)
    fidelity_score: float = Field(ge=0.0, le=1.0)
    safety_ok: bool
    safety_reason: str = Field(min_length=1)
    safety_categories: list[SafetyCategory]
    reason: str
    summary: str
    key_claim: str
    hook_points: list[str]
    tone: str
    output_language: Literal["ko"] = "ko"

    @model_validator(mode="after")
    def safety_fields_must_match_decision(self) -> "LlmAssessment":
        """Keep rejected assessments from carrying generated content downstream."""

        explanations = (
            self.safety_reason,
            self.reason,
            self.provocation_reason,
            self.mass_appeal_reason,
        )
        if any(
            not any("가" <= character <= "힣" for character in text)
            for text in explanations
        ):
            raise ValueError("assessment explanations must contain Korean text")
        if self.provocation_score != PROVOCATION_SCORES[self.provocation_band]:
            raise ValueError("provocation_score must match its anchor score")
        if self.mass_appeal_score != MASS_APPEAL_SCORES[self.mass_appeal_band]:
            raise ValueError("mass_appeal_score must match its anchor score")
        if self.safety_ok:
            if self.safety_categories:
                raise ValueError("safety_categories must be empty when safety_ok is true")
            generated_text = " ".join(
                [self.summary, self.key_claim, *self.hook_points, self.tone]
            )
            if not any("가" <= character <= "힣" for character in generated_text):
                raise ValueError("safe assessment output must contain Korean text")
            return self
        if not self.safety_categories:
            raise ValueError("safety_categories must identify at least one rejection category")
        if self.summary or self.key_claim or self.hook_points or self.tone:
            raise ValueError("unsafe assessments must leave generated output fields empty")
        return self


class CuratedItem(StrictModel):
    """Curated artifact containing transformed content without original text."""

    item_id: str
    source_id: str
    url: HttpUrl
    source_language: Literal["ko", "en"]
    output_language: Literal["ko"]
    summary: str
    key_claim: str
    hook_points: list[str]
    tone: str
    pass_: bool = Field(alias="pass", serialization_alias="pass")
    reaction_score: float = Field(ge=0.0, le=1.0)
    provocation_score: float = Field(ge=0.0, le=1.0)
    provocation_band: ProvocationBand
    provocation_reason: str
    mass_appeal_score: float = Field(ge=0.0, le=1.0)
    mass_appeal_band: MassAppealBand
    mass_appeal_reason: str
    curation_score: float = Field(ge=0.0, le=1.0)
    fidelity_score: float = Field(ge=0.0, le=1.0)
    safety_ok: Literal[True]
    safety_reason: str
    safety_categories: list[SafetyCategory]
    curation_reason: str
    model: str


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
    """Public generation artifact containing no copied source body."""

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
    """Durable per-item checkpoint for resumable generation execution."""

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
