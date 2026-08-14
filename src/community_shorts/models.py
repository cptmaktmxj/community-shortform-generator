"""Typed data contracts shared across pipeline stages."""

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


class StrictModel(BaseModel):
    """Base contract that rejects undeclared fields."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Metrics(StrictModel):
    """Community reaction counters for an item."""

    likes: int = Field(default=0, ge=0)
    comments: int = Field(default=0, ge=0)


class Comment(StrictModel):
    """A selected community comment used only as Stage 2 input."""

    text: str = Field(min_length=1)
    likes: int = Field(default=0, ge=0)


class RawItem(StrictModel):
    """Internal Stage 1 artifact containing source material."""

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
    """Schema-constrained response produced by the Stage 2 model."""

    provocation_score: float = Field(ge=0.0, le=1.0)
    mass_appeal_score: float = Field(ge=0.0, le=1.0)
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

        explanations = (self.safety_reason, self.reason)
        if any(
            not any("가" <= character <= "힣" for character in text)
            for text in explanations
        ):
            raise ValueError("assessment explanations must contain Korean text")
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
    """Stage 2 artifact containing transformed content without original text."""

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
    mass_appeal_score: float = Field(ge=0.0, le=1.0)
    curation_score: float = Field(ge=0.0, le=1.0)
    fidelity_score: float = Field(ge=0.0, le=1.0)
    safety_ok: Literal[True]
    safety_reason: str
    safety_categories: list[SafetyCategory]
    curation_reason: str
    model: str
