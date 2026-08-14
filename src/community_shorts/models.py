"""Typed data contracts shared across pipeline stages."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


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
    safe: bool
    reason: str
    summary: str
    key_claim: str
    hook_points: list[str]
    tone: str
    output_language: Literal["ko"]


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
    curation_reason: str
    model: str
