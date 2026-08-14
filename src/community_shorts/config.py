"""Load and validate pipeline configuration."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError


class ConfigError(ValueError):
    """Raised when source configuration cannot be used safely."""


class SourceConfig(BaseModel):
    """Configuration shared by one community source adapter."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    adapter: Literal["geeknews", "hackernews", "reddit", "dcinside"]
    url: HttpUrl
    language: Literal["ko", "en"]
    enabled: bool = False
    fetch_limit: int = Field(default=20, ge=1, le=100)
    rate_limit_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    subreddit: str | None = None
    gallery_id: str | None = None


class AppConfig(BaseModel):
    """Runtime settings for storage and the local LLM endpoint."""

    model_config = ConfigDict(extra="forbid")

    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model: str = "K-EXAONE-236B-A23B"
    llm_api_key: str = "EMPTY"
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    user_agent: str = "community-shortform-generator/0.1 (contact: local-operator)"


def load_sources(path: Path) -> list[SourceConfig]:
    """Read a YAML registry and return validated, uniquely identified sources."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw_sources = payload.get("sources", [])
        sources = [SourceConfig.model_validate(item) for item in raw_sources]
    except (OSError, yaml.YAMLError, ValidationError, AttributeError) as exc:
        adapter = _unknown_adapter_from(exc)
        if adapter is not None:
            raise ConfigError(f"Unsupported adapter: {adapter}") from exc
        raise ConfigError(f"Invalid source configuration: {exc}") from exc

    seen: set[str] = set()
    for source in sources:
        if source.source_id in seen:
            raise ConfigError(f"Duplicate source_id: {source.source_id}")
        seen.add(source.source_id)
    return sources


def _unknown_adapter_from(exc: Exception) -> str | None:
    """Extract an unsupported adapter value from a Pydantic validation error."""

    if not isinstance(exc, ValidationError):
        return None
    for error in exc.errors():
        if error.get("loc", ())[-1:] == ("adapter",):
            value = error.get("input")
            return str(value)
    return None
