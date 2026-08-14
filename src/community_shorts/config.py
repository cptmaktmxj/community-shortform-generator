"""Load and validate pipeline configuration."""

import os
from pathlib import Path
from typing import Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError


class ConfigError(ValueError):
    """Raised when pipeline configuration cannot be used safely."""


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


class HttpConfig(BaseModel):
    """HTTP identity and its optional environment override name."""

    model_config = ConfigDict(extra="forbid")

    user_agent: str = Field(min_length=1)
    user_agent_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")


class LlmConfig(BaseModel):
    """Non-secret LLM settings and the environment name holding its key."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["openai", "fixture"]
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    api_key_default: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0)


class RedditConfig(BaseModel):
    """Environment lookup configuration for Reddit OAuth credentials."""

    model_config = ConfigDict(extra="forbid")

    access_token_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")


class ResolvedAppConfig(BaseModel):
    """Runtime values after resolving configured environment lookups."""

    model_config = ConfigDict(extra="forbid")

    user_agent: str
    llm_mode: Literal["openai", "fixture"]
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_timeout_seconds: float
    reddit_access_token: str | None


class AppConfig(BaseModel):
    """Validated non-secret application configuration loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    http: HttpConfig
    llm: LlmConfig
    reddit: RedditConfig

    def resolve(self, environ: Mapping[str, str] | None = None) -> ResolvedAppConfig:
        """Resolve secrets and overrides using environment names declared in YAML."""

        values = os.environ if environ is None else environ
        return ResolvedAppConfig(
            user_agent=values.get(self.http.user_agent_env, self.http.user_agent),
            llm_mode=self.llm.mode,
            llm_base_url=self.llm.base_url,
            llm_model=self.llm.model,
            llm_api_key=values.get(self.llm.api_key_env, self.llm.api_key_default),
            llm_timeout_seconds=self.llm.timeout_seconds,
            reddit_access_token=values.get(self.reddit.access_token_env),
        )


def load_app_config(path: Path) -> AppConfig:
    """Read and validate non-secret runtime configuration from YAML."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return AppConfig.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError, AttributeError) as exc:
        raise ConfigError(f"Invalid application configuration: {exc}") from exc


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
