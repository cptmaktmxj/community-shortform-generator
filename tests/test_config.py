from pathlib import Path

import pytest

from community_shorts.config import ConfigError, load_app_config, load_sources


SOURCES_YAML = """
sources:
  - source_id: geeknews
    adapter: geeknews
    url: https://news.hada.io/new
    language: ko
    enabled: true
  - source_id: hackernews
    adapter: hackernews
    url: https://news.ycombinator.com/newest
    language: en
    enabled: false
  - source_id: reddit_hacking
    adapter: reddit
    url: https://www.reddit.com/r/hacking/
    language: en
    enabled: false
  - source_id: reddit_ai_agents
    adapter: reddit
    url: https://www.reddit.com/r/AI_Agents/
    language: en
    enabled: false
  - source_id: reddit_openai
    adapter: reddit
    url: https://www.reddit.com/r/OpenAI/
    language: en
    enabled: false
  - source_id: reddit_geminiai
    adapter: reddit
    url: https://www.reddit.com/r/GeminiAI/
    language: en
    enabled: false
  - source_id: reddit_claude
    adapter: reddit
    url: https://www.reddit.com/r/claude/
    language: en
    enabled: false
  - source_id: dcinside_singularity
    adapter: dcinside
    url: https://gall.dcinside.com/mgallery/board/lists/?id=thesingularity
    language: ko
    enabled: false
"""

APP_CONFIG_YAML = """
http:
  user_agent: config-user-agent
  user_agent_env: TEST_USER_AGENT
llm:
  mode: fixture
  base_url: http://127.0.0.1:11434/v1
  model: qwen3:8b
  api_key_env: TEST_LLM_API_KEY
  api_key_default: local-development
  timeout_seconds: 45
generation_llm:
  mode: openai
  base_url: https://api.openai.com/v1
  model: gpt-5.4-mini
  api_key_env: TEST_GENERATION_KEY
  timeout_seconds: 120
script:
  target_seconds: 45
  min_seconds: 30
  max_seconds: 60
  playback_speed: 1.2
  base_spoken_units_per_second: 4.3
  max_revisions: 2
reddit:
  access_token_env: TEST_REDDIT_TOKEN
"""


def test_load_sources_registers_eight_sources_and_only_geeknews_is_enabled(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(SOURCES_YAML, encoding="utf-8")

    sources = load_sources(path)

    assert len(sources) == 8
    assert [source.source_id for source in sources if source.enabled] == ["geeknews"]
    assert {source.language for source in sources} == {"ko", "en"}


def test_load_sources_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    duplicate = SOURCES_YAML.replace("source_id: hackernews", "source_id: geeknews")
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ConfigError, match="Duplicate source_id: geeknews"):
        load_sources(path)


def test_load_sources_rejects_unknown_adapter(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(
        SOURCES_YAML.replace("adapter: hackernews", "adapter: unknown"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Unsupported adapter: unknown"):
        load_sources(path)


def test_load_app_config_resolves_secrets_from_declared_environment_names(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch runtime code that hardcodes credential environment variable names."""

    path = tmp_path / "config.yaml"
    path.write_text(APP_CONFIG_YAML, encoding="utf-8")
    monkeypatch.setenv("TEST_USER_AGENT", "environment-user-agent")
    monkeypatch.setenv("TEST_LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("TEST_REDDIT_TOKEN", "test-reddit-token")

    resolved = load_app_config(path).resolve()

    assert resolved.user_agent == "environment-user-agent"
    assert resolved.llm_mode == "fixture"
    assert resolved.llm_base_url == "http://127.0.0.1:11434/v1"
    assert resolved.llm_model == "qwen3:8b"
    assert resolved.llm_api_key == "test-llm-key"
    assert resolved.llm_timeout_seconds == 45
    assert resolved.reddit_access_token == "test-reddit-token"


def test_load_app_config_uses_non_secret_fallbacks_when_environment_is_empty(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch optional local development credentials becoming mandatory."""

    path = tmp_path / "config.yaml"
    path.write_text(APP_CONFIG_YAML, encoding="utf-8")
    for name in ("TEST_USER_AGENT", "TEST_LLM_API_KEY", "TEST_REDDIT_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    resolved = load_app_config(path).resolve()

    assert resolved.user_agent == "config-user-agent"
    assert resolved.llm_api_key == "local-development"
    assert resolved.reddit_access_token is None


def test_generation_config_resolves_key_without_hardcoding(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch the OpenAI secret being placed in committed YAML."""

    path = tmp_path / "config.yaml"
    path.write_text(APP_CONFIG_YAML, encoding="utf-8")
    monkeypatch.setenv("TEST_GENERATION_KEY", "test-secret")

    config = load_app_config(path)
    resolved = config.resolve()

    assert resolved.generation_llm_model == "gpt-5.4-mini"
    assert resolved.generation_llm_api_key == "test-secret"
    assert resolved.script_timing.playback_speed == 1.2
    assert "test-secret" not in config.model_dump_json()


def test_generation_config_rejects_invalid_duration_bounds(tmp_path: Path) -> None:
    """Catch a target interval that can never classify a valid script."""

    path = tmp_path / "config.yaml"
    path.write_text(
        APP_CONFIG_YAML.replace("min_seconds: 30", "min_seconds: 61"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="duration bounds"):
        load_app_config(path)
