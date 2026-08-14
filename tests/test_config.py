from pathlib import Path

import pytest

from community_shorts.config import ConfigError, load_sources


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
