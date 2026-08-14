"""Adapter protocol and factory for configured community sources."""

from datetime import datetime
from typing import Protocol

from community_shorts.config import SourceConfig
from community_shorts.http import HttpClient
from community_shorts.models import RawItem


class SourceAdapter(Protocol):
    """Fetch new normalized records from one source."""

    source_id: str

    async def fetch_new(self, since: datetime) -> list[RawItem]:
        """Return source records newer than the supplied timestamp."""


def build_adapter(
    config: SourceConfig,
    client: HttpClient,
    *,
    reddit_access_token: str | None = None,
) -> SourceAdapter:
    """Construct the source-specific adapter selected by configuration."""

    if config.adapter == "geeknews":
        from community_shorts.adapters.geeknews import GeekNewsAdapter

        return GeekNewsAdapter(config, client)
    if config.adapter == "hackernews":
        from community_shorts.adapters.hackernews import HackerNewsAdapter

        return HackerNewsAdapter(config, client)
    if config.adapter == "reddit":
        from community_shorts.adapters.reddit import RedditAdapter

        if config.enabled and not reddit_access_token:
            raise ValueError(f"Reddit OAuth token is required for {config.source_id}")
        return RedditAdapter(config, client, access_token=reddit_access_token or "")
    if config.adapter == "dcinside":
        from community_shorts.adapters.dcinside import DCInsideAdapter

        return DCInsideAdapter(config, client)
    raise ValueError(f"Unsupported adapter: {config.adapter}")
