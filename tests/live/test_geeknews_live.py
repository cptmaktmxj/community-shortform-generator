from datetime import UTC, datetime, timedelta

import httpx
import pytest

from community_shorts.adapters.geeknews import GeekNewsAdapter
from community_shorts.config import SourceConfig
from community_shorts.http import HttpClient


@pytest.mark.live
@pytest.mark.asyncio
async def test_geeknews_live_returns_at_least_one_well_formed_item() -> None:
    config = SourceConfig(
        source_id="geeknews",
        adapter="geeknews",
        url="https://news.hada.io/new",
        language="ko",
        enabled=True,
        fetch_limit=1,
        rate_limit_seconds=0.1,
    )
    async with httpx.AsyncClient(follow_redirects=True) as raw_client:
        client = HttpClient(
            raw_client,
            user_agent="community-shortform-generator/0.1 live-smoke-test",
            rate_limit_seconds=0.1,
            jitter_seconds=0,
        )
        items = await GeekNewsAdapter(config, client).fetch_new(
            datetime.now(UTC) - timedelta(days=7)
        )

    assert items
    assert all(item.source_id == "geeknews" for item in items)
    assert all(str(item.url).startswith("https://news.hada.io/topic?id=") for item in items)
