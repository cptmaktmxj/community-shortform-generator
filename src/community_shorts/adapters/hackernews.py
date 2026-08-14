"""Official Hacker News Firebase API adapter."""

from datetime import UTC, datetime
from html import unescape
from typing import Any, Sequence

from community_shorts.config import SourceConfig
from community_shorts.http import HttpClient
from community_shorts.models import Metrics, RawItem


API_ROOT = "https://hacker-news.firebaseio.com/v0"


def parse_items(
    records: Sequence[dict[str, Any]],
    *,
    since: datetime,
    fetched_at: datetime,
) -> list[RawItem]:
    """Normalize recent story objects returned by the official HN API."""

    items: list[RawItem] = []
    for record in records:
        if record.get("type") != "story" or record.get("deleted") or record.get("dead"):
            continue
        created = datetime.fromtimestamp(int(record.get("time", 0)), tz=UTC)
        if created < since.astimezone(UTC):
            continue
        item_id = str(record.get("id", ""))
        title = unescape(str(record.get("title", "")).strip())
        if not item_id.isdigit() or not title:
            continue
        items.append(
            RawItem(
                item_id=f"hackernews:{item_id}",
                source_id="hackernews",
                url=str(record.get("url") or f"https://news.ycombinator.com/item?id={item_id}"),
                title=title,
                body=unescape(str(record.get("text", ""))),
                metrics=Metrics(
                    likes=max(int(record.get("score", 0)), 0),
                    comments=max(int(record.get("descendants", 0)), 0),
                ),
                source_language="en",
                fetched_at=fetched_at,
            )
        )
    return items


class HackerNewsAdapter:
    """Collect newest Hacker News stories through the official API."""

    def __init__(self, config: SourceConfig, client: HttpClient) -> None:
        self.source_id = config.source_id
        self._config = config
        self._client = client

    async def fetch_new(self, since: datetime) -> list[RawItem]:
        """Fetch configured newest IDs and normalize their item objects."""

        raw_ids = await self._client.get_json(f"{API_ROOT}/newstories.json")
        ids = list(raw_ids)[: self._config.fetch_limit] if isinstance(raw_ids, list) else []
        records: list[dict[str, Any]] = []
        for item_id in ids:
            record = await self._client.get_json(f"{API_ROOT}/item/{item_id}.json")
            if isinstance(record, dict):
                records.append(record)
        return parse_items(records, since=since, fetched_at=datetime.now(UTC))
