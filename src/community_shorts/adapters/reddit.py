"""OAuth Reddit community adapter shared by configured subreddits."""

from datetime import UTC, datetime
from typing import Any

from community_shorts.config import SourceConfig
from community_shorts.http import HttpClient
from community_shorts.models import Metrics, RawItem


def parse_listing(
    payload: object,
    *,
    source: SourceConfig,
    since: datetime,
    fetched_at: datetime,
) -> list[RawItem]:
    """Normalize a Reddit listing response while excluding stickied posts."""

    if not isinstance(payload, dict):
        return []
    children = payload.get("data", {}).get("children", [])
    items: list[RawItem] = []
    for child in children:
        data: dict[str, Any] = child.get("data", {}) if isinstance(child, dict) else {}
        if data.get("stickied") or data.get("removed_by_category"):
            continue
        created = datetime.fromtimestamp(float(data.get("created_utc", 0)), tz=UTC)
        if created < since.astimezone(UTC):
            continue
        reddit_id = str(data.get("id", "")).strip()
        title = str(data.get("title", "")).strip()
        if not reddit_id or not title:
            continue
        permalink = str(data.get("permalink", ""))
        items.append(
            RawItem(
                item_id=f"{source.source_id}:{reddit_id}",
                source_id=source.source_id,
                url=f"https://www.reddit.com{permalink}",
                title=title,
                body=str(data.get("selftext", "")),
                metrics=Metrics(
                    likes=max(int(data.get("ups", 0)), 0),
                    comments=max(int(data.get("num_comments", 0)), 0),
                ),
                source_language=source.language,
                fetched_at=fetched_at,
            )
        )
    return items


class RedditAdapter:
    """Collect newest posts from one OAuth-authorized subreddit."""

    def __init__(self, config: SourceConfig, client: HttpClient, *, access_token: str) -> None:
        self.source_id = config.source_id
        self._config = config
        self._client = client
        self._access_token = access_token

    async def fetch_new(self, since: datetime) -> list[RawItem]:
        """Fetch a subreddit listing with the supplied bearer token."""

        subreddit = self._config.subreddit or ""
        url = f"https://oauth.reddit.com/r/{subreddit}/new?limit={self._config.fetch_limit}&raw_json=1"
        payload = await self._client.get_json(
            url,
            headers={"Authorization": f"Bearer {self._access_token}"},
        )
        return parse_listing(
            payload,
            source=self._config,
            since=since,
            fetched_at=datetime.now(UTC),
        )
