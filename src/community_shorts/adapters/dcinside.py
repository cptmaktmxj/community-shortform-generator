"""DCInside minor gallery HTML adapter."""

import re
from datetime import UTC, datetime
from urllib.parse import urljoin

from selectolax.parser import HTMLParser, Node

from community_shorts.config import SourceConfig
from community_shorts.http import HttpClient
from community_shorts.models import Metrics, RawItem


_NUMBER = re.compile(r"\d[\d,]*")


def _number(text: str | None) -> int:
    """Return the first integer embedded in gallery text."""

    match = _NUMBER.search(text or "")
    return int(match.group(0).replace(",", "")) if match else 0


def _text(node: Node | None) -> str:
    """Return collapsed text for an optional HTML node."""

    return " ".join(node.text(separator=" ", strip=True).split()) if node else ""


def parse_listing(
    html: str,
    *,
    source: SourceConfig,
    fetched_at: datetime,
) -> list[RawItem]:
    """Normalize ordinary post rows from a DCInside gallery list."""

    items: list[RawItem] = []
    for row in HTMLParser(html).css("tr.ub-content"):
        number = row.attributes.get("data-no", "") or _text(row.css_first(".gall_num"))
        if not number.isdigit():
            continue
        link = row.css_first(".gall_tit a")
        title = _text(link)
        if link is None or not title:
            continue
        replies = _number(_text(row.css_first(".reply_num")))
        items.append(
            RawItem(
                item_id=f"{source.source_id}:{number}",
                source_id=source.source_id,
                url=urljoin(str(source.url), link.attributes.get("href", "")),
                title=title,
                body="",
                metrics=Metrics(
                    likes=_number(_text(row.css_first(".gall_recommend"))),
                    comments=replies,
                ),
                source_language=source.language,
                fetched_at=fetched_at,
            )
        )
    return items


class DCInsideAdapter:
    """Collect visible posts from one configured DCInside gallery."""

    def __init__(self, config: SourceConfig, client: HttpClient) -> None:
        self.source_id = config.source_id
        self._config = config
        self._client = client

    async def fetch_new(self, since: datetime) -> list[RawItem]:
        """Fetch and normalize a gallery list page."""

        del since  # List pages do not expose a full stable timestamp.
        html = await self._client.get_text(str(self._config.url))
        return parse_listing(
            html,
            source=self._config,
            fetched_at=datetime.now(UTC),
        )[: self._config.fetch_limit]
