"""GeekNews listing and discussion adapter."""

import re
from datetime import UTC, datetime
from urllib.parse import parse_qs, urljoin, urlsplit

from selectolax.parser import HTMLParser, Node

from community_shorts.config import SourceConfig
from community_shorts.http import HttpClient
from community_shorts.models import Comment, Metrics, RawItem


_NUMBER = re.compile(r"\d[\d,]*")


def _number(text: str | None) -> int:
    """Return the first non-negative integer embedded in source text."""

    match = _NUMBER.search(text or "")
    return int(match.group(0).replace(",", "")) if match else 0


def _text(node: Node | None) -> str:
    """Return collapsed text for an optional HTML node."""

    return " ".join(node.text(separator=" ", strip=True).split()) if node else ""


def parse_listing(html: str, *, fetched_at: datetime) -> list[RawItem]:
    """Parse normalized GeekNews items from a `/new` listing page."""

    tree = HTMLParser(html)
    items: list[RawItem] = []
    for row in tree.css(".topic_row"):
        title_node = row.css_first(".topic-title-heading")
        if title_node is None:
            title_node = row.css_first(".topictitle")
        if title_node is None:
            continue
        discussion = row.css_first("a[data-topic-comment-topic-id]")
        if discussion is None:
            discussion = row.css_first(".topic-comments")
        if discussion is None:
            discussion = row.css_first('a[href*="topic?id="]')
        discussion_href = discussion.attributes.get("href", "") if discussion else ""
        row_id = row.attributes.get("data-topic-state-id", "") or row.attributes.get("data-id", "")
        query_id = parse_qs(urlsplit(discussion_href).query).get("id", [""])[0]
        item_id = query_id or row_id
        if not item_id.isdigit():
            continue
        points_node = next(
            (
                span
                for span in row.css(".topicinfo span")
                if span.attributes.get("id", "").startswith("tp")
            ),
            None,
        )
        comments = _number(
            discussion.attributes.get("data-topic-comment-count", "") if discussion else ""
        )
        if comments == 0:
            comments = _number(_text(discussion))
        items.append(
            RawItem(
                item_id=f"geeknews:{item_id}",
                source_id="geeknews",
                url=f"https://news.hada.io/topic?id={item_id}",
                title=_text(title_node),
                body=_text(row.css_first(".topicdesc")),
                metrics=Metrics(
                    likes=_number(_text(points_node or row.css_first(".votenum"))),
                    comments=comments,
                ),
                source_language="ko",
                fetched_at=fetched_at,
            )
        )
    return items


def parse_comments(html: str, *, limit: int = 3) -> list[Comment]:
    """Parse and rank visible GeekNews discussion comments."""

    comments: list[Comment] = []
    for node in HTMLParser(html).css(".comment"):
        text = _text(node.css_first(".comment_content") or node.css_first(".comment-body"))
        if text:
            comments.append(Comment(text=text, likes=_number(_text(node.css_first(".votenum")))))
    comments.sort(key=lambda comment: (-comment.likes, comment.text))
    return comments[:limit]


class GeekNewsAdapter:
    """Collect current GeekNews topics and their highest-rated comments."""

    def __init__(self, config: SourceConfig, client: HttpClient) -> None:
        self.source_id = config.source_id
        self._config = config
        self._client = client

    async def fetch_new(self, since: datetime) -> list[RawItem]:
        """Fetch the listing and enrich up to the configured item limit."""

        del since  # GeekNews listing has relative time text but no stable timestamp attribute.
        fetched_at = datetime.now(UTC)
        html = await self._client.get_text(str(self._config.url))
        items = parse_listing(html, fetched_at=fetched_at)[: self._config.fetch_limit]
        enriched: list[RawItem] = []
        for item in items:
            detail = await self._client.get_text(str(item.url))
            enriched.append(item.model_copy(update={"top_comments": parse_comments(detail)}))
        return enriched
