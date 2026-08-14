"""Rule-based candidate reduction before expensive local LLM calls."""

import re
from dataclasses import dataclass
from typing import Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from community_shorts.models import RawItem
from community_shorts.scoring import reaction_scores


_TRACKING_KEYS = {"fbclid", "gclid", "ref", "ref_src"}
_SPAM_PATTERNS = (
    re.compile(r"무료\s*코인", re.IGNORECASE),
    re.compile(r"추천인\s*(?:링크|코드)", re.IGNORECASE),
    re.compile(r"광고\s*도배", re.IGNORECASE),
)


@dataclass(frozen=True)
class ScoredRawItem:
    """A source item paired with its deterministic reaction score."""

    raw: RawItem
    reaction_score: float


def canonical_url(url: str) -> str:
    """Remove fragments and known tracking parameters for duplicate detection."""

    parts = urlsplit(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_KEYS
    ]
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/") or "/",
            urlencode(sorted(query)),
            "",
        )
    )


def _is_spam(item: RawItem) -> bool:
    """Identify explicit configured spam phrases in the visible item text."""

    content = f"{item.title}\n{item.body}"
    return any(pattern.search(content) for pattern in _SPAM_PATTERNS)


def prefilter(
    items: Sequence[RawItem],
    *,
    global_limit: int,
    per_source_limit: int,
) -> list[ScoredRawItem]:
    """Deduplicate, remove spam, rank, and enforce global/source call limits."""

    clean = [item for item in items if (item.title.strip() or item.body.strip()) and not _is_spam(item)]
    scores = reaction_scores(clean)
    ranked = sorted(
        clean,
        key=lambda item: (-scores[item.item_id], -item.fetched_at.timestamp(), item.item_id),
    )

    seen_urls: set[str] = set()
    source_counts: dict[str, int] = {}
    selected: list[ScoredRawItem] = []
    for item in ranked:
        normalized = canonical_url(str(item.url))
        if normalized in seen_urls:
            continue
        if source_counts.get(item.source_id, 0) >= per_source_limit:
            continue
        seen_urls.add(normalized)
        source_counts[item.source_id] = source_counts.get(item.source_id, 0) + 1
        selected.append(ScoredRawItem(raw=item, reaction_score=scores[item.item_id]))
        if len(selected) >= global_limit:
            break
    return selected
