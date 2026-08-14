"""Deterministic engagement normalization and selection gates."""

import math
from collections import defaultdict
from typing import Sequence

from community_shorts.models import RawItem


def _average_rank_percentiles(values: list[float]) -> list[float]:
    """Return average-rank percentiles while assigning equal values equal ranks."""

    count = len(values)
    if count == 1:
        return [0.5]
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    result = [0.0] * count
    start = 0
    while start < count:
        end = start + 1
        while end < count and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + end - 1) / 2
        percentile = average_rank / (count - 1)
        for position in range(start, end):
            result[indexed[position][0]] = percentile
        start = end
    return result


def reaction_scores(items: Sequence[RawItem]) -> dict[str, float]:
    """Compute 55/45 source-local percentiles for likes and comments."""

    grouped: dict[str, list[RawItem]] = defaultdict(list)
    for item in items:
        grouped[item.source_id].append(item)

    scores: dict[str, float] = {}
    for source_items in grouped.values():
        likes = [math.log1p(item.metrics.likes) for item in source_items]
        comments = [math.log1p(item.metrics.comments) for item in source_items]
        like_percentiles = _average_rank_percentiles(likes)
        comment_percentiles = _average_rank_percentiles(comments)
        for item, like_score, comment_score in zip(
            source_items,
            like_percentiles,
            comment_percentiles,
            strict=True,
        ):
            scores[item.item_id] = round(like_score * 0.55 + comment_score * 0.45, 6)
    return scores


def curation_score(reaction: float, provocation: float, mass_appeal: float) -> float:
    """Weight audience-fit content signals above source-local reactions."""

    return reaction * 0.20 + provocation * 0.40 + mass_appeal * 0.40


def passes_gates(
    *,
    score: float,
    provocation: float,
    mass_appeal: float,
    fidelity: float,
    safety_ok: bool,
) -> bool:
    """Require combined quality and every independent Stage 2 threshold."""

    return (
        safety_ok
        and score >= 0.62
        and provocation >= 0.35
        and mass_appeal >= 0.45
        and fidelity >= 0.75
    )
