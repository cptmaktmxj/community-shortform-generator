from datetime import UTC, datetime

import pytest

from community_shorts.models import Metrics, RawItem
from community_shorts.scoring import curation_score, passes_gates, reaction_scores


def item(item_id: str, source: str, likes: int, comments: int) -> RawItem:
    return RawItem(
        item_id=item_id,
        source_id=source,
        url=f"https://example.com/{item_id}",
        title=item_id,
        body="useful body",
        metrics=Metrics(likes=likes, comments=comments),
        source_language="en",
        fetched_at=datetime(2026, 8, 14, tzinfo=UTC),
    )


def test_curation_score_uses_40_25_35_weights() -> None:
    assert curation_score(1.0, 0.4, 0.2) == pytest.approx(0.57)


def test_fidelity_below_point_75_fails_even_with_high_scores() -> None:
    assert passes_gates(score=0.99, fidelity=0.74, safe=True) is False
    assert passes_gates(score=0.61, fidelity=1.0, safe=True) is False
    assert passes_gates(score=0.99, fidelity=1.0, safe=False) is False
    assert passes_gates(score=0.62, fidelity=0.75, safe=True) is True


def test_reaction_scores_use_source_local_percentiles() -> None:
    items = [
        item("a:low", "a", 0, 0),
        item("a:high", "a", 10, 10),
        item("b:only", "b", 1000, 1000),
    ]

    scores = reaction_scores(items)

    assert scores == {"a:low": 0.0, "a:high": 1.0, "b:only": 0.5}


def test_reaction_scores_average_tied_ranks() -> None:
    items = [
        item("a:1", "a", 10, 0),
        item("a:2", "a", 10, 0),
        item("a:3", "a", 20, 2),
    ]

    scores = reaction_scores(items)

    assert scores["a:1"] == pytest.approx(0.25)
    assert scores["a:2"] == pytest.approx(0.25)
    assert scores["a:3"] == pytest.approx(1.0)
