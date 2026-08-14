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


def test_curation_score_uses_20_40_40_weights() -> None:
    """Catch source reaction dominating audience-fit content signals."""

    assert curation_score(1.0, 0.5, 0.7) == pytest.approx(0.68)


@pytest.mark.parametrize(
    ("score", "provocation", "mass_appeal", "fidelity", "safety_ok"),
    [
        (0.619, 0.5, 0.5, 0.9, True),
        (0.9, 0.34, 0.7, 0.9, True),
        (0.9, 0.7, 0.44, 0.9, True),
        (0.9, 0.7, 0.7, 0.74, True),
        (0.9, 0.7, 0.7, 0.9, False),
    ],
)
def test_each_independent_gate_can_reject(
    score: float,
    provocation: float,
    mass_appeal: float,
    fidelity: float,
    safety_ok: bool,
) -> None:
    """Catch a strong combined score bypassing one mandatory quality gate."""

    assert passes_gates(
        score=score,
        provocation=provocation,
        mass_appeal=mass_appeal,
        fidelity=fidelity,
        safety_ok=safety_ok,
    ) is False


def test_all_independent_gates_accept_the_boundary() -> None:
    assert passes_gates(
        score=0.62,
        provocation=0.35,
        mass_appeal=0.45,
        fidelity=0.75,
        safety_ok=True,
    ) is True


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
