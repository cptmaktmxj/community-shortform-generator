import pytest

from community_shorts.config import ScriptTimingConfig
from community_shorts.duration import (
    classify_duration,
    estimate_duration,
    pause_seconds,
    spoken_units,
)


def timing(**overrides: float | int) -> ScriptTimingConfig:
    """Build a valid timing configuration with optional test-specific values."""

    values: dict[str, float | int] = {
        "target_seconds": 45,
        "min_seconds": 30,
        "max_seconds": 60,
        "playback_speed": 1.2,
        "base_spoken_units_per_second": 4.0,
        "max_revisions": 2,
    }
    return ScriptTimingConfig(**(values | overrides))


def test_hangul_and_sentence_pause_are_scaled_by_playback_speed() -> None:
    """Catch playback speed being applied only to words or only to pauses."""

    estimate = estimate_duration(
        "가나다라.", timing(min_seconds=0.5, target_seconds=1, max_seconds=2)
    )

    assert estimate.seconds == pytest.approx((4 / 4.0 + 0.35) / 1.2)


def test_paragraph_pause_replaces_adjacent_sentence_pause() -> None:
    """Catch a paragraph boundary being double-counted as two separate pauses."""

    assert pause_seconds("첫 문장.\n\n둘째 문장") == pytest.approx(0.45)


def test_english_abbreviation_and_number_cost_more_than_raw_character_count() -> None:
    """Catch English abbreviations and numbers being treated as silent markup."""

    assert spoken_units("GPT-5") > spoken_units("가나다")


@pytest.mark.parametrize(
    ("seconds", "classification"),
    [
        (29.9, "short"),
        (30.0, "acceptable"),
        (40.0, "ideal"),
        (50.0, "ideal"),
        (60.0, "acceptable"),
        (60.1, "long"),
    ],
)
def test_duration_boundaries(seconds: float, classification: str) -> None:
    """Lock the approved inclusive 30-60 and ideal 40-50 second intervals."""

    assert classify_duration(seconds, timing()) == classification


def test_public_estimate_is_rounded_only_after_classification() -> None:
    """Catch rounding an overlong result into the accepted duration interval."""

    config = timing(
        min_seconds=0.1,
        target_seconds=0.2,
        max_seconds=0.3,
        playback_speed=1.0,
        base_spoken_units_per_second=3 / 0.3004,
    )

    estimate = estimate_duration("가나다", config)

    assert estimate.seconds == 0.3
    assert estimate.classification == "long"
