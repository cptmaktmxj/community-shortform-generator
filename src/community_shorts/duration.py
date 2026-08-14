"""Deterministic narration-duration estimation without invoking a TTS service."""

import re
from dataclasses import dataclass
from typing import Literal

from community_shorts.config import ScriptTimingConfig


DurationClassification = Literal["short", "ideal", "acceptable", "long"]

_SPOKEN_TOKEN_PATTERN = re.compile(
    r"[가-힣]|[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z]+|\d+"
)
_PARAGRAPH_PATTERN = re.compile(
    r"(?:[.!?。！？][ \t]*)?(?:\r?\n[ \t]*){2,}"
)
_SHORT_PAUSE_PATTERN = re.compile(r"[,:;，：；]")
_SENTENCE_PAUSE_PATTERN = re.compile(r"[.!?。！？]")


@dataclass(frozen=True, slots=True)
class DurationEstimate:
    """Rounded public estimate and classification from the unrounded duration."""

    seconds: float
    classification: DurationClassification


def spoken_units(text: str) -> float:
    """Calculate language-aware spoken units while ignoring markup and whitespace."""

    units = 0.0
    for match in _SPOKEN_TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        if len(token) == 1 and ("가" <= token <= "힣" or _is_cjk(token)):
            units += 1.0
        elif token.isdigit():
            units += max(1.5, len(token) * 1.5)
        elif token.isupper():
            units += max(2.0, len(token) * 1.25)
        else:
            units += max(2.0, len(token) * 0.65)
    return units


def pause_seconds(text: str) -> float:
    """Estimate punctuation pauses without double-counting paragraph boundaries."""

    paragraph_count = len(_PARAGRAPH_PATTERN.findall(text))
    remaining = _PARAGRAPH_PATTERN.sub(" ", text)
    return (
        paragraph_count * 0.45
        + len(_SHORT_PAUSE_PATTERN.findall(remaining)) * 0.15
        + len(_SENTENCE_PAUSE_PATTERN.findall(remaining)) * 0.35
    )


def classify_duration(
    seconds: float, config: ScriptTimingConfig
) -> DurationClassification:
    """Classify a raw duration against the approved inclusive intervals."""

    if seconds < config.min_seconds:
        return "short"
    if seconds > config.max_seconds:
        return "long"
    if 40.0 <= seconds <= 50.0:
        return "ideal"
    return "acceptable"


def estimate_duration(text: str, config: ScriptTimingConfig) -> DurationEstimate:
    """Estimate playback duration and classify before rounding the public value."""

    natural_seconds = (
        spoken_units(text) / config.base_spoken_units_per_second + pause_seconds(text)
    )
    raw_seconds = natural_seconds / config.playback_speed
    return DurationEstimate(
        seconds=round(raw_seconds, 3),
        classification=classify_duration(raw_seconds, config),
    )


def _is_cjk(character: str) -> bool:
    """Return whether a single character is in a common CJK ideograph block."""

    return "\u3400" <= character <= "\u4dbf" or "\u4e00" <= character <= "\u9fff"
