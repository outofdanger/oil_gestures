from __future__ import annotations

import pytest

from oil_gestures.core.enums import GestureName, Handedness, RecognitionSource
from oil_gestures.core.types import LandmarkPacket
from oil_gestures.gestures.static.static_recognizer import (
    StaticGestureRecognizer,
    StaticRecognizerConfig,
)


def _packet(
    *,
    hand_detected: bool = True,
    raw_gesture: str | None = "Closed_Fist",
    raw_gesture_score: float = 0.9,
    timestamp: float = 123.0,
) -> LandmarkPacket:
    return LandmarkPacket(
        hand_detected=hand_detected,
        landmarks=object() if hand_detected else None,
        handedness=Handedness.RIGHT,
        confidence=1.0,
        timestamp=timestamp,
        raw_gesture=raw_gesture,
        raw_gesture_score=raw_gesture_score,
    )


@pytest.mark.parametrize(
    "category, expected",
    [
        ("Closed_Fist", GestureName.FIST),
        ("Open_Palm", GestureName.OPEN_PALM),
        ("Thumb_Up", GestureName.THUMB_UP),
        ("Victory", GestureName.VICTORY),
    ],
)
def test_canned_categories_map_to_gesture_names(category: str, expected: GestureName) -> None:
    recognizer = StaticGestureRecognizer()
    result = recognizer.update(_packet(raw_gesture=category))

    assert result is not None
    assert result.name == expected
    assert result.source == RecognitionSource.MEDIAPIPE
    assert result.confidence == pytest.approx(0.9)
    assert result.timestamp == 123.0


@pytest.mark.parametrize("category", ["None", "Pointing_Up", "ILoveYou", "Thumb_Down"])
def test_unmapped_categories_are_ignored(category: str) -> None:
    recognizer = StaticGestureRecognizer()
    assert recognizer.update(_packet(raw_gesture=category)) is None


def test_low_confidence_is_rejected() -> None:
    recognizer = StaticGestureRecognizer(StaticRecognizerConfig(min_confidence=0.70))
    assert recognizer.update(_packet(raw_gesture_score=0.5)) is None
    assert recognizer.update(_packet(raw_gesture_score=0.8)) is not None


def test_no_hand_or_no_gesture_returns_none() -> None:
    recognizer = StaticGestureRecognizer()
    assert recognizer.update(_packet(hand_detected=False)) is None
    assert recognizer.update(_packet(raw_gesture=None)) is None


def test_disabled_recognizer_returns_none() -> None:
    recognizer = StaticGestureRecognizer(StaticRecognizerConfig(enabled=False))
    assert recognizer.update(_packet()) is None
