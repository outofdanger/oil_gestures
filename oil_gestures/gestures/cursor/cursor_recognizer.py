from __future__ import annotations

from dataclasses import dataclass

from oil_gestures.core.constants import (
    DEFAULT_CURSOR_GESTURE_CONFIDENCE,
    DEFAULT_INDEX_PINCH_THUMB_TIP,
    DEFAULT_INDEX_PINCH_TIP,
    DEFAULT_INDEX_PINCH_TRACKING_ENABLED,
    DEFAULT_INDEX_RELEASE_RATIO,
    DEFAULT_INDEX_SQUEEZE_RATIO,
    DEFAULT_MIDDLE_PINCH_PRESS_RATIO,
    DEFAULT_MIDDLE_PINCH_RELEASE_RATIO,
    DEFAULT_MIDDLE_PINCH_THUMB_TIP,
    DEFAULT_MIDDLE_PINCH_TIP,
    DEFAULT_MIDDLE_PINCH_TRACKING_ENABLED,
)
from oil_gestures.core.enums import GestureName, RecognitionSource
from oil_gestures.core.types import GestureResult, LandmarkPacket
from oil_gestures.vision.landmark_utils import as_landmark_list, pinch_ratio


@dataclass(frozen=True)
class CursorGestureConfig:
    enabled: bool = True
    confidence: float = DEFAULT_CURSOR_GESTURE_CONFIDENCE
    index_pinch_tracking_enabled: bool = DEFAULT_INDEX_PINCH_TRACKING_ENABLED
    index_pinch_thumb_tip: int = DEFAULT_INDEX_PINCH_THUMB_TIP
    index_pinch_tip: int = DEFAULT_INDEX_PINCH_TIP
    index_squeeze_ratio: float = DEFAULT_INDEX_SQUEEZE_RATIO
    index_release_ratio: float = DEFAULT_INDEX_RELEASE_RATIO
    middle_pinch_tracking_enabled: bool = DEFAULT_MIDDLE_PINCH_TRACKING_ENABLED
    middle_pinch_thumb_tip: int = DEFAULT_MIDDLE_PINCH_THUMB_TIP
    middle_pinch_tip: int = DEFAULT_MIDDLE_PINCH_TIP
    middle_pinch_press_ratio: float = DEFAULT_MIDDLE_PINCH_PRESS_RATIO
    middle_pinch_release_ratio: float = DEFAULT_MIDDLE_PINCH_RELEASE_RATIO


class CursorGestureRecognizer:
    """Rule-based recognizer used exclusively by cursor control."""

    def __init__(self, config: CursorGestureConfig | None = None) -> None:
        self.config = config or CursorGestureConfig()
        self.index_pressed = False
        self.middle_pressed = False

    def reset(self) -> None:
        self.index_pressed = False
        self.middle_pressed = False

    def update(self, packet: LandmarkPacket) -> GestureResult | None:
        if not self.config.enabled:
            return None
        if not packet.hand_detected or packet.landmarks is None:
            self.reset()
            return None

        landmarks = as_landmark_list(packet.landmarks)

        middle_pinch = self._update_middle_pinch(landmarks, packet.timestamp)
        if middle_pinch is not None:
            return middle_pinch

        index_pinch = self._update_index_pinch(landmarks, packet.timestamp)
        if index_pinch is not None:
            return index_pinch

        return self._result(GestureName.INDEX_MCP, packet.timestamp)

    def _update_middle_pinch(self, landmarks, timestamp: float) -> GestureResult | None:
        if not self.config.middle_pinch_tracking_enabled:
            return None
        ratio = pinch_ratio(
            landmarks,
            self.config.middle_pinch_thumb_tip,
            self.config.middle_pinch_tip,
        )
        if not self.middle_pressed and ratio <= self.config.middle_pinch_press_ratio:
            self.middle_pressed = True
            return self._result(GestureName.MIDDLE_PINCH, timestamp)
        if self.middle_pressed and ratio >= self.config.middle_pinch_release_ratio:
            self.middle_pressed = False
        return None

    def _update_index_pinch(self, landmarks, timestamp: float) -> GestureResult | None:
        if not self.config.index_pinch_tracking_enabled:
            return None
        ratio = pinch_ratio(
            landmarks,
            self.config.index_pinch_thumb_tip,
            self.config.index_pinch_tip,
        )
        if not self.index_pressed and ratio <= self.config.index_squeeze_ratio:
            self.index_pressed = True
            return self._result(GestureName.INDEX_SQUEEZE, timestamp)
        if self.index_pressed and ratio >= self.config.index_release_ratio:
            self.index_pressed = False
            return self._result(GestureName.INDEX_RELEASE, timestamp)
        return None

    def _result(self, name: GestureName, timestamp: float) -> GestureResult:
        return GestureResult(
            name=name,
            confidence=self.config.confidence,
            source=RecognitionSource.CURSOR_RULES,
            timestamp=timestamp,
        )
