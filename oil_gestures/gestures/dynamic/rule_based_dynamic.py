from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque

from oil_gestures.core.enums import GestureName, RecognitionSource
from oil_gestures.core.types import GestureResult, LandmarkPacket
from oil_gestures.vision.landmark_utils import as_landmark_list, pinch_ratio


@dataclass(frozen=True)
class RuleBasedDynamicConfig:
    enabled: bool = True
    fallback_gesture: GestureName = GestureName.POINTING_INDEX
    source: RecognitionSource = RecognitionSource.DYNAMIC_RULES
    confidence: float = 1.0
    require_hand: bool = True
    pinch_tracking_enabled: bool = True
    press_thumb_tip: int = 4
    press_index_tip: int = 8
    press_ratio: float = 0.10
    release_ratio: float = 0.14
    middle_pinch_tracking_enabled: bool = True
    middle_pinch_thumb_tip: int = 4
    middle_pinch_tip: int = 12
    middle_pinch_press_ratio: float = 0.10
    middle_pinch_release_ratio: float = 0.14
    rotation_tracking_enabled: bool = True
    rotation_window: int = 8
    rotation_threshold_radians: float = 0.85
    rotation_cooldown_seconds: float = 0.70


class RuleBasedDynamicRecognizer:
    """
    Lightweight MVP dynamic recognizer.

    It keeps the working pinch behavior from the cursor prototype, but lives in
    the gesture layer so cursor control can be only one optional consumer of
    gesture results.
    """

    def __init__(self, config: RuleBasedDynamicConfig | None = None) -> None:
        self.config = config or RuleBasedDynamicConfig()
        self.pressed = False
        self.middle_pressed = False
        self._angles: Deque[tuple[float, float]] = deque(maxlen=max(2, self.config.rotation_window))
        self._last_rotation_time: float | None = None

    def reset(self) -> None:
        self.pressed = False
        self.middle_pressed = False
        self._angles.clear()
        self._last_rotation_time = None

    def _result(self, name: GestureName, timestamp: float, confidence: float | None = None) -> GestureResult:
        return GestureResult(
            name=name,
            confidence=self.config.confidence if confidence is None else confidence,
            source=self.config.source,
            timestamp=timestamp,
        )

    @staticmethod
    def _angle_delta(start: float, end: float) -> float:
        delta = end - start
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        return delta

    def _rotation_result(self, packet: LandmarkPacket, landmarks) -> GestureResult | None:
        if not self.config.rotation_tracking_enabled or len(landmarks) <= 5:
            return None

        wrist = landmarks[0]
        index_mcp = landmarks[5]
        angle = math.atan2(float(index_mcp.y) - float(wrist.y), float(index_mcp.x) - float(wrist.x))
        self._angles.append((packet.timestamp, angle))
        if len(self._angles) < self._angles.maxlen:
            return None

        first_time, first_angle = self._angles[0]
        last_time, last_angle = self._angles[-1]
        if last_time <= first_time:
            return None

        if self._last_rotation_time is not None:
            if packet.timestamp - self._last_rotation_time < self.config.rotation_cooldown_seconds:
                return None

        delta = self._angle_delta(first_angle, last_angle)
        if abs(delta) < self.config.rotation_threshold_radians:
            return None

        self._last_rotation_time = packet.timestamp
        self._angles.clear()
        name = GestureName.ROTATE_CLOCKWISE if delta > 0.0 else GestureName.ROTATE_COUNTERCLOCKWISE
        confidence = min(1.0, abs(delta) / math.pi)
        return self._result(name, packet.timestamp, confidence=max(confidence, 0.70))

    def update(self, packet: LandmarkPacket) -> GestureResult | None:
        if not self.config.enabled:
            return None
        if self.config.require_hand and (not packet.hand_detected or packet.landmarks is None):
            self.reset()
            return None
        if not packet.hand_detected or packet.landmarks is None:
            return None

        landmarks = as_landmark_list(packet.landmarks)

        if self.config.middle_pinch_tracking_enabled:
            middle_ratio = pinch_ratio(
                landmarks,
                self.config.middle_pinch_thumb_tip,
                self.config.middle_pinch_tip,
            )
            if not self.middle_pressed and middle_ratio <= self.config.middle_pinch_press_ratio:
                self.middle_pressed = True
                return self._result(GestureName.MIDDLE_PINCH, packet.timestamp)
            if self.middle_pressed and middle_ratio >= self.config.middle_pinch_release_ratio:
                self.middle_pressed = False

        if self.config.pinch_tracking_enabled:
            ratio = pinch_ratio(landmarks, self.config.press_thumb_tip, self.config.press_index_tip)
            if not self.pressed and ratio <= self.config.press_ratio:
                self.pressed = True
                return self._result(GestureName.SQUEEZE, packet.timestamp)
            if self.pressed and ratio >= self.config.release_ratio:
                self.pressed = False
                return self._result(GestureName.RELEASE, packet.timestamp)

        rotation = self._rotation_result(packet, landmarks)
        if rotation is not None:
            return rotation

        return self._result(self.config.fallback_gesture, packet.timestamp)
