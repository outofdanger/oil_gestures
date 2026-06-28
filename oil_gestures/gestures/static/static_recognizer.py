from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from oil_gestures.core.constants import DEFAULT_STATIC_CONFIDENCE_THRESHOLD
from oil_gestures.core.enums import GestureName, RecognitionSource
from oil_gestures.core.types import GestureResult, LandmarkPacket


# MediaPipe built-in (canned) gesture category names -> project gesture names.
# Categories not listed here (e.g. "None", "Pointing_Up", "ILoveYou",
# "Thumb_Down") are intentionally ignored by the static subsystem.
DEFAULT_GESTURE_NAME_MAP: dict[str, GestureName] = {
    "Closed_Fist": GestureName.FIST,
    "Open_Palm": GestureName.OPEN_PALM,
    "Thumb_Up": GestureName.THUMB_UP,
    "Victory": GestureName.VICTORY,
}


@dataclass(frozen=True)
class StaticRecognizerConfig:
    enabled: bool = True
    min_confidence: float = DEFAULT_STATIC_CONFIDENCE_THRESHOLD
    gesture_name_map: Mapping[str, GestureName] = field(
        default_factory=lambda: dict(DEFAULT_GESTURE_NAME_MAP)
    )


class StaticGestureRecognizer:
    """
    Static gesture recognition backed by MediaPipe's built-in (canned) gesture
    classifier.

    It reads the canned gesture carried on the LandmarkPacket (produced by
    vision.mediapipe_gesture.MediaPipeGestureRecognizer) and maps the MediaPipe
    category name to a project GestureName. No hand-written geometric rules are
    involved.
    """

    def __init__(self, config: StaticRecognizerConfig | None = None) -> None:
        self.config = config or StaticRecognizerConfig()

    def update(self, packet: LandmarkPacket) -> GestureResult | None:
        if not self.config.enabled or not packet.hand_detected:
            return None
        if packet.raw_gesture is None:
            return None
        if packet.raw_gesture_score < self.config.min_confidence:
            return None

        name = self.config.gesture_name_map.get(packet.raw_gesture)
        if name is None:
            return None

        return GestureResult(
            name=name,
            confidence=packet.raw_gesture_score,
            source=RecognitionSource.MEDIAPIPE,
            timestamp=packet.timestamp,
        )
