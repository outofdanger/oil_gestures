from __future__ import annotations

from dataclasses import dataclass

from oil_gestures.core.enums import GestureName, RecognitionSource
from oil_gestures.core.types import GestureResult, LandmarkPacket
from oil_gestures.gestures.static.rules import StaticRuleConfig, classify_static_gesture
from oil_gestures.vision.landmark_utils import as_landmark_list


@dataclass(frozen=True)
class StaticRecognizerConfig:
    enabled: bool = True
    min_confidence: float = 0.70
    rule_config: StaticRuleConfig = StaticRuleConfig()


class StaticGestureRecognizer:
    def __init__(self, config: StaticRecognizerConfig | None = None) -> None:
        self.config = config or StaticRecognizerConfig()

    def update(self, packet: LandmarkPacket) -> GestureResult | None:
        if not self.config.enabled or not packet.hand_detected or packet.landmarks is None:
            return None

        landmarks = as_landmark_list(packet.landmarks)
        name = classify_static_gesture(landmarks, self.config.rule_config)
        if name == GestureName.UNKNOWN:
            return None

        return GestureResult(
            name=name,
            confidence=max(self.config.min_confidence, packet.confidence),
            source=RecognitionSource.STATIC_RULES,
            timestamp=packet.timestamp,
        )
