from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from oil_gestures.core.constants import DEFAULT_POINTER_LANDMARK
from oil_gestures.core.types import LandmarkPacket, PointerPosition
from oil_gestures.vision.landmark_utils import as_landmark_list, landmark_index


@dataclass(frozen=True)
class HandPointerConfig:
    pointer_source: str = DEFAULT_POINTER_LANDMARK


class HandPointer:
    def __init__(self, config: HandPointerConfig) -> None:
        self.config = config
        self.pointer_index = landmark_index(config.pointer_source)

    def extract(self, packet: LandmarkPacket) -> PointerPosition:
        if not packet.hand_detected or packet.landmarks is None:
            return PointerPosition(0.0, 0.0, False, 0.0, packet.timestamp)

        landmarks: Sequence = as_landmark_list(packet.landmarks)
        if len(landmarks) <= self.pointer_index:
            return PointerPosition(0.0, 0.0, False, 0.0, packet.timestamp)

        landmark = landmarks[self.pointer_index]
        return PointerPosition(
            x=float(landmark.x),
            y=float(landmark.y),
            visible=True,
            confidence=packet.confidence,
            timestamp=packet.timestamp,
        )
