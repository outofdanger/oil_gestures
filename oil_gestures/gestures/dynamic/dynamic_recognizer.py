from __future__ import annotations

from dataclasses import dataclass

from oil_gestures.core.constants import DEFAULT_DYNAMIC_CONFIDENCE_THRESHOLD, DEFAULT_SEQUENCE_LENGTH
from oil_gestures.core.enums import RecognitionSource
from oil_gestures.core.types import GestureResult, LandmarkPacket
from oil_gestures.gestures.dynamic.dynamic_model import DynamicGestureModel


@dataclass(frozen=True)
class DynamicRecognizerConfig:
    enabled: bool = True
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    min_confidence: float = DEFAULT_DYNAMIC_CONFIDENCE_THRESHOLD


class DynamicGestureRecognizer:
    """Model-only dynamic recognition facade; it contains no cursor rules."""

    def __init__(
        self,
        config: DynamicRecognizerConfig | None = None,
        model: DynamicGestureModel | None = None,
    ) -> None:
        self.config = config or DynamicRecognizerConfig()
        self.model = model

    def reset(self) -> None:
        if self.model is not None:
            self.model.reset()

    def update(self, packet: LandmarkPacket) -> GestureResult | None:
        if not self.config.enabled or self.model is None:
            return None
        result = self.model.update(packet)
        if (
            result is None
            or result.source != RecognitionSource.DYNAMIC_MODEL
            or result.confidence < self.config.min_confidence
        ):
            return None
        return result
