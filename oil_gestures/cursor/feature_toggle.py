from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from oil_gestures.core.constants import (
    DEFAULT_CURSOR_TOGGLE_CONFIDENCE,
    DEFAULT_CURSOR_TOGGLE_COOLDOWN_SECONDS,
    DEFAULT_CURSOR_TOGGLE_GESTURE,
)
from oil_gestures.core.enums import GestureName, RecognitionSource
from oil_gestures.core.types import GestureResult


@dataclass(frozen=True)
class CursorFeatureToggleConfig:
    initial_enabled: bool = False
    toggle_gesture: GestureName = GestureName(DEFAULT_CURSOR_TOGGLE_GESTURE)
    min_confidence: float = DEFAULT_CURSOR_TOGGLE_CONFIDENCE
    cooldown_seconds: float = DEFAULT_CURSOR_TOGGLE_COOLDOWN_SECONDS
    allowed_sources: tuple[RecognitionSource, ...] = field(
        default_factory=lambda: (
            RecognitionSource.DYNAMIC_RULES,
            RecognitionSource.DYNAMIC_MODEL,
            RecognitionSource.STATIC_RULES,
        )
    )


@dataclass(frozen=True)
class CursorFeatureToggleResult:
    enabled: bool
    toggled: bool
    source_gesture: GestureName = GestureName.UNKNOWN


class CursorFeatureToggle:
    """
    Owns whether cursor-control is active.

    Gesture recognition keeps running regardless of this state; the cursor layer
    is only allowed to execute when this toggle is enabled.
    """

    def __init__(self, config: CursorFeatureToggleConfig) -> None:
        self.config = config
        self.enabled = config.initial_enabled
        self._last_toggle_time: float | None = None

    def reset(self, enabled: bool | None = None) -> None:
        self.enabled = self.config.initial_enabled if enabled is None else enabled
        self._last_toggle_time = None

    def _ready(self, timestamp: float) -> bool:
        if self._last_toggle_time is None:
            return True
        return timestamp - self._last_toggle_time >= self.config.cooldown_seconds

    def update(
        self,
        results: Iterable[GestureResult | None],
        timestamp: float,
    ) -> CursorFeatureToggleResult:
        for result in results:
            if result is None:
                continue
            if result.name != self.config.toggle_gesture:
                continue
            if result.confidence < self.config.min_confidence:
                continue
            if result.source not in self.config.allowed_sources:
                continue
            if not self._ready(timestamp):
                return CursorFeatureToggleResult(self.enabled, False, result.name)

            self.enabled = not self.enabled
            self._last_toggle_time = timestamp
            return CursorFeatureToggleResult(self.enabled, True, result.name)

        return CursorFeatureToggleResult(self.enabled, False)
