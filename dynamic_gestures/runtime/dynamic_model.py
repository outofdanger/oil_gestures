from __future__ import annotations

from typing import Protocol

from oil_gestures.core.types import GestureResult, LandmarkPacket


class DynamicGestureModel(Protocol):
    """Contract for learned dynamic-gesture models."""

    def update(self, packet: LandmarkPacket) -> GestureResult | None: ...

    def reset(self) -> None: ...
