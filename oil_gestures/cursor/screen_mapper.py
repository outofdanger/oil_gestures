from __future__ import annotations

import platform
from dataclasses import dataclass

from oil_gestures.core.constants import (
    DEFAULT_CURSOR_INVERT_Y,
    DEFAULT_CURSOR_MARGIN_BOTTOM,
    DEFAULT_CURSOR_MARGIN_TOP,
    DEFAULT_CURSOR_MARGIN_X,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_STRICT_SCREEN_BOUNDS,
)
from oil_gestures.core.logger import get_logger
from oil_gestures.core.types import PointerPosition, ScreenPosition
from oil_gestures.cursor.backends import DesktopBounds, PlatformBackend, get_platform_backend


logger = get_logger(__name__)

# Backward-compatible public name; platform backends use DesktopBounds directly.
Rect = DesktopBounds


@dataclass(frozen=True)
class ScreenMapperConfig:
    margin_x: float = DEFAULT_CURSOR_MARGIN_X
    margin_top: float = DEFAULT_CURSOR_MARGIN_TOP
    margin_bottom: float = DEFAULT_CURSOR_MARGIN_BOTTOM
    invert_y: bool = DEFAULT_CURSOR_INVERT_Y
    fallback_width: int = DEFAULT_FRAME_WIDTH
    fallback_height: int = DEFAULT_FRAME_HEIGHT
    strict_screen_bounds: bool = DEFAULT_STRICT_SCREEN_BOUNDS


class ScreenMapper:
    def __init__(self, config: ScreenMapperConfig) -> None:
        self.config = config
        self.system = platform.system()
        self._platform_backend: PlatformBackend = get_platform_backend(self.system)
        self.bounds = self._get_virtual_desktop_bounds()

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _fallback_bounds(self) -> Rect:
        if self.config.strict_screen_bounds:
            raise RuntimeError("Could not read a valid screen size for cursor mapping.")
        fallback = Rect(0.0, 0.0, float(self.config.fallback_width), float(self.config.fallback_height))
        logger.warning(
            "Could not read a valid screen size; using fallback bounds %sx%s.",
            self.config.fallback_width,
            self.config.fallback_height,
        )
        return fallback

    def _validated(self, bounds: DesktopBounds) -> Rect:
        if bounds.width <= 0.0 or bounds.height <= 0.0:
            return self._fallback_bounds()
        return Rect(bounds.x, bounds.y, bounds.width, bounds.height)

    def _get_virtual_desktop_bounds(self) -> Rect:
        try:
            return self._validated(self._platform_backend.get_desktop_bounds())
        except RuntimeError as exc:
            logger.warning("Could not read %s desktop bounds: %s", self.system, exc)
            return self._fallback_bounds()

    def map(self, pointer: PointerPosition) -> ScreenPosition | None:
        if not pointer.visible:
            return None

        usable_x = self._clamp(
            (pointer.x - self.config.margin_x) / max(0.001, 1.0 - 2.0 * self.config.margin_x),
            0.0,
            1.0,
        )
        usable_y = self._clamp(
            (pointer.y - self.config.margin_top)
            / max(0.001, 1.0 - self.config.margin_top - self.config.margin_bottom),
            0.0,
            1.0,
        )
        if self.config.invert_y:
            usable_y = 1.0 - usable_y

        return ScreenPosition(
            x=int(round(self.bounds.x + usable_x * max(0.0, self.bounds.width - 1.0))),
            y=int(round(self.bounds.y + usable_y * max(0.0, self.bounds.height - 1.0))),
            timestamp=pointer.timestamp,
        )
