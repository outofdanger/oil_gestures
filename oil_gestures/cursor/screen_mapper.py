from __future__ import annotations

import platform
from dataclasses import dataclass

from oil_gestures.core.constants import DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH
from oil_gestures.core.logger import get_logger
from oil_gestures.core.types import PointerPosition, ScreenPosition

if platform.system() == "Windows":
    import ctypes
elif platform.system() == "Darwin":
    try:
        import Quartz
    except Exception:
        Quartz = None
else:
    Quartz = None

logger = get_logger(__name__)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def max_x(self) -> float:
        return self.x + self.width

    @property
    def max_y(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class ScreenMapperConfig:
    margin_x: float = 0.08
    margin_top: float = 0.10
    margin_bottom: float = 0.12
    invert_y: bool = False
    fallback_width: int = DEFAULT_FRAME_WIDTH
    fallback_height: int = DEFAULT_FRAME_HEIGHT
    strict_screen_bounds: bool = False


class ScreenMapper:
    def __init__(self, config: ScreenMapperConfig) -> None:
        self.config = config
        self.system = platform.system()
        self.bounds = self._get_virtual_desktop_bounds()

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _display_ids(self) -> list[int]:
        if self.system != "Darwin" or Quartz is None:
            return []
        try:
            result = Quartz.CGGetActiveDisplayList(32, None, None)
            if isinstance(result, tuple) and len(result) >= 3:
                error, display_ids, count = result[:3]
                if error == 0 and display_ids:
                    return list(display_ids[:count])
        except Exception:
            pass
        return [Quartz.CGMainDisplayID()]

    def _windows_bounds(self) -> Rect:
        user32 = ctypes.windll.user32
        return Rect(
            float(user32.GetSystemMetrics(76)),
            float(user32.GetSystemMetrics(77)),
            float(user32.GetSystemMetrics(78)),
            float(user32.GetSystemMetrics(79)),
        )

    def _fallback_bounds(self) -> Rect:
        if self.config.strict_screen_bounds:
            raise RuntimeError("Could not read a valid screen size for cursor mapping.")
        fallback = Rect(0.0, 0.0, float(self.config.fallback_width), float(self.config.fallback_height))
        logger.warning(
            "Could not read a valid screen size; using dry-run fallback bounds %sx%s.",
            self.config.fallback_width,
            self.config.fallback_height,
        )
        return fallback

    def _validated(self, bounds: Rect) -> Rect:
        if bounds.width <= 0.0 or bounds.height <= 0.0:
            return self._fallback_bounds()
        return bounds

    def _get_virtual_desktop_bounds(self) -> Rect:
        if self.system == "Windows":
            return self._validated(self._windows_bounds())
        if self.system != "Darwin" or Quartz is None:
            return self._fallback_bounds()

        rects: list[Rect] = []
        for display_id in self._display_ids():
            bounds = Quartz.CGDisplayBounds(display_id)
            rects.append(
                Rect(
                    float(bounds.origin.x),
                    float(bounds.origin.y),
                    float(bounds.size.width),
                    float(bounds.size.height),
                )
            )

        if not rects:
            bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
            return self._validated(
                Rect(
                    float(bounds.origin.x),
                    float(bounds.origin.y),
                    float(bounds.size.width),
                    float(bounds.size.height),
                )
            )

        min_x = min(rect.x for rect in rects)
        min_y = min(rect.y for rect in rects)
        max_x = max(rect.max_x for rect in rects)
        max_y = max(rect.max_y for rect in rects)
        return self._validated(Rect(min_x, min_y, max_x - min_x, max_y - min_y))

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
            x=int(round(self.bounds.x + usable_x * self.bounds.width)),
            y=int(round(self.bounds.y + usable_y * self.bounds.height)),
            timestamp=pointer.timestamp,
        )
