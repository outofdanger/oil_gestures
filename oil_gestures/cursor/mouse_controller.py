from __future__ import annotations

import platform
import time
from dataclasses import dataclass

from oil_gestures.core.constants import DEFAULT_CURSOR_DRY_RUN
from oil_gestures.core.enums import CursorAction, MouseAction
from oil_gestures.core.logger import get_logger
from oil_gestures.core.types import MouseControlResult, ScreenPosition
from oil_gestures.cursor.backends import MouseBackend, MouseButton, PlatformBackend, get_platform_backend


logger = get_logger(__name__)


@dataclass(frozen=True)
class MouseControllerConfig:
    dry_run: bool = DEFAULT_CURSOR_DRY_RUN


class MouseController:
    """Platform-neutral facade for cursor movement and mouse-button actions."""

    def __init__(self, config: MouseControllerConfig) -> None:
        self.config = config
        self.system = platform.system()
        self._platform_backend: PlatformBackend = get_platform_backend(self.system)
        self._mouse_backend: MouseBackend = self._platform_backend.create_mouse_backend(config.dry_run)

    def accessibility_status(self) -> bool | None:
        return self._platform_backend.accessibility_status()

    def request_accessibility_prompt(self) -> None:
        self._platform_backend.request_accessibility_prompt()

    def diagnostics(self) -> dict[str, object]:
        data: dict[str, object] = {
            "system": self.system,
            "dry_run": self.config.dry_run,
            "backend": self.backend_name,
            "accessibility": self.accessibility_status(),
        }
        data.update(self._platform_backend.diagnostics())
        try:
            data["position"] = self.get_position()
        except Exception as exc:
            data["position_error"] = f"{type(exc).__name__}: {exc}"
        return data

    def get_position(self) -> tuple[float, float]:
        return self._mouse_backend.get_position()

    def move_to(self, position: ScreenPosition) -> MouseControlResult:
        executed = self._mouse_backend.move_to(position.x, position.y)
        if self.config.dry_run:
            logger.info("%s to (%s, %s)", MouseAction.MOVE.value, position.x, position.y)
        return MouseControlResult(MouseAction.MOVE, position, executed, position.timestamp)

    def click(self, button: MouseButton, position: ScreenPosition | None = None) -> MouseControlResult:
        timestamp = position.timestamp if position is not None else time.perf_counter()
        action = MouseAction.LEFT_CLICK if button == "left" else MouseAction.RIGHT_CLICK
        executed = self._mouse_backend.click(button, self._point(position))
        if self.config.dry_run:
            logger.info("%s", action.value)
        return MouseControlResult(action, position, executed, timestamp)

    def mouse_down(self, position: ScreenPosition | None = None) -> MouseControlResult:
        timestamp = position.timestamp if position is not None else time.perf_counter()
        executed = self._mouse_backend.button_down("left", self._point(position))
        if self.config.dry_run:
            logger.info("MOUSE_DOWN")
        return MouseControlResult(MouseAction.MOUSE_DOWN, position, executed, timestamp)

    def mouse_up(self, position: ScreenPosition | None = None) -> MouseControlResult:
        timestamp = position.timestamp if position is not None else time.perf_counter()
        executed = self._mouse_backend.button_up("left", self._point(position))
        if self.config.dry_run:
            logger.info("MOUSE_UP")
        return MouseControlResult(MouseAction.MOUSE_UP, position, executed, timestamp)

    def execute(self, action: CursorAction, position: ScreenPosition | None = None) -> MouseControlResult:
        if action == CursorAction.MOVE_CURSOR and position is not None:
            return self.move_to(position)
        if action == CursorAction.GRAB:
            return self.mouse_down(position)
        if action == CursorAction.RELEASE:
            return self.mouse_up(position)
        timestamp = position.timestamp if position is not None else time.perf_counter()
        return MouseControlResult(MouseAction.NONE, position, False, timestamp)

    @property
    def backend_name(self) -> str:
        return self._mouse_backend.name

    def close(self) -> None:
        self._mouse_backend.close()

    @staticmethod
    def _point(position: ScreenPosition | None) -> tuple[float, float] | None:
        if position is None:
            return None
        return (float(position.x), float(position.y))
