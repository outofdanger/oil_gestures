from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Literal

if platform.system() == "Windows":
    import ctypes
    import ctypes.wintypes
elif platform.system() == "Darwin":
    try:
        import Quartz
    except Exception:
        Quartz = None
else:
    Quartz = None

from oil_gestures.core.constants import DEFAULT_CURSOR_DRY_RUN
from oil_gestures.core.enums import CursorAction, MouseAction
from oil_gestures.core.logger import get_logger
from oil_gestures.core.types import MouseControlResult, ScreenPosition

MouseButton = Literal["left", "right"]
logger = get_logger(__name__)


@dataclass(frozen=True)
class MouseControllerConfig:
    dry_run: bool = DEFAULT_CURSOR_DRY_RUN


class MouseController:
    def __init__(self, config: MouseControllerConfig) -> None:
        self.config = config
        self.system = platform.system()
        self.user32 = None
        self._dry_run_position = (0.0, 0.0)
        self._permission_warning_logged = False
        self._left_button_down = False

        if self.system == "Windows":
            self.user32 = ctypes.windll.user32
            try:
                self.user32.SetProcessDPIAware()
            except Exception:
                pass
            return

        if self.system == "Darwin" and Quartz is not None:
            return

        if self.config.dry_run:
            return

        raise RuntimeError("Real cursor control is supported on macOS and Windows only.")

    def accessibility_status(self) -> bool | None:
        if self.system != "Darwin" or Quartz is None:
            return None
        try:
            return bool(Quartz.CGPreflightPostEventAccess())
        except Exception:
            return None

    def request_accessibility_prompt(self) -> None:
        if self.system != "Darwin" or Quartz is None:
            return
        try:
            Quartz.CGRequestPostEventAccess()
        except Exception:
            pass

    def _real_events_allowed(self) -> bool:
        if self.config.dry_run or self.system != "Darwin":
            return True

        status = self.accessibility_status()
        if status is not False:
            return True

        if not self._permission_warning_logged:
            logger.error(
                "macOS Accessibility permission is not granted; click/down/up events are blocked. "
                "Cursor movement will still be attempted with CGWarpMouseCursorPosition. "
                "Grant permission to the app running Python, then restart the demo."
            )
            self.request_accessibility_prompt()
            self._permission_warning_logged = True
        return False

    def diagnostics(self) -> dict[str, object]:
        data: dict[str, object] = {
            "system": self.system,
            "dry_run": self.config.dry_run,
            "quartz_available": Quartz is not None,
            "accessibility": self.accessibility_status(),
        }
        try:
            data["position"] = self.get_position()
        except Exception as exc:
            data["position_error"] = f"{type(exc).__name__}: {exc}"
        return data

    def get_position(self) -> tuple[float, float]:
        if self.config.dry_run:
            return self._dry_run_position

        if self.system == "Windows":
            point = ctypes.wintypes.POINT()
            self.user32.GetCursorPos(ctypes.byref(point))
            return (float(point.x), float(point.y))

        event = Quartz.CGEventCreate(None)
        location = Quartz.CGEventGetLocation(event)
        return (float(location.x), float(location.y))

    def move_to(self, position: ScreenPosition) -> MouseControlResult:
        action = MouseAction.MOVE
        if self.config.dry_run:
            self._dry_run_position = (float(position.x), float(position.y))
            logger.info("%s to (%s, %s)", action.value, position.x, position.y)
            return MouseControlResult(action, position, False, position.timestamp)

        if self.system == "Windows":
            self.user32.SetCursorPos(int(position.x), int(position.y))
            return MouseControlResult(action, position, True, position.timestamp)

        cg_point = Quartz.CGPoint(float(position.x), float(position.y))
        moved = False
        try:
            display_id = Quartz.CGMainDisplayID()
            if display_id:
                Quartz.CGDisplayMoveCursorToPoint(display_id, cg_point)
                moved = True
        except Exception:
            pass

        try:
            Quartz.CGWarpMouseCursorPosition(cg_point)
            moved = True
        except Exception as exc:
            logger.error("Could not warp mouse cursor to (%s, %s): %s", position.x, position.y, exc)
            return MouseControlResult(action, position, False, position.timestamp)

        return MouseControlResult(action, position, moved, position.timestamp)

    def _event_point(self, position: ScreenPosition | None = None):
        if position is not None:
            return Quartz.CGPoint(float(position.x), float(position.y))
        x, y = self.get_position()
        return Quartz.CGPoint(float(x), float(y))

    def click(self, button: MouseButton, position: ScreenPosition | None = None) -> MouseControlResult:
        timestamp = position.timestamp if position is not None else time.perf_counter()
        action = MouseAction.LEFT_CLICK if button == "left" else MouseAction.RIGHT_CLICK
        if self.config.dry_run:
            logger.info("%s", action.value)
            return MouseControlResult(action, position, False, timestamp)

        if not self._real_events_allowed():
            return MouseControlResult(action, position, False, timestamp)

        if self.system == "Windows":
            down, up = ((0x0002, 0x0004) if button == "left" else (0x0008, 0x0010))
            self.user32.mouse_event(down, 0, 0, 0, 0)
            self.user32.mouse_event(up, 0, 0, 0, 0)
            return MouseControlResult(action, position, True, timestamp)

        down, up, mouse_button = (
            (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp, Quartz.kCGMouseButtonLeft)
            if button == "left"
            else (Quartz.kCGEventRightMouseDown, Quartz.kCGEventRightMouseUp, Quartz.kCGMouseButtonRight)
        )
        cg_point = self._event_point(position)
        for event_type in (down, up):
            event = Quartz.CGEventCreateMouseEvent(None, event_type, cg_point, mouse_button)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        return MouseControlResult(action, position, True, timestamp)

    def mouse_down(self, position: ScreenPosition | None = None) -> MouseControlResult:
        timestamp = position.timestamp if position is not None else time.perf_counter()
        if self.config.dry_run:
            self._left_button_down = True
            logger.info("MOUSE_DOWN")
            return MouseControlResult(MouseAction.MOUSE_DOWN, position, False, timestamp)

        if not self._real_events_allowed():
            return MouseControlResult(MouseAction.MOUSE_DOWN, position, False, timestamp)

        if self.system == "Windows":
            self.user32.mouse_event(0x0002, 0, 0, 0, 0)
            self._left_button_down = True
            return MouseControlResult(MouseAction.MOUSE_DOWN, position, True, timestamp)

        cg_point = self._event_point(position)
        event = Quartz.CGEventCreateMouseEvent(
            None,
            Quartz.kCGEventLeftMouseDown,
            cg_point,
            Quartz.kCGMouseButtonLeft,
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        self._left_button_down = True
        return MouseControlResult(MouseAction.MOUSE_DOWN, position, True, timestamp)

    def mouse_up(self, position: ScreenPosition | None = None) -> MouseControlResult:
        timestamp = position.timestamp if position is not None else time.perf_counter()
        if self.config.dry_run:
            self._left_button_down = False
            logger.info("MOUSE_UP")
            return MouseControlResult(MouseAction.MOUSE_UP, position, False, timestamp)

        if not self._real_events_allowed():
            self._left_button_down = False
            return MouseControlResult(MouseAction.MOUSE_UP, position, False, timestamp)

        if self.system == "Windows":
            self.user32.mouse_event(0x0004, 0, 0, 0, 0)
            self._left_button_down = False
            return MouseControlResult(MouseAction.MOUSE_UP, position, True, timestamp)

        cg_point = self._event_point(position)
        event = Quartz.CGEventCreateMouseEvent(
            None,
            Quartz.kCGEventLeftMouseUp,
            cg_point,
            Quartz.kCGMouseButtonLeft,
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        self._left_button_down = False
        return MouseControlResult(MouseAction.MOUSE_UP, position, True, timestamp)

    def execute(self, action: CursorAction, position: ScreenPosition | None = None) -> MouseControlResult:
        if action == CursorAction.MOVE_CURSOR and position is not None:
            return self.move_to(position)
        if action == CursorAction.SELECT:
            return self.click("left", position)
        if action == CursorAction.RIGHT_CLICK:
            return self.click("right", position)
        if action == CursorAction.GRAB:
            return self.mouse_down(position)
        if action == CursorAction.RELEASE:
            return self.mouse_up(position)
        timestamp = position.timestamp if position is not None else time.perf_counter()
        return MouseControlResult(MouseAction.NONE, position, False, timestamp)

    def reassociate_hardware_mouse(self) -> None:
        if self.system != "Darwin" or Quartz is None:
            return
        try:
            Quartz.CGAssociateMouseAndMouseCursorPosition(True)
        except Exception:
            pass
