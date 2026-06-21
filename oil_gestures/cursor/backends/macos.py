from __future__ import annotations

from typing import Any

from oil_gestures.core.logger import get_logger
from oil_gestures.cursor.backends.base import DesktopBounds, MouseBackend, MouseButton, MousePoint
from oil_gestures.cursor.backends.dry_run import DryRunMouseBackend


logger = get_logger(__name__)


def _load_quartz() -> Any | None:
    try:
        import Quartz

        return Quartz
    except Exception:
        return None


class MacOSMouseBackend:
    name = "quartz"

    def __init__(self, quartz: Any, platform_backend: "MacOSPlatformBackend") -> None:
        self._quartz = quartz
        self._platform_backend = platform_backend
        self._permission_warning_logged = False

    def get_position(self) -> MousePoint:
        event = self._quartz.CGEventCreate(None)
        location = self._quartz.CGEventGetLocation(event)
        return (float(location.x), float(location.y))

    def move_to(self, x: int, y: int) -> bool:
        point = self._quartz.CGPoint(float(x), float(y))
        moved = False
        try:
            display_id = self._quartz.CGMainDisplayID()
            if display_id:
                self._quartz.CGDisplayMoveCursorToPoint(display_id, point)
                moved = True
        except Exception:
            pass

        try:
            self._quartz.CGWarpMouseCursorPosition(point)
            return True
        except Exception as exc:
            logger.error("Could not warp mouse cursor to (%s, %s): %s", x, y, exc)
            return moved

    def drag_to(self, x: int, y: int) -> bool:
        if not self._events_allowed():
            self.move_to(x, y)
            return False

        point = self._quartz.CGPoint(float(x), float(y))
        try:
            event = self._quartz.CGEventCreateMouseEvent(
                None,
                self._quartz.kCGEventLeftMouseDragged,
                point,
                self._quartz.kCGMouseButtonLeft,
            )
            self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, event)
            return True
        except Exception as exc:
            logger.error("Could not drag mouse cursor to (%s, %s): %s", x, y, exc)
            return False

    def click(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        if not self._events_allowed():
            return False
        down, up, quartz_button = self._button_events(button)
        point = self._event_point(position)
        for event_type in (down, up):
            event = self._quartz.CGEventCreateMouseEvent(None, event_type, point, quartz_button)
            self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, event)
        return True

    def button_down(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        if not self._events_allowed():
            return False
        down, _up, quartz_button = self._button_events(button)
        event = self._quartz.CGEventCreateMouseEvent(
            None,
            down,
            self._event_point(position),
            quartz_button,
        )
        self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, event)
        return True

    def button_up(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        if not self._events_allowed():
            return False
        _down, up, quartz_button = self._button_events(button)
        event = self._quartz.CGEventCreateMouseEvent(
            None,
            up,
            self._event_point(position),
            quartz_button,
        )
        self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, event)
        return True

    def close(self) -> None:
        try:
            self._quartz.CGAssociateMouseAndMouseCursorPosition(True)
        except Exception:
            pass

    def _events_allowed(self) -> bool:
        status = self._platform_backend.accessibility_status()
        if status is not False:
            return True
        if not self._permission_warning_logged:
            logger.error(
                "macOS Accessibility permission is not granted; click/down/up events are blocked. "
                "Cursor movement will still be attempted. Grant permission to the app running Python, "
                "then restart the demo."
            )
            self._platform_backend.request_accessibility_prompt()
            self._permission_warning_logged = True
        return False

    def _event_point(self, position: MousePoint | None) -> Any:
        if position is None:
            position = self.get_position()
        return self._quartz.CGPoint(float(position[0]), float(position[1]))

    def _button_events(self, button: MouseButton) -> tuple[int, int, int]:
        if button == "left":
            return (
                self._quartz.kCGEventLeftMouseDown,
                self._quartz.kCGEventLeftMouseUp,
                self._quartz.kCGMouseButtonLeft,
            )
        return (
            self._quartz.kCGEventRightMouseDown,
            self._quartz.kCGEventRightMouseUp,
            self._quartz.kCGMouseButtonRight,
        )


class MacOSPlatformBackend:
    name = "macos"

    def __init__(self) -> None:
        self._quartz = _load_quartz()

    def create_mouse_backend(self, dry_run: bool) -> MouseBackend:
        if dry_run:
            return DryRunMouseBackend()
        if self._quartz is None:
            raise RuntimeError(
                "Real cursor control on macOS requires pyobjc-framework-Quartz. "
                "Install project dependencies with: python -m pip install -r requirements.txt"
            )
        return MacOSMouseBackend(self._quartz, self)

    def get_desktop_bounds(self) -> DesktopBounds:
        if self._quartz is None:
            raise RuntimeError("Quartz is unavailable; macOS desktop bounds cannot be read.")

        rects = [self._display_bounds(display_id) for display_id in self._display_ids()]
        if not rects:
            rects = [self._display_bounds(self._quartz.CGMainDisplayID())]

        min_x = min(rect.x for rect in rects)
        min_y = min(rect.y for rect in rects)
        max_x = max(rect.x + rect.width for rect in rects)
        max_y = max(rect.y + rect.height for rect in rects)
        return DesktopBounds(min_x, min_y, max_x - min_x, max_y - min_y)

    def accessibility_status(self) -> bool | None:
        if self._quartz is None:
            return None
        try:
            return bool(self._quartz.CGPreflightPostEventAccess())
        except Exception:
            return None

    def request_accessibility_prompt(self) -> None:
        if self._quartz is None:
            return
        try:
            self._quartz.CGRequestPostEventAccess()
        except Exception:
            pass

    def diagnostics(self) -> dict[str, object]:
        return {"quartz_available": self._quartz is not None}

    def _display_ids(self) -> list[int]:
        try:
            result = self._quartz.CGGetActiveDisplayList(32, None, None)
            if isinstance(result, tuple) and len(result) >= 3:
                error, display_ids, count = result[:3]
                if error == 0 and display_ids:
                    return list(display_ids[:count])
        except Exception:
            pass
        return []

    def _display_bounds(self, display_id: int) -> DesktopBounds:
        bounds = self._quartz.CGDisplayBounds(display_id)
        return DesktopBounds(
            float(bounds.origin.x),
            float(bounds.origin.y),
            float(bounds.size.width),
            float(bounds.size.height),
        )
