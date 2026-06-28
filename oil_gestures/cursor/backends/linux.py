from __future__ import annotations

import os

from oil_gestures.core.constants import DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH
from oil_gestures.cursor.backends.base import DesktopBounds, MouseBackend, MouseButton, MousePoint
from oil_gestures.cursor.backends.dry_run import DryRunMouseBackend


_LEFT_BUTTON = 1
_RIGHT_BUTTON = 3


def linux_session_type() -> str:
    configured = os.environ.get("XDG_SESSION_TYPE", "").strip().lower()
    if configured in {"wayland", "x11"}:
        return configured
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return configured or "unknown"


class _LinuxX11Connection:
    def __init__(self, require_xtest: bool) -> None:
        session_type = linux_session_type()
        if require_xtest and session_type == "wayland":
            raise RuntimeError(
                "XTEST-based cursor control is not available in a native Wayland session. "
                "Use the uinput backend (default for real-mouse on Wayland), log in with an "
                "X11/Xorg session, or keep cursor.dry_run enabled."
            )
        if not os.environ.get("DISPLAY"):
            raise RuntimeError(
                "Linux desktop integration needs an X11 display, but DISPLAY is not set. "
                "Run the app from the graphical session or keep cursor.dry_run enabled."
            )

        try:
            from Xlib import X, display
            from Xlib.ext import xtest
        except ImportError as exc:
            raise RuntimeError(
                "Linux X11 integration requires python-xlib. "
                "Install project dependencies with: python -m pip install -r requirements.txt"
            ) from exc

        try:
            self.display = display.Display()
        except Exception as exc:
            raise RuntimeError(f"Could not connect to the X11 display {os.environ.get('DISPLAY')!r}: {exc}") from exc

        if require_xtest and not self.display.has_extension("XTEST"):
            self.display.close()
            raise RuntimeError("The active X11 server does not provide the XTEST extension required for mouse events.")

        self.x = X
        self.xtest = xtest
        self.root = self.display.screen().root
        self._closed = False

    def get_desktop_bounds(self) -> DesktopBounds:
        try:
            query_screens = getattr(self.display, "xinerama_query_screens", None)
            if query_screens is not None:
                reply = query_screens()
                screens = getattr(reply, "screens", None)
                if screens is None:
                    screens = getattr(reply, "_data", {}).get("screens", [])
                if screens:
                    min_x = min(float(screen.x) for screen in screens)
                    min_y = min(float(screen.y) for screen in screens)
                    max_x = max(float(screen.x + screen.width) for screen in screens)
                    max_y = max(float(screen.y + screen.height) for screen in screens)
                    return DesktopBounds(min_x, min_y, max_x - min_x, max_y - min_y)
        except Exception:
            pass

        geometry = self.root.get_geometry()
        return DesktopBounds(0.0, 0.0, float(geometry.width), float(geometry.height))

    def close(self) -> None:
        if not self._closed:
            self.display.close()
            self._closed = True


class LinuxX11MouseBackend:
    name = "x11-xtest"

    def __init__(self) -> None:
        self._connection = _LinuxX11Connection(require_xtest=True)

    def get_position(self) -> MousePoint:
        pointer = self._connection.root.query_pointer()
        return (float(pointer.root_x), float(pointer.root_y))

    def move_to(self, x: int, y: int) -> bool:
        self._connection.xtest.fake_input(
            self._connection.display,
            self._connection.x.MotionNotify,
            x=int(x),
            y=int(y),
        )
        self._connection.display.sync()
        return True

    def drag_to(self, x: int, y: int) -> bool:
        return self.move_to(x, y)

    def click(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        button_number = self._button_number(button)
        self._connection.xtest.fake_input(
            self._connection.display,
            self._connection.x.ButtonPress,
            button_number,
        )
        self._connection.xtest.fake_input(
            self._connection.display,
            self._connection.x.ButtonRelease,
            button_number,
        )
        self._connection.display.sync()
        return True

    def button_down(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        self._connection.xtest.fake_input(
            self._connection.display,
            self._connection.x.ButtonPress,
            self._button_number(button),
        )
        self._connection.display.sync()
        return True

    def button_up(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        self._connection.xtest.fake_input(
            self._connection.display,
            self._connection.x.ButtonRelease,
            self._button_number(button),
        )
        self._connection.display.sync()
        return True

    def close(self) -> None:
        self._connection.close()

    @staticmethod
    def _button_number(button: MouseButton) -> int:
        return _LEFT_BUTTON if button == "left" else _RIGHT_BUTTON


class LinuxPlatformBackend:
    name = "linux"

    def create_mouse_backend(self, dry_run: bool) -> MouseBackend:
        if dry_run:
            return DryRunMouseBackend()
        if linux_session_type() == "wayland":
            from oil_gestures.cursor.backends.linux_uinput import LinuxUInputMouseBackend

            return LinuxUInputMouseBackend(self._bounds_for_real_mouse())
        return LinuxX11MouseBackend()

    def _bounds_for_real_mouse(self) -> DesktopBounds:
        try:
            return self.get_desktop_bounds()
        except RuntimeError:
            return DesktopBounds(0.0, 0.0, float(DEFAULT_FRAME_WIDTH), float(DEFAULT_FRAME_HEIGHT))

    def get_desktop_bounds(self) -> DesktopBounds:
        connection = _LinuxX11Connection(require_xtest=False)
        try:
            return connection.get_desktop_bounds()
        finally:
            connection.close()

    def accessibility_status(self) -> bool | None:
        return None

    def request_accessibility_prompt(self) -> None:
        return None

    def diagnostics(self) -> dict[str, object]:
        return {"display_server": linux_session_type()}
