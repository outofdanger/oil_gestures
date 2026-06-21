from __future__ import annotations

from oil_gestures.cursor.backends.base import DesktopBounds, MouseBackend, MouseButton, MousePoint
from oil_gestures.cursor.backends.dry_run import DryRunMouseBackend


_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79

_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010


def _windows_api():
    try:
        import ctypes
        import ctypes.wintypes

        return ctypes, ctypes.wintypes, ctypes.windll.user32
    except (AttributeError, ImportError) as exc:
        raise RuntimeError("The Win32 cursor API is unavailable in this Python environment.") from exc


class WindowsMouseBackend:
    name = "win32"

    def __init__(self) -> None:
        self._ctypes, self._wintypes, self._user32 = _windows_api()
        try:
            self._user32.SetProcessDPIAware()
        except Exception:
            pass

    def get_position(self) -> MousePoint:
        point = self._wintypes.POINT()
        self._user32.GetCursorPos(self._ctypes.byref(point))
        return (float(point.x), float(point.y))

    def move_to(self, x: int, y: int) -> bool:
        return bool(self._user32.SetCursorPos(int(x), int(y)))

    def click(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        down, up = self._button_flags(button)
        self._user32.mouse_event(down, 0, 0, 0, 0)
        self._user32.mouse_event(up, 0, 0, 0, 0)
        return True

    def button_down(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        down, _up = self._button_flags(button)
        self._user32.mouse_event(down, 0, 0, 0, 0)
        return True

    def button_up(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        _down, up = self._button_flags(button)
        self._user32.mouse_event(up, 0, 0, 0, 0)
        return True

    def close(self) -> None:
        return None

    @staticmethod
    def _button_flags(button: MouseButton) -> tuple[int, int]:
        if button == "left":
            return (_MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP)
        return (_MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP)


class WindowsPlatformBackend:
    name = "windows"

    def create_mouse_backend(self, dry_run: bool) -> MouseBackend:
        return DryRunMouseBackend() if dry_run else WindowsMouseBackend()

    def get_desktop_bounds(self) -> DesktopBounds:
        _ctypes, _wintypes, user32 = _windows_api()
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
        return DesktopBounds(
            float(user32.GetSystemMetrics(_SM_XVIRTUALSCREEN)),
            float(user32.GetSystemMetrics(_SM_YVIRTUALSCREEN)),
            float(user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN)),
            float(user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN)),
        )

    def accessibility_status(self) -> bool | None:
        return None

    def request_accessibility_prompt(self) -> None:
        return None

    def diagnostics(self) -> dict[str, object]:
        return {}
