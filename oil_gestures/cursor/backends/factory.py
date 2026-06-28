from __future__ import annotations

from oil_gestures.core.constants import PLATFORM_LINUX, PLATFORM_MACOS, PLATFORM_WINDOWS
from oil_gestures.cursor.backends.base import DesktopBounds, MouseBackend, PlatformBackend
from oil_gestures.cursor.backends.dry_run import DryRunMouseBackend


class UnsupportedPlatformBackend:
    def __init__(self, system: str) -> None:
        self.name = system.lower() or "unknown"
        self._system = system or "Unknown"

    def create_mouse_backend(self, dry_run: bool) -> MouseBackend:
        if dry_run:
            return DryRunMouseBackend()
        raise RuntimeError(f"Real cursor control is not supported on {self._system}.")

    def get_desktop_bounds(self) -> DesktopBounds:
        raise RuntimeError(f"Desktop bounds are not supported on {self._system}.")

    def accessibility_status(self) -> bool | None:
        return None

    def request_accessibility_prompt(self) -> None:
        return None

    def diagnostics(self) -> dict[str, object]:
        return {}


def get_platform_backend(system: str) -> PlatformBackend:
    if system == PLATFORM_WINDOWS:
        from oil_gestures.cursor.backends.windows import WindowsPlatformBackend

        return WindowsPlatformBackend()
    if system == PLATFORM_MACOS:
        from oil_gestures.cursor.backends.macos import MacOSPlatformBackend

        return MacOSPlatformBackend()
    if system == PLATFORM_LINUX:
        from oil_gestures.cursor.backends.linux import LinuxPlatformBackend

        return LinuxPlatformBackend()
    return UnsupportedPlatformBackend(system)
