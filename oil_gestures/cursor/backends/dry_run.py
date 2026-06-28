from __future__ import annotations

from oil_gestures.cursor.backends.base import MouseButton, MousePoint


class DryRunMouseBackend:
    name = "dry-run"

    def __init__(self) -> None:
        self._position: MousePoint = (0.0, 0.0)

    def get_position(self) -> MousePoint:
        return self._position

    def move_to(self, x: int, y: int) -> bool:
        self._position = (float(x), float(y))
        return False

    def drag_to(self, x: int, y: int) -> bool:
        self._position = (float(x), float(y))
        return False

    def click(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        return False

    def button_down(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        return False

    def button_up(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        return False

    def close(self) -> None:
        return None
