from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


MouseButton = Literal["left", "right"]
MousePoint = tuple[float, float]


@dataclass(frozen=True)
class DesktopBounds:
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


class MouseBackend(Protocol):
    name: str

    def get_position(self) -> MousePoint: ...

    def move_to(self, x: int, y: int) -> bool: ...

    def click(self, button: MouseButton, position: MousePoint | None = None) -> bool: ...

    def button_down(self, button: MouseButton, position: MousePoint | None = None) -> bool: ...

    def button_up(self, button: MouseButton, position: MousePoint | None = None) -> bool: ...

    def close(self) -> None: ...


class PlatformBackend(Protocol):
    name: str

    def create_mouse_backend(self, dry_run: bool) -> MouseBackend: ...

    def get_desktop_bounds(self) -> DesktopBounds: ...

    def accessibility_status(self) -> bool | None: ...

    def request_accessibility_prompt(self) -> None: ...

    def diagnostics(self) -> dict[str, object]: ...
