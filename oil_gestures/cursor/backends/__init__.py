from oil_gestures.cursor.backends.base import (
    DesktopBounds,
    MouseBackend,
    MouseButton,
    PlatformBackend,
)
from oil_gestures.cursor.backends.factory import get_platform_backend

__all__ = [
    "DesktopBounds",
    "MouseBackend",
    "MouseButton",
    "PlatformBackend",
    "get_platform_backend",
]
