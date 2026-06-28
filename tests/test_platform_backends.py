from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import oil_gestures.cursor.backends.linux_uinput as linux_uinput_module
import oil_gestures.cursor.backends.macos as macos_backend_module
import oil_gestures.cursor.backends.windows as windows_backend_module
from oil_gestures.core.enums import MouseAction
from oil_gestures.core.types import PointerPosition, ScreenPosition
from oil_gestures.cursor import mouse_controller, screen_mapper
from oil_gestures.cursor.backends.base import DesktopBounds
from oil_gestures.cursor.backends.dry_run import DryRunMouseBackend
from oil_gestures.cursor.backends.factory import get_platform_backend
from oil_gestures.cursor.backends.linux import (
    LinuxPlatformBackend,
    LinuxX11MouseBackend,
    linux_session_type,
)
from oil_gestures.cursor.backends.macos import MacOSPlatformBackend
from oil_gestures.cursor.backends.windows import WindowsPlatformBackend
from oil_gestures.cursor.mouse_controller import MouseController, MouseControllerConfig
from oil_gestures.cursor.screen_mapper import Rect, ScreenMapper, ScreenMapperConfig
from oil_gestures.vision import camera
from oil_gestures.vision.camera import CameraStream


class FakeMouseBackend:
    name = "fake-mouse"

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.closed = False

    def get_position(self) -> tuple[float, float]:
        return (321.0, 654.0)

    def move_to(self, x: int, y: int) -> bool:
        self.calls.append(("move", x, y))
        return True

    def drag_to(self, x: int, y: int) -> bool:
        self.calls.append(("drag", x, y))
        return True

    def click(self, button: str, position=None) -> bool:
        self.calls.append(("click", button, position))
        return True

    def button_down(self, button: str, position=None) -> bool:
        self.calls.append(("down", button, position))
        return True

    def button_up(self, button: str, position=None) -> bool:
        self.calls.append(("up", button, position))
        return True

    def close(self) -> None:
        self.closed = True


class FakePlatformBackend:
    name = "fake-platform"

    def __init__(self) -> None:
        self.mouse = FakeMouseBackend()
        self.permission_requested = False

    def create_mouse_backend(self, dry_run: bool):
        return DryRunMouseBackend() if dry_run else self.mouse

    def get_desktop_bounds(self) -> DesktopBounds:
        return DesktopBounds(-1920.0, 0.0, 3840.0, 1080.0)

    def accessibility_status(self) -> bool | None:
        return True

    def request_accessibility_prompt(self) -> None:
        self.permission_requested = True

    def diagnostics(self) -> dict[str, object]:
        return {"adapter": self.name}


def test_factory_routes_every_supported_operating_system() -> None:
    assert isinstance(get_platform_backend("Windows"), WindowsPlatformBackend)
    assert isinstance(get_platform_backend("Darwin"), MacOSPlatformBackend)
    assert isinstance(get_platform_backend("Linux"), LinuxPlatformBackend)


def test_mouse_controller_delegates_to_platform_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    platform_backend = FakePlatformBackend()
    monkeypatch.setattr(mouse_controller.platform, "system", lambda: "TestOS")
    monkeypatch.setattr(mouse_controller, "get_platform_backend", lambda _system: platform_backend)

    controller = MouseController(MouseControllerConfig(dry_run=False))
    position = ScreenPosition(100, 200, 1.0)

    assert controller.backend_name == "fake-mouse"
    assert controller.get_position() == (321.0, 654.0)
    assert controller.move_to(position).action == MouseAction.MOVE
    assert controller.drag_to(position).action == MouseAction.DRAG
    assert controller.click("right", position).executed
    assert controller.mouse_down(position).executed
    assert controller.mouse_up(position).executed
    assert controller.diagnostics()["adapter"] == "fake-platform"

    assert platform_backend.mouse.calls == [
        ("move", 100, 200),
        ("drag", 100, 200),
        ("click", "right", (100.0, 200.0)),
        ("down", "left", (100.0, 200.0)),
        ("up", "left", (100.0, 200.0)),
    ]

    controller.close()
    assert platform_backend.mouse.closed


def test_screen_mapper_delegates_desktop_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    platform_backend = FakePlatformBackend()
    monkeypatch.setattr(screen_mapper.platform, "system", lambda: "TestOS")
    monkeypatch.setattr(screen_mapper, "get_platform_backend", lambda _system: platform_backend)

    mapper = ScreenMapper(ScreenMapperConfig(margin_x=0.0, margin_top=0.0, margin_bottom=0.0))
    mapped = mapper.map(
        PointerPosition(x=0.5, y=0.5, visible=True, confidence=1.0, timestamp=1.0)
    )
    bottom_right = mapper.map(
        PointerPosition(x=1.0, y=1.0, visible=True, confidence=1.0, timestamp=2.0)
    )

    assert mapper.bounds == Rect(-1920.0, 0.0, 3840.0, 1080.0)
    assert mapped == ScreenPosition(0, 540, 1.0)
    assert bottom_right == ScreenPosition(1919, 1079, 2.0)


def test_dry_run_backend_keeps_an_in_memory_position() -> None:
    backend = DryRunMouseBackend()

    assert not backend.move_to(10, 20)
    assert backend.get_position() == (10.0, 20.0)
    assert not backend.drag_to(30, 40)
    assert backend.get_position() == (30.0, 40.0)
    assert not backend.click("left")


def test_windows_backend_owns_win32_cursor_and_desktop_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    mouse_events: list[int] = []
    cursor_positions: list[tuple[int, int]] = []

    class FakePoint:
        x = 12
        y = 34

    class FakeUser32:
        @staticmethod
        def SetProcessDPIAware() -> None:
            return None

        @staticmethod
        def GetSystemMetrics(metric: int) -> int:
            return {76: -1920, 77: 0, 78: 3840, 79: 1080}[metric]

        @staticmethod
        def GetCursorPos(_point) -> int:
            return 1

        @staticmethod
        def SetCursorPos(x: int, y: int) -> int:
            cursor_positions.append((x, y))
            return 1

        @staticmethod
        def mouse_event(event: int, *_args) -> None:
            mouse_events.append(event)

    ctypes_module = SimpleNamespace(byref=lambda value: value)
    wintypes_module = SimpleNamespace(POINT=FakePoint)
    monkeypatch.setattr(
        windows_backend_module,
        "_windows_api",
        lambda: (ctypes_module, wintypes_module, FakeUser32()),
    )

    platform_backend = WindowsPlatformBackend()
    assert platform_backend.get_desktop_bounds() == DesktopBounds(-1920.0, 0.0, 3840.0, 1080.0)

    mouse = platform_backend.create_mouse_backend(dry_run=False)
    assert mouse.get_position() == (12.0, 34.0)
    assert mouse.move_to(100, 200)
    assert mouse.drag_to(150, 250)
    assert mouse.click("right")
    assert mouse.button_down("left")
    assert mouse.button_up("left")
    assert mouse_events == [0x0008, 0x0010, 0x0002, 0x0004]
    assert cursor_positions == [(100, 200), (150, 250)]


def test_macos_backend_owns_quartz_cursor_and_desktop_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeQuartz:
        kCGEventLeftMouseDown = 1
        kCGEventLeftMouseUp = 2
        kCGEventRightMouseDown = 3
        kCGEventRightMouseUp = 4
        kCGEventLeftMouseDragged = 8
        kCGMouseButtonLeft = 5
        kCGMouseButtonRight = 6
        kCGHIDEventTap = 7

        def __init__(self) -> None:
            self.posted: list[tuple[int, object]] = []
            self.reassociated = False

        @staticmethod
        def CGPreflightPostEventAccess() -> bool:
            return True

        @staticmethod
        def CGRequestPostEventAccess() -> None:
            return None

        @staticmethod
        def CGGetActiveDisplayList(_limit, _displays, _count):
            return (0, [1, 2], 2)

        @staticmethod
        def CGDisplayBounds(display_id: int):
            if display_id == 1:
                return SimpleNamespace(
                    origin=SimpleNamespace(x=-1920, y=0),
                    size=SimpleNamespace(width=1920, height=1080),
                )
            return SimpleNamespace(
                origin=SimpleNamespace(x=0, y=0),
                size=SimpleNamespace(width=1920, height=1080),
            )

        @staticmethod
        def CGMainDisplayID() -> int:
            return 2

        @staticmethod
        def CGPoint(x: float, y: float):
            return (x, y)

        @staticmethod
        def CGEventCreate(_source):
            return object()

        @staticmethod
        def CGEventGetLocation(_event):
            return SimpleNamespace(x=12, y=34)

        @staticmethod
        def CGDisplayMoveCursorToPoint(_display_id, _point) -> None:
            return None

        @staticmethod
        def CGWarpMouseCursorPosition(_point) -> None:
            return None

        @staticmethod
        def CGEventCreateMouseEvent(_source, event_type, point, button):
            return (event_type, point, button)

        def CGEventPost(self, tap: int, event) -> None:
            self.posted.append((tap, event))

        def CGAssociateMouseAndMouseCursorPosition(self, associated: bool) -> None:
            self.reassociated = associated

    quartz = FakeQuartz()
    monkeypatch.setattr(macos_backend_module, "_load_quartz", lambda: quartz)

    platform_backend = MacOSPlatformBackend()
    assert platform_backend.get_desktop_bounds() == DesktopBounds(-1920.0, 0.0, 3840.0, 1080.0)
    assert platform_backend.accessibility_status()

    mouse = platform_backend.create_mouse_backend(dry_run=False)
    assert mouse.get_position() == (12.0, 34.0)
    assert mouse.move_to(100, 200)
    assert mouse.drag_to(150, 250)
    assert mouse.click("right", (100.0, 200.0))
    assert mouse.button_down("left", (100.0, 200.0))
    assert mouse.button_up("left", (100.0, 200.0))
    mouse.close()

    assert len(quartz.posted) == 5
    assert quartz.posted[0][1][0] == quartz.kCGEventLeftMouseDragged
    assert quartz.reassociated


def test_linux_camera_prefers_v4l2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera.platform, "system", lambda: "Linux")

    candidates = CameraStream._candidate_backends()

    assert candidates[0] == getattr(camera.cv2, "CAP_V4L2", camera.cv2.CAP_ANY)
    assert candidates[-1] == camera.cv2.CAP_ANY


def test_wayland_session_is_rejected_before_loading_xlib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("DISPLAY", ":1")

    assert linux_session_type() == "wayland"
    with pytest.raises(RuntimeError, match="Wayland"):
        LinuxX11MouseBackend()


def test_wayland_session_routes_real_mouse_to_uinput_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.delenv("DISPLAY", raising=False)

    captured_bounds: list[DesktopBounds] = []

    class FakeUInputBackend:
        name = "uinput"

        def __init__(self, bounds: DesktopBounds) -> None:
            captured_bounds.append(bounds)

    monkeypatch.setattr(linux_uinput_module, "LinuxUInputMouseBackend", FakeUInputBackend)

    platform_backend = LinuxPlatformBackend()

    assert isinstance(platform_backend.create_mouse_backend(dry_run=True), DryRunMouseBackend)

    mouse = platform_backend.create_mouse_backend(dry_run=False)
    assert isinstance(mouse, FakeUInputBackend)
    # No DISPLAY at all (not even XWayland) means bounds fall back to the default frame size.
    assert captured_bounds == [DesktopBounds(0.0, 0.0, 1280.0, 720.0)]


def test_linux_uinput_backend_emits_expected_ioctls_and_events(monkeypatch: pytest.MonkeyPatch) -> None:
    ioctl_calls: list[tuple[int, int]] = []
    writes: list[bytes] = []
    closed: list[int] = []

    def fake_open(path: str, _flags: int) -> int:
        assert path == linux_uinput_module._UINPUT_PATH
        return 7

    def fake_ioctl(fd: int, request: int, arg: int = 0) -> int:
        assert fd == 7
        ioctl_calls.append((request, arg))
        return 0

    def fake_write(fd: int, data: bytes) -> int:
        assert fd == 7
        writes.append(data)
        return len(data)

    def fake_close(fd: int) -> None:
        closed.append(fd)

    monkeypatch.setattr(linux_uinput_module.os, "open", fake_open)
    monkeypatch.setattr(linux_uinput_module.fcntl, "ioctl", fake_ioctl)
    monkeypatch.setattr(linux_uinput_module.os, "write", fake_write)
    monkeypatch.setattr(linux_uinput_module.os, "close", fake_close)
    monkeypatch.setattr(linux_uinput_module.time, "sleep", lambda _seconds: None)

    backend = linux_uinput_module.LinuxUInputMouseBackend(DesktopBounds(0.0, 0.0, 1920.0, 1080.0))

    assert (linux_uinput_module._UI_SET_EVBIT, linux_uinput_module._EV_KEY) in ioctl_calls
    assert (linux_uinput_module._UI_SET_ABSBIT, linux_uinput_module._ABS_Y) in ioctl_calls
    assert (linux_uinput_module._UI_DEV_CREATE, 0) in ioctl_calls
    assert len(writes) == 1 and len(writes[0]) == 1116
    assert backend.get_position() == (960.0, 540.0)

    event = linux_uinput_module._EVENT_STRUCT

    writes.clear()
    assert backend.move_to(100, 200) is True
    assert writes == [
        event.pack(0, 0, linux_uinput_module._EV_ABS, linux_uinput_module._ABS_X, 100),
        event.pack(0, 0, linux_uinput_module._EV_ABS, linux_uinput_module._ABS_Y, 200),
        event.pack(0, 0, linux_uinput_module._EV_SYN, linux_uinput_module._SYN_REPORT, 0),
    ]
    assert backend.get_position() == (100.0, 200.0)

    writes.clear()
    assert backend.click("right") is True
    assert writes == [
        event.pack(0, 0, linux_uinput_module._EV_KEY, linux_uinput_module._BTN_RIGHT, 1),
        event.pack(0, 0, linux_uinput_module._EV_SYN, linux_uinput_module._SYN_REPORT, 0),
        event.pack(0, 0, linux_uinput_module._EV_KEY, linux_uinput_module._BTN_RIGHT, 0),
        event.pack(0, 0, linux_uinput_module._EV_SYN, linux_uinput_module._SYN_REPORT, 0),
    ]

    backend.close()
    assert (linux_uinput_module._UI_DEV_DESTROY, 0) in ioctl_calls
    assert closed == [7]


def test_linux_x11_backend_and_desktop_bounds_use_xlib(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[object, ...]] = []
    displays: list[object] = []

    class FakeRoot:
        @staticmethod
        def query_pointer():
            return SimpleNamespace(root_x=11, root_y=22)

        @staticmethod
        def get_geometry():
            return SimpleNamespace(width=1920, height=1080)

    class FakeDisplay:
        def __init__(self) -> None:
            self.root = FakeRoot()
            self.closed = False
            displays.append(self)

        @staticmethod
        def has_extension(name: str) -> bool:
            return name == "XTEST"

        def screen(self):
            return SimpleNamespace(root=self.root)

        @staticmethod
        def sync() -> None:
            events.append(("sync",))

        def close(self) -> None:
            self.closed = True

    x_module = SimpleNamespace(MotionNotify=6, ButtonPress=4, ButtonRelease=5)
    display_module = SimpleNamespace(Display=FakeDisplay)
    xtest_module = SimpleNamespace(
        fake_input=lambda *args, **kwargs: events.append(("event", *args[1:], kwargs))
    )

    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setitem(sys.modules, "Xlib", SimpleNamespace(X=x_module, display=display_module))
    monkeypatch.setitem(sys.modules, "Xlib.ext", SimpleNamespace(xtest=xtest_module))

    platform_backend = LinuxPlatformBackend()
    assert platform_backend.get_desktop_bounds() == DesktopBounds(0.0, 0.0, 1920.0, 1080.0)

    mouse = platform_backend.create_mouse_backend(dry_run=False)
    assert mouse.get_position() == (11.0, 22.0)
    assert mouse.move_to(100, 200)
    assert mouse.drag_to(150, 250)
    assert mouse.click("right")
    assert mouse.button_down("left")
    assert mouse.button_up("left")
    mouse.close()

    assert ("event", 6, {"x": 100, "y": 200}) in events
    assert ("event", 6, {"x": 150, "y": 250}) in events
    assert ("event", 4, 3, {}) in events
    assert ("event", 5, 3, {}) in events
    assert ("event", 4, 1, {}) in events
    assert ("event", 5, 1, {}) in events
    assert all(display.closed for display in displays)
