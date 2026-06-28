from __future__ import annotations

import fcntl
import os
import struct
import time

from oil_gestures.cursor.backends.base import DesktopBounds, MouseButton, MousePoint


_UINPUT_PATH = "/dev/uinput"
_UINPUT_MAX_NAME_SIZE = 80
_ABS_CNT = 64

_EV_SYN = 0x00
_EV_KEY = 0x01
_EV_ABS = 0x03
_SYN_REPORT = 0
_BTN_LEFT = 0x110
_BTN_RIGHT = 0x111
_ABS_X = 0x00
_ABS_Y = 0x01

_IOC_WRITE = 1


def _ioc(direction: int, type_char: str, number: int, size: int) -> int:
    """Mirrors the asm-generic/ioctl.h request-encoding macro used by the kernel uinput ABI."""
    return (direction << 30) | (size << 16) | (ord(type_char) << 8) | number


_UI_SET_EVBIT = _ioc(_IOC_WRITE, "U", 100, struct.calcsize("i"))
_UI_SET_KEYBIT = _ioc(_IOC_WRITE, "U", 101, struct.calcsize("i"))
_UI_SET_ABSBIT = _ioc(_IOC_WRITE, "U", 103, struct.calcsize("i"))
_UI_DEV_CREATE = _ioc(0, "U", 1, 0)
_UI_DEV_DESTROY = _ioc(0, "U", 2, 0)

# struct input_event on 64-bit Linux: long tv_sec; long tv_usec; u16 type; u16 code; s32 value;
_EVENT_STRUCT = struct.Struct("llHHi")


class LinuxUInputMouseBackend:
    """Virtual absolute pointer injected through /dev/uinput.

    Unlike XTEST, uinput events are delivered through the kernel evdev/libinput
    stack, the same path a physical touchscreen uses. That makes them visible
    to the compositor itself, so this backend moves the real cursor under a
    native Wayland session as well as under X11/XWayland.
    """

    name = "uinput"

    def __init__(self, bounds: DesktopBounds) -> None:
        self._bounds = bounds
        self._x = int(bounds.x + bounds.width / 2.0)
        self._y = int(bounds.y + bounds.height / 2.0)
        self._fd = self._create_device(bounds)

    @staticmethod
    def _create_device(bounds: DesktopBounds) -> int:
        try:
            fd = os.open(_UINPUT_PATH, os.O_WRONLY | os.O_NONBLOCK)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{_UINPUT_PATH} does not exist. Load the kernel module with: sudo modprobe uinput"
            ) from exc
        except PermissionError as exc:
            raise RuntimeError(
                f"No permission to open {_UINPUT_PATH}. Add your user to the 'input' group "
                "(sudo usermod -aG input $USER, then log out and back in) or add a udev rule, "
                'e.g. /etc/udev/rules.d/99-uinput.rules with: KERNEL=="uinput", GROUP="input", MODE="0660", '
                "then: sudo udevadm control --reload-rules && sudo udevadm trigger"
            ) from exc

        try:
            fcntl.ioctl(fd, _UI_SET_EVBIT, _EV_KEY)
            fcntl.ioctl(fd, _UI_SET_KEYBIT, _BTN_LEFT)
            fcntl.ioctl(fd, _UI_SET_KEYBIT, _BTN_RIGHT)
            fcntl.ioctl(fd, _UI_SET_EVBIT, _EV_ABS)
            fcntl.ioctl(fd, _UI_SET_ABSBIT, _ABS_X)
            fcntl.ioctl(fd, _UI_SET_ABSBIT, _ABS_Y)
            os.write(fd, LinuxUInputMouseBackend._build_setup_payload(bounds))
            fcntl.ioctl(fd, _UI_DEV_CREATE)
        except OSError as exc:
            os.close(fd)
            raise RuntimeError(f"Could not initialize the uinput virtual mouse device: {exc}") from exc

        # Give the kernel input subsystem a moment to register the new evdev node.
        time.sleep(0.05)
        return fd

    @staticmethod
    def _build_setup_payload(bounds: DesktopBounds) -> bytes:
        name = b"oil-gestures-virtual-mouse"
        name = name[: _UINPUT_MAX_NAME_SIZE - 1].ljust(_UINPUT_MAX_NAME_SIZE, b"\x00")
        input_id = struct.pack("HHHH", 3, 0x1234, 0x5678, 1)  # bustype=BUS_USB, arbitrary vendor/product
        ff_effects_max = struct.pack("I", 0)

        absmax = [0] * _ABS_CNT
        absmin = [0] * _ABS_CNT
        absfuzz = [0] * _ABS_CNT
        absflat = [0] * _ABS_CNT
        absmin[_ABS_X] = int(bounds.x)
        absmax[_ABS_X] = int(bounds.x + max(1.0, bounds.width - 1.0))
        absmin[_ABS_Y] = int(bounds.y)
        absmax[_ABS_Y] = int(bounds.y + max(1.0, bounds.height - 1.0))

        return (
            name
            + input_id
            + ff_effects_max
            + struct.pack(f"{_ABS_CNT}i", *absmax)
            + struct.pack(f"{_ABS_CNT}i", *absmin)
            + struct.pack(f"{_ABS_CNT}i", *absfuzz)
            + struct.pack(f"{_ABS_CNT}i", *absflat)
        )

    def _emit(self, ev_type: int, code: int, value: int) -> None:
        os.write(self._fd, _EVENT_STRUCT.pack(0, 0, ev_type, code, value))

    def _sync(self) -> None:
        self._emit(_EV_SYN, _SYN_REPORT, 0)

    @staticmethod
    def _button_code(button: MouseButton) -> int:
        return _BTN_LEFT if button == "left" else _BTN_RIGHT

    def get_position(self) -> MousePoint:
        return (float(self._x), float(self._y))

    def move_to(self, x: int, y: int) -> bool:
        self._x = int(x)
        self._y = int(y)
        self._emit(_EV_ABS, _ABS_X, self._x)
        self._emit(_EV_ABS, _ABS_Y, self._y)
        self._sync()
        return True

    def drag_to(self, x: int, y: int) -> bool:
        return self.move_to(x, y)

    def click(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        if position is not None:
            self.move_to(int(position[0]), int(position[1]))
        code = self._button_code(button)
        self._emit(_EV_KEY, code, 1)
        self._sync()
        self._emit(_EV_KEY, code, 0)
        self._sync()
        return True

    def button_down(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        if position is not None:
            self.move_to(int(position[0]), int(position[1]))
        self._emit(_EV_KEY, self._button_code(button), 1)
        self._sync()
        return True

    def button_up(self, button: MouseButton, position: MousePoint | None = None) -> bool:
        self._emit(_EV_KEY, self._button_code(button), 0)
        self._sync()
        return True

    def close(self) -> None:
        try:
            fcntl.ioctl(self._fd, _UI_DEV_DESTROY)
        except OSError:
            pass
        os.close(self._fd)
