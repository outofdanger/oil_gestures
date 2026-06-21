from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Iterator

import cv2

from oil_gestures.core.constants import (
    DEFAULT_CAMERA_ID,
    DEFAULT_CAMERA_FOURCC,
    DEFAULT_FPS,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_MIRROR_FRAME,
    LINUX_DEFAULT_CAMERA_FOURCC,
    MINIMUM_TARGET_FPS,
    PLATFORM_LINUX,
    PLATFORM_MACOS,
    PLATFORM_WINDOWS,
)
from oil_gestures.core.logger import get_logger
from oil_gestures.core.types import FramePacket


logger = get_logger(__name__)


@dataclass(frozen=True)
class CameraConfig:
    device_id: int = DEFAULT_CAMERA_ID
    width: int = DEFAULT_FRAME_WIDTH
    height: int = DEFAULT_FRAME_HEIGHT
    fps: int = DEFAULT_FPS
    mirror: bool = DEFAULT_MIRROR_FRAME
    preferred_fourcc: str = DEFAULT_CAMERA_FOURCC


class CameraStream:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "CameraStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def open(self) -> None:
        self.capture = self._open_capture()
        diagnostics = self.diagnostics()
        logger.info(
            "Camera opened: backend=%s, actual=%sx%s@%s FPS, FOURCC=%s; "
            "requested=%sx%s@%s FPS",
            diagnostics["backend"],
            diagnostics["width"],
            diagnostics["height"],
            diagnostics["fps"],
            diagnostics["fourcc"],
            self.config.width,
            self.config.height,
            self.config.fps,
        )
        actual_fps = float(diagnostics["fps"])
        if 0.0 < actual_fps < MINIMUM_TARGET_FPS:
            logger.warning(
                "Camera reports only %.1f FPS. On Linux, verify that FOURCC is MJPG; "
                "otherwise try 640x480@30 in configs/default.yaml.",
                actual_fps,
            )

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None

    def diagnostics(self) -> dict[str, object]:
        if self.capture is None:
            return {
                "backend": "closed",
                "width": 0,
                "height": 0,
                "fps": 0.0,
                "fourcc": "unknown",
            }
        try:
            backend_name = self.capture.getBackendName()
        except Exception:
            backend_name = "unknown"
        return {
            "backend": backend_name,
            "width": int(round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": round(float(self.capture.get(cv2.CAP_PROP_FPS)), 1),
            "fourcc": self._decode_fourcc(self.capture.get(cv2.CAP_PROP_FOURCC)),
        }

    @staticmethod
    def _candidate_backends() -> list[int]:
        system = platform.system()
        if system == PLATFORM_WINDOWS:
            candidates = [
                getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
                getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
                cv2.CAP_ANY,
            ]
        elif system == PLATFORM_MACOS:
            candidates = [getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY), cv2.CAP_ANY]
        elif system == PLATFORM_LINUX:
            candidates = [getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), cv2.CAP_ANY]
        else:
            candidates = [cv2.CAP_ANY]
        return list(dict.fromkeys(candidates))

    def _open_capture(self) -> cv2.VideoCapture:
        last_error = "unknown error"
        for backend in self._candidate_backends():
            capture = cv2.VideoCapture(self.config.device_id, backend)
            if not capture.isOpened():
                capture.release()
                last_error = f"camera index {self.config.device_id} did not open"
                continue

            self._configure_capture(capture)

            ok, frame = capture.read()
            if ok and frame is not None:
                return capture

            capture.release()
            last_error = "camera opened but did not return frames"

        hint = " Check /dev/video* access and video-group membership." if platform.system() == PLATFORM_LINUX else ""
        raise RuntimeError(
            f"Could not open webcam: {last_error}. Check Camera permission and camera index.{hint}"
        )

    def _configure_capture(self, capture: cv2.VideoCapture) -> None:
        fourcc = self._preferred_fourcc()
        if fourcc is not None:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _preferred_fourcc(self) -> str | None:
        configured = self.config.preferred_fourcc.strip().upper()
        if configured in {"", "NONE", "DISABLED"}:
            return None
        if configured == "AUTO":
            return LINUX_DEFAULT_CAMERA_FOURCC if platform.system() == PLATFORM_LINUX else None
        if len(configured) != 4:
            raise ValueError("camera.preferred_fourcc must be AUTO, NONE, or a four-character code.")
        return configured

    @staticmethod
    def _decode_fourcc(value: float) -> str:
        code = int(value)
        decoded = "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))
        return decoded.strip("\x00 ") or "unknown"

    def read(self) -> FramePacket | None:
        if self.capture is None:
            raise RuntimeError("CameraStream must be used as a context manager.")

        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None

        height, width = frame.shape[:2]
        return FramePacket(frame=frame, width=width, height=height, timestamp=time.perf_counter())

    def frames(self) -> Iterator[FramePacket]:
        while True:
            packet = self.read()
            if packet is None:
                break
            yield packet


class Camera(CameraStream):
    def __init__(
        self,
        device_id: int = DEFAULT_CAMERA_ID,
        width: int = DEFAULT_FRAME_WIDTH,
        height: int = DEFAULT_FRAME_HEIGHT,
        fps: int = DEFAULT_FPS,
    ) -> None:
        super().__init__(CameraConfig(device_id=device_id, width=width, height=height, fps=fps))
        self.open()
