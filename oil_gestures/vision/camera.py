from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Iterator

import cv2

from oil_gestures.core.constants import (
    DEFAULT_CAMERA_ID,
    DEFAULT_FPS,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
)
from oil_gestures.core.types import FramePacket


@dataclass(frozen=True)
class CameraConfig:
    device_id: int = DEFAULT_CAMERA_ID
    width: int = DEFAULT_FRAME_WIDTH
    height: int = DEFAULT_FRAME_HEIGHT
    fps: int = DEFAULT_FPS
    mirror: bool = False


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

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None

    @staticmethod
    def _candidate_backends() -> list[int]:
        system = platform.system()
        if system == "Windows":
            return [
                getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
                getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
                cv2.CAP_ANY,
            ]
        if system == "Darwin":
            return [getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY), cv2.CAP_ANY]
        return [cv2.CAP_ANY]

    def _open_capture(self) -> cv2.VideoCapture:
        last_error = "unknown error"
        for backend in self._candidate_backends():
            capture = cv2.VideoCapture(self.config.device_id, backend)
            if not capture.isOpened():
                capture.release()
                last_error = f"camera index {self.config.device_id} did not open"
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            capture.set(cv2.CAP_PROP_FPS, self.config.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok, frame = capture.read()
            if ok and frame is not None:
                return capture

            capture.release()
            last_error = "camera opened but did not return frames"

        raise RuntimeError(f"Could not open webcam: {last_error}. Check Camera permission and camera index.")

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
