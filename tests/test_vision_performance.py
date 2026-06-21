from __future__ import annotations

import numpy as np
import pytest

from app.app_config import load_config
from oil_gestures.core.types import FramePacket
from oil_gestures.vision import camera
from oil_gestures.vision.camera import CameraConfig, CameraStream
from oil_gestures.vision.frame_processor import FrameProcessorConfig, process_frame


class RecordingCapture:
    def __init__(self) -> None:
        self.settings: list[tuple[int, float]] = []

    def set(self, property_id: int, value: float) -> bool:
        self.settings.append((property_id, value))
        return True


def test_linux_camera_requests_mjpg_before_resolution_and_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(camera.platform, "system", lambda: "Linux")
    capture = RecordingCapture()
    stream = CameraStream(CameraConfig(width=1280, height=720, fps=30))

    stream._configure_capture(capture)

    assert capture.settings[0] == (
        camera.cv2.CAP_PROP_FOURCC,
        camera.cv2.VideoWriter_fourcc(*"MJPG"),
    )
    assert (camera.cv2.CAP_PROP_FRAME_WIDTH, 1280) in capture.settings
    assert (camera.cv2.CAP_PROP_FRAME_HEIGHT, 720) in capture.settings
    assert (camera.cv2.CAP_PROP_FPS, 30) in capture.settings


def test_auto_fourcc_does_not_override_non_linux_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(camera.platform, "system", lambda: "Darwin")
    capture = RecordingCapture()
    stream = CameraStream(CameraConfig())

    stream._configure_capture(capture)

    assert all(property_id != camera.cv2.CAP_PROP_FOURCC for property_id, _ in capture.settings)


def test_default_inference_frame_is_smaller_than_display_frame() -> None:
    config = load_config()
    frame = np.zeros((config.camera.height, config.camera.width, 3), dtype=np.uint8)
    packet = FramePacket(
        frame=frame,
        width=config.camera.width,
        height=config.camera.height,
        timestamp=123.0,
    )

    inference_packet = process_frame(
        packet,
        FrameProcessorConfig(
            width=config.mediapipe.input_width,
            height=config.mediapipe.input_height,
        ),
    )

    assert config.camera.fps == 30
    assert config.camera.preferred_fourcc == "AUTO"
    assert (inference_packet.width, inference_packet.height) == (640, 360)
    assert inference_packet.frame.shape == (360, 640, 3)
    assert inference_packet.timestamp == packet.timestamp
