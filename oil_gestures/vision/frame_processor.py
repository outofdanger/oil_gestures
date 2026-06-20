from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2

from oil_gestures.core.types import FramePacket


@dataclass(frozen=True)
class FrameProcessorConfig:
    width: int | None = None
    height: int | None = None
    mirror: bool = False


def _resize_dimensions(frame: Any, width: int | None, height: int | None) -> tuple[int, int] | None:
    frame_height, frame_width = frame.shape[:2]
    if width is None and height is None:
        return None
    if width is not None and height is not None:
        return (int(width), int(height))
    if width is not None:
        scale = float(width) / float(frame_width)
        return (int(width), max(1, int(round(frame_height * scale))))
    scale = float(height) / float(frame_height)
    return (max(1, int(round(frame_width * scale))), int(height))


def resize_frame(frame: Any, width: int | None = None, height: int | None = None) -> Any:
    dimensions = _resize_dimensions(frame, width, height)
    if dimensions is None:
        return frame
    current_height, current_width = frame.shape[:2]
    if dimensions == (current_width, current_height):
        return frame
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_LINEAR)


def mirror_frame(frame: Any) -> Any:
    return cv2.flip(frame, 1)


def _convert_color(value: FramePacket | Any, color_code: int) -> FramePacket | Any:
    if isinstance(value, FramePacket):
        converted = cv2.cvtColor(value.frame, color_code)
        height, width = converted.shape[:2]
        return FramePacket(frame=converted, width=width, height=height, timestamp=value.timestamp)
    return cv2.cvtColor(value, color_code)


def bgr_to_rgb(value: FramePacket | Any) -> FramePacket | Any:
    return _convert_color(value, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(value: FramePacket | Any) -> FramePacket | Any:
    return _convert_color(value, cv2.COLOR_RGB2BGR)


def process_frame(packet: FramePacket, config: FrameProcessorConfig) -> FramePacket:
    frame = resize_frame(packet.frame, config.width, config.height)
    if config.mirror:
        frame = mirror_frame(frame)
    height, width = frame.shape[:2]
    return FramePacket(frame=frame, width=width, height=height, timestamp=packet.timestamp)
