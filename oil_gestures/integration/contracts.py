from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


CONTRACT_VERSION = 1
RUNTIME_CONTRACT = "oil_gestures.ml.runtime"
CAMERA_FRAME_CONTRACT = "oil_gestures.ml.camera_frame"


@dataclass(frozen=True)
class FrameContract:
    width: int
    height: int
    mirrored: bool


@dataclass(frozen=True)
class PerformanceContract:
    fps: float
    inference_ms: float


@dataclass(frozen=True)
class PointerContract:
    x: float
    y: float
    visible: bool
    confidence: float


@dataclass(frozen=True)
class HandContract:
    detected: bool
    handedness: str
    confidence: float
    pointer: PointerContract | None


@dataclass(frozen=True)
class GestureContract:
    name: str
    confidence: float
    source: str


@dataclass(frozen=True)
class GesturesContract:
    static: GestureContract | None
    dynamic: GestureContract | None
    cursor: GestureContract | None


@dataclass(frozen=True)
class ScreenPositionContract:
    x: int
    y: int


@dataclass(frozen=True)
class CursorContract:
    enabled: bool
    pressed: bool
    action: str
    screen_position: ScreenPositionContract | None


@dataclass(frozen=True)
class MLRuntimeEvent:
    """Serializable snapshot emitted once for every processed ML frame."""

    contract: str
    version: int
    stream_id: str
    sequence: int
    timestamp: float
    frame: FrameContract
    performance: PerformanceContract
    hand: HandContract
    gestures: GesturesContract
    cursor: CursorContract
    paused: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class CameraFrameEvent:
    """Optional JPEG camera frame correlated with MLRuntimeEvent by sequence."""

    contract: str
    version: int
    stream_id: str
    sequence: int
    timestamp: float
    width: int
    height: int
    mirrored: bool
    encoding: str
    data_base64: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))
