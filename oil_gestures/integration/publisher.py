from __future__ import annotations

import base64
import time
import uuid
from dataclasses import dataclass

from oil_gestures.core.types import (
    CursorControlResult,
    FramePacket,
    GestureResult,
    LandmarkPacket,
    PointerPosition,
)
from oil_gestures.integration.contracts import (
    CAMERA_FRAME_CONTRACT,
    CONTRACT_VERSION,
    RUNTIME_CONTRACT,
    CameraFrameEvent,
    CursorContract,
    FrameContract,
    GestureContract,
    GesturesContract,
    HandContract,
    MLRuntimeEvent,
    PerformanceContract,
    PointerContract,
    ScreenPositionContract,
)
from oil_gestures.integration.ndjson_server import NDJSONBroadcastServer


@dataclass(frozen=True)
class MLIntegrationPublisherConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    publish_camera: bool = False
    camera_fps: float = 10.0
    jpeg_quality: int = 80

    def __post_init__(self) -> None:
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if self.camera_fps <= 0.0:
            raise ValueError("camera_fps must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")


class MLIntegrationPublisher:
    """Publishes stable ML contracts without depending on UI or 3D code."""

    def __init__(self, config: MLIntegrationPublisherConfig | None = None) -> None:
        self.config = config or MLIntegrationPublisherConfig()
        self.server = NDJSONBroadcastServer(self.config.host, self.config.port)
        self.stream_id = str(uuid.uuid4())
        self._sequence = 0
        self._last_camera_publish = float("-inf")

    @property
    def address(self) -> tuple[str, int]:
        return self.server.address

    @property
    def wants_camera_frame(self) -> bool:
        return self.config.publish_camera and self.server.client_count > 0

    def __enter__(self) -> "MLIntegrationPublisher":
        try:
            self.server.start()
        except OSError as exc:
            raise RuntimeError(
                f"Could not start ML contract server on "
                f"{self.config.host}:{self.config.port} ({exc}). "
                f"The port may already be in use; pick another with --event-port."
            ) from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.server.close()

    @staticmethod
    def _gesture(result: GestureResult | None) -> GestureContract | None:
        if result is None:
            return None
        return GestureContract(
            name=result.name.value,
            confidence=float(result.confidence),
            source=result.source.value,
        )

    def publish_runtime(
        self,
        *,
        frame_packet: FramePacket,
        landmark_packet: LandmarkPacket,
        static_gesture: GestureResult | None,
        dynamic_gesture: GestureResult | None,
        cursor_gesture: GestureResult | None,
        pointer: PointerPosition | None,
        cursor_result: CursorControlResult,
        cursor_enabled: bool,
        cursor_pressed: bool,
        paused: bool,
        fps: float,
        inference_ms: float,
        mirrored: bool,
    ) -> MLRuntimeEvent:
        self._sequence += 1
        timestamp = time.time()
        pointer_contract = None
        if pointer is not None:
            pointer_contract = PointerContract(
                x=float(pointer.x),
                y=float(pointer.y),
                visible=bool(pointer.visible),
                confidence=float(pointer.confidence),
            )
        screen_position = None
        if cursor_result.screen_position is not None:
            screen_position = ScreenPositionContract(
                x=int(cursor_result.screen_position.x),
                y=int(cursor_result.screen_position.y),
            )
        event = MLRuntimeEvent(
            contract=RUNTIME_CONTRACT,
            version=CONTRACT_VERSION,
            stream_id=self.stream_id,
            sequence=self._sequence,
            timestamp=timestamp,
            frame=FrameContract(
                width=int(frame_packet.width),
                height=int(frame_packet.height),
                mirrored=mirrored,
            ),
            performance=PerformanceContract(
                fps=float(fps),
                inference_ms=float(inference_ms),
            ),
            hand=HandContract(
                detected=bool(landmark_packet.hand_detected),
                handedness=landmark_packet.handedness.value,
                confidence=float(landmark_packet.confidence),
                pointer=pointer_contract,
            ),
            gestures=GesturesContract(
                static=self._gesture(static_gesture),
                dynamic=self._gesture(dynamic_gesture),
                cursor=self._gesture(cursor_gesture),
            ),
            cursor=CursorContract(
                enabled=bool(cursor_enabled),
                pressed=bool(cursor_pressed),
                action=cursor_result.action.value,
                screen_position=screen_position,
            ),
            paused=bool(paused),
        )
        self.server.publish(event)
        return event

    def publish_camera_frame(
        self,
        frame,
        runtime_event: MLRuntimeEvent,
        *,
        mirrored: bool,
    ) -> CameraFrameEvent | None:
        if not self.wants_camera_frame:
            return None
        now = time.monotonic()
        interval = 1.0 / self.config.camera_fps
        if now - self._last_camera_publish < interval:
            return None

        import cv2

        encoded, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )
        if not encoded:
            return None
        height, width = frame.shape[:2]
        event = CameraFrameEvent(
            contract=CAMERA_FRAME_CONTRACT,
            version=CONTRACT_VERSION,
            stream_id=runtime_event.stream_id,
            sequence=runtime_event.sequence,
            timestamp=runtime_event.timestamp,
            width=int(width),
            height=int(height),
            mirrored=mirrored,
            encoding="jpeg/base64",
            data_base64=base64.b64encode(buffer.tobytes()).decode("ascii"),
        )
        self.server.publish(event)
        self._last_camera_publish = now
        return event
