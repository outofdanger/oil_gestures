from __future__ import annotations

import base64
import json
import socket
import time
from pathlib import Path

import numpy as np
import pytest

from oil_gestures.core.enums import (
    CursorAction,
    GestureName,
    Handedness,
    RecognitionSource,
)
from oil_gestures.core.types import (
    CursorControlResult,
    FramePacket,
    GestureResult,
    LandmarkPacket,
    PointerPosition,
    ScreenPosition,
)
from oil_gestures.integration.contracts import (
    CAMERA_FRAME_CONTRACT,
    CONTRACT_VERSION,
    RUNTIME_CONTRACT,
)
from oil_gestures.integration.ndjson_server import NDJSONBroadcastServer
from oil_gestures.integration.publisher import (
    MLIntegrationPublisher,
    MLIntegrationPublisherConfig,
)


class RecordingServer:
    client_count = 1

    def __init__(self) -> None:
        self.events = []

    def publish(self, event) -> None:
        self.events.append(event)


def build_runtime_event(publisher: MLIntegrationPublisher):
    timestamp = 10.0
    return publisher.publish_runtime(
        frame_packet=FramePacket(np.zeros((2, 3, 3), dtype=np.uint8), 3, 2, timestamp),
        landmark_packet=LandmarkPacket(
            hand_detected=True,
            landmarks=[],
            handedness=Handedness.RIGHT,
            confidence=0.95,
            timestamp=timestamp,
        ),
        static_gesture=GestureResult(
            GestureName.FIST,
            0.9,
            RecognitionSource.MEDIAPIPE,
            timestamp,
        ),
        dynamic_gesture=None,
        cursor_gesture=None,
        pointer=PointerPosition(0.25, 0.75, True, 0.95, timestamp),
        cursor_result=CursorControlResult(
            CursorAction.MOVE_CURSOR,
            ScreenPosition(100, 200, timestamp),
            GestureName.INDEX_MCP,
            1.0,
            timestamp,
        ),
        cursor_enabled=True,
        cursor_pressed=False,
        paused=False,
        fps=29.5,
        inference_ms=12.3,
        mirrored=True,
    )


def test_runtime_contract_is_serializable_and_transport_extensible() -> None:
    publisher = MLIntegrationPublisher()
    recording_server = RecordingServer()
    publisher.server = recording_server  # type: ignore[assignment]

    event = build_runtime_event(publisher)
    payload = json.loads(event.to_json())

    assert payload["contract"] == RUNTIME_CONTRACT
    assert payload["version"] == CONTRACT_VERSION
    assert payload["sequence"] == 1
    assert payload["gestures"]["static"]["name"] == "FIST"
    assert payload["gestures"]["dynamic"] is None
    assert payload["hand"]["pointer"]["x"] == 0.25
    assert payload["cursor"]["screen_position"] == {"x": 100, "y": 200}
    assert recording_server.events == [event]


def test_camera_contract_correlates_with_runtime_snapshot() -> None:
    publisher = MLIntegrationPublisher(
        MLIntegrationPublisherConfig(publish_camera=True, camera_fps=10.0, jpeg_quality=70)
    )
    recording_server = RecordingServer()
    publisher.server = recording_server  # type: ignore[assignment]
    runtime_event = build_runtime_event(publisher)

    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    camera_event = publisher.publish_camera_frame(frame, runtime_event, mirrored=True)

    assert camera_event is not None
    assert camera_event.contract == CAMERA_FRAME_CONTRACT
    assert camera_event.stream_id == runtime_event.stream_id
    assert camera_event.sequence == runtime_event.sequence
    assert camera_event.width == 6
    assert camera_event.height == 4
    assert base64.b64decode(camera_event.data_base64).startswith(b"\xff\xd8")


def test_ndjson_server_broadcasts_complete_event_lines_to_independent_consumers() -> None:
    server = NDJSONBroadcastServer(port=0).start()
    ui_connection = socket.create_connection(server.address, timeout=2.0)
    scene_connection = socket.create_connection(server.address, timeout=2.0)
    ui_connection.settimeout(2.0)
    scene_connection.settimeout(2.0)
    try:
        deadline = time.monotonic() + 2.0
        while server.client_count < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.client_count == 2

        server.publish({"contract": "test", "version": 1})
        with (
            ui_connection.makefile("r", encoding="utf-8") as ui_stream,
            scene_connection.makefile("r", encoding="utf-8") as scene_stream,
        ):
            expected = {"contract": "test", "version": 1}
            assert json.loads(ui_stream.readline()) == expected
            assert json.loads(scene_stream.readline()) == expected
    finally:
        ui_connection.close()
        scene_connection.close()
        server.close()


def test_published_json_schema_is_available_to_non_python_consumers() -> None:
    schema_path = Path(__file__).resolve().parents[1] / "contracts" / "ml_events.v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$defs"]["runtimeEvent"]["properties"]["contract"]["const"] == RUNTIME_CONTRACT
    assert (
        schema["$defs"]["cameraFrameEvent"]["properties"]["contract"]["const"]
        == CAMERA_FRAME_CONTRACT
    )


def _load_schema() -> dict:
    schema_path = Path(__file__).resolve().parents[1] / "contracts" / "ml_events.v1.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def test_runtime_event_matches_published_json_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    event = build_runtime_event(MLIntegrationPublisher())
    jsonschema.validate(json.loads(event.to_json()), _load_schema())


def test_camera_event_matches_published_json_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    publisher = MLIntegrationPublisher(MLIntegrationPublisherConfig(publish_camera=True))
    publisher.server = RecordingServer()  # type: ignore[assignment]
    runtime_event = build_runtime_event(publisher)
    camera_event = publisher.publish_camera_frame(
        np.zeros((4, 6, 3), dtype=np.uint8), runtime_event, mirrored=True
    )
    assert camera_event is not None
    jsonschema.validate(json.loads(camera_event.to_json()), _load_schema())


def test_future_gesture_names_do_not_break_v1_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    payload = json.loads(build_runtime_event(MLIntegrationPublisher()).to_json())
    # Gestures the ML does not produce yet must still validate as v1 (extensible
    # transport, not a closed enum). See docs/interaction_spec.md.
    payload["gestures"]["dynamic"] = {
        "name": "SWIPE_LEFT",
        "confidence": 0.9,
        "source": "DYNAMIC_MODEL",
    }
    jsonschema.validate(payload, _load_schema())
