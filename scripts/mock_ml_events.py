"""Fake ML producer for autonomous UI / 3D development.

This script impersonates the real ML runtime: it opens the same NDJSON/TCP
endpoint and speaks the exact same versioned contract, but generates scripted
fake data instead of using a camera or MediaPipe.

UI and 3D developers run this INSTEAD of the full ML pipeline, so they can build
and test their reactions without a webcam, without models, and in CI. It also
emits gestures that ML does not produce yet (swipes, valve, wrist rotation),
matching the frozen vocabulary in docs/interaction_spec.md. When the real ML is
ready, the stream is identical and consumers need no changes.

Run:
    python scripts/mock_ml_events.py
    python scripts/mock_ml_events.py --camera          # also fake camera frames
    python scripts/mock_ml_events.py --port 8765 --fps 30
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
STEP_SECONDS = 1.2

# (channel, gesture_name, source). channel is one of static/dynamic/cursor.
# Covers the full frozen vocabulary from docs/interaction_spec.md, including
# gestures the real ML does not produce yet ("planned").
SCENARIO: tuple[tuple[str, str, str], ...] = (
    ("static", "OPEN_PALM", "STATIC_RULES"),
    ("static", "FIST", "STATIC_RULES"),
    ("static", "THUMB_UP", "STATIC_RULES"),
    ("static", "OK_SIGN", "STATIC_RULES"),
    ("static", "POINT", "STATIC_RULES"),
    ("dynamic", "SWIPE_LEFT", "DYNAMIC_MODEL"),
    ("dynamic", "SWIPE_RIGHT", "DYNAMIC_MODEL"),
    ("dynamic", "SPREAD", "DYNAMIC_MODEL"),
    ("dynamic", "CLENCH", "DYNAMIC_MODEL"),
    ("dynamic", "WRIST_ROTATE_CW", "DYNAMIC_MODEL"),
    ("dynamic", "WRIST_ROTATE_CCW", "DYNAMIC_MODEL"),
    ("cursor", "INDEX_MCP", "CURSOR_RULES"),
    ("cursor", "INDEX_SQUEEZE", "CURSOR_RULES"),
    ("cursor", "INDEX_RELEASE", "CURSOR_RULES"),
    ("cursor", "MIDDLE_PINCH", "CURSOR_RULES"),
    ("static", "VICTORY", "STATIC_RULES"),
)


def _gesture(channel: str, step: tuple[str, str, str]) -> GestureContract | None:
    active_channel, name, source = step
    if channel != active_channel:
        return None
    return GestureContract(name=name, confidence=0.9, source=source)


def build_runtime_event(sequence: int, stream_id: str, elapsed: float) -> MLRuntimeEvent:
    step = SCENARIO[int(elapsed // STEP_SECONDS) % len(SCENARIO)]
    _, active_name, _ = step

    # Pointer travels a smooth circle so consumers see continuous motion.
    px = 0.5 + 0.3 * math.cos(elapsed)
    py = 0.5 + 0.3 * math.sin(elapsed)

    cursor_enabled = active_name in {"INDEX_MCP", "INDEX_SQUEEZE", "INDEX_RELEASE", "MIDDLE_PINCH"}
    pressed = active_name == "INDEX_SQUEEZE"
    if active_name == "MIDDLE_PINCH":
        action = "RIGHT_CLICK"
    elif active_name == "INDEX_SQUEEZE":
        action = "GRAB"
    elif active_name == "INDEX_RELEASE":
        action = "RELEASE"
    elif cursor_enabled:
        action = "MOVE_CURSOR"
    else:
        action = "NONE"

    return MLRuntimeEvent(
        contract=RUNTIME_CONTRACT,
        version=CONTRACT_VERSION,
        stream_id=stream_id,
        sequence=sequence,
        timestamp=time.time(),
        frame=FrameContract(width=FRAME_WIDTH, height=FRAME_HEIGHT, mirrored=True),
        performance=PerformanceContract(
            fps=round(30.0 + 2.0 * math.sin(elapsed), 1),
            inference_ms=round(12.0 + 1.5 * math.cos(elapsed), 1),
        ),
        hand=HandContract(
            detected=True,
            handedness="RIGHT",
            confidence=0.95,
            pointer=PointerContract(x=round(px, 4), y=round(py, 4), visible=True, confidence=0.95),
        ),
        gestures=GesturesContract(
            static=_gesture("static", step),
            dynamic=_gesture("dynamic", step),
            cursor=_gesture("cursor", step),
        ),
        cursor=CursorContract(
            enabled=cursor_enabled,
            pressed=pressed,
            action=action,
            screen_position=ScreenPositionContract(
                x=int(px * SCREEN_WIDTH),
                y=int(py * SCREEN_HEIGHT),
            ),
        ),
        paused=active_name == "OPEN_PALM",
    )


def _make_camera_encoder(jpeg_quality: int):
    """Return a function(elapsed) -> base64 JPEG, or None if cv2 is unavailable."""
    try:
        import base64

        import cv2
        import numpy as np
    except ImportError:
        return None

    def encode(elapsed: float) -> str | None:
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        shade = int(80 + 60 * (0.5 + 0.5 * math.sin(elapsed)))
        frame[:] = (shade, 40, 90)
        cv2.putText(
            frame,
            "MOCK ML CAMERA",
            (30, FRAME_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            return None
        return base64.b64encode(buffer.tobytes()).decode("ascii")

    return encode


def main() -> int:
    parser = argparse.ArgumentParser(description="Fake ML producer for UI/3D development.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=30.0, help="Runtime event rate.")
    parser.add_argument("--camera", action="store_true", help="Also emit fake JPEG camera frames.")
    parser.add_argument("--camera-fps", type=float, default=10.0)
    parser.add_argument("--camera-jpeg-quality", type=int, default=80)
    args = parser.parse_args()

    if args.fps <= 0:
        raise SystemExit("--fps must be positive")

    encode_camera = None
    if args.camera:
        encode_camera = _make_camera_encoder(args.camera_jpeg_quality)
        if encode_camera is None:
            print("Camera disabled: install opencv-contrib-python and numpy for --camera.")

    server = NDJSONBroadcastServer(args.host, args.port).start()
    host, port = server.address
    stream_id = str(uuid.uuid4())
    print(f"Mock ML contract stream on {host}:{port} (camera={'ON' if encode_camera else 'OFF'}).")
    print("Connect with: python scripts/consume_ml_events.py")
    print("Press Ctrl+C to stop.")

    sequence = 0
    start = time.monotonic()
    frame_interval = 1.0 / args.fps
    camera_interval = 1.0 / args.camera_fps if encode_camera else None
    last_camera = float("-inf")

    try:
        while True:
            tick = time.monotonic()
            elapsed = tick - start
            sequence += 1
            runtime_event = build_runtime_event(sequence, stream_id, elapsed)
            server.publish(runtime_event)

            if (
                encode_camera is not None
                and camera_interval is not None
                and server.client_count > 0
                and tick - last_camera >= camera_interval
            ):
                data = encode_camera(elapsed)
                if data is not None:
                    server.publish(
                        CameraFrameEvent(
                            contract=CAMERA_FRAME_CONTRACT,
                            version=CONTRACT_VERSION,
                            stream_id=runtime_event.stream_id,
                            sequence=runtime_event.sequence,
                            timestamp=runtime_event.timestamp,
                            width=FRAME_WIDTH,
                            height=FRAME_HEIGHT,
                            mirrored=True,
                            encoding="jpeg/base64",
                            data_base64=data,
                        )
                    )
                    last_camera = tick

            sleep_for = frame_interval - (time.monotonic() - tick)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\nStopping mock producer.")
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
