from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oil_gestures.core.constants import (
    DEFAULT_LANDMARK_COUNT,
    DEFAULT_LANDMARK_DIMENSIONS,
    DEFAULT_MAX_HANDS,
    DEFAULT_MEDIAPIPE_MODEL_PATH,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_TRACKING_CONFIDENCE,
    DEFAULT_SAFE_EXIT_KEY,
)
from oil_gestures.core.enums import GestureName, Handedness
from oil_gestures.core.logger import get_logger
from oil_gestures.vision.camera import CameraConfig, CameraStream
from oil_gestures.vision.drawing import draw_landmarks
from oil_gestures.vision.frame_processor import bgr_to_rgb, mirror_frame
from oil_gestures.vision.landmark_utils import as_landmark_list

logger = get_logger(__name__)

WINDOW_NAME = "Dataset recorder | MediaPipe Tasks HandLandmarker"
QUIT_KEY = ord((DEFAULT_SAFE_EXIT_KEY or "q")[:1])

# Cap the capture/processing loop to this rate so recorded sequences keep a
# stable cadence regardless of how fast the camera or MediaPipe can run.
MAX_FPS = 20
MIN_FRAME_INTERVAL = 1.0 / MAX_FPS

REQUIRED_LABEL_NAMES = (
    "IDLE",
    "POINTING_INDEX",
    "SQUEEZE",
    "RELEASE",
    "ROTATE_CLOCKWISE",
    "ROTATE_COUNTERCLOCKWISE",
    "SWIPE_LEFT",
    "SWIPE_RIGHT",
)


def build_labels() -> dict[int, str]:
    # Prefer GestureName.value so the recorder stays in sync with core/enums.py;
    # fall back to literal strings only if a future enum edit drops a label.
    if all(hasattr(GestureName, name) for name in REQUIRED_LABEL_NAMES):
        values = [getattr(GestureName, name).value for name in REQUIRED_LABEL_NAMES]
    else:
        values = list(REQUIRED_LABEL_NAMES)
    return {ord(str(index + 1)): value for index, value in enumerate(values)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record MediaPipe HandLandmarker landmark sequences for dynamic gesture training."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=str, default="data/raw")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", type=str, default="MJPG")
    parser.add_argument("--model", type=str, default=DEFAULT_MEDIAPIPE_MODEL_PATH)
    parser.add_argument(
        "--no-display-mirror",
        dest="display_mirror",
        action="store_false",
        default=True,
        help="Disable mirroring of the display window (saved landmarks are never mirrored).",
    )
    parser.add_argument("--start-delay", type=float, default=0.25)
    return parser.parse_args(argv)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


@dataclass
class DetectionResult:
    hand_detected: bool
    image_landmarks: Any | None
    world_landmarks: Any | None
    handedness: Handedness
    timestamp_ms: int


class HandLandmarkerSession:
    def __init__(
        self,
        model_path: Path,
        max_hands: int = DEFAULT_MAX_HANDS,
        min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = DEFAULT_MIN_TRACKING_CONFIDENCE,
    ) -> None:
        self.model_path = model_path
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp = None
        self.landmarker = None
        self._started_at = time.perf_counter()

    def __enter__(self) -> "HandLandmarkerSession":
        try:
            import mediapipe as mp
        except Exception as exc:
            raise RuntimeError(
                "MediaPipe is required. Install it with: python -m pip install mediapipe"
            ) from exc

        if not hasattr(mp, "tasks") or not hasattr(mp.tasks, "vision"):
            raise RuntimeError(
                "Installed mediapipe does not expose mp.tasks.vision.HandLandmarker. "
                f"Imported mediapipe from: {getattr(mp, '__file__', 'unknown location')}"
            )

        self.mp = mp
        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
        self.landmarker = None
        self.mp = None

    def detect(self, rgb_frame: np.ndarray, timestamp: float) -> DetectionResult:
        mp = self.mp
        timestamp_ms = max(0, int((timestamp - self._started_at) * 1000))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        hand_landmarks_list = list(result.hand_landmarks or [])
        if not hand_landmarks_list:
            return DetectionResult(False, None, None, Handedness.UNKNOWN, timestamp_ms)

        image_landmarks = hand_landmarks_list[0]

        world_landmarks_list = list(getattr(result, "hand_world_landmarks", None) or [])
        world_landmarks = world_landmarks_list[0] if world_landmarks_list else None

        handedness = Handedness.UNKNOWN
        handedness_lists = list(getattr(result, "handedness", None) or [])
        if handedness_lists and handedness_lists[0]:
            label = getattr(handedness_lists[0][0], "category_name", "") or ""
            if label.lower() == "left":
                handedness = Handedness.LEFT
            elif label.lower() == "right":
                handedness = Handedness.RIGHT

        return DetectionResult(True, image_landmarks, world_landmarks, handedness, timestamp_ms)


def landmarks_to_array(landmarks: Any, count: int, dims: int) -> np.ndarray:
    if landmarks is None:
        return np.full((count, dims), np.nan, dtype=np.float32)
    points = as_landmark_list(landmarks)
    array = np.array([[lm.x, lm.y, lm.z] for lm in points], dtype=np.float32)
    if array.shape == (count, dims):
        return array
    padded = np.full((count, dims), np.nan, dtype=np.float32)
    rows = min(count, array.shape[0])
    padded[:rows] = array[:rows]
    return padded


def mirrored_landmarks_for_display(landmarks: Any, mirror: bool) -> Sequence:
    points = as_landmark_list(landmarks)
    if not mirror:
        return points
    return [SimpleNamespace(x=1.0 - float(p.x), y=float(p.y), z=float(getattr(p, "z", 0.0))) for p in points]


@dataclass
class CameraSettings:
    device_id: int
    width: int
    height: int
    fps: int
    fourcc: str


def open_camera(settings: CameraSettings) -> CameraStream:
    camera_config = CameraConfig(device_id=settings.device_id, width=settings.width, height=settings.height, fps=settings.fps)
    stream = CameraStream(camera_config)
    stream.open()
    capture = stream.capture
    fourcc_code = cv2.VideoWriter_fourcc(*settings.fourcc)
    capture.set(cv2.CAP_PROP_FOURCC, fourcc_code)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    capture.set(cv2.CAP_PROP_FPS, settings.fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return stream


def log_actual_camera_settings(stream: CameraStream) -> None:
    capture = stream.capture
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    logger.info(
        f"Camera actual settings: width={actual_width} height={actual_height} "
        f"fps={actual_fps:.2f} fourcc={fourcc_str!r}"
    )


class LoopFpsMeter:
    def __init__(self) -> None:
        self._count = 0
        self._last = time.perf_counter()
        self._value = 0.0

    def update(self) -> float:
        self._count += 1
        now = time.perf_counter()
        elapsed = now - self._last
        if elapsed >= 0.5:
            current = self._count / elapsed
            self._value = current if self._value == 0.0 else self._value + (current - self._value) * 0.25
            self._count = 0
            self._last = now
        return self._value


@dataclass
class RecordingSession:
    label: str
    target_len: int
    image_landmarks: list = field(default_factory=list)
    world_landmarks: list = field(default_factory=list)
    handedness: list = field(default_factory=list)
    timestamps_ms: list = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.image_landmarks)

    def is_complete(self) -> bool:
        return self.count >= self.target_len

    def append(self, image_landmarks: np.ndarray, world_landmarks: np.ndarray, handedness: str, timestamp_ms: int) -> None:
        self.image_landmarks.append(image_landmarks)
        self.world_landmarks.append(world_landmarks)
        self.handedness.append(handedness)
        self.timestamps_ms.append(timestamp_ms)


def save_sample(output_dir: Path, session: RecordingSession, landmark_count: int, landmark_dimensions: int) -> Path:
    label_dir = output_dir / session.label
    label_dir.mkdir(parents=True, exist_ok=True)
    sample_index = len(list(label_dir.glob(f"{session.label}_*.npz"))) + 1
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = label_dir / f"{session.label}_{timestamp_tag}_{sample_index:04d}.npz"

    timestamps_ms = np.array(session.timestamps_ms, dtype=np.int64)
    if len(timestamps_ms) >= 2 and timestamps_ms[-1] > timestamps_ms[0]:
        measured_fps = float(len(timestamps_ms) - 1) * 1000.0 / float(timestamps_ms[-1] - timestamps_ms[0])
    else:
        measured_fps = 0.0

    np.savez_compressed(
        path,
        image_landmarks=np.stack(session.image_landmarks).astype(np.float32),
        world_landmarks=np.stack(session.world_landmarks).astype(np.float32),
        label=session.label,
        handedness=np.array(session.handedness, dtype="<U16"),
        timestamps_ms=timestamps_ms,
        measured_fps=np.float32(measured_fps),
        sequence_length=session.target_len,
        landmark_count=landmark_count,
        landmark_dimensions=landmark_dimensions,
        created_at=datetime.now().isoformat(),
    )
    return path


def draw_status_overlay(
    frame: np.ndarray,
    labels: dict[int, str],
    detection: DetectionResult,
    loop_fps: float,
    session: RecordingSession | None,
    pending_label: str | None,
    pending_seconds_left: float,
    seq_len: int,
    hand_lost_warning: bool,
) -> None:
    lines = [WINDOW_NAME]
    for key_code, label in labels.items():
        lines.append(f"{chr(key_code)}: {label}")
    lines.append("q: quit")
    lines.append(f"HAND: {'YES' if detection.hand_detected else 'NO'}")
    lines.append(f"handedness {detection.handedness.value}")
    lines.append(f"Loop FPS {loop_fps:4.1f}")

    if session is not None:
        lines.append(f"RECORDING: {session.label} | {session.count}/{seq_len}")
    elif pending_label is not None:
        lines.append(f"Starting {pending_label} in {max(0.0, pending_seconds_left):.2f}s")
    else:
        lines.append("Press 1-8 to record one sample")

    y = 24
    for index, line in enumerate(lines):
        color = (0, 215, 255) if index == 0 else (235, 235, 235)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22

    if hand_lost_warning:
        warning = "Hand lost: keep hand visible"
        cv2.putText(frame, warning, (12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, warning, (12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 60, 255), 2, cv2.LINE_AA)


def main() -> int:
    args = parse_args()

    model_path = resolve_path(args.model)
    if not model_path.is_file():
        raise SystemExit(
            f"hand_landmarker model not found at: {model_path}. "
            "Place the MediaPipe Tasks HandLandmarker model at assets/models/mediapipe/hand_landmarker.task."
        )

    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = build_labels()
    camera_settings = CameraSettings(
        device_id=args.camera, width=args.width, height=args.height, fps=args.fps, fourcc=args.fourcc
    )

    camera_stream: CameraStream | None = None
    try:
        camera_stream = open_camera(camera_settings)
        log_actual_camera_settings(camera_stream)

        with HandLandmarkerSession(model_path=model_path) as landmarker:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            fps_meter = LoopFpsMeter()
            session: RecordingSession | None = None
            pending_label: str | None = None
            pending_until: float = 0.0

            while True:
                frame_start = time.perf_counter()
                packet = camera_stream.read()
                if packet is None:
                    logger.warning("Camera returned no frame; stopping.")
                    break

                rgb_packet = bgr_to_rgb(packet)
                detection = landmarker.detect(rgb_packet.frame, packet.timestamp)
                loop_fps = fps_meter.update()

                display_frame = packet.frame.copy()
                if args.display_mirror:
                    display_frame = mirror_frame(display_frame)

                hand_lost_warning = False
                if detection.hand_detected:
                    draw_landmarks(
                        display_frame,
                        mirrored_landmarks_for_display(detection.image_landmarks, args.display_mirror),
                    )
                elif session is not None:
                    hand_lost_warning = True

                now = time.perf_counter()
                if pending_label is not None and now >= pending_until:
                    session = RecordingSession(label=pending_label, target_len=args.seq_len)
                    pending_label = None

                if session is not None and detection.hand_detected:
                    image_array = landmarks_to_array(
                        detection.image_landmarks, DEFAULT_LANDMARK_COUNT, DEFAULT_LANDMARK_DIMENSIONS
                    )
                    world_array = landmarks_to_array(
                        detection.world_landmarks, DEFAULT_LANDMARK_COUNT, DEFAULT_LANDMARK_DIMENSIONS
                    )
                    session.append(image_array, world_array, detection.handedness.value, detection.timestamp_ms)
                    if session.is_complete():
                        saved_path = save_sample(output_dir, session, DEFAULT_LANDMARK_COUNT, DEFAULT_LANDMARK_DIMENSIONS)
                        logger.info(f"Saved sample: {saved_path}")
                        session = None

                draw_status_overlay(
                    display_frame,
                    labels,
                    detection,
                    loop_fps,
                    session,
                    pending_label,
                    pending_until - now,
                    args.seq_len,
                    hand_lost_warning,
                )
                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == QUIT_KEY:
                    break
                if pending_label is None and session is None and key in labels:
                    pending_label = labels[key]
                    pending_until = now + args.start_delay

                elapsed = time.perf_counter() - frame_start
                if elapsed < MIN_FRAME_INTERVAL:
                    time.sleep(MIN_FRAME_INTERVAL - elapsed)
    finally:
        if camera_stream is not None:
            camera_stream.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
