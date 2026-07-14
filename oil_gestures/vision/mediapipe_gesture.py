from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np

from oil_gestures.core.enums import Handedness
from oil_gestures.core.types import FramePacket, LandmarkPacket
from oil_gestures.vision.landmark_utils import as_landmark_list


GESTURE_RECOGNIZER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)


class MediaPipeGestureRecognizer:
    """
    Single-inference MediaPipe backend.

    One GestureRecognizer pass yields both the 21 hand landmarks (consumed by the
    cursor and dynamic subsystems) and MediaPipe's built-in (canned) gesture
    category (consumed by the static gesture subsystem). It replaces
    MediaPipeHandLandmarker so static gestures and cursor tracking share a single
    inference per frame instead of running two MediaPipe graphs.

    The canned gesture and its score are attached to the returned LandmarkPacket
    as ``raw_gesture`` / ``raw_gesture_score``; mapping the MediaPipe category
    name to a project GestureName is left to the static recognizer.
    """

    def __init__(
        self,
        model_path: str,
        max_hands: int = 1,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.65,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.instance = None
        self.mp = None
        self._started_at = time.perf_counter()

    @staticmethod
    def _import_mediapipe():
        try:
            import mediapipe as mp
        except Exception as exc:
            raise RuntimeError(
                "MediaPipe is required. Install it with: python -m pip install --upgrade mediapipe"
            ) from exc
        return mp

    def __enter__(self) -> "MediaPipeGestureRecognizer":
        mp = self._import_mediapipe()
        self.mp = mp
        try:
            BaseOptions = mp.tasks.BaseOptions
            GestureRecognizer = mp.tasks.vision.GestureRecognizer
            GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
        except Exception as exc:
            raise RuntimeError(
                "Installed mediapipe does not expose mp.tasks.vision.GestureRecognizer. "
                "Upgrade it with: python -m pip install --upgrade mediapipe. "
                f"Imported mediapipe from: {getattr(mp, '__file__', 'unknown location')}"
            ) from exc

        model_path = self._resolve_model_path()
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.instance = GestureRecognizer.create_from_options(options)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.instance is not None:
            close = getattr(self.instance, "close", None)
            if callable(close):
                close()
        self.instance = None
        self.mp = None

    def _resolve_model_path(self) -> Path:
        path = self.model_path
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if path.exists():
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(GESTURE_RECOGNIZER_MODEL_URL, path)
        except Exception as exc:
            raise RuntimeError(
                "Could not find or download gesture_recognizer.task. "
                f"Save it manually to: {path}"
            ) from exc
        return path

    def detect(self, frame_packet: FramePacket) -> LandmarkPacket:
        if self.instance is None or self.mp is None:
            return LandmarkPacket(False, None, Handedness.UNKNOWN, 0.0, frame_packet.timestamp)

        mp = self.mp
        rgb_frame = np.ascontiguousarray(frame_packet.frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = max(0, int((frame_packet.timestamp - self._started_at) * 1000))
        result = self.instance.recognize_for_video(mp_image, timestamp_ms)

        landmarks = list(result.hand_landmarks or [])
        if not landmarks:
            return LandmarkPacket(False, None, Handedness.UNKNOWN, 0.0, frame_packet.timestamp)

        world_landmarks_list = list(getattr(result, "hand_world_landmarks", None) or [])
        world_landmarks = world_landmarks_list[0] if world_landmarks_list else None

        handedness = Handedness.UNKNOWN
        confidence = 1.0
        if getattr(result, "handedness", None):
            category = result.handedness[0][0]
            label = getattr(category, "category_name", "")
            confidence = float(getattr(category, "score", 1.0))
            if label.lower() == "left":
                handedness = Handedness.LEFT
            elif label.lower() == "right":
                handedness = Handedness.RIGHT

        raw_gesture: str | None = None
        raw_gesture_score = 0.0
        gestures = list(result.gestures or [])
        if gestures and gestures[0]:
            top = gestures[0][0]
            raw_gesture = getattr(top, "category_name", None)
            raw_gesture_score = float(getattr(top, "score", 0.0))

        return LandmarkPacket(
            hand_detected=True,
            landmarks=landmarks[0],
            handedness=handedness,
            confidence=confidence,
            timestamp=frame_packet.timestamp,
            raw_gesture=raw_gesture,
            raw_gesture_score=raw_gesture_score,
            world_landmarks=world_landmarks,
        )

    @staticmethod
    def landmarks_as_sequence(packet: LandmarkPacket) -> Sequence:
        return as_landmark_list(packet.landmarks)
