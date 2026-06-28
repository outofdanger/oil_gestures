from __future__ import annotations

import shutil
import time
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np

from oil_gestures.core.enums import Handedness
from oil_gestures.core.types import FramePacket, LandmarkPacket
from oil_gestures.vision.landmark_utils import as_landmark_list


HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


class MediaPipeHandLandmarker:
    def __init__(
        self,
        model_path: str,
        max_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.65,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.kind: str | None = None
        self.instance = None
        self.mp = None
        self.hands_module = None
        self.drawing_utils = None
        self.drawing_styles = None
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

    def __enter__(self) -> "MediaPipeHandLandmarker":
        mp = self._import_mediapipe()
        self.mp = mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.kind = "legacy"
            self.hands_module = mp.solutions.hands
            self.drawing_utils = getattr(mp.solutions, "drawing_utils", None)
            self.drawing_styles = getattr(mp.solutions, "drawing_styles", None)
            self.instance = self.hands_module.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            return self

        try:
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
        except Exception as exc:
            raise RuntimeError(
                "Installed mediapipe exposes neither mp.solutions.hands nor mp.tasks.vision. "
                f"Imported mediapipe from: {getattr(mp, '__file__', 'unknown location')}"
            ) from exc

        model_path = self._resolve_model_path()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.kind = "tasks"
        self.instance = HandLandmarker.create_from_options(options)
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

        source = Path("/Users/babidzhon/Project/hand_landmarker.task")
        if source.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, path)
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, path)
        except Exception as exc:
            raise RuntimeError(
                "Could not find or download hand_landmarker.task. "
                f"Save it manually to: {path}"
            ) from exc
        return path

    def detect(self, frame_packet: FramePacket) -> LandmarkPacket:
        if self.instance is None or self.kind is None or self.mp is None:
            return LandmarkPacket(False, None, Handedness.UNKNOWN, 0.0, frame_packet.timestamp)

        mp = self.mp
        rgb_frame = frame_packet.frame

        if self.kind == "legacy":
            rgb_frame.flags.writeable = False
            result = self.instance.process(rgb_frame)
            rgb_frame.flags.writeable = True
            landmarks = list(result.multi_hand_landmarks or [])
            if not landmarks:
                return LandmarkPacket(False, None, Handedness.UNKNOWN, 0.0, frame_packet.timestamp)
            return LandmarkPacket(True, landmarks[0], Handedness.UNKNOWN, 1.0, frame_packet.timestamp)

        timestamp_ms = max(0, int((frame_packet.timestamp - self._started_at) * 1000))
        rgb_frame = np.ascontiguousarray(rgb_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.instance.detect_for_video(mp_image, timestamp_ms)
        landmarks = list(result.hand_landmarks or [])
        if not landmarks:
            return LandmarkPacket(False, None, Handedness.UNKNOWN, 0.0, frame_packet.timestamp)

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

        return LandmarkPacket(True, landmarks[0], handedness, confidence, frame_packet.timestamp)

    @staticmethod
    def landmarks_as_sequence(packet: LandmarkPacket) -> Sequence:
        return as_landmark_list(packet.landmarks)
