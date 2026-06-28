from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app_config import load_config
from oil_gestures.vision.camera import CameraConfig as StreamCameraConfig
from oil_gestures.vision.camera import CameraStream
from oil_gestures.vision.drawing import draw_landmarks, draw_overlay
from oil_gestures.vision.frame_processor import FrameProcessorConfig, bgr_to_rgb, process_frame
from oil_gestures.vision.mediapipe_landmarker import MediaPipeHandLandmarker


class FpsMeter:
    def __init__(self) -> None:
        self.frame_count = 0
        self.last_time = time.perf_counter()
        self.average = 0.0

    def update(self) -> float:
        self.frame_count += 1
        now = time.perf_counter()
        elapsed = now - self.last_time
        if elapsed >= 0.5:
            current = self.frame_count / elapsed
            self.average = current if self.average == 0.0 else self.average + (current - self.average) * 0.25
            self.frame_count = 0
            self.last_time = now
        return self.average


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check camera and MediaPipe hand landmarks.")
    parser.add_argument("--config-dir", type=str, default=None, help="Path to the configs directory.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_config(args.config_dir)
    camera_config = StreamCameraConfig(
        device_id=config.camera.device_id,
        width=config.camera.width,
        height=config.camera.height,
        fps=config.camera.fps,
        preferred_fourcc=config.camera.preferred_fourcc,
    )
    frame_processor_config = FrameProcessorConfig(
        width=None,
        height=None,
        mirror=config.camera.mirror,
    )
    inference_frame_processor_config = FrameProcessorConfig(
        width=config.mediapipe.input_width,
        height=config.mediapipe.input_height,
        mirror=False,
    )
    fps_meter = FpsMeter()
    safe_exit = ord(config.runtime.safe_exit_key[:1] or "q")

    with CameraStream(camera_config) as camera, MediaPipeHandLandmarker(
        model_path=config.mediapipe.model_path,
        max_hands=config.mediapipe.max_hands,
        model_complexity=config.mediapipe.model_complexity,
        min_detection_confidence=config.mediapipe.min_detection_confidence,
        min_tracking_confidence=config.mediapipe.min_tracking_confidence,
    ) as landmarker:
        cv2.namedWindow(config.runtime.window_name, cv2.WINDOW_NORMAL)
        try:
            for frame_packet in camera.frames():
                display_packet = process_frame(frame_packet, frame_processor_config)
                inference_packet = process_frame(display_packet, inference_frame_processor_config)
                rgb_packet = bgr_to_rgb(inference_packet)
                landmark_packet = landmarker.detect(rgb_packet)
                fps = fps_meter.update()

                if landmark_packet.hand_detected:
                    draw_landmarks(display_packet.frame, landmark_packet.landmarks)

                status = "ACTIVE" if landmark_packet.hand_detected else "NO HAND"
                draw_overlay(display_packet.frame, status=status, fps=fps)
                cv2.imshow(config.runtime.window_name, display_packet.frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, safe_exit):
                    break
        finally:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
