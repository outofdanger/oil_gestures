from __future__ import annotations

import argparse
import sys
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app_config import OilGesturesConfig, load_config
from oil_gestures.core.constants import (
    DEFAULT_SAFE_EXIT_KEY,
    DEFAULT_STRICT_SCREEN_BOUNDS,
    FPS_METER_SAMPLE_INTERVAL_SECONDS,
    FPS_METER_SMOOTHING_ALPHA,
    MOUSE_TEST_STEP_SECONDS,
    MOUSE_TEST_TARGETS,
    OPENCV_ESCAPE_KEY_CODE,
    PAUSE_KEY_CHARACTERS,
)
from oil_gestures.core.logger import get_logger
from oil_gestures.core.types import ScreenPosition
from oil_gestures.cursor.action_mapper import CursorActionMapper
from oil_gestures.cursor.cursor_pipeline import CursorPipeline
from oil_gestures.cursor.cursor_smoothing import CursorSmoother, CursorSmoothingConfig
from oil_gestures.cursor.hand_pointer import HandPointer, HandPointerConfig
from oil_gestures.cursor.mouse_controller import MouseController, MouseControllerConfig
from oil_gestures.cursor.screen_mapper import ScreenMapper, ScreenMapperConfig
from oil_gestures.gestures.cursor.cursor_recognizer import CursorGestureRecognizer
from oil_gestures.gestures.dynamic.dynamic_recognizer import DynamicGestureRecognizer
from oil_gestures.gestures.static.static_recognizer import StaticGestureRecognizer


logger = get_logger(__name__)


class FpsMeter:
    def __init__(self) -> None:
        self.frame_count = 0
        self.last_time = time.perf_counter()
        self.average = 0.0

    def update(self) -> float:
        self.frame_count += 1
        now = time.perf_counter()
        elapsed = now - self.last_time
        if elapsed >= FPS_METER_SAMPLE_INTERVAL_SECONDS:
            current = self.frame_count / elapsed
            self.average = (
                current
                if self.average == 0.0
                else self.average + (current - self.average) * FPS_METER_SMOOTHING_ALPHA
            )
            self.frame_count = 0
            self.last_time = now
        return self.average


def build_cursor_pipeline(config: OilGesturesConfig) -> CursorPipeline:
    hand_pointer = HandPointer(HandPointerConfig(pointer_source=config.cursor.pointer_source))
    screen_mapper = ScreenMapper(
        ScreenMapperConfig(
            margin_x=config.cursor.margin_x,
            margin_top=config.cursor.margin_top,
            margin_bottom=config.cursor.margin_bottom,
            invert_y=config.cursor.invert_y,
            fallback_width=config.camera.width,
            fallback_height=config.camera.height,
            strict_screen_bounds=DEFAULT_STRICT_SCREEN_BOUNDS,
        )
    )
    smoother = CursorSmoother(
        CursorSmoothingConfig(
            alpha=config.cursor.smoothing_alpha,
            min_cutoff=config.cursor.min_cutoff,
            beta=config.cursor.beta,
            derivative_cutoff=config.cursor.derivative_cutoff,
            dead_zone_points=config.cursor.dead_zone_points,
            max_speed_points_per_second=config.cursor.max_speed_points_per_second,
        )
    )
    action_mapper = CursorActionMapper.from_strings(config.cursor_actions.mapping)
    mouse = MouseController(MouseControllerConfig(dry_run=config.cursor.dry_run))
    return CursorPipeline(
        hand_pointer=hand_pointer,
        screen_mapper=screen_mapper,
        smoother=smoother,
        action_mapper=action_mapper,
        mouse=mouse,
        reacquire_frames=config.cursor.reacquire_frames,
        lost_reset_seconds=config.cursor.lost_reset_seconds,
        action_cooldown_seconds=config.cursor_actions.cooldown_seconds,
        enabled=config.cursor.enabled,
    )


def run(config: OilGesturesConfig) -> int:
    import cv2

    from oil_gestures.vision.camera import CameraConfig as StreamCameraConfig
    from oil_gestures.vision.camera import CameraStream
    from oil_gestures.vision.drawing import draw_landmarks, draw_overlay, draw_pointer_cursor
    from oil_gestures.vision.frame_processor import FrameProcessorConfig, bgr_to_rgb, process_frame
    from oil_gestures.vision.mediapipe_landmarker import MediaPipeHandLandmarker

    pipeline = build_cursor_pipeline(config)
    if config.cursor.dry_run:
        logger.info("Dry-run mode: real OS mouse is disabled. Use --real-mouse to move/click the mouse.")
        mouse_status = "DRY-RUN"
    else:
        mouse_status = f"REAL {pipeline.mouse.backend_name.upper()}"
        logger.info("Real mouse mode is enabled with the %s backend.", pipeline.mouse.backend_name)

    mouse_permission = pipeline.mouse.accessibility_status()
    if mouse_permission is False and not config.cursor.dry_run:
        mouse_status = "REAL MOVE-ONLY"
        pipeline.mouse.request_accessibility_prompt()
        logger.error(
            "macOS Accessibility permission is not granted. "
            "Open System Settings -> Privacy & Security -> Accessibility, "
            "allow the app running Python, then restart the demo. "
            "Movement will still be attempted, but clicks may be blocked."
        )

    camera_config = StreamCameraConfig(
        device_id=config.camera.device_id,
        width=config.camera.width,
        height=config.camera.height,
        fps=config.camera.fps,
    )
    frame_processor_config = FrameProcessorConfig(
        width=config.camera.width,
        height=config.camera.height,
        mirror=config.camera.mirror,
    )

    fps_meter = FpsMeter()
    static_recognizer = StaticGestureRecognizer(config.static)
    dynamic_recognizer = DynamicGestureRecognizer(config.dynamic)
    cursor_recognizer = CursorGestureRecognizer(config.cursor_gestures)
    pipeline.enabled = config.cursor.enabled
    paused = False
    safe_exit = ord(config.runtime.safe_exit_key[:1] or DEFAULT_SAFE_EXIT_KEY)
    pause_keys = tuple(ord(character) for character in PAUSE_KEY_CHARACTERS)

    with ExitStack() as stack:
        stack.callback(pipeline.mouse.close)
        camera = stack.enter_context(CameraStream(camera_config))
        landmarker = stack.enter_context(
            MediaPipeHandLandmarker(
                model_path=config.mediapipe.model_path,
                max_hands=config.mediapipe.max_hands,
                model_complexity=config.mediapipe.model_complexity,
                min_detection_confidence=config.mediapipe.min_detection_confidence,
                min_tracking_confidence=config.mediapipe.min_tracking_confidence,
            )
        )
        if config.runtime.show_camera_feed:
            cv2.namedWindow(config.runtime.window_name, cv2.WINDOW_NORMAL)

        try:
            for frame_packet in camera.frames():
                display_packet = process_frame(frame_packet, frame_processor_config)
                rgb_packet = bgr_to_rgb(display_packet)
                landmark_packet = landmarker.detect(rgb_packet)
                static_gesture = static_recognizer.update(landmark_packet)
                dynamic_gesture = dynamic_recognizer.update(landmark_packet)
                if pipeline.enabled:
                    cursor_gesture = cursor_recognizer.update(landmark_packet)
                else:
                    cursor_recognizer.reset()
                    cursor_gesture = None

                result = pipeline.process(
                    landmark_packet,
                    cursor_gesture=cursor_gesture,
                    paused=paused,
                )
                fps = fps_meter.update()

                if config.runtime.show_camera_feed:
                    if config.runtime.show_landmarks and landmark_packet.hand_detected:
                        draw_landmarks(
                            display_packet.frame,
                            landmark_packet.landmarks,
                            pointer_index=pipeline.hand_pointer.pointer_index,
                        )

                    if landmark_packet.hand_detected:
                        draw_pointer_cursor(
                            display_packet.frame,
                            pipeline.state.pointer,
                            pressed=pipeline.state.pressed or cursor_recognizer.index_pressed,
                        )

                    if config.runtime.show_debug_overlay:
                        if paused:
                            status = "PAUSED"
                        elif not landmark_packet.hand_detected:
                            status = "NO HAND"
                        elif pipeline.enabled:
                            status = "CURSOR ON"
                        else:
                            status = "GESTURE RECOGNITION"
                        gestures = (dynamic_gesture, static_gesture, cursor_gesture)
                        gesture_status = " | ".join(
                            gesture.name.value for gesture in gestures if gesture is not None
                        )
                        draw_overlay(
                            display_packet.frame,
                            status=status,
                            pointer=pipeline.state.pointer,
                            screen_position=result.screen_position,
                            action_status=pipeline.state.action_status,
                            gesture_status=gesture_status,
                            feature_status="cursor ON" if pipeline.enabled else "cursor OFF",
                            pressed=pipeline.state.pressed or cursor_recognizer.index_pressed,
                            mouse_status=mouse_status,
                            fps=fps,
                            paused=paused,
                        )

                    cv2.imshow(config.runtime.window_name, display_packet.frame)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF

                if key in (OPENCV_ESCAPE_KEY_CODE, safe_exit):
                    break
                if key in pause_keys:
                    paused = not paused
                    pipeline.reset()
                    dynamic_recognizer.reset()
                    cursor_recognizer.reset()

        finally:
            pipeline.reset()
            cv2.destroyAllWindows()

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Oil Gestures gesture-recognition demo.")
    parser.add_argument("--config-dir", type=str, default=None, help="Path to the configs directory.")
    parser.add_argument("--dry-run", action="store_true", help="Disable real OS mouse movement/clicks.")
    parser.add_argument("--real-mouse", action="store_true", help="Enable real OS mouse movement/clicks.")
    parser.add_argument("--cursor-on", action="store_true", help="Start optional cursor-control feature enabled.")
    parser.add_argument("--cursor-off", action="store_true", help="Start optional cursor-control feature disabled.")
    parser.add_argument("--request-permission", action="store_true", help="Ask macOS for Accessibility permission.")
    parser.add_argument("--mouse-diagnostics", action="store_true", help="Print mouse backend status and exit.")
    parser.add_argument("--test-mouse-move", action="store_true", help="Try direct mouse movement and exit.")
    return parser


def print_mouse_diagnostics(config: OilGesturesConfig) -> int:
    pipeline = build_cursor_pipeline(config)
    try:
        diagnostics = pipeline.mouse.diagnostics()
        print("Mouse diagnostics")
        for key, value in diagnostics.items():
            print(f"{key}: {value}")
        print(f"screen_bounds: {pipeline.screen_mapper.bounds}")
        print(f"pointer_source: {config.cursor.pointer_source}")
        print(f"dry_run_config: {config.cursor.dry_run}")
        print(f"cursor_initially_enabled: {config.cursor.enabled}")
        print(f"cursor_recognition_enabled: {config.cursor_gestures.enabled}")
    finally:
        pipeline.mouse.close()
    return 0


def test_mouse_move(config: OilGesturesConfig) -> int:
    pipeline = build_cursor_pipeline(config)
    mouse = pipeline.mouse
    bounds = pipeline.screen_mapper.bounds
    now = time.perf_counter()
    targets = [
        ScreenPosition(
            int(bounds.x + bounds.width * x_ratio),
            int(bounds.y + bounds.height * y_ratio),
            now + index * MOUSE_TEST_STEP_SECONDS,
        )
        for index, (x_ratio, y_ratio) in enumerate(MOUSE_TEST_TARGETS)
    ]

    try:
        print("Trying direct mouse movement")
        print(f"dry_run: {mouse.config.dry_run}")
        print(f"accessibility: {mouse.accessibility_status()}")
        print(f"bounds: {bounds}")
        for target in targets:
            result = mouse.move_to(target)
            print(f"MOVE {target.x},{target.y} executed={result.executed} position={mouse.get_position()}")
            time.sleep(MOUSE_TEST_STEP_SECONDS)
    finally:
        mouse.close()
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.real_mouse and args.dry_run:
        raise SystemExit("--real-mouse and --dry-run cannot be used together.")
    if args.cursor_on and args.cursor_off:
        raise SystemExit("--cursor-on and --cursor-off cannot be used together.")

    config = load_config(args.config_dir)
    cursor_overrides = {}
    if args.real_mouse:
        cursor_overrides["dry_run"] = False
    if args.dry_run:
        cursor_overrides["dry_run"] = True
    if args.cursor_on:
        cursor_overrides["enabled"] = True
    if args.cursor_off:
        cursor_overrides["enabled"] = False
    if cursor_overrides:
        config = OilGesturesConfig(
            camera=config.camera,
            runtime=config.runtime,
            mediapipe=config.mediapipe,
            static=config.static,
            dynamic=config.dynamic,
            cursor=replace(config.cursor, **cursor_overrides),
            cursor_gestures=config.cursor_gestures,
            cursor_actions=config.cursor_actions,
        )
    if args.request_permission:
        permission_pipeline = build_cursor_pipeline(config)
        try:
            permission_pipeline.mouse.request_accessibility_prompt()
        finally:
            permission_pipeline.mouse.close()

    if args.mouse_diagnostics:
        return print_mouse_diagnostics(config)

    if args.test_mouse_move:
        return test_mouse_move(config)

    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
