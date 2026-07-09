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
    MINIMUM_TARGET_FPS,
    OPENCV_ESCAPE_KEY_CODE,
    PAUSE_KEY_CHARACTERS,
    PERFORMANCE_LOG_INTERVAL_SECONDS,
    PERFORMANCE_TIMING_SMOOTHING_ALPHA,
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
from oil_gestures.gestures.decision.gesture_fusion import GestureFusion
from oil_gestures.gestures.decision.gesture_toggle import GestureToggle
from oil_gestures.gestures.dynamic.dynamic_recognizer import DynamicGestureRecognizer
from oil_gestures.gestures.dynamic.model_loader import DynamicModelLoaderConfig, load_dynamic_model
from oil_gestures.gestures.static.static_recognizer import StaticGestureRecognizer
from oil_gestures.integration.publisher import (
    MLIntegrationPublisher,
    MLIntegrationPublisherConfig,
)


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


class PerformanceMeter:
    def __init__(self) -> None:
        self.inference_ms = 0.0
        self.last_log_time = time.perf_counter()

    def update_inference(self, elapsed_seconds: float) -> None:
        current_ms = elapsed_seconds * 1000.0
        if self.inference_ms == 0.0:
            self.inference_ms = current_ms
        else:
            alpha = PERFORMANCE_TIMING_SMOOTHING_ALPHA
            self.inference_ms += (current_ms - self.inference_ms) * alpha

    def should_log(self, now: float) -> bool:
        if now - self.last_log_time < PERFORMANCE_LOG_INTERVAL_SECONDS:
            return False
        self.last_log_time = now
        return True


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
        grab_sensitivity=config.cursor.grab_sensitivity,
    )


def run(
    config: OilGesturesConfig,
    integration_publisher: MLIntegrationPublisher | None = None,
) -> int:
    import cv2

    from oil_gestures.vision.camera import CameraConfig as StreamCameraConfig
    from oil_gestures.vision.camera import CameraStream
    from oil_gestures.vision.drawing import draw_landmarks, draw_overlay, draw_pointer_cursor
    from oil_gestures.vision.frame_processor import FrameProcessorConfig, bgr_to_rgb, process_frame
    from oil_gestures.vision.mediapipe_gesture import MediaPipeGestureRecognizer

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
        preferred_fourcc=config.camera.preferred_fourcc,
        threaded=config.camera.threaded,
    )
    preview_width = config.runtime.preview_width or None
    frame_processor_config = FrameProcessorConfig(
        width=preview_width,
        height=None,
        mirror=config.camera.mirror,
    )
    inference_frame_processor_config = FrameProcessorConfig(
        width=config.mediapipe.input_width,
        height=config.mediapipe.input_height,
        mirror=config.camera.mirror,
    )

    fps_meter = FpsMeter()
    performance_meter = PerformanceMeter()
    static_recognizer = StaticGestureRecognizer(config.static)
    dynamic_model = (
        load_dynamic_model(
            DynamicModelLoaderConfig(
                stgcn_checkpoint_path=config.dynamic.stgcn_checkpoint_path,
                bilstm_checkpoint_path=config.dynamic.bilstm_checkpoint_path,
                min_confidence=config.dynamic.min_confidence,
                veto_floor=config.dynamic.veto_floor,
                device=config.dynamic.device,
                directional_lockout_seconds=config.dynamic.directional_lockout_seconds,
            )
        )
        if config.dynamic.stgcn_checkpoint_path and config.dynamic.bilstm_checkpoint_path
        else None
    )
    dynamic_recognizer = DynamicGestureRecognizer(config.dynamic, model=dynamic_model)
    gesture_fusion = GestureFusion(swipe_cooldown_seconds=config.dynamic.swipe_cooldown_seconds)
    cursor_recognizer = CursorGestureRecognizer(config.cursor_gestures)
    cursor_toggle = GestureToggle(config.cursor_toggle)
    pipeline.enabled = config.cursor.enabled
    if config.cursor_toggle.enabled:
        logger.info(
            "Cursor mode starts %s. Hold the %s gesture for %.2fs to toggle it.",
            "ON" if pipeline.enabled else "OFF",
            config.cursor_toggle.target.value,
            config.cursor_toggle.hold_seconds,
        )
    paused = False
    safe_exit = ord(config.runtime.safe_exit_key[:1] or DEFAULT_SAFE_EXIT_KEY)
    pause_keys = tuple(ord(character) for character in PAUSE_KEY_CHARACTERS)

    with ExitStack() as stack:
        stack.callback(pipeline.mouse.close)
        event_publisher = None
        if integration_publisher is not None:
            event_publisher = stack.enter_context(integration_publisher)
            host, port = event_publisher.address
            logger.info(
                "ML contract stream listening on %s:%d (camera=%s).",
                host,
                port,
                "ON" if event_publisher.config.publish_camera else "OFF",
            )
        camera = stack.enter_context(CameraStream(camera_config))
        landmarker = stack.enter_context(
            MediaPipeGestureRecognizer(
                model_path=config.mediapipe.gesture_model_path,
                max_hands=config.mediapipe.max_hands,
                min_detection_confidence=config.mediapipe.min_detection_confidence,
                min_tracking_confidence=config.mediapipe.min_tracking_confidence,
            )
        )
        if config.runtime.show_camera_feed:
            cv2.namedWindow(config.runtime.window_name, cv2.WINDOW_NORMAL)

        try:
            for frame_packet in camera.frames():
                inference_packet = process_frame(frame_packet, inference_frame_processor_config)
                rgb_packet = bgr_to_rgb(inference_packet)
                inference_started = time.perf_counter()
                landmark_packet = landmarker.detect(rgb_packet)
                performance_meter.update_inference(time.perf_counter() - inference_started)
                static_gesture = static_recognizer.update(landmark_packet)
                dynamic_gesture = dynamic_recognizer.update(landmark_packet)
                # Edge-trigger discrete gestures (THUMB_UP, FIST, SWIPE_*,
                # POINTING_INDEX) before anything downstream sees them, so
                # holding one doesn't repeat the same action every frame - see
                # gestures/decision/decision_layer.py. cursor_toggle below
                # intentionally keeps using the raw static_gesture: its own
                # dwell+release-latch needs the continuous per-frame signal.
                fused_gestures = gesture_fusion.fuse(static_gesture, dynamic_gesture)

                toggle_gesture = static_gesture.name if static_gesture is not None else None
                if not paused and cursor_toggle.update(toggle_gesture, landmark_packet.timestamp):
                    pipeline.enabled = not pipeline.enabled
                    pipeline.reset()
                    cursor_recognizer.reset()
                    logger.info(
                        "Cursor mode toggled %s via %s gesture.",
                        "ON" if pipeline.enabled else "OFF",
                        config.cursor_toggle.target.value,
                    )

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
                now = time.perf_counter()
                if config.runtime.debug and fps > 0.0 and performance_meter.should_log(now):
                    log_method = logger.warning if fps < MINIMUM_TARGET_FPS else logger.info
                    log_method(
                        "Performance: %.1f FPS, MediaPipe %.1f ms, camera %dx%d, inference %dx%d",
                        fps,
                        performance_meter.inference_ms,
                        frame_packet.width,
                        frame_packet.height,
                        inference_packet.width,
                        inference_packet.height,
                    )

                runtime_event = None
                if event_publisher is not None:
                    runtime_event = event_publisher.publish_runtime(
                        frame_packet=frame_packet,
                        landmark_packet=landmark_packet,
                        static_gesture=fused_gestures.static,
                        dynamic_gesture=fused_gestures.dynamic,
                        cursor_gesture=cursor_gesture,
                        pointer=pipeline.state.pointer,
                        cursor_result=result,
                        cursor_enabled=pipeline.enabled,
                        cursor_pressed=pipeline.state.pressed or cursor_recognizer.index_pressed,
                        paused=paused,
                        fps=fps,
                        inference_ms=performance_meter.inference_ms,
                        mirrored=config.camera.mirror,
                    )

                display_packet = None
                needs_camera_frame = (
                    event_publisher is not None and event_publisher.wants_camera_frame
                )
                if config.runtime.show_camera_feed or needs_camera_frame:
                    display_packet = process_frame(frame_packet, frame_processor_config)

                if needs_camera_frame and event_publisher is not None and runtime_event is not None:
                    event_publisher.publish_camera_frame(
                        display_packet.frame,
                        runtime_event,
                        mirrored=config.camera.mirror,
                    )

                if config.runtime.show_camera_feed and display_packet is not None:
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
                        gestures = (fused_gestures.dynamic, fused_gestures.static, cursor_gesture)
                        gesture_status = " | ".join(
                            gesture.name.value for gesture in gestures if gesture is not None
                        )
                        feature_status = "cursor ON" if pipeline.enabled else "cursor OFF"
                        if cursor_toggle.progress > 0.0:
                            feature_status += (
                                f" | hold {config.cursor_toggle.target.value} "
                                f"{int(cursor_toggle.progress * 100)}%"
                            )
                        draw_overlay(
                            display_packet.frame,
                            status=status,
                            pointer=pipeline.state.pointer,
                            screen_position=result.screen_position,
                            action_status=pipeline.state.action_status,
                            gesture_status=gesture_status,
                            feature_status=feature_status,
                            pressed=pipeline.state.pressed or cursor_recognizer.index_pressed,
                            mouse_status=mouse_status,
                            fps=fps,
                            paused=paused,
                        )

                    cv2.imshow(config.runtime.window_name, display_packet.frame)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = -1

                if key in (OPENCV_ESCAPE_KEY_CODE, safe_exit):
                    break
                if key in pause_keys:
                    paused = not paused
                    pipeline.reset()
                    dynamic_recognizer.reset()
                    cursor_recognizer.reset()
                    cursor_toggle.reset()
                    gesture_fusion.reset()

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
    parser.add_argument(
        "--event-server",
        action="store_true",
        help="Publish versioned ML contracts over a local NDJSON/TCP stream.",
    )
    parser.add_argument("--event-host", default="127.0.0.1", help="ML contract server host.")
    parser.add_argument("--event-port", type=int, default=8765, help="ML contract server port.")
    parser.add_argument(
        "--publish-camera",
        action="store_true",
        help="Also publish throttled JPEG camera-frame contracts.",
    )
    parser.add_argument(
        "--camera-publish-fps",
        type=float,
        default=10.0,
        help="Maximum camera contract frame rate.",
    )
    parser.add_argument(
        "--camera-jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality for camera contracts (1-100).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the local OpenCV window; consumers can use the contracts.",
    )
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
    if args.publish_camera and not args.event_server:
        raise SystemExit("--publish-camera requires --event-server.")

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
    runtime_overrides = {}
    if args.headless:
        runtime_overrides["show_camera_feed"] = False
    if cursor_overrides or runtime_overrides:
        config = OilGesturesConfig(
            camera=config.camera,
            runtime=replace(config.runtime, **runtime_overrides),
            mediapipe=config.mediapipe,
            static=config.static,
            dynamic=config.dynamic,
            cursor=replace(config.cursor, **cursor_overrides),
            cursor_gestures=config.cursor_gestures,
            cursor_actions=config.cursor_actions,
            cursor_toggle=config.cursor_toggle,
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

    integration_publisher = None
    if args.event_server:
        try:
            publisher_config = MLIntegrationPublisherConfig(
                host=args.event_host,
                port=args.event_port,
                publish_camera=args.publish_camera,
                camera_fps=args.camera_publish_fps,
                jpeg_quality=args.camera_jpeg_quality,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        integration_publisher = MLIntegrationPublisher(publisher_config)

    return run(config, integration_publisher=integration_publisher)


if __name__ == "__main__":
    raise SystemExit(main())
