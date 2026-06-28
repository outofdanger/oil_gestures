from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oil_gestures.core.constants import (
    DEFAULT_CAMERA_ID,
    DEFAULT_CAMERA_FOURCC,
    DEFAULT_CAMERA_THREADED,
    DEFAULT_CURSOR_BETA,
    DEFAULT_CURSOR_DEAD_ZONE_POINTS,
    DEFAULT_CURSOR_DERIVATIVE_CUTOFF,
    DEFAULT_CURSOR_DRY_RUN,
    DEFAULT_CURSOR_ENABLED,
    DEFAULT_CURSOR_INVERT_Y,
    DEFAULT_CURSOR_LOST_RESET_SECONDS,
    DEFAULT_CURSOR_MARGIN_BOTTOM,
    DEFAULT_CURSOR_MARGIN_TOP,
    DEFAULT_CURSOR_MARGIN_X,
    DEFAULT_CURSOR_MAX_SPEED_POINTS_PER_SECOND,
    DEFAULT_CURSOR_MIN_CUTOFF,
    DEFAULT_CURSOR_REACQUIRE_FRAMES,
    DEFAULT_CURSOR_SMOOTHING_ALPHA,
    DEFAULT_CURSOR_TOGGLE_COOLDOWN_SECONDS,
    DEFAULT_CURSOR_TOGGLE_ENABLED,
    DEFAULT_CURSOR_TOGGLE_GESTURE,
    DEFAULT_CURSOR_TOGGLE_HOLD_SECONDS,
    DEFAULT_DEBUG_MODE,
    DEFAULT_FPS,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_MAX_HANDS,
    DEFAULT_MEDIAPIPE_GESTURE_MODEL_PATH,
    DEFAULT_MEDIAPIPE_INPUT_HEIGHT,
    DEFAULT_MEDIAPIPE_INPUT_WIDTH,
    DEFAULT_MEDIAPIPE_MODEL_PATH,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_TRACKING_CONFIDENCE,
    DEFAULT_MIRROR_FRAME,
    DEFAULT_MODEL_COMPLEXITY,
    DEFAULT_MOUSE_ACTION_COOLDOWN_SECONDS,
    DEFAULT_POINTER_LANDMARK,
    DEFAULT_PREVIEW_WIDTH,
    DEFAULT_SAFE_EXIT_KEY,
    DEFAULT_SHOW_CAMERA_FEED,
    DEFAULT_SHOW_DEBUG_OVERLAY,
    DEFAULT_SHOW_LANDMARKS,
    DEFAULT_WINDOW_NAME,
)
from oil_gestures.core.enums import GestureName
from oil_gestures.cursor.action_mapper import DEFAULT_CURSOR_MAPPING
from oil_gestures.gestures.cursor.cursor_recognizer import CursorGestureConfig
from oil_gestures.gestures.decision.gesture_toggle import GestureToggleConfig
from oil_gestures.gestures.dynamic.dynamic_recognizer import DynamicRecognizerConfig
from oil_gestures.gestures.static.static_recognizer import StaticRecognizerConfig

try:
    import yaml
except Exception:  # pragma: no cover - handled with a clear runtime error.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CameraConfig:
    device_id: int = DEFAULT_CAMERA_ID
    width: int = DEFAULT_FRAME_WIDTH
    height: int = DEFAULT_FRAME_HEIGHT
    fps: int = DEFAULT_FPS
    mirror: bool = DEFAULT_MIRROR_FRAME
    preferred_fourcc: str = DEFAULT_CAMERA_FOURCC
    threaded: bool = DEFAULT_CAMERA_THREADED


@dataclass(frozen=True)
class RuntimeConfig:
    debug: bool = DEFAULT_DEBUG_MODE
    show_camera_feed: bool = DEFAULT_SHOW_CAMERA_FEED
    show_landmarks: bool = DEFAULT_SHOW_LANDMARKS
    show_debug_overlay: bool = DEFAULT_SHOW_DEBUG_OVERLAY
    preview_width: int = DEFAULT_PREVIEW_WIDTH
    safe_exit_key: str = DEFAULT_SAFE_EXIT_KEY
    window_name: str = DEFAULT_WINDOW_NAME


@dataclass(frozen=True)
class MediaPipeConfig:
    max_hands: int = DEFAULT_MAX_HANDS
    model_complexity: int = DEFAULT_MODEL_COMPLEXITY
    min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE
    min_tracking_confidence: float = DEFAULT_MIN_TRACKING_CONFIDENCE
    model_path: str = DEFAULT_MEDIAPIPE_MODEL_PATH
    gesture_model_path: str = DEFAULT_MEDIAPIPE_GESTURE_MODEL_PATH
    input_width: int = DEFAULT_MEDIAPIPE_INPUT_WIDTH
    input_height: int = DEFAULT_MEDIAPIPE_INPUT_HEIGHT


@dataclass(frozen=True)
class CursorConfig:
    enabled: bool = DEFAULT_CURSOR_ENABLED
    dry_run: bool = DEFAULT_CURSOR_DRY_RUN
    pointer_source: str = DEFAULT_POINTER_LANDMARK
    margin_x: float = DEFAULT_CURSOR_MARGIN_X
    margin_top: float = DEFAULT_CURSOR_MARGIN_TOP
    margin_bottom: float = DEFAULT_CURSOR_MARGIN_BOTTOM
    invert_y: bool = DEFAULT_CURSOR_INVERT_Y
    smoothing_alpha: float = DEFAULT_CURSOR_SMOOTHING_ALPHA
    min_cutoff: float = DEFAULT_CURSOR_MIN_CUTOFF
    beta: float = DEFAULT_CURSOR_BETA
    derivative_cutoff: float = DEFAULT_CURSOR_DERIVATIVE_CUTOFF
    dead_zone_points: float = DEFAULT_CURSOR_DEAD_ZONE_POINTS
    max_speed_points_per_second: float = DEFAULT_CURSOR_MAX_SPEED_POINTS_PER_SECOND
    reacquire_frames: int = DEFAULT_CURSOR_REACQUIRE_FRAMES
    lost_reset_seconds: float = DEFAULT_CURSOR_LOST_RESET_SECONDS


@dataclass(frozen=True)
class CursorActionsConfig:
    cooldown_seconds: float = DEFAULT_MOUSE_ACTION_COOLDOWN_SECONDS
    mapping: dict[str, str] = field(
        default_factory=lambda: {
            gesture.value: action.value for gesture, action in DEFAULT_CURSOR_MAPPING.items()
        }
    )


@dataclass(frozen=True)
class OilGesturesConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    static: StaticRecognizerConfig = field(default_factory=StaticRecognizerConfig)
    dynamic: DynamicRecognizerConfig = field(default_factory=DynamicRecognizerConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    cursor_gestures: CursorGestureConfig = field(default_factory=CursorGestureConfig)
    cursor_actions: CursorActionsConfig = field(default_factory=CursorActionsConfig)
    cursor_toggle: GestureToggleConfig = field(default_factory=GestureToggleConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configs. Install it with: python -m pip install PyYAML")
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping at the root of {path}.")
    return loaded


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def load_config(config_dir: str | Path | None = None) -> OilGesturesConfig:
    root = PROJECT_ROOT if config_dir is None else Path(config_dir).expanduser().resolve().parents[0]
    configs = PROJECT_ROOT / "configs" if config_dir is None else Path(config_dir).expanduser().resolve()

    default_yaml = _load_yaml(configs / "default.yaml")
    gestures_yaml = _load_yaml(configs / "gestures.yaml")
    model_yaml = _load_yaml(configs / "model_config.yaml")

    camera_section = default_yaml.get("camera", {})
    runtime_section = default_yaml.get("runtime", {})
    app_section = default_yaml.get("app", {})
    static_section = gestures_yaml.get("static", {})
    dynamic_section = gestures_yaml.get("dynamic", {})
    cursor_section = gestures_yaml.get("cursor", {})
    cursor_recognition_section = cursor_section.get("recognition", {})
    cursor_activation_section = cursor_section.get("activation", {})
    cursor_actions_section = gestures_yaml.get("cursor_actions", {})
    mediapipe_section = model_yaml.get("mediapipe", {})

    camera_defaults = CameraConfig()
    runtime_defaults = RuntimeConfig()
    media_defaults = MediaPipeConfig()
    static_defaults = StaticRecognizerConfig()
    dynamic_defaults = DynamicRecognizerConfig()
    cursor_defaults = CursorConfig()
    cursor_gesture_defaults = CursorGestureConfig()
    cursor_actions_defaults = CursorActionsConfig()
    cursor_toggle_defaults = GestureToggleConfig()

    model_path = str(mediapipe_section.get("model_path", media_defaults.model_path))
    if not Path(model_path).is_absolute():
        model_path = str(root / model_path)

    gesture_model_path = str(
        mediapipe_section.get("gesture_model_path", media_defaults.gesture_model_path)
    )
    if not Path(gesture_model_path).is_absolute():
        gesture_model_path = str(root / gesture_model_path)

    toggle_gesture_name = str(
        cursor_activation_section.get("gesture", cursor_toggle_defaults.target.value)
    )
    try:
        toggle_target = GestureName(toggle_gesture_name)
    except ValueError:
        toggle_target = cursor_toggle_defaults.target

    return OilGesturesConfig(
        camera=CameraConfig(
            device_id=_as_int(camera_section.get("device_id"), camera_defaults.device_id),
            width=_as_int(camera_section.get("width"), camera_defaults.width),
            height=_as_int(camera_section.get("height"), camera_defaults.height),
            fps=_as_int(camera_section.get("fps"), camera_defaults.fps),
            mirror=_as_bool(camera_section.get("mirror"), camera_defaults.mirror),
            preferred_fourcc=str(
                camera_section.get("preferred_fourcc", camera_defaults.preferred_fourcc)
            ),
            threaded=_as_bool(camera_section.get("threaded"), camera_defaults.threaded),
        ),
        runtime=RuntimeConfig(
            debug=_as_bool(app_section.get("debug"), runtime_defaults.debug),
            show_camera_feed=_as_bool(
                runtime_section.get("show_camera_feed"),
                runtime_defaults.show_camera_feed,
            ),
            show_landmarks=_as_bool(
                runtime_section.get("show_landmarks"),
                runtime_defaults.show_landmarks,
            ),
            show_debug_overlay=_as_bool(
                runtime_section.get("show_debug_overlay"),
                runtime_defaults.show_debug_overlay,
            ),
            preview_width=max(
                0,
                _as_int(runtime_section.get("preview_width"), runtime_defaults.preview_width),
            ),
            safe_exit_key=str(runtime_section.get("safe_exit_key", runtime_defaults.safe_exit_key)),
            window_name=str(runtime_section.get("window_name", runtime_defaults.window_name)),
        ),
        mediapipe=MediaPipeConfig(
            max_hands=max(1, _as_int(mediapipe_section.get("max_hands"), media_defaults.max_hands)),
            model_complexity=_as_int(
                mediapipe_section.get("model_complexity"),
                media_defaults.model_complexity,
            ),
            min_detection_confidence=_as_float(
                mediapipe_section.get("min_detection_confidence"),
                media_defaults.min_detection_confidence,
            ),
            min_tracking_confidence=_as_float(
                mediapipe_section.get("min_tracking_confidence"),
                media_defaults.min_tracking_confidence,
            ),
            model_path=model_path,
            gesture_model_path=gesture_model_path,
            input_width=max(
                1,
                _as_int(mediapipe_section.get("input_width"), media_defaults.input_width),
            ),
            input_height=max(
                1,
                _as_int(mediapipe_section.get("input_height"), media_defaults.input_height),
            ),
        ),
        static=StaticRecognizerConfig(
            enabled=_as_bool(static_section.get("enabled"), static_defaults.enabled),
            min_confidence=_as_float(
                static_section.get("min_confidence"),
                static_defaults.min_confidence,
            ),
        ),
        dynamic=DynamicRecognizerConfig(
            enabled=_as_bool(dynamic_section.get("enabled"), dynamic_defaults.enabled),
            sequence_length=max(
                1,
                _as_int(dynamic_section.get("sequence_length"), dynamic_defaults.sequence_length),
            ),
            min_confidence=_as_float(
                dynamic_section.get("min_confidence"),
                dynamic_defaults.min_confidence,
            ),
            veto_floor=_as_float(
                dynamic_section.get("veto_floor"),
                dynamic_defaults.veto_floor,
            ),
            swipe_cooldown_seconds=_as_float(
                dynamic_section.get("swipe_cooldown_seconds"),
                dynamic_defaults.swipe_cooldown_seconds,
            ),
            stgcn_checkpoint_path=(
                dynamic_defaults.stgcn_checkpoint_path
                if "stgcn_checkpoint_path" not in dynamic_section
                else dynamic_section.get("stgcn_checkpoint_path")
            ),
            bilstm_checkpoint_path=(
                dynamic_defaults.bilstm_checkpoint_path
                if "bilstm_checkpoint_path" not in dynamic_section
                else dynamic_section.get("bilstm_checkpoint_path")
            ),
        ),
        cursor=CursorConfig(
            enabled=_as_bool(cursor_section.get("enabled"), cursor_defaults.enabled),
            dry_run=_as_bool(cursor_section.get("dry_run"), cursor_defaults.dry_run),
            pointer_source=str(cursor_section.get("pointer_source", cursor_defaults.pointer_source)),
            margin_x=_as_float(cursor_section.get("margin_x"), cursor_defaults.margin_x),
            margin_top=_as_float(cursor_section.get("margin_top"), cursor_defaults.margin_top),
            margin_bottom=_as_float(cursor_section.get("margin_bottom"), cursor_defaults.margin_bottom),
            invert_y=_as_bool(cursor_section.get("invert_y"), cursor_defaults.invert_y),
            smoothing_alpha=_as_float(
                cursor_section.get("smoothing_alpha"),
                cursor_defaults.smoothing_alpha,
            ),
            min_cutoff=_as_float(cursor_section.get("min_cutoff"), cursor_defaults.min_cutoff),
            beta=_as_float(cursor_section.get("beta"), cursor_defaults.beta),
            derivative_cutoff=_as_float(
                cursor_section.get("derivative_cutoff"),
                cursor_defaults.derivative_cutoff,
            ),
            dead_zone_points=_as_float(
                cursor_section.get("dead_zone_points"),
                cursor_defaults.dead_zone_points,
            ),
            max_speed_points_per_second=_as_float(
                cursor_section.get("max_speed_points_per_second"),
                cursor_defaults.max_speed_points_per_second,
            ),
            reacquire_frames=max(
                1,
                _as_int(cursor_section.get("reacquire_frames"), cursor_defaults.reacquire_frames),
            ),
            lost_reset_seconds=_as_float(
                cursor_section.get("lost_reset_seconds"),
                cursor_defaults.lost_reset_seconds,
            ),
        ),
        cursor_gestures=CursorGestureConfig(
            enabled=_as_bool(
                cursor_recognition_section.get("enabled"),
                cursor_gesture_defaults.enabled,
            ),
            confidence=_as_float(
                cursor_recognition_section.get("confidence"),
                cursor_gesture_defaults.confidence,
            ),
            index_pinch_tracking_enabled=_as_bool(
                cursor_recognition_section.get("index_pinch_tracking_enabled"),
                cursor_gesture_defaults.index_pinch_tracking_enabled,
            ),
            index_pinch_thumb_tip=_as_int(
                cursor_recognition_section.get("index_pinch_thumb_tip"),
                cursor_gesture_defaults.index_pinch_thumb_tip,
            ),
            index_pinch_tip=_as_int(
                cursor_recognition_section.get("index_pinch_tip"),
                cursor_gesture_defaults.index_pinch_tip,
            ),
            index_squeeze_ratio=_as_float(
                cursor_recognition_section.get("index_squeeze_ratio"),
                cursor_gesture_defaults.index_squeeze_ratio,
            ),
            index_release_ratio=_as_float(
                cursor_recognition_section.get("index_release_ratio"),
                cursor_gesture_defaults.index_release_ratio,
            ),
            middle_pinch_tracking_enabled=_as_bool(
                cursor_recognition_section.get("middle_pinch_tracking_enabled"),
                cursor_gesture_defaults.middle_pinch_tracking_enabled,
            ),
            middle_pinch_thumb_tip=_as_int(
                cursor_recognition_section.get("middle_pinch_thumb_tip"),
                cursor_gesture_defaults.middle_pinch_thumb_tip,
            ),
            middle_pinch_tip=_as_int(
                cursor_recognition_section.get("middle_pinch_tip"),
                cursor_gesture_defaults.middle_pinch_tip,
            ),
            middle_pinch_press_ratio=_as_float(
                cursor_recognition_section.get("middle_pinch_press_ratio"),
                cursor_gesture_defaults.middle_pinch_press_ratio,
            ),
            middle_pinch_release_ratio=_as_float(
                cursor_recognition_section.get("middle_pinch_release_ratio"),
                cursor_gesture_defaults.middle_pinch_release_ratio,
            ),
        ),
        cursor_actions=CursorActionsConfig(
            cooldown_seconds=_as_float(
                cursor_actions_section.get("cooldown_seconds"),
                cursor_actions_defaults.cooldown_seconds,
            ),
            mapping=dict(cursor_actions_section.get("mapping", cursor_actions_defaults.mapping)),
        ),
        cursor_toggle=GestureToggleConfig(
            enabled=_as_bool(
                cursor_activation_section.get("enabled"),
                cursor_toggle_defaults.enabled,
            ),
            target=toggle_target,
            hold_seconds=max(
                0.0,
                _as_float(
                    cursor_activation_section.get("hold_seconds"),
                    cursor_toggle_defaults.hold_seconds,
                ),
            ),
            cooldown_seconds=max(
                0.0,
                _as_float(
                    cursor_activation_section.get("cooldown_seconds"),
                    cursor_toggle_defaults.cooldown_seconds,
                ),
            ),
        ),
    )
