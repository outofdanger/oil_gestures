from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oil_gestures.core.constants import (
    DEFAULT_CURSOR_DRY_RUN,
    DEFAULT_CURSOR_SMOOTHING_ALPHA,
    DEFAULT_CURSOR_TOGGLE_CONFIDENCE,
    DEFAULT_CURSOR_TOGGLE_COOLDOWN_SECONDS,
    DEFAULT_CURSOR_TOGGLE_GESTURE,
    DEFAULT_MOUSE_ACTION_COOLDOWN_SECONDS,
    DEFAULT_POINTER_LANDMARK,
)
from oil_gestures.core.enums import GestureName, RecognitionSource
from oil_gestures.cursor.action_mapper import DEFAULT_DYNAMIC_MAPPING, DEFAULT_STATIC_FALLBACK_MAPPING
from oil_gestures.gestures.dynamic.rule_based_dynamic import RuleBasedDynamicConfig
from oil_gestures.gestures.static.rules import StaticRuleConfig
from oil_gestures.gestures.static.static_recognizer import StaticRecognizerConfig

try:
    import yaml
except Exception:  # pragma: no cover - handled with a clear runtime error.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CameraConfig:
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 60
    mirror: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    debug: bool = True
    show_camera_feed: bool = True
    show_landmarks: bool = True
    show_debug_overlay: bool = True
    safe_exit_key: str = "q"
    window_name: str = "Oil Gestures"


@dataclass(frozen=True)
class MediaPipeConfig:
    max_hands: int = 1
    model_complexity: int = 1
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.65
    model_path: str = "assets/models/mediapipe/hand_landmarker.task"


@dataclass(frozen=True)
class CursorConfig:
    enabled: bool = False
    dry_run: bool = DEFAULT_CURSOR_DRY_RUN
    pointer_source: str = DEFAULT_POINTER_LANDMARK
    toggle_gesture: str = DEFAULT_CURSOR_TOGGLE_GESTURE
    toggle_min_confidence: float = DEFAULT_CURSOR_TOGGLE_CONFIDENCE
    toggle_cooldown_seconds: float = DEFAULT_CURSOR_TOGGLE_COOLDOWN_SECONDS
    margin_x: float = 0.08
    margin_top: float = 0.10
    margin_bottom: float = 0.12
    invert_y: bool = False
    smoothing_alpha: float = DEFAULT_CURSOR_SMOOTHING_ALPHA
    min_cutoff: float = 7.0
    beta: float = 0.080
    derivative_cutoff: float = 1.0
    dead_zone_points: float = 0.0
    max_speed_points_per_second: float = 50000.0
    reacquire_frames: int = 1
    lost_reset_seconds: float = 0.20


@dataclass(frozen=True)
class CursorActionsConfig:
    cooldown_seconds: float = DEFAULT_MOUSE_ACTION_COOLDOWN_SECONDS
    dynamic_mapping: dict[str, str] = field(
        default_factory=lambda: {gesture.value: action.value for gesture, action in DEFAULT_DYNAMIC_MAPPING.items()}
    )
    static_fallback_mapping: dict[str, str] = field(
        default_factory=lambda: {
            gesture.value: action.value for gesture, action in DEFAULT_STATIC_FALLBACK_MAPPING.items()
        }
    )


@dataclass(frozen=True)
class OilGesturesConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    static: StaticRecognizerConfig = field(default_factory=StaticRecognizerConfig)
    dynamic: RuleBasedDynamicConfig = field(default_factory=RuleBasedDynamicConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    cursor_actions: CursorActionsConfig = field(default_factory=CursorActionsConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configs. Install it with: python3 -m pip install PyYAML")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


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
    cursor_section = gestures_yaml.get("cursor", {})
    cursor_actions_section = gestures_yaml.get("cursor_actions", {})
    static_section = gestures_yaml.get("static", {})
    dynamic_section = gestures_yaml.get("dynamic", {})
    mediapipe_section = model_yaml.get("mediapipe", {})

    cursor_defaults = CursorConfig()
    cursor_actions_defaults = CursorActionsConfig()
    static_defaults = StaticRecognizerConfig()
    dynamic_defaults = RuleBasedDynamicConfig()
    media_defaults = MediaPipeConfig()

    model_path = str(mediapipe_section.get("model_path", media_defaults.model_path))
    if not Path(model_path).is_absolute():
        model_path = str(root / model_path)

    return OilGesturesConfig(
        camera=CameraConfig(
            device_id=_as_int(camera_section.get("device_id"), 0),
            width=_as_int(camera_section.get("width"), 1280),
            height=_as_int(camera_section.get("height"), 720),
            fps=_as_int(camera_section.get("fps"), 60),
            mirror=_as_bool(camera_section.get("mirror"), True),
        ),
        runtime=RuntimeConfig(
            debug=_as_bool(app_section.get("debug"), True),
            show_camera_feed=_as_bool(runtime_section.get("show_camera_feed"), True),
            show_landmarks=_as_bool(runtime_section.get("show_landmarks"), True),
            show_debug_overlay=_as_bool(runtime_section.get("show_debug_overlay"), True),
            safe_exit_key=str(runtime_section.get("safe_exit_key", "q")),
            window_name=str(runtime_section.get("window_name", "Oil Gestures")),
        ),
        mediapipe=MediaPipeConfig(
            max_hands=max(1, _as_int(mediapipe_section.get("max_hands"), media_defaults.max_hands)),
            model_complexity=_as_int(mediapipe_section.get("model_complexity"), media_defaults.model_complexity),
            min_detection_confidence=_as_float(
                mediapipe_section.get("min_detection_confidence"),
                media_defaults.min_detection_confidence,
            ),
            min_tracking_confidence=_as_float(
                mediapipe_section.get("min_tracking_confidence"),
                media_defaults.min_tracking_confidence,
            ),
            model_path=model_path,
        ),
        static=StaticRecognizerConfig(
            enabled=_as_bool(static_section.get("enabled"), static_defaults.enabled),
            min_confidence=_as_float(static_section.get("min_confidence"), static_defaults.min_confidence),
            rule_config=StaticRuleConfig(
                ok_pinch_ratio=_as_float(
                    static_section.get("ok_pinch_ratio"),
                    static_defaults.rule_config.ok_pinch_ratio,
                ),
                finger_extension_margin=_as_float(
                    static_section.get("finger_extension_margin"),
                    static_defaults.rule_config.finger_extension_margin,
                ),
                fist_curl_margin=_as_float(
                    static_section.get("fist_curl_margin"),
                    static_defaults.rule_config.fist_curl_margin,
                ),
                open_palm_min_extended=max(
                    1,
                    _as_int(
                        static_section.get("open_palm_min_extended"),
                        static_defaults.rule_config.open_palm_min_extended,
                    ),
                ),
            ),
        ),
        dynamic=RuleBasedDynamicConfig(
            enabled=_as_bool(dynamic_section.get("enabled"), dynamic_defaults.enabled),
            fallback_gesture=GestureName(
                str(dynamic_section.get("fallback_gesture", dynamic_defaults.fallback_gesture.value))
            ),
            source=RecognitionSource(str(dynamic_section.get("source", dynamic_defaults.source.value))),
            confidence=_as_float(dynamic_section.get("confidence"), dynamic_defaults.confidence),
            require_hand=_as_bool(dynamic_section.get("require_hand"), dynamic_defaults.require_hand),
            pinch_tracking_enabled=_as_bool(
                dynamic_section.get("pinch_tracking_enabled"),
                dynamic_defaults.pinch_tracking_enabled,
            ),
            press_thumb_tip=_as_int(dynamic_section.get("press_thumb_tip"), dynamic_defaults.press_thumb_tip),
            press_index_tip=_as_int(dynamic_section.get("press_index_tip"), dynamic_defaults.press_index_tip),
            press_ratio=_as_float(dynamic_section.get("press_ratio"), dynamic_defaults.press_ratio),
            release_ratio=_as_float(dynamic_section.get("release_ratio"), dynamic_defaults.release_ratio),
            middle_pinch_tracking_enabled=_as_bool(
                dynamic_section.get("middle_pinch_tracking_enabled"),
                dynamic_defaults.middle_pinch_tracking_enabled,
            ),
            middle_pinch_thumb_tip=_as_int(
                dynamic_section.get("middle_pinch_thumb_tip"),
                dynamic_defaults.middle_pinch_thumb_tip,
            ),
            middle_pinch_tip=_as_int(dynamic_section.get("middle_pinch_tip"), dynamic_defaults.middle_pinch_tip),
            middle_pinch_press_ratio=_as_float(
                dynamic_section.get("middle_pinch_press_ratio"),
                dynamic_defaults.middle_pinch_press_ratio,
            ),
            middle_pinch_release_ratio=_as_float(
                dynamic_section.get("middle_pinch_release_ratio"),
                dynamic_defaults.middle_pinch_release_ratio,
            ),
            rotation_tracking_enabled=_as_bool(
                dynamic_section.get("rotation_tracking_enabled"),
                dynamic_defaults.rotation_tracking_enabled,
            ),
            rotation_window=max(2, _as_int(dynamic_section.get("rotation_window"), dynamic_defaults.rotation_window)),
            rotation_threshold_radians=_as_float(
                dynamic_section.get("rotation_threshold_radians"),
                dynamic_defaults.rotation_threshold_radians,
            ),
            rotation_cooldown_seconds=_as_float(
                dynamic_section.get("rotation_cooldown_seconds"),
                dynamic_defaults.rotation_cooldown_seconds,
            ),
        ),
        cursor=CursorConfig(
            enabled=_as_bool(cursor_section.get("enabled"), cursor_defaults.enabled),
            dry_run=_as_bool(cursor_section.get("dry_run"), cursor_defaults.dry_run),
            pointer_source=str(cursor_section.get("pointer_source", cursor_defaults.pointer_source)),
            toggle_gesture=str(cursor_section.get("toggle_gesture", cursor_defaults.toggle_gesture)),
            toggle_min_confidence=_as_float(
                cursor_section.get("toggle_min_confidence"),
                cursor_defaults.toggle_min_confidence,
            ),
            toggle_cooldown_seconds=_as_float(
                cursor_section.get("toggle_cooldown_seconds"),
                cursor_defaults.toggle_cooldown_seconds,
            ),
            margin_x=_as_float(cursor_section.get("margin_x"), cursor_defaults.margin_x),
            margin_top=_as_float(cursor_section.get("margin_top"), cursor_defaults.margin_top),
            margin_bottom=_as_float(cursor_section.get("margin_bottom"), cursor_defaults.margin_bottom),
            invert_y=_as_bool(cursor_section.get("invert_y"), cursor_defaults.invert_y),
            smoothing_alpha=_as_float(cursor_section.get("smoothing_alpha"), cursor_defaults.smoothing_alpha),
            min_cutoff=_as_float(cursor_section.get("min_cutoff"), cursor_defaults.min_cutoff),
            beta=_as_float(cursor_section.get("beta"), cursor_defaults.beta),
            derivative_cutoff=_as_float(cursor_section.get("derivative_cutoff"), cursor_defaults.derivative_cutoff),
            dead_zone_points=_as_float(cursor_section.get("dead_zone_points"), cursor_defaults.dead_zone_points),
            max_speed_points_per_second=_as_float(
                cursor_section.get("max_speed_points_per_second"),
                cursor_defaults.max_speed_points_per_second,
            ),
            reacquire_frames=max(1, _as_int(cursor_section.get("reacquire_frames"), cursor_defaults.reacquire_frames)),
            lost_reset_seconds=_as_float(cursor_section.get("lost_reset_seconds"), cursor_defaults.lost_reset_seconds),
        ),
        cursor_actions=CursorActionsConfig(
            cooldown_seconds=_as_float(
                cursor_actions_section.get("cooldown_seconds"),
                cursor_actions_defaults.cooldown_seconds,
            ),
            dynamic_mapping=dict(
                cursor_actions_section.get("dynamic_mapping", cursor_actions_defaults.dynamic_mapping)
            ),
            static_fallback_mapping=dict(
                cursor_actions_section.get(
                    "static_fallback_mapping",
                    cursor_actions_defaults.static_fallback_mapping,
                )
            ),
        ),
    )
