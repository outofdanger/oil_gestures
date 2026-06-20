from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.app_config import load_config
from oil_gestures.core.enums import CursorAction, GestureName, Handedness, MouseAction, RecognitionSource
from oil_gestures.core.types import GestureResult, LandmarkPacket, ScreenPosition
from oil_gestures.cursor.action_mapper import CursorActionMapper
from oil_gestures.cursor.feature_toggle import CursorFeatureToggle, CursorFeatureToggleConfig
from oil_gestures.cursor.cursor_pipeline import CursorPipeline
from oil_gestures.cursor.hand_pointer import HandPointer, HandPointerConfig
from oil_gestures.cursor.mouse_controller import MouseController, MouseControllerConfig
from oil_gestures.gestures.dynamic.rule_based_dynamic import RuleBasedDynamicConfig, RuleBasedDynamicRecognizer


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0


class PassthroughScreenMapper:
    def map(self, pointer):
        if not pointer.visible:
            return None
        return ScreenPosition(int(pointer.x * 1000), int(pointer.y * 1000), pointer.timestamp)


class PassthroughSmoother:
    def reset(self, current_position=None):
        pass

    def apply(self, target, current_position=None):
        return target


class RecordingMouse:
    def __init__(self) -> None:
        self.moves = []
        self.executed = []

    def get_position(self):
        return (0.0, 0.0)

    def move_to(self, position):
        self.moves.append(position)

    def execute(self, action, position=None):
        self.executed.append(action)


def _open_hand() -> list[Landmark]:
    return [
        Landmark(0.50, 0.90),
        Landmark(0.40, 0.80),
        Landmark(0.35, 0.70),
        Landmark(0.30, 0.60),
        Landmark(0.20, 0.50),
        Landmark(0.50, 0.55),
        Landmark(0.50, 0.40),
        Landmark(0.50, 0.25),
        Landmark(0.50, 0.10),
        Landmark(0.60, 0.55),
        Landmark(0.60, 0.40),
        Landmark(0.60, 0.25),
        Landmark(0.60, 0.10),
        Landmark(0.70, 0.55),
        Landmark(0.70, 0.40),
        Landmark(0.70, 0.25),
        Landmark(0.70, 0.10),
        Landmark(0.80, 0.60),
        Landmark(0.80, 0.45),
        Landmark(0.80, 0.30),
        Landmark(0.80, 0.15),
    ]


def _packet(timestamp: float, landmarks: list[Landmark] | None = None) -> LandmarkPacket:
    return LandmarkPacket(landmarks is not None, landmarks, Handedness.UNKNOWN, 0.9, timestamp)


def _gesture(name: GestureName, timestamp: float) -> GestureResult:
    return GestureResult(name, 0.9, RecognitionSource.DYNAMIC_RULES, timestamp)


def _static_gesture(name: GestureName, timestamp: float) -> GestureResult:
    return GestureResult(name, 0.9, RecognitionSource.STATIC_RULES, timestamp)


def _cursor_pipeline(
    mouse: RecordingMouse,
    cooldown: float = 0.0,
    action_mapper: CursorActionMapper | None = None,
) -> CursorPipeline:
    return CursorPipeline(
        hand_pointer=HandPointer(HandPointerConfig()),
        screen_mapper=PassthroughScreenMapper(),
        smoother=PassthroughSmoother(),
        action_mapper=action_mapper or CursorActionMapper(),
        mouse=mouse,
        reacquire_frames=1,
        lost_reset_seconds=0.20,
        action_cooldown_seconds=cooldown,
    )


def test_hand_pointer_default_uses_index_mcp_point_5() -> None:
    landmarks = _open_hand()
    pointer = HandPointer(HandPointerConfig())
    packet = LandmarkPacket(True, landmarks, Handedness.UNKNOWN, 1.0, 123.0)

    result = pointer.extract(packet)

    assert result.visible
    assert pointer.pointer_index == 5
    assert result.x == landmarks[5].x
    assert result.y == landmarks[5].y


def test_action_mapper_prefers_dynamic_gesture() -> None:
    mapper = CursorActionMapper()
    position = ScreenPosition(100, 200, 123.0)
    dynamic = GestureResult(GestureName.SQUEEZE, 0.88, RecognitionSource.DYNAMIC_RULES, 123.0)
    static = GestureResult(GestureName.OK_SIGN, 0.99, RecognitionSource.STATIC_RULES, 123.0)

    result = mapper.map(dynamic_result=dynamic, static_result=static, screen_position=position)

    assert result.action == CursorAction.GRAB
    assert result.source_gesture == GestureName.SQUEEZE
    assert result.screen_position == position


def test_action_mapper_uses_static_fallback() -> None:
    mapper = CursorActionMapper()
    static = GestureResult(GestureName.OK_SIGN, 0.91, RecognitionSource.STATIC_RULES, 123.0)

    result = mapper.map(static_result=static)

    assert result.action == CursorAction.SELECT
    assert result.source_gesture == GestureName.OK_SIGN


def test_action_mapper_maps_rotation_to_pressure() -> None:
    mapper = CursorActionMapper()
    clockwise = GestureResult(GestureName.ROTATE_CLOCKWISE, 0.8, RecognitionSource.DYNAMIC_RULES, 123.0)
    counterclockwise = GestureResult(
        GestureName.ROTATE_COUNTERCLOCKWISE,
        0.8,
        RecognitionSource.DYNAMIC_RULES,
        124.0,
    )

    assert mapper.map(dynamic_result=clockwise).action == CursorAction.INCREASE_PRESSURE
    assert mapper.map(dynamic_result=counterclockwise).action == CursorAction.DECREASE_PRESSURE


def test_action_mapper_does_not_use_middle_pinch_as_cursor_action() -> None:
    mapper = CursorActionMapper()
    gesture = GestureResult(GestureName.MIDDLE_PINCH, 0.8, RecognitionSource.DYNAMIC_RULES, 123.0)

    result = mapper.map(dynamic_result=gesture)

    assert result.action == CursorAction.NONE
    assert result.source_gesture == GestureName.UNKNOWN


def test_action_mapper_rejects_drag_mapping_for_this_issue() -> None:
    with pytest.raises(ValueError, match="DRAG"):
        CursorActionMapper.from_strings({"SQUEEZE": "DRAG"}, {})


def test_dynamic_recognizer_produces_pointing_index_for_detected_hand() -> None:
    config = RuleBasedDynamicConfig(enabled=True)
    packet = LandmarkPacket(True, _open_hand(), Handedness.UNKNOWN, 0.9, 123.0)

    result = RuleBasedDynamicRecognizer(config).update(packet)

    assert result is not None
    assert result.name == GestureName.POINTING_INDEX
    assert result.source == RecognitionSource.DYNAMIC_RULES


def test_dynamic_recognizer_tracks_press_transitions() -> None:
    config = RuleBasedDynamicConfig(enabled=True, pinch_tracking_enabled=True)
    recognizer = RuleBasedDynamicRecognizer(config)
    landmarks = _open_hand()
    packet = LandmarkPacket(True, landmarks, Handedness.UNKNOWN, 0.9, 123.0)

    assert recognizer.update(packet).name == GestureName.POINTING_INDEX

    landmarks[4] = Landmark(0.50, 0.10)
    landmarks[8] = Landmark(0.51, 0.10)
    packet = LandmarkPacket(True, landmarks, Handedness.UNKNOWN, 0.9, 124.0)
    assert recognizer.update(packet).name == GestureName.SQUEEZE
    assert recognizer.pressed

    packet = LandmarkPacket(True, landmarks, Handedness.UNKNOWN, 0.9, 125.0)
    assert recognizer.update(packet).name == GestureName.POINTING_INDEX

    landmarks[8] = Landmark(0.80, 0.10)
    packet = LandmarkPacket(True, landmarks, Handedness.UNKNOWN, 0.9, 126.0)
    assert recognizer.update(packet).name == GestureName.RELEASE
    assert not recognizer.pressed


def test_dynamic_recognizer_tracks_middle_pinch_toggle_gesture() -> None:
    config = RuleBasedDynamicConfig(enabled=True, middle_pinch_tracking_enabled=True)
    recognizer = RuleBasedDynamicRecognizer(config)
    landmarks = _open_hand()

    landmarks[4] = Landmark(0.60, 0.10)
    landmarks[8] = Landmark(0.25, 0.10)
    landmarks[12] = Landmark(0.61, 0.10)
    packet = LandmarkPacket(True, landmarks, Handedness.UNKNOWN, 0.9, 124.0)
    assert recognizer.update(packet).name == GestureName.MIDDLE_PINCH
    assert recognizer.middle_pressed

    packet = LandmarkPacket(True, landmarks, Handedness.UNKNOWN, 0.9, 125.0)
    assert recognizer.update(packet).name == GestureName.POINTING_INDEX

    landmarks[12] = Landmark(0.90, 0.10)
    packet = LandmarkPacket(True, landmarks, Handedness.UNKNOWN, 0.9, 126.0)
    assert recognizer.update(packet).name == GestureName.POINTING_INDEX
    assert not recognizer.middle_pressed


def test_dynamic_recognizer_resets_button_state_when_hand_is_lost() -> None:
    config = RuleBasedDynamicConfig(enabled=True, middle_pinch_tracking_enabled=True)
    recognizer = RuleBasedDynamicRecognizer(config)
    recognizer.pressed = True
    recognizer.middle_pressed = True

    assert recognizer.update(_packet(200.0, None)) is None
    assert not recognizer.pressed
    assert not recognizer.middle_pressed


def test_cursor_feature_toggle_uses_gesture_without_mouse_action() -> None:
    toggle = CursorFeatureToggle(
        CursorFeatureToggleConfig(
            initial_enabled=False,
            toggle_gesture=GestureName.MIDDLE_PINCH,
            cooldown_seconds=0.5,
        )
    )

    first = toggle.update([_gesture(GestureName.MIDDLE_PINCH, 1.0)], 1.0)
    blocked = toggle.update([_gesture(GestureName.MIDDLE_PINCH, 1.2)], 1.2)
    second = toggle.update([_gesture(GestureName.MIDDLE_PINCH, 1.6)], 1.6)

    assert first.toggled and first.enabled
    assert not blocked.toggled and blocked.enabled
    assert second.toggled and not second.enabled


def test_cursor_pipeline_does_not_repeat_mouse_down_while_pressed() -> None:
    mouse = RecordingMouse()
    pipeline = _cursor_pipeline(mouse)

    pipeline.process(_packet(1.0, _open_hand()), dynamic_gesture=_gesture(GestureName.SQUEEZE, 1.0))
    pipeline.process(_packet(1.1, _open_hand()), dynamic_gesture=_gesture(GestureName.SQUEEZE, 1.1))

    assert mouse.executed == [CursorAction.GRAB]
    assert pipeline.state.pressed


def test_cursor_pipeline_releases_mouse_when_hand_is_lost() -> None:
    mouse = RecordingMouse()
    pipeline = _cursor_pipeline(mouse)

    pipeline.process(_packet(1.0, _open_hand()), dynamic_gesture=_gesture(GestureName.SQUEEZE, 1.0))
    pipeline.process(_packet(1.1, None))

    assert mouse.executed == [CursorAction.GRAB, CursorAction.RELEASE]
    assert not pipeline.state.pressed


def test_cursor_pipeline_debounces_repeated_selects() -> None:
    mouse = RecordingMouse()
    pipeline = _cursor_pipeline(mouse, cooldown=0.5)

    pipeline.process(_packet(1.0, _open_hand()), static_gesture=_static_gesture(GestureName.OK_SIGN, 1.0))
    pipeline.process(_packet(1.2, _open_hand()), static_gesture=_static_gesture(GestureName.OK_SIGN, 1.2))
    pipeline.process(_packet(1.6, _open_hand()), static_gesture=_static_gesture(GestureName.OK_SIGN, 1.6))

    assert mouse.executed == [CursorAction.SELECT, CursorAction.SELECT]


def test_cursor_pipeline_disabled_does_not_execute_cursor_actions() -> None:
    mouse = RecordingMouse()
    pipeline = _cursor_pipeline(mouse)
    pipeline.enabled = False

    pipeline.process(_packet(1.0, _open_hand()), dynamic_gesture=_gesture(GestureName.SQUEEZE, 1.0))
    pipeline.process(_packet(1.1, _open_hand()), dynamic_gesture=_gesture(GestureName.SQUEEZE, 1.1))

    assert mouse.executed == []
    assert not pipeline.state.pressed


def test_mouse_controller_reports_move_while_left_button_is_down() -> None:
    mouse = MouseController(MouseControllerConfig(dry_run=True))
    position = ScreenPosition(10, 20, 1.0)

    assert mouse.move_to(position).action == MouseAction.MOVE
    mouse.mouse_down(position)
    assert mouse.move_to(ScreenPosition(30, 40, 1.1)).action == MouseAction.MOVE
    mouse.mouse_up(ScreenPosition(30, 40, 1.2))
    assert mouse.move_to(ScreenPosition(50, 60, 1.3)).action == MouseAction.MOVE


def test_loaded_config_is_safe_and_gesture_first_by_default() -> None:
    config = load_config()

    assert config.cursor.dry_run
    assert not config.cursor.enabled
    assert config.cursor.toggle_gesture == GestureName.MIDDLE_PINCH.value
