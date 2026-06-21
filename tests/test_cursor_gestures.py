from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.app_config import load_config
from oil_gestures.core.enums import CursorAction, GestureName, Handedness, MouseAction, RecognitionSource
from oil_gestures.core.types import GestureResult, LandmarkPacket, ScreenPosition
from oil_gestures.cursor.action_mapper import CursorActionMapper
from oil_gestures.cursor.cursor_pipeline import CursorPipeline
from oil_gestures.cursor.hand_pointer import HandPointer, HandPointerConfig
from oil_gestures.cursor.mouse_controller import MouseController, MouseControllerConfig
from oil_gestures.gestures.cursor.cursor_recognizer import CursorGestureConfig, CursorGestureRecognizer
from oil_gestures.gestures.dynamic.dynamic_recognizer import DynamicGestureRecognizer


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
        self.drags = []
        self.executed = []

    def get_position(self):
        return (0.0, 0.0)

    def move_to(self, position):
        self.moves.append(position)

    def drag_to(self, position):
        self.drags.append(position)

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


def _cursor_gesture(name: GestureName, timestamp: float) -> GestureResult:
    return GestureResult(name, 0.9, RecognitionSource.CURSOR_RULES, timestamp)


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


def test_action_mapper_maps_only_cursor_gestures() -> None:
    mapper = CursorActionMapper()
    position = ScreenPosition(100, 200, 123.0)

    squeeze = mapper.map(_cursor_gesture(GestureName.INDEX_SQUEEZE, 123.0), position)
    middle_pinch = mapper.map(_cursor_gesture(GestureName.MIDDLE_PINCH, 123.0), position)
    static = mapper.map(
        GestureResult(GestureName.FIST, 0.99, RecognitionSource.MEDIAPIPE, 123.0),
        position,
    )
    dynamic = mapper.map(
        GestureResult(GestureName.INDEX_SQUEEZE, 0.99, RecognitionSource.DYNAMIC_MODEL, 123.0),
        position,
    )

    assert squeeze.action == CursorAction.GRAB
    assert squeeze.source_gesture == GestureName.INDEX_SQUEEZE
    assert middle_pinch.action == CursorAction.RIGHT_CLICK
    assert middle_pinch.source_gesture == GestureName.MIDDLE_PINCH
    assert static.action == CursorAction.NONE
    assert static.source_gesture == GestureName.UNKNOWN
    assert dynamic.action == CursorAction.NONE


def test_middle_pinch_maps_to_right_click() -> None:
    result = CursorActionMapper().map(_cursor_gesture(GestureName.MIDDLE_PINCH, 123.0))

    assert result.action == CursorAction.RIGHT_CLICK
    assert result.source_gesture == GestureName.MIDDLE_PINCH


def test_old_cursor_gesture_names_are_not_accepted() -> None:
    with pytest.raises(ValueError):
        CursorActionMapper.from_strings({"SQUEEZE": "GRAB"})


def test_cursor_recognizer_produces_index_mcp_for_detected_hand() -> None:
    packet = LandmarkPacket(True, _open_hand(), Handedness.UNKNOWN, 0.9, 123.0)

    result = CursorGestureRecognizer().update(packet)

    assert result is not None
    assert result.name == GestureName.INDEX_MCP
    assert result.source == RecognitionSource.CURSOR_RULES


def test_cursor_recognizer_tracks_index_press_transitions() -> None:
    recognizer = CursorGestureRecognizer(CursorGestureConfig(index_pinch_tracking_enabled=True))
    landmarks = _open_hand()

    assert recognizer.update(_packet(123.0, landmarks)).name == GestureName.INDEX_MCP

    landmarks[4] = Landmark(0.50, 0.10)
    landmarks[8] = Landmark(0.51, 0.10)
    assert recognizer.update(_packet(124.0, landmarks)).name == GestureName.INDEX_SQUEEZE
    assert recognizer.index_pressed

    assert recognizer.update(_packet(125.0, landmarks)).name == GestureName.INDEX_MCP

    landmarks[8] = Landmark(0.80, 0.10)
    assert recognizer.update(_packet(126.0, landmarks)).name == GestureName.INDEX_RELEASE
    assert not recognizer.index_pressed


def test_cursor_recognizer_tracks_middle_pinch_without_toggling_state() -> None:
    recognizer = CursorGestureRecognizer(CursorGestureConfig(middle_pinch_tracking_enabled=True))
    landmarks = _open_hand()
    landmarks[4] = Landmark(0.60, 0.10)
    landmarks[8] = Landmark(0.25, 0.10)
    landmarks[12] = Landmark(0.61, 0.10)

    assert recognizer.update(_packet(124.0, landmarks)).name == GestureName.MIDDLE_PINCH
    assert recognizer.middle_pressed
    assert recognizer.update(_packet(125.0, landmarks)).name == GestureName.INDEX_MCP

    landmarks[12] = Landmark(0.90, 0.10)
    assert recognizer.update(_packet(126.0, landmarks)).name == GestureName.INDEX_MCP
    assert not recognizer.middle_pressed


def test_cursor_recognizer_resets_when_hand_is_lost() -> None:
    recognizer = CursorGestureRecognizer()
    recognizer.index_pressed = True
    recognizer.middle_pressed = True

    assert recognizer.update(_packet(200.0, None)) is None
    assert not recognizer.index_pressed
    assert not recognizer.middle_pressed


def test_general_dynamic_recognizer_has_no_rule_based_cursor_fallback() -> None:
    packet = LandmarkPacket(True, _open_hand(), Handedness.UNKNOWN, 0.9, 123.0)

    assert DynamicGestureRecognizer().update(packet) is None


def test_general_dynamic_recognizer_accepts_only_dynamic_model_results() -> None:
    class Model:
        def __init__(self, source: RecognitionSource) -> None:
            self.source = source

        def update(self, packet: LandmarkPacket) -> GestureResult:
            return GestureResult(GestureName.IDLE, 0.9, self.source, packet.timestamp)

        def reset(self) -> None:
            pass

    packet = LandmarkPacket(True, _open_hand(), Handedness.UNKNOWN, 0.9, 123.0)

    accepted = DynamicGestureRecognizer(model=Model(RecognitionSource.DYNAMIC_MODEL)).update(packet)
    rejected = DynamicGestureRecognizer(model=Model(RecognitionSource.CURSOR_RULES)).update(packet)

    assert accepted is not None
    assert accepted.name == GestureName.IDLE
    assert rejected is None


def test_cursor_pipeline_does_not_repeat_mouse_down_while_pressed() -> None:
    mouse = RecordingMouse()
    pipeline = _cursor_pipeline(mouse)

    pipeline.process(_packet(1.0, _open_hand()), _cursor_gesture(GestureName.INDEX_SQUEEZE, 1.0))
    pipeline.process(_packet(1.1, _open_hand()), _cursor_gesture(GestureName.INDEX_SQUEEZE, 1.1))

    assert mouse.executed == [CursorAction.GRAB]
    assert pipeline.state.pressed


def test_cursor_pipeline_releases_mouse_when_hand_is_lost() -> None:
    mouse = RecordingMouse()
    pipeline = _cursor_pipeline(mouse)

    pipeline.process(_packet(1.0, _open_hand()), _cursor_gesture(GestureName.INDEX_SQUEEZE, 1.0))
    pipeline.process(_packet(1.1, None))

    assert mouse.executed == [CursorAction.GRAB, CursorAction.RELEASE]
    assert not pipeline.state.pressed


def test_cursor_pipeline_drags_while_index_is_squeezed() -> None:
    mouse = RecordingMouse()
    pipeline = _cursor_pipeline(mouse)

    pipeline.process(_packet(1.0, _open_hand()), _cursor_gesture(GestureName.INDEX_SQUEEZE, 1.0))
    pipeline.process(_packet(1.1, _open_hand()), _cursor_gesture(GestureName.INDEX_MCP, 1.1))
    pipeline.process(_packet(1.2, _open_hand()), _cursor_gesture(GestureName.INDEX_RELEASE, 1.2))

    assert mouse.executed == [CursorAction.GRAB, CursorAction.RELEASE]
    assert len(mouse.drags) == 2
    assert not pipeline.state.pressed


def test_cursor_pipeline_executes_middle_pinch_as_right_click() -> None:
    mouse = RecordingMouse()
    pipeline = _cursor_pipeline(mouse)

    result = pipeline.process(
        _packet(1.0, _open_hand()),
        _cursor_gesture(GestureName.MIDDLE_PINCH, 1.0),
    )

    assert result.action == CursorAction.RIGHT_CLICK
    assert mouse.executed == [CursorAction.RIGHT_CLICK]


def test_cursor_pipeline_disabled_does_not_execute_cursor_actions() -> None:
    mouse = RecordingMouse()
    pipeline = _cursor_pipeline(mouse)
    pipeline.enabled = False

    pipeline.process(_packet(1.0, _open_hand()), _cursor_gesture(GestureName.INDEX_SQUEEZE, 1.0))

    assert mouse.executed == []
    assert not pipeline.state.pressed


def test_mouse_controller_reports_move_while_left_button_is_down() -> None:
    mouse = MouseController(MouseControllerConfig(dry_run=True))
    position = ScreenPosition(10, 20, 1.0)

    assert mouse.move_to(position).action == MouseAction.MOVE
    mouse.mouse_down(position)
    assert mouse.drag_to(ScreenPosition(30, 40, 1.1)).action == MouseAction.DRAG
    mouse.mouse_up(ScreenPosition(30, 40, 1.2))
    assert mouse.move_to(ScreenPosition(50, 60, 1.3)).action == MouseAction.MOVE


def test_mouse_controller_executes_right_click_action() -> None:
    mouse = MouseController(MouseControllerConfig(dry_run=True))

    result = mouse.execute(CursorAction.RIGHT_CLICK, ScreenPosition(10, 20, 1.0))

    assert result.action == MouseAction.RIGHT_CLICK


def test_loaded_config_keeps_cursor_secondary_and_isolated() -> None:
    config = load_config()

    assert config.cursor.dry_run
    assert not config.cursor.enabled
    assert not hasattr(config.cursor, "toggle_gesture")
    assert config.cursor_actions.mapping == {
        "INDEX_MCP": "MOVE_CURSOR",
        "INDEX_SQUEEZE": "GRAB",
        "INDEX_RELEASE": "RELEASE",
        "MIDDLE_PINCH": "RIGHT_CLICK",
    }
