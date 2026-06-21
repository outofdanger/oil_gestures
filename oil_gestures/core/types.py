from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oil_gestures.core.enums import (
    CommandName,
    CursorAction,
    GestureName,
    Handedness,
    MouseAction,
    RecognitionSource,
)


@dataclass
class FramePacket:
    """
    Frame data produced by the camera/frame processing layer.

    Producer:
        vision.camera.py / vision.frame_processor.py

    Consumers:
        vision.mediapipe_landmarker.py
        ui.camera_widget.py
        vision.drawing.py
    """

    frame: Any
    width: int
    height: int
    timestamp: float


@dataclass
class LandmarkPacket:
    """
    Hand landmark data produced by the MediaPipe layer.

    MediaPipe is the single source of hand landmarks in this project.
    The exact landmarks type is flexible for MVP because it may be a MediaPipe
    object, list of normalized points, numpy array, or custom structure.

    Producer:
        vision.mediapipe_landmarker.py

    Consumers:
        gestures.static.static_recognizer.py
        gestures.dynamic.sequence_buffer.py
        gestures.cursor.cursor_recognizer.py
        cursor.hand_pointer.py
        vision.drawing.py
    """

    hand_detected: bool
    landmarks: Any | None
    handedness: Handedness
    confidence: float
    timestamp: float


@dataclass
class GestureResult:
    """
    Result produced by an independent gesture-recognition subsystem.

    Examples:
        static: OPEN_PALM, FIST, OK_SIGN
        cursor: INDEX_MCP, INDEX_SQUEEZE, INDEX_RELEASE, MIDDLE_PINCH

    Producers:
        gestures.static.static_recognizer.py
        gestures.dynamic.dynamic_recognizer.py
        gestures.cursor.cursor_recognizer.py

    Consumers:
        gestures.decision.gesture_fusion.py
        commands.command_mapper.py
        cursor.action_mapper.py
        ui.status_panel.py
    """

    name: GestureName
    confidence: float
    source: RecognitionSource
    timestamp: float


@dataclass
class PointerPosition:
    """
    Normalized hand pointer position in camera coordinates.

    Usually produced from INDEX_MCP, INDEX_TIP, or palm center.

    Coordinate range:
        x: 0.0 - 1.0
        y: 0.0 - 1.0

    Producer:
        cursor.hand_pointer.py

    Consumers:
        cursor.screen_mapper.py
        cursor.cursor_smoothing.py
        ui.debug_panel.py
    """

    x: float
    y: float
    visible: bool
    confidence: float
    timestamp: float


@dataclass
class ScreenPosition:
    """
    Cursor position mapped to screen pixel coordinates.

    Producer:
        cursor.screen_mapper.py

    Consumers:
        cursor.mouse_controller.py
        ui.debug_panel.py
        simulator or 3D scene integration layer
    """

    x: int
    y: int
    timestamp: float


@dataclass
class CursorControlResult:
    """
    High-level result of cursor-control mode.

    This result combines:
        - screen cursor position
        - high-level cursor action
        - source gesture that caused the action, if any

    Example:
        INDEX_MCP -> CursorAction.MOVE_CURSOR
        INDEX_SQUEEZE -> CursorAction.GRAB
        INDEX_RELEASE -> CursorAction.RELEASE

    Producer:
        cursor.cursor_pipeline.py / cursor.action_mapper.py

    Consumers:
        ui.status_panel.py
        simulator layer
        optional mouse action translation layer
    """

    action: CursorAction
    screen_position: ScreenPosition | None
    source_gesture: GestureName
    confidence: float
    timestamp: float


@dataclass
class MouseControlResult:
    """
    Low-level mouse control result.

    This is used only after high-level CursorAction is translated into
    real OS-level mouse actions.

    Example:
        CursorAction.GRAB -> MouseAction.MOUSE_DOWN
        CursorAction.RELEASE -> MouseAction.MOUSE_UP

    Producer:
        cursor.mouse_controller.py or cursor action execution layer

    Consumers:
        ui.debug_panel.py
        logs
    """

    action: MouseAction
    screen_position: ScreenPosition | None
    executed: bool
    timestamp: float


@dataclass
class CommandResult:
    """
    High-level simulator command result.

    This is mainly for the future gesture-command mode, where gestures directly
    trigger simulator commands instead of going through cursor interaction.

    Producer:
        commands.command_mapper.py

    Consumers:
        simulator.simulator_controller.py
        ui.status_panel.py
        logs
    """

    command: CommandName
    source_gesture: GestureName
    confidence: float
    timestamp: float


@dataclass
class SimulatorStateSnapshot:
    """
    Current state snapshot of the simulator or mock simulator.

    Producer:
        simulator.simulator_state.py
        simulator.mock_simulator.py

    Consumers:
        ui.status_panel.py
        ui.debug_panel.py
        logs
    """

    selected_object: str | None
    valve_open: bool
    valve_rotation_degrees: float
    pump_running: bool
    emergency_stop: bool
    timestamp: float
