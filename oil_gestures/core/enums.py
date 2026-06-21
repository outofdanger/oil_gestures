from enum import Enum

class GestureName(str, Enum):
    """
    Shared gesture names used by independent recognition subsystems.
    Do not use raw gesture strings in implementation modules.
    """

    UNKNOWN = "UNKNOWN"
    IDLE = "IDLE"

    # Static gestures (MediaPipe canned gestures)
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
    THUMB_UP = "THUMB_UP"
    VICTORY = "VICTORY"

    # Cursor gestures
    INDEX_MCP = "INDEX_MCP"
    INDEX_SQUEEZE = "INDEX_SQUEEZE"
    INDEX_RELEASE = "INDEX_RELEASE"
    MIDDLE_PINCH = "MIDDLE_PINCH"


class RecognitionSource(str, Enum):
    """
    Shows which subsystem produced a gesture result.
    """

    STATIC_RULES = "STATIC_RULES"
    CURSOR_RULES = "CURSOR_RULES"
    DYNAMIC_MODEL = "DYNAMIC_MODEL"
    MEDIAPIPE = "MEDIAPIPE"
    FALLBACK = "FALLBACK"


class Handedness(str, Enum):
    """
    Hand side detected by the vision layer.
    """

    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UNKNOWN = "UNKNOWN"


class MouseAction(str, Enum):
    """
    Mouse action in cursor-control mode
    """
    NONE = "NONE"
    MOVE = "MOVE"
    LEFT_CLICK = "LEFT_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    MOUSE_DOWN = "MOUSE_DOWN"
    MOUSE_UP = "MOUSE_UP"
    DRAG = "DRAG"


class CursorAction(str, Enum):
    """
    High-level actions produced only by the cursor gesture subsystem.
    """
    NONE = "NONE"
    MOVE_CURSOR = "MOVE_CURSOR"
    RIGHT_CLICK = "RIGHT_CLICK"
    GRAB = "GRAB"
    RELEASE = "RELEASE"


class InteractionMode(str, Enum):
    """
    High-level interaction modes of the system.
    """

    CURSOR_CONTROL = "CURSOR_CONTROL"
    GESTURE_COMMAND = "GESTURE_COMMAND"
    DEMO = "DEMO"


class CommandName(str, Enum):
    """
    High-level simulator commands for future direct gesture-command mode.
    Cursor mode may still indirectly trigger these through the 3D interface.
    """

    NONE = "NONE"
    SELECT_OBJECT = "SELECT_OBJECT"
    GRAB_OBJECT = "GRAB_OBJECT"
    RELEASE_OBJECT = "RELEASE_OBJECT"
    INCREASE_WELL_PRESSURE = "INCREASE_WELL_PRESSURE"
    DECREASE_WELL_PRESSURE = "DECREASE_WELL_PRESSURE"
    OPEN_VALVE = "OPEN_VALVE"
    CLOSE_VALVE = "CLOSE_VALVE"
    START_PUMP = "START_PUMP"
    STOP_PUMP = "STOP_PUMP"
    EMERGENCY_STOP = "EMERGENCY_STOP"
