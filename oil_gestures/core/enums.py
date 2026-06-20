from enum import Enum

class GestureName(str, Enum):
    """
    Shared gesture names used by static and dynamic recognition layers.
    Do not use raw gesture strings in implementation modules.
    """

    UNKNOWN = "UNKNOWN"
    IDLE = "IDLE"

    # Static gestures
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
    OK_SIGN = "OK_SIGN"

    # Dynamic gestures
    SQUEEZE = "SQUEEZE"
    RELEASE = "RELEASE"
    MIDDLE_PINCH = "MIDDLE_PINCH"
    ROTATE_CLOCKWISE = "ROTATE_CLOCKWISE"
    ROTATE_COUNTERCLOCKWISE = "ROTATE_COUNTERCLOCKWISE"
    POINTING_INDEX = "POINTING_INDEX"


class RecognitionSource(str, Enum):
    """
    Shows which subsystem produced a gesture result.
    """

    STATIC_RULES = "STATIC_RULES"
    DYNAMIC_RULES = "DYNAMIC_RULES"
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
    High-level cursor-mode actions produced from hand tracking,
    static gestures, and dynamic gestures.
    """
    NONE = "NONE"
    MOVE_CURSOR = "MOVE_CURSOR"
    SELECT = "SELECT"
    RIGHT_CLICK = "RIGHT_CLICK"
    GRAB = "GRAB"
    RELEASE = "RELEASE"
    DRAG = "DRAG"
    INCREASE_PRESSURE = "INCREASE_PRESSURE"
    DECREASE_PRESSURE = "DECREASE_PRESSURE"


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
