"""
Project-wide default constants.

If a value should be configurable by the user, it should also be duplicated
or overridden in configs/*.yaml.
"""

# Camera defaults
DEFAULT_CAMERA_ID = 0
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_MIRROR_FRAME = True

# MediaPipe / hand landmarks
DEFAULT_LANDMARK_COUNT = 21
DEFAULT_LANDMARK_DIMENSIONS = 3

# Gesture recognition thresholds
DEFAULT_STATIC_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_DYNAMIC_CONFIDENCE_THRESHOLD = 0.75

# Static gesture smoothing
DEFAULT_STATIC_SMOOTHING_WINDOW = 5

# Dynamic gesture sequence
DEFAULT_SEQUENCE_LENGTH = 40

# Cursor control
DEFAULT_POINTER_LANDMARK = "INDEX_TIP"
DEFAULT_CURSOR_SMOOTHING_ALPHA = 0.35
DEFAULT_CURSOR_DRY_RUN = True

# Mouse / cursor action safety
DEFAULT_MOUSE_ACTION_COOLDOWN_SECONDS = 0.50

# Runtime
DEFAULT_DEBUG_MODE = True
DEFAULT_SAFE_EXIT_KEY = "q"

# Logging
DEFAULT_LOGGER_NAME = "oil_gestures"