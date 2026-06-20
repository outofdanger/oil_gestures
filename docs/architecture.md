# Architecture

Oil Gestures is organized as a layered runtime. Each layer has one job and
communicates through dataclasses from `oil_gestures.core.types`.

## Runtime Flow

1. `app.app_config.load_config()` reads YAML config files into typed dataclasses.
2. `oil_gestures.vision.camera.CameraStream` opens OpenCV and produces raw `FramePacket`.
3. `oil_gestures.vision.frame_processor` mirrors/resizes frames and prepares RGB packets for MediaPipe.
4. `oil_gestures.vision.mediapipe_landmarker.MediaPipeHandLandmarker` produces `LandmarkPacket`.
5. `oil_gestures.gestures.static.StaticGestureRecognizer` produces static `GestureResult` values.
6. `oil_gestures.gestures.dynamic.RuleBasedDynamicRecognizer` produces dynamic `GestureResult` values.
7. Gesture results can be mapped to simulator commands through `oil_gestures.commands`.
8. Gesture results can also toggle optional features such as cursor control.
9. When cursor control is enabled, `oil_gestures.cursor.CursorPipeline` consumes landmarks and gestures.
10. `oil_gestures.vision.drawing` renders debug landmarks, gestures, and feature state.

## Cursor Feature Contract

Cursor control is a feature, not the main product path. Gesture recognition keeps
running whether cursor control is enabled or disabled.

Default cursor state:

- initially disabled;
- dry-run enabled;
- toggle gesture: `MIDDLE_PINCH`;
- real OS mouse control only with `--real-mouse` or config override.

Cursor action mappings:

- `POINTING_INDEX` -> `MOVE_CURSOR`
- `SQUEEZE` -> `GRAB`
- `RELEASE` -> `RELEASE`
- `ROTATE_CLOCKWISE` -> `INCREASE_PRESSURE`
- `ROTATE_COUNTERCLOCKWISE` -> `DECREASE_PRESSURE`
- `OK_SIGN` -> `SELECT` fallback
- `FIST` -> `GRAB` fallback
- `OPEN_PALM` -> `RELEASE` fallback

`DRAG` remains in the core enum only as a future contract. It is not mapped,
executed, or tested as working behavior in the current issue.

## Module Boundaries

- Camera code stays in `vision/camera.py`.
- Frame resizing, mirroring, and BGR/RGB conversion stay in `vision/frame_processor.py`.
- MediaPipe hand detection stays in `vision/mediapipe_landmarker.py`.
- Reusable landmark math stays in `vision/landmark_utils.py`.
- Static and dynamic gesture recognition stay in `gestures/`.
- Cursor pointer, mapping, smoothing, action mapping, and mouse execution stay in `cursor/`.
- `app/main.py` composes modules and owns no gesture algorithm.
