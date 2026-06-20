# Project Checkup

Source checked: https://github.com/outofdanger/oil_gestures/issues/2

## Architecture Status

- `vision/camera.py` owns webcam open/read/release and returns `FramePacket`.
- `vision/frame_processor.py` owns resize, mirror, and BGR/RGB conversion.
- `vision/mediapipe_landmarker.py` owns MediaPipe hand landmark detection and returns `LandmarkPacket`.
- `vision/landmark_utils.py` owns reusable landmark math.
- `vision/drawing.py` owns visualization helpers only.
- `gestures/static/` owns static gesture recognition.
- `gestures/dynamic/` owns dynamic gesture recognition.
- `cursor/` owns optional cursor-control feature modules.
- `app/main.py` composes the runtime and does not own gesture algorithms.

## Issue #2 Requirements

- MediaPipe remains the single source of hand landmarks.
- Static and dynamic gesture layers are separate from cursor control.
- Cursor control consumes `LandmarkPacket` plus recognized `GestureResult` values.
- Required cursor mappings are present:
  - `POINTING_INDEX` -> `MOVE_CURSOR`
  - `SQUEEZE` -> `GRAB`
  - `RELEASE` -> `RELEASE`
  - `ROTATE_CLOCKWISE` -> `INCREASE_PRESSURE`
  - `ROTATE_COUNTERCLOCKWISE` -> `DECREASE_PRESSURE`
  - static fallbacks: `OK_SIGN`, `FIST`, `OPEN_PALM`
- Real mouse control is disabled by default through `dry_run: true`.
- Cursor control is initially disabled by default and toggled by `MIDDLE_PINCH`.
- `DRAG` remains only as a future enum contract and is not mapped or executed.

## Extra Above Issue #2

- Added a cursor feature toggle so hand tracking/control is not the main runtime.
- Added lightweight static gesture rules for `OPEN_PALM`, `FIST`, `OK_SIGN`.
- Moved the prototype pinch behavior into `RuleBasedDynamicRecognizer`.
- Added tests for safe defaults, cursor toggle, disabled cursor pipeline, and drag rejection.

## Manual Checks Still Needed

- Run `python3 scripts/check_camera.py` on a machine with webcam permission.
- Run `python3 scripts/run_demo.py` and toggle cursor control with `MIDDLE_PINCH`.
- Run `python3 scripts/run_demo.py --cursor-on --real-mouse` only when real OS mouse control is intentionally needed.
