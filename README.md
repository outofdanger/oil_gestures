# Oil Gestures

Webcam-based gesture recognition for oil-simulator demos.

The core product is the gesture-recognition pipeline:

```text
Camera -> MediaPipe landmarks -> independent recognizers -> commands/features
```

MediaPipe is the single source of hand landmarks. Cursor control is an optional
secondary subsystem with its own recognizer; it does not consume static or
learned dynamic gesture results as mouse actions.

## Current Features

- Static gesture recognizer for `OPEN_PALM`, `FIST`, and `OK_SIGN`.
- Model-only dynamic-recognition interface, ready for a learned model.
- Isolated rule-based cursor recognizer for `INDEX_MCP`, `INDEX_SQUEEZE`,
  `INDEX_RELEASE`, and `MIDDLE_PINCH`.
- Optional cursor-control feature, initially disabled and manually testable with `--cursor-on`.
- Safe cursor dry-run by default: real OS mouse movement is disabled unless
  explicitly requested.
- Real cursor control on macOS, Windows, and Linux/X11.
- `scripts/check_camera.py` checks camera + MediaPipe landmarks only.
- `scripts/run_demo.py` runs the gesture-recognition demo and optional cursor feature.

## Project Layout

```text
app/
  app_config.py        Typed config dataclasses and YAML loading
  main.py              Runtime composition and OpenCV loop
oil_gestures/
  vision/              Camera, MediaPipe, drawing, landmark helpers
  gestures/            Independent static, dynamic-model, and cursor recognizers
  cursor/              Pointer mapping plus platform-neutral mouse facade/backends
  commands/            Gesture-to-command mapping
configs/               YAML runtime defaults
assets/models/         Local model files
scripts/               Thin executable scripts
tests/                 Unit tests for pure logic
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

macOS needs Camera permission for the app that runs Python. Accessibility
permission is needed only when real mouse control is intentionally enabled.

On Linux, camera capture prefers V4L2 and real cursor control uses X11/XTest.
Native Wayland sessions continue to support recognition and cursor dry-run, but
global mouse injection requires logging in with an X11/Xorg session.

## Run

Gesture recognition with the secondary cursor subsystem initially off:

```bash
python3 scripts/run_demo.py
```

Automatic activation by a learned dynamic gesture is intentionally left for a
later integration. `MIDDLE_PINCH` is recognized by the cursor subsystem but does
not toggle it.

Start with cursor feature on, still in safe dry-run mode:

```bash
python3 scripts/run_demo.py --cursor-on
```

Enable real mouse control intentionally:

```bash
python3 scripts/run_demo.py --cursor-on --real-mouse
```

Run mouse diagnostics:

```bash
python3 app/main.py --mouse-diagnostics
```

On Linux, verify the real X11 backend before starting the camera:

```bash
python3 app/main.py --mouse-diagnostics --real-mouse
```
