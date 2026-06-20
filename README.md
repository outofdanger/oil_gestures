# Oil Gestures

Webcam-based gesture recognition for oil-simulator demos.

The core product is the gesture-recognition pipeline:

```text
Camera -> MediaPipe landmarks -> static gestures -> dynamic gestures -> commands/features
```

MediaPipe is the single source of hand landmarks. Cursor control is an optional
feature on top of recognized gestures; it is not the main runtime mode.

## Current Features

- Static gesture recognizer for `OPEN_PALM`, `FIST`, and `OK_SIGN`.
- Rule-based dynamic recognizer for `POINTING_INDEX`, `SQUEEZE`, `RELEASE`,
  `MIDDLE_PINCH`, `ROTATE_CLOCKWISE`, and `ROTATE_COUNTERCLOCKWISE`.
- Optional cursor-control feature that can be toggled by gesture.
- Safe cursor dry-run by default: real OS mouse movement is disabled unless
  explicitly requested.
- `scripts/check_camera.py` checks camera + MediaPipe landmarks only.
- `scripts/run_demo.py` runs the gesture-recognition demo and optional cursor feature.

## Project Layout

```text
app/
  app_config.py        Typed config dataclasses and YAML loading
  main.py              Runtime composition and OpenCV loop
oil_gestures/
  vision/              Camera, MediaPipe, drawing, landmark helpers
  gestures/            Static/dynamic gesture recognizers and decision helpers
  cursor/              Optional pointer extraction, screen mapping, mouse actions
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

## Run

Gesture recognition with cursor feature initially off:

```bash
python3 scripts/run_demo.py
```

Use the configured toggle gesture, default `MIDDLE_PINCH`, to turn cursor control
on or off while the demo is running.

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
