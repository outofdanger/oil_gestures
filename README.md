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
- Real cursor control on macOS, Windows, Linux/X11 (XTEST), and Linux/Wayland
  (via a `/dev/uinput` virtual pointer).
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

The supported runtime is standard **CPython 3.14.x**; the repository pins
CPython 3.14.6 in `.python-version`. The free-threaded `3.14t` build is not part
of the supported runtime yet.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.14 -m venv .venv` and activate
it with `.venv\Scripts\activate`.

macOS needs Camera permission for the app that runs Python. Accessibility
permission is needed only when real mouse control is intentionally enabled.

On Linux, camera capture prefers V4L2. Real cursor control uses the X11 XTEST
extension on X11/Xorg sessions, and a `/dev/uinput` virtual pointer device on
native Wayland sessions (XTEST only affects XWayland clients, not the
compositor, so it cannot move the real cursor under Wayland). The uinput path
needs read/write access to `/dev/uinput`: either run as a member of the
`input` group (`sudo usermod -aG input $USER`, then log out and back in) or
add a udev rule such as `KERNEL=="uinput", GROUP="input", MODE="0660"` in
`/etc/udev/rules.d/`, plus the kernel module loaded (`sudo modprobe uinput`).
Without that access, `--real-mouse` raises a clear error instead of silently
falling back to dry-run.

The prebuilt dependency set supports macOS on Apple Silicon, Windows x86-64,
and Linux x86-64 with glibc 2.28 or newer. MediaPipe/OpenCV do not currently
provide the complete wheel set for macOS Intel, Linux ARM64, or Windows ARM64.

## Run

Gesture recognition with the secondary cursor subsystem initially off:

```bash
python scripts/run_demo.py
```

Automatic activation by a learned dynamic gesture is intentionally left for a
later integration. `MIDDLE_PINCH` performs a right click and does not toggle
cursor control. Holding `INDEX_SQUEEZE`, moving the hand, and then producing
`INDEX_RELEASE` performs drag-and-drop.

Start with cursor feature on, still in safe dry-run mode:

```bash
python scripts/run_demo.py --cursor-on
```

Enable real mouse control intentionally:

```bash
python scripts/run_demo.py --cursor-on --real-mouse
```

Run mouse diagnostics:

```bash
python app/main.py --mouse-diagnostics
```

On Linux, verify the real X11 backend before starting the camera:

```bash
python app/main.py --mouse-diagnostics --real-mouse
```
