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

- Static gesture recognizer for `OPEN_PALM`, `FIST`, `THUMB_UP`, `VICTORY`
  (MediaPipe canned classifier).
- Learned dynamic-gesture ensemble (ST-GCN + BiLSTM) for `POINTING_INDEX`,
  `SWIPE_LEFT/RIGHT`, `ROTATE_CW/CCW`, `SQUEEZE`, `RELEASE`; runs on CUDA / MPS
  when available. See [`dynamic_gestures/README.md`](dynamic_gestures/README.md).
- PySide6 + PyVista **3D oil-rig simulator UI** driven by gestures over the
  contract (valves, manometers, controller, level gauge). See
  [`docs/command_mapping.md`](docs/command_mapping.md) for the gesture → action map.
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
  ui_main.py           3D simulator UI entry point (contract consumer)
  run_ui.py            Launches ML producer + UI together
oil_gestures/
  vision/              Camera, MediaPipe, drawing, landmark helpers
  gestures/            Independent static, dynamic-ensemble, and cursor recognizers
  cursor/              Pointer mapping plus platform-neutral mouse facade/backends
  commands/            Gesture-to-command mapping helpers
  integration/         Versioned NDJSON/TCP contract (producer/consumer boundary)
  simulator/           PyVista 3D scene, model, render scheduler/profile
  ui/                  PySide6 window, control panel, gesture->scene controller
configs/               YAML runtime defaults
assets/models/         MediaPipe .task + trained PyTorch .pt checkpoints
dynamic_gestures/      Offline dynamic-gesture training pipeline + datasets
scripts/               Thin executable scripts
tests/                 Unit tests for pure logic
```

## Setup

The supported runtime is standard **CPython 3.12.x**; the repository pins
CPython 3.12.3 in `.python-version`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate
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

For real-time processing, Linux/V4L2 requests MJPG at 1280x720@30 by default,
while MediaPipe receives a separate 640x360 inference frame. The terminal logs
the actual backend, resolution, FPS, and FOURCC accepted by the camera driver.

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

## 3D Simulator UI

Launch the ML producer and the PySide6 + PyVista 3D UI together (they talk only
over the contract; the launcher wires GPU/render env and starts both):

```bash
python app/run_ui.py
```

The UI is an autonomous consumer: it can also run against the fake producer
(`python scripts/mock_ml_events.py`) with no camera. Gesture → scene actions are
documented in [`docs/command_mapping.md`](docs/command_mapping.md).

## Autonomous integration

Run ML as an independent producer of versioned JSON contracts:

```bash
python scripts/run_demo.py --event-server --publish-camera --headless
```

UI and 3D applications connect independently to `127.0.0.1:8765`. The
language-neutral schema, transport rules, and consumer example are documented
in [`docs/integration_contract.md`](docs/integration_contract.md).

On Linux, verify the real X11 backend before starting the camera:

```bash
python app/main.py --mouse-diagnostics --real-mouse
```
