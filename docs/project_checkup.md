# Project Checkup

## Architecture Status

- MediaPipe is the single source of hand landmarks.
- `gestures/static/` owns general static recognition.
- `gestures/dynamic/` is model-only and contains no cursor rules.
- `gestures/cursor/` owns the four rule-based cursor gestures.
- `cursor/` owns pointer mapping, smoothing, actions, and OS mouse backends.
- `app/main.py` runs the independent subsystems in one OpenCV loop.

## Cursor Contract

- `INDEX_MCP` -> `MOVE_CURSOR`
- `INDEX_SQUEEZE` -> `GRAB`
- `INDEX_RELEASE` -> `RELEASE`
- `MIDDLE_PINCH` -> `RIGHT_CLICK`.
- Movement between `INDEX_SQUEEZE` and `INDEX_RELEASE` -> `DRAG`.
- Static gestures are not cursor fallbacks.
- Real mouse control and cursor activation are disabled by default.

Automatic cursor activation after a learned dynamic gesture remains a future
runtime integration. Until then, `--cursor-on` is available for manual testing.

## Manual Checks

- Confirm `python --version` reports CPython 3.14.x (validated baseline: 3.14.6).
- Run `python -m pip check` and `python -m pytest -q` inside the project virtual environment.
- Run `python scripts/check_camera.py` on a machine with webcam permission.
- Run `python scripts/run_demo.py --cursor-on` to inspect cursor recognition in dry-run mode.
- Add `--real-mouse` only when real OS mouse control is intentionally needed.
