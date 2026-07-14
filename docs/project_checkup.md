# Project Checkup

## Architecture Status

- MediaPipe is the single source of hand landmarks.
- `gestures/static/` owns static recognition (canned MediaPipe categories).
- `gestures/dynamic/` runs a trained ST-GCN + BiLSTM ensemble (no cursor rules);
  loaded by `gestures.dynamic.model_loader`.
- `gestures/cursor/` owns the four rule-based cursor gestures.
- `cursor/` owns pointer mapping, smoothing, actions, and OS mouse backends.
- `app/main.py` is the ML producer: runs the subsystems in one OpenCV loop and
  publishes the versioned NDJSON/TCP contract (`integration/`).
- `oil_gestures/ui/` + `oil_gestures/simulator/` are the autonomous UI/3D
  consumer (PySide6 + PyVista). `SimulatorController` (Qt-free) maps gestures →
  scene actions per `docs/command_mapping.md`; the `Controller` performs them.
- ML and UI never import each other - only the contract joins them.

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
