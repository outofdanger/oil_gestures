# Architecture

Oil Gestures is organized as a layered runtime. Each layer has one job and
communicates through dataclasses from `oil_gestures.core.types`.

## Runtime Flow

1. `app.app_config.load_config()` reads YAML config files into typed dataclasses.
2. `oil_gestures.vision.camera.CameraStream` opens OpenCV and produces raw `FramePacket`.
3. `oil_gestures.vision.frame_processor` mirrors/resizes frames and prepares RGB packets for MediaPipe.
4. `oil_gestures.vision.mediapipe_landmarker.MediaPipeHandLandmarker` produces `LandmarkPacket`.
5. `gestures.static.StaticGestureRecognizer` produces independent static results.
6. `gestures.dynamic.DynamicGestureRecognizer` accepts results only from a learned model.
7. When cursor control is enabled, `gestures.cursor.CursorGestureRecognizer` produces cursor-only results.
8. `cursor.CursorPipeline` consumes only cursor results and translates them to mouse actions.
9. `oil_gestures.vision.drawing` renders all subsystem results in the same OpenCV window.

## Cursor Feature Contract

Cursor control is a secondary feature, not the main recognition path. It shares
landmarks and the OpenCV window with static/dynamic recognition, but it owns its
recognizer, configuration, mappings, and OS interaction.

Default cursor state:

- initially disabled;
- dry-run enabled;
- manual activation through config or `--cursor-on` for now;
- real OS mouse control only with `--real-mouse` or config override.

Cursor action mappings:

- `INDEX_MCP` -> `MOVE_CURSOR`
- `INDEX_SQUEEZE` -> `GRAB`
- `INDEX_RELEASE` -> `RELEASE`
- `MIDDLE_PINCH` -> no action yet

Static and learned dynamic gesture results never enter `CursorActionMapper`.
Future activation of cursor control by a learned dynamic gesture belongs to the
runtime orchestration layer and is not implemented yet.

## Platform Backends

```text
MouseController ──> PlatformBackend ──> MouseBackend
ScreenMapper ─────> PlatformBackend ──> DesktopBounds
                            │
                Windows / macOS / Linux / dry-run
```

The factory selects one platform adapter. Application-facing modules depend on
the contracts from `cursor/backends/base.py`, not on native APIs.

## Module Boundaries

- Camera code stays in `vision/camera.py`.
- Frame resizing, mirroring, and BGR/RGB conversion stay in `vision/frame_processor.py`.
- MediaPipe hand detection stays in `vision/mediapipe_landmarker.py`.
- Reusable landmark math stays in `vision/landmark_utils.py`.
- Static, learned dynamic, and cursor gesture recognition live in separate subpackages of `gestures/`.
- Cursor pointer, mapping, smoothing, action mapping, and mouse execution stay in `cursor/`.
- Platform-specific cursor integration stays behind common contracts in `cursor/backends/base.py`.
  Windows, macOS, Linux/X11, and dry-run implementations live in sibling backend modules; their
  native dependencies are loaded only when the matching platform needs them.
- `app/main.py` composes modules and owns no gesture algorithm.
