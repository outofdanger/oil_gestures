# ML Integration Contract v1

The ML runtime is an autonomous producer. UI and 3D applications are autonomous
consumers and must not import vision, MediaPipe, cursor, or application runtime
modules. Their only shared dependency is the versioned JSON contract in
`contracts/ml_events.v1.schema.json`.

## Run the producer

Metadata only, with the existing local preview:

```bash
python scripts/run_demo.py --event-server
```

Metadata plus camera frames, without a local OpenCV window:

```bash
python scripts/run_demo.py --event-server --publish-camera --headless
```

The default endpoint is `127.0.0.1:8765`. It is a UTF-8 TCP stream containing
one complete JSON object per line (NDJSON). Multiple consumers may connect and
disconnect independently. The ML loop never waits for a consumer; a slow
consumer may miss old snapshots and must use the latest `sequence` value.

The endpoint can be changed with `--event-host` and `--event-port`. Keep the
default loopback host unless remote access is intentionally required.

## Consume the stream

The dependency-free reference client is:

```bash
python scripts/consume_ml_events.py
```

Python applications can use the same reader directly:

```python
from oil_gestures.integration import RUNTIME_CONTRACT, iter_events

for event in iter_events("127.0.0.1", 8765):
    if event["contract"] == RUNTIME_CONTRACT:
        print(event["gestures"], event["hand"]["pointer"])
```

Other languages only need a TCP client, newline framing, JSON decoding, and the
published JSON Schema. No Python-specific objects are sent over the boundary.

## Runtime snapshots

`oil_gestures.ml.runtime` is emitted for every processed frame and contains:

- source frame dimensions and mirror flag;
- FPS and MediaPipe inference time;
- hand detection, handedness, confidence, and normalized pointer coordinates;
- independent static, dynamic, and cursor gesture results;
- cursor mode, pressed state, action, and optional screen position;
- pause state.

Gesture names and sources are strings, not a closed transport enum. Future
dynamic gestures can therefore be added without breaking v1 consumers. A
consumer must ignore gesture names it does not support.

`timestamp` is Unix time in seconds and is valid across processes. `stream_id`
changes each time the ML producer starts. `sequence` starts at 1 and increases
for every processed frame.

## Camera frames

`oil_gestures.ml.camera_frame` is optional and emitted only when
`--publish-camera` is set and at least one consumer is connected. Frames are
clean, mirrored preview frames without the local OpenCV overlay. They are JPEG
bytes encoded as base64. `stream_id` and `sequence` correlate a camera frame
with its runtime snapshot, so each UI can render its own overlays.

Camera publication defaults to 10 FPS and JPEG quality 80. Use
`--camera-publish-fps` and `--camera-jpeg-quality` to tune UI bandwidth without
changing ML inference frequency.

The camera-frame `width`/`height` are the preview size and may differ from
`frame` in the runtime snapshot, which reports the full camera resolution. Hand
`pointer` coordinates are normalized to `[0, 1]`, so overlays scale to whichever
frame a consumer renders.

## Current gesture availability

The contract exposes whatever the current recognizers produce. Today this is:

- static: `OPEN_PALM`, `FIST`, `THUMB_UP`, `VICTORY`;
- cursor: `INDEX_MCP`, `INDEX_SQUEEZE`, `INDEX_RELEASE`, `MIDDLE_PINCH`;
- dynamic: `null` until a learned dynamic model is connected.

Planned gestures such as pointing, OK, swipe, clench/release, and wrist rotation
can be added by the ML subsystem later without changing this transport or the UI
and 3D connection code.
