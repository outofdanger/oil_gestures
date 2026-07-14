# Gesture List

Recognition subsystems are independent channels: **static** (canned MediaPipe
classifier), **dynamic** (learned ST-GCN + BiLSTM ensemble), and **cursor**
(rule-based). The ML runtime only recognizes gestures and publishes them on the
versioned contract; the meaning of each gesture (gesture → scene action) lives
in the UI/3D consumer — see [`command_mapping.md`](command_mapping.md).

## Static Recognition (`gestures/static`)

Maps MediaPipe's built-in categories to `GestureName`:

| Gesture | MediaPipe category | Scene meaning (command_mapping.md) |
| --- | --- | --- |
| `OPEN_PALM` | `Open_Palm` | none (unmapped) |
| `FIST` | `Closed_Fist` | none (emergency stop moved to menu / controller) |
| `THUMB_UP` | `Thumb_Up` | activate the selected detail |
| `VICTORY` | `Victory` | toggle cursor mode |

## Dynamic Recognition (`gestures/dynamic`)

Learned ensemble (ST-GCN leads, BiLSTM confirms). Checkpoints and training live
in `dynamic_gestures/`; the runtime loader is `gestures.dynamic.model_loader`.

| Gesture | Scene meaning |
| --- | --- |
| `POINTING_INDEX` | toggle the selected detail's context menu |
| `SWIPE_LEFT` / `SWIPE_RIGHT` | cycle selection forward / back (wraps) |
| `ROTATE_CLOCKWISE` / `ROTATE_COUNTERCLOCKWISE` | open / close the selected valve's % (continuous) |
| `SQUEEZE` / `RELEASE` | zoom into the selected assembly / return to main view |

`IDLE` is the trained "no intentional motion" baseline (never acted on).
`TRANSITION` (9th class in the `_transition` checkpoints) captures the hand's
return stroke and is dropped by the runtime as "no gesture". Opposite-direction
pairs are additionally guarded by a rest-gated directional lockout — see
`gestures/dynamic/model_loader.py`.

## Cursor Recognition (`gestures/cursor`)

Isolated rule-based recognizer; active only in cursor mode (toggled by VICTORY).
Static/dynamic scene gestures are muted while cursor mode is on.

| Gesture | Cursor action |
| --- | --- |
| `INDEX_MCP` | `MOVE_CURSOR` |
| `INDEX_SQUEEZE` | `GRAB` / mouse down |
| `INDEX_RELEASE` | `RELEASE` / mouse up |
| `MIDDLE_PINCH` | `RIGHT_CLICK` |

`INDEX_MCP` also names the default MediaPipe landmark used as the cursor point
(landmark `5`). Moving while `INDEX_SQUEEZE` is held emits drag events until
`INDEX_RELEASE`.
