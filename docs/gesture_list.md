# Gesture List

Recognition subsystems are independent. Static and learned dynamic results are
not used as cursor actions.

## Static Recognition

| Gesture | Source | Purpose |
| --- | --- | --- |
| `OPEN_PALM` | static rules | general recognition / future command |
| `FIST` | static rules | general recognition / future command |
| `OK_SIGN` | static rules | general recognition / future command |

## Dynamic Recognition

`gestures/dynamic` is reserved for learned temporal models. It has no rule-based
cursor fallback and currently produces no result until a model is connected.

## Cursor Recognition

The isolated rule-based recognizer lives in `gestures/cursor`:

| Gesture | Cursor action |
| --- | --- |
| `INDEX_MCP` | `MOVE_CURSOR` |
| `INDEX_SQUEEZE` | `GRAB` / mouse down |
| `INDEX_RELEASE` | `RELEASE` / mouse up |
| `MIDDLE_PINCH` | `RIGHT_CLICK` |

`INDEX_MCP` also names the default MediaPipe landmark used as the cursor point
(landmark `5`). Moving while `INDEX_SQUEEZE` is held emits drag events until
`INDEX_RELEASE`. `MIDDLE_PINCH` does not toggle cursor control.
