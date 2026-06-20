# Gesture List

## Recognized Gestures

| Gesture | Source | Current purpose |
| --- | --- | --- |
| `OPEN_PALM` | static rules | release fallback / future command |
| `FIST` | static rules | grab fallback / future command |
| `OK_SIGN` | static rules | select fallback |
| `POINTING_INDEX` | dynamic rules | cursor movement when cursor feature is enabled |
| `SQUEEZE` | dynamic rules | grab / mouse down |
| `RELEASE` | dynamic rules | release / mouse up |
| `MIDDLE_PINCH` | dynamic rules | toggle cursor-control feature |
| `ROTATE_CLOCKWISE` | dynamic rules | increase pressure |
| `ROTATE_COUNTERCLOCKWISE` | dynamic rules | decrease pressure |

## Optional Cursor Control

Cursor position is calculated from the configured pointer landmark. The default
pointer source is `INDEX_MCP` / point `5`.

Cursor actions come from recognized gestures only while the cursor feature is
enabled:

| Gesture | Cursor action |
| --- | --- |
| `POINTING_INDEX` | `MOVE_CURSOR` |
| `SQUEEZE` | `GRAB` |
| `RELEASE` | `RELEASE` |
| `ROTATE_CLOCKWISE` | `INCREASE_PRESSURE` |
| `ROTATE_COUNTERCLOCKWISE` | `DECREASE_PRESSURE` |
| `OK_SIGN` | `SELECT` fallback |
| `FIST` | `GRAB` fallback |
| `OPEN_PALM` | `RELEASE` fallback |

`DRAG` is reserved in core contracts for future work and is intentionally not
implemented in the current cursor feature.
