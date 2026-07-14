from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from oil_gestures.core.enums import CursorAction, GestureName, RecognitionSource
from oil_gestures.core.types import CursorControlResult, GestureResult, ScreenPosition


DEFAULT_CURSOR_MAPPING: dict[GestureName, CursorAction] = {
    GestureName.INDEX_MCP: CursorAction.MOVE_CURSOR,
    GestureName.INDEX_SQUEEZE: CursorAction.GRAB,
    GestureName.INDEX_RELEASE: CursorAction.RELEASE,
    GestureName.MIDDLE_PINCH: CursorAction.RIGHT_CLICK,
}


@dataclass(frozen=True)
class CursorActionMapperConfig:
    mapping: Mapping[GestureName, CursorAction] = field(
        default_factory=lambda: dict(DEFAULT_CURSOR_MAPPING)
    )


class CursorActionMapper:
    """Maps cursor-only gesture results to cursor actions."""

    def __init__(self, config: CursorActionMapperConfig | None = None) -> None:
        self.config = config or CursorActionMapperConfig()

    @classmethod
    def from_strings(cls, mapping: Mapping[str, str] | None = None) -> "CursorActionMapper":
        if mapping is None:
            return cls()
        parsed: dict[GestureName, CursorAction] = {}
        for gesture_name, action_name in mapping.items():
            gesture = GestureName(gesture_name)
            # Cursor mappings accept only cursor-channel gestures. A name that is
            # a valid GestureName but belongs to another channel (e.g. the
            # dynamic "SQUEEZE" vs the cursor "INDEX_SQUEEZE") is rejected here so
            # a mis-typed config fails loudly instead of silently mapping a
            # non-cursor gesture onto a cursor action.
            if gesture not in DEFAULT_CURSOR_MAPPING:
                accepted = ", ".join(sorted(g.value for g in DEFAULT_CURSOR_MAPPING))
                raise ValueError(
                    f"{gesture_name!r} is not a cursor gesture; "
                    f"cursor mappings accept only: {accepted}"
                )
            parsed[gesture] = CursorAction(action_name)
        return cls(CursorActionMapperConfig(parsed))

    def map(
        self,
        cursor_gesture: GestureResult | None = None,
        screen_position: ScreenPosition | None = None,
        timestamp: float | None = None,
    ) -> CursorControlResult:
        is_cursor_result = (
            cursor_gesture is not None
            and cursor_gesture.source == RecognitionSource.CURSOR_RULES
        )
        action = self.config.mapping.get(cursor_gesture.name) if is_cursor_result else None
        if timestamp is not None:
            result_timestamp = timestamp
        elif cursor_gesture is not None:
            result_timestamp = cursor_gesture.timestamp
        else:
            result_timestamp = 0.0
        return CursorControlResult(
            action=action or CursorAction.NONE,
            screen_position=screen_position,
            source_gesture=(
                cursor_gesture.name
                if is_cursor_result and action is not None
                else GestureName.UNKNOWN
            ),
            confidence=(
                cursor_gesture.confidence
                if is_cursor_result and action is not None
                else 0.0
            ),
            timestamp=result_timestamp,
        )
