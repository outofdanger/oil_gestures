from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from oil_gestures.core.enums import CursorAction, GestureName, RecognitionSource
from oil_gestures.core.types import CursorControlResult, GestureResult, ScreenPosition


DEFAULT_CURSOR_MAPPING: dict[GestureName, CursorAction] = {
    GestureName.INDEX_MCP: CursorAction.MOVE_CURSOR,
    GestureName.INDEX_SQUEEZE: CursorAction.GRAB,
    GestureName.INDEX_RELEASE: CursorAction.RELEASE,
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
        parsed = {
            GestureName(gesture_name): CursorAction(action_name)
            for gesture_name, action_name in mapping.items()
        }
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
