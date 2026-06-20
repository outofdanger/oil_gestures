from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from oil_gestures.core.enums import CursorAction, GestureName, RecognitionSource
from oil_gestures.core.types import CursorControlResult, GestureResult, ScreenPosition


DEFAULT_DYNAMIC_MAPPING: dict[GestureName, CursorAction] = {
    GestureName.POINTING_INDEX: CursorAction.MOVE_CURSOR,
    GestureName.SQUEEZE: CursorAction.GRAB,
    GestureName.RELEASE: CursorAction.RELEASE,
    GestureName.ROTATE_CLOCKWISE: CursorAction.INCREASE_PRESSURE,
    GestureName.ROTATE_COUNTERCLOCKWISE: CursorAction.DECREASE_PRESSURE,
}

DEFAULT_STATIC_FALLBACK_MAPPING: dict[GestureName, CursorAction] = {
    GestureName.OK_SIGN: CursorAction.SELECT,
    GestureName.FIST: CursorAction.GRAB,
    GestureName.OPEN_PALM: CursorAction.RELEASE,
}

DYNAMIC_SOURCES = {RecognitionSource.DYNAMIC_RULES, RecognitionSource.DYNAMIC_MODEL}
STATIC_SOURCES = {RecognitionSource.STATIC_RULES}


@dataclass(frozen=True)
class CursorActionMapperConfig:
    dynamic_mapping: Mapping[GestureName, CursorAction] = field(
        default_factory=lambda: dict(DEFAULT_DYNAMIC_MAPPING)
    )
    static_fallback_mapping: Mapping[GestureName, CursorAction] = field(
        default_factory=lambda: dict(DEFAULT_STATIC_FALLBACK_MAPPING)
    )


class CursorActionMapper:
    def __init__(self, config: CursorActionMapperConfig | None = None) -> None:
        self.config = config or CursorActionMapperConfig()

    @classmethod
    def from_strings(
        cls,
        dynamic_mapping: Mapping[str, str] | None = None,
        static_fallback_mapping: Mapping[str, str] | None = None,
    ) -> "CursorActionMapper":
        parsed_dynamic = cls._parse_mapping(dynamic_mapping, DEFAULT_DYNAMIC_MAPPING)
        parsed_static = cls._parse_mapping(static_fallback_mapping, DEFAULT_STATIC_FALLBACK_MAPPING)
        return cls(CursorActionMapperConfig(parsed_dynamic, parsed_static))

    @staticmethod
    def _parse_mapping(
        mapping: Mapping[str, str] | None,
        default: Mapping[GestureName, CursorAction],
    ) -> dict[GestureName, CursorAction]:
        if mapping is None:
            return dict(default)
        parsed: dict[GestureName, CursorAction] = {}
        for gesture, action in mapping.items():
            parsed_action = CursorAction(action)
            if parsed_action == CursorAction.DRAG:
                raise ValueError("CursorAction.DRAG is reserved for a future issue and cannot be mapped yet.")
            parsed[GestureName(gesture)] = parsed_action
        return parsed

    @staticmethod
    def _result_timestamp(
        dynamic_result: GestureResult | None,
        static_result: GestureResult | None,
        timestamp: float | None,
    ) -> float:
        if timestamp is not None:
            return timestamp
        if dynamic_result is not None:
            return dynamic_result.timestamp
        if static_result is not None:
            return static_result.timestamp
        return 0.0

    def map(
        self,
        dynamic_result: GestureResult | None = None,
        static_result: GestureResult | None = None,
        screen_position: ScreenPosition | None = None,
        timestamp: float | None = None,
    ) -> CursorControlResult:
        source = dynamic_result
        action = self.config.dynamic_mapping.get(dynamic_result.name) if dynamic_result is not None else None

        if action is None and static_result is not None:
            source = static_result
            action = self.config.static_fallback_mapping.get(static_result.name)

        return CursorControlResult(
            action=action or CursorAction.NONE,
            screen_position=screen_position,
            source_gesture=source.name if source is not None and action is not None else GestureName.UNKNOWN,
            confidence=source.confidence if source is not None and action is not None else 0.0,
            timestamp=self._result_timestamp(dynamic_result, static_result, timestamp),
        )

    def map_sequence(
        self,
        results: Iterable[GestureResult],
        screen_position: ScreenPosition | None = None,
        timestamp: float | None = None,
    ) -> CursorControlResult:
        dynamic_result: GestureResult | None = None
        static_result: GestureResult | None = None
        for result in results:
            if result.source in DYNAMIC_SOURCES:
                if dynamic_result is None or result.confidence > dynamic_result.confidence:
                    dynamic_result = result
            elif result.source in STATIC_SOURCES:
                if static_result is None or result.confidence > static_result.confidence:
                    static_result = result
        return self.map(dynamic_result, static_result, screen_position, timestamp)
