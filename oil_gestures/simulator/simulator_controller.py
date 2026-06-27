from __future__ import annotations

from dataclasses import dataclass

from oil_gestures.commands.command_mapper import CommandMapper
from oil_gestures.core.enums import CommandName, GestureName, RecognitionSource
from oil_gestures.core.types import GestureResult

CONTRACT_NAME = "oil_gestures.ml.runtime"

# Source of truth for static-gesture -> command mapping: docs/command_mapping.md.
# Only gestures the static recognizer can actually produce today (canned
# MediaPipe categories) get a real command. OPEN_PALM has no assigned command
# in that doc (THUMB_UP replaced it for "open valve") - it is intentionally
# left unmapped rather than guessed.
STATIC_GESTURE_COMMANDS: dict[str, str] = {
    "FIST": "EMERGENCY_STOP",
    "THUMB_UP": "OPEN_VALVE",
}

# Dynamic-channel gestures from docs/command_mapping.md whose control is not
# implemented yet (no trained dynamic model - see gestures/dynamic, decision
# layer is empty). Shown to the operator as a labelled, honest stub instead of
# silently doing nothing. Note: scripts/mock_ml_events.py still emits some of
# these under older names (WRIST_ROTATE_CW/CCW, SPREAD, CLENCH) that don't
# match GestureName - that vocabulary mismatch is a separate, pre-existing
# issue and is not resolved by this stub table.
DYNAMIC_GESTURE_LABELS: dict[GestureName, str] = {
    GestureName.POINTING_INDEX: "👉 Указание (открыть меню объекта)",
    GestureName.SWIPE_LEFT: "👈 Свайп влево (переключение вентиля)",
    GestureName.SWIPE_RIGHT: "👉 Свайп вправо (переключение вентиля)",
    GestureName.ROTATE_CLOCKWISE: "🔄 По часовой (давление +)",
    GestureName.ROTATE_COUNTERCLOCKWISE: "🔄 Против часовой (давление -)",
}

NOT_IMPLEMENTED_SUFFIX = " - управление пока не реализовано"


@dataclass
class GestureEventResult:
    """What a single ML contract event means for the scene/UI."""

    message: str | None = None
    action_taken: bool = False
    emergency: bool = False
    # Current static gesture for a dedicated, always-accurate UI readout,
    # independent of the action/log message above.
    gesture_name: str | None = None
    gesture_confidence: float = 0.0


class SimulatorController:
    """
    Owns the "gesture -> scene action" mapping. The concrete static-gesture
    commands come from docs/command_mapping.md (the current source of truth);
    docs/interaction_spec.md still defines the channel/mode semantics around
    it. Takes an already-parsed oil_gestures.ml.runtime contract event and
    decides what happens to the scene model.

    Knows nothing about Qt/PySide or sockets, so it stays an autonomous
    consumer of the ML contract as required by docs/integration_contract.md -
    the UI layer owns the transport, this layer owns the interpretation.
    """

    def __init__(self, model, command_mapper: CommandMapper | None = None) -> None:
        self.model = model
        self._command_mapper = command_mapper or CommandMapper.from_strings(
            STATIC_GESTURE_COMMANDS
        )

    def handle_event(self, event: dict) -> GestureEventResult:
        if event.get("contract") != CONTRACT_NAME:
            return GestureEventResult()

        gestures = event.get("gestures") or {}
        cursor = event.get("cursor") or {}
        static = gestures.get("static")
        dynamic = gestures.get("dynamic")

        if isinstance(static, dict):
            result = self._apply_static_gesture(static)
        elif isinstance(dynamic, dict):
            result = self._apply_dynamic_gesture(dynamic)
        else:
            result = GestureEventResult()

        cursor_message = self._cursor_message(cursor)
        if cursor_message:
            result.message = (
                cursor_message if not result.message else f"{result.message}  |  {cursor_message}"
            )

        if result.message is None and not result.action_taken:
            hand = event.get("hand") or {}
            result.message = "👋 Рука есть" if hand.get("detected") else None

        return result

    def _cursor_message(self, cursor: dict) -> str | None:
        if not cursor.get("enabled"):
            return None
        action = cursor.get("action", "NONE")
        return f"🖐️ Курсор: {action}"

    def _apply_static_gesture(self, static: dict) -> GestureEventResult:
        name = static.get("name")
        confidence = static.get("confidence", 0) or 0.0
        result = GestureEventResult(gesture_name=name, gesture_confidence=confidence)

        try:
            gesture_result = GestureResult(
                name=GestureName(name),
                confidence=confidence,
                source=RecognitionSource.STATIC_RULES,
                timestamp=0.0,
            )
        except ValueError:
            result.message = self._format_generic(name, confidence)
            return result

        command = self._command_mapper.map(gesture_result).command
        detail = self.model.get_highlighted()

        if command == CommandName.EMERGENCY_STOP:
            result.emergency = True
            result.message = "✊ Аварийная остановка течения"
            return result

        if command == CommandName.OPEN_VALVE:
            # THUMB_UP - тумблер на наведённом вентиле: открывает закрытый,
            # закрывает открытый (по уточнению, не только "открыть").
            if detail and hasattr(detail, "open") and hasattr(detail, "close") and not detail.has_animation():
                if detail.is_open:
                    self.model.execute_action(detail, "close")
                    result.message = f"👍 Закрыть: {detail.name}"
                else:
                    self.model.execute_action(detail, "open")
                    result.message = f"👍 Открыть: {detail.name}"
                result.action_taken = True
            else:
                result.message = "👍 Большой палец вверх (нет цели)"
            return result

        if gesture_result.name == GestureName.VICTORY:
            result.message = "✌ Победа (курсор вкл/выкл)"
            return result

        # OPEN_PALM and anything else not in docs/command_mapping.md: show the
        # gesture, but do not guess a command for it.
        result.message = self._format_generic(name, confidence)
        return result

    def _apply_dynamic_gesture(self, dynamic: dict) -> GestureEventResult:
        name = dynamic.get("name")
        confidence = dynamic.get("confidence", 0) or 0.0

        try:
            label = DYNAMIC_GESTURE_LABELS.get(GestureName(name))
        except ValueError:
            label = None

        if label is None:
            return GestureEventResult()

        return GestureEventResult(message=f"{label}{NOT_IMPLEMENTED_SUFFIX}")

    def _format_generic(self, name: str | None, confidence) -> str:
        try:
            return f"✋ {name} ({float(confidence):.0%})"
        except (TypeError, ValueError):
            return f"✋ {name}"
