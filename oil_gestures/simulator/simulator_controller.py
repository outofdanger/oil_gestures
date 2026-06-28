from __future__ import annotations

from dataclasses import dataclass

from oil_gestures.commands.command_dispatcher import CommandDispatcher
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

# Dynamic-channel gestures from docs/command_mapping.md that are still pure
# display stubs - no trained signal drives them yet. POINTING_INDEX and
# SWIPE_LEFT/RIGHT are handled separately below (real behaviour, not a stub).
NOT_IMPLEMENTED_LABELS: dict[GestureName, str] = {
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
    # POINTING_INDEX: ask the Qt layer to open the same context menu right-click
    # would (SimulatorController has no Qt access of its own).
    open_menu: bool = False
    # THUMB_UP executed an armed menu action: ask the Qt layer to close the
    # context menu it opened for open_menu above.
    close_menu: bool = False
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

    Menu flow (POINTING_INDEX / THUMB_UP): in cursor-off gesture mode there is
    no mouse to click a context menu, so THUMB_UP only acts on a detail once
    POINTING_INDEX has explicitly "armed" it (opened its menu) - mirrors a
    right-click followed by picking the only available action. ``_armed_detail``
    tracks this; it resets whenever the selected detail changes (swipe) or once
    THUMB_UP consumes it.
    """

    def __init__(
        self,
        model,
        command_mapper: CommandMapper | None = None,
        dispatcher: CommandDispatcher | None = None,
    ) -> None:
        self.model = model
        self._command_mapper = command_mapper or CommandMapper.from_strings(
            STATIC_GESTURE_COMMANDS
        )
        self._dispatcher = dispatcher or CommandDispatcher()
        self._armed_detail = None

    def clear_armed(self) -> None:
        """Drop the current menu-armed detail (menu closed by any means)."""
        self._armed_detail = None

    def handle_event(self, event: dict) -> GestureEventResult:
        if event.get("contract") != CONTRACT_NAME:
            return GestureEventResult()

        cursor = event.get("cursor") or {}
        # Cursor-off gesture mode has no mouse hover, so nothing would ever
        # get selected without this: default to the first selectable detail.
        if not cursor.get("enabled") and self.model.get_highlighted() is None:
            self.model.highlight_first()

        gestures = event.get("gestures") or {}
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

        command_result = self._command_mapper.map(gesture_result)
        command = command_result.command

        if command == CommandName.EMERGENCY_STOP:
            self._dispatcher.dispatch(command_result, self.model)
            result.emergency = True
            result.message = "✊ Аварийная остановка течения"
            return result

        if command == CommandName.OPEN_VALVE:
            self._apply_thumb_up(command_result, result)
            return result

        if gesture_result.name == GestureName.VICTORY:
            result.message = "✌ Победа (курсор вкл/выкл)"
            return result

        # OPEN_PALM and anything else not in docs/command_mapping.md: show the
        # gesture, but do not guess a command for it.
        result.message = self._format_generic(name, confidence)
        return result

    def _apply_thumb_up(self, command_result, result: GestureEventResult) -> None:
        """THUMB_UP toggles open/close, but only on a detail armed by
        POINTING_INDEX first (menu opened) - see class docstring."""
        detail = self.model.get_highlighted()

        if detail is None or detail is not self._armed_detail:
            result.message = "👍 Сначала откройте меню (👉)"
            return

        if not (hasattr(detail, "open") and hasattr(detail, "close")) or detail.has_animation():
            result.message = "👍 Большой палец вверх (нет цели)"
            return

        action = self._dispatcher.dispatch(command_result, self.model, detail=detail)
        result.message = f"👍 {'Открыть' if action == 'open' else 'Закрыть'}: {detail.name}"
        result.action_taken = True
        result.close_menu = True
        self._armed_detail = None

    def _apply_dynamic_gesture(self, dynamic: dict) -> GestureEventResult:
        name = dynamic.get("name")
        confidence = dynamic.get("confidence", 0) or 0.0
        result = GestureEventResult(gesture_name=None, gesture_confidence=0.0)

        try:
            gesture_name = GestureName(name)
        except ValueError:
            return result

        if gesture_name == GestureName.POINTING_INDEX:
            self._apply_pointing_index(result)
            return result

        if gesture_name in (GestureName.SWIPE_LEFT, GestureName.SWIPE_RIGHT):
            self._apply_swipe(gesture_name, result)
            return result

        label = NOT_IMPLEMENTED_LABELS.get(gesture_name)
        if label is not None:
            result.message = f"{label}{NOT_IMPLEMENTED_SUFFIX}"
        return result

    def _apply_pointing_index(self, result: GestureEventResult) -> None:
        detail = self.model.get_highlighted()
        if detail is None or not self.model.get_menu_actions(detail):
            result.message = "👉 Указание (нет цели)"
            return

        self._armed_detail = detail
        result.open_menu = True
        result.action_taken = True
        result.message = f"👉 Меню: {detail.name}"

    def _apply_swipe(self, gesture_name: GestureName, result: GestureEventResult) -> None:
        if gesture_name == GestureName.SWIPE_LEFT:
            self.model.highlight_previous()
        else:
            self.model.highlight_next()
        # Selection changed - any menu armed for the previous detail is stale,
        # so drop the arming AND ask the UI to close that now-orphaned menu
        # (harmless if none is open - the Qt layer guards on that).
        self._armed_detail = None
        result.close_menu = True
        result.action_taken = True

        detail = self.model.get_highlighted()
        arrow = "👈" if gesture_name == GestureName.SWIPE_LEFT else "👉"
        result.message = f"{arrow} Выбрано: {detail.name}" if detail else f"{arrow} Нет деталей"

    def _format_generic(self, name: str | None, confidence) -> str:
        try:
            return f"✋ {name} ({float(confidence):.0%})"
        except (TypeError, ValueError):
            return f"✋ {name}"
