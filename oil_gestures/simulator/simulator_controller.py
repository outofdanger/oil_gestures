from __future__ import annotations

from dataclasses import dataclass

from oil_gestures.commands.command_dispatcher import CommandDispatcher
from oil_gestures.commands.command_mapper import CommandMapper
from oil_gestures.core.enums import CommandName, GestureName, RecognitionSource
from oil_gestures.core.types import GestureResult

CONTRACT_NAME = "oil_gestures.ml.runtime"

# Static-gesture -> command mapping (docs/command_mapping.md). FIST is dispatched
# as an emergency stop here; THUMB_UP is handled as a generic "activate selected
# detail" (the Qt layer decides what activation means per detail type), so it no
# longer needs a concrete command here beyond marking intent.
STATIC_GESTURE_COMMANDS: dict[str, str] = {
    "FIST": "EMERGENCY_STOP",
    "THUMB_UP": "OPEN_VALVE",
}


@dataclass
class GestureEventResult:
    """What a single ML contract event means for the scene/UI. The Qt layer
    (Controller) reads these flags and performs the actual scene/Qt work."""

    message: str | None = None
    action_taken: bool = False
    emergency: bool = False
    # POINTING_INDEX: open the selected detail's context menu (preview).
    open_menu: bool = False
    # Close a gesture-opened menu (selection changed / detail activated).
    close_menu: bool = False
    # THUMB_UP: activate/click the currently selected detail.
    activate: bool = False
    # SQUEEZE / RELEASE: zoom into the selected assembly / return to main view.
    zoom_in: bool = False
    zoom_out: bool = False
    # ROTATE_CLOCKWISE / CCW: open the selected valve a bit more (+1) / less (-1)
    # (its % open slider). The chain logic turns valve openness into pressure.
    rotate_step: int = 0
    # Whether the cursor (mouse-control) mode is active this frame - the UI uses
    # it to show the mode and to know scene gestures were intentionally skipped.
    cursor_enabled: bool = False
    # Current gesture for a dedicated UI readout, independent of the message.
    gesture_name: str | None = None
    gesture_confidence: float = 0.0


class SimulatorController:
    """
    Owns the "gesture -> scene action" interpretation (docs/command_mapping.md).
    Qt-free: it only sets flags on GestureEventResult; the Controller performs
    the actual scene/Qt work (activation, zoom, pressure, menu), reusing the
    colleagues' click/focus logic. Stays an autonomous contract consumer per
    docs/integration_contract.md.

    Modes are mutually exclusive: when cursor (mouse-control) mode is on, scene
    gestures are NOT interpreted here (only cursor status is reported), so a
    SWIPE/THUMB_UP can't move the scene while the hand is driving the cursor.
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
        if cursor.get("enabled"):
            # Cursor mode: hand drives the OS mouse; scene gestures are muted so
            # the two modes never fight. Report only the cursor status/mode.
            return GestureEventResult(
                cursor_enabled=True,
                message=self._cursor_message(cursor),
            )

        # Gesture (navigation/control) mode. No mouse hover here, so nothing
        # would ever get selected without defaulting to the first detail.
        if self.model.get_highlighted() is None:
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

        if gesture_result.name == GestureName.THUMB_UP:
            # Activate the currently selected detail - the Controller decides
            # what activation means per type (valve toggle, remove manometer,
            # press controller button, ...). Close any open preview menu.
            result.activate = True
            result.close_menu = True
            result.action_taken = True
            result.message = "👍 Активировать выбранное"
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
        result = GestureEventResult(gesture_name=None, gesture_confidence=0.0)

        try:
            gesture_name = GestureName(name)
        except ValueError:
            return result

        if gesture_name == GestureName.POINTING_INDEX:
            self._apply_pointing_index(result)
        elif gesture_name == GestureName.SWIPE_LEFT:
            self._apply_swipe(result)
        elif gesture_name == GestureName.ROTATE_CLOCKWISE:
            result.rotate_step = 1
            result.message = "🔄 Открыть вентиль больше"
        elif gesture_name == GestureName.ROTATE_COUNTERCLOCKWISE:
            result.rotate_step = -1
            result.message = "🔄 Прикрыть вентиль"
        elif gesture_name == GestureName.SQUEEZE:
            result.zoom_in = True
            result.message = "🤏 Приблизить"
        elif gesture_name == GestureName.RELEASE:
            result.zoom_out = True
            result.message = "🖐 Общий вид"
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

    def _apply_swipe(self, result: GestureEventResult) -> None:
        # Single-direction cycling (SWIPE_RIGHT is dropped upstream - a swipe's
        # return stroke reads as the opposite swipe). Steps forward and wraps
        # last -> first over the current selection scope (all big nodes, or the
        # controller's buttons when zoomed into it - see Model selection scope).
        self.model.highlight_next()
        # Selection changed - any open preview menu is stale.
        self._armed_detail = None
        result.close_menu = True
        result.action_taken = True

        detail = self.model.get_highlighted()
        result.message = f"👉 Выбрано: {detail.name}" if detail else "👉 Нет деталей"

    def _format_generic(self, name: str | None, confidence) -> str:
        try:
            return f"✋ {name} ({float(confidence):.0%})"
        except (TypeError, ValueError):
            return f"✋ {name}"
