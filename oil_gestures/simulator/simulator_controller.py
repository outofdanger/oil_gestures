from __future__ import annotations

from dataclasses import dataclass

from oil_gestures.core.enums import GestureName

CONTRACT_NAME = "oil_gestures.ml.runtime"


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

    def __init__(self, model) -> None:
        self.model = model
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
            gesture_name = GestureName(name)
        except ValueError:
            result.message = self._format_generic(name, confidence)
            return result

        if gesture_name == GestureName.THUMB_UP:
            # Activate the currently selected detail - the Controller decides
            # what activation means per type (valve toggle, remove manometer,
            # press controller button, ...). Close any open preview menu.
            result.activate = True
            result.close_menu = True
            result.action_taken = True
            result.message = "👍 Активировать выбранное"
        elif gesture_name == GestureName.VICTORY:
            result.message = "✌ Победа (курсор вкл/выкл)"
        else:
            # FIST больше не аварийка (пересекается со SQUEEZE-зумом): аварийный
            # стоп теперь только в меню и на контроллере. OPEN_PALM/прочее -
            # просто показать жест.
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
        elif gesture_name in (GestureName.SWIPE_LEFT, GestureName.SWIPE_RIGHT):
            self._apply_swipe(gesture_name, result)
        elif gesture_name == GestureName.ROTATE_CLOCKWISE:
            result.rotate_step = 1
            result.close_menu = True  # чтобы не осталось стухшего превью-меню
            result.message = "🔄 Открыть вентиль больше"
        elif gesture_name == GestureName.ROTATE_COUNTERCLOCKWISE:
            result.rotate_step = -1
            result.close_menu = True
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

    def _apply_swipe(self, gesture_name: GestureName, result: GestureEventResult) -> None:
        # Two-directional cycling over the current selection scope (all big
        # nodes, or the controller's buttons when zoomed in - see Model
        # selection scope). The return-stroke false-opposite is handled upstream
        # by the directional lockout in gestures.dynamic.model_loader, so both
        # SWIPE_LEFT and SWIPE_RIGHT are safe to use again.
        if gesture_name == GestureName.SWIPE_RIGHT:
            self.model.highlight_previous()
        else:
            self.model.highlight_next()
        # Selection changed - any open preview menu is stale.
        self._armed_detail = None
        result.close_menu = True
        result.action_taken = True

        detail = self.model.get_highlighted()
        arrow = "👈" if gesture_name == GestureName.SWIPE_RIGHT else "👉"
        result.message = f"{arrow} Выбрано: {detail.name}" if detail else f"{arrow} Нет деталей"

    def _format_generic(self, name: str | None, confidence) -> str:
        try:
            return f"✋ {name} ({float(confidence):.0%})"
        except (TypeError, ValueError):
            return f"✋ {name}"
