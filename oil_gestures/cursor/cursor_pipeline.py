from __future__ import annotations

from dataclasses import dataclass, field

from oil_gestures.core.enums import CursorAction, GestureName
from oil_gestures.core.types import CursorControlResult, GestureResult, LandmarkPacket, PointerPosition, ScreenPosition
from oil_gestures.cursor.action_mapper import CursorActionMapper
from oil_gestures.cursor.cursor_smoothing import CursorSmoother
from oil_gestures.cursor.hand_pointer import HandPointer
from oil_gestures.cursor.mouse_controller import MouseController
from oil_gestures.cursor.screen_mapper import ScreenMapper


@dataclass
class CursorPipelineState:
    pointer: PointerPosition | None = None
    screen_position: ScreenPosition | None = None
    action_status: str = ""
    pressed: bool = False
    valid_detection_frames: int = 0
    last_seen_time: float | None = None
    last_action_times: dict[CursorAction, float] = field(default_factory=dict)
    # Absolute screen point where the current grab started; movement while
    # pressed is scaled around it by grab_sensitivity. None when not grabbing.
    grab_anchor: tuple[float, float] | None = None


class CursorPipeline:
    def __init__(
        self,
        hand_pointer: HandPointer,
        screen_mapper: ScreenMapper,
        smoother: CursorSmoother,
        action_mapper: CursorActionMapper,
        mouse: MouseController,
        reacquire_frames: int,
        lost_reset_seconds: float,
        action_cooldown_seconds: float = 0.0,
        enabled: bool = True,
        grab_sensitivity: float = 1.0,
    ) -> None:
        self.hand_pointer = hand_pointer
        self.screen_mapper = screen_mapper
        self.smoother = smoother
        self.action_mapper = action_mapper
        self.mouse = mouse
        self.reacquire_frames = reacquire_frames
        self.lost_reset_seconds = lost_reset_seconds
        self.action_cooldown_seconds = max(0.0, action_cooldown_seconds)
        self.enabled = enabled
        # Clamp to a small positive floor so a grab can never fully freeze the
        # cursor; 1.0 (or more) means "no reduction", matching old behaviour.
        self.grab_sensitivity = max(0.05, grab_sensitivity)
        self.state = CursorPipelineState()

    def reset(self) -> None:
        self._release_held_button(self.state.last_seen_time or 0.0, self.state.screen_position)
        self.state.valid_detection_frames = 0
        self.state.pointer = None
        self.state.screen_position = None
        self.state.action_status = ""
        self.state.pressed = False
        self.state.grab_anchor = None
        self.state.last_seen_time = None
        self.state.last_action_times.clear()
        self._reset_smoother_to_mouse()

    def _control_enabled(self, timestamp: float, paused: bool) -> bool:
        if paused or not self.enabled:
            return False
        if self.state.valid_detection_frames < self.reacquire_frames:
            return False
        if self.state.last_seen_time is None:
            return False
        return timestamp - self.state.last_seen_time <= self.lost_reset_seconds

    def _action_ready(self, action: CursorAction, timestamp: float) -> bool:
        if self.action_cooldown_seconds <= 0.0:
            return True
        previous = self.state.last_action_times.get(action)
        return previous is None or timestamp - previous >= self.action_cooldown_seconds

    def _mark_action(self, action: CursorAction, timestamp: float) -> None:
        self.state.last_action_times[action] = timestamp

    def _release_held_button(
        self,
        timestamp: float,
        screen_position: ScreenPosition | None = None,
    ) -> None:
        if not self.state.pressed:
            return

        position = screen_position or self.state.screen_position
        self.mouse.execute(CursorAction.RELEASE, position)
        self.state.pressed = False
        self.state.grab_anchor = None
        self._mark_action(CursorAction.RELEASE, timestamp)

    def _apply_grab_gain(self, position: ScreenPosition) -> ScreenPosition:
        """While a grab is held, scale movement around the grab anchor by
        grab_sensitivity so the same hand travel moves the cursor less (precise
        drag). No-op when not grabbing or sensitivity >= 1.0."""
        if (
            not self.state.pressed
            or self.state.grab_anchor is None
            or self.grab_sensitivity >= 1.0
        ):
            return position
        anchor_x, anchor_y = self.state.grab_anchor
        scaled_x = anchor_x + (position.x - anchor_x) * self.grab_sensitivity
        scaled_y = anchor_y + (position.y - anchor_y) * self.grab_sensitivity
        return ScreenPosition(int(round(scaled_x)), int(round(scaled_y)), position.timestamp)

    def _reset_smoother_to_mouse(self) -> None:
        try:
            self.smoother.reset(self.mouse.get_position())
        except Exception:
            self.smoother.reset(None)

    def _execute_discrete_action(self, action: CursorAction, position: ScreenPosition, timestamp: float) -> bool:
        if action == CursorAction.RIGHT_CLICK:
            if self.state.pressed or not self._action_ready(action, timestamp):
                return False
            self.mouse.execute(action, position)
            self._mark_action(action, timestamp)
            return True

        if action == CursorAction.GRAB:
            if self.state.pressed or not self._action_ready(action, timestamp):
                return False
            self.mouse.execute(action, position)
            self.state.pressed = True
            # Anchor the precise-drag scaling at the (absolute) grab point.
            self.state.grab_anchor = (float(position.x), float(position.y))
            self._mark_action(action, timestamp)
            return True

        if action == CursorAction.RELEASE:
            if not self.state.pressed:
                return False
            self.mouse.execute(action, position)
            self.state.pressed = False
            self.state.grab_anchor = None
            self._mark_action(action, timestamp)
            return True

        return False

    def _set_action_status(self, result: CursorControlResult) -> None:
        if result.action == CursorAction.NONE:
            self.state.action_status = ""
            return
        if result.source_gesture == GestureName.UNKNOWN:
            self.state.action_status = result.action.value
            return
        self.state.action_status = f"{result.source_gesture.value} -> {result.action.value}"

    def process(
        self,
        packet: LandmarkPacket,
        cursor_gesture: GestureResult | None = None,
        paused: bool = False,
    ) -> CursorControlResult:
        if not packet.hand_detected or packet.landmarks is None:
            previous_seen_time = self.state.last_seen_time
            self._release_held_button(packet.timestamp, self.state.screen_position)
            self.state.valid_detection_frames = 0
            self.state.pointer = None
            self.state.screen_position = None
            self.state.action_status = ""
            self.state.pressed = False
            self.state.grab_anchor = None
            if previous_seen_time is not None and packet.timestamp - previous_seen_time <= self.lost_reset_seconds:
                self.state.last_seen_time = previous_seen_time
            else:
                self.state.last_seen_time = None
                self._reset_smoother_to_mouse()
            self.state.last_action_times.clear()
            return CursorControlResult(CursorAction.NONE, None, GestureName.UNKNOWN, 0.0, packet.timestamp)

        pointer = self.hand_pointer.extract(packet)
        self.state.pointer = pointer
        screen_position = self.screen_mapper.map(pointer)
        self.state.screen_position = screen_position
        self.state.valid_detection_frames += 1
        self.state.last_seen_time = packet.timestamp

        if screen_position is None or not self._control_enabled(packet.timestamp, paused):
            if screen_position is None or paused or not self.enabled:
                self._release_held_button(packet.timestamp, screen_position)
            self.state.action_status = ""
            return CursorControlResult(
                CursorAction.NONE,
                screen_position,
                GestureName.UNKNOWN,
                pointer.confidence,
                packet.timestamp,
            )

        smoothed_position = self.smoother.apply(screen_position)
        # While a grab is held, scale movement down around the grab anchor for a
        # precise drag; identical to smoothed_position otherwise. The smoother
        # keeps tracking the absolute position, so releasing resumes normal
        # absolute mapping (the cursor eases back toward the hand).
        effective_position = self._apply_grab_gain(smoothed_position)
        self.state.screen_position = effective_position

        result = self.action_mapper.map(
            cursor_gesture=cursor_gesture,
            screen_position=effective_position,
            timestamp=packet.timestamp,
        )
        if result.action == CursorAction.NONE:
            result = CursorControlResult(
                CursorAction.NONE,
                effective_position,
                GestureName.UNKNOWN,
                pointer.confidence,
                packet.timestamp,
            )
        elif result.action == CursorAction.MOVE_CURSOR:
            if self.state.pressed:
                self.mouse.drag_to(effective_position)
            else:
                self.mouse.move_to(effective_position)
        else:
            if result.action == CursorAction.RELEASE and self.state.pressed:
                self.mouse.drag_to(effective_position)
            else:
                self.mouse.move_to(effective_position)
            self._execute_discrete_action(result.action, effective_position, packet.timestamp)

        self._set_action_status(result)
        return result
