from __future__ import annotations

from dataclasses import dataclass, field

from oil_gestures.core.enums import GestureName

# Gestures that represent a discrete action (open a menu, toggle a valve,
# step a selection) must fire exactly once per "show", not once per frame -
# holding THUMB_UP for a second at 30 FPS must not toggle a valve 30 times.
# ROTATE_CLOCKWISE/CCW are deliberately excluded: their *magnitude* matters
# continuously while held (pressure goes up/down for as long as the gesture
# is shown), so gating them here would break that, not fix anything.
DEFAULT_EDGE_TRIGGERED: frozenset[GestureName] = frozenset(
    {
        GestureName.FIST,
        GestureName.THUMB_UP,
        GestureName.SWIPE_LEFT,
        GestureName.SWIPE_RIGHT,
        GestureName.POINTING_INDEX,
    }
)


@dataclass(frozen=True)
class DecisionLayerConfig:
    edge_triggered: frozenset[GestureName] = field(default_factory=lambda: DEFAULT_EDGE_TRIGGERED)
    # Minimum time between two fires in this channel, even for *different*
    # edge-triggered gestures - without this, frame-to-frame model flicker
    # between e.g. SWIPE_LEFT and SWIPE_RIGHT each look like "gesture changed"
    # and fire immediately, turning the smallest hand motion into a rapid
    # back-and-forth of selection changes.
    cooldown_seconds: float = 0.5


class DecisionLayer:
    """
    Turns "a gesture is being held" into "a gesture just happened, once".

    Generalizes the dwell+release-latch pattern already used by GestureToggle
    (built only for the VICTORY cursor toggle) to any gesture that should not
    repeat every frame while held. Feed it one channel's recognized gesture
    name (or None) once per frame; it returns the same name back only on the
    frame the gesture should be acted on, and None on every later frame of the
    same hold - so a downstream consumer (e.g. the published ML contract)
    naturally sees "no gesture" while the operator is still holding still,
    instead of the same classification repeated dozens of times a second.

    Also enforces a minimum cooldown between two fires regardless of whether
    the gesture changed in between - a hold+release latch alone only stops a
    *steady* gesture from repeating; it does nothing about a noisy model
    alternating between two different edge-triggered gestures frame to frame
    (each alternation looks like a fresh "gesture changed" event and would
    otherwise fire immediately).

    One instance per recognition channel (static, dynamic) - gestures from
    different channels never share state.
    """

    def __init__(self, config: DecisionLayerConfig | None = None) -> None:
        self.config = config or DecisionLayerConfig()
        self._last_gesture: GestureName | None = None
        self._fired_for_current_hold = False
        self._last_fired_at: float | None = None

    def reset(self) -> None:
        self._last_gesture = None
        self._fired_for_current_hold = False
        self._last_fired_at = None

    def decide(self, gesture: GestureName | None, timestamp: float | None = None) -> GestureName | None:
        if gesture != self._last_gesture:
            # Gesture changed (including a release to None): re-arm.
            self._last_gesture = gesture
            self._fired_for_current_hold = False

        if gesture is None:
            return None

        if gesture not in self.config.edge_triggered:
            # Continuous gesture - pass through unchanged every frame.
            return gesture

        if self._fired_for_current_hold:
            return None

        if (
            timestamp is not None
            and self._last_fired_at is not None
            and timestamp - self._last_fired_at < self.config.cooldown_seconds
        ):
            # Too soon after the last fire in this channel - swallow it.
            # Deliberately do NOT latch _fired_for_current_hold here: if the
            # operator keeps holding the gesture, it should fire as soon as the
            # cooldown elapses (hold-to-repeat at the cooldown rate), and a
            # gesture blocked only by a transient flicker shouldn't need a full
            # release + re-show to become live again. A steady held gesture
            # still fires only once - that case is the latch above, not here.
            return None

        self._fired_for_current_hold = True
        self._last_fired_at = timestamp
        return gesture
