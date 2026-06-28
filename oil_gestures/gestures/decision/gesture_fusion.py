from __future__ import annotations

from dataclasses import dataclass

from oil_gestures.core.types import GestureResult
from oil_gestures.gestures.decision.decision_layer import DecisionLayer, DecisionLayerConfig

# How long to wait between two fires of the dynamic channel. The learned model
# can flip between SWIPE_LEFT and SWIPE_RIGHT frame to frame on the smallest
# hand motion; without this, every flip steps the selection, so a still hand
# turns into a storm of selection changes. ~0.6s -> at most ~1.5 steps/sec.
SWIPE_COOLDOWN_SECONDS = 0.6


@dataclass(frozen=True)
class FusedGestures:
    """One frame's worth of post-decision gesture results, ready to publish."""

    static: GestureResult | None
    dynamic: GestureResult | None


class GestureFusion:
    """
    Applies a DecisionLayer per recognition channel and reassembles the frame.

    Does not collapse static/dynamic into a single "winning" gesture - they
    are independent recognition subsystems (per docs/interaction_spec.md) and
    the UI already handles them arriving in parallel (e.g. cursor-mode message
    next to a static-gesture action). What this *does* do is stop a held
    gesture from being published as the same classification on every frame -
    see DecisionLayer.

    Cursor gestures are not gated here: they drive continuous pointer motion,
    not a discrete recognized gesture, so there is nothing to debounce.
    """

    def __init__(self, swipe_cooldown_seconds: float = SWIPE_COOLDOWN_SECONDS) -> None:
        # Static channel relies on the per-hold latch alone (cooldown 0): one
        # of its gestures is the emergency stop (FIST), and a channel-wide
        # cooldown could swallow an emergency FIST issued moments after a
        # THUMB_UP. The latch still stops a held gesture repeating every frame.
        self._static_decision = DecisionLayer(DecisionLayerConfig(cooldown_seconds=0.0))
        # Dynamic channel gets a real cooldown to tame SWIPE_LEFT/RIGHT jitter
        # (tunable via configs/gestures.yaml: dynamic.swipe_cooldown_seconds).
        self._dynamic_decision = DecisionLayer(
            DecisionLayerConfig(cooldown_seconds=swipe_cooldown_seconds)
        )

    def reset(self) -> None:
        self._static_decision.reset()
        self._dynamic_decision.reset()

    def fuse(
        self,
        static_gesture: GestureResult | None,
        dynamic_gesture: GestureResult | None,
    ) -> FusedGestures:
        static_name = static_gesture.name if static_gesture is not None else None
        dynamic_name = dynamic_gesture.name if dynamic_gesture is not None else None

        static_timestamp = static_gesture.timestamp if static_gesture is not None else None
        dynamic_timestamp = dynamic_gesture.timestamp if dynamic_gesture is not None else None
        decided_static = self._static_decision.decide(static_name, static_timestamp)
        decided_dynamic = self._dynamic_decision.decide(dynamic_name, dynamic_timestamp)

        return FusedGestures(
            static=static_gesture if decided_static is not None else None,
            dynamic=dynamic_gesture if decided_dynamic is not None else None,
        )
