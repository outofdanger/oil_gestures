from __future__ import annotations

from oil_gestures.core.types import GestureResult, LandmarkPacket
from oil_gestures.gestures.dynamic.rule_based_dynamic import RuleBasedDynamicConfig, RuleBasedDynamicRecognizer


class DynamicGestureRecognizer:
    """
    Facade for the dynamic gesture layer.

    A learned model can be plugged in later without changing app/runtime code.
    For the MVP this delegates to the rule-based recognizer.
    """

    def __init__(self, rule_based: RuleBasedDynamicRecognizer | None = None) -> None:
        self.rule_based = rule_based or RuleBasedDynamicRecognizer(RuleBasedDynamicConfig())

    def reset(self) -> None:
        self.rule_based.reset()

    def update(self, packet: LandmarkPacket) -> GestureResult | None:
        return self.rule_based.update(packet)
