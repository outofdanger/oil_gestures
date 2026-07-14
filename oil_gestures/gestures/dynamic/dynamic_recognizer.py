from __future__ import annotations

from dataclasses import dataclass

from oil_gestures.core.constants import DEFAULT_DYNAMIC_CONFIDENCE_THRESHOLD, DEFAULT_SEQUENCE_LENGTH
from oil_gestures.core.enums import RecognitionSource
from oil_gestures.core.types import GestureResult, LandmarkPacket
from oil_gestures.gestures.dynamic.dynamic_model import DynamicGestureModel


@dataclass(frozen=True)
class DynamicRecognizerConfig:
    enabled: bool = True
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    # ST-GCN (the lead/trigger) must reach this probability on a non-IDLE class
    # to fire. ST-GCN reads hand *pose*, so it triggers even on small motion -
    # keep this fairly high, but BiLSTM confirmation (veto_floor) is the bigger
    # lever against accidental swipes.
    min_confidence: float = DEFAULT_DYNAMIC_CONFIDENCE_THRESHOLD
    # BiLSTM (the confirm/veto) must give the ST-GCN-proposed class at least
    # this probability or the trigger is suppressed. BiLSTM reads *motion*, so
    # a high value here means "only fire when the hand actually moved like the
    # gesture", which is what kills false swipes from tiny hand jitter.
    veto_floor: float = 0.20
    # Minimum time between two dynamic-channel fires (SWIPE_LEFT/RIGHT,
    # POINTING_INDEX). Rate-limits stepping through elements; does not affect
    # recognition quality or the continuous ROTATE gestures.
    swipe_cooldown_seconds: float = 0.6
    # Torch device for the ensemble. "auto" = CUDA (discrete NVIDIA) -> MPS
    # (Apple Silicon GPU) -> CPU. Force with "cuda" / "mps" / "cpu". The models
    # and their per-frame inputs are moved to this device in
    # gestures.dynamic.model_loader.
    device: str = "auto"
    # Rest floor for the directional lockout (gestures.dynamic.model_loader):
    # after a direction fires, its opposite is suppressed until the hand rests
    # for this long. The return stroke is suppressed until the hand actually
    # stops (any duration), so this stays small.
    directional_lockout_seconds: float = 0.4
    # Checkpoint pair loaded by load_dynamic_model() in
    # gestures.dynamic.model_loader, which runs the "ST-GCN leads + BiLSTM
    # confirms" ensemble (validated live in
    # dynamic_gestures/scripts/test_dynamic_model.py before this lived here).
    # Each model reads its own window length from its checkpoint, so neither
    # uses sequence_length above. Either set to None disables dynamic
    # recognition entirely (no model loaded), same as today.
    stgcn_checkpoint_path: str | None = "assets/models/pytorch/dynamic_stgcn_transition.pt"
    bilstm_checkpoint_path: str | None = "assets/models/pytorch/dynamic_bilstm_transition.pt"


class DynamicGestureRecognizer:
    """Model-only dynamic recognition facade; it contains no cursor rules."""

    def __init__(
        self,
        config: DynamicRecognizerConfig | None = None,
        model: DynamicGestureModel | None = None,
    ) -> None:
        self.config = config or DynamicRecognizerConfig()
        self.model = model

    def reset(self) -> None:
        if self.model is not None:
            self.model.reset()

    def update(self, packet: LandmarkPacket) -> GestureResult | None:
        if not self.config.enabled or self.model is None:
            return None
        result = self.model.update(packet)
        if (
            result is None
            or result.source != RecognitionSource.DYNAMIC_MODEL
            or result.confidence < self.config.min_confidence
        ):
            return None
        return result
