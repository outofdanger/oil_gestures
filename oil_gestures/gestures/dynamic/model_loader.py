from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from oil_gestures.core.constants import DEFAULT_DYNAMIC_CONFIDENCE_THRESHOLD
from oil_gestures.core.enums import GestureName, RecognitionSource
from oil_gestures.core.logger import get_logger
from oil_gestures.core.types import GestureResult, LandmarkPacket
from oil_gestures.gestures.dynamic.feature_extractor import (
    landmarks_to_array,
    normalize_sequence,
    pack_bilstm_features,
    pack_stgcn_features,
    resample_sequence,
)
from oil_gestures.gestures.dynamic.sequence_buffer import SequenceBuffer, SequenceBufferConfig
from oil_gestures.gestures.dynamic.smoothing import ProbabilitySmoother, ProbabilitySmoothingConfig

logger = get_logger(__name__)

# IDLE is the "no intentional motion" baseline every checkpoint is trained
# against (see process_dynamic_dataset.py) - never something to act on.
_BASELINE_LABEL = GestureName.IDLE.value

# Continuous gestures: their value matters every frame for as long as they are
# held (ROTATE turns pressure up/down continuously), so they must keep emitting
# and are exempt from the discrete-gesture refractory below. Everything else
# (SWIPE_LEFT, POINTING_INDEX, SQUEEZE, RELEASE) is a one-shot discrete action.
_CONTINUOUS_LABELS: frozenset[str] = frozenset(
    {GestureName.ROTATE_CLOCKWISE.value, GestureName.ROTATE_COUNTERCLOCKWISE.value}
)

# Opposite-direction pairs. The *return stroke* of a gesture (the hand moving
# back to neutral after a rotate/swipe/squeeze) is itself recognized as the
# OPPOSITE gesture, which would undo the action. Instead of disabling one
# direction outright (the old SWIPE_RIGHT hack), we lock out the opposite
# direction for a short window after a direction was last active - see the
# directional lockout in EnsembleDynamicGestureModel.update().
_OPPOSITE_LABELS: dict[str, str] = {
    GestureName.SWIPE_LEFT.value: GestureName.SWIPE_RIGHT.value,
    GestureName.SWIPE_RIGHT.value: GestureName.SWIPE_LEFT.value,
    GestureName.ROTATE_CLOCKWISE.value: GestureName.ROTATE_COUNTERCLOCKWISE.value,
    GestureName.ROTATE_COUNTERCLOCKWISE.value: GestureName.ROTATE_CLOCKWISE.value,
    GestureName.SQUEEZE.value: GestureName.RELEASE.value,
    GestureName.RELEASE.value: GestureName.SQUEEZE.value,
}


class _BiLSTMGestureClassifier(nn.Module):
    """
    Mirrors dynamic_gestures/scripts/train_dynamic_model.py:BiLSTMGestureClassifier
    layer-for-layer so a checkpoint trained there loads via load_state_dict
    unmodified. Kept self-contained instead of importing the training script,
    which is a CLI tool (argparse, sys.path hacks), not a library.
    """

    def __init__(
        self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        forward_last = h_n[-2]
        backward_last = h_n[-1]
        combined = torch.cat([forward_last, backward_last], dim=-1)
        combined = self.dropout(combined)
        return self.classifier(combined)


class _SpatialGraphConv(nn.Module):
    """Graph convolution over the K adjacency subsets. Mirrors
    train_stgcn_model.py:SpatialGraphConv exactly."""

    def __init__(self, in_channels: int, out_channels: int, num_subsets: int) -> None:
        super().__init__()
        self.num_subsets = num_subsets
        self.conv = nn.Conv2d(in_channels, out_channels * num_subsets, kernel_size=1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V) ; adjacency: (K, V, V)
        x = self.conv(x)
        n, kc, t, v = x.shape
        x = x.view(n, self.num_subsets, kc // self.num_subsets, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, adjacency)
        return x.contiguous()


class _STGCNBlock(nn.Module):
    """Spatial graph conv + temporal conv with residual connection. Mirrors
    train_stgcn_model.py:STGCNBlock exactly."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_subsets: int,
        t_kernel: int,
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        padding = (t_kernel - 1) // 2
        self.gcn = _SpatialGraphConv(in_channels, out_channels, num_subsets)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (t_kernel, 1), (stride, 1), (padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        res = 0 if self.residual is None else self.residual(x)
        x = self.gcn(x, adjacency)
        x = self.tcn(x)
        return self.relu(x + res)


class _STGCN(nn.Module):
    """ST-GCN classifier for single-hand landmark sequences (N, C, T, V, M).
    Mirrors train_stgcn_model.py:STGCN exactly so a trained checkpoint loads
    via load_state_dict unmodified."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        adjacency: np.ndarray,
        base_channels: int,
        t_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        num_subsets, num_nodes, _ = adjacency.shape
        self.register_buffer("adjacency", torch.tensor(adjacency, dtype=torch.float32))
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        c1, c2 = base_channels, base_channels * 2
        self.blocks = nn.ModuleList(
            [
                _STGCNBlock(in_channels, c1, num_subsets, t_kernel, residual=False),
                _STGCNBlock(c1, c1, num_subsets, t_kernel, dropout=dropout),
                _STGCNBlock(c1, c1, num_subsets, t_kernel, dropout=dropout),
                _STGCNBlock(c1, c2, num_subsets, t_kernel, stride=2, dropout=dropout),
                _STGCNBlock(c2, c2, num_subsets, t_kernel, dropout=dropout),
            ]
        )
        self.edge_importance = nn.ParameterList(
            [nn.Parameter(torch.ones_like(self.adjacency)) for _ in self.blocks]
        )
        self.classifier = nn.Linear(c2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V, M)
        n, c, t, v, m = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)

        for block, importance in zip(self.blocks, self.edge_importance):
            x = block(x, self.adjacency * importance)

        x = nn.functional.avg_pool2d(x, x.shape[2:])
        x = x.view(n, m, -1).mean(dim=1)
        return self.classifier(x)


@dataclass(frozen=True)
class DynamicModelLoaderConfig:
    stgcn_checkpoint_path: str = "assets/models/pytorch/dynamic_stgcn_merged.pt"
    bilstm_checkpoint_path: str = "assets/models/pytorch/dynamic_bilstm_merged.pt"
    device: str = "auto"
    # Minimum ST-GCN probability for its top class to count as a trigger at all.
    min_confidence: float = DEFAULT_DYNAMIC_CONFIDENCE_THRESHOLD
    # Minimum BiLSTM probability for the ST-GCN-proposed class to count as
    # CONFIRMED (unless it's already in BiLSTM's top-2). Lower = trust
    # ST-GCN more (snappier, more false fires); higher = stricter (fewer
    # false fires, slightly slower). Matches test_dynamic_model.py's default.
    veto_floor: float = 0.20
    # EMA weight for the newest frame in each model's own probability smoother.
    smoothing_alpha: float = 0.5
    # After a direction (SWIPE_LEFT, ROTATE_CW, SQUEEZE, ...) is active, its
    # opposite is suppressed for this long - the hand's return stroke reads as
    # the opposite gesture and would undo the action. Larger = safer against
    # accidental reversals but slower to switch direction on purpose.
    directional_lockout_seconds: float = 0.8


def _ensemble_decision(
    stgcn_probs: np.ndarray,
    bilstm_probs: np.ndarray,
    class_names: list[str],
    min_confidence: float,
    veto_floor: float,
) -> str | None:
    """Dual-model decision: ST-GCN LEADS (fast trigger), BiLSTM CONFIRMS (veto).

    Based on dynamic_gestures/scripts/test_dynamic_model.py:ensemble_decision.
    ST-GCN reads spatial hand *pose* (world landmarks), so it crosses the
    confidence threshold even on small motion - it drives the trigger. BiLSTM
    reads *motion/velocity*, so requiring it to independently agree is what
    rejects accidental swipes from tiny hand jitter: the proposed class is
    CONFIRMED only if BiLSTM gives it at least ``veto_floor`` probability,
    otherwise it is VETOED (treated as no gesture).

    The original ported logic also confirmed when the class merely landed in
    BiLSTM's top-2; that escape hatch was dropped because it let small
    movements through (during a tiny jitter BiLSTM ranks IDLE #1 and a swipe
    #2, which satisfied top-2 no matter how high veto_floor was set). Now
    veto_floor is the single, real confirmation gate.
    """
    lead = int(np.argmax(stgcn_probs))
    lead_label = class_names[lead]
    if lead_label == _BASELINE_LABEL or float(stgcn_probs[lead]) < min_confidence:
        return None
    confirmed = float(bilstm_probs[lead]) >= veto_floor
    return lead_label if confirmed else None


class EnsembleDynamicGestureModel:
    """
    Loads the trained ST-GCN + BiLSTM checkpoint pair and implements the
    DynamicGestureModel Protocol (update/reset), running the same dual-model
    "ST-GCN leads + BiLSTM confirms" logic validated live in
    dynamic_gestures/scripts/test_dynamic_model.py.

    Maintains one shared rolling buffer of (image_landmarks, world_landmarks)
    pairs sized for the larger of the two checkpoints' windows, resampling
    down to each model's own ``target_len`` before normalizing/packing -
    today both checkpoints happen to use target_len=20, but this does not
    assume they stay equal.
    """

    def __init__(self, config: DynamicModelLoaderConfig | None = None) -> None:
        self.config = config or DynamicModelLoaderConfig()
        self.device = self._resolve_device(self.config.device)

        stgcn_path = Path(self.config.stgcn_checkpoint_path)
        bilstm_path = Path(self.config.bilstm_checkpoint_path)
        for path in (stgcn_path, bilstm_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Dynamic gesture checkpoint not found at {path}. Train it with "
                    "dynamic_gestures/scripts/train_stgcn_model.py / train_dynamic_model.py, "
                    "or point DynamicRecognizerConfig elsewhere."
                )

        stgcn_checkpoint = torch.load(stgcn_path, map_location=self.device, weights_only=False)
        if stgcn_checkpoint.get("model_type") != "STGCN":
            raise ValueError(f"{stgcn_path} is not an STGCN checkpoint.")
        bilstm_checkpoint = torch.load(bilstm_path, map_location=self.device, weights_only=False)
        if bilstm_checkpoint.get("model_type") != "BiLSTMGestureClassifier":
            raise ValueError(f"{bilstm_path} is not a BiLSTMGestureClassifier checkpoint.")

        self._class_names: list[str] = list(bilstm_checkpoint["class_names"])
        if list(stgcn_checkpoint["class_names"]) != self._class_names:
            raise ValueError(
                f"Class name mismatch between {stgcn_path} and {bilstm_path}; "
                "the ensemble requires both checkpoints to share the same class_names order."
            )

        adjacency = np.asarray(stgcn_checkpoint["adjacency"], dtype=np.float32)
        self._stgcn_model = _STGCN(
            in_channels=stgcn_checkpoint["in_channels"],
            num_classes=stgcn_checkpoint["num_classes"],
            adjacency=adjacency,
            base_channels=stgcn_checkpoint["base_channels"],
            t_kernel=stgcn_checkpoint["t_kernel"],
            dropout=stgcn_checkpoint["dropout"],
        ).to(self.device)
        self._stgcn_model.load_state_dict(stgcn_checkpoint["model_state_dict"])
        self._stgcn_model.eval()
        self._stgcn_target_len = int(stgcn_checkpoint["target_len"])

        self._bilstm_model = _BiLSTMGestureClassifier(
            input_dim=bilstm_checkpoint["input_dim"],
            hidden_size=bilstm_checkpoint["hidden_size"],
            num_layers=bilstm_checkpoint["num_layers"],
            num_classes=bilstm_checkpoint["num_classes"],
            dropout=bilstm_checkpoint["dropout"],
        ).to(self.device)
        self._bilstm_model.load_state_dict(bilstm_checkpoint["model_state_dict"])
        self._bilstm_model.eval()
        self._bilstm_target_len = int(bilstm_checkpoint["target_len"])
        self._bilstm_add_velocity = bool(bilstm_checkpoint["add_velocity"])

        self._window = max(self._stgcn_target_len, self._bilstm_target_len)
        self._buffer: SequenceBuffer[tuple[np.ndarray, np.ndarray]] = SequenceBuffer(
            SequenceBufferConfig(max_length=self._window)
        )
        smoothing_config = ProbabilitySmoothingConfig(alpha=self.config.smoothing_alpha)
        self._stgcn_smoother = ProbabilitySmoother(smoothing_config)
        self._bilstm_smoother = ProbabilitySmoother(smoothing_config)
        # Discrete-gesture refractory: after emitting a one-shot gesture, stay
        # silent until the hand returns to rest (a frame the ensemble reads as
        # baseline/no-gesture). Without this, the *return stroke* of a swipe -
        # the hand moving back to centre - is itself a real motion the model
        # classifies as the OPPOSITE swipe, causing back-and-forth selection
        # "rollbacks". Keyed on actual hand state, not a timer.
        self._ready_to_fire = True
        # Directional lockout state: last active direction + its timestamp, to
        # suppress the opposite gesture during the hand's return stroke.
        self._last_dir_label: str | None = None
        self._last_dir_ts: float = 0.0
        logger.info(
            "Loaded dynamic gesture ensemble: ST-GCN %s (lead, window=%d) + BiLSTM %s "
            "(confirm, window=%d), classes=%s, on %s.",
            stgcn_path,
            self._stgcn_target_len,
            bilstm_path,
            self._bilstm_target_len,
            self._class_names,
            self.device,
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "cpu":
            return torch.device("cpu")
        if device == "cuda":
            return torch.device("cuda")
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def reset(self) -> None:
        self._buffer.clear()
        self._stgcn_smoother.reset()
        self._bilstm_smoother.reset()
        # Hand lost / paused = a fresh session: let the next gesture fire.
        self._ready_to_fire = True
        self._last_dir_label = None
        self._last_dir_ts = 0.0

    def update(self, packet: LandmarkPacket) -> GestureResult | None:
        if not packet.hand_detected:
            self.reset()
            return None

        image_frame = landmarks_to_array(packet.landmarks)
        world_frame = landmarks_to_array(packet.world_landmarks)
        if image_frame is None or world_frame is None:
            self.reset()
            return None

        self._buffer.append((image_frame, world_frame))
        if not self._buffer.is_full():
            return None

        entries = self._buffer.as_list()
        image_window = np.stack([image for image, _ in entries])
        world_window = np.stack([world for _, world in entries])

        stgcn_input = resample_sequence(world_window, self._stgcn_target_len)
        stgcn_norm = normalize_sequence(stgcn_input)
        bilstm_input = resample_sequence(image_window, self._bilstm_target_len)
        bilstm_norm = normalize_sequence(bilstm_input)
        if stgcn_norm is None or bilstm_norm is None:
            return None

        stgcn_features = pack_stgcn_features(stgcn_norm)[np.newaxis, ..., np.newaxis]
        bilstm_features = pack_bilstm_features(bilstm_norm, self._bilstm_add_velocity)[np.newaxis, ...]

        with torch.no_grad():
            stgcn_logits = self._stgcn_model(torch.from_numpy(stgcn_features).to(self.device))
            stgcn_probs = torch.softmax(stgcn_logits, dim=-1)[0].cpu().numpy()
            bilstm_logits = self._bilstm_model(torch.from_numpy(bilstm_features).to(self.device))
            bilstm_probs = torch.softmax(bilstm_logits, dim=-1)[0].cpu().numpy()

        stgcn_probs = self._stgcn_smoother.smooth(stgcn_probs)
        bilstm_probs = self._bilstm_smoother.smooth(bilstm_probs)

        label = _ensemble_decision(
            stgcn_probs, bilstm_probs, self._class_names, self.config.min_confidence, self.config.veto_floor
        )
        if label is None:
            # Hand at rest / no confident gesture: re-arm so the next discrete
            # gesture can fire. This is the boundary that ends one swipe's
            # refractory window.
            self._ready_to_fire = True
            return None

        # Directional lockout: if this label is the opposite of a direction that
        # was active within directional_lockout_seconds, it is the return stroke
        # of that gesture - suppress it (without re-anchoring the lockout, so the
        # lockout stays measured from the real direction). Otherwise record it as
        # the active direction. Lets both directions of a pair stay enabled
        # (SWIPE_LEFT/RIGHT, ROTATE_CW/CCW, SQUEEZE/RELEASE) without rollbacks.
        opposite = _OPPOSITE_LABELS.get(label)
        if opposite is not None:
            ts = packet.timestamp
            if (
                self._last_dir_label == opposite
                and ts - self._last_dir_ts < self.config.directional_lockout_seconds
            ):
                return None
            self._last_dir_label = label
            self._last_dir_ts = ts

        try:
            gesture_name = GestureName(label)
        except ValueError:
            return None

        confidence = float(stgcn_probs[self._class_names.index(label)])
        result = GestureResult(
            name=gesture_name,
            confidence=confidence,
            source=RecognitionSource.DYNAMIC_MODEL,
            timestamp=packet.timestamp,
        )

        # Continuous gestures (ROTATE) must emit every frame while held - never
        # gate them on the refractory, and don't let them disarm it.
        if label in _CONTINUOUS_LABELS:
            return result

        # Discrete gesture: emit once, then stay silent until a rest re-arms us
        # (above). Suppresses the swipe's return-stroke and any held repeat.
        if not self._ready_to_fire:
            return None
        self._ready_to_fire = False
        return result


def load_dynamic_model(config: DynamicModelLoaderConfig | None = None) -> EnsembleDynamicGestureModel | None:
    """Best-effort loader: returns None (dynamic recognition stays disabled,
    same as today) if either checkpoint is missing or unreadable, instead of
    crashing the whole app - static/cursor recognition must keep working
    regardless of whether dynamic-gesture checkpoints are present."""
    try:
        return EnsembleDynamicGestureModel(config)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.warning("Dynamic gesture model not loaded: %s", exc)
        return None
