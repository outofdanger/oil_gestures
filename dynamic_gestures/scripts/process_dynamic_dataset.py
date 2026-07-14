from __future__ import annotations

import argparse
import csv
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:
    torch = None

# dynamic_gestures/scripts/<file>.py -> parents[2] is the repository root, which
# is where the ``oil_gestures`` runtime package lives. Relative paths passed on the
# CLI are resolved against this root by ``resolve_path`` below.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oil_gestures.vision.landmark_utils import HAND_EDGES, LANDMARK_INDEX  # noqa: E402

try:
    import pandas as pd
except Exception:
    pd = None

LABEL_TO_ID = {
    "IDLE": 0,
    "POINTING_INDEX": 1,
    "SQUEEZE": 2,
    "RELEASE": 3,
    "ROTATE_CLOCKWISE": 4,
    "ROTATE_COUNTERCLOCKWISE": 5,
    "SWIPE_LEFT": 6,
    "SWIPE_RIGHT": 7,
    "TRANSITION": 8,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

# TRANSITION has no raw recordings of its own: it is synthesized by time-reversing
# swipe clips. A reversed SWIPE_LEFT is the hand travelling back to rest with the
# residual "left swipe" pose - exactly the return stroke that live recognition
# otherwise misreads as SWIPE_RIGHT. Only swipes are reversed: a reversed ROTATE_CW
# is a genuine ROTATE_CCW and a reversed SQUEEZE is close to a real RELEASE, so
# reversing those would poison their true classes.
TRANSITION_LABEL = "TRANSITION"
REVERSIBLE_LABELS = ("SWIPE_LEFT", "SWIPE_RIGHT")

WRIST_INDEX = LANDMARK_INDEX["WRIST"]
MIDDLE_MCP_INDEX = LANDMARK_INDEX["MIDDLE_MCP"]
EXPECTED_LANDMARK_COUNT = 21
EXPECTED_LANDMARK_DIMS = 3
MIN_SCALE = 1e-4

# Horizontal-mirror augmentation: flipping the x axis turns a gesture into its
# left/right mirror image, so direction-bearing labels must be swapped to stay
# correct. Labels not listed here map to themselves (a flipped IDLE is still IDLE).
LR_FLIP_LABELS = {
    "SWIPE_LEFT": "SWIPE_RIGHT",
    "SWIPE_RIGHT": "SWIPE_LEFT",
    "ROTATE_CLOCKWISE": "ROTATE_COUNTERCLOCKWISE",
    "ROTATE_COUNTERCLOCKWISE": "ROTATE_CLOCKWISE",
}

MANIFEST_COLUMNS = (
    "path",
    "label",
    "label_id",
    "split",
    "is_mirrored",
    "is_reversed",
    "raw_seq_len",
    "target_len",
    "measured_fps",
    "feature_dim",
    "created_at",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process raw dynamic gesture landmark recordings into a PyTorch-ready dataset."
    )
    parser.add_argument("--input", type=str, default="dynamic_gestures/data/raw")
    parser.add_argument("--output", type=str, default="dynamic_gestures/data/processed/dynamic_gestures_v1.pt")
    parser.add_argument("--manifest", type=str, default="dynamic_gestures/data/processed/dynamic_gestures_manifest.csv")
    parser.add_argument("--target-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-velocity",
        dest="add_velocity",
        action="store_false",
        default=True,
        help="Disable velocity (frame-difference) features.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        default=True,
        help=(
            "Disable train-split data augmentation. Augmentation only ever touches the "
            "train split; val/test stay pristine for honest evaluation. Pass this to "
            "reproduce the un-augmented baseline."
        ),
    )
    parser.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        default=True,
        help="Disable horizontal-mirror (left/right swap) augmentation. No-op when --no-augment is set.",
    )
    parser.add_argument(
        "--no-transitions",
        dest="transitions",
        action="store_false",
        default=True,
        help=(
            "Disable the synthetic TRANSITION class (time-reversed swipes = the hand's "
            "return stroke). Unlike mirror augmentation, TRANSITION samples are generated "
            "in every split (each reversal stays in its source clip's split), because a "
            "class absent from val/test could not be measured at all."
        ),
    )
    parser.add_argument(
        "--exclude-labels",
        type=str,
        nargs="*",
        default=[],
        metavar="LABEL",
        help=(
            "Gesture labels to leave out of the built dataset (raw recordings are kept untouched). "
            "Remaining classes are renumbered to a contiguous 0..N-1 id space, so the model trains "
            "as a genuine (N)-class classifier. Example: --exclude-labels POINTING_INDEX"
        ),
    )
    return parser.parse_args(argv)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


@dataclass
class ProcessedSample:
    path: Path
    # Normalized landmark sequences kept un-flattened as (T, 21, 3) so the same
    # sample can be packed into both the flat BiLSTM layout and the graph
    # (C, T, V) ST-GCN layout, and mirrored in place after the split.
    norm_image: np.ndarray
    norm_world: np.ndarray
    label: str
    label_id: int
    raw_seq_len: int
    measured_fps: float
    created_at: str
    is_mirrored: bool = False
    is_reversed: bool = False


def load_raw_npz(path: Path) -> dict[str, Any] | None:
    try:
        with np.load(path, allow_pickle=True) as data:
            if "image_landmarks" not in data.files or "label" not in data.files:
                warnings.warn(f"Skipping {path}: missing image_landmarks or label.")
                return None
            image_landmarks = np.asarray(data["image_landmarks"])
            # world_landmarks are metric, hand-relative 3D coordinates that are far
            # less perspective-distorted than image_landmarks; they are the node
            # features for the ST-GCN branch. Fall back to image landmarks if a
            # recording predates them so older data still loads.
            if "world_landmarks" in data.files:
                world_landmarks = np.asarray(data["world_landmarks"])
            else:
                world_landmarks = image_landmarks
            label_value = data["label"]
            label = str(label_value.item()) if hasattr(label_value, "item") else str(label_value)
            raw_seq_len = int(data["sequence_length"]) if "sequence_length" in data.files else int(image_landmarks.shape[0])
            measured_fps = float(data["measured_fps"]) if "measured_fps" in data.files else 0.0
            created_at = str(data["created_at"].item()) if "created_at" in data.files else ""
            return {
                "image_landmarks": image_landmarks,
                "world_landmarks": world_landmarks,
                "label": label,
                "raw_seq_len": raw_seq_len,
                "measured_fps": measured_fps,
                "created_at": created_at,
            }
    except Exception as exc:
        warnings.warn(f"Skipping {path}: failed to load ({exc}).")
        return None


def is_valid_shape(image_landmarks: np.ndarray) -> bool:
    return (
        image_landmarks.ndim == 3
        and image_landmarks.shape[1] == EXPECTED_LANDMARK_COUNT
        and image_landmarks.shape[2] == EXPECTED_LANDMARK_DIMS
    )


def resample_sequence(landmarks: np.ndarray, target_len: int) -> np.ndarray:
    source_len = landmarks.shape[0]
    if source_len == target_len:
        return landmarks.astype(np.float32)
    if source_len == 1:
        return np.repeat(landmarks, target_len, axis=0).astype(np.float32)

    source_t = np.linspace(0.0, 1.0, source_len)
    target_t = np.linspace(0.0, 1.0, target_len)
    flat = landmarks.reshape(source_len, -1)
    resampled = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        resampled[:, col] = np.interp(target_t, source_t, flat[:, col])
    return resampled.reshape(target_len, landmarks.shape[1], landmarks.shape[2])


def normalize_sequence(
    landmarks: np.ndarray, min_scale: float = MIN_SCALE, anchor: str = "per_frame"
) -> np.ndarray | None:
    wrist = landmarks[:, WRIST_INDEX : WRIST_INDEX + 1, :]
    relative = landmarks - wrist
    distances = np.linalg.norm(relative[:, MIDDLE_MCP_INDEX, :], axis=-1)
    scale = float(np.mean(distances))
    if scale < min_scale:
        return None
    if anchor == "per_frame":
        # Legacy mode: every frame re-centered at its own wrist. Hand pose only;
        # global translation of the hand is erased.
        return (relative / scale).astype(np.float32)
    if anchor != "first_frame":
        raise ValueError(f"Unknown anchor mode '{anchor}'; expected 'per_frame' or 'first_frame'.")
    # Anchor the whole window at the FIRST frame's wrist: per-frame pose keeps its
    # global trajectory, so translation-driven gestures (swipes, drifts) remain
    # visible after normalization. Scale stays the per-frame hand size average.
    centered = landmarks - wrist[:1]
    return (centered / scale).astype(np.float32)


def add_velocity_features(features: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(features)
    velocity[1:] = features[1:] - features[:-1]
    return np.concatenate([features, velocity], axis=-1)


def _resample_normalize(landmarks: np.ndarray, target_len: int) -> np.ndarray | None:
    if not is_valid_shape(landmarks) or not np.isfinite(landmarks).all():
        return None
    resampled = resample_sequence(landmarks, target_len)
    return normalize_sequence(resampled)


def process_file(path: Path, target_len: int) -> tuple[ProcessedSample | None, str | None]:
    raw = load_raw_npz(path)
    if raw is None:
        return None, "load_failed"

    label = raw["label"]
    if label not in LABEL_TO_ID:
        return None, "unknown_label"

    image_landmarks = raw["image_landmarks"]
    if not is_valid_shape(image_landmarks):
        warnings.warn(f"Skipping {path}: invalid shape {image_landmarks.shape}, expected (T, 21, 3).")
        return None, "invalid_shape"
    if not np.isfinite(image_landmarks).all():
        warnings.warn(f"Skipping {path}: non-finite values in image_landmarks.")
        return None, "non_finite"

    norm_image = _resample_normalize(image_landmarks, target_len)
    norm_world = _resample_normalize(raw["world_landmarks"], target_len)
    if norm_image is None or norm_world is None:
        warnings.warn(f"Skipping {path}: hand scale too small or invalid after normalization.")
        return None, "scale_too_small"

    sample = ProcessedSample(
        path=path,
        norm_image=norm_image,
        norm_world=norm_world,
        label=label,
        label_id=LABEL_TO_ID[label],
        raw_seq_len=raw["raw_seq_len"],
        measured_fps=raw["measured_fps"],
        created_at=raw["created_at"],
    )
    return sample, None


def mirror_sample(sample: ProcessedSample) -> ProcessedSample:
    """Horizontally mirror a sample (negate x) and swap its left/right label."""
    mirrored_image = sample.norm_image.copy()
    mirrored_world = sample.norm_world.copy()
    mirrored_image[..., 0] *= -1.0
    mirrored_world[..., 0] *= -1.0
    mirrored_label = LR_FLIP_LABELS.get(sample.label, sample.label)
    return ProcessedSample(
        path=sample.path,
        norm_image=mirrored_image,
        norm_world=mirrored_world,
        label=mirrored_label,
        label_id=LABEL_TO_ID[mirrored_label],
        raw_seq_len=sample.raw_seq_len,
        measured_fps=sample.measured_fps,
        created_at=sample.created_at,
        is_mirrored=True,
        is_reversed=sample.is_reversed,
    )


def reversed_transition_sample(sample: ProcessedSample) -> ProcessedSample:
    """Time-reverse a swipe into a TRANSITION (return-stroke) sample.

    Reversing the already-normalized arrays equals normalizing a reversed raw
    clip: per-frame wrist centering is frame-local, the scale is a mean over
    frames (order-free), and linear resampling is time-symmetric. This would
    NOT hold for first_frame anchoring, where the anchor frame changes.
    """
    return ProcessedSample(
        path=sample.path,
        norm_image=sample.norm_image[::-1].copy(),
        norm_world=sample.norm_world[::-1].copy(),
        label=TRANSITION_LABEL,
        label_id=LABEL_TO_ID[TRANSITION_LABEL],
        raw_seq_len=sample.raw_seq_len,
        measured_fps=sample.measured_fps,
        created_at=sample.created_at,
        is_reversed=True,
    )


def pack_bilstm_features(norm_image: np.ndarray, add_velocity: bool) -> np.ndarray:
    """Flat per-frame feature layout (T, 63) or (T, 126) with velocity, for the BiLSTM."""
    flattened = norm_image.reshape(norm_image.shape[0], -1)
    features = add_velocity_features(flattened) if add_velocity else flattened
    return features.astype(np.float32)


def pack_stgcn_features(norm_world: np.ndarray) -> np.ndarray:
    """Graph layout (C, T, V) from (T, V, C) world landmarks, for ST-GCN."""
    return np.transpose(norm_world, (2, 0, 1)).astype(np.float32)


def build_normalized_adjacency(num_nodes: int, edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Symmetric normalized adjacency with self-loops: D^-1/2 (A + I) D^-1/2."""
    adjacency = np.eye(num_nodes, dtype=np.float32)
    for i, j in edges:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    degree = adjacency.sum(axis=1)
    d_inv_sqrt = np.zeros_like(degree)
    np.divide(1.0, np.sqrt(degree), out=d_inv_sqrt, where=degree > 0)
    normalizer = np.diag(d_inv_sqrt)
    return (normalizer @ adjacency @ normalizer).astype(np.float32)


def split_label_indices(indices: list[int], train_ratio: float, val_ratio: float, rng: np.random.Generator) -> dict[int, str]:
    shuffled = list(indices)
    rng.shuffle(shuffled)
    n = len(shuffled)
    assignment: dict[int, str] = {}

    if n < 3:
        for idx in shuffled:
            assignment[idx] = "train"
        return assignment

    n_train = int(round(n * train_ratio))
    n_train = max(1, min(n_train, n - 2))
    n_val = int(round(n * val_ratio))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val

    for idx in shuffled[:n_train]:
        assignment[idx] = "train"
    for idx in shuffled[n_train : n_train + n_val]:
        assignment[idx] = "val"
    for idx in shuffled[n_train + n_val :]:
        assignment[idx] = "test"
    return assignment


def build_splits(
    samples: list[ProcessedSample], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> list[str]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("train/val/test ratios must sum to a positive value.")
    train_ratio, val_ratio, test_ratio = (ratio / ratio_sum for ratio in (train_ratio, val_ratio, test_ratio))

    indices_by_label: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        indices_by_label.setdefault(sample.label, []).append(index)

    splits = [""] * len(samples)
    rng = np.random.default_rng(seed)
    for label in sorted(indices_by_label):
        indices = sorted(indices_by_label[label])
        if len(indices) < 3:
            warnings.warn(
                f"Label '{label}' has only {len(indices)} sample(s); putting all of them in train."
            )
        assignment = split_label_indices(indices, train_ratio, val_ratio, rng)
        for index, split_name in assignment.items():
            splits[index] = split_name
    return splits


def stack_split(
    selected: list[ProcessedSample],
    add_velocity: bool,
    feature_dim: int,
    stgcn_channels: int,
    target_len: int,
    num_nodes: int,
) -> dict[str, torch.Tensor]:
    """Pack a list of samples into both BiLSTM (N, T, F) and ST-GCN (N, C, T, V, M) tensors."""
    if not selected:
        return {
            "bilstm": torch.zeros((0, target_len, feature_dim), dtype=torch.float32),
            "stgcn": torch.zeros((0, stgcn_channels, target_len, num_nodes, 1), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        }
    bilstm = np.stack([pack_bilstm_features(s.norm_image, add_velocity) for s in selected]).astype(np.float32)
    # ST-GCN expects (N, C, T, V, M); M=1 (single hand). Add the trailing person axis.
    stgcn = np.stack([pack_stgcn_features(s.norm_world) for s in selected]).astype(np.float32)
    stgcn = stgcn[..., np.newaxis]
    labels = np.array([s.label_id for s in selected], dtype=np.int64)
    return {
        "bilstm": torch.from_numpy(bilstm),
        "stgcn": torch.from_numpy(stgcn),
        "labels": torch.from_numpy(labels),
    }


def write_manifest(
    manifest_path: Path,
    labeled_samples: list[tuple[ProcessedSample, str]],
    target_len: int,
    feature_dim: int,
) -> None:
    rows = [
        {
            "path": str(sample.path),
            "label": sample.label,
            "label_id": sample.label_id,
            "split": split_name,
            "is_mirrored": int(sample.is_mirrored),
            "is_reversed": int(sample.is_reversed),
            "raw_seq_len": sample.raw_seq_len,
            "target_len": target_len,
            "measured_fps": sample.measured_fps,
            "feature_dim": feature_dim,
            "created_at": sample.created_at,
        }
        for sample, split_name in labeled_samples
    ]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if pd is not None:
        pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
        return

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()

    if torch is None:
        raise SystemExit(
            "PyTorch is required to save the processed dataset. Install it with: "
            "python -m pip install torch"
        )

    input_dir = resolve_path(args.input)
    output_path = resolve_path(args.output)
    manifest_path = resolve_path(args.manifest)

    raw_files = sorted(input_dir.glob("*/*.npz"))

    samples: list[ProcessedSample] = []
    skip_reasons: dict[str, int] = {}
    for path in raw_files:
        sample, skip_reason = process_file(path, args.target_len)
        if sample is None:
            skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
            continue
        samples.append(sample)

    skipped_count = sum(skip_reasons.values())

    if not samples:
        print("No valid samples were processed. Nothing to save.")
        if skip_reasons:
            print(f"Skip reasons: {skip_reasons}")
        return 1

    exclude_labels = {label.strip().upper() for label in args.exclude_labels}
    unknown_excludes = sorted(exclude_labels - set(LABEL_TO_ID))
    if unknown_excludes:
        raise SystemExit(
            f"--exclude-labels got unknown label(s): {unknown_excludes}. Known labels: {sorted(LABEL_TO_ID)}"
        )

    # TRANSITION exists only as time-reversed swipes; when disabled (flag, explicit
    # exclude, or no swipe sources to reverse) it must also vanish from class_names,
    # or the model would train with a class that has zero samples.
    has_reversal_sources = any(
        sample.label in REVERSIBLE_LABELS and sample.label not in exclude_labels for sample in samples
    )
    transitions_enabled = args.transitions and TRANSITION_LABEL not in exclude_labels and has_reversal_sources
    if not transitions_enabled:
        exclude_labels.add(TRANSITION_LABEL)

    if exclude_labels:
        samples = [sample for sample in samples if sample.label not in exclude_labels]
        if not samples:
            print(f"All samples were excluded by --exclude-labels {sorted(exclude_labels)}. Nothing to save.")
            return 1

    # Contiguous label space over the labels that remain, ordered by their original id so the
    # class order stays stable. The model trains as a genuine len(class_names)-class classifier.
    remaining_labels = [
        label for label, _ in sorted(LABEL_TO_ID.items(), key=lambda kv: kv[1]) if label not in exclude_labels
    ]
    label_to_id = {label: index for index, label in enumerate(remaining_labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    class_names = remaining_labels

    feature_dim = pack_bilstm_features(samples[0].norm_image, args.add_velocity).shape[1]
    stgcn_channels = samples[0].norm_world.shape[2]
    num_nodes = EXPECTED_LANDMARK_COUNT

    splits = build_splits(samples, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    labeled_samples: list[tuple[ProcessedSample, str]] = list(zip(samples, splits))

    # Synthetic TRANSITION class: every swipe also contributes its time-reversed
    # return stroke. Generated in ALL splits (a new class must be measurable on
    # val/test), but each reversal inherits its source clip's split, so no clip's
    # information ever crosses a split boundary. Runs before mirroring so train
    # TRANSITION samples get mirrored like everything else.
    transition_sources = 0
    if transitions_enabled:
        reversed_samples: list[tuple[ProcessedSample, str]] = []
        for sample, split in labeled_samples:
            if sample.label in REVERSIBLE_LABELS:
                reversed_samples.append((reversed_transition_sample(sample), split))
        transition_sources = len(reversed_samples)
        labeled_samples.extend(reversed_samples)

    # Mirror augmentation only ever expands the train split, so val/test remain a
    # faithful, leakage-free measure of generalization.
    mirror_enabled = args.augment and args.mirror
    train_before = sum(1 for _, split in labeled_samples if split == "train")
    if mirror_enabled:
        mirrored: list[tuple[ProcessedSample, str]] = []
        for sample, split in labeled_samples:
            if split != "train":
                continue
            mirror = mirror_sample(sample)
            # A mirror can flip a kept label into an excluded one (e.g. excluding SWIPE_LEFT
            # while keeping SWIPE_RIGHT); drop those so excluded classes never reappear.
            if mirror.label in exclude_labels:
                continue
            mirrored.append((mirror, "train"))
        labeled_samples.extend(mirrored)
    train_after = sum(1 for _, split in labeled_samples if split == "train")

    # Assign the contiguous (possibly remapped) ids now that all originals + mirrors exist.
    for sample, _ in labeled_samples:
        sample.label_id = label_to_id[sample.label]

    def select(split_name: str) -> list[ProcessedSample]:
        return [sample for sample, split in labeled_samples if split == split_name]

    train = stack_split(select("train"), args.add_velocity, feature_dim, stgcn_channels, args.target_len, num_nodes)
    val = stack_split(select("val"), args.add_velocity, feature_dim, stgcn_channels, args.target_len, num_nodes)
    test = stack_split(select("test"), args.add_velocity, feature_dim, stgcn_channels, args.target_len, num_nodes)

    adjacency = torch.from_numpy(build_normalized_adjacency(num_nodes, HAND_EDGES))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # --- BiLSTM branch: flat per-frame features (N, T, feature_dim) ---
            "X_train": train["bilstm"],
            "y_train": train["labels"],
            "X_val": val["bilstm"],
            "y_val": val["labels"],
            "X_test": test["bilstm"],
            "y_test": test["labels"],
            # --- ST-GCN branch: graph layout (N, C, T, V, M) on world landmarks ---
            "X_train_stgcn": train["stgcn"],
            "X_val_stgcn": val["stgcn"],
            "X_test_stgcn": test["stgcn"],
            "adjacency": adjacency,
            "edges": [list(edge) for edge in HAND_EDGES],
            "stgcn_meta": {
                "layout": "N,C,T,V,M",
                "num_nodes": num_nodes,
                "num_channels": stgcn_channels,
                "num_persons": 1,
                "node_features": "world_landmarks",
            },
            "class_names": class_names,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "excluded_labels": sorted(exclude_labels),
            "target_len": args.target_len,
            "feature_dim": feature_dim,
            "normalization": {
                "method": "wrist_center_middle_mcp_scale",
                "wrist_index": WRIST_INDEX,
                "middle_mcp_index": MIDDLE_MCP_INDEX,
                "min_scale": MIN_SCALE,
            },
            "add_velocity": args.add_velocity,
            "augmentation": {
                "mirror": mirror_enabled,
                "train_size_before": train_before,
                "train_size_after": train_after,
                "transitions": transitions_enabled,
                "transition_sources": transition_sources,
            },
        },
        output_path,
    )

    write_manifest(manifest_path, labeled_samples, args.target_len, feature_dim)

    samples_per_class: dict[str, int] = {}
    for sample in samples:
        samples_per_class[sample.label] = samples_per_class.get(sample.label, 0) + 1

    print(f"Loaded samples: {len(samples)}")
    print(f"Skipped samples: {skipped_count} {skip_reasons if skip_reasons else ''}")
    if exclude_labels:
        print(f"Excluded labels (kept on disk, not trained): {sorted(exclude_labels)}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Augmentation: mirror={mirror_enabled} train {train_before} -> {train_after}")
    print(f"TRANSITION (reversed swipes): enabled={transitions_enabled} sources={transition_sources}")
    print(f"Train/val/test sizes: {train['labels'].shape[0]}/{val['labels'].shape[0]}/{test['labels'].shape[0]}")
    print(f"Output .pt path: {output_path}")
    print(f"Manifest path: {manifest_path}")
    print(f"BiLSTM X_train shape: {tuple(train['bilstm'].shape)} (feature_dim={feature_dim})")
    print(f"ST-GCN X_train shape: {tuple(train['stgcn'].shape)} (C,T,V,M; adjacency={tuple(adjacency.shape)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())