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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oil_gestures.vision.landmark_utils import LANDMARK_INDEX  # noqa: E402

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
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

WRIST_INDEX = LANDMARK_INDEX["WRIST"]
MIDDLE_MCP_INDEX = LANDMARK_INDEX["MIDDLE_MCP"]
EXPECTED_LANDMARK_COUNT = 21
EXPECTED_LANDMARK_DIMS = 3
MIN_SCALE = 1e-4

MANIFEST_COLUMNS = (
    "path",
    "label",
    "label_id",
    "split",
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
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed/dynamic_gestures_v1.pt")
    parser.add_argument("--manifest", type=str, default="data/processed/dynamic_gestures_manifest.csv")
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
    return parser.parse_args(argv)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


@dataclass
class ProcessedSample:
    path: Path
    features: np.ndarray
    label: str
    label_id: int
    raw_seq_len: int
    measured_fps: float
    created_at: str


def load_raw_npz(path: Path) -> dict[str, Any] | None:
    try:
        with np.load(path, allow_pickle=True) as data:
            if "image_landmarks" not in data.files or "label" not in data.files:
                warnings.warn(f"Skipping {path}: missing image_landmarks or label.")
                return None
            image_landmarks = np.asarray(data["image_landmarks"])
            label_value = data["label"]
            label = str(label_value.item()) if hasattr(label_value, "item") else str(label_value)
            raw_seq_len = int(data["sequence_length"]) if "sequence_length" in data.files else int(image_landmarks.shape[0])
            measured_fps = float(data["measured_fps"]) if "measured_fps" in data.files else 0.0
            created_at = str(data["created_at"].item()) if "created_at" in data.files else ""
            return {
                "image_landmarks": image_landmarks,
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


def normalize_sequence(landmarks: np.ndarray, min_scale: float = MIN_SCALE) -> np.ndarray | None:
    wrist = landmarks[:, WRIST_INDEX : WRIST_INDEX + 1, :]
    centered = landmarks - wrist
    distances = np.linalg.norm(centered[:, MIDDLE_MCP_INDEX, :], axis=-1)
    scale = float(np.mean(distances))
    if scale < min_scale:
        return None
    return (centered / scale).astype(np.float32)


def add_velocity_features(features: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(features)
    velocity[1:] = features[1:] - features[:-1]
    return np.concatenate([features, velocity], axis=-1)


def process_file(path: Path, target_len: int, add_velocity: bool) -> tuple[ProcessedSample | None, str | None]:
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

    resampled = resample_sequence(image_landmarks, target_len)
    normalized = normalize_sequence(resampled)
    if normalized is None:
        warnings.warn(f"Skipping {path}: hand scale too small after normalization.")
        return None, "scale_too_small"

    flattened = normalized.reshape(target_len, -1)
    features = add_velocity_features(flattened) if add_velocity else flattened

    sample = ProcessedSample(
        path=path,
        features=features.astype(np.float32),
        label=label,
        label_id=LABEL_TO_ID[label],
        raw_seq_len=raw["raw_seq_len"],
        measured_fps=raw["measured_fps"],
        created_at=raw["created_at"],
    )
    return sample, None


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
    samples: list[ProcessedSample], splits: list[str], split_name: str, feature_dim: int, target_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = [sample for sample, split in zip(samples, splits) if split == split_name]
    if not selected:
        return (
            torch.zeros((0, target_len, feature_dim), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
        )
    features = np.stack([sample.features for sample in selected]).astype(np.float32)
    labels = np.array([sample.label_id for sample in selected], dtype=np.int64)
    return torch.from_numpy(features), torch.from_numpy(labels)


def write_manifest(
    manifest_path: Path, samples: list[ProcessedSample], splits: list[str], target_len: int, feature_dim: int
) -> None:
    rows = [
        {
            "path": str(sample.path),
            "label": sample.label,
            "label_id": sample.label_id,
            "split": split_name,
            "raw_seq_len": sample.raw_seq_len,
            "target_len": target_len,
            "measured_fps": sample.measured_fps,
            "feature_dim": feature_dim,
            "created_at": sample.created_at,
        }
        for sample, split_name in zip(samples, splits)
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
        sample, skip_reason = process_file(path, args.target_len, args.add_velocity)
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

    feature_dim = samples[0].features.shape[1]
    splits = build_splits(samples, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    X_train, y_train = stack_split(samples, splits, "train", feature_dim, args.target_len)
    X_val, y_val = stack_split(samples, splits, "val", feature_dim, args.target_len)
    X_test, y_test = stack_split(samples, splits, "test", feature_dim, args.target_len)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "class_names": [ID_TO_LABEL[label_id] for label_id in sorted(ID_TO_LABEL)],
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL,
            "target_len": args.target_len,
            "feature_dim": feature_dim,
            "normalization": {
                "method": "wrist_center_middle_mcp_scale",
                "wrist_index": WRIST_INDEX,
                "middle_mcp_index": MIDDLE_MCP_INDEX,
                "min_scale": MIN_SCALE,
            },
            "add_velocity": args.add_velocity,
        },
        output_path,
    )

    write_manifest(manifest_path, samples, splits, args.target_len, feature_dim)

    samples_per_class: dict[str, int] = {}
    for sample in samples:
        samples_per_class[sample.label] = samples_per_class.get(sample.label, 0) + 1

    print(f"Loaded samples: {len(samples)}")
    print(f"Skipped samples: {skipped_count} {skip_reasons if skip_reasons else ''}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Train/val/test sizes: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")
    print(f"Output .pt path: {output_path}")
    print(f"Manifest path: {manifest_path}")
    print(f"X_train shape: {tuple(X_train.shape)}")
    print(f"feature_dim: {feature_dim}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())1211