from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# dynamic_gestures/scripts/<file>.py -> parents[2] is the repository root (home of
# the ``oil_gestures`` runtime package); SCRIPTS_DIR enables the sibling-script
# imports below (process/train modules live next to this file).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
for extra_path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from oil_gestures.core.enums import GestureName  # noqa: E402

# Reuse the exact training-time model definition, preprocessing and metric helpers
# so the verification path matches how the checkpoint was produced.
from process_dynamic_dataset import (  # noqa: E402
    LABEL_TO_ID,
    pack_bilstm_features,
    pack_stgcn_features,
    process_file,
)
from train_stgcn_model import STGCN  # noqa: E402
from train_dynamic_model import (  # noqa: E402
    TORCH_AVAILABLE,
    BiLSTMGestureClassifier,
    compute_classification_metrics,
    compute_confusion_matrix,
    resolve_path,
)

if TORCH_AVAILABLE:
    import torch

# Canonical dynamic-gesture vocabulary, taken from the shared GestureName enum
# (the contract every recognition subsystem agrees on). IDLE is the "no gesture"
# baseline class the dynamic model is trained against.
BASELINE_LABEL = GestureName.IDLE.value
DYNAMIC_GESTURE_LABELS = [
    GestureName.POINTING_INDEX.value,
    GestureName.SQUEEZE.value,
    GestureName.RELEASE.value,
    GestureName.ROTATE_CLOCKWISE.value,
    GestureName.ROTATE_COUNTERCLOCKWISE.value,
    GestureName.SWIPE_LEFT.value,
    GestureName.SWIPE_RIGHT.value,
]
EXPECTED_MODEL_LABELS = [BASELINE_LABEL, *DYNAMIC_GESTURE_LABELS]

GESTURE_DESCRIPTIONS = {
    "IDLE": "No intentional motion (resting hand / baseline)",
    "POINTING_INDEX": "Index finger extended and held",
    "SQUEEZE": "Hand closing into a grab",
    "RELEASE": "Hand opening from a grab",
    "ROTATE_CLOCKWISE": "Hand rotating clockwise",
    "ROTATE_COUNTERCLOCKWISE": "Hand rotating counter-clockwise",
    "SWIPE_LEFT": "Hand moving to the left",
    "SWIPE_RIGHT": "Hand moving to the right",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a trained dynamic-gesture model: list the gestures it recognizes "
        "and report how well it recognizes each one."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="dynamic_gestures/models/dynamic_bilstm.pt",
        help="Path to the trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dynamic_gestures/data/processed/dynamic_gestures_v1.pt",
        help="Processed dataset used to evaluate per-gesture recognition.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which dataset split to evaluate on.",
    )
    parser.add_argument(
        "--npz",
        type=str,
        default=None,
        help="Run a single raw recording (*.npz) through the model and show the prediction.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to show for a single --npz recording.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used while scoring the dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--no-eval",
        dest="evaluate_dataset",
        action="store_false",
        default=True,
        help="Only print the gesture vocabulary and model metadata; skip dataset evaluation.",
    )
    return parser.parse_args(argv)


def resolve_device(device_arg: str) -> "torch.device":
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested via --device cuda but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_checkpoint(checkpoint_path: Path, device: "torch.device") -> tuple["torch.nn.Module", dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BiLSTMGestureClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_model(checkpoint_path: Path, device: "torch.device") -> tuple["torch.nn.Module", dict[str, Any], str]:
    """Load either a BiLSTM or an ST-GCN checkpoint, auto-detected by model_type."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") == "STGCN":
        adjacency = np.asarray(checkpoint["adjacency"], dtype=np.float32)
        model = STGCN(
            in_channels=checkpoint["in_channels"],
            num_classes=checkpoint["num_classes"],
            adjacency=adjacency,
            base_channels=checkpoint["base_channels"],
            t_kernel=checkpoint["t_kernel"],
            dropout=checkpoint["dropout"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint, "stgcn"
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    return model, checkpoint, "bilstm"


def predict_logits(
    model: "torch.nn.Module", features: "torch.Tensor", device: "torch.device", batch_size: int
) -> np.ndarray:
    logits_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = features[start : start + batch_size].to(device)
            logits_chunks.append(model(batch).detach().cpu().numpy())
    if not logits_chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(logits_chunks, axis=0)


def print_header(title: str) -> None:
    print("=" * 64)
    print(f" {title}")
    print("=" * 64)


def print_model_summary(checkpoint_path: Path, checkpoint: dict[str, Any], device: "torch.device", kind: str) -> None:
    print_header("Dynamic gesture model — verification")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Model type : {checkpoint.get('model_type', 'unknown')}")
    print(f"Device     : {device}")
    if kind == "stgcn":
        print(
            f"Network    : base_channels={checkpoint['base_channels']}, "
            f"t_kernel={checkpoint['t_kernel']}, dropout={checkpoint['dropout']}"
        )
        print(
            f"Input      : target_len={checkpoint['target_len']}, channels={checkpoint['in_channels']}, "
            f"nodes={checkpoint['num_nodes']}, node_features={checkpoint.get('node_features')}"
        )
    else:
        print(
            f"Network    : hidden_size={checkpoint['hidden_size']}, "
            f"num_layers={checkpoint['num_layers']}, dropout={checkpoint['dropout']}"
        )
        print(
            f"Input      : target_len={checkpoint['target_len']}, "
            f"feature_dim={checkpoint['feature_dim']}, add_velocity={checkpoint['add_velocity']}"
        )
    best_epoch = checkpoint.get("best_epoch")
    best_f1 = checkpoint.get("best_val_macro_f1")
    if best_epoch is not None and best_f1 is not None:
        print(f"Trained    : best_epoch={best_epoch}, best_val_macro_f1={best_f1:.4f}")
    test_metrics = checkpoint.get("test_metrics")
    if isinstance(test_metrics, dict) and "accuracy" in test_metrics:
        print(
            f"Saved test : accuracy={test_metrics['accuracy']:.4f}, "
            f"macro_f1={test_metrics.get('macro_f1', float('nan')):.4f}"
        )
    print()


def print_gesture_vocabulary(class_names: list[str]) -> None:
    print("Recognized gestures (vocabulary the model can output):")
    print(f"{'id':>3}  {'name':<26} {'kind':<9} {'contract':<8} description")
    print("-" * 64)
    for class_id, name in enumerate(class_names):
        if name == BASELINE_LABEL:
            kind = "baseline"
        elif name in DYNAMIC_GESTURE_LABELS:
            kind = "dynamic"
        else:
            kind = "?"
        in_contract = "ok" if name in EXPECTED_MODEL_LABELS else "UNKNOWN"
        description = GESTURE_DESCRIPTIONS.get(name, "")
        print(f"{class_id:>3}  {name:<26} {kind:<9} {in_contract:<8} {description}")
    print()

    model_label_set = set(class_names)
    expected_set = set(EXPECTED_MODEL_LABELS)
    missing = [label for label in EXPECTED_MODEL_LABELS if label not in model_label_set]
    extra = sorted(model_label_set - expected_set)
    if not missing and not extra:
        print(f"Contract check: OK — model covers all {len(DYNAMIC_GESTURE_LABELS)} dynamic gestures + IDLE baseline.")
    else:
        if missing:
            print(f"Contract check: WARNING — expected gestures missing from the model: {missing}")
        if extra:
            print(f"Contract check: WARNING — model emits gestures not in the contract: {extra}")
    print()


def select_split(dataset: dict[str, Any], split: str, kind: str) -> tuple["torch.Tensor", "torch.Tensor"]:
    suffix = "_stgcn" if kind == "stgcn" else ""
    if split == "all":
        features = torch.cat(
            [dataset[f"X_train{suffix}"], dataset[f"X_val{suffix}"], dataset[f"X_test{suffix}"]], dim=0
        )
        labels = torch.cat([dataset["y_train"], dataset["y_val"], dataset["y_test"]], dim=0)
        return features, labels
    return dataset[f"X_{split}{suffix}"], dataset[f"y_{split}"]


def print_confusion_matrix(confusion: np.ndarray, class_names: list[str]) -> None:
    num_classes = len(class_names)
    print("Confusion matrix (rows = true gesture, columns = predicted):")
    header = "true \\ pred".ljust(26) + "".join(f"{class_id:>6}" for class_id in range(num_classes))
    print(header)
    for class_id, name in enumerate(class_names):
        row = f"{class_id:>2} {name:<23}" + "".join(f"{int(value):>6}" for value in confusion[class_id])
        print(row)
    print()
    print("Column legend: " + ", ".join(f"{class_id}={name}" for class_id, name in enumerate(class_names)))
    print()


def evaluate_dataset(
    model: "torch.nn.Module",
    dataset: dict[str, Any],
    class_names: list[str],
    split: str,
    device: "torch.device",
    batch_size: int,
    kind: str,
) -> None:
    features, labels = select_split(dataset, split, kind)
    if features.shape[0] == 0:
        print(f"Split '{split}' has no samples; nothing to evaluate.")
        return

    logits = predict_logits(model, features, device, batch_size)
    predictions = np.argmax(logits, axis=-1)
    y_true = labels.numpy()
    num_classes = len(class_names)
    confusion = compute_confusion_matrix(y_true, predictions, num_classes)
    metrics = compute_classification_metrics(confusion)
    support = confusion.sum(axis=1)

    print(f"Per-gesture recognition on '{split}' split ({features.shape[0]} samples):")
    print(f"{'id':>3}  {'gesture':<26} {'precision':>9} {'recall':>8} {'f1':>8} {'support':>8}")
    print("-" * 64)
    for class_id, name in enumerate(class_names):
        print(
            f"{class_id:>3}  {name:<26} "
            f"{metrics['precision_per_class'][class_id]:>9.3f} "
            f"{metrics['recall_per_class'][class_id]:>8.3f} "
            f"{metrics['f1_per_class'][class_id]:>8.3f} "
            f"{int(support[class_id]):>8}"
        )
    print("-" * 64)
    print(f"Overall accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1         : {metrics['macro_f1']:.4f}")
    print()
    print_confusion_matrix(confusion, class_names)


def check_single_recording(
    model: "torch.nn.Module",
    checkpoint: dict[str, Any],
    npz_path: Path,
    device: "torch.device",
    top_k: int,
    kind: str,
) -> int:
    class_names: list[str] = checkpoint["class_names"]
    sample, skip_reason = process_file(npz_path, target_len=checkpoint["target_len"])
    if sample is None:
        print(f"Could not process {npz_path}: {skip_reason}")
        return 1

    if kind == "stgcn":
        packed = pack_stgcn_features(sample.norm_world)[np.newaxis, ..., np.newaxis]
        features = torch.from_numpy(packed.astype(np.float32))
    else:
        features = torch.from_numpy(
            pack_bilstm_features(sample.norm_image, checkpoint["add_velocity"])
        ).unsqueeze(0)
    logits = predict_logits(model, features, device, batch_size=1)[0]
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    ranking = np.argsort(probabilities)[::-1]

    print_header("Single recording check")
    print(f"File        : {npz_path}")
    print(f"True label  : {sample.label}")
    predicted = class_names[int(ranking[0])]
    verdict = "correct" if predicted == sample.label else "WRONG"
    print(f"Predicted   : {predicted}  ({probabilities[ranking[0]] * 100:.1f}%)  -> {verdict}")
    print()
    top_k = max(1, min(top_k, len(class_names)))
    print(f"Top-{top_k} predictions:")
    for rank in range(top_k):
        class_id = int(ranking[rank])
        print(f"  {rank + 1}. {class_names[class_id]:<26} {probabilities[class_id] * 100:6.2f}%")
    return 0


def main() -> int:
    if not TORCH_AVAILABLE:
        raise SystemExit("PyTorch is required. Install it with: python -m pip install torch")

    args = parse_args()
    device = resolve_device(args.device)

    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise SystemExit(
            f"Checkpoint not found at: {checkpoint_path}. Train the model first with "
            "scripts/train_dynamic_model.py."
        )

    model, checkpoint, kind = load_model(checkpoint_path, device)
    class_names: list[str] = checkpoint["class_names"]

    print_model_summary(checkpoint_path, checkpoint, device, kind)
    print_gesture_vocabulary(class_names)

    if args.npz is not None:
        npz_path = resolve_path(args.npz)
        if not npz_path.is_file():
            raise SystemExit(f"Recording not found at: {npz_path}")
        return check_single_recording(model, checkpoint, npz_path, device, args.top_k, kind)

    if not args.evaluate_dataset:
        return 0

    data_path = resolve_path(args.data)
    if not data_path.is_file():
        print(
            f"Processed dataset not found at: {data_path}. "
            "Skipping evaluation (run scripts/process_dynamic_dataset.py to enable it)."
        )
        return 0

    dataset = torch.load(data_path, map_location="cpu", weights_only=False)
    dataset_class_names: list[str] = dataset["class_names"]
    if dataset_class_names != class_names:
        print(
            "WARNING: dataset class order differs from the checkpoint; "
            "metrics use the checkpoint's class names.\n"
        )
    if kind == "stgcn" and "X_test_stgcn" not in dataset:
        print(
            "Dataset has no ST-GCN tensors; re-run scripts/process_dynamic_dataset.py to enable "
            "evaluation for this checkpoint."
        )
        return 0
    evaluate_dataset(model, dataset, class_names, args.split, device, args.batch_size, kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
