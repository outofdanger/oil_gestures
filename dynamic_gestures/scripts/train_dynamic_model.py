from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow ops that are not yet implemented on the MPS (Apple Silicon GPU) backend to
# transparently fall back to CPU instead of raising. Must be set before torch import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# dynamic_gestures/scripts/<file>.py -> parents[2] is the repository root (home of
# the ``oil_gestures`` runtime package). CLI-relative paths resolve against it.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oil_gestures.core.logger import get_logger  # noqa: E402

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

logger = get_logger(__name__)


# oil_gestures/gestures/dynamic/dynamic_model.py only defines a runtime Protocol
# (DynamicGestureModel) for inference-time consumers; it has no trainable nn.Module
# to reuse, so the BiLSTM classifier is defined locally here.
if TORCH_AVAILABLE:

    class BiLSTMGestureClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float) -> None:
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

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            _, (h_n, _) = self.lstm(x)
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            combined = torch.cat([forward_last, backward_last], dim=-1)
            combined = self.dropout(combined)
            return self.classifier(combined)

else:
    BiLSTMGestureClassifier = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BiLSTM classifier on processed dynamic gesture landmark sequences."
    )
    parser.add_argument("--data", type=str, default="dynamic_gestures/data/processed/dynamic_gestures_v1.pt")
    parser.add_argument("--output", type=str, default="dynamic_gestures/models/dynamic_bilstm.pt")
    parser.add_argument("--onnx-output", type=str, default="dynamic_gestures/models/dynamic_bilstm.onnx")
    parser.add_argument("--report", type=str, default="dynamic_gestures/data/processed/dynamic_training_report.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device. 'auto' prefers CUDA, then MPS (Apple Silicon GPU), then CPU.",
    )
    # --- Apple Silicon / performance tuning ---------------------------------
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker processes. 0 is fastest for this small in-memory dataset; "
            "raise it only if data loading becomes the bottleneck. Forced to 0 when --preload is on."
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help=(
            "Intra-op CPU threads (torch.set_num_threads). 0 leaves the PyTorch default. "
            "On M3 Pro try 6 to pin work to the performance cores."
        ),
    )
    parser.add_argument(
        "--no-preload",
        dest="preload",
        action="store_false",
        default=True,
        help=(
            "Disable preloading the full dataset into device (GPU/unified) memory. "
            "Preloading is on by default and removes per-batch host->device copies."
        ),
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help=(
            "Enable autocast mixed precision on CUDA/MPS. Note: LSTM/RNN ops stay fp32 under "
            "autocast, so the speedup for this model is small; off by default."
        ),
    )
    parser.add_argument(
        "--no-onnx",
        dest="export_onnx",
        action="store_false",
        default=True,
        help="Disable ONNX export of the best model.",
    )
    return parser.parse_args(argv)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def mps_available() -> bool:
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if mps_available() and hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def resolve_device(device_arg: str) -> "torch.device":
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested via --device cuda but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not mps_available():
            raise SystemExit(
                "MPS was requested via --device mps but it is unavailable. "
                "Requires Apple Silicon and a PyTorch build with MPS support."
            )
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_backend(device: "torch.device", num_threads: int) -> None:
    """Apply device-specific performance settings."""
    if num_threads and num_threads > 0:
        torch.set_num_threads(num_threads)
    # TF32 / high-precision matmul paths (no-op on MPS/CPU, helps on CUDA).
    with contextlib.suppress(Exception):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def autocast_context(device: "torch.device", enabled: bool):
    if enabled and device.type in ("cuda", "mps"):
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def compute_class_weights(y_train: "torch.Tensor", num_classes: int) -> "torch.Tensor":
    counts = torch.bincount(y_train, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    return counts.sum() / (num_classes * counts)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix


def compute_classification_metrics(confusion: np.ndarray) -> dict[str, Any]:
    num_classes = confusion.shape[0]
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = float(correct / total) if total > 0 else 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for class_index in range(num_classes):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum() - true_positive)
        false_negative = int(confusion[class_index, :].sum() - true_positive)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_scores.append(float(f1_score))

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return {
        "accuracy": accuracy,
        "precision_per_class": precisions,
        "recall_per_class": recalls,
        "f1_per_class": f1_scores,
        "macro_f1": macro_f1,
    }


def evaluate(
    model: "nn.Module",
    loader: "DataLoader",
    criterion: "nn.Module",
    device: "torch.device",
    num_classes: int,
    non_blocking: bool,
    use_amp: bool,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=non_blocking)
            y_batch = y_batch.to(device, non_blocking=non_blocking)
            with autocast_context(device, use_amp):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

            total_loss += float(loss.item()) * X_batch.size(0)
            total_count += X_batch.size(0)
            predictions = torch.argmax(logits, dim=-1)
            all_true.extend(y_batch.detach().cpu().tolist())
            all_pred.extend(predictions.detach().cpu().tolist())

    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    confusion = compute_confusion_matrix(np.array(all_true), np.array(all_pred), num_classes)
    metrics = compute_classification_metrics(confusion)
    metrics["loss"] = avg_loss
    metrics["confusion_matrix"] = confusion.tolist()
    return metrics


def train_one_epoch(
    model: "nn.Module",
    loader: "DataLoader",
    criterion: "nn.Module",
    optimizer: "torch.optim.Optimizer",
    device: "torch.device",
    grad_clip: float,
    non_blocking: bool,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    correct = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=non_blocking)
        y_batch = y_batch.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item()) * X_batch.size(0)
        total_count += X_batch.size(0)
        predictions = torch.argmax(logits, dim=-1)
        correct += int((predictions == y_batch).sum().item())

    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    accuracy = correct / total_count if total_count > 0 else 0.0
    return avg_loss, accuracy


def export_onnx(model: "nn.Module", onnx_path: Path, target_len: int, feature_dim: int) -> tuple[bool, str]:
    # Export from a CPU copy: MPS/CUDA tracing for ONNX export is unreliable, and the
    # exported graph is device-agnostic anyway.
    try:
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_model = copy.deepcopy(model).to("cpu").eval()
        dummy_input = torch.randn(1, target_len, feature_dim)
        torch.onnx.export(
            cpu_model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=17,
        )
        return True, "exported"
    except Exception as exc:
        logger.warning(f"ONNX export failed: {exc}")
        return False, f"failed: {exc}"


def main() -> int:
    args = parse_args()

    if not TORCH_AVAILABLE:
        raise SystemExit("PyTorch is required to train this model. Install it with: python -m pip install torch")

    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_backend(device, args.num_threads)

    # Preloading keeps the whole dataset resident in device memory, eliminating per-batch
    # host->device copies. It requires single-process loading (workers cannot share GPU tensors).
    preload = args.preload and device.type in ("cuda", "mps")
    num_workers = 0 if preload else args.num_workers
    non_blocking = device.type == "cuda"
    use_amp = args.amp

    print(f"Using device: {device} | torch {torch.__version__} | threads={torch.get_num_threads()}")
    print(
        f"Performance: preload={preload} num_workers={num_workers} amp={use_amp} "
        f"(MPS available={mps_available()})"
    )

    data_path = resolve_path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Processed dataset not found at: {data_path}. Run scripts/process_dynamic_dataset.py first.")

    dataset = torch.load(data_path, map_location="cpu", weights_only=False)

    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_val, y_val = dataset["X_val"], dataset["y_val"]
    X_test, y_test = dataset["X_test"], dataset["y_test"]
    class_names: list[str] = dataset["class_names"]
    label_to_id: dict[str, int] = dataset["label_to_id"]
    id_to_label: dict[int, str] = dataset["id_to_label"]
    target_len: int = dataset["target_len"]
    feature_dim: int = dataset["feature_dim"]
    normalization: dict[str, Any] = dataset["normalization"]
    add_velocity: bool = dataset["add_velocity"]
    num_classes = len(class_names)

    # Class weights derived from the CPU label tensor (bincount), before any device move.
    class_weights = compute_class_weights(y_train, num_classes).to(device)

    if preload:
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

    pin_memory = device.type == "cuda" and not preload
    loader_kwargs: dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = BiLSTMGestureClassifier(
        input_dim=feature_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.grad_clip, non_blocking, use_amp
        )
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes, non_blocking, use_amp)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no val_macro_f1 improvement for {args.patience} epochs.")
                break

    model.load_state_dict(best_state_dict)
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes, non_blocking, use_amp)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_class_test_metrics = {
        class_names[index]: {
            "precision": test_metrics["precision_per_class"][index],
            "recall": test_metrics["recall_per_class"][index],
            "f1": test_metrics["f1_per_class"][index],
        }
        for index in range(num_classes)
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "BiLSTMGestureClassifier",
            "input_dim": feature_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "num_classes": num_classes,
            "class_names": class_names,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "target_len": target_len,
            "feature_dim": feature_dim,
            "normalization": normalization,
            "add_velocity": add_velocity,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "test_metrics": test_metrics,
        },
        output_path,
    )

    onnx_path = resolve_path(args.onnx_output)
    onnx_exported = False
    onnx_status = "skipped: --no-onnx"
    if args.export_onnx:
        onnx_exported, onnx_status = export_onnx(model, onnx_path, target_len, feature_dim)

    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_path": str(data_path),
        "checkpoint_path": str(output_path),
        "onnx_output_path": str(onnx_path) if args.export_onnx else None,
        "onnx_export_status": onnx_status,
        "device": str(device),
        "runtime": {
            "torch_version": torch.__version__,
            "preload_to_device": preload,
            "num_workers": num_workers,
            "num_threads": torch.get_num_threads(),
            "amp": use_amp,
            "mps_available": mps_available(),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        },
        "class_names": class_names,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(X_test.shape[0]),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "per_class_test_metrics": per_class_test_metrics,
        "confusion_matrix": test_metrics["confusion_matrix"],
    }

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro F1: {best_val_macro_f1:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Saved PyTorch checkpoint: {output_path}")
    if args.export_onnx:
        if onnx_exported:
            print(f"Saved ONNX model: {onnx_path}")
        else:
            print(f"ONNX export warning: {onnx_status}")
    else:
        print("ONNX export skipped (--no-onnx).")
    print(f"Saved training report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
