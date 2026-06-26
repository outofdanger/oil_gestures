from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow ops not yet implemented on the MPS (Apple Silicon GPU) backend to fall back
# to CPU instead of raising. Must be set before torch import.
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
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Graph construction (spatial-configuration partitioning, Yan et al. 2018)
# ---------------------------------------------------------------------------
def _bfs_hops(num_nodes: int, edges: list[tuple[int, int]], center: int) -> dict[int, int]:
    neighbors: dict[int, set[int]] = {i: set() for i in range(num_nodes)}
    for i, j in edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    dist = {center: 0}
    queue = [center]
    while queue:
        node = queue.pop(0)
        for nxt in neighbors[node]:
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    # Disconnected nodes (shouldn't happen for a hand) get a large distance.
    for node in range(num_nodes):
        dist.setdefault(node, num_nodes)
    return dist


def _edge2mat(link: list[tuple[int, int]], num_nodes: int) -> np.ndarray:
    matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in link:
        matrix[j, i] = 1.0
    return matrix


def _normalize_digraph(matrix: np.ndarray) -> np.ndarray:
    in_degree = matrix.sum(axis=0)
    norm = np.zeros_like(matrix)
    for node in range(matrix.shape[0]):
        if in_degree[node] > 0:
            norm[node, node] = in_degree[node] ** -1
    return matrix @ norm


def build_partitioned_adjacency(num_nodes: int, edges: list[tuple[int, int]], center: int = 0) -> np.ndarray:
    """Return (K=3, V, V): self-loops, centripetal (inward), centrifugal (outward).

    For each bone, the endpoint closer (in hop distance) to ``center`` is the
    parent. Inward edges point toward the center, outward edges away. This is the
    canonical ST-GCN spatial partitioning and lets each subset learn a distinct
    spatial-relation kernel.
    """
    hops = _bfs_hops(num_nodes, edges, center)
    self_link = [(i, i) for i in range(num_nodes)]
    inward: list[tuple[int, int]] = []
    outward: list[tuple[int, int]] = []
    for i, j in edges:
        if hops[i] <= hops[j]:
            nearer, farther = i, j
        else:
            nearer, farther = j, i
        inward.append((farther, nearer))   # toward center
        outward.append((nearer, farther))  # away from center

    identity = _edge2mat(self_link, num_nodes)
    centripetal = _normalize_digraph(_edge2mat(inward, num_nodes))
    centrifugal = _normalize_digraph(_edge2mat(outward, num_nodes))
    return np.stack([identity, centripetal, centrifugal]).astype(np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:

    class SpatialGraphConv(nn.Module):
        """Graph convolution over the K adjacency subsets."""

        def __init__(self, in_channels: int, out_channels: int, num_subsets: int) -> None:
            super().__init__()
            self.num_subsets = num_subsets
            self.conv = nn.Conv2d(in_channels, out_channels * num_subsets, kernel_size=1)

        def forward(self, x: "torch.Tensor", adjacency: "torch.Tensor") -> "torch.Tensor":
            # x: (N, C, T, V) ; adjacency: (K, V, V)
            x = self.conv(x)
            n, kc, t, v = x.shape
            x = x.view(n, self.num_subsets, kc // self.num_subsets, t, v)
            x = torch.einsum("nkctv,kvw->nctw", x, adjacency)
            return x.contiguous()

    class STGCNBlock(nn.Module):
        """Spatial graph conv + temporal conv with residual connection."""

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
            assert t_kernel % 2 == 1, "temporal kernel must be odd to keep length with padding"
            padding = (t_kernel - 1) // 2
            self.gcn = SpatialGraphConv(in_channels, out_channels, num_subsets)
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

        def forward(self, x: "torch.Tensor", adjacency: "torch.Tensor") -> "torch.Tensor":
            res = 0 if self.residual is None else self.residual(x)
            x = self.gcn(x, adjacency)
            x = self.tcn(x)
            return self.relu(x + res)

    class STGCN(nn.Module):
        """ST-GCN classifier for single-hand landmark sequences (N, C, T, V, M)."""

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
            # Adjacency is a fixed buffer; edge_importance makes each subset's graph
            # learnable (reweights connections) — a standard ST-GCN accuracy boost.
            self.register_buffer("adjacency", torch.tensor(adjacency, dtype=torch.float32))
            self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

            c1, c2 = base_channels, base_channels * 2
            self.blocks = nn.ModuleList(
                [
                    STGCNBlock(in_channels, c1, num_subsets, t_kernel, residual=False),
                    STGCNBlock(c1, c1, num_subsets, t_kernel, dropout=dropout),
                    STGCNBlock(c1, c1, num_subsets, t_kernel, dropout=dropout),
                    STGCNBlock(c1, c2, num_subsets, t_kernel, stride=2, dropout=dropout),
                    STGCNBlock(c2, c2, num_subsets, t_kernel, dropout=dropout),
                ]
            )
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones_like(self.adjacency)) for _ in self.blocks]
            )
            self.classifier = nn.Linear(c2, num_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (N, C, T, V, M)
            n, c, t, v, m = x.shape
            x = x.permute(0, 4, 3, 1, 2).contiguous().view(n * m, v * c, t)
            x = self.data_bn(x)
            x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)

            for block, importance in zip(self.blocks, self.edge_importance):
                x = block(x, self.adjacency * importance)

            # Global average pooling over time and joints, then average over persons.
            x = F.avg_pool2d(x, x.shape[2:])
            x = x.view(n, m, -1).mean(dim=1)
            return self.classifier(x)

else:
    STGCN = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ST-GCN classifier on processed dynamic gesture skeleton sequences."
    )
    parser.add_argument("--data", type=str, default="dynamic_gestures/data/processed/dynamic_gestures_v1.pt")
    parser.add_argument("--output", type=str, default="dynamic_gestures/models/dynamic_stgcn.pt")
    parser.add_argument("--onnx-output", type=str, default="dynamic_gestures/models/dynamic_stgcn.onnx")
    parser.add_argument("--report", type=str, default="dynamic_gestures/data/processed/stgcn_training_report.json")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--t-kernel", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.02, help="Cosine floor as a fraction of --lr.")
    parser.add_argument("--patience", type=int, default=20)
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=6,
        help="Intra-op CPU threads (torch.set_num_threads). 6 pins work to the M3 Pro performance cores.",
    )
    parser.add_argument(
        "--no-preload",
        dest="preload",
        action="store_false",
        default=True,
        help="Disable preloading the dataset into device (unified/GPU) memory.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable autocast mixed precision on CUDA/MPS.",
    )
    parser.add_argument("--no-onnx", dest="export_onnx", action="store_false", default=True)
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
            raise SystemExit("CUDA requested via --device cuda but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not mps_available():
            raise SystemExit("MPS requested via --device mps but it is unavailable (needs Apple Silicon).")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_backend(device: "torch.device", num_threads: int) -> None:
    if num_threads and num_threads > 0:
        torch.set_num_threads(num_threads)
    with contextlib.suppress(Exception):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def autocast_context(device: "torch.device", enabled: bool):
    if enabled and device.type in ("cuda", "mps"):
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def cosine_warmup_lambda(epoch_index: int, warmup_epochs: int, total_epochs: int, min_ratio: float) -> float:
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        return (epoch_index + 1) / warmup_epochs
    progress = (epoch_index - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    progress = min(1.0, max(0.0, progress))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


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
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_scores.append(float(f1))

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return {
        "accuracy": accuracy,
        "precision_per_class": precisions,
        "recall_per_class": recalls,
        "f1_per_class": f1_scores,
        "macro_f1": macro_f1,
    }


def evaluate(model, loader, criterion, device, num_classes, non_blocking, use_amp) -> dict[str, Any]:
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


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip, non_blocking, use_amp):
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


def export_onnx(model, onnx_path: Path, target_len: int, channels: int, num_nodes: int) -> tuple[bool, str]:
    try:
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_model = copy.deepcopy(model).to("cpu").eval()
        dummy_input = torch.randn(1, channels, target_len, num_nodes, 1)
        torch.onnx.export(
            cpu_model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
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

    preload = args.preload and device.type in ("cuda", "mps")
    num_workers = 0 if preload else args.num_workers
    non_blocking = device.type == "cuda"
    use_amp = args.amp

    print(f"Using device: {device} | torch {torch.__version__} | threads={torch.get_num_threads()}")
    print(f"Performance: preload={preload} num_workers={num_workers} amp={use_amp} (MPS available={mps_available()})")

    data_path = resolve_path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Processed dataset not found at: {data_path}. Run scripts/process_dynamic_dataset.py first.")

    dataset = torch.load(data_path, map_location="cpu", weights_only=False)
    if "X_train_stgcn" not in dataset:
        raise SystemExit(
            "This dataset has no ST-GCN tensors. Re-run scripts/process_dynamic_dataset.py to regenerate "
            "data with the graph (X_train_stgcn / adjacency / edges)."
        )

    X_train, y_train = dataset["X_train_stgcn"], dataset["y_train"]
    X_val, y_val = dataset["X_val_stgcn"], dataset["y_val"]
    X_test, y_test = dataset["X_test_stgcn"], dataset["y_test"]
    class_names: list[str] = dataset["class_names"]
    label_to_id: dict[str, int] = dataset["label_to_id"]
    id_to_label: dict[int, str] = dataset["id_to_label"]
    target_len: int = dataset["target_len"]
    meta = dataset["stgcn_meta"]
    num_nodes = int(meta["num_nodes"])
    channels = int(meta["num_channels"])
    edges = [tuple(edge) for edge in dataset["edges"]]
    num_classes = len(class_names)

    adjacency = build_partitioned_adjacency(num_nodes, edges, center=0)

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

    model = STGCN(
        in_channels=channels,
        num_classes=num_classes,
        adjacency=adjacency,
        base_channels=args.base_channels,
        t_kernel=args.t_kernel,
        dropout=args.dropout,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"ST-GCN params: {num_params:,} | base_channels={args.base_channels} t_kernel={args.t_kernel}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch_index: cosine_warmup_lambda(
            epoch_index, args.warmup_epochs, args.epochs, args.min_lr_ratio
        ),
    )

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
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | lr={current_lr:.5f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
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
            "model_type": "STGCN",
            "in_channels": channels,
            "num_nodes": num_nodes,
            "num_classes": num_classes,
            "base_channels": args.base_channels,
            "t_kernel": args.t_kernel,
            "dropout": args.dropout,
            "adjacency": adjacency.tolist(),
            "edges": [list(edge) for edge in edges],
            "class_names": class_names,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "target_len": target_len,
            "node_features": meta.get("node_features"),
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
        onnx_exported, onnx_status = export_onnx(model, onnx_path, target_len, channels, num_nodes)

    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model_type": "STGCN",
        "dataset_path": str(data_path),
        "checkpoint_path": str(output_path),
        "onnx_output_path": str(onnx_path) if args.export_onnx else None,
        "onnx_export_status": onnx_status,
        "device": str(device),
        "num_params": num_params,
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
            "base_channels": args.base_channels,
            "t_kernel": args.t_kernel,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "warmup_epochs": args.warmup_epochs,
            "min_lr_ratio": args.min_lr_ratio,
            "patience": args.patience,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        },
        "class_names": class_names,
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
        print(f"Saved ONNX model: {onnx_path}" if onnx_exported else f"ONNX export warning: {onnx_status}")
    print(f"Saved training report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
