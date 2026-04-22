"""
train_pyg.py
============
Training script for VulnGAT — proper GNN with message passing
over def-use and control-flow edges.

Usage:
    python3 src/train_pyg.py
    python3 src/train_pyg.py --augment --epochs 80
    python3 src/train_pyg.py --json_dir data/parsed_ir --out results/pyg_gnn
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    hamming_loss, classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyg_dataset import IRGraphDataset, NUM_CLASSES, get_cwe_from_filename
from pyg_model   import build_pyg_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_pyg")

CLASS_NAMES = ["non-vulnerable", "CWE-476", "CWE-191", "CWE-190", "CWE-369"]
RANDOM_SEED = 42


# ════════════════════════════════════════════════════════════════════════
# 1. AUGMENTATION
# ════════════════════════════════════════════════════════════════════════

def augment_dataset(dataset, n_synthetic: int = 300) -> list:
    """
    Synthetic multi-label augmentation on PyG graphs.
    Merges node features of two graphs with different CWEs.
    Returns list of original + synthetic Data objects.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # Group indices by CWE class (skip class 0 = non-vulnerable)
    class_indices = {i: [] for i in range(1, NUM_CLASSES)}
    for idx in range(len(dataset)):
        data = dataset[idx]
        label = data.y.squeeze(0)
        for c in range(1, NUM_CLASSES):
            if label[c].item() == 1.0:
                class_indices[c].append(idx)

    active_classes = [c for c, idxs in class_indices.items() if len(idxs) > 0]
    if len(active_classes) < 2:
        logger.warning("Not enough classes for augmentation.")
        return list(range(len(dataset)))

    synthetic = []
    for _ in range(n_synthetic):
        c1, c2 = rng.choice(active_classes, size=2, replace=False)
        i1 = rng.choice(class_indices[c1])
        i2 = rng.choice(class_indices[c2])

        d1 = dataset[i1]
        d2 = dataset[i2]

        # Average node features (use the larger graph's structure)
        if d1.num_nodes >= d2.num_nodes:
            base, other = d1, d2
        else:
            base, other = d2, d1

        # Pad smaller node feature matrix to match larger
        n_base  = base.num_nodes
        n_other = other.num_nodes
        if n_other < n_base:
            pad = torch.zeros(n_base - n_other, other.x.size(1))
            x_other = torch.cat([other.x, pad], dim=0)
        else:
            x_other = other.x[:n_base]

        x_new = (base.x + x_other) / 2.0

        # OR the labels
        y_new = torch.clamp(base.y + other.y, 0, 1)

        synthetic.append(Data(
            x          = x_new,
            edge_index = base.edge_index,
            edge_attr  = base.edge_attr,
            y          = y_new,
            num_nodes  = n_base,
        ))

    logger.info("Augmented: +%d synthetic multi-label graphs", len(synthetic))
    return synthetic


# ════════════════════════════════════════════════════════════════════════
# 2. FOCAL LOSS + WEIGHTED LOSS
# ════════════════════════════════════════════════════════════════════════

class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    Down-weights easy-to-classify samples so the model focuses on
    hard examples (e.g., CWE-190/191 vs safe arithmetic).

    FL(p) = -alpha * (1-p)^gamma * log(p)       for y=1
            -(1-alpha) * p^gamma * log(1-p)      for y=0

    gamma=2 is standard; higher values focus more on hard examples.
    """
    def __init__(self, gamma: float = 2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # BCE per element (no reduction)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        # p_t = probability of correct class
        p_t = targets * probs + (1 - targets) * (1 - probs)
        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        # Apply pos_weight for class imbalance
        if self.pos_weight is not None:
            weight = targets * self.pos_weight + (1 - targets)
            loss = loss * weight

        return loss.mean()


def compute_pos_weights(dataset, device: torch.device) -> torch.Tensor:
    all_labels = torch.cat([dataset[i].y for i in range(len(dataset))], dim=0)
    pos    = all_labels.sum(dim=0)
    neg    = len(dataset) - pos
    weights = torch.where(pos > 0, neg / pos.clamp(min=1), torch.ones_like(pos))
    
    # Cap weights — uncapped weights cause high recall but terrible precision
    weights = torch.clamp(weights, max=3.0)
    
    logger.info("Pos weights (capped): %s", dict(zip(CLASS_NAMES, weights.numpy().round(2))))
    return weights.to(device)


# ════════════════════════════════════════════════════════════════════════
# 3. METRICS
# ════════════════════════════════════════════════════════════════════════

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_p  = precision_score(y_true, y_pred, average="micro", zero_division=0)
    macro_p  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    micro_r  = recall_score(y_true, y_pred, average="micro", zero_division=0)
    macro_r  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    hl       = hamming_loss(y_true, y_pred)

    p_arr = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_arr = recall_score(y_true, y_pred, average=None, zero_division=0)
    f_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
    sup   = y_true.sum(axis=0).astype(int)

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": round(float(p_arr[i]), 4),
            "recall":    round(float(r_arr[i]), 4),
            "f1":        round(float(f_arr[i]), 4),
            "support":   int(sup[i]),
        }

    return {
        "hamming_loss":    round(hl,       4),
        "micro_f1":        round(micro_f1, 4),
        "macro_f1":        round(macro_f1, 4),
        "micro_precision": round(micro_p,  4),
        "macro_precision": round(macro_p,  4),
        "micro_recall":    round(micro_r,  4),
        "macro_recall":    round(macro_r,  4),
        "per_class":       per_class,
    }


def print_metrics(metrics: dict, y_true: np.ndarray, y_pred: np.ndarray):
    sep = "─" * 65
    print(f"\n{sep}")
    print("  PyG GAT RESULTS  (Multi-label Vulnerability Detection)")
    print(sep)
    print(f"  Hamming Loss     : {metrics['hamming_loss']:.4f}")
    print(f"  Micro F1         : {metrics['micro_f1']:.4f}")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  Micro Precision  : {metrics['micro_precision']:.4f}")
    print(f"  Macro Precision  : {metrics['macro_precision']:.4f}")
    print(f"  Micro Recall     : {metrics['micro_recall']:.4f}")
    print(f"  Macro Recall     : {metrics['macro_recall']:.4f}")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<18} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-' * 50}")
    for name, m in metrics["per_class"].items():
        print(f"  {name:<18} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['support']:>8}")
    print(sep)
    print("\n  Full sklearn report:")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0, digits=4,
    ))


# ════════════════════════════════════════════════════════════════════════
# 4. TRAIN / EVAL LOOPS
# ════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device, log_gradients: bool = False) -> float:
    model.train()
    total_loss = 0.0
    max_grad_norm = 0.0
    for batch in loader:
        batch = batch.to(device)

        # Handle empty edge case
        if batch.edge_index.size(1) == 0:
            continue

        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss   = criterion(logits, batch.y.squeeze(1))
        loss.backward()

        # Track gradient norms for stability monitoring
        if log_gradients:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    max_grad_norm = max(max_grad_norm, grad_norm)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    if log_gradients and max_grad_norm > 0:
        logger.info("  Max gradient norm: %.4f", max_grad_norm)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, threshold: float = 0.5) -> tuple:
    model.eval()
    total_loss  = 0.0
    all_probs   = []
    all_labels  = []

    for batch in loader:
        batch  = batch.to(device)

        if batch.edge_index.size(1) == 0:
            continue

        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss   = criterion(logits, batch.y.squeeze(1))
        total_loss += loss.item() * batch.num_graphs

        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.squeeze(1).cpu().numpy())

    probs  = np.concatenate(all_probs,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds  = (probs >= threshold).astype(int)

    avg_loss = total_loss / len(loader.dataset)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    return avg_loss, micro_f1, labels, preds

def balance_dataset(data_list: list, max_ratio: float = 3.0) -> list:
    import random
    rng = random.Random(42)
    
    vuln    = [d for d in data_list if d.y.squeeze(0)[0].item() < 0.5]
    nonvuln = [d for d in data_list if d.y.squeeze(0)[0].item() >= 0.5]
    
    n_keep  = min(len(nonvuln), int(len(vuln) * max_ratio))
    nonvuln = rng.sample(nonvuln, n_keep)
    
    balanced = vuln + nonvuln
    rng.shuffle(balanced)
    logger.info("Balanced: %d vuln + %d non-vuln = %d total",
                len(vuln), n_keep, len(balanced))
    return balanced

# ════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ════════════════════════════════════════════════════════════════════════

def run(args):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load dataset ─────────────────────────────────────────────────
    logger.info("Loading dataset (building graphs + caching) ...")
    dataset = IRGraphDataset(args.json_dir, cache_dir=args.cache_dir)
    logger.info("Total graphs: %d", len(dataset))

    # ── Train/val/test split ─────────────────────────────────────────
    n       = len(dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(RANDOM_SEED)).tolist()

    n_test  = int(n * 0.20)
    n_val   = int(n * 0.10)
    n_train = n - n_test - n_val

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    logger.info("Split: train=%d  val=%d  test=%d", n_train, n_val, n_test)

    # ── Optional augmentation on training set ────────────────────────
    train_data = [dataset[i] for i in train_idx]
    train_data = balance_dataset(train_data, max_ratio=3.0)
    if args.augment:
        # Build temporary subset for augmentation
        class TempDataset:
            def __init__(self, data_list):
                self._data = data_list
            def __len__(self):
                return len(self._data)
            def __getitem__(self, i):
                return self._data[i]

        synthetic = augment_dataset(TempDataset(train_data), n_synthetic=args.n_synthetic)
        train_data = train_data + synthetic
        logger.info("Train set after augmentation: %d", len(train_data))

    val_data  = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────
    model = build_pyg_model(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("VulnGAT parameters: %d", n_params)

    # ── Loss ─────────────────────────────────────────────────────────
    class TempDs:
        def __init__(self, dl): self._d = dl
        def __len__(self): return len(self._d)
        def __getitem__(self, i): return self._d[i]

    pos_weights = compute_pos_weights(TempDs(train_data), device)
    if args.focal:
        criterion = FocalBCEWithLogitsLoss(gamma=args.gamma, pos_weight=pos_weights)
        logger.info("Using FocalBCEWithLogitsLoss (gamma=%.1f)", args.gamma)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # ── Optimizer + scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ────────────────────────────────────────────────
    best_val_f1  = 0.0
    best_epoch   = 0
    patience_ctr = 0
    history      = []

    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val F1':>8}  {'LR':>10}")
    print(sep)

    for epoch in range(1, args.epochs + 1):
        train_loss              = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1, _, _ = eval_epoch(model, val_loader, criterion, device, args.threshold)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss,   4),
            "val_f1":     round(val_f1,     4),
        })

        improved = "✓" if val_f1 > best_val_f1 else ""
        print(f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>8.4f}  "
              f"{val_f1:>8.4f}  {lr:>10.2e}  {improved}")

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_epoch   = epoch
            patience_ctr = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_f1":      val_f1,
                "args":        vars(args),
            }, out_dir / "best_pyg_model.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                logger.info("Early stopping at epoch %d (best epoch %d, val F1=%.4f)",
                            epoch, best_epoch, best_val_f1)
                break

    print(sep)
    logger.info("Best val F1: %.4f at epoch %d", best_val_f1, best_epoch)

    # ── Test evaluation ───────────────────────────────────────────────
    logger.info("Loading best model for test evaluation ...")
    ckpt = torch.load(out_dir / "best_pyg_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    _, test_f1, y_true, y_pred = eval_epoch(
        model, test_loader, criterion, device, args.threshold
    )
    metrics = evaluate(y_true, y_pred)
    print_metrics(metrics, y_true, y_pred)

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "model":       "VulnGAT (GATv2 + def-use + CFG edges)",
        "best_epoch":  best_epoch,
        "best_val_f1": round(best_val_f1, 4),
        "augmented":   args.augment,
        "threshold":   args.threshold,
        "metrics":     metrics,
        "history":     history,
    }

    with open(out_dir / "pyg_metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved → %s/pyg_metrics.json", out_dir)

    # ── Compare with previous baseline ───────────────────────────────
    baseline_path = Path("results/gnn/gnn_metrics.json")
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        b = baseline["metrics"]
        print("\n  COMPARISON vs MLP-GNN baseline:")
        print(f"  {'Metric':<20} {'MLP-GNN':>10} {'PyG GAT':>10} {'Δ':>8}")
        print(f"  {'-'*50}")
        for key in ["micro_f1", "macro_f1", "hamming_loss"]:
            bval = b[key]
            gval = metrics[key]
            delta = gval - bval
            sign  = "+" if delta > 0 else ""
            print(f"  {key:<20} {bval:>10.4f} {gval:>10.4f} {sign}{delta:>7.4f}")


# ════════════════════════════════════════════════════════════════════════
# 6. ARGS
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train VulnGAT on LLVM IR graphs")
    p.add_argument("--json_dir",    type=str,   default="data/parsed_ir")
    p.add_argument("--cache_dir",   type=str,   default="data/pyg_cache")
    p.add_argument("--out",         type=str,   default="results/pyg_gnn")
    p.add_argument("--epochs",      type=int,   default=80)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--patience",    type=int,   default=15)
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--augment",     action="store_true")
    p.add_argument("--n_synthetic", type=int,   default=500)
    p.add_argument("--focal",       action="store_true",
                   help="Use Focal Loss instead of BCE (better for minority classes)")
    p.add_argument("--gamma",       type=float, default=2.0,
                   help="Focal loss gamma (higher = more focus on hard examples)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)