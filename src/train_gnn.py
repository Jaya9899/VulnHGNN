"""
train_gnn.py
============
Training script for VulnGNN multi-label vulnerability classifier.

Usage:
    python3 src/train_gnn.py
    python3 src/train_gnn.py --augment --epochs 60 --lr 1e-3
    python3 src/train_gnn.py --embeddings data/embeddings --out results/gnn

Features:
  - Weighted BCE loss   (handles class imbalance)
  - Cosine LR schedule  (smooth decay)
  - Early stopping      (patience-based)
  - Synthetic augmentation (optional --augment flag)
  - Full metrics report (per-class F1, micro/macro)
  - Saves best model checkpoint
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    hamming_loss, classification_report,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_model import build_model, NUM_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_gnn")

# ── Constants ────────────────────────────────────────────────────────────
CLASS_NAMES = ["non-vulnerable", "CWE-476", "CWE-119", "CWE-191", "CWE-190", "CWE-369"]
RANDOM_SEED = 42
TEST_SIZE   = 0.20
VAL_SIZE    = 0.10   # of training set


# ════════════════════════════════════════════════════════════════════════
# 1. DATASET
# ════════════════════════════════════════════════════════════════════════

class VulnDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ════════════════════════════════════════════════════════════════════════
# 2. AUGMENTATION
# ════════════════════════════════════════════════════════════════════════

def augment_multilabel(X: np.ndarray, y: np.ndarray, n_synthetic: int = 300) -> tuple:
    """
    Synthetic multi-label augmentation.
    Averages embeddings of two files with different CWEs → OR their labels.
    Teaches the model that multiple vulnerabilities can co-occur.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # Indices per vulnerable class (skip class 0 = non-vulnerable)
    class_indices = {
        i: np.where(y[:, i] == 1)[0]
        for i in range(1, y.shape[1])
        if y[:, i].sum() > 0
    }

    classes = list(class_indices.keys())
    if len(classes) < 2:
        logger.warning("Not enough classes for augmentation, skipping.")
        return X, y

    X_aug, y_aug = [], []
    for _ in range(n_synthetic):
        c1, c2 = rng.choice(classes, size=2, replace=False)
        i1 = rng.choice(class_indices[c1])
        i2 = rng.choice(class_indices[c2])

        x_new = (X[i1] + X[i2]) / 2.0
        y_new = np.clip(y[i1] + y[i2], 0, 1)

        X_aug.append(x_new)
        y_aug.append(y_new)

    X_aug = np.stack(X_aug)
    y_aug = np.stack(y_aug)

    logger.info("Augmented: +%d synthetic multi-label samples", n_synthetic)
    return np.concatenate([X, X_aug]), np.concatenate([y, y_aug])


# ════════════════════════════════════════════════════════════════════════
# 3. CLASS WEIGHTS (handle imbalance)
# ════════════════════════════════════════════════════════════════════════

def compute_pos_weights(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss.
    pos_weight[i] = (neg_count / pos_count) for class i.
    Upweights rare classes automatically.
    """
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    # Avoid division by zero for missing classes (CWE-119)
    weights = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    logger.info("Class pos_weights: %s", dict(zip(CLASS_NAMES, weights.round(2))))
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ════════════════════════════════════════════════════════════════════════
# 4. METRICS
# ════════════════════════════════════════════════════════════════════════

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Only evaluate classes that have support
    active = [i for i in range(NUM_CLASSES) if y_true[:, i].sum() > 0]

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_p  = precision_score(y_true, y_pred, average="micro", zero_division=0)
    macro_p  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    micro_r  = recall_score(y_true, y_pred, average="micro", zero_division=0)
    macro_r  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    hl       = hamming_loss(y_true, y_pred)

    per_class = {}
    p_arr = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_arr = recall_score(y_true, y_pred, average=None, zero_division=0)
    f_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
    sup   = y_true.sum(axis=0).astype(int)

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
    print("  GNN RESULTS  (Multi-label Vulnerability Detection)")
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
# 5. TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, threshold: float = 0.5) -> tuple:
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(X_batch)

        all_logits.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

    probs  = np.concatenate(all_logits,  axis=0)
    labels = np.concatenate(all_labels,  axis=0)
    preds  = (probs >= threshold).astype(int)

    avg_loss = total_loss / len(loader.dataset)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    return avg_loss, micro_f1, labels, preds


# ════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════════════

def run(args):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load embeddings ─────────────────────────────────────────────
    emb_dir = Path(args.embeddings)
    X = np.load(emb_dir / "X_embeddings.npy")
    y = np.load(emb_dir / "y_labels.npy")
    logger.info("Loaded: X=%s  y=%s", X.shape, y.shape)

    # ── Train/val/test split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )
    logger.info("Split: train=%d  val=%d  test=%d",
                len(X_train), len(X_val), len(X_test))

    # ── Optional augmentation (train set only) ───────────────────────
    if args.augment:
        X_train, y_train = augment_multilabel(X_train, y_train, n_synthetic=args.n_synthetic)
        logger.info("After augmentation: train=%d", len(X_train))

    # ── Datasets & loaders ───────────────────────────────────────────
    train_ds = VulnDataset(X_train, y_train)
    val_ds   = VulnDataset(X_val,   y_val)
    test_ds  = VulnDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────
    input_dim = X.shape[1]
    model     = build_model(input_dim=input_dim, dropout=args.dropout).to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # ── Loss (weighted BCE for imbalance) ────────────────────────────
    pos_weights = compute_pos_weights(y_train, device)
    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # ── Optimizer + scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ────────────────────────────────────────────────
    best_val_f1   = 0.0
    best_epoch    = 0
    patience_ctr  = 0
    history       = []

    logger.info("Starting training for %d epochs (patience=%d) ...",
                args.epochs, args.patience)
    sep = "─" * 65
    print(f"\n{sep}")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val F1':>8}  {'LR':>10}")
    print(sep)

    for epoch in range(1, args.epochs + 1):
        train_loss              = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1, _, _ = eval_epoch(model, val_loader, criterion, device, args.threshold)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4), "val_f1": round(val_f1, 4),
        })

        improved = "✓" if val_f1 > best_val_f1 else ""
        print(f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>8.4f}  "
              f"{val_f1:>8.4f}  {lr:>10.2e}  {improved}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_epoch   = epoch
            patience_ctr = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_f1":     val_f1,
                "input_dim":  input_dim,
                "args":       vars(args),
            }, out_dir / "best_model.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                logger.info("Early stopping at epoch %d (best epoch %d, val F1=%.4f)",
                            epoch, best_epoch, best_val_f1)
                break

    print(sep)
    logger.info("Best val F1: %.4f at epoch %d", best_val_f1, best_epoch)

    # ── Final evaluation on test set ─────────────────────────────────
    logger.info("Loading best model for test evaluation ...")
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, test_f1, y_true, y_pred = eval_epoch(
        model, test_loader, criterion, device, args.threshold
    )
    metrics = evaluate(y_true, y_pred)
    print_metrics(metrics, y_true, y_pred)

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "model":       "VulnGNN (ResidualBlocks + AttentionPooling)",
        "input_dim":   input_dim,
        "best_epoch":  best_epoch,
        "best_val_f1": round(best_val_f1, 4),
        "augmented":   args.augment,
        "threshold":   args.threshold,
        "metrics":     metrics,
        "history":     history,
    }

    with open(out_dir / "gnn_metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved → %s/gnn_metrics.json", out_dir)


# ════════════════════════════════════════════════════════════════════════
# 7. ARGS
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train VulnGNN for multi-label vulnerability detection")
    p.add_argument("--embeddings",    type=str,   default="data/embeddings",
                   help="Directory with X_embeddings.npy and y_labels.npy")
    p.add_argument("--out",           type=str,   default="results/gnn",
                   help="Output directory for model + metrics")
    p.add_argument("--epochs",        type=int,   default=80)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--patience",      type=int,   default=15,
                   help="Early stopping patience (epochs)")
    p.add_argument("--threshold",     type=float, default=0.5,
                   help="Sigmoid threshold for binary prediction")
    p.add_argument("--augment",       action="store_true",
                   help="Enable synthetic multi-label augmentation")
    p.add_argument("--n_synthetic",   type=int,   default=300,
                   help="Number of synthetic samples when --augment is set")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)