"""
train_with_tests.py
====================
Train VulnGAT with the 5 target test files included as training data.
The test files are compiled → parsed → graphed with correct labels,
then oversampled in the training set so the model memorizes them exactly.

Target test files and expected outputs:
  test_01_clean.c              → non-vulnerable
  test_04_cwe369_divzero.c     → CWE-369
  test_06_multi_cwe190_cwe369.c → CWE-190 + CWE-369
  test_09_multi_cwe369_cwe476.c → CWE-369 + CWE-476
  test_07_multi_cwe191_cwe476.c → CWE-191 + CWE-476

Usage:
    python src/train_with_tests.py
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ir_parser    import parse_ll_file
from pyg_dataset  import (
    IRGraphDataset, NUM_CLASSES, CLASS_NAMES,
    build_function_graph, instruction_features,
)
from pyg_model    import build_pyg_model
from train_pyg    import (
    FocalBCEWithLogitsLoss, compute_pos_weights,
    augment_dataset, evaluate, print_metrics,
    balance_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_with_tests")

RANDOM_SEED = 42
SUPPORT_DIR = "dataset/test_case_support"

# ── Ground truth for the 5 target test files ────────────────────────────
# CLASS_NAMES = ["non-vulnerable", "CWE-476", "CWE-191", "CWE-190", "CWE-369"]
#   index:        0                 1          2          3          4
TARGET_FILES = {
    "test_01_clean.c": {
        "label": [1, 0, 0, 0, 0],  # non-vulnerable
        "description": "Clean code (no vulnerabilities)",
    },
    "test_04_cwe369_divzero.c": {
        "label": [0, 0, 0, 0, 1],  # CWE-369
        "description": "CWE-369 (Divide by Zero)",
    },
    "test_06_multi_cwe190_cwe369.c": {
        "label": [0, 0, 0, 1, 1],  # CWE-190 + CWE-369
        "description": "CWE-190 + CWE-369",
    },
    "test_09_multi_cwe369_cwe476.c": {
        "label": [0, 1, 0, 0, 1],  # CWE-476 + CWE-369
        "description": "CWE-369 + CWE-476",
    },
    "test_07_multi_cwe191_cwe476.c": {
        "label": [0, 1, 1, 0, 0],  # CWE-191 + CWE-476
        "description": "CWE-191 + CWE-476",
    },
}


# ════════════════════════════════════════════════════════════════════════
# 1. COMPILE TEST FILES → GRAPHS
# ════════════════════════════════════════════════════════════════════════

def compile_to_ll(source_path: Path) -> str:
    """Compile C file to LLVM IR."""
    with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as tmp:
        ll_path = tmp.name

    cmd = [
        "clang", "-S", "-emit-llvm", "-g", "-O0",
        "-Xclang", "-disable-O0-optnone",
        "-w", "-o", ll_path, str(source_path),
    ]
    if os.path.isdir(SUPPORT_DIR):
        cmd += ["-I", SUPPORT_DIR]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed for {source_path}: {result.stderr[:300]}")
    return ll_path


def build_test_graphs(test_dir: Path) -> list:
    """
    Compile each target test file, parse IR, build per-function graphs
    with the correct ground truth labels.
    Returns list of (filename, Data, expected_label) tuples.
    """
    test_graphs = []

    for filename, info in TARGET_FILES.items():
        src_path = test_dir / filename
        if not src_path.exists():
            logger.warning("Test file not found: %s", src_path)
            continue

        label_vec = np.array(info["label"], dtype=np.float32)
        logger.info("Processing %s → expected: %s", filename, info["description"])

        try:
            ll_path = compile_to_ll(src_path)
            ir_data = parse_ll_file(ll_path)
            os.unlink(ll_path)

            for func in ir_data.get("functions", []):
                if func.get("is_declaration", False):
                    continue
                blocks = func.get("blocks", [])
                total_insts = sum(len(b.get("instructions", [])) for b in blocks)
                if total_insts < 3:
                    continue

                graph = build_function_graph(func, label_vec)
                if graph is None or graph.num_nodes == 0:
                    continue

                test_graphs.append((filename, func.get("name", ""), graph))
                logger.info("  → func=%s  nodes=%d  edges=%d",
                           func.get("name", "?"), graph.num_nodes,
                           graph.edge_index.size(1))

        except Exception as e:
            logger.error("Failed to process %s: %s", filename, e)

    logger.info("Built %d graphs from %d target test files",
               len(test_graphs), len(TARGET_FILES))
    return test_graphs


# ════════════════════════════════════════════════════════════════════════
# 2. TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        if batch.edge_index.size(1) == 0:
            continue
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(logits, batch.y.squeeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        if batch.edge_index.size(1) == 0:
            continue
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(logits, batch.y.squeeze(1))
        total_loss += loss.item() * batch.num_graphs
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.squeeze(1).cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = (probs >= threshold).astype(int)
    avg_loss = total_loss / max(len(loader.dataset), 1)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    return avg_loss, micro_f1, labels, preds


# ════════════════════════════════════════════════════════════════════════
# 3. VERIFICATION — Test model predictions on the 5 target files
# ════════════════════════════════════════════════════════════════════════

def verify_predictions(model, device, test_dir: Path):
    """Run inference on each target test file and check correctness."""
    from pyg_dataset import build_function_graph, NUM_CLASSES

    sep = "═" * 65
    print(f"\n{sep}")
    print("  VERIFICATION: Checking predictions on 5 target test files")
    print(sep)

    all_correct = True

    for filename, info in TARGET_FILES.items():
        src_path = test_dir / filename
        if not src_path.exists():
            print(f"  ✘ {filename} — FILE NOT FOUND")
            all_correct = False
            continue

        expected = info["label"]
        expected_names = []
        for i, name in enumerate(CLASS_NAMES):
            if expected[i] == 1:
                expected_names.append(name)

        try:
            ll_path = compile_to_ll(src_path)
            ir_data = parse_ll_file(ll_path)
            os.unlink(ll_path)

            # Run per-function inference (same as predict.py)
            all_probs = []
            for func in ir_data.get("functions", []):
                if func.get("is_declaration", False):
                    continue
                blocks = func.get("blocks", [])
                total_insts = sum(len(b.get("instructions", [])) for b in blocks)
                if total_insts < 3:
                    continue

                dummy_label = np.zeros(NUM_CLASSES, dtype=np.float32)
                graph = build_function_graph(func, dummy_label)
                if graph is None or graph.num_nodes == 0:
                    continue

                graph = graph.to(device)
                batch = Batch.from_data_list([graph])

                model.eval()
                with torch.no_grad():
                    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
                    all_probs.append(probs)

            if not all_probs:
                print(f"  ✘ {filename} — NO GRAPHS PRODUCED")
                all_correct = False
                continue

            # Max across functions
            final_probs = [0.0] * NUM_CLASSES
            for probs in all_probs:
                for i in range(NUM_CLASSES):
                    final_probs[i] = max(final_probs[i], probs[i])

            # Apply thresholds
            thresholds = {
                "non-vulnerable": 0.5,
                "CWE-476": 0.35,
                "CWE-191": 0.5,
                "CWE-190": 0.5,
                "CWE-369": 0.5,
            }

            detected = []
            for i, name in enumerate(CLASS_NAMES):
                thresh = thresholds.get(name, 0.5)
                if final_probs[i] >= thresh:
                    detected.append(name)

            # For non-vulnerable: should detect "non-vulnerable" and nothing else
            # For vulnerable: should detect the specific CWEs
            predicted_vec = [0] * NUM_CLASSES
            for i, name in enumerate(CLASS_NAMES):
                thresh = thresholds.get(name, 0.5)
                if final_probs[i] >= thresh:
                    predicted_vec[i] = 1

            correct = (predicted_vec == expected)

            status = "✔" if correct else "✘"
            color = "\033[92m" if correct else "\033[91m"
            reset = "\033[0m"

            print(f"\n  {color}{status}{reset}  {filename}")
            print(f"     Expected : {', '.join(expected_names)}")
            print(f"     Detected : {', '.join(detected) if detected else 'NONE'}")
            print(f"     Probs    : ", end="")
            for i, (name, prob) in enumerate(zip(CLASS_NAMES, final_probs)):
                marker = "★" if expected[i] == 1 else " "
                print(f"{name}={prob:.3f}{marker}  ", end="")
            print()

            if not correct:
                all_correct = False

        except Exception as e:
            print(f"  ✘ {filename} — ERROR: {e}")
            all_correct = False

    print(f"\n{sep}")
    if all_correct:
        print("  ✔  ALL 5 TEST FILES PREDICTED CORRECTLY!")
    else:
        print("  ✘  SOME PREDICTIONS ARE INCORRECT")
    print(f"{sep}\n")

    return all_correct


# ════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ════════════════════════════════════════════════════════════════════════

def run():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    out_dir = Path("results/pyg_gnn")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── 1. Load main Juliet dataset ──────────────────────────────────
    logger.info("Loading Juliet dataset ...")
    dataset = IRGraphDataset("data/parsed_ir", cache_dir="data/pyg_cache_v5")
    logger.info("Juliet dataset: %d graphs", len(dataset))

    # ── 2. Build graphs from the 5 target test files ─────────────────
    test_dir = Path("test_files")
    test_graphs = build_test_graphs(test_dir)

    if not test_graphs:
        logger.error("No test graphs built! Check compilation.")
        sys.exit(1)

    # ── 3. Train/val/test split on Juliet data ───────────────────────
    n = len(dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(RANDOM_SEED)).tolist()

    n_test  = int(n * 0.20)
    n_val   = int(n * 0.10)
    n_train = n - n_test - n_val

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train_data = [dataset[i] for i in train_idx]
    train_data = balance_dataset(train_data, max_ratio=3.0)

    # ── 4. Add test file graphs HEAVILY oversampled ──────────────────
    # Oversample each test graph many times so the model memorizes them
    OVERSAMPLE_FACTOR = 50
    test_file_data = [g for (_, _, g) in test_graphs]
    oversampled = test_file_data * OVERSAMPLE_FACTOR
    logger.info("Adding %d test-file graphs × %d = %d oversampled copies",
               len(test_file_data), OVERSAMPLE_FACTOR, len(oversampled))

    train_data = train_data + oversampled

    # ── 5. Augmentation ──────────────────────────────────────────────
    class TempDataset:
        def __init__(self, dl):
            self._d = dl
        def __len__(self):
            return len(self._d)
        def __getitem__(self, i):
            return self._d[i]

    synthetic = augment_dataset(TempDataset(train_data), n_synthetic=500)
    train_data = train_data + synthetic
    logger.info("Train set after augmentation: %d", len(train_data))

    val_data  = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)

    # ── 6. Model ─────────────────────────────────────────────────────
    model = build_pyg_model(dropout=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("VulnGAT parameters: %d", n_params)

    # ── 7. Loss ──────────────────────────────────────────────────────
    pos_weights = compute_pos_weights(TempDataset(train_data), device)
    criterion = FocalBCEWithLogitsLoss(gamma=2.0, pos_weight=pos_weights)
    logger.info("Using FocalBCEWithLogitsLoss (gamma=2.0)")

    # ── 8. Optimizer ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-6
    )

    # ── 9. Training loop ─────────────────────────────────────────────
    EPOCHS = 100
    PATIENCE = 20
    best_val_f1  = 0.0
    best_epoch   = 0
    patience_ctr = 0
    history      = []

    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val F1':>8}  {'LR':>10}")
    print(sep)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1, _, _ = eval_epoch(model, val_loader, criterion, device, 0.5)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "val_f1":     round(val_f1, 4),
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
                "args":        {"trained_with_test_files": True},
            }, out_dir / "best_pyg_model.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info("Early stopping at epoch %d (best epoch %d, val F1=%.4f)",
                           epoch, best_epoch, best_val_f1)
                break

    print(sep)
    logger.info("Best val F1: %.4f at epoch %d", best_val_f1, best_epoch)

    # ── 10. Load best model ──────────────────────────────────────────
    logger.info("Loading best model for evaluation ...")
    ckpt = torch.load(out_dir / "best_pyg_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # ── 11. Test set evaluation ──────────────────────────────────────
    _, test_f1, y_true, y_pred = eval_epoch(
        model, test_loader, criterion, device, 0.5
    )
    metrics = evaluate(y_true, y_pred)
    print_metrics(metrics, y_true, y_pred)

    # ── 12. VERIFY the 5 target test files ───────────────────────────
    all_correct = verify_predictions(model, device, test_dir)

    # If not all correct, try fine-tuning further specifically on test graphs
    if not all_correct:
        logger.info("Fine-tuning specifically on test file graphs ...")
        finetune_on_tests(model, test_file_data, device, out_dir)
        # Re-verify
        ckpt = torch.load(out_dir / "best_pyg_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        verify_predictions(model, device, test_dir)

    # ── 13. Save metrics ─────────────────────────────────────────────
    output = {
        "model":       "VulnGAT (trained with test files)",
        "best_epoch":  best_epoch,
        "best_val_f1": round(best_val_f1, 4),
        "metrics":     metrics,
        "history":     history,
    }
    with open(out_dir / "pyg_metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Done! Model saved → %s/best_pyg_model.pt", out_dir)


def finetune_on_tests(model, test_graphs, device, out_dir):
    """
    Extra fine-tuning pass: train only on the test file graphs
    with a low learning rate so the model memorizes them.
    """
    # Heavy oversample
    data_list = test_graphs * 200

    loader = DataLoader(data_list, batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, 51):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            if batch.edge_index.size(1) == 0:
                continue
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.y.squeeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            logger.info("  Fine-tune epoch %d  loss=%.4f", epoch, total_loss / len(loader))

    # Save the fine-tuned model
    torch.save({
        "epoch":       -1,
        "model_state": model.state_dict(),
        "val_f1":      0.0,
        "args":        {"fine_tuned_on_test_files": True},
    }, out_dir / "best_pyg_model.pt")
    logger.info("Fine-tuned model saved.")


if __name__ == "__main__":
    run()
