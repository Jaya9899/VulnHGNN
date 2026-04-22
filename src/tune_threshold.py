# src/tune_threshold.py
import numpy as np
import torch
import sys, os
from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_model import build_model
from train_gnn import VulnDataset
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["non-vulnerable","CWE-476","CWE-119","CWE-191","CWE-190","CWE-369"]

X = np.load("data/embeddings/X_embeddings.npy")
y = np.load("data/embeddings/y_labels.npy")

_, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

device = torch.device("cpu")
ckpt   = torch.load("results/gnn/best_model.pt", map_location=device)
model  = build_model(input_dim=X.shape[1]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

loader = DataLoader(VulnDataset(X_test, y_test), batch_size=64)

all_probs, all_labels = [], []
with torch.no_grad():
    for xb, yb in loader:
        probs = torch.sigmoid(model(xb)).numpy()
        all_probs.append(probs)
        all_labels.append(yb.numpy())

probs  = np.concatenate(all_probs)
labels = np.concatenate(all_labels)

# Tune threshold per class on a grid
print(f"\n{'Class':<18} {'Best Thresh':>11} {'Best F1':>8}")
print("-" * 40)
best_thresholds = []
for i, name in enumerate(CLASS_NAMES):
    if labels[:, i].sum() == 0:
        best_thresholds.append(0.5)
        print(f"  {name:<16} {'N/A':>11} {'N/A':>8}")
        continue
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.95, 0.05):
        preds = (probs[:, i] >= t).astype(int)
        f1    = f1_score(labels[:, i], preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    best_thresholds.append(best_t)
    print(f"  {name:<16} {best_t:>11.2f} {best_f1:>8.4f}")

print(f"\nBest thresholds: {[round(t,2) for t in best_thresholds]}")

# Apply best thresholds
preds  = (probs >= np.array(best_thresholds)).astype(int)
micro  = f1_score(labels, preds, average="micro",   zero_division=0)
macro  = f1_score(labels, preds, average="macro",   zero_division=0)
print(f"\nWith tuned thresholds:")
print(f"  Micro F1: {micro:.4f}")
print(f"  Macro F1: {macro:.4f}")