import os
import re
import random
import logging
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("baseline_features")

LABEL_IDS   = [0,   2,        3,        17,       18,       25]
CLASS_NAMES = [
    "non-vulnerable",
    "CWE-476",
    "CWE-119",
    "CWE-191",
    "CWE-190",
    "CWE-369",
]
NUM_CLASSES = len(CLASS_NAMES)

_LABEL_TO_INDEX = {lid: idx for idx, lid in enumerate(LABEL_IDS)}


_OPCODE_RE = re.compile(
    r'(?:%.+\s*=\s*)?'     # optional result register
    r'([a-z][a-z0-9_.]*)',  # opcode (first lowercase token)
)

_KNOWN_OPS = {
    # Terminator
    "ret", "br", "switch", "indirectbr", "invoke", "resume",
    "unreachable", "callbr",
    # Binary
    "add", "fadd", "sub", "fsub", "mul", "fmul",
    "udiv", "sdiv", "fdiv", "urem", "srem", "frem",
    # Bitwise
    "shl", "lshr", "ashr", "and", "or", "xor",
    # Memory
    "alloca", "load", "store", "getelementptr", "fence",
    "cmpxchg", "atomicrmw",
    # Cast
    "trunc", "zext", "sext", "fptrunc", "fpext",
    "fptoui", "fptosi", "uitofp", "sitofp",
    "ptrtoint", "inttoptr", "bitcast", "addrspacecast",
    # Other
    "icmp", "fcmp", "phi", "select", "call", "va_arg",
    "landingpad", "extractelement", "insertelement",
    "shufflevector", "extractvalue", "insertvalue",
    "freeze",
}

def _get_labels_from_filename(fname: str) -> set:
    labels = set()
    if "CWE476" in fname: labels.add(2)
    if "CWE119" in fname: labels.add(3)
    if "CWE191" in fname: labels.add(17)
    if "CWE190" in fname: labels.add(18)
    if "CWE369" in fname: labels.add(25)
    
    # If no specific CWE is found (e.g., _io files), it is non-vulnerable
    if not labels:
        labels.add(0)
        
    return labels


def _extract_opcodes(ll_path: Path) -> list:
    """Return list of opcodes from a .ll file (preserving order)."""
    opcodes = []
    with open(ll_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            # Skip metadata, comments, declarations, attributes, types
            if not line or line.startswith(";") or line.startswith("!"):
                continue
            if line.startswith("declare ") or line.startswith("attributes "):
                continue
            if line.startswith("@") or line.startswith("source_filename"):
                continue
            if line.startswith("target "):
                continue
            if line.startswith("define "):
                continue
            if line == "}":
                continue
            # Try to extract an opcode
            m = _OPCODE_RE.match(line)
            if m:
                op = m.group(1)
                if op in _KNOWN_OPS:
                    opcodes.append(op)
    return opcodes


def _sample_id_from_filename(fname: str) -> str:
    base = fname.replace(".ll", "")
    parts = base.split("_")
    return parts[0]


def build_feature_matrix(
    ir_dir,
    max_samples: Optional[int] = None,
) -> tuple:
    ir_dir = Path(ir_dir)

    # ── 1. Discover .ll files ──────────────────────────────────────────
    ll_files = list(ir_dir.glob("*.ll"))
    if max_samples is not None:
        ll_files = random.sample(ll_files, min(max_samples, len(ll_files)))
    else:
        ll_files = sorted(ll_files)
    logger.info("Processing %d .ll files from %s", len(ll_files), ir_dir)

    # ── 2. Extract opcodes and labels per file ────────────────────────
    rows = []       # list of Counter dicts
    labels = []     # list of multi-hot arrays
    meta = []       # list of metadata dicts
    all_unigrams = set()
    all_bigrams = set()

    for ll_file in ll_files:
        opcodes = _extract_opcodes(ll_file)
        sid = _sample_id_from_filename(ll_file.name)
        file_labels = _get_labels_from_filename(ll_file.name)

        # Unigram counts
        uni_counts = Counter(opcodes)
        for op in uni_counts:
            all_unigrams.add(op)

        # Bigram counts
        bi_counts = Counter()
        for i in range(len(opcodes) - 1):
            bigram = f"{opcodes[i]}_{opcodes[i+1]}"
            bi_counts[bigram] += 1
            all_bigrams.add(bigram)

        # Combined row
        row = {}
        for op, cnt in uni_counts.items():
            row[f"op_{op}"] = cnt
        for bg, cnt in bi_counts.items():
            row[f"bigram_{bg}"] = cnt
        rows.append(row)

        # Multi-hot label
        y_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        for lbl in file_labels:
            if lbl in _LABEL_TO_INDEX:
                y_vec[_LABEL_TO_INDEX[lbl]] = 1.0
        labels.append(y_vec)

        meta.append({
            "filename": ll_file.name,
            "sample_id": sid,
            "n_opcodes": len(opcodes),
        })

    # ── 4. Assemble DataFrame ─────────────────────────────────────────
    X = pd.DataFrame(rows).fillna(0).astype(np.float32)

    # Sort columns: unigrams first, then bigrams
    uni_cols = sorted([c for c in X.columns if c.startswith("op_")])
    bi_cols  = sorted([c for c in X.columns if c.startswith("bigram_")])
    X = X[uni_cols + bi_cols]

    y = np.array(labels, dtype=np.float32)

    logger.info(
        "Feature matrix: %d samples × %d features (%d unigrams, %d bigrams)",
        X.shape[0], X.shape[1], len(uni_cols), len(bi_cols),
    )

    return X, y, meta
