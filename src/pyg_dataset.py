"""
pyg_dataset.py  (v4 — function-level graphs, good() as non-vuln)
===========================================================
Builds ONE PyG graph per FUNCTION instead of one per file.
Labels are assigned based on function name, not filename:

  _bad()  function  → CWE label extracted from filename
  good*() function  → non-vulnerable (safe version of vulnerable code;
                       critical negative examples with arithmetic ops)
  io      functions → non-vulnerable (label 0)
  is_declaration    → SKIPPED (external/STL functions)

Changes from v3:
  - good*() functions are now INCLUDED as non-vulnerable training data.
    They contain arithmetic/pointer operations with proper bounds checking,
    teaching the model that arithmetic alone ≠ vulnerability.
  - CWE-119 removed from class list (0 samples in dataset)

Result:
  ~1150 vulnerable graphs   (one per _bad function)
  ~4000+ non-vulnerable graphs (_io + good* functions)
  Model sees safe arithmetic code → no more CWE-369 bias
"""

import re
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from const_graph import _get_result, _get_operands, _get_cfg_targets

logger = logging.getLogger("pyg_dataset")

# ── Label scheme (CWE-119 removed — 0 samples in dataset) ───────────────
CLASS_NAMES = ["non-vulnerable", "CWE-476", "CWE-191", "CWE-190", "CWE-369"]
NUM_CLASSES  = len(CLASS_NAMES)

CWE_FILENAME_MAP = {
    "CWE476": 1,
    "CWE191": 2,
    "CWE190": 3,
    "CWE369": 4,
}

# ── Opcode vocabulary ─────────────────────────────────────────────────────
KNOWN_OPS = [
    "ret", "br", "switch", "indirectbr", "invoke", "resume", "unreachable", "callbr",
    "add", "fadd", "sub", "fsub", "mul", "fmul", "udiv", "sdiv", "fdiv",
    "urem", "srem", "frem", "shl", "lshr", "ashr", "and", "or", "xor",
    "alloca", "load", "store", "getelementptr", "fence", "cmpxchg", "atomicrmw",
    "trunc", "zext", "sext", "fptrunc", "fpext", "fptoui", "fptosi",
    "uitofp", "sitofp", "ptrtoint", "inttoptr", "bitcast", "addrspacecast",
    "icmp", "fcmp", "phi", "select", "call", "va_arg", "landingpad",
    "extractelement", "insertelement", "shufflevector",
    "extractvalue", "insertvalue", "freeze",
]
OP_TO_IDX = {op: i + 1 for i, op in enumerate(KNOWN_OPS)}
NUM_OPS   = len(KNOWN_OPS) + 1

OPERAND_TYPES    = ["i1", "i8", "i16", "i32", "i64", "float", "double",
                    "ptr", "i8*", "i32*", "i64*"]
INST_FEATURE_DIM = 22


# ════════════════════════════════════════════════════════════════════════
# 1. LABELING LOGIC
# ════════════════════════════════════════════════════════════════════════

def get_cwe_from_filename(filename: str) -> Optional[int]:
    """Extract CWE class index from filename. Returns None if not found."""
    for cwe_str, idx in CWE_FILENAME_MAP.items():
        if cwe_str in filename:
            return idx
    return None


def get_function_label(func_name: str, file_cwe: Optional[int]) -> Optional[np.ndarray]:
    """
    Determine label vector for a function based on its name.

    Rules:
      - Contains '_bad'   → vulnerable, use file_cwe as label
      - Contains 'good'   → non-vulnerable (safe version of vulnerable code;
                             these include arithmetic/pointer ops with proper
                             bounds checking — critical negative examples)
      - file_cwe is None  → _io file → non-vulnerable
      - Anything else in a CWE file → SKIP (helper/setup functions)

    Returns None to SKIP the function entirely.
    """
    name_lower = func_name.lower()
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)

    # Vulnerable function
    if "_bad" in name_lower:
        if file_cwe is not None:
            vec[file_cwe] = 1.0
            return vec
        else:
            return None  # _bad but no CWE in filename — skip

    # good*() functions — safe rewrites of vulnerable code.
    # These are CRITICAL for training: they contain arithmetic/pointer
    # operations with proper bounds checking, teaching the model that
    # arithmetic alone ≠ vulnerability.
    if "good" in name_lower:
        vec[0] = 1.0  # non-vulnerable
        return vec

    # _io files (file_cwe is None) → genuine non-vulnerable code
    if file_cwe is None:
        vec[0] = 1.0
        return vec

    # Any other helper function inside a CWE file → skip
    return None


# ════════════════════════════════════════════════════════════════════════
# 2. INSTRUCTION FEATURES
# ════════════════════════════════════════════════════════════════════════

def instruction_features(inst: dict, position_in_block: float) -> np.ndarray:
    """
    Per-instruction feature vector (22 dims):
      [0]      opcode index (normalized)
      [1]      position in block
      [2-12]   operand type flags (11 types)
      [13]     is_terminator
      [14]     is_memory_op
      [15]     is_call
      [16]     is_comparison
      [17]     is_cast
      --- NEW: vulnerability-specific features ---
      [18]     is_add_or_mul  (CWE-190 overflow signal)
      [19]     is_sub         (CWE-191 underflow signal)
      [20]     is_div         (CWE-369 divzero signal)
      [21]     uses_i32       (32-bit arithmetic is overflow-prone)
    """
    opcode = inst.get("opcode", "").lower().strip()
    op_idx = OP_TO_IDX.get(opcode, 0) / NUM_OPS
    text   = inst.get("text", "")

    type_flags = np.array(
        [1.0 if t in text else 0.0 for t in OPERAND_TYPES],
        dtype=np.float32,
    )
    flags = np.array([
        float(opcode in {"ret", "br", "switch", "indirectbr", "invoke", "resume", "callbr"}),
        float(opcode in {"load", "store", "alloca", "getelementptr", "cmpxchg", "atomicrmw"}),
        float(opcode == "call"),
        float(opcode in {"icmp", "fcmp"}),
        float(opcode in {"trunc", "zext", "sext", "fptrunc", "fpext", "fptoui",
                         "fptosi", "uitofp", "sitofp", "ptrtoint", "inttoptr",
                         "bitcast", "addrspacecast"}),
    ], dtype=np.float32)

    # Vulnerability-specific features
    vuln_flags = np.array([
        float(opcode in {"add", "mul", "fadd", "fmul"}),   # CWE-190 overflow
        float(opcode in {"sub", "fsub"}),                   # CWE-191 underflow
        float(opcode in {"sdiv", "udiv", "fdiv",            # CWE-369 divzero
                         "srem", "urem", "frem"}),
        float("i32" in text and opcode in {"add", "sub",    # 32-bit = overflow-prone
                                           "mul", "sdiv", "udiv"}),
    ], dtype=np.float32)

    return np.concatenate([[op_idx, position_in_block], type_flags, flags, vuln_flags])


# ════════════════════════════════════════════════════════════════════════
# 3. SINGLE-FUNCTION GRAPH BUILDER
# ════════════════════════════════════════════════════════════════════════

def build_function_graph(func: dict, label_vec: np.ndarray) -> Optional[Data]:
    """
    Build a PyG graph for a single function.
    Nodes = instructions, Edges = sequential + data_flow + cfg
    """
    blocks = func.get("blocks", [])
    if not blocks:
        return None

    all_features  = []
    edges         = []
    edge_types    = []

    result_to_node: dict[str, int] = {}
    block_ranges:   dict[str, tuple[int, int]] = {}
    node_idx = 0

    # ── Pass 1: nodes + sequential edges + SSA def map ──────────────
    for block in blocks:
        blabel = block.get("label", f"b{node_idx}")
        insts  = block.get("instructions", [])
        if not insts:
            continue

        block_start = node_idx
        n = len(insts)

        for i, inst in enumerate(insts):
            pos  = i / max(n - 1, 1)
            feat = instruction_features(inst, pos)
            all_features.append(feat)

            result = _get_result(inst.get("text", ""))
            if result:
                result_to_node[result] = node_idx

            if i > 0:
                edges.append((node_idx - 1, node_idx))
                edge_types.append(0)  # sequential

            node_idx += 1

        block_ranges[blabel] = (block_start, node_idx - 1)

    if node_idx == 0:
        return None

    # ── Pass 2: data flow edges ──────────────────────────────────────
    cur_idx = 0
    for block in blocks:
        for inst in block.get("instructions", []):
            text   = inst.get("text", "")
            result = _get_result(text)

            for operand in _get_operands(text):
                if operand == result:
                    continue
                if operand in result_to_node:
                    src = result_to_node[operand]
                    dst = cur_idx
                    if src != dst:
                        edges.append((src, dst))
                        edge_types.append(1)  # data_flow

            cur_idx += 1

    # ── Pass 3: CFG edges ────────────────────────────────────────────
    for block in blocks:
        blabel     = block.get("label", "")
        insts      = block.get("instructions", [])
        if not insts:
            continue
        terminator = insts[-1]
        opcode     = terminator.get("opcode", "")
        text       = terminator.get("text",   "")

        for target_label in _get_cfg_targets(text, opcode):
            if target_label in block_ranges and blabel in block_ranges:
                src = block_ranges[blabel][1]
                dst = block_ranges[target_label][0]
                if src != dst:
                    edges.append((src, dst))
                    edge_types.append(2)  # cfg

    # ── Assemble ─────────────────────────────────────────────────────
    x = torch.tensor(np.stack(all_features), dtype=torch.float32)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros(0,      dtype=torch.long)

    y = torch.tensor(label_vec, dtype=torch.float32).unsqueeze(0)

    return Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = y,
        num_nodes  = x.size(0),
    )


# ════════════════════════════════════════════════════════════════════════
# 4. DATASET CLASS
# ════════════════════════════════════════════════════════════════════════

class IRGraphDataset(Dataset):
    """
    PyG Dataset — one graph per function (not per file).
    Cache keys: {filestem}__{funcname}.pt
    """

    def __init__(self, json_dir: str, cache_dir: str = "/home/jlux1/pyg_cache_v3"):
        self.json_dir  = Path(json_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.json_files = sorted(self.json_dir.glob("*.json"))
        logger.info("Found %d JSON files in %s", len(self.json_files), json_dir)

        self._graph_paths: list[Path] = []
        self._process_all()

        super().__init__()

    def _safe_cache_key(self, filestem: str, func_name: str) -> str:
        safe_func = re.sub(r'[^\w]', '_', func_name)[:80]
        return f"{filestem}__{safe_func}.pt"

    def _process_all(self):
        processed = skipped = failed = 0
        label_counts = np.zeros(NUM_CLASSES, dtype=int)

        for jf in self.json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    ir_data = json.load(f)

                file_cwe = get_cwe_from_filename(jf.stem)

                for func in ir_data.get("functions", []):
                    if func.get("is_declaration", False):
                        continue

                    func_name = func.get("name", "")
                    blocks    = func.get("blocks", [])
                    total_insts = sum(
                        len(b.get("instructions", [])) for b in blocks
                    )
                    if total_insts < 3:
                        skipped += 1
                        continue

                    label_vec = get_function_label(func_name, file_cwe)
                    if label_vec is None:
                        skipped += 1
                        continue

                    cache_key  = self._safe_cache_key(jf.stem, func_name)
                    cache_path = self.cache_dir / cache_key

                    if cache_path.exists():
                        self._graph_paths.append(cache_path)
                        processed += 1
                        continue

                    graph = build_function_graph(func, label_vec)
                    if graph is None or graph.num_nodes == 0:
                        skipped += 1
                        continue

                    torch.save(graph, cache_path)
                    self._graph_paths.append(cache_path)
                    label_counts += label_vec.astype(int)
                    processed += 1

            except Exception as e:
                logger.warning("Failed %s: %s", jf.name, e)
                failed += 1

        logger.info("Dataset ready: %d function graphs (%d skipped, %d failed)",
                    processed, skipped, failed)
        logger.info("Label distribution (new graphs only):")
        for i, name in enumerate(CLASS_NAMES):
            logger.info("  %-20s %d", name, label_counts[i])

    def len(self):
        return len(self._graph_paths)

    def get(self, idx):
        return torch.load(self._graph_paths[idx], weights_only=False)


# ════════════════════════════════════════════════════════════════════════
# 5. INFERENCE HELPER  (used by predict.py)
# ════════════════════════════════════════════════════════════════════════

def build_pyg_graph(ir_data: dict, filename: str) -> Optional[Data]:
    """
    For inference: build a single graph from all non-declaration functions
    in the file, merged into one graph.
    """
    all_features  = []
    edges         = []
    edge_types    = []
    result_to_node: dict[str, int] = {}
    block_ranges:   dict[str, tuple[int, int]] = {}
    node_idx = 0

    for func in ir_data.get("functions", []):
        if func.get("is_declaration", False):
            continue

        blocks = func.get("blocks", [])

        for block in blocks:
            blabel = f"{func['name']}_{block.get('label', f'b{node_idx}')}"
            insts  = block.get("instructions", [])
            if not insts:
                continue

            block_start = node_idx
            n = len(insts)

            for i, inst in enumerate(insts):
                pos  = i / max(n - 1, 1)
                feat = instruction_features(inst, pos)
                all_features.append(feat)

                result = _get_result(inst.get("text", ""))
                if result:
                    result_to_node[result] = node_idx

                if i > 0:
                    edges.append((node_idx - 1, node_idx))
                    edge_types.append(0)

                node_idx += 1

            block_ranges[blabel] = (block_start, node_idx - 1)

        for block in blocks:
            for inst in block.get("instructions", []):
                text   = inst.get("text", "")
                result = _get_result(text)
                for operand in _get_operands(text):
                    if operand == result:
                        continue
                    if operand in result_to_node:
                        src = result_to_node[operand]
                        dst = node_idx - 1
                        if src != dst:
                            edges.append((src, dst))
                            edge_types.append(1)

        for block in blocks:
            blabel     = f"{func['name']}_{block.get('label', '')}"
            insts      = block.get("instructions", [])
            if not insts:
                continue
            terminator = insts[-1]
            for target in _get_cfg_targets(
                terminator.get("text", ""), terminator.get("opcode", "")
            ):
                full_target = f"{func['name']}_{target}"
                if full_target in block_ranges and blabel in block_ranges:
                    src = block_ranges[blabel][1]
                    dst = block_ranges[full_target][0]
                    if src != dst:
                        edges.append((src, dst))
                        edge_types.append(2)

    if node_idx == 0:
        return None

    x = torch.tensor(np.stack(all_features), dtype=torch.float32)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros(0,      dtype=torch.long)

    label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    cwe = get_cwe_from_filename(filename)
    if cwe:
        label_vec[cwe] = 1.0
    else:
        label_vec[0] = 1.0

    return Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = torch.tensor(label_vec).unsqueeze(0),
        num_nodes  = x.size(0),
    )


# ════════════════════════════════════════════════════════════════════════
# 6. QUICK TEST
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    dataset = IRGraphDataset("data/parsed_ir", cache_dir="/home/jlux1/pyg_cache_v3")

    print(f"\nDataset size : {len(dataset)}")
    print(f"\nLabel distribution across ALL graphs:")

    counts = np.zeros(NUM_CLASSES, dtype=int)
    for i in range(len(dataset)):
        g = dataset[i]
        counts += g.y.squeeze(0).numpy().astype(int)

    for i, name in enumerate(CLASS_NAMES):
        pct = 100 * counts[i] / max(len(dataset), 1)
        print(f"  {name:<20} {counts[i]:>5}  ({pct:.1f}%)")

    print(f"\nSample graph:")
    g = dataset[0]
    print(f"  Nodes : {g.num_nodes}")
    print(f"  Edges : {g.edge_index.size(1)}")
    print(f"  Label : {g.y}")
    if g.edge_attr.numel() > 0:
        t = g.edge_attr.tolist()
        print(f"  Sequential : {t.count(0)}")
        print(f"  Data flow  : {t.count(1)}")
        print(f"  CFG        : {t.count(2)}")