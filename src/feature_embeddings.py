"""
feature_embeddings.py
=====================
Three-level feature embedding for heterogeneous LLVM IR graphs.

Features:
  1. Instruction-level  : opcode index + operand type flags + position
  2. Block-level        : aggregated opcode bag + CFG degree + size
  3. Graph-level        : structural stats + opcode unigram distribution
  4. Opcode2Vec         : learned dense opcode embeddings via Word2Vec

Usage:
    python3 src/feature_embeddings.py --input data/parsed_ir --output data/embeddings
"""

import os
import re
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("feature_embeddings")

# ── Label scheme ────────────────────────────────────────────────────────
LABEL_IDS   = [0,  2,       3,       17,      18,      25]
CLASS_NAMES = ["non-vulnerable","CWE-476","CWE-119","CWE-191","CWE-190","CWE-369"]
NUM_CLASSES = len(CLASS_NAMES)
_LABEL_TO_INDEX = {lid: idx for idx, lid in enumerate(LABEL_IDS)}

CWE_FILENAME_MAP = {
    "CWE476": 2,
    "CWE119": 3,
    "CWE191": 17,
    "CWE190": 18,
    "CWE369": 25,
}

# ── Known LLVM opcodes (ordered list → index lookup) ────────────────────
KNOWN_OPS = [
    "ret","br","switch","indirectbr","invoke","resume","unreachable","callbr",
    "add","fadd","sub","fsub","mul","fmul","udiv","sdiv","fdiv","urem","srem","frem",
    "shl","lshr","ashr","and","or","xor",
    "alloca","load","store","getelementptr","fence","cmpxchg","atomicrmw",
    "trunc","zext","sext","fptrunc","fpext","fptoui","fptosi","uitofp","sitofp",
    "ptrtoint","inttoptr","bitcast","addrspacecast",
    "icmp","fcmp","phi","select","call","va_arg","landingpad",
    "extractelement","insertelement","shufflevector","extractvalue","insertvalue",
    "freeze",
]
OP_TO_IDX = {op: i + 1 for i, op in enumerate(KNOWN_OPS)}  # 0 = unknown
NUM_OPS   = len(KNOWN_OPS) + 1

# Operand type flags (detected from instruction text)
OPERAND_TYPES = ["i1","i8","i16","i32","i64","float","double","ptr","i8*","i32*","i64*"]


# ════════════════════════════════════════════════════════════════════════
# 1. HELPERS
# ════════════════════════════════════════════════════════════════════════

def load_ir_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_label_vector(filename: str) -> np.ndarray:
    """Extract multi-hot label vector from filename."""
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    matched = False
    for cwe_str, lid in CWE_FILENAME_MAP.items():
        if cwe_str in filename:
            vec[_LABEL_TO_INDEX[lid]] = 1.0
            matched = True
    if not matched:
        vec[0] = 1.0  # non-vulnerable
    return vec


def _get_opcode(inst: dict) -> str:
    """Safely extract opcode string from instruction dict."""
    return inst.get("opcode", "").lower().strip()


def _get_blocks(func: dict) -> list:
    """Get blocks from a function dict using correct key."""
    return func.get("blocks", [])  # ← your JSON uses 'blocks' not 'basic_blocks'


# ════════════════════════════════════════════════════════════════════════
# 2. INSTRUCTION-LEVEL FEATURES
# ════════════════════════════════════════════════════════════════════════

def instruction_features(inst: dict, position_in_block: float) -> np.ndarray:
    """
    Per-instruction feature vector (18-dim):
      [0]      opcode index (normalized 0–1)
      [1]      position in block (0=first, 1=last)
      [2–12]   operand type flags from instruction text
      [13]     is_terminator
      [14]     is_memory_op
      [15]     is_call
      [16]     is_comparison
      [17]     is_cast
    """
    opcode = _get_opcode(inst)
    op_idx = OP_TO_IDX.get(opcode, 0) / NUM_OPS  # normalized

    # Use 'text' key (your actual key) for operand type detection
    raw = inst.get("text", "")
    type_flags = np.array(
        [1.0 if t in raw else 0.0 for t in OPERAND_TYPES],
        dtype=np.float32,
    )

    is_terminator = float(opcode in {"ret","br","switch","indirectbr","invoke","resume","callbr"})
    is_memory     = float(opcode in {"load","store","alloca","getelementptr","cmpxchg","atomicrmw"})
    is_call       = float(opcode == "call")
    is_comparison = float(opcode in {"icmp","fcmp"})
    is_cast       = float(opcode in {"trunc","zext","sext","fptrunc","fpext","fptoui",
                                      "fptosi","uitofp","sitofp","ptrtoint","inttoptr",
                                      "bitcast","addrspacecast"})

    return np.concatenate([
        [op_idx, position_in_block],
        type_flags,
        [is_terminator, is_memory, is_call, is_comparison, is_cast],
    ]).astype(np.float32)  # 18-dim


INST_FEATURE_DIM = 18


# ════════════════════════════════════════════════════════════════════════
# 3. BLOCK-LEVEL FEATURES
# ════════════════════════════════════════════════════════════════════════

def block_features(block: dict, cfg_in_degree: int, cfg_out_degree: int) -> np.ndarray:
    """
    Per-block feature vector (5 + NUM_OPS dims):
      [0]     block size normalized
      [1]     CFG in-degree normalized
      [2]     CFG out-degree normalized
      [3]     is_entry block
      [4]     is_exit block
      [5–N]   opcode unigram bag (normalized by block size)
    """
    instructions = block.get("instructions", [])
    n = max(len(instructions), 1)

    op_bag = np.zeros(NUM_OPS, dtype=np.float32)
    for inst in instructions:
        opcode = _get_opcode(inst)
        idx = OP_TO_IDX.get(opcode, 0)
        op_bag[idx] += 1.0
    op_bag /= n  # normalize

    structural = np.array([
        min(n, 100) / 100.0,
        min(cfg_in_degree,  10) / 10.0,
        min(cfg_out_degree, 10) / 10.0,
        float(cfg_in_degree  == 0),   # entry block
        float(cfg_out_degree == 0),   # exit block
    ], dtype=np.float32)

    return np.concatenate([structural, op_bag])


BLOCK_FEATURE_DIM = 5 + NUM_OPS


# ════════════════════════════════════════════════════════════════════════
# 4. CFG DEGREE COMPUTATION (derived from block terminators)
# ════════════════════════════════════════════════════════════════════════

def compute_cfg_degrees(blocks: list) -> tuple[dict, dict]:
    """
    Derive CFG in/out degrees from terminator instructions.
    Looks for 'br' instructions with label references in their text.
    Returns (in_degree, out_degree) dicts keyed by block label.
    """
    in_deg  = Counter()
    out_deg = Counter()

    # Map label → index for quick lookup
    label_set = {b.get("label", f"b{i}") for i, b in enumerate(blocks)}

    for block in blocks:
        label = block.get("label", "")
        insts = block.get("instructions", [])
        if not insts:
            continue

        # Look at terminator (last instruction)
        terminator = insts[-1]
        op   = _get_opcode(terminator)
        text = terminator.get("text", "")

        # Count successors by finding label references in terminator text
        successors = set()
        if op in {"br", "switch", "indirectbr", "invoke", "callbr"}:
            # Find all %label or label references in text
            for token in re.findall(r'%[\w.]+', text):
                candidate = token.lstrip('%')
                if candidate in label_set and candidate != label:
                    successors.add(candidate)

        out_deg[label] = len(successors)
        for s in successors:
            in_deg[s] += 1

    return dict(in_deg), dict(out_deg)


# ════════════════════════════════════════════════════════════════════════
# 5. GRAPH-LEVEL FEATURES
# ════════════════════════════════════════════════════════════════════════

def graph_features(ir_data: dict) -> np.ndarray:
    """
    Whole-graph feature vector (6 + NUM_OPS dims):
      [0]   total instruction count  (log-normalized)
      [1]   total block count        (log-normalized)
      [2]   total function count     (normalized)
      [3]   avg instructions per block
      [4]   avg blocks per function
      [5]   cyclomatic complexity estimate
      [6–N] opcode unigram distribution (normalized)
    """
    functions = ir_data.get("functions", [])
    n_funcs   = max(len(functions), 1)

    all_insts  = []
    all_blocks = []
    cfg_edges  = 0
    op_counts  = np.zeros(NUM_OPS, dtype=np.float32)

    for func in functions:
        blocks = _get_blocks(func)
        all_blocks.extend(blocks)

        in_deg, out_deg = compute_cfg_degrees(blocks)
        cfg_edges += sum(out_deg.values())

        for block in blocks:
            insts = block.get("instructions", [])
            all_insts.extend(insts)
            for inst in insts:
                opcode = _get_opcode(inst)
                op_counts[OP_TO_IDX.get(opcode, 0)] += 1

    n_blocks = max(len(all_blocks), 1)
    n_insts  = max(len(all_insts),  1)

    cyclomatic = (cfg_edges - n_blocks + 2 * n_funcs) / n_funcs

    op_dist = op_counts / op_counts.sum() if op_counts.sum() > 0 else op_counts

    structural = np.array([
        np.log1p(n_insts)  / 10.0,
        np.log1p(n_blocks) / 5.0,
        min(n_funcs, 50)   / 50.0,
        min(n_insts / n_blocks, 100) / 100.0,
        min(n_blocks / n_funcs, 50)  / 50.0,
        min(max(cyclomatic, 0), 50)  / 50.0,
    ], dtype=np.float32)

    return np.concatenate([structural, op_dist])


GRAPH_FEATURE_DIM = 6 + NUM_OPS


# ════════════════════════════════════════════════════════════════════════
# 6. OPCODE2VEC — learned dense opcode embeddings
# ════════════════════════════════════════════════════════════════════════

def collect_opcode_sequences(ir_data: dict) -> list:
    """Extract one opcode sequence per function (like sentences for Word2Vec)."""
    sequences = []
    for func in ir_data.get("functions", []):
        seq = []
        for block in _get_blocks(func):           # ← fixed key
            for inst in block.get("instructions", []):
                op = _get_opcode(inst)
                if op in OP_TO_IDX:
                    seq.append(op)
        if seq:
            sequences.append(seq)
    return sequences


def train_opcode2vec(all_sequences: list, embed_dim: int = 32) -> dict:
    """
    Train skip-gram Word2Vec on opcode sequences.
    Returns dict: opcode → np.ndarray(embed_dim,)
    """
    if not all_sequences:
        logger.error("No opcode sequences collected — check JSON structure.")
        return {}

    try:
        from gensim.models import Word2Vec
        logger.info("Training Opcode2Vec (embed_dim=%d) on %d sequences ...",
                    embed_dim, len(all_sequences))
        model = Word2Vec(
            sentences=all_sequences,
            vector_size=embed_dim,
            window=5,
            min_count=1,      # ← lowered from 2 so rare opcodes are included
            workers=4,
            sg=1,             # skip-gram
            epochs=10,
            seed=42,
        )
        embeddings = {op: model.wv[op] for op in model.wv.index_to_key}
        logger.info("Opcode2Vec trained. Vocabulary: %d opcodes", len(embeddings))
        return embeddings
    except ImportError:
        logger.warning("gensim not installed. pip install gensim --break-system-packages")
        return {}


def get_opcode2vec_embedding(
    ir_data: dict,
    opcode_embeddings: dict,
    embed_dim: int = 32,
) -> np.ndarray:
    """
    Mean + max pool Opcode2Vec embeddings across all instructions.
    Output: 2 * embed_dim vector.
    """
    vecs = []
    for func in ir_data.get("functions", []):
        for block in _get_blocks(func):           # ← fixed key
            for inst in block.get("instructions", []):
                op = _get_opcode(inst)
                if op in opcode_embeddings:
                    vecs.append(opcode_embeddings[op])

    if not vecs:
        return np.zeros(embed_dim * 2, dtype=np.float32)

    mat       = np.stack(vecs, axis=0)
    mean_pool = mat.mean(axis=0)
    max_pool  = mat.max(axis=0)
    return np.concatenate([mean_pool, max_pool]).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
# 7. COMBINED EMBEDDING PER GRAPH
# ════════════════════════════════════════════════════════════════════════

def embed_graph(ir_data: dict, opcode_embeddings: dict, embed_dim: int = 32) -> dict:
    """
    Build all embeddings for a single graph.

    Returns dict:
      graph_vec       : graph-level structural + opcode dist vector
      opcode2vec_vec  : mean+max pooled Opcode2Vec vector
      combined_vec    : concatenation of both (for RF/MLP use)
      block_vecs      : list of block-level feature vectors
      inst_vecs       : list of (block_idx, inst_idx, feature_vec) tuples
    """
    g_vec    = graph_features(ir_data)
    o2v      = get_opcode2vec_embedding(ir_data, opcode_embeddings, embed_dim)
    combined = np.concatenate([g_vec, o2v])

    block_vecs = []
    inst_vecs  = []

    for func in ir_data.get("functions", []):
        blocks = _get_blocks(func)                # ← fixed key
        in_deg, out_deg = compute_cfg_degrees(blocks)

        for b_idx, block in enumerate(blocks):
            label = block.get("label", f"b{b_idx}")
            in_d  = in_deg.get(label,  0)
            out_d = out_deg.get(label, 0)
            b_vec = block_features(block, in_d, out_d)
            block_vecs.append(b_vec)

            insts = block.get("instructions", [])
            n     = max(len(insts), 1)
            for i_idx, inst in enumerate(insts):
                pos   = i_idx / n
                i_vec = instruction_features(inst, pos)
                inst_vecs.append((b_idx, i_idx, i_vec))

    return {
        "graph_vec":      g_vec,
        "opcode2vec_vec": o2v,
        "combined_vec":   combined,
        "block_vecs":     block_vecs,
        "inst_vecs":      inst_vecs,
    }


# ════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ════════════════════════════════════════════════════════════════════════

def process_all(input_dir: Path, output_dir: Path, embed_dim: int = 32):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        logger.error("No .json files found in %s", input_dir)
        return

    logger.info("Found %d JSON files", len(json_files))

    # ── Pass 1: collect opcode sequences for Opcode2Vec ─────────────
    logger.info("Pass 1: Collecting opcode sequences ...")
    all_sequences = []
    ir_cache = {}

    for jf in json_files:
        try:
            data = load_ir_json(jf)
            ir_cache[jf.name] = data
            seqs = collect_opcode_sequences(data)
            all_sequences.extend(seqs)
        except Exception as e:
            logger.warning("Failed to load %s: %s", jf.name, e)

    logger.info("Collected %d opcode sequences from %d files",
                len(all_sequences), len(ir_cache))

    # Train Opcode2Vec
    opcode_embeddings = train_opcode2vec(all_sequences, embed_dim=embed_dim)

    # ── Pass 2: embed each graph ─────────────────────────────────────
    logger.info("Pass 2: Embedding all graphs ...")
    results   = []
    labels    = []
    filenames = []
    failed    = []

    for jf in json_files:
        if jf.name not in ir_cache:
            continue
        try:
            ir_data  = ir_cache[jf.name]
            emb      = embed_graph(ir_data, opcode_embeddings, embed_dim)
            label    = get_label_vector(jf.stem)

            results.append(emb["combined_vec"])
            labels.append(label)
            filenames.append(jf.name)

        except Exception as e:
            logger.warning("Failed to embed %s: %s", jf.name, e)
            failed.append(jf.name)

    if not results:
        logger.error("No graphs embedded successfully. Check your JSON structure.")
        return

    # ── Save outputs ─────────────────────────────────────────────────
    X = np.stack(results, axis=0)
    y = np.stack(labels,  axis=0)

    np.save(output_dir / "X_embeddings.npy", X)
    np.save(output_dir / "y_labels.npy",     y)

    with open(output_dir / "filenames.json", "w") as f:
        json.dump(filenames, f, indent=2)

    if opcode_embeddings:
        o2v_dict = {k: v.tolist() for k, v in opcode_embeddings.items()}
        with open(output_dir / "opcode2vec.json", "w") as f:
            json.dump(o2v_dict, f, indent=2)

    logger.info("─" * 55)
    logger.info("Done!")
    logger.info("  Samples embedded : %d", len(results))
    logger.info("  Embedding matrix : %s  → X_embeddings.npy", X.shape)
    logger.info("  Label matrix     : %s  → y_labels.npy",     y.shape)
    logger.info("  Failed files     : %d", len(failed))
    logger.info("  Dims — graph: %d  |  opcode2vec: %d  |  combined: %d",
                GRAPH_FEATURE_DIM, embed_dim * 2, X.shape[1])

    if failed:
        with open(output_dir / "failed_embeddings.txt", "w") as f:
            f.writelines(n + "\n" for n in failed)
        logger.info("  Failed list → failed_embeddings.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature embeddings for LLVM IR graphs")
    parser.add_argument("--input",     type=Path, default=Path("data/parsed_ir"),
                        help="Directory with parsed IR .json files")
    parser.add_argument("--output",    type=Path, default=Path("data/embeddings"),
                        help="Output directory for embeddings")
    parser.add_argument("--embed_dim", type=int,  default=32,
                        help="Opcode2Vec embedding dimension (default: 32)")
    args = parser.parse_args()

    process_all(args.input, args.output, args.embed_dim)