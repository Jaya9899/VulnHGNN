"""
run_pipeline.py
===============
Batch graph construction from parsed IR JSON files.

Iterates over all .json files in the parsed_ir directory, builds
a heterogeneous graph (NetworkX DiGraph) for each using const_graph,
and saves the collection as a pickle file for downstream use
(e.g. conversion to PyTorch Geometric HeteroData).

Usage:
    python src/run_pipeline.py --parsed_dir data/parsed_ir --out data/graphs.pkl
"""

import os
import sys
import json
import glob
import pickle
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from const_graph import build_heterogeneous_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARSED_DIR = _ROOT / "data" / "parsed_ir"
DEFAULT_OUT = _ROOT / "data" / "graphs.pkl"


def build_all_graphs(parsed_dir: Path, limit: int = None):
    """
    Build heterogeneous graphs from all .json files in parsed_dir.

    Returns
    -------
    graphs : list[dict]
        Each dict has keys: 'filename', 'sample_id', 'graph' (nx.DiGraph),
        'n_nodes', 'n_edges', 'n_blocks', 'n_instructions'.
    """
    json_files = sorted(glob.glob(str(parsed_dir / "*.json")))
    if limit is not None:
        json_files = json_files[:limit]

    logger.info("Building graphs from %d JSON files in %s", len(json_files), parsed_dir)

    graphs = []
    failed = []

    for i, fpath in enumerate(json_files):
        fname = os.path.basename(fpath)
        sample_id = fname.replace(".json", "").split("_")[0]

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                ir_data = json.load(f)

            G = build_heterogeneous_graph(ir_data)

            block_nodes = sum(
                1 for _, d in G.nodes(data=True) if d.get("node_type") == "block"
            )
            inst_nodes = sum(
                1 for _, d in G.nodes(data=True)
                if d.get("node_type") == "instruction"
            )

            graphs.append({
                "filename": fname,
                "sample_id": sample_id,
                "graph": G,
                "n_nodes": G.number_of_nodes(),
                "n_edges": G.number_of_edges(),
                "n_blocks": block_nodes,
                "n_instructions": inst_nodes,
            })

        except Exception as exc:
            logger.warning("FAILED %s: %s", fname, exc)
            failed.append(fname)

        if (i + 1) % 200 == 0 or (i + 1) == len(json_files):
            logger.info(
                "  Progress: %d / %d  (failed: %d)",
                i + 1, len(json_files), len(failed),
            )

    return graphs, failed


def print_summary(graphs, failed):
    """Print a summary of the graph construction results."""
    if not graphs:
        logger.warning("No graphs were built!")
        return

    total_nodes = sum(g["n_nodes"] for g in graphs)
    total_edges = sum(g["n_edges"] for g in graphs)
    total_blocks = sum(g["n_blocks"] for g in graphs)
    total_insts = sum(g["n_instructions"] for g in graphs)

    sep = "─" * 55
    print(f"\n{sep}")
    print("  GRAPH CONSTRUCTION SUMMARY")
    print(sep)
    print(f"  Total graphs built : {len(graphs)}")
    print(f"  Failed             : {len(failed)}")
    print(f"  Total nodes        : {total_nodes}")
    print(f"  Total edges        : {total_edges}")
    print(f"  Total blocks       : {total_blocks}")
    print(f"  Total instructions : {total_insts}")
    print(f"  Avg nodes/graph    : {total_nodes / len(graphs):.1f}")
    print(f"  Avg edges/graph    : {total_edges / len(graphs):.1f}")
    print(sep)

    if failed:
        print(f"\n  Failed files ({len(failed)}):")
        for f in failed[:20]:
            print(f"    {f}")
        if len(failed) > 20:
            print(f"    ... and {len(failed) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Build heterogeneous graphs from parsed IR JSON files."
    )
    parser.add_argument(
        "--parsed_dir", type=Path, default=DEFAULT_PARSED_DIR,
        help="Directory containing parsed .json files (default: data/parsed_ir)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="Output pickle file (default: data/graphs.pkl)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N files (for testing)",
    )
    args = parser.parse_args()

    if not args.parsed_dir.exists():
        logger.error("Parsed IR directory not found: %s", args.parsed_dir)
        sys.exit(1)

    graphs, failed = build_all_graphs(args.parsed_dir, limit=args.limit)
    print_summary(graphs, failed)

    # Save to pickle
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(graphs, f)
    logger.info("Saved %d graphs → %s", len(graphs), args.out)


if __name__ == "__main__":
    main()
