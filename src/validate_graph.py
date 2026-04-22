import networkx as nx
import json
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from const_graph import build_heterogeneous_graph

def validate_graph(G, name="", quiet=False):
    issues = []
    tag = f"[{name}] " if name else ""

    block_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "block"]
    inst_nodes  = [n for n, d in G.nodes(data=True) if d.get("node_type") == "instruction"]

    #every instruction must have exactly one parent block
    for inst in inst_nodes:
        parents = [u for u, v, d in G.in_edges(inst, data=True)
                   if d.get("edge_type") == "contains"]
        if len(parents) != 1:
            issues.append(f"{tag}Inst '{inst}' has {len(parents)} parent blocks (expected 1)")

    #every block must contain at least one instruction
    for block in block_nodes:
        children = [v for u, v, d in G.out_edges(block, data=True)
                    if d.get("edge_type") == "contains"]
        if not children:
            issues.append(f"{tag}Block '{block}' has no instructions")

    # no CFG self-loops
    for u, v, d in G.edges(data=True):
        if u == v and d.get("edge_type") == "control_flow":
            issues.append(f"{tag}CFG self-loop at '{u}'")

    func_blocks: dict[str, list[str]] = {}
    for block in block_nodes:
        func_name = G.nodes[block].get("func", "unknown")
        func_blocks.setdefault(func_name, []).append(block)

    for func_name, fblocks in func_blocks.items():
        # build subgraph of block nodes+their instr children
        inst_of_func = [
            v for b in fblocks
            for _, v, d in G.out_edges(b, data=True)
            if d.get("edge_type") == "contains"
        ]
        subgraph = G.subgraph(list(fblocks) + list(inst_of_func))
        if subgraph.number_of_nodes() > 1 and not nx.is_weakly_connected(subgraph):
            issues.append(f"{tag}Function '{func_name}' subgraph is not weakly connected")

    if not quiet:
        if not issues:
            print(f"[OK]   {name or 'Graph'}")
        else:
            for issue in issues:
                print(f"[FAIL] {issue}")

    return len(issues) == 0, issues


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate graphs built from parsed IR JSON files")
    parser.add_argument("--input", type=str, default="data/parsed_ir",
                        help="Directory containing .json files (default: data/parsed_ir)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only validate the first N files (default: all)")
    args = parser.parse_args()

    ir_dir = args.input
    files  = sorted(glob.glob(os.path.join(ir_dir, "*.json")))
    if args.limit:
        files = random.sample(files, min(args.limit, len(files)))

    print(f"Validating {len(files)} files in '{ir_dir}'...\n")
    passed = failed = 0
    failed_list = []

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            ir_data = json.load(f)

        G    = build_heterogeneous_graph(ir_data)
        name = os.path.basename(fpath)
        ok, _   = validate_graph(G, name=name)
        if ok:
            passed += 1
        else:
            failed += 1
            failed_list.append(name)

    print(f"\nSUMMARY")
    print(f"Total : {len(files)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    if failed_list:
        print("\nFailed files:")
        for f in failed_list:
            print(f"  {f}")