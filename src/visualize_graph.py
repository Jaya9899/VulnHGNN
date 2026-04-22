import os
import sys
import json
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from const_graph import build_heterogeneous_graph


def visualize_graph(G, func_name=None, save_path="graph_sample.png"):
    all_funcs = list({
        d["func"]
        for n, d in G.nodes(data=True)
        if d.get("node_type") == "block"
    })

    if not all_funcs:
        print("[WARN] No block nodes found in graph.")
        return

    if func_name is None:
        func_name = all_funcs[0]
        print(f"No function specified — showing: '{func_name}'")
    elif func_name not in all_funcs:
        print(f"[WARN] '{func_name}' not found. Available: {all_funcs[:5]}")
        return

    block_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "block" and d.get("func") == func_name
    ]
    inst_nodes = [
        v
        for b in block_nodes
        for _, v, d in G.out_edges(b, data=True)
        if d.get("edge_type") == "contains"
    ]

    subG = G.subgraph(block_nodes + inst_nodes).copy()

    print(f"\nFunction : {func_name}")
    print(f"Blocks   : {len(block_nodes)}")
    print(f"Insts    : {len(inst_nodes)}")
    print(f"Edges    : {subG.number_of_edges()}")
    cfg = [(u,v) for u,v,d in subG.edges(data=True) if d.get("edge_type")=="control_flow"]
    dfg = [(u,v) for u,v,d in subG.edges(data=True) if d.get("edge_type")=="data_flow"]
    print(f"  CFG edges : {len(cfg)}")
    print(f"  DFG edges : {len(dfg)}")

    pos = {}
    block_nodes_sorted = sorted(block_nodes, key=lambda n: G.nodes[n]["label"])
    x_gap   = 3.0   # horizontal spacing between blocks
    y_block = 0.0   # y level for blocks
    y_start = -1.5  # y level where instructions start

    for col, b in enumerate(block_nodes_sorted):
        x = col * x_gap
        pos[b] = (x, y_block)
        children = [
            v for _, v, d in G.out_edges(b, data=True)
            if d.get("edge_type") == "contains"
        ]
        for row, inst in enumerate(children):
            pos[inst] = (x, y_start - row * 0.6)

    node_colors = []
    for n in subG.nodes():
        if subG.nodes[n].get("node_type") == "block":
            node_colors.append("#AED6F1")  
        else:
            node_colors.append("#A9DFBF") 

    edge_colors = []
    for u, v, d in subG.edges(data=True):
        et = d.get("edge_type", "")
        if et == "control_flow":
            edge_colors.append("red")
        elif et == "data_flow":
            edge_colors.append("blue")
        else:
            edge_colors.append("#AAAAAA")  

    labels = {}
    for n, d in subG.nodes(data=True):
        if d.get("node_type") == "block":
            labels[n] = f"[{d['label']}]"
        else:
            labels[n] = d.get("opcode", "?")

    fig, ax = plt.subplots(figsize=(max(12, len(block_nodes) * 3), 10))
    nx.draw(
        subG, pos, ax=ax,
        labels=labels,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=800,
        font_size=7,
        arrows=True,
        arrowsize=12,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1"
    )

    legend = [
        mpatches.Patch(color="#AED6F1", label="Block node"),
        mpatches.Patch(color="#A9DFBF", label="Instruction node"),
        mpatches.Patch(color="red",     label="Control flow (CFG)"),
        mpatches.Patch(color="blue",    label="Data flow (DFG)"),
        mpatches.Patch(color="#AAAAAA", label="Contains"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    ax.set_title(f"Program Graph — {func_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    import argparse
    import glob as _glob

    _root = os.path.join(os.path.dirname(__file__), "..")
    default_dir = os.path.join(_root, "data", "parsed_ir")

    parser = argparse.ArgumentParser(
        description="Visualize a heterogeneous graph from parsed IR JSON"
    )
    parser.add_argument(
        "--input", type=str, default=default_dir,
        help="Path to a .json file or directory of .json files (default: data/parsed_ir)"
    )
    parser.add_argument(
        "--output", type=str, default="graph_sample.png",
        help="Output image path (default: graph_sample.png)"
    )
    args = parser.parse_args()

    # Resolve input: single file or first file in directory
    if os.path.isdir(args.input):
        candidates = sorted(_glob.glob(os.path.join(args.input, "*.json")))
        if not candidates:
            print(f"[ERROR] No .json files found in {args.input}")
            sys.exit(1)
        json_path = candidates[0]
        print(f"Using first file: {os.path.basename(json_path)}")
    else:
        json_path = args.input

    with open(json_path, "r", encoding="utf-8") as f:
        ir_data = json.load(f)

    G = build_heterogeneous_graph(ir_data)

    funcs = sorted({
        d["func"]
        for _, d in G.nodes(data=True)
        if d.get("node_type") == "block"
    })
    print(f"Functions in {os.path.basename(json_path)}:")
    for fn in funcs:
        bcount = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "block" and d.get("func") == fn)
        print(f"  {fn}  ({bcount} blocks)")

    chosen = next(
        (fn for fn in funcs
         if sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "block" and d.get("func") == fn) > 1),
        funcs[0]
    )
    visualize_graph(G, func_name=chosen, save_path=args.output)