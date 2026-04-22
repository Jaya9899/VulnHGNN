import networkx as nx
import json
import re
import os

def _get_result(text):
    m = re.match(r'^\s*(%[\w\.]+)\s*=', text)
    return m.group(1) if m else None

def _get_operands(text):
    rhs = text.split("=", 1)[1] if "=" in text else text
    return re.findall(r'%[\w\.]+', rhs)

def _get_cfg_targets(text, opcode):
    if opcode not in ("br", "switch", "invoke"):
        return []
    return re.findall(r'label\s+%(\w+)', text)


def build_heterogeneous_graph(ir_json):
    G = nx.DiGraph()

    for func in ir_json.get("functions", []):
        if func.get("is_declaration"):
            continue   # skip external declarations, they have no blocks

        fname = func["name"]
        block_ids      = {}   # label is basically block_node_id
        result_to_inst = {}   # SSA var is the inst_node_id  (for DFG)
        for block in func.get("blocks", []):
            blabel     = block["label"]
            block_node = f"block_{fname}_{blabel}"
            G.add_node(block_node, node_type="block", func=fname, label=blabel)
            block_ids[blabel] = block_node

            for idx, inst in enumerate(block.get("instructions", [])):
                text   = inst["text"]
                opcode = inst.get("opcode", "unknown")
                result = _get_result(text)
                inst_node = f"inst_{fname}_{blabel}_{idx}"
                G.add_node(inst_node,
                           node_type="instruction",
                           opcode=opcode,
                           text=text,
                           result=result or "")

                G.add_edge(block_node, inst_node, edge_type="contains")
                if result:
                    result_to_inst[result] = inst_node
                for operand in _get_operands(text):
                    if operand in result_to_inst:
                        G.add_edge(result_to_inst[operand], inst_node,
                                   edge_type="data_flow")

        for block in func.get("blocks", []):
            blabel     = block["label"]
            block_node = block_ids[blabel]

            for inst in block.get("instructions", []):
                for target_label in _get_cfg_targets(inst["text"], inst.get("opcode","")):
                    dst = block_ids.get(target_label)
                    if dst:
                        G.add_edge(block_node, dst, edge_type="control_flow")

    return G


if __name__ == "__main__":
    import glob

    ir_dir  = os.path.join(os.path.dirname(__file__), "..", "data", "parsed_ir")
    files   = sorted(glob.glob(os.path.join(ir_dir, "*.json")))[:5]

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            ir_data = json.load(f)

        G = build_heterogeneous_graph(ir_data)

        block_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "block"]
        inst_nodes  = [n for n, d in G.nodes(data=True) if d.get("node_type") == "instruction"]
        cfg_edges   = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "control_flow"]
        dfg_edges   = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "data_flow"]

        print(f"\n{os.path.basename(fpath)}")
        print(f"Nodes : {G.number_of_nodes()}  (blocks={len(block_nodes)}, insts={len(inst_nodes)})")
        print(f"Edges : {G.number_of_edges()}  (CFG={len(cfg_edges)}, DFG={len(dfg_edges)})")