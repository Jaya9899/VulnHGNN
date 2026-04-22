"""
localization.py — Vulnerability Localization Engine for VulnHGNN.

Identifies which LLVM IR instructions are responsible for each predicted
vulnerability using a combination of:
  1. Attention weights from GATv2Conv layers
  2. Gradient-based importance (saliency) w.r.t. predicted CWE scores
  3. Top-k selection of vulnerable instruction nodes
  4. Subgraph extraction via DFG/CFG expansion

Output per CWE:
  - Ranked list of instruction nodes with importance scores
  - Vulnerable subgraph (edges + nodes)
  - Mapping to LLVM IR line context
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger("localization")


# ════════════════════════════════════════════════════════════════════════
# 1. ATTENTION EXTRACTION
# ════════════════════════════════════════════════════════════════════════

class AttentionExtractor:
    """
    Hooks into GATv2Conv layers to capture attention coefficients
    during the forward pass.

    Usage:
        extractor = AttentionExtractor(model)
        logits = model(x, edge_index, edge_attr, batch)
        attention_scores = extractor.get_node_attention_scores(num_nodes)
        extractor.remove_hooks()
    """

    def __init__(self, model):
        self.model = model
        self._hooks = []
        self._attention_weights = []
        self._edge_indices = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on all GATv2Conv layers."""
        from torch_geometric.nn import GATv2Conv

        for name, module in self.model.named_modules():
            if isinstance(module, GATv2Conv):
                hook = module.register_forward_hook(self._capture_attention(name))
                self._hooks.append(hook)
                logger.debug("Registered attention hook on %s", name)

    def _capture_attention(self, layer_name):
        """Create a hook function that captures attention weights."""
        def hook_fn(module, inputs, output):
            # GATv2Conv returns (out, attention_weights) when
            # return_attention_weights=True. But by default it only
            # returns out. We need to extract attention from the module.
            # 
            # GATv2Conv stores alpha internally during forward pass.
            # We can access it via the module's internal state.
            # 
            # Alternative: we capture edge_index from input and
            # compute attention contribution from the output.
            if isinstance(output, tuple) and len(output) == 2:
                out, alpha = output
                self._attention_weights.append({
                    "layer": layer_name,
                    "alpha": alpha.detach().cpu(),
                })
            else:
                # When return_attention_weights is not set,
                # we store the edge_index for later scoring
                edge_index = inputs[1] if len(inputs) > 1 else None
                if edge_index is not None:
                    self._edge_indices.append({
                        "layer": layer_name,
                        "edge_index": edge_index.detach().cpu(),
                    })
        return hook_fn

    def clear(self):
        """Clear captured attention data."""
        self._attention_weights.clear()
        self._edge_indices.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_node_attention_scores(self, num_nodes: int) -> torch.Tensor:
        """
        Aggregate attention weights across all layers and heads
        to produce a single attention score per node.

        For each node, the score is the mean of all incoming
        attention weights across all layers and all heads.

        Returns:
            Tensor of shape [num_nodes] with attention scores.
        """
        scores = torch.zeros(num_nodes, dtype=torch.float32)

        if self._attention_weights:
            # We have actual attention weights
            for entry in self._attention_weights:
                alpha = entry["alpha"]  # [num_edges, heads]
                # Average across heads
                if alpha.dim() == 2:
                    alpha_mean = alpha.mean(dim=1)  # [num_edges]
                else:
                    alpha_mean = alpha.squeeze()

                # We need edge_index to map attention to destination nodes
                # Find the corresponding edge_index
                for ei_entry in self._edge_indices:
                    if ei_entry["layer"] == entry["layer"]:
                        edge_index = ei_entry["edge_index"]
                        dst_nodes = edge_index[1]  # destination nodes
                        for i, dst in enumerate(dst_nodes):
                            if i < len(alpha_mean):
                                scores[dst] += alpha_mean[i].item()
                        break

        if scores.sum() == 0 and self._edge_indices:
            # Fallback: use edge degree as proxy for attention
            for entry in self._edge_indices:
                edge_index = entry["edge_index"]
                dst_nodes = edge_index[1]
                for dst in dst_nodes:
                    if dst < num_nodes:
                        scores[dst] += 1.0

        # Normalize to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()

        return scores


# ════════════════════════════════════════════════════════════════════════
# 2. GRADIENT-BASED IMPORTANCE (SALIENCY)
# ════════════════════════════════════════════════════════════════════════

def compute_gradient_scores(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    batch: torch.Tensor,
    target_class: int,
) -> torch.Tensor:
    """
    Compute gradient of the predicted CWE score w.r.t. node features.

    The gradient magnitude per node indicates how much that node's
    features contribute to the CWE prediction.

    Args:
        model: Trained VulnGAT model
        x: Node features [num_nodes, feat_dim]
        edge_index: [2, num_edges]
        edge_attr: [num_edges]
        batch: [num_nodes]
        target_class: CWE class index to compute gradient for

    Returns:
        Tensor [num_nodes] — gradient magnitude per node
    """
    model.eval()

    # Enable gradients on input features
    x_input = x.clone().detach().requires_grad_(True)

    # Forward pass
    logits = model(x_input, edge_index, edge_attr, batch)
    probs = torch.sigmoid(logits)

    # Get score for target class
    target_score = probs[0, target_class]

    # Backward pass to compute gradients
    model.zero_grad()
    target_score.backward(retain_graph=True)

    if x_input.grad is None:
        logger.warning("No gradient computed for target class %d", target_class)
        return torch.zeros(x.size(0))

    # Gradient magnitude per node = L2 norm of gradient vector
    grad_magnitude = x_input.grad.detach().cpu().norm(dim=1)  # [num_nodes]

    # Normalize to [0, 1]
    if grad_magnitude.max() > 0:
        grad_magnitude = grad_magnitude / grad_magnitude.max()

    return grad_magnitude


# ════════════════════════════════════════════════════════════════════════
# 3. COMBINED NODE IMPORTANCE SCORING
# ════════════════════════════════════════════════════════════════════════

def compute_node_importance(
    attention_scores: torch.Tensor,
    gradient_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Combine attention and gradient scores into final node importance.

    Formula: node_score = attention_score × gradient_magnitude

    Args:
        attention_scores: [num_nodes] — aggregated attention per node
        gradient_scores: [num_nodes] — gradient magnitude per node

    Returns:
        [num_nodes] — combined importance score
    """
    assert attention_scores.shape == gradient_scores.shape, \
        f"Shape mismatch: attention {attention_scores.shape} vs gradient {gradient_scores.shape}"

    combined = attention_scores * gradient_scores

    # Normalize to [0, 1]
    if combined.max() > 0:
        combined = combined / combined.max()

    return combined


# ════════════════════════════════════════════════════════════════════════
# 4. TOP-K INSTRUCTION SELECTION
# ════════════════════════════════════════════════════════════════════════

def select_vulnerable_nodes(
    scores: torch.Tensor,
    top_k_ratio: float = 0.15,
    min_nodes: int = 3,
    max_nodes: int = 50,
) -> list:
    """
    Select top-k instruction nodes as vulnerability hotspots.

    Args:
        scores: [num_nodes] importance scores
        top_k_ratio: Fraction of nodes to select (e.g., 0.15 = top 15%)
        min_nodes: Minimum number of nodes to return
        max_nodes: Maximum number of nodes to return

    Returns:
        List of (node_index, score) tuples, sorted by score descending.
    """
    num_nodes = scores.size(0)
    k = int(num_nodes * top_k_ratio)
    k = max(k, min_nodes)
    k = min(k, max_nodes, num_nodes)

    # Get top-k indices
    topk_scores, topk_indices = torch.topk(scores, k)

    selected = [
        (int(idx), float(score))
        for idx, score in zip(topk_indices, topk_scores)
        if score > 0  # Only include nodes with non-zero scores
    ]

    return selected


# ════════════════════════════════════════════════════════════════════════
# 5. SUBGRAPH EXTRACTION
# ════════════════════════════════════════════════════════════════════════

def extract_vulnerable_subgraph(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    selected_nodes: list,
    max_hops: int = 2,
) -> dict:
    """
    Extract minimal connected subgraph around selected vulnerable nodes.

    Expansion strategy:
      - Primary: follow data flow edges (edge_type=1)
      - Secondary: follow control flow edges (edge_type=2)
      - Skip: sequential edges (edge_type=0) unless connecting selected nodes

    Args:
        edge_index: [2, num_edges]
        edge_attr: [num_edges] edge types
        selected_nodes: List of (node_idx, score) tuples
        max_hops: Number of hops for expansion

    Returns:
        dict with 'nodes', 'edges', 'edge_types'
    """
    selected_set = {idx for idx, _ in selected_nodes}
    expanded = set(selected_set)

    edge_index_np = edge_index.cpu().numpy()
    edge_attr_np = edge_attr.cpu().numpy()

    # Build adjacency for fast lookup
    # src → [(dst, edge_type), ...]
    adj_forward = {}
    adj_backward = {}
    for i in range(edge_index_np.shape[1]):
        src = int(edge_index_np[0, i])
        dst = int(edge_index_np[1, i])
        etype = int(edge_attr_np[i])

        if src not in adj_forward:
            adj_forward[src] = []
        adj_forward[src].append((dst, etype))

        if dst not in adj_backward:
            adj_backward[dst] = []
        adj_backward[dst].append((src, etype))

    # Expand via data flow (primary) and CFG (secondary)
    for hop in range(max_hops):
        frontier = set()
        for node in expanded:
            # Forward: follow data flow edges
            for dst, etype in adj_forward.get(node, []):
                if etype == 1:  # data_flow — primary
                    frontier.add(dst)
                elif etype == 2:  # cfg — secondary
                    frontier.add(dst)

            # Backward: follow data flow edges
            for src, etype in adj_backward.get(node, []):
                if etype == 1:  # data_flow — primary
                    frontier.add(src)

        expanded.update(frontier)

    # Extract edges within the subgraph
    subgraph_edges = []
    subgraph_edge_types = []
    for i in range(edge_index_np.shape[1]):
        src = int(edge_index_np[0, i])
        dst = int(edge_index_np[1, i])
        if src in expanded and dst in expanded:
            subgraph_edges.append((src, dst))
            subgraph_edge_types.append(int(edge_attr_np[i]))

    return {
        "nodes": sorted(expanded),
        "edges": subgraph_edges,
        "edge_types": subgraph_edge_types,
        "num_nodes": len(expanded),
        "num_edges": len(subgraph_edges),
    }


# ════════════════════════════════════════════════════════════════════════
# 6. MAP NODES TO IR CONTEXT
# ════════════════════════════════════════════════════════════════════════

def map_nodes_to_ir(
    selected_nodes: list,
    ir_data: dict,
    func_name: Optional[str] = None,
) -> list:
    """
    Map selected node indices back to LLVM IR instructions.

    Reconstructs the instruction list in the same order used during
    graph construction (see pyg_dataset.py: build_function_graph).

    Args:
        selected_nodes: List of (node_idx, score) tuples
        ir_data: Parsed IR JSON dict
        func_name: If provided, only look at this function

    Returns:
        List of dicts with 'node_idx', 'score', 'opcode', 'text',
        'function', 'block', 'instruction_idx'
    """
    # Reconstruct node → instruction mapping
    instruction_map = []
    node_idx = 0

    for func in ir_data.get("functions", []):
        if func.get("is_declaration", False):
            continue
        if func_name and func.get("name", "") != func_name:
            continue

        fname = func.get("name", "unknown")
        for block in func.get("blocks", []):
            blabel = block.get("label", "entry")
            for i, inst in enumerate(block.get("instructions", [])):
                instruction_map.append({
                    "node_idx": node_idx,
                    "function": fname,
                    "block": blabel,
                    "instruction_idx": i,
                    "opcode": inst.get("opcode", "unknown"),
                    "text": inst.get("text", ""),
                })
                node_idx += 1

    # Match selected nodes to instructions
    mapped = []
    idx_to_inst = {entry["node_idx"]: entry for entry in instruction_map}

    for node_idx, score in selected_nodes:
        if node_idx in idx_to_inst:
            entry = idx_to_inst[node_idx].copy()
            entry["score"] = score
            mapped.append(entry)
        else:
            mapped.append({
                "node_idx": node_idx,
                "score": score,
                "opcode": "unknown",
                "text": f"[node {node_idx} — not in IR map]",
                "function": "unknown",
                "block": "unknown",
                "instruction_idx": -1,
            })

    # Sort by score descending
    mapped.sort(key=lambda x: x["score"], reverse=True)
    return mapped


# ════════════════════════════════════════════════════════════════════════
# 7. MAIN LOCALIZATION ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

def localize_vulnerabilities(
    model: torch.nn.Module,
    graph_data,
    ir_data: dict,
    predicted_cwes: list,
    device: torch.device,
    top_k_ratio: float = 0.15,
    func_name: Optional[str] = None,
) -> dict:
    """
    Full localization pipeline for a single graph.

    Steps:
      1. Extract attention weights from GATv2Conv layers
      2. Compute gradient of each CWE score w.r.t. node features
      3. Combine attention × gradient for node importance
      4. Select top-k vulnerable instruction nodes
      5. Extract minimal connected subgraph
      6. Map nodes back to LLVM IR instructions

    Args:
        model: Trained VulnGAT model
        graph_data: PyG Data object (single graph)
        ir_data: Parsed IR JSON dict
        predicted_cwes: List of predicted CWE class names (e.g., ["CWE-190", "CWE-369"])
        device: torch device
        top_k_ratio: Fraction of nodes to select
        func_name: Optional function name to restrict localization

    Returns:
        dict with per-CWE localization results
    """
    from torch_geometric.data import Batch

    CLASS_NAMES = ["non-vulnerable", "CWE-476", "CWE-191", "CWE-190", "CWE-369"]

    if not predicted_cwes:
        return {"cwes": {}, "message": "No vulnerabilities to localize."}

    # Prepare graph for inference
    graph = graph_data.to(device)
    batch_data = Batch.from_data_list([graph])

    num_nodes = graph.num_nodes

    # Step 1: Attention extraction
    extractor = AttentionExtractor(model)
    extractor.clear()

    model.eval()
    with torch.no_grad():
        _ = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)

    attention_scores = extractor.get_node_attention_scores(num_nodes)
    extractor.remove_hooks()

    # Step 2-6: Per-CWE localization
    results = {}

    for cwe in predicted_cwes:
        if cwe == "non-vulnerable":
            continue

        if cwe not in CLASS_NAMES:
            logger.warning("Unknown CWE: %s — skipping localization", cwe)
            continue

        target_class = CLASS_NAMES.index(cwe)

        # Step 2: Gradient scoring for this CWE
        gradient_scores = compute_gradient_scores(
            model,
            batch_data.x, batch_data.edge_index,
            batch_data.edge_attr, batch_data.batch,
            target_class,
        )

        # Step 3: Combined importance
        importance = compute_node_importance(attention_scores, gradient_scores)

        # Step 4: Select top-k nodes
        selected = select_vulnerable_nodes(importance, top_k_ratio=top_k_ratio)

        # Step 5: Extract subgraph
        subgraph = extract_vulnerable_subgraph(
            graph.edge_index.cpu(),
            graph.edge_attr.cpu(),
            selected,
        )

        # Step 6: Map to IR instructions
        ir_mapping = map_nodes_to_ir(selected, ir_data, func_name=func_name)

        results[cwe] = {
            "target_class": target_class,
            "num_nodes_scored": num_nodes,
            "num_selected": len(selected),
            "top_nodes": selected,
            "vulnerable_subgraph": subgraph,
            "ir_instructions": ir_mapping,
        }

        logger.info(
            "Localized %s: %d/%d nodes selected (top score: %.4f)",
            cwe, len(selected), num_nodes,
            selected[0][1] if selected else 0.0,
        )

    return {
        "cwes": results,
        "num_total_nodes": num_nodes,
        "attention_scores_summary": {
            "mean": float(attention_scores.mean()),
            "max": float(attention_scores.max()),
            "nonzero": int((attention_scores > 0).sum()),
        },
    }
