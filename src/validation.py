"""
validation.py — Validation Layer for VulnHGNN self-healing pipeline.

All repairs must be verified before acceptance. This module implements:
  1. GNN Re-detection: Pass patched IR through the model to verify the
     vulnerability is no longer predicted.
  2. No-new-vulnerabilities check: Ensure repairs don't introduce new CWEs.

Acceptance criteria:
  - Original vulnerability is removed (no longer predicted above threshold)
  - No new vulnerabilities are introduced
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("validation")

CLASS_NAMES = ["non-vulnerable", "CWE-476", "CWE-191", "CWE-190", "CWE-369"]


# ════════════════════════════════════════════════════════════════════════
# 1. GNN RE-DETECTION
# ════════════════════════════════════════════════════════════════════════

def run_gnn_redetection(
    patched_ir_content: str,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Parse patched LLVM IR → build graph → run GNN → return predictions.

    This validates that the patched IR no longer triggers vulnerability
    predictions from the model.

    Args:
        patched_ir_content: Raw LLVM IR text after repair
        model: Trained VulnGAT model
        device: torch device
        threshold: Detection threshold

    Returns:
        dict with 'success', 'predictions', 'detected_cwes'
    """
    from ir_parser import parse_ll_file
    from pyg_dataset import build_function_graph, NUM_CLASSES, instruction_features
    from torch_geometric.data import Batch
    import numpy as np

    # Write patched IR to temp file for parsing
    with tempfile.NamedTemporaryFile(
        suffix=".ll", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(patched_ir_content)
        tmp_path = tmp.name

    try:
        # Parse the patched IR
        ir_data = parse_ll_file(tmp_path)

        if not ir_data or not ir_data.get("functions"):
            return {
                "success": False,
                "error": "Patched IR produced no parseable functions.",
                "predictions": [],
                "detected_cwes": [],
            }

        # Run inference on each function (same as predict.py approach)
        all_probs = []
        model.eval()

        for func in ir_data.get("functions", []):
            if func.get("is_declaration", False):
                continue
            blocks = func.get("blocks", [])
            total_insts = sum(len(b.get("instructions", [])) for b in blocks)
            if total_insts < 3:
                continue

            dummy_label = np.zeros(NUM_CLASSES, dtype=np.float32)
            try:
                graph = build_function_graph(func, dummy_label)
                if graph is None or graph.num_nodes == 0:
                    continue

                graph = graph.to(device)
                batch = Batch.from_data_list([graph])

                with torch.no_grad():
                    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
                    all_probs.append(probs)

            except Exception as e:
                logger.warning("Failed to process function %s: %s", func.get("name", "?"), e)
                continue

        if not all_probs:
            return {
                "success": False,
                "error": "No functions could be processed from patched IR.",
                "predictions": [],
                "detected_cwes": [],
            }

        # Aggregate max probabilities across all functions
        probs_final = [0.0] * NUM_CLASSES
        for probs in all_probs:
            for i in range(NUM_CLASSES):
                probs_final[i] = max(probs_final[i], probs[i])

        # Build predictions
        predictions = []
        detected_cwes = []
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs_final)):
            detected = prob >= threshold
            predictions.append({
                "class": name,
                "probability": round(prob, 4),
                "detected": detected,
            })
            if detected and name != "non-vulnerable":
                detected_cwes.append(name)

        return {
            "success": True,
            "predictions": predictions,
            "detected_cwes": detected_cwes,
            "ir_data": ir_data,
        }

    except Exception as e:
        logger.error("GNN re-detection failed: %s", e)
        return {
            "success": False,
            "error": str(e),
            "predictions": [],
            "detected_cwes": [],
        }

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ════════════════════════════════════════════════════════════════════════
# 2. NO-NEW-VULNERABILITIES CHECK
# ════════════════════════════════════════════════════════════════════════

def check_no_new_vulnerabilities(
    original_cwes: list,
    patched_cwes: list,
) -> dict:
    """
    Ensure the repair didn't introduce new vulnerabilities.

    Args:
        original_cwes: CWEs detected in the original code
        patched_cwes: CWEs detected in the patched code

    Returns:
        dict with 'passed', 'new_cwes', 'removed_cwes', 'remaining_cwes'
    """
    original_set = set(original_cwes)
    patched_set = set(patched_cwes)

    new_cwes = patched_set - original_set
    removed_cwes = original_set - patched_set
    remaining_cwes = original_set & patched_set

    passed = len(new_cwes) == 0

    if new_cwes:
        logger.warning("NEW vulnerabilities introduced by repair: %s", new_cwes)
    if removed_cwes:
        logger.info("Vulnerabilities successfully removed: %s", removed_cwes)
    if remaining_cwes:
        logger.info("Vulnerabilities still present: %s", remaining_cwes)

    return {
        "passed": passed,
        "new_cwes": sorted(new_cwes),
        "removed_cwes": sorted(removed_cwes),
        "remaining_cwes": sorted(remaining_cwes),
    }


# ════════════════════════════════════════════════════════════════════════
# 3. FULL VALIDATION
# ════════════════════════════════════════════════════════════════════════

def validate_repair(
    original_ir: str,
    patched_ir: str,
    model: torch.nn.Module,
    device: torch.device,
    original_predictions: dict,
    threshold: float = 0.5,
) -> dict:
    """
    Full validation of a repair:
      Step 1: Re-run GNN detection on patched IR
      Step 2: Check no new vulnerabilities introduced

    Args:
        original_ir: Original LLVM IR text
        patched_ir: Patched LLVM IR text after repair
        model: Trained VulnGAT model
        device: torch device
        original_predictions: Detection results from original code
        threshold: Detection threshold

    Returns:
        dict with 'accepted', 'reason', validation details
    """
    result = {
        "accepted": False,
        "reason": "",
        "redetection": None,
        "new_vuln_check": None,
    }

    # Extract original CWEs
    original_cwes = []
    for pred in original_predictions.get("predictions", []):
        if pred.get("detected") and pred.get("class") != "non-vulnerable":
            original_cwes.append(pred["class"])

    if not original_cwes:
        # Nothing to validate if no original vulnerabilities
        original_preds = original_predictions.get("probabilities", [])
        for pred in original_preds:
            if pred.get("detected") and pred.get("class") != "non-vulnerable":
                original_cwes.append(pred["class"])

    # Step 1: GNN Re-detection
    logger.info("Step 1: Running GNN re-detection on patched IR...")
    redetection = run_gnn_redetection(patched_ir, model, device, threshold)
    result["redetection"] = {
        "success": redetection["success"],
        "detected_cwes": redetection.get("detected_cwes", []),
        "predictions": redetection.get("predictions", []),
    }

    if not redetection["success"]:
        result["reason"] = f"Re-detection failed: {redetection.get('error', 'unknown')}"
        return result

    # Step 2: No-new-vulnerabilities check
    logger.info("Step 2: Checking for new vulnerabilities...")
    patched_cwes = redetection.get("detected_cwes", [])
    new_vuln_check = check_no_new_vulnerabilities(original_cwes, patched_cwes)
    result["new_vuln_check"] = new_vuln_check

    if not new_vuln_check["passed"]:
        result["reason"] = (
            f"Repair introduced new vulnerabilities: "
            f"{', '.join(new_vuln_check['new_cwes'])}"
        )
        return result

    # Determine acceptance
    removed = new_vuln_check["removed_cwes"]
    remaining = new_vuln_check["remaining_cwes"]

    if remaining:
        result["accepted"] = False
        result["reason"] = (
            f"Partial fix: removed {removed} but {remaining} remain. "
            f"Requires additional repair iterations."
        )
    else:
        result["accepted"] = True
        result["reason"] = f"All vulnerabilities removed: {removed}. Repair accepted."

    logger.info("Validation result: %s — %s", 
                "ACCEPTED" if result["accepted"] else "REJECTED",
                result["reason"])

    return result
