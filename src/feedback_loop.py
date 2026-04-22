"""
feedback_loop.py — Iterative self-healing loop for VulnHGNN.

Orchestrates the cycle:
    Detect → Localize → Repair → Validate → Re-detect

Continues until:
  - No vulnerabilities remain
  - No valid repair can be applied
  - Maximum iterations reached
"""

import os
import sys
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("feedback_loop")


def run_healing_loop(
    source_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    support_dir: str = "dataset/test_case_support",
    threshold: float = 0.5,
    max_iterations: int = 5,
    top_k_ratio: float = 0.15,
    confidence_threshold: float = 0.6,
) -> dict:
    """
    Full iterative healing loop:

    1. Compile C/C++ source → LLVM IR
    2. Detect vulnerabilities (GNN)
    3. If vulnerabilities found:
       a. Localize vulnerable instructions (attention + gradient)
       b. Apply IR-level repairs
       c. Validate (GNN re-detection)
       d. If validation fails, loop back to step 2 with patched IR
    4. Return final clean IR + repair log

    Args:
        source_path: Path to C/C++ source file
        model: Trained VulnGAT model
        device: torch device
        support_dir: Path to Juliet test case support headers
        threshold: Detection threshold
        max_iterations: Maximum repair iterations
        top_k_ratio: Fraction of nodes to select for localization
        confidence_threshold: Minimum confidence for predictions

    Returns:
        dict with full pipeline results including repair log
    """
    from predict import compile_to_ll, CLASS_THRESHOLDS
    from ir_parser import parse_ll_file
    from pyg_dataset import build_function_graph, NUM_CLASSES
    from localization import localize_vulnerabilities
    from ir_healer import repair_ir
    from validation import validate_repair, run_gnn_redetection
    from security import validate_ir_input, filter_low_confidence

    CLASS_NAMES = ["non-vulnerable", "CWE-476", "CWE-191", "CWE-190", "CWE-369"]

    t0 = time.time()

    result = {
        "success": False,
        "source_file": str(source_path),
        "iterations": [],
        "final_status": "UNKNOWN",
        "original_cwes": [],
        "remaining_cwes": [],
        "total_patches": 0,
        "original_ir": None,
        "patched_ir": None,
        "elapsed_sec": 0,
    }

    # ── Step 1: Compile to LLVM IR ──────────────────────────────────
    logger.info("Step 1: Compiling %s → LLVM IR ...", source_path.name)
    ll_path = None
    try:
        ll_path = compile_to_ll(source_path, support_dir)
        with open(ll_path, "r", encoding="utf-8", errors="replace") as f:
            original_ir = f.read()
    except Exception as e:
        result["error"] = f"Compilation failed: {e}"
        result["elapsed_sec"] = time.time() - t0
        return result
    finally:
        if ll_path and os.path.exists(ll_path):
            os.unlink(ll_path)

    # ── Input validation ────────────────────────────────────────────
    is_valid, msg = validate_ir_input(original_ir)
    if not is_valid:
        result["error"] = f"IR validation failed: {msg}"
        result["elapsed_sec"] = time.time() - t0
        return result

    result["original_ir"] = original_ir
    current_ir = original_ir

    # ── Iterative loop ──────────────────────────────────────────────
    for iteration in range(1, max_iterations + 1):
        logger.info("═══ Iteration %d/%d ═══", iteration, max_iterations)
        iter_result = {
            "iteration": iteration,
            "detected_cwes": [],
            "localization": None,
            "repair": None,
            "validation": None,
        }

        # ── Step 2: Detect ──────────────────────────────────────────
        logger.info("  [Detect] Running GNN inference...")
        detection = run_gnn_redetection(current_ir, model, device, threshold)

        if not detection["success"]:
            iter_result["error"] = f"Detection failed: {detection.get('error')}"
            result["iterations"].append(iter_result)
            break

        detected_cwes = detection.get("detected_cwes", [])
        iter_result["detected_cwes"] = detected_cwes

        if iteration == 1:
            result["original_cwes"] = list(detected_cwes)

        if not detected_cwes:
            logger.info("  ✔ No vulnerabilities detected — code is clean!")
            result["iterations"].append(iter_result)
            result["final_status"] = "CLEAN"
            result["remaining_cwes"] = []
            break

        logger.info("  Detected CWEs: %s", detected_cwes)

        # ── Step 3: Localize ────────────────────────────────────────
        logger.info("  [Localize] Computing node importance scores...")
        ir_data = detection.get("ir_data")

        if ir_data is None:
            # Need to re-parse
            with tempfile.NamedTemporaryFile(
                suffix=".ll", delete=False, mode="w", encoding="utf-8"
            ) as tmp:
                tmp.write(current_ir)
                tmp_path = tmp.name
            try:
                ir_data = parse_ll_file(tmp_path)
            finally:
                os.unlink(tmp_path)

        localization_result = _run_localization(
            model, ir_data, detected_cwes, device, top_k_ratio
        )
        iter_result["localization"] = {
            "cwes": {
                cwe: {
                    "num_selected": loc.get("num_selected", 0),
                    "top_instructions": [
                        {"text": inst.get("text", "")[:80], "score": inst.get("score", 0)}
                        for inst in loc.get("ir_instructions", [])[:5]
                    ],
                }
                for cwe, loc in localization_result.get("cwes", {}).items()
            }
        }

        # Collect vulnerable instructions for repair targeting
        vuln_instructions = []
        for cwe, loc in localization_result.get("cwes", {}).items():
            for inst in loc.get("ir_instructions", []):
                inst_copy = dict(inst)
                inst_copy["cwe"] = cwe
                vuln_instructions.append(inst_copy)

        # ── Step 4: Repair ──────────────────────────────────────────
        logger.info("  [Repair] Applying IR-level patches...")
        repair_result = repair_ir(current_ir, detected_cwes, vuln_instructions)
        iter_result["repair"] = {
            "total_patches": repair_result.get("total_patches", 0),
            "patches": repair_result.get("patches", []),
            "is_valid_ir": repair_result.get("is_valid", False),
        }

        if repair_result.get("total_patches", 0) == 0:
            logger.warning("  No patches applied — cannot repair further.")
            result["iterations"].append(iter_result)
            result["final_status"] = "NO_REPAIR_AVAILABLE"
            result["remaining_cwes"] = detected_cwes
            break

        result["total_patches"] += repair_result["total_patches"]
        patched_ir = repair_result["patched_ir"]

        # ── Step 5: Validate ────────────────────────────────────────
        logger.info("  [Validate] Verifying repair...")
        validation_result = validate_repair(
            original_ir=current_ir,
            patched_ir=patched_ir,
            model=model,
            device=device,
            original_predictions={
                "predictions": [
                    {"class": cwe, "detected": True}
                    for cwe in detected_cwes
                ]
            },
            threshold=threshold,
        )
        iter_result["validation"] = {
            "accepted": validation_result.get("accepted", False),
            "reason": validation_result.get("reason", ""),
            "new_cwes": validation_result.get("new_vuln_check", {}).get("new_cwes", []),
            "removed_cwes": validation_result.get("new_vuln_check", {}).get("removed_cwes", []),
            "remaining_cwes": validation_result.get("new_vuln_check", {}).get("remaining_cwes", []),
        }

        result["iterations"].append(iter_result)

        # Update IR for next iteration
        current_ir = patched_ir

        # Check if repair was fully accepted
        remaining = validation_result.get("new_vuln_check", {}).get("remaining_cwes", [])
        new_cwes = validation_result.get("new_vuln_check", {}).get("new_cwes", [])

        if new_cwes:
            logger.warning("  ✘ Repair introduced new vulnerabilities: %s", new_cwes)
            result["final_status"] = "REPAIR_INTRODUCED_NEW_VULNS"
            result["remaining_cwes"] = list(set(remaining) | set(new_cwes))
            break

        if not remaining:
            logger.info("  ✔ All vulnerabilities repaired!")
            result["final_status"] = "FULLY_HEALED"
            result["remaining_cwes"] = []
            break

        logger.info("  Remaining CWEs: %s — continuing loop...", remaining)
        result["remaining_cwes"] = remaining

    else:
        # Loop exhausted max iterations
        result["final_status"] = "MAX_ITERATIONS_REACHED"
        logger.warning("Max iterations (%d) reached.", max_iterations)

    result["patched_ir"] = current_ir
    result["success"] = True
    result["elapsed_sec"] = round(time.time() - t0, 3)

    logger.info(
        "Pipeline complete: status=%s, iterations=%d, patches=%d, elapsed=%.2fs",
        result["final_status"],
        len(result["iterations"]),
        result["total_patches"],
        result["elapsed_sec"],
    )

    return result


def _run_localization(model, ir_data, detected_cwes, device, top_k_ratio):
    """
    Helper to run localization on each function in the IR data.
    Returns aggregated localization results.
    """
    from pyg_dataset import build_function_graph, NUM_CLASSES
    from localization import localize_vulnerabilities
    import numpy as np

    aggregated = {"cwes": {}, "num_total_nodes": 0}

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

            func_ir_data = {
                "functions": [func],
                "source_file": ir_data.get("source_file", ""),
            }

            loc_result = localize_vulnerabilities(
                model=model,
                graph_data=graph,
                ir_data=func_ir_data,
                predicted_cwes=detected_cwes,
                device=device,
                top_k_ratio=top_k_ratio,
                func_name=func.get("name"),
            )

            # Merge results
            for cwe, data in loc_result.get("cwes", {}).items():
                if cwe not in aggregated["cwes"]:
                    aggregated["cwes"][cwe] = data
                else:
                    existing = aggregated["cwes"][cwe]
                    existing["ir_instructions"].extend(data.get("ir_instructions", []))
                    existing["num_selected"] += data.get("num_selected", 0)

            aggregated["num_total_nodes"] += loc_result.get("num_total_nodes", 0)

        except Exception as e:
            logger.warning("Localization failed for function %s: %s",
                          func.get("name", "?"), e)

    return aggregated
