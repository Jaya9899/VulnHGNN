"""
pipeline.py — Complete VulnHGNN System Pipeline (§11).

End-to-end pipeline:
    C/C++ Source
    → LLVM IR (.ll)
    → JSON Representation
    → Heterogeneous Graph (CFG + DFG + Containment)
    → VulnGAT Model
    → Multi-label CWE Prediction
    → Node Scoring (Attention + Gradient)
    → Vulnerable Instruction Selection
    → Subgraph Extraction
    → Rule-Based IR Repair Engine
    → Patched LLVM IR
    → Validation (GNN re-detection)
    → Final Clean IR Output
    → Carbon Emission Logging

Final outputs (§12):
    - List of detected CWE vulnerabilities
    - Exact vulnerable LLVM IR instructions
    - Extracted vulnerable subgraph
    - Patched LLVM IR code
    - Validation status
    - Energy and carbon metrics

Usage:
    python src/pipeline.py test_files/test_02_cwe190_overflow.c
    python src/pipeline.py test_files/ --batch
    python src/pipeline.py myfile.c --output results/pipeline_output
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ANSI colors
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
WHITE  = "\033[97m"


def run_full_pipeline(
    source_path: Path,
    model_path: str = "results/pyg_gnn_v9/best_pyg_model.pt",
    output_dir: Path = None,
    support_dir: str = "dataset/test_case_support",
    max_repair_iterations: int = 5,
    carbon_tracking: bool = True,
    threshold: float = 0.5,
    top_k_ratio: float = 0.15,
    confidence_threshold: float = 0.6,
    save_patched_ir: bool = True,
) -> dict:
    """
    Complete end-to-end VulnHGNN pipeline.

    Args:
        source_path: Path to C/C++ source file
        model_path: Path to trained model checkpoint
        output_dir: Directory to save outputs (default: results/pipeline_output)
        support_dir: Path to Juliet test case support headers
        max_repair_iterations: Max healing loop iterations
        carbon_tracking: Enable CodeCarbon tracking
        threshold: Detection threshold
        top_k_ratio: Localization node selection ratio
        confidence_threshold: Min confidence for predictions
        save_patched_ir: Save patched IR to file

    Returns:
        dict with all §12 outputs
    """
    from predict import load_model
    from feedback_loop import run_healing_loop
    from carbon_tracker import CarbonTracker
    from security import validate_ir_input, filter_low_confidence, check_model_stability

    if output_dir is None:
        output_dir = Path("results/pipeline_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Initialize carbon tracker
    tracker = CarbonTracker(
        project_name="VulnHGNN",
        output_dir=str(output_dir),
    ) if carbon_tracking else None

    # ── Step 1: Load model ──────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model from %s on %s ...", model_path, device)

    try:
        model, ckpt = load_model(model_path, device)
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": str(e),
            "file": str(source_path),
        }

    # ── Step 2: Model stability check ───────────────────────────────
    stability = check_model_stability(model)
    if not stability["healthy"]:
        logger.error("Model health check FAILED: %s", stability)
        return {
            "success": False,
            "error": "Model health check failed",
            "model_stability": stability,
            "file": str(source_path),
        }

    # ── Step 3: Run healing loop with carbon tracking ───────────────
    logger.info("Running full pipeline for %s ...", source_path.name)

    if tracker:
        tracker.start_phase("full_pipeline")

    loop_result = run_healing_loop(
        source_path=source_path,
        model=model,
        device=device,
        support_dir=support_dir,
        threshold=threshold,
        max_iterations=max_repair_iterations,
        top_k_ratio=top_k_ratio,
        confidence_threshold=confidence_threshold,
    )

    if tracker:
        tracker.stop_phase()

    # ── Step 4: Assemble final output (§12) ─────────────────────────
    elapsed = round(time.time() - t0, 3)

    # Get carbon metrics
    carbon_metrics = None
    if tracker:
        carbon_report = tracker.get_report()
        carbon_metrics = carbon_report.get("totals", {})
        tracker.save_report(str(output_dir / f"carbon_{source_path.stem}.json"))

    # Collect vulnerable instructions from iterations
    vulnerable_instructions = []
    vulnerable_subgraph = {}
    for iteration in loop_result.get("iterations", []):
        loc_data = iteration.get("localization") or {}
        for cwe, loc in (loc_data.get("cwes") or {}).items():
            for inst in loc.get("top_instructions", []):
                vulnerable_instructions.append({
                    "cwe": cwe,
                    "text": inst.get("text", ""),
                    "score": inst.get("score", 0),
                })

    # Build final output
    final_output = {
        "success": loop_result.get("success", False),
        "file": str(source_path),
        "file_name": source_path.name,
        "elapsed_sec": elapsed,

        # §12.1: Detected CWE vulnerabilities
        "detected_cwes": loop_result.get("original_cwes", []),

        # §12.2: Vulnerable LLVM IR instructions
        "vulnerable_instructions": vulnerable_instructions,

        # §12.3: Vulnerable subgraph (from localization)
        "vulnerable_subgraph": vulnerable_subgraph,

        # §12.4: Patched LLVM IR
        "patched_ir": loop_result.get("patched_ir"),
        "has_patched_ir": loop_result.get("patched_ir") is not None,

        # §12.5: Validation status
        "validation_status": loop_result.get("final_status", "UNKNOWN"),
        "remaining_cwes": loop_result.get("remaining_cwes", []),
        "total_patches": loop_result.get("total_patches", 0),
        "num_iterations": len(loop_result.get("iterations", [])),

        # §12.6: Energy and carbon metrics
        "carbon_metrics": carbon_metrics,

        # Detailed iteration log
        "iterations": loop_result.get("iterations", []),

        # Model info
        "model": {
            "path": model_path,
            "device": str(device),
            "parameters": stability.get("total_parameters", 0),
            "healthy": stability["healthy"],
        },
    }

    # ── Step 5: Save outputs ────────────────────────────────────────
    # Save main result JSON
    result_path = output_dir / f"{source_path.stem}_result.json"
    # Remove IR content from JSON (too large)
    json_output = {k: v for k, v in final_output.items()
                   if k not in ("patched_ir",)}
    with open(result_path, "w") as f:
        json.dump(json_output, f, indent=2)

    # Save patched IR separately
    if save_patched_ir and final_output.get("patched_ir"):
        ir_path = output_dir / f"{source_path.stem}_patched.ll"
        with open(ir_path, "w", encoding="utf-8") as f:
            f.write(final_output["patched_ir"])
        logger.info("Patched IR saved → %s", ir_path)

    logger.info("Results saved → %s", result_path)

    return final_output


def print_pipeline_result(result: dict):
    """Pretty-print pipeline results to console."""
    sep = "═" * 65
    sep2 = "─" * 65

    print(f"\n{CYAN}{BOLD}{sep}{RESET}")
    print(f"{CYAN}{BOLD}  VulnHGNN — Complete Pipeline Results{RESET}")
    print(f"{CYAN}{BOLD}{sep}{RESET}\n")

    print(f"  {BOLD}File:{RESET}    {result.get('file_name', '?')}")
    print(f"  {BOLD}Time:{RESET}    {result.get('elapsed_sec', 0):.2f}s")
    print(f"  {BOLD}Status:{RESET}  ", end="")

    status = result.get("validation_status", "UNKNOWN")
    if status == "FULLY_HEALED":
        print(f"{GREEN}{BOLD}✔ FULLY HEALED{RESET}")
    elif status == "CLEAN":
        print(f"{GREEN}{BOLD}✔ CLEAN — No vulnerabilities{RESET}")
    elif status == "NO_REPAIR_AVAILABLE":
        print(f"{YELLOW}{BOLD}⚠ No repair available{RESET}")
    else:
        print(f"{RED}{BOLD}✘ {status}{RESET}")

    # Detected CWEs
    cwes = result.get("detected_cwes", [])
    if cwes:
        print(f"\n  {BOLD}Detected Vulnerabilities:{RESET}")
        for cwe in cwes:
            print(f"    {RED}● {cwe}{RESET}")
    else:
        print(f"\n  {GREEN}No vulnerabilities detected{RESET}")

    # Patches
    total_patches = result.get("total_patches", 0)
    if total_patches > 0:
        print(f"\n  {BOLD}Repairs:{RESET}")
        print(f"    Patches applied: {total_patches}")
        print(f"    Iterations:      {result.get('num_iterations', 0)}")

    remaining = result.get("remaining_cwes", [])
    if remaining:
        print(f"\n  {YELLOW}Remaining CWEs: {', '.join(remaining)}{RESET}")

    # Vulnerable instructions
    vuln_insts = result.get("vulnerable_instructions", [])
    if vuln_insts:
        print(f"\n  {BOLD}Vulnerable Instructions (top 5):{RESET}")
        print(f"  {sep2}")
        for inst in vuln_insts[:5]:
            text = inst.get("text", "")[:60]
            score = inst.get("score", 0)
            cwe = inst.get("cwe", "")
            print(f"    [{cwe}] {text}  (score: {score:.4f})")

    # Carbon metrics
    carbon = result.get("carbon_metrics")
    if carbon:
        print(f"\n  {BOLD}Carbon Emissions:{RESET}")
        print(f"    Duration:    {carbon.get('duration_sec', 0):.2f}s")
        print(f"    Energy:      {carbon.get('energy_kwh', 0):.6f} kWh")
        print(f"    CO₂:         {carbon.get('co2_kg', 0):.8f} kg")

    print(f"\n{CYAN}{BOLD}{sep}{RESET}\n")


# ════════════════════════════════════════════════════════════════════════
# BATCH MODE
# ════════════════════════════════════════════════════════════════════════

def run_batch_pipeline(
    folder: Path,
    model_path: str,
    output_dir: Path,
    **kwargs,
) -> list:
    """Run full pipeline on all C/C++ files in a folder."""
    files = sorted(list(folder.glob("*.c")) + list(folder.glob("*.cpp")))

    if not files:
        print(f"{RED}No .c/.cpp files found in {folder}{RESET}")
        return []

    print(f"\n{CYAN}{BOLD}{'═'*65}{RESET}")
    print(f"{CYAN}{BOLD}  VulnHGNN — Batch Pipeline: {len(files)} files{RESET}")
    print(f"{CYAN}{BOLD}{'═'*65}{RESET}\n")

    all_results = []
    summary = {"clean": 0, "healed": 0, "partial": 0, "failed": 0}

    for i, fpath in enumerate(files, 1):
        print(f"\n{BOLD}{'─'*65}{RESET}")
        print(f"{BOLD}  [{i}/{len(files)}] {fpath.name}{RESET}")

        result = run_full_pipeline(
            source_path=fpath,
            model_path=model_path,
            output_dir=output_dir,
            **kwargs,
        )

        all_results.append(result)
        print_pipeline_result(result)

        status = result.get("validation_status", "UNKNOWN")
        if status == "CLEAN":
            summary["clean"] += 1
        elif status == "FULLY_HEALED":
            summary["healed"] += 1
        elif status in ("NO_REPAIR_AVAILABLE", "MAX_ITERATIONS_REACHED"):
            summary["partial"] += 1
        else:
            summary["failed"] += 1

    # Batch summary
    print(f"\n{CYAN}{BOLD}{'═'*65}{RESET}")
    print(f"{CYAN}{BOLD}  BATCH SUMMARY{RESET}")
    print(f"{CYAN}{'─'*65}{RESET}")
    print(f"  Total files    : {len(files)}")
    print(f"  {GREEN}Clean          : {summary['clean']}{RESET}")
    print(f"  {GREEN}Fully healed   : {summary['healed']}{RESET}")
    print(f"  {YELLOW}Partial repair : {summary['partial']}{RESET}")
    print(f"  {RED}Failed         : {summary['failed']}{RESET}")
    print(f"{CYAN}{BOLD}{'═'*65}{RESET}")

    # Save batch results
    batch_path = output_dir / "batch_results.json"
    batch_output = []
    for r in all_results:
        batch_output.append({
            k: v for k, v in r.items() if k not in ("patched_ir",)
        })
    with open(batch_path, "w") as f:
        json.dump(batch_output, f, indent=2)
    print(f"\n  Batch results saved → {batch_path}\n")

    return all_results


# ════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VulnHGNN — Complete Vulnerability Detection & Self-Healing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py test_files/test_02_cwe190_overflow.c
  python src/pipeline.py test_files/ --batch
  python src/pipeline.py myfile.c --output results/my_output
  python src/pipeline.py myfile.c --no-carbon --max-iterations 3
        """,
    )

    parser.add_argument(
        "input", type=str,
        help="Path to .c/.cpp file or directory (with --batch)",
    )
    parser.add_argument(
        "--model", type=str,
        default="results/pyg_gnn_v9/best_pyg_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output", type=str,
        default="results/pipeline_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--support", type=str,
        default="dataset/test_case_support",
        help="Path to support headers",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run on all .c/.cpp files in input directory",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=5,
        help="Maximum repair iterations (default: 5)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Detection threshold (default: 0.5)",
    )
    parser.add_argument(
        "--top-k", type=float, default=0.15,
        help="Top-k ratio for localization (default: 0.15)",
    )
    parser.add_argument(
        "--no-carbon", action="store_true",
        help="Disable carbon emissions tracking",
    )
    parser.add_argument(
        "--no-save-ir", action="store_true",
        help="Don't save patched IR files",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if args.batch:
        if not input_path.is_dir():
            print(f"{RED}ERROR: --batch requires a directory{RESET}")
            sys.exit(1)
        run_batch_pipeline(
            folder=input_path,
            model_path=args.model,
            output_dir=output_dir,
            support_dir=args.support,
            max_repair_iterations=args.max_iterations,
            carbon_tracking=not args.no_carbon,
            threshold=args.threshold,
            top_k_ratio=args.top_k,
            save_patched_ir=not args.no_save_ir,
        )
    else:
        if not input_path.exists():
            print(f"{RED}ERROR: File not found: {input_path}{RESET}")
            sys.exit(1)
        if input_path.suffix.lower() not in (".c", ".cpp"):
            print(f"{RED}ERROR: Input must be .c or .cpp{RESET}")
            sys.exit(1)

        result = run_full_pipeline(
            source_path=input_path,
            model_path=args.model,
            output_dir=output_dir,
            support_dir=args.support,
            max_repair_iterations=args.max_iterations,
            carbon_tracking=not args.no_carbon,
            threshold=args.threshold,
            top_k_ratio=args.top_k,
            save_patched_ir=not args.no_save_ir,
        )
        print_pipeline_result(result)


if __name__ == "__main__":
    main()
