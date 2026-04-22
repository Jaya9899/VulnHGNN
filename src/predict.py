#!/usr/bin/env python3
"""
predict.py: Vulnerability detection pipeline (Compile -> Parse -> Graph -> Predict).
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ir_parser   import parse_ll_file
from pyg_model   import build_pyg_model
from pyg_dataset import build_function_graph, NUM_CLASSES
# --- Constants ---
CLASS_NAMES = ["non-vulnerable", "CWE-476", "CWE-191", "CWE-190", "CWE-369"]

CWE_INFO = {
    "non-vulnerable": {
        "full_name": "Non-Vulnerable",
        "description": "No known vulnerability pattern detected.",
        "severity":    "NONE",
        "color":       "\033[92m",   # green
    },
    "CWE-476": {
        "full_name": "NULL Pointer Dereference",
        "description": "Pointer may be used without checking for NULL.",
        "severity":    "HIGH",
        "color":       "\033[91m",   # red
    },
    "CWE-191": {
        "full_name": "Integer Underflow",
        "description": "Integer subtraction may wrap below minimum value.",
        "severity":    "HIGH",
        "color":       "\033[91m",
    },
    "CWE-190": {
        "full_name": "Integer Overflow",
        "description": "Integer arithmetic may exceed maximum value.",
        "severity":    "HIGH",
        "color":       "\033[91m",
    },
    "CWE-369": {
        "full_name": "Divide by Zero",
        "description": "Division operation may use a zero divisor.",
        "severity":    "MEDIUM",
        "color":       "\033[93m",   # yellow
    },
}

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4}

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"

DEFAULT_MODEL     = "results/pyg_gnn_v9/best_pyg_model.pt"
DEFAULT_SUPPORT   = "dataset/test_case_support"
DEFAULT_THRESHOLD = 0.5


# --- Pipeline Steps ---

def compile_to_ll(source_path: Path, support_dir: str) -> str:
    """
    Compile .c/.cpp to LLVM IR (.ll) using clang.
    Returns the .ll content as a string (written to temp file).
    """
    suffix    = source_path.suffix.lower()
    compiler  = "clang++" if suffix == ".cpp" else "clang"

    with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as tmp:
        ll_path = tmp.name

    cmd = [
        compiler, "-S", "-emit-llvm", "-g", "-O0",
        "-Xclang", "-disable-O0-optnone",
        "-w",   # suppress all warnings
        "-o", ll_path,
        str(source_path),
    ]

    # Add support dir if it exists (for Juliet test case headers)
    if support_dir and os.path.isdir(support_dir):
        cmd += ["-I", support_dir]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        os.unlink(ll_path)
        raise RuntimeError(
            f"Compilation failed:\n{result.stderr[:500]}"
        )

    return ll_path


def parse_ll(ll_path: str) -> dict:
    """Parse .ll file into IR JSON dict using ir_parser.py."""
    return parse_ll_file(ll_path)


# build_graph removed — run_model_per_function() is the active code path


def run_model(graph, model, device: torch.device, thresholds: dict) -> dict:
    """Run VulnGAT inference and return prediction dict."""
    model.eval()

    # Add batch dimension (single graph → batch of 1)
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([graph]).to(device)

    with torch.no_grad():
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs  = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    thresholds_dict = thresholds if thresholds else {}
    predictions = []
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        thresh = thresholds_dict.get(name, 0.5)
        predictions.append({
            "class":      name,
            "probability": round(prob, 4),
            "detected":   prob >= thresh,
        })

    return {
        "probabilities": predictions,
        "thresholds":     thresholds,
    }


# --- Output Formatting ---

def print_banner():
    print(f"\n{CYAN}{BOLD}{'═' * 62}{RESET}")
    print(f"{CYAN}{BOLD}  VulnGAT — Vulnerability Detector{RESET}")
    print(f"{CYAN}{BOLD}  Powered by Graph Attention Networks on LLVM IR{RESET}")
    print(f"{CYAN}{BOLD}{'═' * 62}{RESET}\n")


def print_result(source_path: Path, result: dict, elapsed: float):
    preds     = result["probabilities"]
    threshold = result["threshold"]

    detected = [p for p in preds if p["detected"]]
    is_vuln  = any(p["class"] != "non-vulnerable" and p["detected"] for p in preds)

    sep  = "─" * 62
    sep2 = "·" * 62

    print(f"{BOLD}  File   : {WHITE}{source_path.name}{RESET}")
    print(f"{DIM}  Path   : {source_path}{RESET}")
    print(f"{DIM}  Time   : {elapsed:.2f}s{RESET}")
    print(f"  {sep}")

    # Overall verdict
    if not is_vuln:
        print(f"\n  {GREEN}{BOLD}✔  VERDICT: CLEAN — No vulnerabilities detected{RESET}\n")
    else:
        vuln_names = [p["class"] for p in detected if p["class"] != "non-vulnerable"]
        print(f"\n  {RED}{BOLD}✘  VERDICT: VULNERABLE — {len(vuln_names)} issue(s) found{RESET}\n")

    # Per-class probabilities
    print(f"  {BOLD}{'Class':<22} {'Prob':>6}  {'Bar':<20}  Status{RESET}")
    print(f"  {sep2}")

    for p in sorted(preds, key=lambda x: x["probability"], reverse=True):
        name  = p["class"]
        prob  = p["probability"]
        det   = p["detected"]
        info  = CWE_INFO.get(name, {})
        color = info.get("color", WHITE) if det else DIM

        # ASCII probability bar
        bar_len = int(prob * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)

        status = f"{color}{'DETECTED' if det else 'ok'}{RESET}"
        print(f"  {color}{name:<22}{RESET} {prob:>6.1%}  {color}{bar}{RESET}  {status}")

    # Detailed vulnerability info
    vuln_preds = [p for p in detected if p["class"] != "non-vulnerable"]
    if vuln_preds:
        print(f"\n  {BOLD}Vulnerability Details:{RESET}")
        print(f"  {sep}")

        # Sort by severity
        vuln_preds.sort(
            key=lambda x: SEVERITY_ORDER.get(
                CWE_INFO.get(x["class"], {}).get("severity", "LOW"), 3
            )
        )

        for p in vuln_preds:
            name = p["class"]
            info = CWE_INFO.get(name, {})
            sev  = info.get("severity", "UNKNOWN")
            sev_color = RED if sev in ("CRITICAL", "HIGH") else YELLOW
            print(f"\n  {info.get('color', WHITE)}{BOLD}  {name} — {info.get('full_name', '')}{RESET}")
            print(f"  {sev_color}  Severity   : {sev}{RESET}")
            print(f"    Description: {info.get('description', '')}")
            print(f"    Confidence : {p['probability']:.1%}")

    print(f"\n  {sep}\n")


def format_json_result(source_path: Path, result: dict, elapsed: float) -> dict:
    """Format result as clean JSON for --json flag."""
    preds    = result["probabilities"]
    detected = [p for p in preds if p["detected"] and p["class"] != "non-vulnerable"]

    thresholds = result.get("thresholds", {})
    return {
        "file":          str(source_path),
        "elapsed_sec":   round(elapsed, 3),
        "threshold":     thresholds.get("CWE-369", 0.5), # report specific threshold for context
        "vulnerable":    len(detected) > 0,
        "vulnerabilities": [
            {
                "class":       p["class"],
                "full_name":   CWE_INFO.get(p["class"], {}).get("full_name", ""),
                "severity":    CWE_INFO.get(p["class"], {}).get("severity", ""),
                "probability": p["probability"],
            }
            for p in detected
        ],
        "all_scores": {p["class"]: p["probability"] for p in preds},
    }


# --- Main Predict Function ---

def predict_file(
    source_path: Path,
    model,
    device: torch.device,
    support_dir: str,
    threshold:   float,
    keep_ll:     bool = False,
    verbose:     bool = True,
) -> dict:
    """
    Full pipeline for a single file.
    Returns result dict.
    """
    ll_path = None
    t0      = time.time()

    try:
        # Step 1: Compile
        if verbose:
            print(f"  {DIM}[1/4] Compiling {source_path.name} → LLVM IR ...{RESET}")
        ll_path = compile_to_ll(source_path, support_dir)

        # Step 2: Parse
        if verbose:
            print(f"  {DIM}[2/4] Parsing LLVM IR ...{RESET}")
        ir_data = parse_ll(ll_path)

        # Step 4: Predict
        if verbose:
            print(f"  {DIM}[4/4] Running VulnGAT inference ...{RESET}")
        result = run_model_per_function(ir_data, model, device, threshold)
        elapsed = time.time() - t0
        result["elapsed"]  = elapsed
        result["filename"] = str(source_path)
        result["success"]  = True
        return result

    except Exception as e:
        elapsed = time.time() - t0
        return {
            "success":  False,
            "error":    str(e),
            "filename": str(source_path),
            "elapsed":  elapsed,
        }
    finally:
        if ll_path and os.path.exists(ll_path) and not keep_ll:
            os.unlink(ll_path)


def count_instructions(ir_data: dict) -> int:
    total = 0
    for func in ir_data.get("functions", []):
        for block in func.get("blocks", []):
            total += len(block.get("instructions", []))
    return total


# --- Batch Mode ---

def run_batch(
    folder: Path,
    model,
    device,
    support_dir,
    threshold,
    output_json: bool,
):
    """Run prediction on all .c/.cpp files in a folder."""
    files = sorted(list(folder.glob("*.c")) + list(folder.glob("*.cpp")))

    if not files:
        print(f"{RED}No .c/.cpp files found in {folder}{RESET}")
        return

    print_banner()
    print(f"{BOLD}  Batch mode: {len(files)} files in {folder}{RESET}\n")

    all_results = []
    passed = failed = vuln = clean = 0

    for i, fpath in enumerate(files, 1):
        print(f"{BOLD}{'─'*62}{RESET}")
        print(f"{BOLD}  [{i}/{len(files)}] {fpath.name}{RESET}")

        result = predict_file(fpath, model, device, support_dir, threshold)

        if not result["success"]:
            print(f"  {RED}✘ ERROR: {result['error']}{RESET}\n")
            failed += 1
            continue

        passed += 1
        print_result(fpath, result, result["elapsed"])

        preds    = result["probabilities"]
        is_vuln  = any(p["class"] != "non-vulnerable" and p["detected"] for p in preds)
        if is_vuln:
            vuln += 1
        else:
            clean += 1

        if output_json:
            all_results.append(format_json_result(fpath, result, result["elapsed"]))

    # Batch summary
    print(f"\n{CYAN}{BOLD}{'═'*62}{RESET}")
    print(f"{CYAN}{BOLD}  BATCH SUMMARY{RESET}")
    print(f"{CYAN}{'─'*62}{RESET}")
    print(f"  Total files  : {len(files)}")
    print(f"  {GREEN}Clean        : {clean}{RESET}")
    print(f"  {RED}Vulnerable   : {vuln}{RESET}")
    if failed:
        print(f"  {YELLOW}Errors       : {failed}{RESET}")
    print(f"{CYAN}{BOLD}{'═'*62}{RESET}\n")

    if output_json:
        out_path = folder / "predictions.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  {DIM}JSON results saved → {out_path}{RESET}\n")


# --- Load Model ---

def load_model(model_path: str, device: torch.device):
    """Load trained VulnGAT from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train first with: python3 src/train_pyg.py --augment"
        )

    ckpt        = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = ckpt["model_state"]["classifier.3.weight"].shape[0]
    input_dim   = ckpt["model_state"]["input_proj.0.weight"].shape[1]
    model       = build_pyg_model(num_classes=num_classes, input_dim=input_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
    
# Calibration thresholds for model bias
CLASS_THRESHOLDS = {
    "non-vulnerable": 0.5,
    "CWE-476":        0.1,
    "CWE-191":        0.5,
    "CWE-190":        0.4,
    "CWE-369":        0.7,
}

def run_model_per_function(ir_data: dict, model, device, threshold: float) -> dict:
    """
    Run inference on each function separately.
    Final prediction = max probability across all functions per class.
    This matches how the model was trained (one function at a time).
    """
    from torch_geometric.data import Batch
    from pyg_dataset import build_function_graph, instruction_features
    import numpy as np

    all_probs = []

    for func in ir_data.get("functions", []):
        # Skip declarations and tiny functions
        if func.get("is_declaration", False):
            continue
        blocks = func.get("blocks", [])
        total_insts = sum(len(b.get("instructions", [])) for b in blocks)
        if total_insts < 3:
            continue

        # Build dummy label (not used for inference)
        import numpy as np
        dummy_label = np.zeros(NUM_CLASSES, dtype=np.float32)

        try:
            graph = build_function_graph(func, dummy_label)
            if graph is None or graph.num_nodes == 0:
                continue

            graph = graph.to(device)
            batch = Batch.from_data_list([graph])

            model.eval()
            with torch.no_grad():
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                probs  = torch.sigmoid(logits).squeeze(0).cpu().tolist()
                
                # Pure GNN Inference (Raw Probs)
                all_probs.append({
                    "probs": probs,
                    "function": func.get("name")
                })

        except Exception:
            continue

    if not all_probs:
        # fallback — return zeros
        probs_final = [0.0] * NUM_CLASSES
        best_funcs = [""] * NUM_CLASSES
    else:
        # Aggregate max probabilities across all functions for file-level verdict
        probs_final = [0.0] * NUM_CLASSES
        best_funcs = [""] * NUM_CLASSES
        for entry in all_probs:
            for i in range(NUM_CLASSES):
                if entry["probs"][i] > probs_final[i]:
                    probs_final[i] = entry["probs"][i]
                    best_funcs[i] = entry["function"]
    
    predictions = []
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs_final)):
        # Use calibrated thresholds
        thresh = CLASS_THRESHOLDS.get(name, threshold)
        
        predictions.append({
            "class":       name,
            "probability": round(prob, 4),
            "detected":    prob >= thresh,
            "function":    best_funcs[i],
        })

    return {
        "probabilities": predictions,
        "thresholds":    CLASS_THRESHOLDS,
    }


# --- Entry Point ---

def main():
    parser = argparse.ArgumentParser(
        description="VulnGAT — Vulnerability Detector for C/C++ source files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 src/predict.py myfile.c
  python3 src/predict.py test_files/test_02_cwe190_overflow.c
  python3 src/predict.py test_files/ --batch
  python3 src/predict.py myfile.c --threshold 0.4 --json
  python3 src/predict.py myfile.c --model results/pyg_gnn/best_pyg_model.pt
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to .c/.cpp file or directory (with --batch)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Path to trained model checkpoint (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--support",
        type=str,
        default=DEFAULT_SUPPORT,
        help="Path to Juliet test case support headers",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Detection threshold 0–1 (default: 0.5)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run on all .c/.cpp files in input directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--keep-ll",
        action="store_true",
        help="Keep intermediate .ll file after prediction",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress pipeline step messages",
    )

    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────
    logging.basicConfig(level=logging.WARNING)  # suppress INFO logs during inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, ckpt = load_model(args.model, device)
    except FileNotFoundError as e:
        print(f"\n{RED}ERROR: {e}{RESET}\n")
        sys.exit(1)

    input_path = Path(args.input)

    # ── Batch mode ─────────────────────────────────────────────────
    if args.batch:
        if not input_path.is_dir():
            print(f"{RED}ERROR: --batch requires a directory, got: {input_path}{RESET}")
            sys.exit(1)
        run_batch(input_path, model, device, args.support, args.threshold, args.json)
        return

    # ── Single file mode ───────────────────────────────────────────
    if not input_path.exists():
        print(f"\n{RED}ERROR: File not found: {input_path}{RESET}\n")
        sys.exit(1)

    if input_path.suffix.lower() not in (".c", ".cpp"):
        print(f"\n{RED}ERROR: Input must be a .c or .cpp file, got: {input_path.suffix}{RESET}\n")
        sys.exit(1)

    print_banner()

    result = predict_file(
        input_path,
        model,
        device,
        args.support,
        args.threshold,
        keep_ll  = args.keep_ll,
        verbose  = not args.quiet,
    )

    if not result["success"]:
        print(f"\n{RED}{'─'*62}{RESET}")
        print(f"{RED}{BOLD}  ✘ PIPELINE ERROR{RESET}")
        print(f"{RED}{'─'*62}{RESET}")
        print(f"\n  {result['error']}\n")
        sys.exit(1)

    print()  # spacing after step messages

    if args.json:
        out = format_json_result(input_path, result, result["elapsed"])
        print(json.dumps(out, indent=2))
    else:
        print_result(input_path, result, result["elapsed"])


if __name__ == "__main__":
    main()