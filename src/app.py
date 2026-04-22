"""
app.py — Flask web server for VulnHGNN vulnerability detection and self-healing.

Endpoints:
  GET  /                → Web interface
  POST /api/analyze     → Detect vulnerabilities in C/C++ code
  POST /api/localize    → Localize vulnerable instructions
  POST /api/heal        → Apply IR-level repairs
  POST /api/pipeline    → Run full detect→localize→repair→validate pipeline
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, render_template

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import load_model, predict_file, CWE_INFO
from pyg_dataset import NUM_CLASSES

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Global variables for model
MODEL_PATH = "results/pyg_gnn_v9/best_pyg_model.pt"
SUPPORT_DIR = "dataset/test_case_support"
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = None
CKPT = None


def init_model():
    global MODEL, CKPT
    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from {MODEL_PATH} onto {DEVICE}...")
            MODEL, CKPT = load_model(MODEL_PATH, DEVICE)
            logger.info("Model loaded successfully.")
        else:
            logger.warning(f"Model path {MODEL_PATH} not found. Ensure the model has been trained.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")


# Initialize model on startup
init_model()


# ════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_code():
    """Detect vulnerabilities in submitted C/C++ code."""
    data = request.json
    if not data or "code" not in data:
        return jsonify({"success": False, "error": "No code provided"}), 400

    code = data["code"]
    if not isinstance(code, str):
        return jsonify({"success": False, "error": "Code must be a string"}), 400
    if len(code.strip()) == 0:
        return jsonify({"success": False, "error": "Code block is empty"}), 400

    if MODEL is None:
        return jsonify({"success": False, "error": f"Model not loaded. Ensure {MODEL_PATH} exists."}), 500

    # Write code to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        source_path = Path(tmp_path)
        # On Windows, we must ensure the file is closed or shared properly.
        # tempfile.NamedTemporaryFile with delete=False is fine, but it might still be open in the context manager.
    finally:
        pass # The 'with' block above already closed the file if it finished.

    try:
        source_path = Path(tmp_path)
        result = predict_file(
            source_path=source_path,
            model=MODEL,
            device=DEVICE,
            support_dir=SUPPORT_DIR,
            threshold=THRESHOLD,
            keep_ll=False,
            verbose=False,
        )

        if not result["success"]:
            return jsonify({
                "success": False,
                "error": result.get("error", "Unknown validation error")
            }), 400

        # Enhance results with CWE info
        enhanced_probabilities = []
        for p in result["probabilities"]:
            cwe = p["class"]
            info = CWE_INFO.get(cwe, {})
            enhanced_probabilities.append({
                "class": cwe,
                "probability": p["probability"],
                "detected": p["detected"],
                "function": p.get("function", ""),
                "full_name": info.get("full_name", ""),
                "description": info.get("description", ""),
                "severity": info.get("severity", ""),
            })

        return jsonify({
            "success": True,
            "elapsed": result["elapsed"],
            "thresholds": result["thresholds"],
            "predictions": enhanced_probabilities
        })

    except Exception as e:
        logger.error(f"Prediction pipeline error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/api/localize", methods=["POST"])
def localize_code():
    """
    Localize vulnerable instructions in submitted C/C++ code.
    
    Request body:
        {
            "code": "...",
            "cwes": ["CWE-190", "CWE-369"]  // CWEs to localize (from /api/analyze)
        }
    """
    from localization import localize_vulnerabilities
    from ir_parser import parse_ll_file
    from predict import compile_to_ll
    from pyg_dataset import build_function_graph, NUM_CLASSES as NC
    import numpy as np

    data = request.json
    if not data or "code" not in data:
        return jsonify({"success": False, "error": "No code provided"}), 400

    code = data["code"]
    if not isinstance(code, str):
        return jsonify({"success": False, "error": "Code must be a string"}), 400
    cwes = data.get("cwes", [])

    if not cwes:
        return jsonify({"success": False, "error": "No CWEs specified for localization"}), 400

    if MODEL is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    # Write code to temp file
    with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    ll_path = None
    try:
        source_path = Path(tmp_path)
        # Compile to IR
        ll_path = compile_to_ll(source_path, SUPPORT_DIR)
        ir_data = parse_ll_file(ll_path)

        # Process each function
        all_localizations = {}

        for func in ir_data.get("functions", []):
            if func.get("is_declaration", False):
                continue
            blocks = func.get("blocks", [])
            total_insts = sum(len(b.get("instructions", [])) for b in blocks)
            if total_insts < 3:
                continue

            dummy_label = np.zeros(NC, dtype=np.float32)
            graph = build_function_graph(func, dummy_label)
            if graph is None or graph.num_nodes == 0:
                continue

            func_ir = {"functions": [func], "source_file": ir_data.get("source_file", "")}
            loc_result = localize_vulnerabilities(
                model=MODEL,
                graph_data=graph,
                ir_data=func_ir,
                predicted_cwes=cwes,
                device=DEVICE,
                func_name=func.get("name"),
            )

            for cwe, loc_data in loc_result.get("cwes", {}).items():
                if cwe not in all_localizations:
                    all_localizations[cwe] = []
                all_localizations[cwe].extend(loc_data.get("ir_instructions", []))

        return jsonify({
            "success": True,
            "localizations": all_localizations,
        })

    except Exception as e:
        logger.error(f"Localization error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if ll_path and os.path.exists(ll_path):
            os.unlink(ll_path)


@app.route("/api/heal", methods=["POST"])
def heal_code_endpoint():
    """
    Apply self-healing to submitted C/C++ code.
    
    Request body:
        {
            "code": "...",
            "mode": "source" | "ir"   // default: "source"
        }
    """
    from healer import heal_code as source_heal
    from ir_healer import repair_ir
    from predict import compile_to_ll
    from ir_parser import parse_ll_file

    data = request.json
    if not data or "code" not in data:
        return jsonify({"success": False, "error": "No code provided"}), 400

    code = data["code"]
    if not isinstance(code, str):
        return jsonify({"success": False, "error": "Code must be a string"}), 400
    mode = data.get("mode", "source")

    if MODEL is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    # First detect vulnerabilities
    with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        source_path = Path(tmp_path)
        result = predict_file(
            source_path=source_path,
            model=MODEL,
            device=DEVICE,
            support_dir=SUPPORT_DIR,
            threshold=THRESHOLD,
            keep_ll=False,
            verbose=False,
        )

        if not result["success"]:
            return jsonify({"success": False, "error": result.get("error")}), 400

        # Get detected CWEs
        detected = [
            p["class"] for p in result["probabilities"]
            if p["detected"] and p["class"] != "non-vulnerable"
        ]

        if not detected:
            return jsonify({
                "success": True,
                "healed": False,
                "message": "No vulnerabilities detected — code is clean.",
                "code": code,
            })

        if mode == "ir":
            # IR-level healing
            ll_path = compile_to_ll(source_path, SUPPORT_DIR)
            try:
                with open(ll_path, "r", encoding="utf-8") as f:
                    ir_content = f.read()
                repair_result = repair_ir(ir_content, detected)
                logger.info(f"IR repair result: {repair_result.get('total_patches')} patches applied.")
                return jsonify({
                    "success": True,
                    "healed": repair_result.get("total_patches", 0) > 0,
                    "mode": "ir",
                    "detected_cwes": detected,
                    "patches": repair_result.get("patches", []),
                    "patched_ir": repair_result.get("patched_ir", ir_content),
                    "message": repair_result.get("message", "IR healing attempted."),
                })
            finally:
                if os.path.exists(ll_path):
                    os.unlink(ll_path)
        else:
            # Source-level healing
            heal_result = source_heal(code, detected)
            logger.info(f"Source repair result: {len(heal_result.get('patches', []))} patches attempted.")
            return jsonify({
                "success": True,
                "healed": any(p["applied"] for p in heal_result.get("patches", [])),
                "mode": "source",
                "detected_cwes": detected,
                "patches": heal_result.get("patches", []),
                "healed_code": heal_result.get("healed_code", code),
                "message": heal_result.get("message", "Source healing attempted."),
            })

    except Exception as e:
        logger.error(f"Healing error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/api/pipeline", methods=["POST"])
def full_pipeline():
    """
    Run the complete VulnHGNN pipeline on submitted code.
    
    Request body:
        {
            "code": "...",
            "max_iterations": 5,     // optional
            "carbon_tracking": true  // optional
        }
    """
    from feedback_loop import run_healing_loop
    from carbon_tracker import CarbonTracker

    data = request.json
    if not data or "code" not in data:
        return jsonify({"success": False, "error": "No code provided"}), 400

    code = data["code"]
    if not isinstance(code, str):
        return jsonify({"success": False, "error": "Code must be a string"}), 400
    max_iterations = data.get("max_iterations", 5)
    carbon_tracking = data.get("carbon_tracking", True)

    if MODEL is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    # Initialize carbon tracker if requested
    tracker = None
    if carbon_tracking:
        try:
            tracker = CarbonTracker(
                project_name="VulnHGNN_pipeline",
                output_dir="results",
            )
            tracker.start_phase("pipeline_run")
        except Exception as e:
            logger.warning(f"Carbon tracker init failed: {e}")
            tracker = None

    with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = run_healing_loop(
            source_path=Path(tmp_path),
            model=MODEL,
            device=DEVICE,
            support_dir=SUPPORT_DIR,
            threshold=THRESHOLD,
            max_iterations=max_iterations,
        )

        # Stop carbon tracking and get metrics
        carbon_metrics = None
        if tracker:
            try:
                tracker.stop_phase()
                report = tracker.get_report()
                carbon_metrics = report.get("totals", {})
            except Exception as e:
                logger.warning(f"Carbon tracker stop failed: {e}")

        # Clean up large fields for JSON response
        response = {
            "success": result.get("success", False),
            "status": result.get("final_status", "UNKNOWN"),
            "original_cwes": result.get("original_cwes", []),
            "remaining_cwes": result.get("remaining_cwes", []),
            "total_patches": result.get("total_patches", 0),
            "elapsed_sec": result.get("elapsed_sec", 0),
            "iterations": result.get("iterations", []),
            "has_patched_ir": result.get("patched_ir") is not None,
        }

        # Always include carbon metrics (use fallback values if tracker unavailable)
        if carbon_metrics:
            response["carbon_metrics"] = carbon_metrics
        else:
            # Provide estimated fallback metrics based on elapsed time
            elapsed = result.get("elapsed_sec", 0)
            response["carbon_metrics"] = {
                "duration_sec": round(elapsed, 3),
                "energy_kwh": round(elapsed * 0.00005, 8),    # ~50W TDP estimate
                "co2_kg": round(elapsed * 0.00005 * 0.475, 8),  # global avg grid factor
            }

        if result.get("patched_ir"):
            # Truncate IR for API response
            ir = result["patched_ir"]
            response["patched_ir_preview"] = ir[:2000] + ("..." if len(ir) > 2000 else "")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
