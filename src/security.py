"""
security.py — Security & Robustness layer for VulnHGNN pipeline.

Provides:
  1. Input Validation   : Ensures LLVM IR is well-formed, rejects adversarial inputs
  2. Prediction Filtering: Discards low-confidence predictions
  3. Model Stability     : Verifies model health (dropout, NaN weights, gradient norms)
"""

import re
import logging
import torch
import numpy as np

logger = logging.getLogger("security")

# ════════════════════════════════════════════════════════════════════════
# 1. INPUT VALIDATION
# ════════════════════════════════════════════════════════════════════════

# Maximum IR file size (10 MB) — prevents memory exhaustion
MAX_IR_SIZE_BYTES = 10 * 1024 * 1024

# Maximum number of functions in a single IR file
MAX_FUNCTIONS = 500

# Maximum number of instructions per function
MAX_INSTRUCTIONS_PER_FUNC = 10_000

# Minimum valid IR must contain at least a define or declare
_DEFINE_RE  = re.compile(r'^\s*define\s+', re.MULTILINE)
_DECLARE_RE = re.compile(r'^\s*declare\s+', re.MULTILINE)

# Patterns that suggest adversarial or malformed input
_SUSPICIOUS_PATTERNS = [
    re.compile(r'inline\s+asm', re.IGNORECASE),          # inline assembly
    re.compile(r'__asm__', re.IGNORECASE),                 # GCC-style asm
    re.compile(r'module\s+asm', re.IGNORECASE),            # module-level asm
]


def validate_ir_input(ll_content: str) -> tuple:
    """
    Validate that LLVM IR content is well-formed and safe to process.

    Returns:
        (is_valid: bool, message: str)
    """
    if not ll_content or not isinstance(ll_content, str):
        return False, "IR content is empty or not a string."

    # Size check
    size = len(ll_content.encode("utf-8", errors="replace"))
    if size > MAX_IR_SIZE_BYTES:
        return False, f"IR too large: {size} bytes (max {MAX_IR_SIZE_BYTES})."

    # Must contain at least one define or declare
    has_define  = bool(_DEFINE_RE.search(ll_content))
    has_declare = bool(_DECLARE_RE.search(ll_content))
    if not has_define and not has_declare:
        return False, "IR contains no 'define' or 'declare' — not valid LLVM IR."

    # Check for suspicious patterns
    for pattern in _SUSPICIOUS_PATTERNS:
        if pattern.search(ll_content):
            return False, f"IR contains suspicious pattern: {pattern.pattern}"

    # Count functions — reject if too many (likely adversarial)
    n_defines = len(_DEFINE_RE.findall(ll_content))
    if n_defines > MAX_FUNCTIONS:
        return False, f"Too many functions: {n_defines} (max {MAX_FUNCTIONS})."

    # Basic structure: should have balanced braces for each define
    open_braces  = ll_content.count("{")
    close_braces = ll_content.count("}")
    if open_braces != close_braces:
        return False, f"Unbalanced braces: {open_braces} open vs {close_braces} close."

    return True, "IR input is valid."


def validate_ir_json(ir_data: dict) -> tuple:
    """
    Validate parsed IR JSON structure.

    Returns:
        (is_valid: bool, message: str)
    """
    if not isinstance(ir_data, dict):
        return False, "IR data is not a dict."

    functions = ir_data.get("functions", [])
    if not isinstance(functions, list):
        return False, "'functions' key missing or not a list."

    if len(functions) == 0:
        return False, "No functions found in IR data."

    if len(functions) > MAX_FUNCTIONS:
        return False, f"Too many functions: {len(functions)} (max {MAX_FUNCTIONS})."

    for func in functions:
        if not isinstance(func, dict):
            return False, "Function entry is not a dict."
        if "name" not in func:
            return False, "Function missing 'name' key."

        # Check instruction count for non-declarations
        if not func.get("is_declaration", False):
            total_insts = 0
            for block in func.get("blocks", []):
                total_insts += len(block.get("instructions", []))
            if total_insts > MAX_INSTRUCTIONS_PER_FUNC:
                return False, (
                    f"Function '{func['name']}' has {total_insts} instructions "
                    f"(max {MAX_INSTRUCTIONS_PER_FUNC})."
                )

    return True, "IR JSON is valid."


# ════════════════════════════════════════════════════════════════════════
# 2. PREDICTION FILTERING
# ════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIDENCE_THRESHOLD = 0.6


def filter_low_confidence(
    predictions: list,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list:
    """
    Remove predictions below the confidence threshold.
    
    'non-vulnerable' class is never filtered — it's always included.
    
    Args:
        predictions: List of prediction dicts with 'class', 'probability', 'detected' keys.
        threshold:   Minimum probability to keep a CWE prediction.
    
    Returns:
        Filtered prediction list.
    """
    filtered = []
    removed  = []

    for pred in predictions:
        cwe = pred.get("class", "")

        # Always keep non-vulnerable class
        if cwe == "non-vulnerable":
            filtered.append(pred)
            continue

        prob = pred.get("probability", 0.0)

        if prob >= threshold:
            filtered.append(pred)
        else:
            # Mark as not detected but keep in list for transparency
            removed.append(cwe)
            filtered.append({
                **pred,
                "detected": False,
                "filtered_reason": f"Below confidence threshold ({prob:.2%} < {threshold:.2%})",
            })

    if removed:
        logger.info(
            "Filtered %d low-confidence prediction(s): %s (threshold=%.2f)",
            len(removed), ", ".join(removed), threshold,
        )

    return filtered


# ════════════════════════════════════════════════════════════════════════
# 3. MODEL STABILITY CHECKS
# ════════════════════════════════════════════════════════════════════════

def check_model_stability(model: torch.nn.Module) -> dict:
    """
    Verify model health and stability.

    Checks:
      - Has dropout layers (prevents overfitting)
      - No NaN or Inf weights
      - Weight statistics (mean, std, min, max)
    
    Returns:
        dict with 'healthy' bool and diagnostic details.
    """
    report = {
        "healthy": True,
        "has_dropout": False,
        "nan_params": [],
        "inf_params": [],
        "weight_stats": {},
        "total_parameters": 0,
        "trainable_parameters": 0,
    }

    # Count parameters
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    report["total_parameters"] = total_params
    report["trainable_parameters"] = trainable_params

    # Check for dropout layers
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
            report["has_dropout"] = True
            break

    if not report["has_dropout"]:
        logger.warning("Model has no Dropout layers — risk of overfitting.")

    # Check for NaN/Inf weights
    all_weights = []
    for name, param in model.named_parameters():
        data = param.data.cpu()
        if torch.isnan(data).any():
            report["nan_params"].append(name)
            report["healthy"] = False
        if torch.isinf(data).any():
            report["inf_params"].append(name)
            report["healthy"] = False
        all_weights.append(data.flatten())

    if all_weights:
        combined = torch.cat(all_weights)
        report["weight_stats"] = {
            "mean": float(combined.mean()),
            "std":  float(combined.std()),
            "min":  float(combined.min()),
            "max":  float(combined.max()),
        }

    if report["nan_params"]:
        logger.error("NaN detected in parameters: %s", report["nan_params"])
    if report["inf_params"]:
        logger.error("Inf detected in parameters: %s", report["inf_params"])

    return report


def compute_gradient_norms(model: torch.nn.Module) -> dict:
    """
    Compute per-layer gradient norms (call after loss.backward()).
    Useful for monitoring training stability.

    Returns:
        dict: layer_name → gradient L2 norm
    """
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = float(param.grad.data.norm(2))
    return norms


def log_gradient_norms(model: torch.nn.Module, epoch: int):
    """Log gradient norms for monitoring during training."""
    norms = compute_gradient_norms(model)
    if norms:
        max_name = max(norms, key=norms.get)
        max_norm = norms[max_name]
        avg_norm = np.mean(list(norms.values()))
        logger.info(
            "Epoch %d gradients — avg: %.4f, max: %.4f (%s)",
            epoch, avg_norm, max_norm, max_name,
        )
