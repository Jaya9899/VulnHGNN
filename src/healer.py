"""
healer.py — Rule-based self-healing module for C/C++ vulnerability remediation.

Applies targeted, CWE-specific code transformations to patch detected vulnerabilities.
Supported CWEs:
  - CWE-190: Integer Overflow  → safe arithmetic with overflow checks
  - CWE-191: Integer Underflow → safe subtraction with underflow guards
  - CWE-369: Divide by Zero    → zero-divisor guards before division
  - CWE-476: NULL Pointer Deref → NULL checks after malloc/calloc/realloc
"""

import re


# ─── CWE-369: Division by Zero ───────────────────────────────────────────────

def _heal_cwe369(code: str) -> str:
    """
    Finds unguarded division operations and wraps them in a zero-check.
    Handles both '/' and '%' operators on integer or float types.
    """
    lines = code.split('\n')
    healed = []
    i = 0

    # Pattern: something = expr / divisor;  or  expr / divisor (standalone)
    # We look for lines with / or % where the divisor is a variable
    div_pattern = re.compile(
        r'^(\s*)'                          # leading whitespace
        r'(.+?)'                           # LHS (e.g., "return (float)sum")
        r'\s*([/%])\s*'                    # operator
        r'(\w+)'                           # divisor variable name
        r'\s*;'                            # semicolon
    )

    while i < len(lines):
        line = lines[i]
        m = div_pattern.match(line)

        if m:
            indent = m.group(1)
            lhs = m.group(2).strip()
            op = m.group(3)
            divisor = m.group(4)

            # Skip if divisor is a numeric literal (not actually risky)
            if re.match(r'^\d+$', divisor):
                healed.append(line)
                i += 1
                continue

            # Check if there's already a guard (look 1-3 lines above)
            already_guarded = False
            for back in range(1, min(4, i + 1)):
                prev = lines[i - back]
                if divisor in prev and ('!= 0' in prev or '> 0' in prev or '== 0' in prev):
                    already_guarded = True
                    break

            if already_guarded:
                healed.append(line)
            else:
                # Determine return type for the fallback value
                # If lhs contains "return", use return with a safe default
                if 'return' in lhs:
                    healed.append(f'{indent}if ({divisor} == 0) {{')
                    healed.append(f'{indent}    fprintf(stderr, "[HEALED] Division by zero prevented for \\"{divisor}\\"\\n");')
                    healed.append(f'{indent}    return 0; /* HEALED: CWE-369 zero-divisor guard */')
                    healed.append(f'{indent}}}')
                    healed.append(line)
                else:
                    healed.append(f'{indent}if ({divisor} != 0) {{ /* HEALED: CWE-369 zero-divisor guard */')
                    healed.append(f'{indent}    {lhs} {op} {divisor};')
                    healed.append(f'{indent}}} else {{')
                    healed.append(f'{indent}    fprintf(stderr, "[HEALED] Division by zero prevented for \\"{divisor}\\"\\n");')
                    healed.append(f'{indent}}}')
        else:
            healed.append(line)

        i += 1

    return '\n'.join(healed)


# ─── CWE-476: NULL Pointer Dereference ───────────────────────────────────────

def _heal_cwe476(code: str) -> str:
    """
    Finds malloc/calloc/realloc calls where the returned pointer is used
    without a NULL check, and inserts one.
    """
    lines = code.split('\n')
    healed = []
    i = 0

    # Pattern: Type *ptr = (Type *)malloc(...)   or   ptr = malloc(...)
    alloc_pattern = re.compile(
        r'^(\s*)'                                # indent
        r'(?:\w[\w\s\*]*\s+)?'                   # optional type decl
        r'(\*?\w+)'                              # pointer variable name
        r'\s*=\s*'                                # assignment
        r'(?:\([^)]*\)\s*)?'                      # optional cast
        r'(malloc|calloc|realloc)\s*\('           # allocator call
    )

    while i < len(lines):
        line = lines[i]
        m = alloc_pattern.match(line)

        if m:
            indent = m.group(1)
            ptr_name = m.group(2).lstrip('*')
            
            # Check if there's already a NULL check within the next 3 lines
            already_guarded = False
            for ahead in range(1, min(4, len(lines) - i)):
                future = lines[i + ahead]
                if ptr_name in future and ('== NULL' in future or '!= NULL' in future or f'!{ptr_name}' in future or f'if ({ptr_name})' in future):
                    already_guarded = True
                    break

            healed.append(line)

            if not already_guarded:
                healed.append(f'{indent}if ({ptr_name} == NULL) {{ /* HEALED: CWE-476 NULL guard */')
                healed.append(f'{indent}    fprintf(stderr, "[HEALED] NULL pointer caught for \\"{ptr_name}\\"\\n");')
                healed.append(f'{indent}    return; /* or return NULL / return -1 as appropriate */')
                healed.append(f'{indent}}}')
        else:
            healed.append(line)

        i += 1

    return '\n'.join(healed)


# ─── CWE-190: Integer Overflow ──────────────────────────────────────────────

def _heal_cwe190(code: str) -> str:
    """
    Finds integer multiplication and addition patterns that could overflow,
    and wraps them in safe overflow-checking guards.
    """
    lines = code.split('\n')
    healed = []
    i = 0

    # Pattern: int var = a * b;  or  int var = a + b;
    arith_pattern = re.compile(
        r'^(\s*)'                       # indent
        r'((?:unsigned\s+)?(?:int|long|short|size_t)\s+)'  # type
        r'(\w+)'                        # result variable
        r'\s*=\s*'                      # =
        r'(\w+)'                        # operand a
        r'\s*([+*])\s*'                # operator
        r'(\w+)'                        # operand b
        r'\s*;'
    )

    while i < len(lines):
        line = lines[i]
        m = arith_pattern.match(line)

        if m:
            indent = m.group(1)
            type_str = m.group(2).strip()
            result = m.group(3)
            op_a = m.group(4)
            operator = m.group(5)
            op_b = m.group(6)

            # Skip if operands are numeric literals
            if re.match(r'^\d+$', op_a) and re.match(r'^\d+$', op_b):
                healed.append(line)
                i += 1
                continue

            if operator == '*':
                healed.append(f'{indent}/* HEALED: CWE-190 overflow guard for {op_a} * {op_b} */')
                healed.append(f'{indent}if ({op_a} != 0 && {op_b} > __INT_MAX__ / {op_a}) {{')
                healed.append(f'{indent}    fprintf(stderr, "[HEALED] Integer overflow prevented in {op_a} * {op_b}\\n");')
                healed.append(f'{indent}    {type_str} {result} = __INT_MAX__;')
                healed.append(f'{indent}}} else {{')
                healed.append(f'{indent}    {line.strip()}')
                healed.append(f'{indent}}}')
            elif operator == '+':
                healed.append(f'{indent}/* HEALED: CWE-190 overflow guard for {op_a} + {op_b} */')
                healed.append(f'{indent}if ({op_a} > __INT_MAX__ - {op_b}) {{')
                healed.append(f'{indent}    fprintf(stderr, "[HEALED] Integer overflow prevented in {op_a} + {op_b}\\n");')
                healed.append(f'{indent}    {type_str} {result} = __INT_MAX__;')
                healed.append(f'{indent}}} else {{')
                healed.append(f'{indent}    {line.strip()}')
                healed.append(f'{indent}}}')
            else:
                healed.append(line)
        else:
            healed.append(line)

        i += 1

    return '\n'.join(healed)


# ─── CWE-191: Integer Underflow ──────────────────────────────────────────────

def _heal_cwe191(code: str) -> str:
    """
    Finds integer subtraction patterns that could underflow,
    and wraps them in underflow-checking guards.
    """
    lines = code.split('\n')
    healed = []
    i = 0

    # Pattern: type var = a - b;
    sub_pattern = re.compile(
        r'^(\s*)'
        r'((?:unsigned\s+)?(?:int|long|short|size_t)\s+)'
        r'(\w+)'
        r'\s*=\s*'
        r'(?:\([^)]*\)\s*)?'           # optional cast
        r'(\w+)'
        r'\s*-\s*'
        r'(?:\([^)]*\)\s*)?'           # optional cast
        r'(\w+)'
        r'\s*;'
    )

    while i < len(lines):
        line = lines[i]
        m = sub_pattern.match(line)

        if m:
            indent = m.group(1)
            type_str = m.group(2).strip()
            result = m.group(3)
            op_a = m.group(4)
            op_b = m.group(5)

            # Skip numeric literals
            if re.match(r'^\d+$', op_a) and re.match(r'^\d+$', op_b):
                healed.append(line)
                i += 1
                continue

            # Check if type is unsigned
            is_unsigned = 'unsigned' in type_str

            if is_unsigned:
                healed.append(f'{indent}/* HEALED: CWE-191 underflow guard for {op_a} - {op_b} */')
                healed.append(f'{indent}if ({op_a} < {op_b}) {{')
                healed.append(f'{indent}    fprintf(stderr, "[HEALED] Unsigned underflow prevented in {op_a} - {op_b}\\n");')
                healed.append(f'{indent}    {type_str} {result} = 0;')
                healed.append(f'{indent}}} else {{')
                healed.append(f'{indent}    {line.strip()}')
                healed.append(f'{indent}}}')
            else:
                # Signed underflow: check if result wraps
                healed.append(f'{indent}/* HEALED: CWE-191 underflow guard for {op_a} - {op_b} */')
                healed.append(f'{indent}if ({op_b} > 0 && {op_a} < (-__INT_MAX__ - 1) + {op_b}) {{')
                healed.append(f'{indent}    fprintf(stderr, "[HEALED] Integer underflow prevented in {op_a} - {op_b}\\n");')
                healed.append(f'{indent}    {type_str} {result} = (-__INT_MAX__ - 1);')
                healed.append(f'{indent}}} else {{')
                healed.append(f'{indent}    {line.strip()}')
                healed.append(f'{indent}}}')
        else:
            healed.append(line)

        i += 1

    return '\n'.join(healed)


# ─── Ensure stdio is included (for fprintf) ─────────────────────────────────

def _ensure_stdio(code: str) -> str:
    """Make sure <stdio.h> is included since healed code uses fprintf."""
    if '#include <stdio.h>' not in code and '#include<stdio.h>' not in code:
        # Insert after last #include or at top
        lines = code.split('\n')
        last_include = -1
        for idx, line in enumerate(lines):
            if line.strip().startswith('#include'):
                last_include = idx
        if last_include >= 0:
            lines.insert(last_include + 1, '#include <stdio.h>  /* added by healer */')
        else:
            lines.insert(0, '#include <stdio.h>  /* added by healer */')
        return '\n'.join(lines)
    return code


# ─── Public API ──────────────────────────────────────────────────────────────

# Maps CWE class names to their healing functions
HEALERS = {
    "CWE-369": _heal_cwe369,
    "CWE-476": _heal_cwe476,
    "CWE-190": _heal_cwe190,
    "CWE-191": _heal_cwe191,
}

# Human-readable descriptions of what each healer does
HEAL_DESCRIPTIONS = {
    "CWE-369": "Inserted zero-divisor guard before division/modulo operations.",
    "CWE-476": "Inserted NULL pointer check after memory allocation calls.",
    "CWE-190": "Inserted overflow boundary check before integer multiplication/addition.",
    "CWE-191": "Inserted underflow boundary check before integer subtraction.",
}


def heal_code(source_code: str, detected_cwes: list[str]) -> dict:
    """
    Apply targeted healing transformations for each detected CWE.

    Args:
        source_code:   The original C/C++ source code string.
        detected_cwes: List of CWE class strings, e.g. ["CWE-369", "CWE-476"].

    Returns:
        dict with:
            - healed_code: The patched source code string.
            - patches: List of dicts describing each patch applied.
            - success: bool
    """
    if not detected_cwes:
        return {
            "success": True,
            "healed_code": source_code,
            "patches": [],
            "message": "No vulnerabilities detected — code is already clean.",
        }

    code = source_code
    patches = []

    for cwe in detected_cwes:
        healer = HEALERS.get(cwe)
        if healer:
            original = code
            code = healer(code)
            if code != original:
                patches.append({
                    "cwe": cwe,
                    "description": HEAL_DESCRIPTIONS.get(cwe, "Applied fix."),
                    "applied": True,
                })
            else:
                patches.append({
                    "cwe": cwe,
                    "description": f"No actionable pattern found for {cwe} (code may use non-standard patterns).",
                    "applied": False,
                })
        else:
            patches.append({
                "cwe": cwe,
                "description": f"No healer available for {cwe}.",
                "applied": False,
            })

    # Ensure stdio.h is present if we injected fprintf calls
    if any(p["applied"] for p in patches):
        code = _ensure_stdio(code)

    return {
        "success": True,
        "healed_code": code,
        "patches": patches,
        "message": f"Applied {sum(1 for p in patches if p['applied'])} patch(es).",
    }
