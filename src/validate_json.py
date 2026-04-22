import os
import json
import argparse

def validate_json_file(filepath):
    result = {
        "valid":             False,
        "errors":            [],
        "num_functions":     0,
        "num_declarations":  0,
        "num_defined_funcs": 0,
        "num_blocks":        0,
        "num_instructions":  0,
        "opcodes":           {},
    }
    errors = result["errors"]

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return result
    except Exception as e:
        errors.append(f"Could not open file: {e}")
        return result

    if "source_file" not in data:
        errors.append("Missing top-level key: 'source_file'")
    if "functions" not in data:
        errors.append("Missing top-level key: 'functions'")
        return result       

    if not isinstance(data["functions"], list):
        errors.append("'functions' must be a list")
        return result

    for fi, func in enumerate(data["functions"]):
        func_tag = f"functions[{fi}]"

        for key in ("name", "return_type", "params", "is_declaration", "blocks"):
            if key not in func:
                errors.append(f"{func_tag}: missing key '{key}'")

        if "name" not in func:
            continue   

        fname = func.get("name", f"?{fi}")
        func_tag = f"@{fname}"

        result["num_functions"] += 1
        if func.get("is_declaration"):
            result["num_declarations"] += 1
        else:
            result["num_defined_funcs"] += 1

        if not isinstance(func.get("blocks", []), list):
            errors.append(f"{func_tag}: 'blocks' must be a list")
            continue

        for bi, block in enumerate(func.get("blocks", [])):
            block_tag = f"{func_tag} block[{bi}]"

            for key in ("label", "instructions"):
                if key not in block:
                    errors.append(f"{block_tag}: missing key '{key}'")

            if not isinstance(block.get("instructions", []), list):
                errors.append(f"{block_tag}: 'instructions' must be a list")
                continue

            result["num_blocks"] += 1

            for ii, instr in enumerate(block.get("instructions", [])):
                instr_tag = f"{block_tag} instr[{ii}]"

                for key in ("text", "opcode"):
                    if key not in instr:
                        errors.append(f"{instr_tag}: missing key '{key}'")

                result["num_instructions"] += 1
                opcode = instr.get("opcode", "unknown")
                result["opcodes"][opcode] = result["opcodes"].get(opcode, 0) + 1

    result["valid"] = len(errors) == 0
    return result

#printing stats
def validate_all(input_dir):
    json_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))

    if not json_files:
        print(f"[WARN] No .json files found in '{input_dir}'")
        return

    total           = len(json_files)
    valid_count     = 0
    invalid_count   = 0
    invalid_list    = []

    # complete stats
    agg = {
        "num_functions":     0,
        "num_declarations":  0,
        "num_defined_funcs": 0,
        "num_blocks":        0,
        "num_instructions":  0,
        "opcodes":           {},
    }

    print(f"Validating {total} JSON files in '{input_dir}' ...\n")

    for filename in json_files:
        filepath = os.path.join(input_dir, filename)
        res = validate_json_file(filepath)

        if res["valid"]:
            valid_count += 1
            status = "[OK]  "
        else:
            invalid_count += 1
            invalid_list.append((filename, res["errors"]))
            status = "[FAIL]"

        for key in ("num_functions", "num_declarations", "num_defined_funcs",
                    "num_blocks", "num_instructions"):
            agg[key] += res[key]
        for opcode, cnt in res["opcodes"].items():
            agg["opcodes"][opcode] = agg["opcodes"].get(opcode, 0) + cnt


    print("VALIDATION SUMMARY")
    print()
    print(f"  Total files checked : {total}")
    print(f"  Valid               : {valid_count}")
    print(f"  Invalid             : {invalid_count}")
    print()
    print("CONTENT STATS (across all files)")
    print()
    print(f"  Total functions   : {agg['num_functions']}")
    print(f"  Defined           : {agg['num_defined_funcs']}")
    print(f"  Declarations      : {agg['num_declarations']}")
    print(f"  Total basic blocks: {agg['num_blocks']}")
    print(f"  Total instructions : {agg['num_instructions']}")
    if agg["num_defined_funcs"] > 0:
        avg_blocks = agg["num_blocks"]        / agg["num_defined_funcs"]
        print(f"  Avg blocks/function : {avg_blocks:.1f}")
    if agg["num_blocks"] > 0:
        avg_instrs = agg["num_instructions"] / agg["num_blocks"]
        print(f"  Avg instrs/block    : {avg_instrs:.1f}")

    print()
    print("Some OPCODES found")
    print()
    top_opcodes = sorted(agg["opcodes"].items(), key=lambda x: x[1], reverse=True)[:10]
    for opcode, count in top_opcodes:
        bar = "#" * min(30, count // max(1, agg["num_instructions"] // 50))
        print(f"  {opcode:<20} {count:>8}   {bar}")

    if invalid_list:
        print()
        print("INVALID FILES")
        print("-" * 40)
        for name, errs in invalid_list:
            print(f"  {name}")
            for err in errs[:5]:   # show at most 5 errors per file
                print(f"    -> {err}")
            if len(errs) > 5:
                print(f"    ... ({len(errs) - 5} more errors)")

    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate parsed IR JSON files and print stats")
    parser.add_argument(
        "--input", type=str, default="data/parsed_ir",
        help="Directory containing .json files  (default: data/parsed_ir)"
    )
    args = parser.parse_args()

    validate_all(args.input)
