import os
import json
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ir_parser import parse_ll_file


def convert_all(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ll_files = [f for f in os.listdir(input_dir) if f.endswith(".ll")]

    if not ll_files:
        print(f"[WARN] No .ll files found in '{input_dir}'")
        return

    success = 0
    failed  = 0
    failed_files = []

    total = len(ll_files)
    print(f"Parsing {total} .ll files from '{input_dir}' ...\n")

    for idx, filename in enumerate(sorted(ll_files), start=1):
        ll_path   = os.path.join(input_dir,  filename)
        json_name = filename.replace(".ll", ".json")
        json_path = os.path.join(output_dir, json_name)

        try:
            data = parse_ll_file(ll_path)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            print(f"  [{idx:>4}/{total}] [OK]   {filename}")
            success = int(success) + 1

        except Exception as e:
            print(f"  [{idx:>4}/{total}] [FAIL] {filename}: {e}")
            failed = int(failed) + 1
            failed_files.append(filename)

    print(f"\nDone: {success} succeeded, {failed} failed.")
    print(f"JSON files written to: '{output_dir}'")

    if failed_files:
        fail_log = os.path.join(output_dir, "failed_parsing.txt")
        with open(fail_log, "w") as f:
            f.writelines(name + "\n" for name in failed_files)
        print(f"Failed files logged to: '{fail_log}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .ll files to .json using ir_parser")
    parser.add_argument(
        "--input",  type=str, default="data/ir",
        help="Directory containing .ll files  (default: data/ir)"
    )
    parser.add_argument(
        "--output", type=str, default="data/parsed_ir",
        help="Directory to write .json files  (default: data/parsed_ir)"
    )
    args = parser.parse_args()

    convert_all(args.input, args.output)
