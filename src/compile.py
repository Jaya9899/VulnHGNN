import subprocess
import os

def _clang_windows_system_includes() -> list[str]:
    if os.name != "nt":
        return []
    include_env = os.environ.get("INCLUDE", "")
    if not include_env:
        return []
    parts = [p.strip() for p in include_env.split(";") if p.strip()]
    # Use -isystem so they behave like system headers (less noise).
    args: list[str] = []
    for p in parts:
        args += ["-isystem", p]
    return args

def compile_to_ir(source_dir, output_dir, support_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success = 0
    failed = 0
    failed_files = []

    win_sys_includes = _clang_windows_system_includes()
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if not (filename.endswith('.c') or filename.endswith('.cpp')):
                continue

            source_path = os.path.join(root, filename)
            folder_id = os.path.basename(root)
            out_name = f"{folder_id}_{filename}".replace('.c', '.ll').replace('.cpp', '.ll')
            ir_path = os.path.join(output_dir, out_name)

            cmd = [
                "clang", "-S", "-emit-llvm", "-g", "-O0",
                "-Xclang", "-disable-O0-optnone",
                "-I", support_dir,
                *win_sys_includes,
                "-Wno-everything",
                source_path, "-o", ir_path
            ]

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"[OK] {folder_id}/{filename}")
                success += 1
            except subprocess.CalledProcessError as e:
                print(f"[FAIL] {folder_id}/{filename}: {e.stderr[:200]}")
                failed += 1
                failed_files.append(f"{folder_id}/{filename}")

    print(f"\nDone: {success} succeeded, {failed} failed")
    if failed_files:
        with open("failed_compilations.txt", "w") as f:
            f.writelines(f"{name}\n" for name in failed_files)
        print(f"Failed files saved to failed_compilations.txt")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compile C files to LLVM IR')
    parser.add_argument('--source', type=str, default='dataset/upload_source_1', help='Root source directory')
    parser.add_argument('--output', type=str, default='data/ir', help='Output directory for .ll files')
    parser.add_argument('--support', type=str, default='dataset/test_case_support', help='Path to Juliet testcasesupport folder')

    args = parser.parse_args()
    compile_to_ir(args.source, args.output, args.support)