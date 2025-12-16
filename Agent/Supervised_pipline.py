#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supervised Learning Interactive Orchestrator
Revised Version: Adapted for execution within the Agent directory
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(
    cmd: list[str],
    cwd: Path,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Execute command with fixed Python interpreter and CWD."""
    if cmd and cmd[0] == "python":
        cmd = [sys.executable] + cmd[1:]
    elif cmd and cmd[0].endswith("python"):
        cmd[0] = sys.executable

    print("\n====== RUN CMD ======")
    print(f"cwd: {cwd}")
    print(" ".join(cmd))
    print("=====================")

    if not capture:
        p = subprocess.run(cmd, cwd=str(cwd))
        if check and p.returncode != 0:
            p2 = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
            print("----- STDOUT (captured after failure) -----")
            print(p2.stdout or "")
            print("----- STDERR (captured after failure) -----")
            print(p2.stderr or "")
            raise RuntimeError(
                f"Command failed (returncode={p2.returncode})\n"
                f"cwd={cwd}\ncmd={' '.join(cmd)}\n"
                f"--- stderr ---\n{p2.stderr}"
            )
        return p
    else:
        p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
        if p.stdout:
            print("----- STDOUT -----")
            print(p.stdout)
        if p.stderr:
            print("----- STDERR -----")
            print(p.stderr)
        if check and p.returncode != 0:
            raise RuntimeError(
                f"Command failed (returncode={p.returncode})\n"
                f"cwd={cwd}\ncmd={' '.join(cmd)}\n"
                f"--- stderr ---\n{p.stderr}"
            )
        return p


def run_regression_pipeline(supervised_dir: Path, force_rerun: bool, capture: bool) -> None:
    # supervised_dir must point to LIGHT_platform-main/Supervised_Learning
    print("\n=== Running Regression Pipeline ===")
    reg_dir = supervised_dir / "Regression_Model"
    run_pipeline = reg_dir / "run_pipeline.py"
    
    if not run_pipeline.exists():
        raise FileNotFoundError(f"Regression pipeline entry not found: {run_pipeline}")

    cmd = ["python", str(run_pipeline)]
    if force_rerun:
        cmd += ["--force_rerun"]

    run_cmd(cmd, cwd=reg_dir, check=True, capture=capture)
    print("✅ Regression pipeline executed successfully.")


def run_classification_pipeline(supervised_dir: Path, force_rerun: bool, capture: bool) -> None:
    # supervised_dir must point to LIGHT_platform-main/Supervised_Learning
    print("\n=== Running Classification Pipeline ===")
    cls_dir = supervised_dir / "Classification_Model"
    cls_main = cls_dir / "classification_main.py"
    
    if not cls_main.exists():
        raise FileNotFoundError(f"Classification pipeline entry not found: {cls_main}")

    cmd = [
        "python", "classification_main.py",
        "--task_name", "SwellingRatio",
        "--raw_csv", "../DataBase/swelling_ratio.csv",
        "--src_col", "Swelling Ratio (times)",
        "--threshold", "9",
        "--polymer_cols", "SMILE A", "SMILE B", "SMILE C",
        "--do_predict",
        "--predict_in_csv", "../High-throughput predict/kmeans-pooled.csv",
        "--predict_source_csv", "../High-throughput predict/kmeans_results.csv",
    ]
    if force_rerun:
        cmd += ["--force_rerun"]

    run_cmd(cmd, cwd=cls_dir, check=True, capture=capture)
    print("✅ Classification pipeline executed successfully.")


def prompt_menu() -> str:
    print("\n==============================")
    print("Please select a task:")
    print("1. Run Regression")
    print("2. Run Classification")
    print("3. Run Both")
    print("4. Exit")
    print("==============================")
    return input("Enter 1/2/3/4 and press Enter: ").strip()


def main():
    ap = argparse.ArgumentParser(description="Supervised Learning Interactive Controller")
    ap.add_argument("--force_rerun", action="store_true", help="Force re-run (skip existing files)")
    ap.add_argument("--capture", action="store_true", help="Capture stdout/stderr (slower)")
    args = ap.parse_args()

    # --- Path Resolution Fix ---
    # 1. Get directory of current script (LIGHT_platform-main/Agent)
    current_path = Path(__file__).resolve()
    current_dir = current_path.parent
    
    # 2. Go up one level to project root (LIGHT_platform-main)
    project_root = current_dir.parent
    
    # 3. Locate Supervised_Learning directory
    # Logic: Whether script is in Agent or Supervised_Learning, try finding it from root first
    supervised_dir = project_root / "Supervised_Learning"
    
    # Simple fallback check
    if not supervised_dir.exists():
        # Maybe script itself is inside Supervised_Learning (legacy support)
        if (current_dir / "Regression_Model").exists():
            supervised_dir = current_dir
        else:
            print(f"❌ Critical Error: Supervised_Learning directory not found.")
            print(f"Current Path: {current_dir}")
            print(f"Attempted Path: {supervised_dir}")
            sys.exit(1)

    print(f"Current Script Path: {current_dir}")
    print(f"Target Supervised Directory: {supervised_dir}")
    print(f"Using Python: {sys.executable}")

    while True:
        choice = prompt_menu()

        if choice == "4":
            print("Selected: Exit. Terminating program.")
            break

        if choice not in {"1", "2", "3"}:
            print("❌ Invalid input. Please try again.")
            continue

        do_reg = choice in {"1", "3"}
        do_cls = choice in {"2", "3"}

        try:
            if do_reg:
                run_regression_pipeline(supervised_dir, force_rerun=args.force_rerun, capture=args.capture)
            if do_cls:
                run_classification_pipeline(supervised_dir, force_rerun=args.force_rerun, capture=args.capture)

            print("\n=== Current task completed === ✅ (Returning to menu)")

        except Exception as e:
            print("\n❌ Task Failed:")
            print(str(e))
            print("(Returning to menu, you can retry)")


if __name__ == "__main__":
    main()