#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classification Model Plotting Orchestrator (Following your script parameter format):
- Automatically finds the fold with the highest Acc_class_1.
- Locates the train/valid CSVs for that fold.
- Calls:
    1) draw_Matrix.py (Confusion Matrix)
    2) draw_ROC.py    (ROC Curve)
- Outputs to:
    results/<task_name>/draw/rf/fold_XX/CM/
    results/<task_name>/draw/rf/fold_XX/ROC/
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd


def run_cmd(cmd):
    print("\n====== RUN CMD ======")
    print(" ".join(cmd))
    print("=====================\n")
    subprocess.run(cmd, check=True)


def find_best_fold(metrics_csv: Path) -> int:
    df = pd.read_csv(metrics_csv)
    if "Acc_class_1" not in df.columns or "fold" not in df.columns:
        raise ValueError("cv10_metrics.csv is missing 'fold' or 'Acc_class_1' columns")

    row = df.sort_values("Acc_class_1", ascending=False).iloc[0]
    best_fold = int(row["fold"])
    best_acc = float(row["Acc_class_1"])
    print(f"[INFO] Best Fold = {best_fold:02d}, Acc_class_1 = {best_acc:.4f}")
    return best_fold


def get_fold_csvs(cv10_dir: Path, fold: int):
    fold2 = f"{fold:02d}"
    train_csv = cv10_dir / f"fold_{fold2}_train.csv"
    valid_csv = cv10_dir / f"fold_{fold2}_valid.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing file: {train_csv}")
    if not valid_csv.exists():
        raise FileNotFoundError(f"Missing file: {valid_csv}")

    return train_csv, valid_csv


def main():
    ap = argparse.ArgumentParser(description="Best Fold Plotting Pipeline for Classification Model")
    ap.add_argument("--task_dir", required=True, help="results/<task_name> directory, e.g., results/SwellingRatio")

    # ---- Confusion Matrix Parameters (Matching your startup script) ----
    ap.add_argument("--y_col", default="y_true", help="Column name for true labels")
    ap.add_argument("--yhat_col", default="y_pred", help="Column name for predicted labels")
    ap.add_argument("--out_train", default="confmat_train.png", help="Output filename for training matrix")
    ap.add_argument("--out_test", default="confmat_valid.png", help="Output filename for validation matrix")
    ap.add_argument("--cmap", default="Blues", help="Colormap for matrix")
    ap.add_argument("--rotate_xticks", type=int, default=0, help="Rotation angle for x-ticks")
    ap.add_argument("--normalize", default="none", choices=["none", "true", "pred", "all"], help="Normalization mode")

    # ---- ROC Parameters (Matching your startup script) ----
    ap.add_argument("--train_color", default="109,109,255", help="Color for training ROC curve")
    ap.add_argument("--test_color", default="#F3A5D9", help="Color for testing ROC curve")
    ap.add_argument("--fill", action="store_true", default=True, help="Fill area under ROC curve (Default: True)")

    # ---- Control Parameters ----
    ap.add_argument("--skip_existing", action="store_true", help="Skip plotting if files already exist")

    args = ap.parse_args()

    task_dir = Path(args.task_dir).resolve()
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task_dir does not exist: {task_dir}")

    # === Find rf_cls_cv10* directory ===
    rf_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("rf_cls_cv10")]
    if not rf_dirs:
        raise RuntimeError(f"No rf_cls_cv10* directory found in {task_dir}")

    rf_dir = rf_dirs[0]
    print(f"[INFO] Using model directory: {rf_dir.name}")

    metrics_csv = rf_dir / "cv10_metrics.csv"
    cv10_dir = rf_dir / "cv10"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_csv}")
    if not cv10_dir.is_dir():
        raise FileNotFoundError(f"Missing cv10 directory: {cv10_dir}")

    # === Step 1: Find best fold ===
    best_fold = find_best_fold(metrics_csv)

    # === Step 2: Locate CSVs ===
    train_csv, valid_csv = get_fold_csvs(cv10_dir, best_fold)

    # === Step 3: Define output directories ===
    base_out = task_dir / "draw" / "rf" / f"fold_{best_fold:02d}"
    cm_out = base_out / "CM"
    roc_out = base_out / "ROC"
    cm_out.mkdir(parents=True, exist_ok=True)
    roc_out.mkdir(parents=True, exist_ok=True)

    expected = [
        cm_out / args.out_train,
        cm_out / args.out_test,
        roc_out / "roc_train.png",
        roc_out / "roc_test.png",
        roc_out / "roc_train_vs_test.png",
    ]
    if args.skip_existing and all(p.exists() for p in expected):
        print(f"[SKIP] Plot results already exist: {base_out}")
        return

    # === Script location ===
    this_dir = Path(__file__).resolve().parent
    matrix_script = this_dir / "draw_Matrix.py"
    roc_script = this_dir / "draw_ROC.py"
    if not matrix_script.is_file():
        raise FileNotFoundError(f"draw_Matrix.py not found: {matrix_script}")
    if not roc_script.is_file():
        raise FileNotFoundError(f"draw_ROC.py not found: {roc_script}")

    # === Step 4: Confusion Matrix (Strictly following your parameter format) ===
    print("=== Plotting Confusion Matrix ===")
    run_cmd([
        sys.executable, str(matrix_script),
        "--csv_train", str(train_csv),
        "--csv_test", str(valid_csv),
        "--y_col", args.y_col,
        "--yhat_col", args.yhat_col,
        "--out_train", args.out_train,
        "--out_test", args.out_test,
        "--out_dir", str(cm_out),
        "--cmap", args.cmap,
        "--rotate_xticks", str(args.rotate_xticks),
        "--normalize", args.normalize,
    ])

    # === Step 5: ROC (Strictly following your parameter format) ===
    print("=== Plotting ROC Curves ===")
    roc_cmd = [
        sys.executable, str(roc_script),
        "--csv_train", str(train_csv),
        "--csv_test", str(valid_csv),
        "--out_dir", str(roc_out),
        "--train_color", args.train_color,
        "--test_color", args.test_color,
    ]
    if args.fill:
        roc_cmd.append("--fill")
    run_cmd(roc_cmd)

    print("\n[DONE] Classification model plotting completed.")
    print(f"[DONE] Output directory: {base_out}")
    print(f"  - Confusion Matrix: {cm_out}")
    print(f"  - ROC: {roc_out}")


if __name__ == "__main__":
    main()