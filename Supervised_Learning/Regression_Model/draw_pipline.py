#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
draw_pipline.py
- Automatically reads the CV metric files for each model and finds the fold with the highest R2.
- Automatically locates the train/test (or valid) prediction CSVs for that fold.
- Calls draw/draw_r2.py to output scatter plots.

Usage (default):
  python draw_pipline.py --out_root results/YoungsModulus
"""

from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


def read_best_fold(metrics_path: Path, metric_col: str = "R2") -> int:
    """
    Compatible with whitespace/comma separation;
    Compatible with 'fold'/'Fold' column names;
    Returns the fold number with the maximum R2.
    """
    # First try reading with whitespace separation
    df = pd.read_csv(metrics_path, sep=r"\s+")
    # If there is only one column but the column name contains a comma, it's comma-separated
    if df.shape[1] == 1 and "," in df.columns[0]:
        df = pd.read_csv(metrics_path)

    df.columns = [c.strip() for c in df.columns]

    fold_col = "fold" if "fold" in df.columns else ("Fold" if "Fold" in df.columns else None)
    if fold_col is None:
        raise ValueError(f"[draw] metrics missing fold/Fold column: {metrics_path}, cols={df.columns.tolist()}")

    if metric_col not in df.columns:
        # Case-insensitive compatibility
        cand = None
        for c in df.columns:
            if c.lower() == metric_col.lower():
                cand = c
                break
        if cand is None:
            raise ValueError(f"[draw] metrics missing {metric_col} column: {metrics_path}, cols={df.columns.tolist()}")
        metric_col = cand

    best_row = df.loc[df[metric_col].idxmax()]
    return int(best_row[fold_col])

def find_fold_pred_csvs(model_dir: Path, best_fold: int, model_type: str):

    fold2 = f"{best_fold:02d}"

    # ===== RF / MLP / SVM =====
    if model_type in {"rf", "mlp", "svm"}:
        train_csv = model_dir / f"fold_{fold2}_train.csv"
        valid_csv = model_dir / f"fold_{fold2}_valid.csv"

        if train_csv.exists() and valid_csv.exists():
            return train_csv, valid_csv

        raise FileNotFoundError(
            f"[draw] {model_type}: {train_csv.name} / {valid_csv.name} not found in {model_dir}"
        )

    # ===== OLS =====
    if model_type == "ols":
        # OLS: Directory names and filenames use non-zero-padded fold numbers
        fold_plain = str(best_fold)
        fold_dir = model_dir / f"fold_{fold_plain}"
    
        if not fold_dir.exists():
            raise FileNotFoundError(f"[draw] OLS: Directory does not exist {fold_dir}")
    
        train_csv = fold_dir / f"fold_{fold_plain}_train.csv"
        valid_csv = fold_dir / f"fold_{fold_plain}_valid.csv"
    
        if train_csv.exists() and valid_csv.exists():
            return train_csv, valid_csv
    
        raise FileNotFoundError(
            f"[draw] OLS: {train_csv.name} / {valid_csv.name} not found in {fold_dir}"
        )

        # Note: The original code had redundant checks here; keeping structure for fidelity
        if train_csv.exists() and valid_csv.exists():
            return train_csv, valid_csv

        raise FileNotFoundError(
            f"[draw] OLS: {train_csv.name} / {valid_csv.name} not found in {fold_dir}"
        )

    raise ValueError(f"[draw] Unknown model_type: {model_type}")


def run_draw(draw_script: Path, train_csv: Path, test_csv: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(draw_script),
        "--train_csv", str(train_csv),
        "--test_csv", str(test_csv),
        "--outdir", str(outdir),
    ]
    print("\n====== DRAW ======")
    print(" ".join(cmd))
    print("==================")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="e.g., results/YoungsModulus")
    ap.add_argument("--metric_col", default="R2", help="Default is R2")
    ap.add_argument("--only", choices=["rf", "mlp", "svm", "ols", "all"], default="all")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    if not out_root.exists():
        raise FileNotFoundError(f"[draw] out_root does not exist: {out_root}")

    draw_script = Path(__file__).parent / "draw" / "draw_r2.py"
    if not draw_script.exists():
        raise FileNotFoundError(f"[draw] Plotting script not found: {draw_script}")

    specs = [
        ("rf",  out_root / "rf_cv10",             out_root / "rf_cv10" / "cv10_metrics.csv"),
        ("mlp", out_root / "runs" / "mlp",        out_root / "runs" / "mlp" / "cv10_metrics.csv"),
        ("svm", out_root / "runs" / "svm",        out_root / "runs" / "svm" / "cv10_metrics.csv"),
        ("ols", out_root / "ols_linear",          out_root / "ols_linear" / "cv_results.csv"),
    ]

    draw_out_root = out_root / "draw"
    draw_out_root.mkdir(parents=True, exist_ok=True)

    for model_type, model_dir, metrics_path in specs:
        if args.only != "all" and args.only != model_type:
            continue

        if not metrics_path.exists():
            print(f"[draw][SKIP] {model_type}: metrics not found {metrics_path}")
            continue
        if not model_dir.exists():
            print(f"[draw][SKIP] {model_type}: model_dir not found {model_dir}")
            continue

        try:
            best_fold = read_best_fold(metrics_path, metric_col=args.metric_col)
            train_csv, test_csv = find_fold_pred_csvs(
                model_dir=model_dir,
                best_fold=best_fold,
                model_type=model_type,
            )

            outdir = draw_out_root / model_type / f"fold_{best_fold:02d}"
            print(f"\n[draw] {model_type}: best_fold={best_fold}")
            print(f"[draw] train_csv: {train_csv}")
            print(f"[draw] test_csv : {test_csv}")
            run_draw(draw_script, train_csv, test_csv, outdir)

            print(f"[draw][OK] {model_type}: Output to {outdir}")

        except Exception as e:
            print(f"[draw][FAIL] {model_type}: {e}")

    print("\n[draw][DONE] Drawing pipeline completed")


if __name__ == "__main__":
    main()