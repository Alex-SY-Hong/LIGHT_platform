#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-Click Pipeline main.py

Assumed Directory Structure:
Regression_Model/
  ├─ main.py                 (This file)
  ├─ main_regression/
  │    ├─ morgan_pooling.py
  │    ├─ baseline_RF.py
  │    ├─ baseline_mlp_svm.py
  │    ├─ baseline_OLS_linear_regression.py
  │    ├─ train_mlp_svm_pipeline.py
  │    └─ ...
  ├─ grid/
  │    ├─ rf_grid_loop.py
  │    ├─ grid_mlp.py
  │    └─ grid_svm.py
  └─ predict/
       └─ predict.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

import json
import shutil
from typing import Optional
import pandas as pd

# ===== Utilities =====
def run_cmd(cmd):
    print("\n====== RUN CMD ======")
    print(" ".join(cmd))
    print("=====================\n")
    subprocess.run(cmd, check=True)


def file_exists_and_valid(filepath: Path, min_size: int = 10) -> bool:
    """Check if file exists and is not empty."""
    if not filepath.exists():
        return False
    if not filepath.is_file():
        return False
    if filepath.stat().st_size < min_size:
        return False
    return True


def json_file_valid(filepath: Path) -> bool:
    """Check if JSON file exists and is valid."""
    if not file_exists_and_valid(filepath):
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


def csv_file_valid(filepath: Path, min_rows: int = 1) -> bool:
    """Check if CSV file exists and is valid."""
    if not file_exists_and_valid(filepath):
        return False
    
    try:
        df = pd.read_csv(filepath, nrows=5)  # Read only first 5 rows for checking
        if len(df) >= min_rows:
            return True
        return False
    except Exception:
        return False


# ==== Common best_params.json Loader ====
def load_best_params(json_path: Path):
    """
    Read best_params.json / mlp_grid_best.json / svm_grid_best.json from the given path.
    - If file doesn't exist: Return {}
    - If it contains {"best_params": {...}}: Automatically extract best_params
    - Automatically convert sklearn Pipeline style keys (e.g., "mlp__activation") into
      "mlp_activation" / "svm_C" format usable by subsequent code.
    """
    if not json_path.is_file():
        print(f"[WARN] Hyperparameter file not found: {json_path}. Using default parameters in main.py.")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    print(f"[INFO] Hyperparameter file detected: {json_path}")
    # Compatible with {"best_params": {...}, "best_r2": ...} structure
    if isinstance(obj, dict) and "best_params" in obj and isinstance(obj["best_params"], dict):
        params_raw = obj["best_params"]
    else:
        params_raw = obj

    # Normalize keys:
    # - "mlp__activation" -> "mlp_activation"
    # - "svm__C" -> "svm_C"
    # - Others remain unchanged
    params_norm = {}
    if isinstance(params_raw, dict):
        for k, v in params_raw.items():
            if "__" in k:
                step, name = k.split("__", 1)
                if step in ("mlp", "svm", "rf"):
                    new_key = f"{step}_{name}"
                else:
                    new_key = name
            else:
                new_key = k
            params_norm[new_key] = v
    else:
        # Extreme cases (e.g., direct list), return as is
        params_norm = params_raw

    print(f"[INFO] Parsed hyperparameters: {params_norm}")
    return params_norm


def select_best_model_from_cv(
    metrics_file: Path,
    fold_models_dir: Path,
    metric_col: str = "R2",
    best_model_name: str = "best_model.joblib",
):

    if not metrics_file.is_file():
        print(f"[WARN] Metrics file not found: {metrics_file}. Skipping best fold selection.")
        return

    if not fold_models_dir.is_dir():
        print(f"[WARN] Model directory not found: {fold_models_dir}. Skipping best fold selection.")
        return

    print(f"[INFO] Reading CV results from {metrics_file}...")

    # Try whitespace separator first
    df = pd.read_csv(metrics_file, sep=r"\s+")
    # If only one column is read and header contains comma, it's actually comma-separated
    if df.shape[1] == 1 and "," in df.columns[0]:
        df = pd.read_csv(metrics_file, sep=",")
    df.columns = [c.strip() for c in df.columns]

    if "fold" not in df.columns:
        print(f"[WARN] Missing 'fold' column in metrics file. Actual columns: {df.columns.tolist()}")
        return

    # Compatible with R2 / r2 / R2_mean etc.
    if metric_col not in df.columns:
        metric_lower = metric_col.lower()
        candidate = None
        for c in df.columns:
            if c.lower() == metric_lower:
                candidate = c
                break
        if candidate is None:
            print(f"[WARN] Missing '{metric_col}' column in metrics file. Actual columns: {df.columns.tolist()}")
            return
        metric_col = candidate

    best_row = df.loc[df[metric_col].idxmax()]
    best_fold = int(best_row["fold"])
    best_score = float(best_row[metric_col])
    print(f"[INFO] Best fold in metrics: fold={best_fold}, {metric_col}={best_score:.4f}")

    # Naming template: fold_01_best_model.joblib
    src_model = fold_models_dir / f"fold_{best_fold:02d}_best_model.joblib"
    if not src_model.is_file():
        print(f"[WARN] Corresponding model file not found: {src_model}")
        print(f"[WARN] Files in current model directory: {[p.name for p in fold_models_dir.iterdir()]}")
        return

    best_model_path = fold_models_dir / best_model_name
    shutil.copy(src_model, best_model_path)
    print(f"[INFO] Best model selected: {src_model.name} -> {best_model_path.name}")


def select_best_ols_model(
    ols_dir: Path,
    metric_col: str = "R2",
    best_model_subdir: str = "fold_models",
    best_model_name: str = "best_model.joblib",
):
    """
    OLS Special:
    - Find the fold (Fold column) with max metric_col from ols_dir/cv_results.csv
    - Retrieve model from ols_dir/fold_{i}/model.pkl
    - Copy to ols_dir/fold_models/best_model.joblib
    """
    metrics_file = ols_dir / "cv_results.csv"
    if not metrics_file.is_file():
        print(f"[WARN] OLS metrics file not found: {metrics_file}. Skipping OLS best fold selection.")
        return

    print(f"[INFO] Reading OLS CV results from {metrics_file}...")
    df = pd.read_csv(metrics_file)
    df.columns = [c.strip() for c in df.columns]

    if "Fold" not in df.columns:
        print(f"[WARN] Missing 'Fold' column in OLS cv_results.csv. Actual columns: {df.columns.tolist()}")
        return
    if metric_col not in df.columns:
        print(f"[WARN] Missing '{metric_col}' column in OLS cv_results.csv. Actual columns: {df.columns.tolist()}")
        return

    best_row = df.loc[df[metric_col].idxmax()]
    best_fold = int(best_row["Fold"])
    best_score = float(best_row[metric_col])
    print(f"[INFO] OLS Best Fold: Fold={best_fold}, {metric_col}={best_score:.4f}")

    # Source model path: ols_linear/fold_{best_fold}/model.pkl
    src_model = ols_dir / f"fold_{best_fold}" / "model.pkl"
    if not src_model.is_file():
        print(f"[WARN] OLS best fold model file not found: {src_model}")
        return

    # Destination path: ols_linear/fold_models/best_model.joblib
    fold_models_dir = ols_dir / best_model_subdir
    fold_models_dir.mkdir(parents=True, exist_ok=True)
    dst_model = fold_models_dir / best_model_name

    shutil.copy(src_model, dst_model)
    print(f"[INFO] OLS best model selected: {src_model} -> {dst_model}")

def collect_model_candidates(out_root: Path, metric_col: str = "R2", dump_json: bool = True):
    """
    Summarize best fold info for RF / MLP / SVM / OLS from 10-fold CV, return candidates list:
        {
          "model_type": "rf/mlp/svm/ols",
          "best_fold": int,
          "best_score": float,
          "metrics_file": Path,
          "model_dir": Path,
          "pred_csv": Optional[Path],
        }

    Optionally write simplified version to out_root / model_candidates_for_llm.json,
    Structure is a list, elements like:
        {"model_type": "rf", "best_fold": 8, "best_score": 0.63, "pred_csv": "predictions/RF_best_pred_xxx.csv"}
    """
    candidates = []

    # ---- RF ----
    rf_dir = out_root / "rf_cv10"
    rf_metrics = rf_dir / "cv10_metrics.csv"
    rf_pred = out_root / "predictions" / "RF_best_pred_*.csv"  # Hint only, real path updated below
    if rf_metrics.is_file():
        df = pd.read_csv(rf_metrics)
        df.columns = [c.strip() for c in df.columns]
        if "fold" in df.columns and metric_col in df.columns:
            row = df.loc[df[metric_col].idxmax()]
            candidates.append({
                "model_type": "rf",
                "best_fold": int(row["fold"]),
                "best_score": float(row[metric_col]),
                "metrics_file": rf_metrics,
                "model_dir": rf_dir,
                # pred_csv placeholder, main will fill real path later
                "pred_csv": None,
            })
        else:
            print(f"[WARN] RF metrics missing fold or {metric_col} column. Actual columns: {df.columns.tolist()}")

    # ---- MLP ----
    mlp_dir = out_root / "runs" / "mlp"
    mlp_metrics = mlp_dir / "cv10_metrics.csv"
    if mlp_metrics.is_file():
        df = pd.read_csv(mlp_metrics)
        df.columns = [c.strip() for c in df.columns]
        if "fold" in df.columns and metric_col in df.columns:
            row = df.loc[df[metric_col].idxmax()]
            candidates.append({
                "model_type": "mlp",
                "best_fold": int(row["fold"]),
                "best_score": float(row[metric_col]),
                "metrics_file": mlp_metrics,
                "model_dir": mlp_dir,
                "pred_csv": None,
            })
        else:
            print(f"[WARN] MLP metrics missing fold or {metric_col} column. Actual columns: {df.columns.tolist()}")

    # ---- SVM ----
    svm_dir = out_root / "runs" / "svm"
    svm_metrics = svm_dir / "cv10_metrics.csv"
    if svm_metrics.is_file():
        df = pd.read_csv(svm_metrics)
        df.columns = [c.strip() for c in df.columns]
        if "fold" in df.columns and metric_col in df.columns:
            row = df.loc[df[metric_col].idxmax()]
            candidates.append({
                "model_type": "svm",
                "best_fold": int(row["fold"]),
                "best_score": float(row[metric_col]),
                "metrics_file": svm_metrics,
                "model_dir": svm_dir,
                "pred_csv": None,
            })
        else:
            print(f"[WARN] SVM metrics missing fold or {metric_col} column. Actual columns: {df.columns.tolist()}")

    # ---- OLS ----
    ols_dir = out_root / "ols_linear"
    ols_metrics = ols_dir / "cv_results.csv"
    if ols_metrics.is_file():
        df = pd.read_csv(ols_metrics)
        df.columns = [c.strip() for c in df.columns]
        if "Fold" in df.columns and metric_col in df.columns:
            row = df.loc[df[metric_col].idxmax()]
            candidates.append({
                "model_type": "ols",
                "best_fold": int(row["Fold"]),
                "best_score": float(row[metric_col]),
                "metrics_file": ols_metrics,
                "model_dir": ols_dir,
                "pred_csv": None,
            })
        else:
            print(f"[WARN] OLS metrics missing Fold or {metric_col} column. Actual columns: {df.columns.tolist()}")

    if not candidates:
        print("[WARN] No available CV metrics found. Cannot build candidate list.")

    # Write a simplified JSON for local LLM
    if dump_json and candidates:
        simple = [
            {
                "model_type": c["model_type"],
                "best_fold": c["best_fold"],
                "best_score": c["best_score"],
                "pred_csv": str(c["pred_csv"]) if c["pred_csv"] is not None else None,
            }
            for c in candidates
        ]
        json_path = out_root / "model_candidates_for_llm.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(simple, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Written model candidate JSON: {json_path}")

    return candidates

# ===== Wrappers for invoking sub-scripts =====
def run_morgan_pooling(
    main_dir: Path,
    raw_csv: str,
    out_csv: str,
    polymer_cols,
    alpha: float,
    radius: int,
    nbits: int,
    target_col: str = None,
):
    script = main_dir / "morgan_pooling.py"
    cmd = [
        sys.executable,
        str(script),
        "--in_csv",
        raw_csv,
        "--out_csv",
        out_csv,
        "--alpha",
        str(alpha),
        "--radius",
        str(radius),
        "--nbits",
        str(nbits),
    ]

    if polymer_cols:
        cmd += ["--polymer_cols", *polymer_cols]

    if target_col:
        cmd += ["--target_col", target_col]

    run_cmd(cmd)


def run_rf_baseline(
    main_dir: Path,
    feat_csv: str,
    target_col: str,
    save_dir: str,
    best_params: Optional[dict] = None,
):

    script = main_dir / "baseline_RF.py"
    cmd = [
        sys.executable,
        str(script),
        "--in_csv",
        feat_csv,
        "--target",
        target_col,
        "--model",
        "rf",
        "--seed",
        "42",
        "--save_dir",
        save_dir,
        "--cv10",
        "--cv_folds",
        "10",
        "--save_train_pred",
    ]

    if best_params:
        # sklearn param names -> baseline_RF.py CLI param names
        key_map = {
            "n_estimators": "rf_n_estimators",
            "max_depth": "rf_max_depth",
            "min_samples_split": "rf_min_samples_split",
            "min_samples_leaf": "rf_min_samples_leaf",
            "max_features": "rf_max_features",
        }

        mapped = {}
        for k, v in best_params.items():
            cli_key = key_map.get(k)
            if cli_key is None:
                # Ignore extra params from grid search with a warning
                print(f"[WARN] Ignoring unknown RF parameter {k}={v}")
                continue

            # Some parameters should be integers, avoid 15.0
            if cli_key in (
                "rf_n_estimators",
                "rf_max_depth",
                "rf_min_samples_split",
                "rf_min_samples_leaf",
            ):
                try:
                    v = int(v)
                except Exception:
                    pass

            mapped[cli_key] = v

        print("[INFO] Applying mapped RF hyperparameters:", mapped)
        for k, v in mapped.items():
            cmd += [f"--{k}", str(v)]
    else:
        # Use hardcoded defaults if no best_params
        cmd += [
            "--rf_n_estimators",
            "400",
            "--rf_max_depth",
            "15",
            "--rf_max_features",
            "0.2",
            "--rf_min_samples_leaf",
            "2",
            "--rf_min_samples_split",
            "12",
        ]

    run_cmd(cmd)


def run_mlp_svm_baseline(
    main_dir: Path,
    feat_csv: str,
    target_col: str,
    out_root: str,
    cv_folds: int,
    mlp_best_params: Optional[dict] = None,
    svm_best_params: Optional[dict] = None,
):
    """
    Corresponding MLP + SVM command:

    python train_mlp_svm_pipeline.py \
      --in_csv  Path/SMILES-pooled-morgan.csv \
      --target  "Young's Modulus (kPa) log10" \
      --out_root Path/runs \
      --cv10 1 \
      --cv_folds 10 \
      [ + some MLP/SVM hyperparameters ]
    """
    script = main_dir / "train_mlp_svm_pipeline.py"
    cmd = [
        sys.executable,
        str(script),
        "--in_csv",
        feat_csv,
        "--target",
        target_col,
        "--out_root",
        out_root,
        "--cv10",
        "1",
        "--cv_folds",
        str(cv_folds),
    ]

    # ===== MLP Parameter Mapping =====
    if mlp_best_params:
        # Keys are already mlp_activation / mlp_alpha / mlp_hidden_layer_sizes / mlp_learning_rate_init
        # Need to map to train_mlp_svm_pipeline.py CLI names
        key_map = {
            "mlp_activation": "mlp_activation",
            "mlp_alpha": "mlp_alpha",
            "mlp_hidden_layer_sizes": "mlp_hidden",       # Note name change
            "mlp_learning_rate_init": "mlp_lr",           # Note name change
        }

        mapped_mlp = {}
        for k, v in mlp_best_params.items():
            cli_key = key_map.get(k)
            if cli_key is None:
                print(f"[WARN] Ignoring unknown MLP parameter {k}={v}")
                continue

            # Special handling for hidden_layer_sizes: list/tuple -> "512,256,128"
            if k == "mlp_hidden_layer_sizes":
                if isinstance(v, (list, tuple)):
                    v = ",".join(str(x) for x in v)
                else:
                    v = str(v)

            mapped_mlp[cli_key] = v

        print("[INFO] Applying mapped MLP hyperparameters:", mapped_mlp)
        for k, v in mapped_mlp.items():
            cmd += [f"--{k}", str(v)]

    # ===== SVM Parameter Mapping =====
    if svm_best_params:
        # Your best_params contain svm_C / svm_epsilon / svm_gamma / svm_kernel,
        # which match train_mlp_svm_pipeline.py CLI exactly.
        print("[INFO] Using SVM hyperparameters from Grid Search:", svm_best_params)
        for k, v in svm_best_params.items():
            # Simple filter to avoid weird keys
            if k not in {"svm_C", "svm_epsilon", "svm_gamma", "svm_kernel"}:
                print(f"[WARN] Ignoring unknown SVM parameter {k}={v}")
                continue
            cmd += [f"--{k}", str(v)]

    run_cmd(cmd)


def run_rf_grid(
    grid_dir: Path,
    feat_csv: str,
    target_col: str,
    save_dir: str,
    id_cols: str,
    test_size: float,
):
    script = grid_dir / "rf_grid_loop.py"
    cmd = [
        sys.executable,
        str(script),
        "--in_csv",
        feat_csv,
        "--target",
        target_col,
        "--save_dir",
        save_dir,
        "--id_cols",
        id_cols,
        "--test_size",
        str(test_size),
        "--seed",
        "42",
        "--final_cv",
        "10",
    ]
    run_cmd(cmd)


def run_mlp_grid(grid_dir: Path, feat_csv: str, target_col: str, save_dir: str):
    script = grid_dir / "grid_mlp.py"
    cmd = [
        sys.executable,
        str(script),
        "--in_csv",
        feat_csv,
        "--target",
        target_col,
        "--save_dir",
        save_dir,
    ]
    run_cmd(cmd)


def run_svm_grid(grid_dir: Path, feat_csv: str, target_col: str, save_dir: str):
    script = grid_dir / "grid_svm.py"
    cmd = [
        sys.executable,
        str(script),
        "--in_csv",
        feat_csv,
        "--target",
        target_col,
        "--save_dir",
        save_dir,
    ]
    run_cmd(cmd)


def run_ols_baseline(
    main_dir: Path,
    feat_csv: str,
    target_col: str,
    save_dir: str,
    n_splits: int = 10,
    seed: int = 42,
):
    script = main_dir / "baseline_OLS_linear_regression.py"
    cmd = [
        sys.executable,
        str(script),
        "--in_csv",
        feat_csv,
        "--target",
        target_col,
        "--out_dir",
        save_dir,
        "--n_splits",
        str(n_splits),
        "--seed",
        str(seed),
    ]
    run_cmd(cmd)


# ===== Prediction Wrapper =====
def run_predict(
    predict_dir: Path,
    in_csv: str,
    source_csv: str,
    out_csv: str,
    model_dir: str,
    target_name: str = "Prediction",
    id_col: str = "row_index",
):
    script = predict_dir / "predict.py"
    cmd = [
        sys.executable,
        str(script),
        "--in_csv",
        in_csv,
        "--source_csv",
        source_csv,
        "--out_csv",
        out_csv,
        "--model_dir",
        model_dir,
        "--target_name",
        target_name,
        "--id_col",
        id_col,
    ]
    run_cmd(cmd)


# ===== Main Controller =====
def main():
    parser = argparse.ArgumentParser(
        description="One-Click: Morgan Fingerprint + RF/MLP/SVM Grid Search + RF/MLP/SVM baseline (10-fold CV) + OLS baseline + Predict"
    )

    # ---- Required: Input / Target / Output Root ----
    parser.add_argument(
        "--raw_csv",
        required=True,
        help="Raw training CSV (contains SMILES + target column)",
    )
    parser.add_argument(
        "--target",
        required=True,
        help='Target column name, e.g., "Young\'s Modulus (kPa) log10"',
    )
    parser.add_argument(
        "--out_root",
        required=True,
        help="Root output directory for the whole task (subdirectories created automatically)",
    )

    # ---- Morgan Parameters ----
    parser.add_argument(
        "--polymer_cols",
        nargs="+",
        default=["SMILE A", "SMILE B", "SMILE C"],
        help='Polymer SMILES column names, e.g.: --polymer_cols "SMILE A" "SMILE B" "SMILE C"',
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=3.0,
        help="--alpha for morgan_pooling, default 3.0",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=3,
        help="Morgan radius, default 3",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=1024,
        help="Morgan fingerprint length, default 1024",
    )

    # ---- Extra Params for RF Grid ----
    parser.add_argument(
        "--rf_id_cols",
        default="SampleID,RecipeID",
        help='--id_cols for rf_grid_loop.py, default "SampleID,RecipeID"',
    )
    parser.add_argument(
        "--rf_test_size",
        type=float,
        default=0.2,
        help="--test_size for rf_grid_loop.py, default 0.2",
    )

    # ---- Specify grid result directories (Optional, default under out_root) ----
    parser.add_argument(
        "--rf_grid_dir",
        help="RF Grid Search result dir (contains best_params.json), default out_root/rf_grid",
    )
    parser.add_argument(
        "--mlp_grid_dir",
        help="MLP Grid Search result dir (contains mlp_grid_best.json), default out_root/mlp_grid",
    )
    parser.add_argument(
        "--svm_grid_dir",
        help="SVM Grid Search result dir (contains svm_grid_best.json), default out_root/svm_grid",
    )

    # ---- Flags: Skip certain training steps ----
    parser.add_argument(
        "--no_rf_baseline",
        action="store_true",
        help="Do not run baseline_RF.py",
    )
    parser.add_argument(
        "--no_mlp_svm_baseline",
        action="store_true",
        help="Do not run train_mlp_svm_pipeline.py",
    )
    parser.add_argument(
        "--no_rf_grid",
        action="store_true",
        help="Do not run rf_grid_loop.py",
    )
    parser.add_argument(
        "--no_mlp_grid",
        action="store_true",
        help="Do not run grid_mlp.py",
    )
    parser.add_argument(
        "--no_svm_grid",
        action="store_true",
        help="Do not run grid_svm.py",
    )
    parser.add_argument(
        "--no_ols_baseline",
        action="store_true",
        help="Do not run baseline_OLS_linear_regression.py",
    )

    # ---- Option to skip existing files ----
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,  # Default on
        help="Skip existing files (default on)",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force rerun all steps (overrides --skip_existing)",
    )

    # ---- Flags + Params: Prediction (Supports High-Throughput Libs) ----
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Perform prediction on all available models (RF/MLP/SVM/OLS) after training",
    )
    parser.add_argument(
        "--predict_in_csv",
        help="Feature file for prediction (e.g., kmeans-pooled.csv). Uses pooled features from training if not provided.",
    )
    parser.add_argument(
        "--predict_source_csv",
        help="Original record CSV for merging output (e.g., kmeans_results.csv). Uses raw_csv if not provided.",
    )
    parser.add_argument(
        "--predict_target_name",
        default="Prediction",
        help="Prefix for prediction result column name, default Prediction",
    )
    parser.add_argument(
        "--overall_best_json",
        help="Global best model JSON selected by local LLM (e.g., overall_best_model.json). If not provided, CV best is used as global best.",
    )
    
    args = parser.parse_args()
    
    # If force_rerun is specified, ignore skip_existing
    if args.force_rerun:
        args.skip_existing = False
        print("[INFO] Force rerun mode enabled. Ignoring existing files.")

    # ---- Directory Setup ----
    ROOT = Path(__file__).resolve().parent
    MAIN_DIR = ROOT / "main_regression"
    GRID_DIR = ROOT / "grid"
    PREDICT_DIR = ROOT / "predict"

    raw_csv_path = Path(args.raw_csv).resolve()
    raw_csv = str(raw_csv_path)
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Morgan features output path: out_root/features/xxx-pooled-morgan.csv
    feat_dir = out_root / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_csv_path = feat_dir / (raw_csv_path.stem + "-pooled-morgan.csv")

    # Default grid result directories (can be overridden by args)
    rf_grid_dir = Path(args.rf_grid_dir).resolve() if args.rf_grid_dir else out_root / "rf_grid"
    mlp_grid_dir = Path(args.mlp_grid_dir).resolve() if args.mlp_grid_dir else out_root / "mlp_grid"
    svm_grid_dir = Path(args.svm_grid_dir).resolve() if args.svm_grid_dir else out_root / "svm_grid"

    rf_cv_dir = out_root / "rf_cv10"

    # ===== STEP 1: Morgan Fingerprint Generation =====
    print("\n===== [STEP 1] Generating Morgan Fingerprints =====")
    
    # Check if feature file exists
    if args.skip_existing and file_exists_and_valid(feat_csv_path, min_size=1024):  # At least 1KB
        print(f"[SKIP] Feature file exists: {feat_csv_path}")
        print("[INFO] Skipping Morgan fingerprint generation.")
    else:
        run_morgan_pooling(
            main_dir=MAIN_DIR,
            raw_csv=raw_csv,
            out_csv=str(feat_csv_path),
            polymer_cols=args.polymer_cols,
            alpha=args.alpha,
            radius=args.radius,
            nbits=args.nbits,
            target_col=args.target,
        )

    # Ensure feature file exists
    if not file_exists_and_valid(feat_csv_path):
        print(f"[ERROR] Feature file generation failed: {feat_csv_path}")
        sys.exit(1)

    # ===== STEP 2: RF Grid Search (Skip if best_params.json exists) =====
    rf_best_params_path = rf_grid_dir / "best_params.json"
    if not args.no_rf_grid:
        if args.skip_existing and json_file_valid(rf_best_params_path):
            print(f"\n===== [STEP 2] Valid RF best_params.json detected. Skipping RF Grid Search. =====")
            print(f"[SKIP] RF hyperparameters file exists: {rf_best_params_path}")
        else:
            print("\n===== [STEP 2] RF Grid Search + Final 10-fold CV =====")
            rf_grid_dir.mkdir(parents=True, exist_ok=True)
            run_rf_grid(
                grid_dir=GRID_DIR,
                feat_csv=str(feat_csv_path),
                target_col=args.target,
                save_dir=str(rf_grid_dir),
                id_cols=args.rf_id_cols,
                test_size=args.rf_test_size,
            )
    else:
        print("\n[SKIP] RF Grid Search disabled by --no_rf_grid")

    # ===== STEP 3: MLP Grid Search =====
    mlp_best_params_path = mlp_grid_dir / "mlp_grid_best.json"
    if not args.no_mlp_grid:
        if args.skip_existing and json_file_valid(mlp_best_params_path):
            print(f"\n===== [STEP 3] Valid {mlp_best_params_path.name} detected. Skipping MLP Grid Search. =====")
            print(f"[SKIP] MLP hyperparameters file exists: {mlp_best_params_path}")
        else:
            print("\n===== [STEP 3] MLP Grid Search + 10-fold CV =====")
            mlp_grid_dir.mkdir(parents=True, exist_ok=True)
            run_mlp_grid(
                grid_dir=GRID_DIR,
                feat_csv=str(feat_csv_path),
                target_col=args.target,
                save_dir=str(mlp_grid_dir),
            )
    else:
        print("\n[SKIP] MLP Grid Search disabled by --no_mlp_grid")

    # ===== STEP 4: SVM Grid Search =====
    svm_best_params_path = svm_grid_dir / "svm_grid_best.json"
    if not args.no_svm_grid:
        if args.skip_existing and json_file_valid(svm_best_params_path):
            print(f"\n===== [STEP 4] Valid {svm_best_params_path.name} detected. Skipping SVM Grid Search. =====")
            print(f"[SKIP] SVM hyperparameters file exists: {svm_best_params_path}")
        else:
            print("\n===== [STEP 4] SVM Grid Search + 10-fold CV =====")
            svm_grid_dir.mkdir(parents=True, exist_ok=True)
            run_svm_grid(
                grid_dir=GRID_DIR,
                feat_csv=str(feat_csv_path),
                target_col=args.target,
                save_dir=str(svm_grid_dir),
            )
    else:
        print("\n[SKIP] SVM Grid Search disabled by --no_svm_grid")

    # ==== Load best_params from respective grid directories ====
    rf_best_params = load_best_params(rf_grid_dir / "best_params.json")
    mlp_best_params = load_best_params(mlp_grid_dir / "mlp_grid_best.json")
    svm_best_params = load_best_params(svm_grid_dir / "svm_grid_best.json")

    # ===== STEP 5: RF baseline (CV10) + Select Best Fold =====
    rf_best_model_path = out_root / "rf_cv10" / "fold_models" / "best_model.joblib"
    
    if not args.no_rf_baseline:
        if args.skip_existing and file_exists_and_valid(rf_best_model_path):
            print(f"\n===== [STEP 5] RF best model exists. Skipping RF Baseline training. =====")
            print(f"[SKIP] RF best model exists: {rf_best_model_path}")
        else:
            print("\n===== [STEP 5] RF Baseline (10-fold CV, prioritizing grid params) =====")
            rf_cv_dir.mkdir(parents=True, exist_ok=True)
            run_rf_baseline(
                main_dir=MAIN_DIR,
                feat_csv=str(feat_csv_path),
                target_col=args.target,
                save_dir=str(rf_cv_dir),
                best_params=rf_best_params,
            )

            rf_metrics_file = rf_cv_dir / "cv10_metrics.csv"
            rf_fold_models_dir = rf_cv_dir / "fold_models"

            # Select best fold only if training just ran
            if not args.skip_existing or not file_exists_and_valid(rf_best_model_path):
                select_best_model_from_cv(
                    metrics_file=rf_metrics_file,
                    fold_models_dir=rf_fold_models_dir,
                    metric_col="R2",
                )
    else:
        print("\n[SKIP] RF Baseline disabled by --no_rf_baseline")

    # ===== STEP 6: MLP + SVM baseline (CV10) + Select Best Fold =====
    mlp_best_model_path = out_root / "runs" / "mlp" / "fold_models" / "best_model.joblib"
    svm_best_model_path = out_root / "runs" / "svm" / "fold_models" / "best_model.joblib"
    
    if not args.no_mlp_svm_baseline:
        # Check if both models exist
        mlp_exists = file_exists_and_valid(mlp_best_model_path)
        svm_exists = file_exists_and_valid(svm_best_model_path)
        
        if args.skip_existing and mlp_exists and svm_exists:
            print(f"\n===== [STEP 6] MLP and SVM best models exist. Skipping MLP+SVM Baseline training. =====")
            print(f"[SKIP] MLP best model exists: {mlp_best_model_path}")
            print(f"[SKIP] SVM best model exists: {svm_best_model_path}")
        else:
            print("\n===== [STEP 6] MLP + SVM Baseline (10-fold CV, prioritizing grid params) =====")
            runs_root = out_root / "runs"

            run_mlp_svm_baseline(
                main_dir=MAIN_DIR,
                feat_csv=str(feat_csv_path),
                target_col=args.target,
                out_root=str(runs_root),
                cv_folds=10,
                mlp_best_params=mlp_best_params,
                svm_best_params=svm_best_params,
            )

            # MLP: Select Best Fold
            mlp_dir = runs_root / "mlp"
            mlp_metrics_file = mlp_dir / "cv10_metrics.csv"
            mlp_fold_models_dir = mlp_dir / "fold_models"

            if not args.skip_existing or not file_exists_and_valid(mlp_best_model_path):
                select_best_model_from_cv(
                    metrics_file=mlp_metrics_file,
                    fold_models_dir=mlp_fold_models_dir,
                    metric_col="R2",
                )

            # SVM: Select Best Fold
            svm_dir = runs_root / "svm"
            svm_metrics_file = svm_dir / "cv10_metrics.csv"
            svm_fold_models_dir = svm_dir / "fold_models"

            if not args.skip_existing or not file_exists_and_valid(svm_best_model_path):
                select_best_model_from_cv(
                    metrics_file=svm_metrics_file,
                    fold_models_dir=svm_fold_models_dir,
                    metric_col="R2",
                )

    else:
        print("\n[SKIP] MLP+SVM Baseline disabled by --no_mlp_svm_baseline")

    # ===== Step 7: OLS Baseline (10-fold CV, No Grid) =====
    ols_best_model_path = out_root / "ols_linear" / "fold_models" / "best_model.joblib"
    
    if not args.no_ols_baseline:
        if args.skip_existing and file_exists_and_valid(ols_best_model_path):
            print(f"\n===== [STEP 7] OLS best model exists. Skipping OLS Baseline training. =====")
            print(f"[SKIP] OLS best model exists: {ols_best_model_path}")
        else:
            print("\n===== [STEP 7] OLS Baseline (10-fold CV, No Grid) =====")
            ols_dir = out_root / "ols_linear"
            ols_dir.mkdir(parents=True, exist_ok=True)

            # 1) Run OLS 10-fold
            run_ols_baseline(
                main_dir=MAIN_DIR,
                feat_csv=str(feat_csv_path),
                target_col=args.target,
                save_dir=str(ols_dir),
                n_splits=10,
                seed=42,
            )

            # 2) Select highest R2 fold from cv_results.csv, copy model.pkl -> fold_models/best_model.joblib
            if not args.skip_existing or not file_exists_and_valid(ols_best_model_path):
                select_best_ols_model(
                    ols_dir=ols_dir,
                    metric_col="R2",
                )
    else:
        print("\n[SKIP] OLS Baseline disabled by --no_ols_baseline")

    # ===== STEP 8: Predict on all available models =====
    if args.do_predict:
        print("\n===== [STEP 8] Predicting on all available models (RF / MLP / SVM / OLS) =====")
        predictions_dir = out_root / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # If high-throughput library provided, use its features & source; else use training data
        if args.predict_in_csv:
            in_csv_for_pred = str(Path(args.predict_in_csv).resolve())
        else:
            in_csv_for_pred = str(feat_csv_path)

        if args.predict_source_csv:
            source_csv_for_pred_path = Path(args.predict_source_csv).resolve()
        else:
            source_csv_for_pred_path = raw_csv_path

        pred_base_name = source_csv_for_pred_path.stem

        # Record prediction file path for each model to write to JSON later
        model_pred_paths = {}

        # ---- RF Prediction ----
        rf_model_dir = rf_cv_dir / "fold_models"
        rf_pred_csv_path = predictions_dir / f"RF_best_pred_{pred_base_name}.csv"
        
        if rf_model_dir.is_dir():
            if args.skip_existing and csv_file_valid(rf_pred_csv_path, min_rows=5):
                print(f"[SKIP] RF prediction results exist: {rf_pred_csv_path}")
                model_pred_paths["rf"] = rf_pred_csv_path
            else:
                print(f"[INFO] Predicting using RF best fold model, output file: {rf_pred_csv_path}")
                run_predict(
                    predict_dir=PREDICT_DIR,
                    in_csv=in_csv_for_pred,
                    source_csv=str(source_csv_for_pred_path),
                    out_csv=str(rf_pred_csv_path),
                    model_dir=str(rf_model_dir),
                    target_name="RF_" + args.predict_target_name,
                    id_col="row_index",
                )
                model_pred_paths["rf"] = rf_pred_csv_path
        else:
            print("[INFO] RF model directory not found, skipping RF prediction.")

        # ---- MLP Prediction ----
        mlp_model_dir = out_root / "runs" / "mlp" / "fold_models"
        mlp_pred_csv_path = predictions_dir / f"MLP_best_pred_{pred_base_name}.csv"
        
        if mlp_model_dir.is_dir():
            if args.skip_existing and csv_file_valid(mlp_pred_csv_path, min_rows=5):
                print(f"[SKIP] MLP prediction results exist: {mlp_pred_csv_path}")
                model_pred_paths["mlp"] = mlp_pred_csv_path
            else:
                print(f"[INFO] Predicting using MLP best fold model, output file: {mlp_pred_csv_path}")
                run_predict(
                    predict_dir=PREDICT_DIR,
                    in_csv=in_csv_for_pred,
                    source_csv=str(source_csv_for_pred_path),
                    out_csv=str(mlp_pred_csv_path),
                    model_dir=str(mlp_model_dir),
                    target_name="MLP_" + args.predict_target_name,
                    id_col="row_index",
                )
                model_pred_paths["mlp"] = mlp_pred_csv_path
        else:
            print("[INFO] MLP model directory not found, skipping MLP prediction.")

        # ---- SVM Prediction ----
        svm_model_dir = out_root / "runs" / "svm" / "fold_models"
        svm_pred_csv_path = predictions_dir / f"SVM_best_pred_{pred_base_name}.csv"
        
        if svm_model_dir.is_dir():
            if args.skip_existing and csv_file_valid(svm_pred_csv_path, min_rows=5):
                print(f"[SKIP] SVM prediction results exist: {svm_pred_csv_path}")
                model_pred_paths["svm"] = svm_pred_csv_path
            else:
                print(f"[INFO] Predicting using SVM best fold model, output file: {svm_pred_csv_path}")
                run_predict(
                    predict_dir=PREDICT_DIR,
                    in_csv=in_csv_for_pred,
                    source_csv=str(source_csv_for_pred_path),
                    out_csv=str(svm_pred_csv_path),
                    model_dir=str(svm_model_dir),
                    target_name="SVM_" + args.predict_target_name,
                    id_col="row_index",
                )
                model_pred_paths["svm"] = svm_pred_csv_path
        else:
            print("[INFO] SVM model directory not found, skipping SVM prediction.")

        # ---- OLS Prediction ----
        ols_model_dir = out_root / "ols_linear" / "fold_models"
        ols_pred_csv_path = predictions_dir / f"OLS_best_pred_{pred_base_name}.csv"
        
        if ols_model_dir.is_dir():
            if args.skip_existing and csv_file_valid(ols_pred_csv_path, min_rows=5):
                print(f"[SKIP] OLS prediction results exist: {ols_pred_csv_path}")
                model_pred_paths["ols"] = ols_pred_csv_path
            else:
                print(f"[INFO] Predicting using OLS best fold model, output file: {ols_pred_csv_path}")
                run_predict(
                    predict_dir=PREDICT_DIR,
                    in_csv=in_csv_for_pred,
                    source_csv=str(source_csv_for_pred_path),
                    out_csv=str(ols_pred_csv_path),
                    model_dir=str(ols_model_dir),
                    target_name="OLS_" + args.predict_target_name,
                    id_col="row_index",
                )
                model_pred_paths["ols"] = ols_pred_csv_path
        else:
            print("[INFO] OLS model directory not found, skipping OLS prediction.")

        # After prediction, regenerate candidates JSON and fill in pred_csv
        candidates = collect_model_candidates(out_root=out_root, metric_col="R2", dump_json=False)

        # Use ROOT.parent as "project root", resulting in e.g.:
        # Regression_Model/results/YoungsModulus/predictions/xxx.csv
        project_root = ROOT.parent

        for c in candidates:
            mt = c["model_type"]
            if mt in model_pred_paths:
                full_path = model_pred_paths[mt].resolve()
                try:
                    # Relative path: Regression_Model/...
                    rel_path = full_path.relative_to(project_root)
                except ValueError:
                    # Fallback to filename if not under project_root
                    rel_path = full_path.name

                # If you want it to start with /Regression_Model/..., add a slash manually
                c["pred_csv"] = "/" + str(rel_path).replace("\\", "/")
                # If you only want Regression_Model/..., use this line:
                # c["pred_csv"] = str(rel_path).replace("\\", "/")

        # Write simplified JSON for LLM
        json_path = out_root / "model_candidates_for_llm.json"
        
        # Check if JSON file already exists
        if args.skip_existing and json_file_valid(json_path):
            print(f"[SKIP] Candidate model JSON exists: {json_path}")
        else:
            simple = [
                {
                    "model_type": c["model_type"],
                    "best_fold": c["best_fold"],
                    "best_score": c["best_score"],
                    "pred_csv": c.get("pred_csv"),
                }
                for c in candidates
            ]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(simple, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Written model candidate JSON (with prediction paths): {json_path}")


    print("\n[DONE] Pipeline completed!")
    print(f"[DONE] Output root: {out_root}")
    
    # Check file existence
    print("\n" + "="*60)
    print("File Existence Check:")
    print("="*60)
    
    files_to_check = [
        ("Morgan Feature File", feat_csv_path),
        ("RF Hyperparams File", rf_best_params_path),
        ("MLP Hyperparams File", mlp_best_params_path),
        ("SVM Hyperparams File", svm_best_params_path),
        ("RF Best Model", rf_best_model_path),
        ("MLP Best Model", mlp_best_model_path),
        ("SVM Best Model", svm_best_model_path),
        ("OLS Best Model", ols_best_model_path),
    ]
    
    for name, path in files_to_check:
        exists = file_exists_and_valid(path)
        status = "✓ Exists" if exists else "✗ Missing"
        print(f"{name:20} {status:15} {path}")
    
    if args.do_predict:
        print("\nPrediction Files:")
        for model_type in ["rf", "mlp", "svm", "ols"]:
            pred_path = out_root / "predictions" / f"{model_type.upper()}_best_pred_{source_csv_for_pred_path.stem}.csv"
            exists = csv_file_valid(pred_path, min_rows=1) if pred_path.exists() else False
            status = "✓ Exists" if exists else "✗ Missing"
            print(f"{model_type.upper()} Prediction:    {status:15} {pred_path}")
    
    candidates_json = out_root / "model_candidates_for_llm.json"
    exists = json_file_valid(candidates_json) if candidates_json.exists() else False
    status = "✓ Exists" if exists else "✗ Missing"
    print(f"\nCandidate Model JSON: {status:15} {candidates_json}")
    
    print("="*60)


if __name__ == "__main__":
    main()
