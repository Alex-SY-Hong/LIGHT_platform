#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load


def main():
    ap = argparse.ArgumentParser(description="RF model prediction and merge with original SMILES (supports regression and classification)")
    ap.add_argument("--in_csv", required=True, help="Input: pooled Morgan or composite feature file")
    ap.add_argument("--source_csv", required=True, help="Input: original SMILES file (for merging output)")
    ap.add_argument("--out_csv", required=True, help="Output: original SMILES + prediction results (+ classification probabilities)")
    ap.add_argument("--model_dir", required=True, help="Model directory (containing best_model.joblib)")
    ap.add_argument("--target_name", type=str, default="Prediction", help="Prediction column name")
    ap.add_argument("--id_col", type=str, default="row_index", help="Matching index column (default: row_index)")
    args = ap.parse_args()

    # === Path Check ===
    in_csv, src_csv, out_csv, model_dir = map(Path, [args.in_csv, args.source_csv, args.out_csv, args.model_dir])
    model_path = model_dir / "best_model.joblib"
    if not in_csv.exists():
        raise FileNotFoundError(f"Input feature file does not exist: {in_csv}")
    if not src_csv.exists():
        raise FileNotFoundError(f"Original SMILES file does not exist: {src_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # === Read Data ===
    df_feat = pd.read_csv(in_csv)
    df_src = pd.read_csv(src_csv)
    print(f"[INFO] Feature file: {df_feat.shape}, SMILES file: {df_src.shape}")

    # === Check and fill row_index ===
    if args.id_col not in df_feat.columns:
        df_feat[args.id_col] = np.arange(len(df_feat))
    if args.id_col not in df_src.columns:
        df_src[args.id_col] = np.arange(len(df_src))

    # === Load Model ===
    model_loaded = load(model_path)
    model = model_loaded
    if isinstance(model_loaded, dict):
        for k in ("model", "rf", "estimator", "pipe"):
            if k in model_loaded and hasattr(model_loaded[k], "predict"):
                model = model_loaded[k]
                break

    # === Auto-identify feature columns ===
    FEATURE_PATTERNS = ("fp_", "morgan_", "ecfp_", "desc_", "frag_", "idx_", "pair_")
    feature_cols = [
        c for c in df_feat.columns
        if any(c.startswith(p) for p in FEATURE_PATTERNS)
        and np.issubdtype(df_feat[c].dtype, np.number)
    ]
    if not feature_cols:
        raise ValueError("No matching feature columns detected. Please check if column prefixes contain fp_/desc_/frag_/idx_/pair_.")

    X = df_feat[feature_cols].astype(np.float32).values
    nfeat_model = getattr(model, "n_features_in_", None)
    print(f"[INFO] Automatically identified {len(feature_cols)} feature columns. Example of first 10: {feature_cols[:10]}")

    # === Auto Dimension Alignment ===
    if nfeat_model and X.shape[1] != nfeat_model:
        diff = nfeat_model - X.shape[1]
        if diff > 0:
            X = np.hstack([X, np.zeros((X.shape[0], diff), dtype=np.float32)])
            print(f"[AUTO] Padding zero columns {diff} -> Input dimension {X.shape[1]}")
        else:
            X = X[:, :nfeat_model]
            print(f"[AUTO] Truncating {-diff} columns -> Input dimension {X.shape[1]}")

    # === Model Prediction (Auto-detect Classification/Regression) ===
    if hasattr(model, "predict_proba"):  # Classification model
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        df_feat[args.target_name] = y_pred
        for i in range(y_prob.shape[1]):
            df_feat[f"{args.target_name}_prob_class{i}"] = y_prob[:, i]
        print(f"[OK] Classification prediction done. Total: {len(y_pred)}, Classes={y_prob.shape[1]}")
    else:  # Regression model
        y_pred = model.predict(X)
        df_feat[args.target_name] = y_pred
        print(f"[OK] Regression prediction done. Total: {len(y_pred)}.")

    # === Merge back to SMILES table ===
    merged = pd.merge(
        df_src,
        df_feat[[args.id_col] + [c for c in df_feat.columns if c.startswith(args.target_name)]],
        on=args.id_col,
        how="left"
    )
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved prediction results -> {out_csv}")
    print(f"[INFO] Output columns: {list(merged.columns)}")
    print(f"[INFO] Feature dimension used: {X.shape[1]}  Model expected: {nfeat_model}")


if __name__ == "__main__":
    main()