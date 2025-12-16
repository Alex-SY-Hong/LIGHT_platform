#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
import warnings
from sklearn.exceptions import DataConversionWarning

# Filter specific warnings
warnings.filterwarnings("ignore", category=UserWarning, 
    message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=DataConversionWarning)


def main():
    # Set up argument parser with description: RF model prediction and merge with original SMILES (supports regression and classification)
    ap = argparse.ArgumentParser(description="RF Model Prediction and Merge with Original SMILES (Supports Regression/Classification)")
    ap.add_argument("--in_csv", required=True, help="Input: Pooled Morgan or composite feature file")
    ap.add_argument("--source_csv", required=True, help="Input: Original SMILES file (for merging output)")
    ap.add_argument("--out_csv", required=True, help="Output: Original SMILES + Prediction Result (+ Classification Probability)")
    ap.add_argument("--model_dir", required=True, help="Model directory (must contain best_model.joblib)")
    ap.add_argument("--target_name", type=str, default="Prediction", help="Name of the prediction column")
    ap.add_argument("--id_col", type=str, default="row_index", help="Matching index column (default: row_index)")
    args = ap.parse_args()

    # === Path check ===
    in_csv, src_csv, out_csv, model_dir = map(Path, [args.in_csv, args.source_csv, args.out_csv, args.model_dir])
    model_path = model_dir / "best_model.joblib"
    if not in_csv.exists():
        raise FileNotFoundError(f"Input feature file not found: {in_csv}")
    if not src_csv.exists():
        raise FileNotFoundError(f"Original SMILES file not found: {src_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # === Read data ===
    df_feat = pd.read_csv(in_csv)
    df_src = pd.read_csv(src_csv)
    print(f"[INFO] Feature file: {df_feat.shape}, SMILES file: {df_src.shape}")

    # === Check and add row_index ===
    if args.id_col not in df_feat.columns:
        df_feat[args.id_col] = np.arange(len(df_feat))
    if args.id_col not in df_src.columns:
        df_src[args.id_col] = np.arange(len(df_src))

    # === Load model ===
    model_loaded = load(model_path)
    model = model_loaded
    # Handle cases where the model is wrapped in a dictionary (e.g., pipeline or estimator)
    if isinstance(model_loaded, dict):
        for k in ("model", "rf", "estimator", "pipe"):
            if k in model_loaded and hasattr(model_loaded[k], "predict"):
                model = model_loaded[k]
                break

    # === Automatic feature column identification ===
    # Patterns for auto-detected feature columns (e.g., fingerprints, descriptors)
    FEATURE_PATTERNS = ("fp_", "morgan_", "ecfp_", "desc_", "frag_", "idx_", "pair_")
    feature_cols = [
        c for c in df_feat.columns
        if any(c.startswith(p) for p in FEATURE_PATTERNS)
        and np.issubdtype(df_feat[c].dtype, np.number)
    ]
    
    # If feature columns are not found, try using all numeric columns
    if not feature_cols:
        # Get all numeric columns
        numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude potential ID columns
        feature_cols = [col for col in numeric_cols if col != args.id_col]
        if not feature_cols:
            raise ValueError("No numeric feature columns detected. Please check the input file.")
        print(f"[INFO] Using all numeric columns as features, total {len(feature_cols)} columns")
    else:
        print(f"[INFO] Auto-detected {len(feature_cols)} feature columns, first 10 examples: {feature_cols[:10]}")

    # Extract feature columns, keeping as DataFrame to preserve column names
    X = df_feat[feature_cols].astype(np.float32)
    
    # Check if the model has the feature_names_in_ attribute
    if hasattr(model, 'feature_names_in_'):
        model_feature_names = list(model.feature_names_in_)
        print(f"[INFO] Model expects {len(model_feature_names)} feature columns")
        
        # Check if feature dimensions match
        if X.shape[1] != len(model_feature_names):
            print(f"[WARN] Feature dimension mismatch: Data {X.shape[1]} vs Model {len(model_feature_names)}")
            
            # Attempt to fix: check for missing features
            missing_features = set(model_feature_names) - set(feature_cols)
            extra_features = set(feature_cols) - set(model_feature_names)
            
            if missing_features:
                print(f"[WARN] Missing feature columns ({len(missing_features)}): {list(missing_features)[:5]}...")
            if extra_features:
                print(f"[WARN] Extra feature columns ({len(extra_features)}): {list(extra_features)[:5]}...")
            
            # Attempt to reorder feature columns
            try:
                # Only keep features expected by the model that are available in the data
                available_features = [col for col in model_feature_names if col in feature_cols]
                if available_features:
                    X = X[available_features]
                    print(f"[INFO] Reordered feature columns, using {len(available_features)} available features")
                    
                    # Check if padding with zeros is necessary
                    if len(available_features) < len(model_feature_names):
                        missing_count = len(model_feature_names) - len(available_features)
                        print(f"[WARN] Insufficient feature columns, padding with {missing_count} zero columns")
                        # Add zero columns for the missing features
                        for missing_name in list(missing_features):
                            # The safest way is to insert the missing column name and set to 0.0
                            # Note: The following logic assumes the missing columns are appended at the end
                            # or that the model's feature_names_in_ dictates the order.
                            X[missing_name] = 0.0
                            
                        # Re-sort to match the model's expected order
                        X = X[model_feature_names]
                    # If we have too many columns in X, they were already dropped by slicing with available_features
                else:
                    print("[ERROR] No available feature columns match the model's expected features")
            except Exception as e:
                print(f"[ERROR] Feature column processing failed: {e}")
    else:
        print("[INFO] Model does not have stored feature names information")
        nfeat_model = getattr(model, "n_features_in_", None)
        if nfeat_model and X.shape[1] != nfeat_model:
            print(f"[WARN] Feature dimension mismatch: Data {X.shape[1]} vs Model {nfeat_model}")
            # Simple dimension alignment
            if nfeat_model > X.shape[1]:
                diff = nfeat_model - X.shape[1]
                for i in range(diff):
                    X[f'extra_{i}'] = 0.0
                print(f"[AUTO] Padded {diff} zero columns -> Input dimension {X.shape[1]}")
            else:
                X = X.iloc[:, :nfeat_model]
                print(f"[AUTO] Truncated {X.shape[1] - nfeat_model} columns -> Input dimension {X.shape[1]}")

    # === Model Prediction (Auto-detect classification/regression) ===
    print(f"[INFO] Starting prediction, feature dimension: {X.shape}")
    
    if hasattr(model, "predict_proba"):  # Classification Model
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        df_feat[args.target_name] = y_pred
        for i in range(y_prob.shape[1]):
            # Use class indices for probability column names
            # Note: For explicit class labels, model.classes_ would be used, but this script sticks to indices.
            df_feat[f"{args.target_name}_prob_class{i}"] = y_prob[:, i]
        print(f"[OK] Classification prediction complete, total {len(y_pred)} samples, num_classes={y_prob.shape[1]}")
    else:  # Regression Model
        y_pred = model.predict(X)
        df_feat[args.target_name] = y_pred
        print(f"[OK] Regression prediction complete, total {len(y_pred)} samples.")

    # === Merge back with SMILES table ===
    # Determine columns to merge
    merge_cols = [args.id_col, args.target_name]
    # Add probability columns (if they exist)
    prob_cols = [col for col in df_feat.columns if col.startswith(f"{args.target_name}_prob_class")]
    merge_cols.extend(prob_cols)
    
    merged = pd.merge(
        df_src,
        df_feat[merge_cols],
        on=args.id_col,
        how="left"
    )
    # Use utf-8-sig for better compatibility with Excel (BOM)
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] Prediction results saved to -> {out_csv}")
    print(f"[INFO] Output columns: {list(merged.columns)}")
    print(f"[INFO] Actual feature dimension used: {X.shape[1]}")


if __name__ == "__main__":
    main()