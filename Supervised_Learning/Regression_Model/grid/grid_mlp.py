#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import argparse


def main():
    # ===== Argument Definition =====
    ap = argparse.ArgumentParser(description="MLP Grid Search")
    ap.add_argument("--in_csv", required=True, help="Input feature CSV file")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--save_dir", default="mlp_grid", help="Output directory")
    args = ap.parse_args()

    in_csv = args.in_csv
    target = args.target
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ===== Data Loading =====
    df = pd.read_csv(in_csv)
    # Select numeric columns and drop the target column to form features X
    X = df.select_dtypes(include=[np.number]).drop(columns=[target])
    y = df[target].values

    # ===== Search Space =====
    param_grid = {
        "mlp__hidden_layer_sizes": [(256,128,64), (512,256,128), (128,64)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [1e-3, 1e-4],
        "mlp__learning_rate_init": [1e-3, 5e-4],
        "mlp__solver": ["adam"],
        "mlp__max_iter": [1000]
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ===== MLP Pipeline =====
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(max_iter=800, random_state=42))
    ])

    print(f"Number of parameter combinations: {len(list(ParameterGrid(param_grid)))}\n")

    results_list = []
    for i, params in enumerate(ParameterGrid(param_grid), 1):
        pipe.set_params(**params)
        scores = []
        for train_idx, valid_idx in kf.split(X, y):
            Xtr, Xva = X.iloc[train_idx], X.iloc[valid_idx]
            ytr, yva = y[train_idx], y[valid_idx]
            pipe.fit(Xtr, ytr)
            y_pred = pipe.predict(Xva)
            r2 = r2_score(yva, y_pred)
            scores.append(r2)
        mean_r2 = np.mean(scores)
        results_list.append({**params, "mean_R2": mean_r2})
        print(f"[{i:03d}/{len(ParameterGrid(param_grid))}] mean_R2 = {mean_r2:.4f} | params = {params}")

    # ===== Save Best Parameters =====
    results_df = pd.DataFrame(results_list)
    best_row = results_df.loc[results_df["mean_R2"].idxmax()]
    best_params = {k: best_row[k] for k in param_grid.keys()}
    best_score = best_row["mean_R2"]

    # Ensure all numeric types are converted to Python built-in types (for JSON serialization)
    best_params = {k: (int(v) if isinstance(v, np.int64) else v) for k, v in best_params.items()}

    with open(save_dir/"mlp_grid_best.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": best_params, "best_r2": float(best_score)}, f, indent=2, ensure_ascii=False)

    print("\nMLP Grid Search Done.")
    print(f"Best R2 = {best_score:.4f}")
    print(f"Best Params = {best_params}")

if __name__ == "__main__":
    main()