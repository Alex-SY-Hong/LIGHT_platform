#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import argparse

def main():
    # ==== Argument Definition ====
    ap = argparse.ArgumentParser(description="SVM Grid Search")
    ap.add_argument("--in_csv", required=True, help="Input feature CSV file")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--save_dir", default="svm_grid", help="Output directory")
    args = ap.parse_args()

    in_csv = args.in_csv
    target = args.target
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ==== Data Loading ====
    df = pd.read_csv(in_csv)
    # Select numeric columns and drop the target column to form features X
    X = df.select_dtypes(include=[np.number]).drop(columns=[target])
    y = df[target].values

    # ==== Search Space ====
    param_grid = {
        "svm__kernel": ["rbf", "poly", "sigmoid"],
        "svm__C": [0.1, 1, 10, 50],
        "svm__epsilon": [0.05, 0.1, 0.2],
        "svm__gamma": ["scale", "auto"]
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ==== Grid Search ====
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVR())
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
        print(f"[{i:03d}/{len(ParameterGrid(param_grid))}] mean_R2 = {mean_r2:.4f}  |  params = {params}")

    # ==== Save Best Parameters ====
    results_df = pd.DataFrame(results_list)
    best_row = results_df.loc[results_df["mean_R2"].idxmax()]
    best_params = {k: best_row[k] for k in param_grid.keys()}
    best_score = best_row["mean_R2"]

    with open(save_dir/"svm_grid_best.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": best_params, "best_r2": float(best_score)}, f, indent=2, ensure_ascii=False)

    print("\nSVM Grid Search Done.")
    print(f"Best R2 = {best_score:.4f}")
    print(f"Best Params = {best_params}")

if __name__ == "__main__":
    main()