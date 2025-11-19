#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ===== 用户配置 =====
CSV_PATH = "Path/SMILES-pooled-morgan.csv"
TARGET_COL = "Young's Modulus (kPa) log10"
RANDOM_STATE = 42
N_SPLITS = 10
OUT_DIR = "Path/runs/linear_ols"

# -------------------------------------------------------------------------
# Step 1. 数据读取函数（支持ASCII长行 + 自动fp列名）
# -------------------------------------------------------------------------
def load_large_feature_csv(path, target_col, fingerprint_prefix="fp_", skip_rows=1):

    clean_path = path.replace(".csv", "_clean.csv")

    if os.path.exists(clean_path):
        print(f"Detected cached clean file: {clean_path}")
        return pd.read_csv(clean_path)

    print(f"Parsing raw ASCII file (skip first {skip_rows} line(s)) ...")
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i < skip_rows:
                continue
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"[\[\]]", "", line)
            parts = re.split(r"[\s,;]+", line)
            try:
                row = [float(x) for x in parts if x != ""]
                data.append(row)
            except ValueError:
                continue

    arr = np.array(data)
    print(f"Parsed manually: shape={arr.shape}")
    n_rows, n_cols = arr.shape

    # 自动修正列数
    if n_cols > 1025:
        print(f"Detected {n_cols} columns (>1025). Truncating to 1025.")
        arr = arr[:, :1025]
        n_cols = 1025
    elif n_cols < 2:
        raise ValueError("数据列数过少，无法区分特征与目标值。")

    X = arr[:, :-1]
    y = arr[:, -1]
    fp_cols = [f"{fingerprint_prefix}{i}" for i in range(X.shape[1])]
    cols = fp_cols + [target_col]

    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df.to_csv(clean_path, index=False, float_format="%.6f")
    print(f"Clean structured CSV saved: {clean_path}")
    return df


def read_csv_robust(path: str, target_col: str) -> pd.DataFrame:

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(200)
        if "," not in head and ";" not in head:
            print("Detected long-line ASCII file. Switching to manual parser ...")
            return load_large_feature_csv(path, target_col)
    except Exception:
        pass

    encodings = ["utf-8", "utf-8-sig", "gb18030", "latin1"]
    seps = [",", "\t", ";"]
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False)
                if df.shape[1] >= 2:
                    print(f"Successfully loaded CSV: encoding={enc}, sep='{sep}', shape={df.shape}")
                    return df
            except Exception:
                continue
    raise RuntimeError("无法读取CSV文件，请检查文件编码或分隔符。")

# -------------------------------------------------------------------------
# Step 2. 加载数据
# -------------------------------------------------------------------------
print(f"Loading data from: {CSV_PATH}")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"文件不存在: {CSV_PATH}")

df = read_csv_robust(CSV_PATH, TARGET_COL)
if TARGET_COL not in df.columns:
    raise KeyError(f"未找到目标列: {TARGET_COL}")

df = df.dropna(subset=[TARGET_COL])
y = df[TARGET_COL].values
X = df.drop(columns=[TARGET_COL]).values
print(f"Final data shape: X={X.shape}, y={y.shape}")

# -------------------------------------------------------------------------
# Step 3. 十折交叉验证
# -------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
results = []

print(f"\nStarting {N_SPLITS}-Fold Cross Validation ...")

for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
    print(f"\nFold {fold_idx}/{N_SPLITS} start...")

    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)

    # 训练集预测
    y_train_pred = model.predict(X_train)
    df_train_pred = pd.DataFrame({
        "idx": train_idx,
        "y_pred": y_train_pred,
        "y_true": y_train,
        "residual": y_train_pred - y_train
    })

    # 验证集预测
    y_valid_pred = model.predict(X_valid)
    df_valid_pred = pd.DataFrame({
        "idx": valid_idx,
        "y_pred": y_valid_pred,
        "y_true": y_valid,
        "residual": y_valid_pred - y_valid
    })

    mse = mean_squared_error(y_valid, y_valid_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_valid, y_valid_pred)
    r2 = r2_score(y_valid, y_valid_pred)
    print(f"Fold {fold_idx} → R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    results.append({"Fold": fold_idx, "R2": r2, "RMSE": rmse, "MAE": mae})

    fold_dir = os.path.join(OUT_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    df_train_pred.to_csv(os.path.join(fold_dir, f"fold_{fold_idx}_train.csv"), index=False, float_format="%.6f")
    df_valid_pred.to_csv(os.path.join(fold_dir, f"fold_{fold_idx}_valid.csv"), index=False, float_format="%.6f")
    joblib.dump(model, os.path.join(fold_dir, "model.pkl"))
    print(f"Saved: {fold_dir}")

# -------------------------------------------------------------------------
# Step 4. 汇总结果保存
# -------------------------------------------------------------------------
cv_results = pd.DataFrame(results)
cv_results.to_csv(os.path.join(OUT_DIR, "cv_results.csv"), index=False, float_format="%.6f")

# -------------------------------------------------------------------------
# Step 5. 输出平均性能
# -------------------------------------------------------------------------
mean_r2 = cv_results["R2"].mean()
std_r2 = cv_results["R2"].std()
mean_rmse = cv_results["RMSE"].mean()
std_rmse = cv_results["RMSE"].std()
mean_mae = cv_results["MAE"].mean()
std_mae = cv_results["MAE"].std()

summary_path = os.path.join(OUT_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write("10-Fold Cross Validation Summary\n")
    f.write(f"R²    = {mean_r2:.4f} ± {std_r2:.4f}\n")
    f.write(f"RMSE  = {mean_rmse:.4f} ± {std_rmse:.4f}\n")
    f.write(f"MAE   = {mean_mae:.4f} ± {std_mae:.4f}\n")

print("\n10-Fold Cross Validation Summary:")
print(f"R²    = {mean_r2:.4f} ± {std_r2:.4f}")
print(f"RMSE  = {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"MAE   = {mean_mae:.4f} ± {std_mae:.4f}")
print(f"Results saved to: {OUT_DIR}")
