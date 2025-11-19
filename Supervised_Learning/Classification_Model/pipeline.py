#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import hashlib


def build_cls_csv(in_csv: str,
                  src_col: str,
                  class_col: str,
                  threshold: float,
                  out_csv: str) -> None:
    df = pd.read_csv(in_csv, low_memory=False)
    if src_col not in df.columns:
        raise SystemExit(f"[ERROR] 源列 '{src_col}' 不在 {in_csv} 中")

    x = pd.to_numeric(df[src_col], errors="coerce")
    df[class_col] = (x >= threshold).astype("Int64")
    df = df.dropna(subset=[class_col]).copy()
    df[class_col] = df[class_col].astype(int)

    # 物理删除连续列，防止信息泄漏
    df = df.drop(columns=[src_col], errors="ignore")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(
        f"[OK] 写出: {out_csv} (rows={len(df)}, cols={df.shape[1]})；已删除列: {src_col}"
    )


def add_fp_hash(csv_path: str, class_col: str) -> None:
    df = pd.read_csv(csv_path, low_memory=False)

    feat = df.select_dtypes(include=[np.number]).columns.tolist()
    feat = [c for c in feat if c != class_col]
    if not feat:
        raise SystemExit("[ERROR] 没有数值特征列可用于哈希分组")

    n = len(df)
    n_unique = df[feat].drop_duplicates().shape[0]
    print(
        f"[INFO] 指纹唯一行数={n_unique}/{n}，重复占比={100*(n-n_unique)/n:.2f}%"
    )

    def rhash(row):
        arr = row.values.astype(np.float32, copy=False)
        return hashlib.md5(arr.tobytes()).hexdigest()

    df["_fp_hash"] = df[feat].apply(rhash, axis=1)
    print("[INFO] _fp_hash unique groups:", df["_fp_hash"].nunique())
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def choose_group_col(csv_path: str, pref_group_col: str) -> str:
    """完全复刻原 Bash 中的 group 选择逻辑。"""
    df = pd.read_csv(csv_path, low_memory=False, dtype="string")

    def good_group(col: str) -> bool:
        if col not in df.columns:
            return False
        nunq = df[col].nunique(dropna=True)
        n = len(df)
        if nunq < 2:
            return False
        avg_sz = n / max(nunq, 1)
        uniq_ratio = nunq / max(n, 1)
        # 只有当平均组大小>=2 且 唯一组比例<=0.9 时才认为可用
        return (avg_sz >= 2.0) and (uniq_ratio <= 0.9)

    if good_group(pref_group_col):
        chosen = pref_group_col
    elif "_fp_hash" in df.columns and good_group("_fp_hash"):
        chosen = "_fp_hash"
    else:
        chosen = ""

    if chosen:
        print(f"[INFO] using group_col: {chosen}")
    else:
        print("[INFO] group nearly singleton -> fallback to StratifiedKFold")

    return chosen


def run_train_rf(
    cls_csv: str,
    class_col: str,
    threshold: float,
    use_cv10: bool,
    chosen_group: str,
    decision_thr: str,
    seed: int,
    save_fold_models: bool,
    base_save_root: str,
) -> None:


    # 公共参数
    common = [
        "--in_csv",
        cls_csv,
        "--target",
        class_col,
        "--task",
        "cls",
        "--seed",
        str(seed),
        "--id_cols",
        "SampleID,RecipeID",
        "--save_train_pred",
        "--rf_n_estimators",
        "1000",
        "--rf_max_depth",
        "14",
        "--rf_max_features",
        "0.2",
        "--rf_min_samples_leaf",
        "0.02",
        "--rf_min_samples_split",
        "0.05",
        "--rf_class_weight",
        "balanced_subsample",
    ]

    if decision_thr != "":
        common += ["--decision_threshold", str(decision_thr)]

    # 生成保存路径前缀（等价于原脚本）
    base_root = Path(base_save_root)

    train_rf_py = Path(__file__).with_name("train_rf.py")
    py_exe = sys.executable

    if use_cv10:
        if chosen_group:
            save_dir = base_root / f"rf_cls_cv10_group_t{int(threshold)}"
            cmd = [
                py_exe,
                str(train_rf_py),
                *common,
                "--group_col",
                chosen_group,
                "--cv10",
            ]
        else:
            save_dir = base_root / f"rf_cls_cv10_t{int(threshold)}"
            cmd = [py_exe, str(train_rf_py), *common, "--cv10"]

        if save_fold_models:
            cmd.append("--save_fold_models")

        cmd += [
            "--save_dir",
            str(save_dir),
            "--oof_path",
            str(save_dir / "cv10_oof.csv"),
        ]
    else:
        if chosen_group:
            save_dir = base_root / f"rf_cls_singlesplit_group_t{int(threshold)}"
            cmd = [
                py_exe,
                str(train_rf_py),
                *common,
                "--group_col",
                chosen_group,
                "--n_splits",
                "5",
                "--fold_idx",
                "0",
                "--save_dir",
                str(save_dir),
            ]
        else:
            save_dir = base_root / f"rf_cls_singlesplit_t{int(threshold)}"
            cmd = [
                py_exe,
                str(train_rf_py),
                *common,
                "--save_dir",
                str(save_dir),
            ]

    save_dir.mkdir(parents=True, exist_ok=True)

    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print(
        f"[DONE] classification (bin-thr={threshold}, cv10={use_cv10}, "
        f"decision-thr={decision_thr if decision_thr != '' else 'default-argmax(0.5)'})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="RF 分类一键管线（等价原 baseline_cls.sh）"
    )
    parser.add_argument(
        "--in_csv",
        required=True,
        help="原始回归表（含连续目标列）的 CSV 路径",
    )
    parser.add_argument(
        "--src_col",
        required=True,
        help="连续目标列名（例如 'Swelling Ratio (times)'）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="二分类阈值，例如 9",
    )
    parser.add_argument(
        "--class_col",
        default="y_class",
        help="生成的分类标签列名（默认 y_class）",
    )
    parser.add_argument(
        "--use_cv10",
        type=int,
        default=1,
        help="1=启用10折CV；0=单次划分（默认 1）",
    )
    parser.add_argument(
        "--pref_group_col",
        default="RecipeID",
        help="优先使用的分组列（默认 RecipeID）",
    )
    parser.add_argument(
        "--decision_threshold",
        default="",
        help="分类决策阈值；留空则使用默认 argmax(0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="随机种子（默认 42）"
    )
    parser.add_argument(
        "--save_fold_models",
        type=int,
        default=1,
        help="CV 时是否保存每折模型 (1/0，默认 1)",
    )
    parser.add_argument(
        "--base_save_root",
        default="Path",
        help="保存目录前缀",
    )

    args = parser.parse_args()

    in_csv = args.in_csv
    src_col = args.src_col
    thr = args.threshold
    class_col = args.class_col
    use_cv10 = bool(args.use_cv10)
    pref_group_col = args.pref_group_col
    decision_thr = args.decision_threshold
    seed = args.seed
    save_fold_models = bool(args.save_fold_models)
    base_save_root = args.base_save_root

    # 1) 连续列 -> y_class + 删除连续列
    cls_csv = Path(in_csv).with_name(
        Path(in_csv).stem + f"_CLS{int(thr)}.csv"
    ).as_posix()
    print(f"[STEP] build {class_col} from '{src_col}' with threshold {thr}")
    build_cls_csv(in_csv, src_col, class_col, thr, cls_csv)

    # 2) 生成 _fp_hash
    add_fp_hash(cls_csv, class_col)

    # 3) 选择分组列
    chosen_group = choose_group_col(cls_csv, pref_group_col)

    # 4) 调用 train_rf.py 训练
    run_train_rf(
        cls_csv=cls_csv,
        class_col=class_col,
        threshold=thr,
        use_cv10=use_cv10,
        chosen_group=chosen_group,
        decision_thr=decision_thr,
        seed=seed,
        save_fold_models=save_fold_models,
        base_save_root=base_save_root,
    )


if __name__ == "__main__":
    main()
