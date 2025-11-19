#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path


def run_mlp(
    script_path: Path,
    in_csv: str,
    target: str,
    save_dir: str,
    cv10: bool,
    cv_folds: int,
    mlp_hidden: str,
    mlp_activation: str,
    mlp_alpha: float,
    mlp_lr: float,
    mlp_max_iter: int,
    mlp_early_stop: bool,
):
    cmd = [
        sys.executable,
        str(script_path),
        "--in_csv",
        in_csv,
        "--target",
        target,
        "--model",
        "mlp",
        "--save_dir",
        save_dir,
        "--save_train_pred",
    ]

    if cv10:
        cmd += ["--cv10", "--cv_folds", str(cv_folds)]

    # 超参数：完全照搬你原来的脚本
    cmd += [
        "--mlp_hidden",
        mlp_hidden,
        "--mlp_activation",
        mlp_activation,
        "--mlp_alpha",
        str(mlp_alpha),
        "--mlp_lr",
        str(mlp_lr),
        "--mlp_max_iter",
        str(mlp_max_iter),
    ]

    if mlp_early_stop:
        cmd.append("--mlp_early_stop")

    print("[MLP CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[MLP] done.")


def run_svm(
    script_path: Path,
    in_csv: str,
    target: str,
    save_dir: str,
    cv10: bool,
    cv_folds: int,
    svm_kernel: str,
    svm_C: float,
    svm_epsilon: float,
    svm_gamma: str,
    use_perm: bool,
):
    cmd = [
        sys.executable,
        str(script_path),
        "--in_csv",
        in_csv,
        "--target",
        target,
        "--model",
        "svm",
        "--save_dir",
        save_dir,
        "--save_train_pred",
    ]

    if cv10:
        cmd += ["--cv10", "--cv_folds", str(cv_folds)]

    # 超参数：完全照搬你原来的脚本
    cmd += [
        "--svm_kernel",
        svm_kernel,
        "--svm_C",
        str(svm_C),
        "--svm_epsilon",
        str(svm_epsilon),
        "--svm_gamma",
        svm_gamma,
    ]

    # 你原来脚本里写的是 `--no_perm`，即默认不跑 permutation test
    if not use_perm:
        cmd.append("--no_perm")

    print("[SVM CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[SVM] done.")


def main():
    parser = argparse.ArgumentParser(
        description="MLP + SVM baseline 一键管线（封装 baseline_mlp_svm.py）"
    )

    parser.add_argument("--in_csv", required=True, help="输入特征 CSV")
    parser.add_argument("--target", required=True, help="目标列名")
    parser.add_argument(
        "--out_root",
        required=True,
        help="输出根目录，会在下面创建 mlp/ 和 svm/ 子目录",
    )

    parser.add_argument(
        "--cv10",
        type=int,
        default=1,
        help="是否使用 10 折CV (1/0)，默认 1",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=10,
        help="CV 折数，默认 10（与你原脚本一致）",
    )

    # ===== MLP 超参数（默认值完全来自你原来的 Bash）=====
    parser.add_argument(
        "--mlp_hidden",
        default="512,256,128",
        help='MLP 隐藏层结构，默认 "512,256,128"',
    )
    parser.add_argument(
        "--mlp_activation",
        default="tanh",
        help="MLP 激活函数，默认 tanh",
    )
    parser.add_argument(
        "--mlp_alpha",
        type=float,
        default=1e-3,
        help="MLP L2 正则，默认 1e-3",
    )
    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=1e-3,
        help="MLP 学习率，默认 1e-3",
    )
    parser.add_argument(
        "--mlp_max_iter",
        type=int,
        default=1000,
        help="MLP 最大迭代轮次，默认 1000",
    )
    parser.add_argument(
        "--no_mlp_early_stop",
        action="store_true",
        help="关闭 MLP 提前停止（默认是开启早停）",
    )

    # ===== SVM 超参数（默认值完全来自你原来的 Bash）=====
    parser.add_argument(
        "--svm_kernel",
        default="rbf",
        help="SVM kernel 类型，默认 rbf",
    )
    parser.add_argument(
        "--svm_C",
        type=float,
        default=10.0,
        help="SVM C 参数，默认 10.0",
    )
    parser.add_argument(
        "--svm_epsilon",
        type=float,
        default=0.2,
        help="SVM epsilon，默认 0.2",
    )
    parser.add_argument(
        "--svm_gamma",
        default="auto",
        help="SVM gamma，默认 auto",
    )
    parser.add_argument(
        "--svm_use_perm",
        action="store_true",
        help="是否启用 permutation test（默认 False，对应原脚本中的 --no_perm）",
    )

    args = parser.parse_args()

    in_csv = args.in_csv
    target = args.target
    out_root = Path(args.out_root).resolve()
    cv10 = bool(args.cv10)
    cv_folds = args.cv_folds

    out_mlp = out_root / "mlp"
    out_svm = out_root / "svm"
    out_mlp.mkdir(parents=True, exist_ok=True)
    out_svm.mkdir(parents=True, exist_ok=True)

    # baseline_mlp_svm.py 路径：假设和本脚本在同一目录
    script_path = Path(__file__).with_name("baseline_mlp_svm.py")

    print(f"[INFO] in_csv   = {in_csv}")
    print(f"[INFO] target   = {target}")
    print(f"[INFO] out_root = {out_root}")
    print(f"[INFO] cv10={cv10}, cv_folds={cv_folds}")
    print(f"[INFO] script   = {script_path}")

    # ===== 1) MLP =====
    print("[INFO] Running MLP ...")
    run_mlp(
        script_path=script_path,
        in_csv=in_csv,
        target=target,
        save_dir=str(out_mlp),
        cv10=cv10,
        cv_folds=cv_folds,
        mlp_hidden=args.mlp_hidden,
        mlp_activation=args.mlp_activation,
        mlp_alpha=args.mlp_alpha,
        mlp_lr=args.mlp_lr,
        mlp_max_iter=args.mlp_max_iter,
        mlp_early_stop=not args.no_mlp_early_stop,
    )

    # ===== 2) SVM =====
    print("[INFO] Running SVM ...")
    run_svm(
        script_path=script_path,
        in_csv=in_csv,
        target=target,
        save_dir=str(out_svm),
        cv10=cv10,
        cv_folds=cv_folds,
        svm_kernel=args.svm_kernel,
        svm_C=args.svm_C,
        svm_epsilon=args.svm_epsilon,
        svm_gamma=args.svm_gamma,
        use_perm=args.svm_use_perm,
    )

    print("[INFO] All done.")


if __name__ == "__main__":
    main()
