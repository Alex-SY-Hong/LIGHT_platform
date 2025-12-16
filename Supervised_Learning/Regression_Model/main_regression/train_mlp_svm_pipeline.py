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
    # 
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

    # Hyperparameters: copied exactly from your original script
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
    # 
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

    # Hyperparameters: copied exactly from your original script
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

    # Your original script used `--no_perm`, meaning permutation test is skipped by default
    if not use_perm:
        cmd.append("--no_perm")

    print("[SVM CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[SVM] done.")


def main():
    # 
    parser = argparse.ArgumentParser(
        description="MLP + SVM baseline one-click pipeline (wraps baseline_mlp_svm.py)"
    )

    parser.add_argument("--in_csv", required=True, help="Input feature CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument(
        "--out_root",
        required=True,
        help="Output root directory; 'mlp/' and 'svm/' subdirectories will be created under this.",
    )

    parser.add_argument(
        "--cv10",
        type=int,
        default=1,
        help="Whether to use 10-fold CV (1/0), default 1",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=10,
        help="Number of CV folds, default 10 (consistent with your original script)",
    )

    # ===== MLP Hyperparameters (Defaults come entirely from your original Bash script) =====
    parser.add_argument(
        "--mlp_hidden",
        default="512,256,128",
        help='MLP hidden layer structure, default "512,256,128"',
    )
    parser.add_argument(
        "--mlp_activation",
        default="tanh",
        help="MLP activation function, default tanh",
    )
    parser.add_argument(
        "--mlp_alpha",
        type=float,
        default=1e-3,
        help="MLP L2 regularization, default 1e-3",
    )
    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=1e-3,
        help="MLP learning rate, default 1e-3",
    )
    parser.add_argument(
        "--mlp_max_iter",
        type=int,
        default=1000,
        help="MLP max iterations, default 1000",
    )
    parser.add_argument(
        "--no_mlp_early_stop",
        action="store_true",
        help="Disable MLP early stopping (early stopping is enabled by default)",
    )

    # ===== SVM Hyperparameters (Defaults come entirely from your original Bash script) =====
    parser.add_argument(
        "--svm_kernel",
        default="rbf",
        help="SVM kernel type, default rbf",
    )
    parser.add_argument(
        "--svm_C",
        type=float,
        default=10.0,
        help="SVM C parameter, default 10.0",
    )
    parser.add_argument(
        "--svm_epsilon",
        type=float,
        default=0.2,
        help="SVM epsilon, default 0.2",
    )
    parser.add_argument(
        "--svm_gamma",
        default="auto",
        help="SVM gamma, default auto",
    )
    parser.add_argument(
        "--svm_use_perm",
        action="store_true",
        help="Whether to enable permutation test (default False, corresponds to --no_perm in the original script)",
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

    # Path to baseline_mlp_svm.py: assumed to be in the same directory as this script
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