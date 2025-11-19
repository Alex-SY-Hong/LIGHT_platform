#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams.update({
    "axes.labelsize": 20,      # 轴标签
    "xtick.labelsize": 18,     # X刻度
    "ytick.labelsize": 18,     # Y刻度
    "legend.fontsize": 15,     # 图例
    "axes.unicode_minus": False,
    "font.sans-serif": ["DejaVu Sans", "Arial", "Microsoft YaHei", "SimHei"]  # 中文兼容
})

# --------- 小工具 ----------
def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _parse_color(s: Optional[str], default: Tuple[float,float,float]):
    if not s: return default
    s = s.strip()
    if s.startswith("#"):
        s = s.lstrip("#")
        r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
        return (r/255, g/255, b/255)
    if "," in s:
        r,g,b = [float(x) for x in s.split(",")]
        return (r/255, g/255, b/255)
    # 颜色名交给 matplotlib 处理（返回字符串）
    return s

def _read_csv(p: Optional[str]) -> Optional[pd.DataFrame]:
    if not p: return None
    pth = Path(p)
    if not pth.is_file(): return None
    return pd.read_csv(pth)

def _auto_prob_col(df: pd.DataFrame, p_col: Optional[str]) -> str:
    if p_col and p_col in df.columns: return p_col
    for c in ["proba_pos", "prob_class1", "prob_1", "pred_proba", "prob", "p1"]:
        if c in df.columns: return c
    # 简单兜底：找 [0,1] 的数值列
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            vals = s.dropna()
            if len(vals) >= 10 and vals.min() >= -0.01 and vals.max() <= 1.01:
                return c
    raise ValueError("未找到概率列，请用 --p_col 指定，例如 proba_pos")

def _prep_y_p(df: pd.DataFrame, y_col: str, p_col: str):
    if y_col not in df.columns: raise ValueError(f"缺少标签列 {y_col}")
    if p_col not in df.columns: raise ValueError(f"缺少概率列 {p_col}")
    d = df[[y_col, p_col]].dropna()
    y = d[y_col].values
    p = d[p_col].astype(float).clip(0,1).values
    uniq = np.unique(y)
    if len(uniq) != 2: raise ValueError(f"{y_col} 不是二分类（唯一值={uniq}）")
    pos = np.sort(uniq)[1]  # 较大者为正类
    y = (y == pos).astype(int)
    return y, p

def _plot_roc_ax(ax, y, p, color, label_prefix=None, lw=2.0, fill=False, alpha=0.12):
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    auc = roc_auc_score(y, p)
    label = f"AUC={auc:.3f}" if not label_prefix else f"{label_prefix} (AUC={auc:.3f})"
    ax.plot(fpr, tpr, color=color, linewidth=lw, label=label)
    if fill:
        # 面积相对 x 轴（0）填充
        ax.fill_between(fpr, tpr, 0, color=color if isinstance(color, tuple) else None, alpha=alpha)

# --------- 主逻辑 ----------
def main():
    ap = argparse.ArgumentParser(description="简洁 ROC 绘图（可调色）")
    ap.add_argument("--csv_train", type=str, default=None, help="训练集预测CSV")
    ap.add_argument("--csv_test",  type=str, default=None, help="测试/验证集预测CSV")
    ap.add_argument("--y_col",     type=str, default="y_true", help="标签列名")
    ap.add_argument("--p_col",     type=str, default=None, help="概率列名（默认自动）")
    ap.add_argument("--out_dir",   type=str, default="figs", help="输出目录")
    ap.add_argument("--lw",        type=float, default=2.0, help="线宽")
    ap.add_argument("--figsize",   type=str, default="6,5", help="图尺寸，如 6,5")
    ap.add_argument("--dpi",       type=int, default=200, help="DPI")
    ap.add_argument("--fill",      action="store_true", help="曲线下方填充面积")
    ap.add_argument("--alpha",     type=float, default=0.12, help="面积透明度")
    ap.add_argument("--train_color", type=str, default="109,109,255", help="训练色：'#6d6dff' 或 '109,109,255'")
    ap.add_argument("--test_color",  type=str, default="#F3A5D9",     help="测试色：'#F3A5D9' 或 '243,165,217'")
    args = ap.parse_args()

    # 解析参数
    try:
        fw, fh = [float(x) for x in args.figsize.split(",")]
    except Exception:
        fw, fh = 6.0, 5.0
    out_dir = Path(args.out_dir); _ensure_dir(out_dir)
    train_df = _read_csv(args.csv_train)
    test_df  = _read_csv(args.csv_test)
    train_c  = _parse_color(args.train_color, (109/255,109/255,255/255))
    test_c   = _parse_color(args.test_color,  (243/255,165/255,217/255))

    # 单图：Train
    if train_df is not None and not train_df.empty:
        y_col, p_col = args.y_col, _auto_prob_col(train_df, args.p_col)
        y, p = _prep_y_p(train_df, y_col, p_col)
        fig, ax = plt.subplots(figsize=(fw, fh), dpi=args.dpi)
        ax.plot([0,1], [0,1], "--", color="k", linewidth=1)  # 黑色 y=x
        _plot_roc_ax(ax, y, p, train_c, lw=args.lw, fill=args.fill, alpha=args.alpha)
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC (Train)", xlim=(0,1), ylim=(0,1))
        ax.grid(alpha=0.3, linestyle=":", linewidth=0.8); ax.legend(loc="lower right")
        fig.tight_layout(); fig.savefig(out_dir/"roc_train.png", bbox_inches="tight"); plt.close(fig)

    # 单图：Test
    if test_df is not None and not test_df.empty:
        y_col, p_col = args.y_col, _auto_prob_col(test_df, args.p_col)
        y, p = _prep_y_p(test_df, y_col, p_col)
        fig, ax = plt.subplots(figsize=(fw, fh), dpi=args.dpi)
        ax.plot([0,1], [0,1], "--", color="k", linewidth=1)
        _plot_roc_ax(ax, y, p, test_c, lw=args.lw, fill=args.fill, alpha=args.alpha)
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC (Test)", xlim=(0,1), ylim=(0,1))
        ax.grid(alpha=0.3, linestyle=":", linewidth=0.8); ax.legend(loc="lower right")
        fig.tight_layout(); fig.savefig(out_dir/"roc_test.png", bbox_inches="tight"); plt.close(fig)

    # 合并：Train vs Test
    if (train_df is not None and not train_df.empty) or (test_df is not None and not test_df.empty):
        fig, ax = plt.subplots(figsize=(fw, fh), dpi=args.dpi)
        # 坐标轴边框粗细
        for s in ax.spines.values():
            s.set_linewidth(2.0)        # ← 想多粗改这里
        
        # 刻度线粗细与长度
        ax.tick_params(axis="both", width=1.6, length=6)
        ax.plot([0,1], [0,1], "--", color="k", linewidth=2)
        if train_df is not None and not train_df.empty:
            y_col, p_col = args.y_col, _auto_prob_col(train_df, args.p_col)
            y, p = _prep_y_p(train_df, y_col, p_col)
            _plot_roc_ax(ax, y, p, train_c, label_prefix="Train", lw=args.lw, fill=args.fill, alpha=args.alpha)
        if test_df is not None and not test_df.empty:
            y_col, p_col = args.y_col, _auto_prob_col(test_df, args.p_col)
            y, p = _prep_y_p(test_df, y_col, p_col)
            _plot_roc_ax(ax, y, p, test_c, label_prefix="Test", lw=args.lw, fill=args.fill, alpha=args.alpha)
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", xlim=(0,1), ylim=(0,1))
        ax.grid(alpha=0.3, linestyle=":", linewidth=2); ax.legend(loc="lower right")
        fig.tight_layout(); fig.savefig(out_dir/"roc_train_vs_test.png", bbox_inches="tight"); plt.close(fig)
        

    print(f"[OK] saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
