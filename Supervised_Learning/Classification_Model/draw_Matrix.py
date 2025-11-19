#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable  # ✅ 新增导入

# ======== 全局统一样式 ========
AX_SPINE_LW = 1.5
TICK_W      = 1.5
TICK_LEN    = 6
FIGSIZE     = (6, 6)
DPI         = 500

plt.rcParams.update({
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "axes.linewidth": AX_SPINE_LW,
    "axes.unicode_minus": False,
    "axes.grid": False,
})

# ============ 工具函数 ============
def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def _parse_cmap(cmap_arg: str):
    s = (cmap_arg or "").strip()
    if s.startswith("#"):
        s = s.lstrip("#")
        r = int(s[0:2], 16)/255; g = int(s[2:4], 16)/255; b = int(s[4:6], 16)/255
        return LinearSegmentedColormap.from_list("single_hex", [(1,1,1), (r,g,b)])
    if "," in s:
        r,g,b = [float(x)/255 for x in s.split(",")]
        return LinearSegmentedColormap.from_list("single_rgb", [(1,1,1), (r,g,b)])
    return cmap_arg

def _load_from_matrix(csv_path: Path) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    if df.shape[0] == df.shape[1]:
        labels = [str(c) for c in df.columns]
        if "unnamed" in str(df.columns[0]).lower():
            maybe_labels = df.iloc[:,0].astype(str).tolist()
            if len(set(maybe_labels)) == df.shape[0]:
                labels = [str(c) for c in df.columns[1:]]
                cm = df.iloc[:,1:].to_numpy()
                return cm, labels
        return df.to_numpy(), labels
    if df.shape[1] == df.shape[0] + 1:
        labels = [str(c) for c in df.columns[1:]]
        cm = df.iloc[:,1:].to_numpy()
        return cm, labels
    raise ValueError(f"{csv_path} 不是 NxN 或 N x (N+1) 的混淆矩阵 CSV")

def _load_from_preds(csv_path: Path, y_col: str, yhat_col: str,
                     labels_arg: Optional[str]) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    if y_col not in df.columns or yhat_col not in df.columns:
        raise ValueError(f"CSV 缺列：{y_col} 或 {yhat_col}")
    y = df[y_col].dropna().astype(str).values
    yhat = df[yhat_col].dropna().astype(str).values
    if labels_arg:
        labels = [s.strip() for s in labels_arg.split(",")]
    else:
        labels = sorted(list(set(list(y) + list(yhat))))
    cm = confusion_matrix(y, yhat, labels=labels)
    return cm, labels

def _normalize_cm(cm: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return cm.astype(int)
    cm = cm.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        if mode == "true":
            denom = cm.sum(axis=1, keepdims=True)
            out = np.divide(cm, denom, where=denom!=0)
        elif mode == "pred":
            denom = cm.sum(axis=0, keepdims=True)
            out = np.divide(cm, denom, where=denom!=0)
        elif mode == "all":
            total = cm.sum()
            out = cm / total if total > 0 else cm
        else:
            raise ValueError("normalize 仅支持 none/true/pred/all")
    return np.nan_to_num(out)

# ============ 绘图函数 ============
def plot_confmat(
    cm: np.ndarray,
    labels: List[str],
    out_path: Path,
    *,
    title: str,
    normalize: str = "none",
    cmap: Union[str, LinearSegmentedColormap] = "Blues",
    figsize: Tuple[float,float] = FIGSIZE,
    dpi: int = DPI,
    rotate_xticks: int = 0,
    # 字号统一
    title_size: int = 20,
    label_size: int = 20,
    tick_size: int  = 20,
    cell_size: int  = 18,
    cbar_tick_size: int = 18,
    # 样式
    spine_lw: float = AX_SPINE_LW,
    tick_w: float = TICK_W,
    tick_len: float = TICK_LEN,
    grid: bool = False,
    fmt_decimals: int = 2
):
    cm_show = _normalize_cm(cm, normalize)
    cm_show = cm_show.T  # ✅ X=True, Y=Pred
    is_float = (normalize != "none")
    fmt = f".{fmt_decimals}f" if is_float else "d"

    # ✅ 使用固定色条布局
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_box_aspect(1)  # ✅ 关键行
    im = ax.imshow(cm_show, cmap=cmap)


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=cbar_tick_size)

    ax.set_xlabel("True label", fontsize=label_size)
    ax.set_ylabel("Pred label", fontsize=label_size)
    ax.set_title(title, fontsize=title_size)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=tick_size, rotation=rotate_xticks)
    ax.set_yticklabels(labels, fontsize=tick_size)

    for s in ax.spines.values():
        s.set_linewidth(spine_lw)
    ax.tick_params(axis="both", width=tick_w, length=tick_len)

    thresh = (cm_show.max() + cm_show.min()) / 2.0
    for i in range(cm_show.shape[0]):
        for j in range(cm_show.shape[1]):
            val = cm_show[i, j]
            txt = format(val, fmt)
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=cell_size,
                    color="white" if val > thresh else "black")

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

# ============ 主流程 ============
def main():
    ap = argparse.ArgumentParser(description="科研统一风格混淆矩阵绘制")
    ap.add_argument("--csv_train", type=str, default=None)
    ap.add_argument("--csv_test",  type=str, default=None)
    ap.add_argument("--train_matrix", action="store_true")
    ap.add_argument("--test_matrix",  action="store_true")
    ap.add_argument("--y_col", type=str, default="y_true")
    ap.add_argument("--yhat_col", type=str, default="y_pred")
    ap.add_argument("--labels", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="figs")
    ap.add_argument("--out_train", type=str, default="confmat_train.png")
    ap.add_argument("--out_test", type=str, default="confmat_test.png")
    ap.add_argument("--train_title", type=str, default=None)
    ap.add_argument("--test_title",  type=str, default=None)
    ap.add_argument("--normalize", type=str, default="none", choices=["none","true","pred","all"])
    ap.add_argument("--cmap", type=str, default="Blues")
    ap.add_argument("--rotate_xticks", type=int, default=0)
    args = ap.parse_args()

    cmap = _parse_cmap(args.cmap)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def _process_one(kind: str, csv_path: Optional[str], is_matrix: bool):
        if not csv_path:
            return
        p = Path(csv_path)
        if not p.is_file():
            raise FileNotFoundError(f"{kind} 文件不存在：{p}")

        cm, labels = (_load_from_matrix(p) if is_matrix else
                      _load_from_preds(p, args.y_col, args.yhat_col, args.labels))

        title = "Swelling Ratio (times) Train" if kind == "train" else "Swelling Ratio (times) Test"
        out_name = f"confmat_{kind}.png"
        out_path = out_dir / out_name

        plot_confmat(cm, labels, out_path=out_path, title=title,
                     normalize=args.normalize, cmap=cmap,
                     figsize=FIGSIZE, dpi=DPI,
                     spine_lw=AX_SPINE_LW, tick_w=TICK_W, tick_len=TICK_LEN)

        print(f"[OK] {kind} → {out_path.resolve()}")

    _process_one("train", args.csv_train, args.train_matrix)
    _process_one("test", args.csv_test, args.test_matrix)

if __name__ == "__main__":
    main()
