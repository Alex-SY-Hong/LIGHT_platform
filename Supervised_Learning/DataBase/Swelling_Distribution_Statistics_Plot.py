#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python Swelling_Distribution_Statistics_Plot.py

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ===== Style Controls =====
AX_SPINE_LW = 1.5
TICK_W = 1.5
TICK_LEN = 6
GRID_LW = 1.0
BAR_EDGE_LW = 1.5

FIGSIZE = (6, 6)
DPI = 1000

# Global Font
plt.rcParams["font.family"] = "Arial"


def main():
    # ===== Input Paths =====
    CSV_FILE = "swelling_ratio.csv"
    TARGET = "Swelling Ratio (times)"
    TARGET_name = "Swelling Ratio (times)"

    # ===== Read Data =====
    encodings = ["utf-8-sig", "gb18030", "latin1"]
    for enc in encodings:
        try:
            df = pd.read_csv(CSV_FILE, encoding=enc)
            print(f"✅ Successfully read file. Encoding: {enc}")
            break
        except Exception as e:
            print(f"Encoding {enc} failed: {e}")
    else:
        raise RuntimeError("❌ Failed to read CSV file. Please check file encoding!")

    if TARGET not in df.columns:
        raise ValueError(f"❌ Column '{TARGET}' does not exist. Please check the CSV file!")

    data = df[TARGET].dropna()

    # ===== Define Break Axis Intervals =====
    left_min, left_max = 0, 150
    right_min, right_max = 280, 310
    fold_text = "150–290 (no data)"

    # ===== Create Canvas (Square) =====
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

    plt.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.12, wspace=0.05)

    COLOR = (243/255, 165/255, 217/255)

    # ===== Left Histogram =====
    ax1.hist(data[data < left_max], bins=20, color=COLOR,
             edgecolor="black", linewidth=BAR_EDGE_LW, alpha=0.9)
    ax1.set_xlim(left_min, left_max)

    # ===== Right Histogram =====
    right_data = data[data >= right_min]
    ax2.hist(right_data, bins=5, range=(right_min, right_max),
             color=COLOR, edgecolor="black", linewidth=BAR_EDGE_LW, alpha=0.9)
    ax2.set_xlim(right_min - 5, right_max + 5)

    # ===== Remove Middle Spines =====
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # ===== Break Symbols (Diagonals) =====
    d = .012
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1.8)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    # ===== Unify Styles =====
    for ax in (ax1, ax2):
        ax.grid(axis='x', linestyle='--', alpha=0.3, linewidth=GRID_LW)
        ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=GRID_LW)
        ax.tick_params(axis="both", labelsize=26, width=TICK_W, length=TICK_LEN, top=False)
        for s in ax.spines.values():
            s.set_linewidth(AX_SPINE_LW)

    # ===== Ticks Adjustment =====
    xticks_left = [tick for tick in ax1.get_xticks() if tick < left_max]
    ax1.set_xticks(xticks_left)

    xticks_right = [tick for tick in ax2.get_xticks() if tick > right_min]
    if len(xticks_right) > 0:
        xticks_right = xticks_right[:-1]
    ax2.set_xticks(xticks_right)

    ax1.tick_params(labelbottom=True)
    ax2.tick_params(labelbottom=True)
    ax1.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_ticks_position('bottom')

    # ===== Labels =====
    fig.text(0.5, 0.02, TARGET_name, ha='center', va='center', fontsize=20)
    ax1.set_ylabel("Frequency", fontsize=20)

    # Hide Y ticks on the right plot
    ax2.tick_params(axis='y', which='both', left=False, right=False,
                    labelleft=False, labelright=False)

    # ===== Enforce Square Image Dimensions =====
    plt.gcf().set_size_inches(6, 6)

    # ===== Save Output =====
    out_path = r"C:\Users\user\Desktop\Swelling_Distribution_square_folded_perfect.png"
    plt.savefig(out_path, dpi=1000, bbox_inches="tight", pad_inches=0.1)
    print(f"\n✅ Image saved to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()