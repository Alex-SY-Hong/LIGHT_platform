#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python Youngs_Distribution_Statistics_Plot.py

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"

# ===== Plotting Parameters (Unified control of line widths) =====
AX_SPINE_LW = 1.5   # Axis spine line width
TICK_W      = 1.5   # Tick line width
TICK_LEN    = 6     # Tick length
GRID_LW     = 1.0   # Grid line width
BAR_EDGE_LW = 1.5   # Bar edge line width


def main():
    # ===== Input Paths =====
    CSV_FILE = "youngs_modulus.csv"
    TARGET = "Young's Modulus (kPa) log10"
    TARGET_name = r"$\log_{10}(\mathrm{Young’s\ Modulus}\;(\mathrm{kPa}))$"

    # ===== Automatically try encodings =====
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

    # ===== Plot Histogram (Square Canvas) =====
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    COLOR = (109/255, 109/255, 255/255)

    ax.hist(
        df[TARGET].dropna(),
        bins=30,
        color=COLOR,
        edgecolor="black",
        linewidth=BAR_EDGE_LW,
        alpha=0.9
    )

    ax.set_xlabel(TARGET_name, fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)

    # ⚠️ Do not set aspect ratio to 1:1
    # ax.set_aspect('equal', adjustable='box')

    # ===== Grid + Axis Style =====
    ax.grid(axis='x', linestyle='--', alpha=0.3, linewidth=GRID_LW)
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=GRID_LW)

    for s in ax.spines.values():
        s.set_linewidth(AX_SPINE_LW)

    ax.tick_params(axis="both", labelsize=20, width=TICK_W, length=TICK_LEN)

    # ===== Save Output =====
    out_path = r"C:\Users\user\Desktop\Youngs_Distribution_square.png"
    plt.savefig(out_path, dpi=500, bbox_inches="tight", pad_inches=0.2)
    print(f"\n✅ Image saved to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()