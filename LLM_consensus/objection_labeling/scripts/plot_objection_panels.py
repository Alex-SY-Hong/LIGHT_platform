"""
Two-panel Nature-style figure: objection distribution against Formulas 4 and 5.

Panel a: grouped bars, percentage of arguments per objection dimension (F4 vs F5).
Panel b: 7x7 co-occurrence matrix, upper triangle F4 / lower triangle F5.

Data source: objection_labeling/results/labeled_arguments.json (scoped labeling run).
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

BASE = Path(__file__).resolve().parent.parent / "results"
DATA_PATH = BASE / "labeled_arguments.json"
OUT_DIR = BASE

C_F4 = "#7b7bff"
C_F5 = "#f4aedc"

DIMENSIONS = [
    "Mechanical_Safety",
    "Swelling_Performance",
    "Endothelialization",
    "SMC_inhibition",
    "Anti_inflammation",
    "Thrombogenicity",
    "Other",
]

SHORT = {
    "Mechanical_Safety": "Mechanical\nsafety",
    "Swelling_Performance": "Swelling",
    "Endothelialization": "Endothel.",
    "SMC_inhibition": "SMC\ninhib.",
    "Anti_inflammation": "Anti-infl.",
    "Thrombogenicity": "Thrombo.",
    "Other": "Other",
}


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_dims(records):
    c = Counter()
    for r in records:
        for dim in r["labels"]:
            c[dim] += 1
    return c


def co_occurrence(records):
    m = Counter()
    for r in records:
        labels = r["labels"]
        for i, d1 in enumerate(labels):
            for d2 in labels[i + 1:]:
                m[tuple(sorted([d1, d2]))] += 1
    return m


def draw_panel_a(ax, f4_counts, f5_counts, n4, n5):
    # Sort dimensions by F4-F5 count difference (descending): contrast visible left-to-right
    order = sorted(DIMENSIONS, key=lambda d: f4_counts.get(d, 0) - f5_counts.get(d, 0), reverse=True)
    x = np.arange(len(order))
    w = 0.38

    f4_pct = [f4_counts.get(d, 0) / n4 * 100 for d in order]
    f5_pct = [f5_counts.get(d, 0) / n5 * 100 for d in order]

    b1 = ax.bar(x - w / 2, f4_pct, w, color=C_F4, label=f"Formula 4 (n={n4})")
    b2 = ax.bar(x + w / 2, f5_pct, w, color=C_F5, edgecolor="#8a5a80", linewidth=0.4,
                label=f"Formula 5 (n={n5})")

    for bars, counts in ((b1, [f4_counts.get(d, 0) for d in order]),
                         (b2, [f5_counts.get(d, 0) for d in order])):
        for rect, cnt in zip(bars, counts):
            ax.annotate(f"{cnt}", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[d] for d in order], fontsize=6)
    ax.set_ylabel("Objection frequency (% of arguments)")
    ax.set_ylim(0, max(f4_pct + f5_pct) * 1.18)
    ax.legend(loc="upper right", fontsize=6, handlelength=1.0, handleheight=1.0)
    ax.text(0.0, 1.02, "Multi-label: column sums exceed 100%", transform=ax.transAxes,
            fontsize=5.5, color="0.35")


def draw_panel_b(ax, f4_co, f5_co, f4_counts, f5_counts):
    n = len(DIMENSIONS)
    vmax = max(list(f4_co.values()) + list(f5_co.values()) + [1])

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")

    for i in range(n):
        for j in range(n):
            d1, d2 = DIMENSIONS[i], DIMENSIONS[j]
            if i == j:
                fc, alpha = "0.92", 1.0
                txt = f"{f4_counts.get(d1, 0)}|{f5_counts.get(d2, 0)}"
                tcolor = "0.15"
                val = 0
            elif j > i:  # upper triangle: F4
                val = f4_co.get(tuple(sorted([d1, d2])), 0)
                fc, alpha = C_F4, 0.12 + 0.88 * val / vmax
                txt = str(val)
            else:  # lower triangle: F5
                val = f5_co.get(tuple(sorted([d1, d2])), 0)
                fc, alpha = C_F5, 0.12 + 0.88 * val / vmax
                txt = str(val)
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fc, alpha=alpha,
                                   edgecolor="white", linewidth=1))
            if i != j:
                tcolor = "white" if val / vmax > 0.55 else "0.15"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=5.5,
                    color=tcolor)

    labels = [SHORT[d].replace("\n", " ") for d in DIMENSIONS]
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels, fontsize=5.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    ax.text(1.0, 1.04, "Upper triangle: Formula 4   Lower triangle: Formula 5",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color="0.35")


def main():
    data = load_data(DATA_PATH)
    f4, f5 = data["formula_4"], data["formula_5"]
    n4, n5 = len(f4), len(f5)

    f4_counts, f5_counts = count_dims(f4), count_dims(f5)
    f4_co, f5_co = co_occurrence(f4), co_occurrence(f5)

    fig = plt.figure(figsize=(7.2, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.42,
                          left=0.06, right=0.985, top=0.86, bottom=0.22)

    ax_a = fig.add_subplot(gs[0, 0])
    draw_panel_a(ax_a, f4_counts, f5_counts, n4, n5)
    ax_a.text(-0.14, 1.12, "a", transform=ax_a.transAxes, fontsize=9, fontweight="bold")

    ax_b = fig.add_subplot(gs[0, 1])
    draw_panel_b(ax_b, f4_co, f5_co, f4_counts, f5_counts)
    ax_b.text(-0.32, 1.12, "b", transform=ax_b.transAxes, fontsize=9, fontweight="bold")

    fig.suptitle("Objection dimensions raised against the two leading formulas",
                 x=0.06, y=0.965, ha="left", fontsize=8, fontweight="bold")

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    stem = OUT_DIR / "objection_panels"
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    print(f"[Okay] saved {stem}.svg/.pdf/.png")


if __name__ == "__main__":
    main()
