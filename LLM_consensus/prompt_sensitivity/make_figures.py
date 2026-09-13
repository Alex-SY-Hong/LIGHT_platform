# Build the Nature-style summary figure for the prompt-sensitivity experiment.
# Reads results/sensitivity_results.json (written by analyze_sensitivity.py) and
# renders a 3-panel figure:
#   a) winner stability: winner identity per model x condition
#   b) rank robustness: Spearman rho of mean Total_Score vs baseline,
#      with the baseline split-half reference band (no-perturbation floor)
#   c) score shifts vs noise floor: mean |delta| per dimension relative to
#      the baseline across-run std (values >1 exceed 1x noise, >2 exceed 2x)
# Colors follow the repo palette rules; labels in English (figure captions rule).

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ENCODING = "utf-8"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "sensitivity_results.json"
OUT_PNG = HERE / "results" / "nature_figure_sensitivity.png"
OUT_PDF = HERE / "results" / "nature_figure_sensitivity.pdf"

MODELS = [
    "gemini-3-pro-preview",
    "gpt-5",
    "grok-4-5",
    "claude-opus-4-6",
]
MODEL_SHORT = ["Gemini-3 Pro", "GPT-5", "Grok-4.5", "Claude Opus 4.6"]
VARIANTS = ["v1_reorder", "v2_paraphrase", "v3_no_optimization"]
VARIANT_SHORT = ["V1 order", "V2 paraphrase", "V3 no-opt"]

DIMS = [
    "Mechanical_Safety",
    "Swelling_Performance",
    "Endothelialization",
    "SMC_inhibition",
    "Anti_inflammation",
    "Thrombogenicity",
]
DIM_SHORT = ["Mech", "Swell", "Endo", "SMC", "AntiInf", "Thromb"]

# Repo palette: main binary + auxiliary used symmetrically.
MAIN_A = "#f4aedc"
MAIN_B = "#7b7bff"
AUX = ["#142864", "#38307c", "#5c3894", "#864088", "#b64858", "#e65028"]
VARIANT_COLORS = [AUX[0], MAIN_B, AUX[5]]  # symmetric 1 / main / 6

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

FORMULA_CMAP = plt.get_cmap("tab10")


def load():
    with open(RESULTS, "r", encoding=ENCODING) as fh:
        return json.load(fh)


def panel_winner(ax, data):
    """Panel a: winner identity per model (rows) x condition (cols)."""
    cols = ["Baseline", "Split-half"] + VARIANT_SHORT
    ncols = len(cols)
    nrows = len(MODELS)
    grid = np.full((nrows, ncols), np.nan)
    annot = [["" for _ in range(ncols)] for _ in range(nrows)]

    for r, m in enumerate(MODELS):
        base = data["baseline"][m]
        grid[r, 0] = base["modal_winner"]
        cnt = base["winner_counts"]
        tot = sum(cnt.values()) or 1
        key = str(base["modal_winner"])
        annot[r][0] = f"F{base['modal_winner']} {cnt.get(key, 0)}/{tot}"
        sh = data["split_half_reference"][m]
        grid[r, 1] = sh["half_a_modal_winner"]
        annot[r][1] = f"F{sh['half_a_modal_winner']}"
        for c, v in enumerate(VARIANTS, start=2):
            entry = data["variants"][v].get(m, {})
            w = entry.get("modal_winner")
            if w is not None:
                grid[r, c] = w
                hit = entry.get("winner_matches_baseline_modal")
                annot[r][c] = f"F{w} {'=' if hit else 'X'}"

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    for r in range(nrows):
        for c in range(ncols):
            val = grid[r, c]
            face = "#ffffff" if np.isnan(val) else FORMULA_CMAP(int(val) - 1)
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=face,
                                       edgecolor="#333333", linewidth=0.4))
            if not np.isnan(val):
                ax.text(c + 0.5, r + 0.5, annot[r][c],
                        ha="center", va="center", fontsize=5.6, color="#111111")
    ax.set_xticks([i + 0.5 for i in range(ncols)])
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks([i + 0.5 for i in range(nrows)])
    ax.set_yticklabels(MODEL_SHORT)
    ax.set_title("a  Winner identity across prompt conditions", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def panel_rho(ax, data):
    """Panel b: Spearman rho vs baseline + split-half reference band."""
    x = np.arange(len(MODELS))
    width = 0.24
    sh_vals = [data["split_half_reference"][m]["spearman_total_score"] for m in MODELS]
    lo, hi = float(np.nanmin(sh_vals)), float(np.nanmax(sh_vals))
    ax.axhspan(lo, hi, color=MAIN_A, alpha=0.35, zorder=0)
    ax.axhline(1.0, color="#999999", linewidth=0.5, linestyle=":", zorder=1)

    for i, (v, vshort) in enumerate(zip(VARIANTS, VARIANT_SHORT)):
        vals = [data["variants"][v].get(m, {}).get("spearman_total_score", np.nan)
                for m in MODELS]
        ax.bar(x + (i - 1) * width, vals, width, label=vshort,
               color=VARIANT_COLORS[i], edgecolor="#333333", linewidth=0.4, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_SHORT, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Spearman $\\rho$ of Total Score ranks\n(vs baseline)")
    ax.axhline(0.5, color="#b64858", linewidth=0.6, linestyle="--", zorder=1)
    ax.text(1.5, 0.53, r"$\rho$=0.5", color="#b64858", fontsize=5.5,
            ha="center")
    ax.set_title("b  Ranking robustness", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
              ncol=1, columnspacing=1.2, handlelength=1.2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def panel_noise(ax, data):
    """Panel c: per-dimension |shift| / noise-floor ratio, averaged over models."""
    x = np.arange(len(DIMS))
    width = 0.24
    for i, (v, vshort) in enumerate(zip(VARIANTS, VARIANT_SHORT)):
        ratios = []
        for d in DIMS:
            per_model = []
            for m in MODELS:
                entry = data["variants"][v].get(m, {})
                shift = entry.get("dim_shifts", {}).get(d, {})
                mv = shift.get("mean_abs_shift") if shift else None
                fl = shift.get("noise_floor") if shift else None
                if mv is not None and fl is not None and fl:
                    per_model.append(mv / fl)
            ratios.append(float(np.nanmean(per_model)) if per_model else np.nan)
        ax.bar(x + (i - 1) * width, ratios, width, label=vshort,
               color=VARIANT_COLORS[i], edgecolor="#333333", linewidth=0.4)

    ax.axhline(1.0, color="#555555", linewidth=0.6, linestyle="--")
    ax.axhline(2.0, color="#e65028", linewidth=0.6, linestyle="--")
    ax.text(len(DIMS) - 0.4, 1.03, "1x noise", fontsize=5.5, color="#555555")
    ax.text(len(DIMS) - 0.4, 2.03, "2x noise", fontsize=5.5, color="#e65028")
    ax.set_xticks(x)
    ax.set_xticklabels(DIM_SHORT)
    ax.set_ylabel("Mean |dimension shift| /\nbaseline noise floor")
    ax.set_title("c  Score shifts vs run-to-run noise", loc="left", fontweight="bold")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def main():
    data = load()
    fig = plt.figure(figsize=(7.0, 6.3))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.0, 1.0], hspace=0.75)

    ax_a = fig.add_subplot(gs[0])
    panel_winner(ax_a, data)

    ax_b = fig.add_subplot(gs[1])
    panel_rho(ax_b, data)

    ax_c = fig.add_subplot(gs[2])
    panel_noise(ax_c, data)

    handles = [Patch(facecolor=FORMULA_CMAP(i), edgecolor="#333333", linewidth=0.4,
                     label=f"F{i + 1}") for i in range(10)]
    fig.legend(handles=handles, ncol=10, loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=5.5)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"[Okay] wrote {OUT_PNG}")
    print(f"[Okay] wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
