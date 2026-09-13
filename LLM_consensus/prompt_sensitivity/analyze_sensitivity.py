# Analyze prompt-sensitivity results.
# Compares each variant (v1/v2/v3) against the baseline dataset (11 runs x 4 models)
# on: winner stability, winner margin, rank correlation of mean Total_Score,
# per-dimension shifts vs baseline noise floor, and top-3 overlap.
# A baseline split-half comparison (runs 0-4 vs 5-10) provides the no-perturbation
# reference level for the same metrics, so variant effects can be judged against
# intrinsic run-to-run variability rather than against zero.

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ENCODING = "utf-8"
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]

# Reuse the root regex extractor: it parses "{model}.md" files whose run headers
# match the format written by run_variants.py.
sys.path.insert(0, str(REPO_ROOT))
from extract_data import LLMDataExtractor  # noqa: E402

DIMS = [
    "Mechanical_Safety",
    "Swelling_Performance",
    "Endothelialization",
    "SMC_inhibition",
    "Anti_inflammation",
    "Thrombogenicity",
]

MODELS = [
    "gemini-3-pro-preview",
    "gpt-5",
    "grok-4-5",
    "claude-opus-4-6",
]

# Substitutes: gateway retired grok-4 / claude-opus-4-5-20251101 mid-experiment,
# so grok-4-5 and claude-opus-4-6 were run against baseline AND variants here.
SUBSTITUTES = {"grok-4-5", "claude-opus-4-6"}

VARIANTS = ["v1_reorder", "v2_paraphrase", "v3_no_optimization"]

RAW_DIR = HERE / "results" / "raw"
OUT_PATH = HERE / "results" / "sensitivity_results.json"


def load_baseline():
    """Load baseline records per model.

    Original models come from the root extracted_data.json (11 runs each).
    Substitute models have no root baseline; theirs is extracted from
    results/raw/baseline_prompt/ (5 runs collected in this experiment).
    """
    src = REPO_ROOT / "extracted_data.json"
    with open(src, "r", encoding=ENCODING) as fh:
        data = json.load(fh)
    extractor = LLMDataExtractor(base_dir=str(RAW_DIR / "baseline_prompt"))
    for model in MODELS:
        if model in SUBSTITUTES:
            df = extractor.extract_data_from_md(model)
            data[model] = df.to_dict("records") if df is not None else []
    return data


def load_variant(variant: str):
    """Extract records from raw variant outputs using the root extractor."""
    extractor = LLMDataExtractor(base_dir=str(RAW_DIR / variant))
    out = {}
    for model in MODELS:
        df = extractor.extract_data_from_md(model)
        out[model] = df.to_dict("records") if df is not None else []
    return out


def mean_scores(records):
    """Mean score per formula for each dimension + Total_Score -> {formula: {...}}."""
    per_formula = {}
    for rec in records:
        f = rec.get("Formula")
        if f is None:
            continue
        per_formula.setdefault(f, []).append(rec)
    means = {}
    for f, recs in sorted(per_formula.items()):
        entry = {}
        for dim in DIMS + ["Total_Score"]:
            vals = [r.get(dim) for r in recs if isinstance(r.get(dim), (int, float))]
            entry[dim] = float(np.mean(vals)) if vals else np.nan
        means[f] = entry
    return means


def winners(records):
    """Stated winner per run; falls back to argmax Total_Score when unstated."""
    by_run = {}
    for rec in records:
        by_run.setdefault(rec.get("Run"), []).append(rec)
    result = {}
    for run, recs in sorted(by_run.items(), key=lambda kv: (kv[0] is None, kv[0])):
        stated = {r.get("Winner") for r in recs if r.get("Winner") is not None}
        if len(stated) == 1:
            result[run] = {"winner": int(next(iter(stated))), "source": "stated"}
        else:
            scored = [
                (r.get("Formula"), r.get("Total_Score"))
                for r in recs
                if isinstance(r.get("Total_Score"), (int, float))
            ]
            if scored:
                best = max(scored, key=lambda t: t[1])[0]
                result[run] = {"winner": int(best), "source": "argmax_fallback"}
    return result


def winner_margin(records):
    """Mean (top1 - top2) of run-level Total_Score values across runs."""
    by_run = {}
    for rec in records:
        by_run.setdefault(rec.get("Run"), []).append(rec)
    margins = []
    for run, recs in by_run.items():
        totals = {}
        for r in recs:
            f = r.get("Formula")
            t = r.get("Total_Score")
            if f is not None and isinstance(t, (int, float)):
                totals[f] = t
        if len(totals) >= 2:
            srt = sorted(totals.values(), reverse=True)
            margins.append(srt[0] - srt[1])
    return float(np.mean(margins)) if margins else np.nan


def modal_winner(winners_by_run):
    """Most frequent winner across runs with tie/absence handling."""
    vals = [w["winner"] for w in winners_by_run.values()]
    if not vals:
        return None, {}
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.items(), key=lambda kv: kv[1])
    return top[0], counts


def rank_vector(means):
    """Total_Score means ordered by formula id 1..10."""
    return np.array([means[f]["Total_Score"] for f in range(1, 11)], dtype=float)


def top3_set(means):
    arr = rank_vector(means)
    if np.all(np.isnan(arr)):
        return set()
    order = np.argsort(-np.nan_to_num(arr, nan=-np.inf))
    return {int(order[i]) + 1 for i in range(min(3, len(order)))}


def noise_floor(baseline_records):
    """Per model/dim: mean over formulas of the across-run std in baseline."""
    by_model = {}
    for model in MODELS:
        recs = baseline_records.get(model, [])
        by_run = {}
        for r in recs:
            by_run.setdefault(r.get("Run"), []).append(r)
        per_formula_runs = {f: {d: [] for d in DIMS} for f in range(1, 11)}
        for run, rrecs in by_run.items():
            for r in rrecs:
                f = r.get("Formula")
                if f is None:
                    continue
                for d in DIMS:
                    v = r.get(d)
                    if isinstance(v, (int, float)):
                        per_formula_runs[f][d].append(v)
        floor = {}
        for d in DIMS:
            stds = [
                float(np.std(per_formula_runs[f][d], ddof=1))
                for f in range(1, 11)
                if len(per_formula_runs[f][d]) >= 2
            ]
            floor[d] = float(np.mean(stds)) if stds else np.nan
        by_model[model] = floor
    return by_model


def compare(cond_means, base_means, floor_for_model):
    """Metrics comparing one condition's mean-score table to a reference table."""
    base_vec, cond_vec = rank_vector(base_means), rank_vector(cond_means)
    ok = ~(np.isnan(base_vec) | np.isnan(cond_vec))
    if ok.sum() == 10:
        rho, rho_p = spearmanr(base_vec[ok], cond_vec[ok])
        rho, rho_p = float(rho), float(rho_p)
    else:
        rho, rho_p = np.nan, np.nan
    shifts = {}
    n_exceed = 0
    for d in DIMS:
        deltas = [
            cond_means[f][d] - base_means[f][d]
            for f in range(1, 11)
            if not (np.isnan(cond_means[f][d]) or np.isnan(base_means[f][d]))
        ]
        mean_abs_shift = float(np.mean(np.abs(deltas))) if deltas else np.nan
        shifts[d] = {"mean_abs_shift": mean_abs_shift, "noise_floor": floor_for_model[d]}
        if not np.isnan(mean_abs_shift) and floor_for_model[d] > 0:
            if mean_abs_shift > 2 * floor_for_model[d]:
                n_exceed += 1
    t3a, t3b = top3_set(base_means), top3_set(cond_means)
    jacc = float(len(t3a & t3b) / len(t3a | t3b)) if (t3a and t3b) else np.nan
    return {
        "spearman_total_score": rho,
        "spearman_p": rho_p,
        "top3_jaccard": jacc,
        "dim_shifts": shifts,
        "dims_exceed_2x_noise": n_exceed,
    }


def main() -> None:
    baseline = load_baseline()
    print(f"[Okay] baseline loaded: {sum(len(v) for v in baseline.values())} records")

    results = {"baseline": {}, "variants": {}, "split_half_reference": {}}
    base_means_by_model = {m: mean_scores(baseline.get(m, [])) for m in MODELS}
    base_win_by_model = {m: winners(baseline.get(m, [])) for m in MODELS}
    floors = noise_floor(baseline)

    for m in MODELS:
        modal, counts = modal_winner(base_win_by_model[m])
        results["baseline"][m] = {
            "modal_winner": modal,
            "winner_counts": counts,
            "winner_margin_mean": winner_margin(baseline.get(m, [])),
        }
        print(f"[Base] {m}: modal winner F{modal}, counts {counts}")

    # Split-half reference: baseline split at the run midpoint (0-4 vs 5-10 for
    # the 11-run originals; 0-2 vs 3-4 for the 5-run substitutes), same metrics,
    # no perturbation.
    for m in MODELS:
        recs = baseline.get(m, [])
        run_ids = sorted({r.get("Run") for r in recs if r.get("Run") is not None})
        first_half = set(run_ids[: len(run_ids) // 2])
        half_a = [r for r in recs if r.get("Run") in first_half]
        half_b = [r for r in recs if r.get("Run") not in first_half]
        ma, mb = mean_scores(half_a), mean_scores(half_b)
        cmp_res = compare(ma, mb, floors[m])
        wa, _ = modal_winner(winners(half_a))
        wb, _ = modal_winner(winners(half_b))
        cmp_res["half_a_modal_winner"] = wa
        cmp_res["half_b_modal_winner"] = wb
        cmp_res["winner_agrees"] = (wa is not None and wa == wb)
        results["split_half_reference"][m] = cmp_res
        print(
            f"[SplitHalf] {m}: winner {wa} vs {wb}, "
            f"rho={cmp_res['spearman_total_score']:.3f}"
        )

    for variant in VARIANTS:
        vdata = load_variant(variant)
        results["variants"][variant] = {}
        for m in MODELS:
            recs = vdata.get(m, [])
            if not recs:
                print(f"[Warn] {variant}/{m}: no records extracted")
                results["variants"][variant][m] = {"error": "no records"}
                continue
            means = mean_scores(recs)
            wby = winners(recs)
            modal, counts = modal_winner(wby)
            base_modal = results["baseline"][m]["modal_winner"]
            cmp_res = compare(means, base_means_by_model[m], floors[m])
            n_runs = len({r.get("Run") for r in recs})
            fallbacks = sum(1 for w in wby.values() if w["source"] == "argmax_fallback")
            entry = {
                "n_runs": n_runs,
                "n_records": len(recs),
                "modal_winner": modal,
                "winner_counts": counts,
                "winner_matches_baseline_modal": (modal is not None and modal == base_modal),
                "winner_margin_mean": winner_margin(recs),
                "argmax_fallback_runs": fallbacks,
                **cmp_res,
            }
            results["variants"][variant][m] = entry
            print(
                f"[Variant] {variant}/{m}: winner F{modal} (base F{base_modal}), "
                f"rho={entry['spearman_total_score']:.3f}, "
                f"jacc={entry['top3_jaccard']:.2f}, "
                f"dims>2xNoise={entry['dims_exceed_2x_noise']}"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding=ENCODING) as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=float)
    print(f"[Okay] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
