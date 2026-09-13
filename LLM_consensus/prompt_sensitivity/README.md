# Prompt Sensitivity Analysis

Robustness analysis of the LLM consensus scoring framework against perturbations of
the scoring prompt (`../prompt.md`). The goal is to show that the platform's headline
conclusions (per-model winners and candidate rankings) are properties of the
evaluation framework rather than artifacts of one specific prompt phrasing.

## Method

Three controlled variants of the baseline prompt are generated programmatically:

| Variant | Class | Perturbation |
|---------|-------|--------------|
| V1 `v1_reorder` | Benign | Candidate presentation order reversed (Formula 10 -> 1) |
| V2 `v2_paraphrase` | Benign | Score anchors, parameter descriptions, and winner rule reworded (meaning-preserving) |
| V3 `v3_no_optimization` | Structural | The "constrained optimization" assessment paradigm removed |

Each variant is executed at temperature 0.0 with the same output contract as the
baseline and parsed with the same extractor (`../extract_data.py`).

Because temperature 0.0 does not guarantee determinism, every metric is interpreted
against a **split-half baseline reference**: each model's unperturbed baseline is
split into two halves and compared with itself, yielding the no-perturbation noise
level for that model. Variant effects are judged against this noise floor, not zero.

### Metrics

1. **Winner stability** - modal winner per condition vs the baseline modal winner
2. **Rank robustness** - Spearman rho of mean Total_Score ranks (variant vs baseline)
3. **Top-3 overlap** - Jaccard index of the top-3 candidate sets
4. **Dimension drift** - mean |delta| per scoring dimension relative to the baseline
   across-run noise floor; shifts > 2x the floor are flagged

## Usage

```bash
poetry shell
cd prompt_sensitivity

python make_variants.py        # build variants/ from ../prompt.md
python run_variants.py         # API calls -> results/raw/<variant>/<model>.md
python analyze_sensitivity.py  # metrics    -> results/sensitivity_results.json
python make_figures.py         # figure     -> results/nature_figure_sensitivity.png
```

`run_variants.py` requires `../.env` with `API_KEY` and `API_URL`, is resume-safe
(completed runs are skipped), and appends responses with the same run-header format
as the root pipeline so the shared extractor can parse them.

## Models and replication

| Model | Baseline | V1 | V2 | V3 | Note |
|-------|----------|----|----|----|------|
| gemini-3-pro-preview | 11 runs (root dataset) | 5 | 5 | 5 | |
| gpt-5 | 11 runs (root dataset) | 5 | 5 | 5 | |
| grok-4-5 | 5 runs (re-baselined here) | 3 | 3 | 3 | successor of retired `grok-4` |
| claude-opus-4-6 | 5 runs (re-baselined here) | 5 | 3 | 3 | successor of retired `claude-opus-4-5-20251101` |

`grok-4` and `claude-opus-4-5-20251101` were permanently retired by the provider
(HTTP 404) during the study; the two successor models were re-baselined inside this
experiment. Conclusions for those rows apply to the successors, not the retired
snapshots. Cross-generation agreement of winners is explicitly **not** claimed.

## Key findings (full details in `report.md`, Chinese)

- **Winner preserved in 10/12** variant x model conditions. Both models with stable
  baselines (Gemini: F4 11/11; GPT-5: F5 10/11) kept their winner under all three
  perturbations, including the structural one.
- The single variant-induced flip (grok-4-5 under V3, F5 -> F4) lies within that
  model's intrinsic run-to-run variability: its baseline winner is a weak plurality
  (3/5) that flips between the halves of its own unperturbed baseline.
- All rank correlations rho >= 0.64 (all p < 0.05, n = 10 candidates); variant
  effects never fall below each model's split-half noise reference in a way that
  would indicate framework fragility.
- Dimension-level scores are the least stable layer (drifts up to 3.2x the noise
  floor) and should not be cited as standalone conclusions.

## Structure

```
prompt_sensitivity/
├── __init__.py             # module doc + pipeline order
├── make_variants.py        # build variants/ from ../prompt.md
├── run_variants.py         # resume-safe API runner
├── analyze_sensitivity.py  # metrics -> results/sensitivity_results.json
├── make_figures.py         # 3-panel Nature-style figure (PNG + PDF)
├── report.md               # full analysis report (Chinese)
├── variants/               # generated prompt variant texts
└── results/
    ├── raw/<variant>/<model>.md   # raw responses, run-header format
    ├── sensitivity_results.json   # all metrics
    ├── nature_figure_sensitivity.png
    └── nature_figure_sensitivity.pdf
```

Colors follow the repo palette rules (main `#f4aedc` / `#7b7bff`, auxiliary colors
used symmetrically); figure labels are in English.
