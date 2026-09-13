# Prompt sensitivity analysis module.
#
# Tests whether the consensus platform's per-model winners and candidate
# rankings are robust to perturbations of the scoring prompt:
#   V1 reorder          - candidate presentation order reversed (benign)
#   V2 paraphrase       - anchors/parameters/winner rule reworded (benign)
#   V3 no-optimization  - constrained-optimization paradigm removed (structural)
#
# Pipeline (run in order, from this directory):
#   python make_variants.py       # build variants/ from the root prompt.md
#   python run_variants.py        # API calls -> results/raw/<variant>/<model>.md
#   python analyze_sensitivity.py # metrics    -> results/sensitivity_results.json
#   python make_figures.py        # figure     -> results/nature_figure_sensitivity.png
#
# Variant effects are interpreted against a split-half baseline reference
# (no-perturbation run-to-run noise), not against zero. See report.md for
# the full analysis and README.md for usage.
