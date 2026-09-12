"""Regenerate labeling_statistics.md from existing labeled_arguments.json (no API calls)."""
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import label_dimensions as ld  # noqa: E402 (handles package import + .env itself)

RESULTS = SCRIPTS_DIR.parent / "results"

with open(RESULTS / "labeled_arguments.json", "r", encoding="utf-8") as f:
    data = json.load(f)

f4 = data["formula_4"]
f5 = data["formula_5"]
all_labeled = f4 + f5
all_cooc, all_counts = ld.build_co_occurrence_matrix(all_labeled)

out = RESULTS / "labeling_statistics.md"
ld.create_statistics_summary(f4, f5, all_counts, all_cooc, out)

print(f"[Okay] regenerated {out}")
print(f"f4 args: {len(f4)}, f5 args: {len(f5)}")
