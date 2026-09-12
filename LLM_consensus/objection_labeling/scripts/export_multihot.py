"""Export objection labels as multi-hot binary vectors.

Reads:    objection_labeling/results/labeled_arguments.json
Writes:   objection_labeling/results/multihot_vectors.json (dimension order + vectors)
          objection_labeling/results/multihot_vectors.csv  (rows = arguments, cols = 7 dims)

Standalone backfill tool; the main labeling pipeline (label_dimensions.py)
writes these tables natively on every run.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from objection_labeling.analysis.multihot import DIMENSIONS, labels_to_vector

BASE = Path(__file__).resolve().parent.parent / "results"


def main():
    with open(BASE / "labeled_arguments.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {"dimension_order": DIMENSIONS, "formula_4": [], "formula_5": []}
    csv_rows = []

    for formula in ("formula_4", "formula_5"):
        for entry in data[formula]:
            vector = labels_to_vector(entry["labels"])
            out[formula].append({"key": entry["key"], "vector": vector})
            csv_rows.append([formula, entry["key"]] + vector)

    with open(BASE / "multihot_vectors.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    with open(BASE / "multihot_vectors.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["formula", "key"] + DIMENSIONS)
        writer.writerows(csv_rows)

    print("[Okay] wrote multihot_vectors.json / .csv")
    for formula in ("formula_4", "formula_5"):
        n = len(out[formula])
        sums = [sum(r["vector"][i] for r in out[formula]) for i in range(len(DIMENSIONS))]
        src_counts = {d: 0 for d in DIMENSIONS}
        for entry in data[formula]:
            for lab in entry["labels"]:
                src_counts[lab] += 1
        match = all(src_counts[d] == s for d, s in zip(DIMENSIONS, sums))
        print(f"{formula}: n_args={n}, per-dim sums={dict(zip(DIMENSIONS, sums))}, matches_source={match}")


if __name__ == "__main__":
    main()
