"""Audit diff between two labeling runs (e.g. different labeler models).

Prints entries whose label sets differ, with full argument text and
old/new labels, plus a total changed count. Serves as the first step
of the human adjudication workflow for inter-rater reliability.

Configure the two run directories below. Each expects a
labeled_arguments.json as written by label_dimensions.py.
"""
import json
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent.parent
RUN_A = MODULE_DIR / "results" / "labeled_arguments.json"                 # canonical
RUN_B = MODULE_DIR / "results_flashlite" / "labeled_arguments.json"       # comparison run


def main():
    old = json.load(open(RUN_A, encoding="utf-8"))
    new = json.load(open(RUN_B, encoding="utf-8"))

    total_changed = 0
    for formula in ("formula_4", "formula_5"):
        o, n = old[formula], new[formula]
        print(f"\n{'='*70}\n{formula}: {len(o)} vs {len(n)} entries")
        assert len(o) == len(n), "count mismatch"
        for i, (eo, en) in enumerate(zip(o, n)):
            assert eo["key"] == en["key"], f"order mismatch at {i}: {eo['key']} vs {en['key']}"
            if set(eo["labels"]) != set(en["labels"]):
                total_changed += 1
                arg = eo["argument"].replace("\n", " ")
                print(f"\n--- [{i}] {eo['key']}")
                print(f"    text : {arg}")
                print(f"    A    : {', '.join(eo['labels'])}")
                print(f"    B    : {', '.join(en['labels'])}")

    print(f"\n{'='*70}\nTotal changed: {total_changed}")


if __name__ == "__main__":
    main()
