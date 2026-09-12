"""Objection labeling module.

LLM-assisted multi-label annotation of anti-formula objections for
Formulas 4 and 5, with multi-hot vector encoding, co-occurrence
statistics, and Nature-style visualization.

Convenience API and scripts:
- scripts/label_dimensions.py       main labeling pipeline (LLM, .env credentials)
- scripts/export_multihot.py        multi-hot JSON/CSV export from labeled data
- scripts/regen_stats.py            regenerate labeling_statistics.md (no API calls)
- scripts/plot_objection_panels.py  two-panel objection distribution figure
- scripts/compare_multihot_runs.py  audit diff between two labeling runs
"""

import json
from pathlib import Path

from objection_labeling.analysis.multihot import (
    DIMENSIONS,
    labels_to_vector,
    vector_to_labels,
)

__version__ = "1.0.0"

__all__ = [
    "DIMENSIONS",
    "labels_to_vector",
    "vector_to_labels",
    "load_labeled_arguments",
]

_RESULTS_PATH = Path(__file__).parent / "results" / "labeled_arguments.json"


def load_labeled_arguments(path=None):
    """Load labeled objections as {formula: [{key, argument, labels, vector}]}.

    Each entry's ``vector`` is a 7-dim multi-hot encoding of ``labels``
    following DIMENSIONS order.
    """
    path = Path(path) if path is not None else _RESULTS_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
