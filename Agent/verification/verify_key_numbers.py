"""Verify the paper's key numbers against tracked artifacts (task 9).

Checks 1-4 run in the LLM_consensus poetry environment, zero API:

1. YM regression pooled OOF R^2        — recomputed from the archived
   OOF csv (pure arithmetic, no sklearn) and matched to the paper claim
   at the claim's stated precision (3 decimals).
2. Classification pooled OOF Acc / F1  — same policy.
3. Consensus reliability numbers        — regenerate via the pipeline's
   declared analyzer entry and compare field-exact to the archived
   reliability_analysis_results.json; the regenerated output must carry
   the full analyzer's signature (winner_margin).
4. Popularity-bias p-values             — seeded double run, byte-equal
   summaries; p-value COUNT is derived, never hardcoded.

Design rule (see decision registry): expected VALUES are paper claims
and may be pinned; structural IDENTITIES (entry point, counts, fold
number) must be derived from artifacts and records. Tolerance tiers are
separate: archived-recompute matches at rounding precision; the
+-0.005 budget belongs only to a future retrain cross-check, which this
script does not perform.

Checks 5-7 (feature regen, clustering, deployment predict) need the
unsupervised/supervised environments and are added separately.

Usage:
    python verify_key_numbers.py --check all [--repo /path/to/repo]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:  # package context
    from .reproducibility_check import compare_json, read_table
except ImportError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reproducibility_check import compare_json, read_table

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
LOG_PATH = REPORTS_DIR / "verify_key_numbers_log.jsonl"

REPO_ROOT = Path(__file__).resolve().parents[2]
YM_OOF_CSV = REPO_ROOT / (
    "Supervised_Learning/Regression_Model/results/YoungsModulus/rf_cv10/cv10_oof.csv"
)
CLS_OOF_CSV = REPO_ROOT / (
    "Supervised_Learning/Classification_Model/results/SwellingRatio/"
    "rf_cls_cv10_t9/cv10_oof.csv"
)
LLM_CONSENSUS = REPO_ROOT / "LLM_consensus"
ARCHIVED_CONSENSUS = LLM_CONSENSUS / "reliability_analysis_results.json"
PIPELINE_DECLARATION = LLM_CONSENSUS / "run_pipeline.py"
CONSENSUS_ENTRY = "reliability_analysis.analyze_reliability"
POPULARITY_SCRIPT = LLM_CONSENSUS / "popularity_bias/scripts/analyze_rigorous_v2.py"
POPULARITY_SUMMARY = (
    LLM_CONSENSUS / "popularity_bias/results/rigorous_analysis_v2_summary.json"
)

# Paper claims (pinned VALUES — the objects under verification).
CLAIM_YM_R2 = 0.528
CLAIM_CLS_ACC = 0.719
CLAIM_CLS_F1 = 0.591
CLAIM_DECIMALS = 3
RETRAIN_BUDGET_TOL = 0.005  # documented tier, NOT used by checks 1-2


@dataclass
class CheckResult:
    """Outcome of one key-number check."""

    name: str
    ok: bool
    detail: Dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        flag = "PASS" if self.ok else "FAIL"
        return f"[{flag}] {self.name}: {json.dumps(self.detail, default=str)}"


# ---------------------------------------------------------------- metrics
def regression_oof_metrics(oof_csv: Path) -> Dict[str, Any]:
    """Pooled OOF R^2 from the archived prediction table (pure numpy)."""
    df = read_table(Path(oof_csv))
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:  # degenerate constant target: perfect or not
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return {"n": int(len(df)), "r2": r2}


def classification_oof_metrics(oof_csv: Path) -> Dict[str, Any]:
    """Pooled OOF accuracy and F1 for the positive class (pure numpy)."""
    df = read_table(Path(oof_csv))
    y_true = df["y_true"].to_numpy(dtype=int)
    y_pred = df["y_pred"].to_numpy(dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = float(np.mean(y_true == y_pred))
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom else 0.0
    return {"n": int(len(df)), "accuracy": acc, "f1_positive": f1}


def verify_round_to_claim(value: float, claim: float, ndigits: int) -> bool:
    """True when the recomputed value rounds to the paper claim."""
    half_ulp = 0.5 * 10 ** (-ndigits)
    return abs(value - claim) <= half_ulp + 1e-12


_P_KEY = re.compile(r"(_p$|^p[_ ]?val)", re.IGNORECASE)  # p_value, p_val, *_p


def count_p_values(node: Any) -> int:
    """Count p-value entries by walking the structure (derived identity)."""
    if isinstance(node, dict):
        total = sum(1 for k in node if _P_KEY.search(str(k)))
        return total + sum(count_p_values(v) for v in node.values())
    if isinstance(node, list):
        return sum(count_p_values(v) for v in node)
    return 0


# ----------------------------------------------------------------- checks
def _subprocess_ok(cmd: List[str], cwd: Path) -> bool:
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        print(
            f"[WARN] {' '.join(cmd)} rc={proc.returncode} tail={tail}",
            file=sys.stderr,
        )
    return proc.returncode == 0


def check_1_ym_oof(oof_csv: Path = YM_OOF_CSV) -> CheckResult:
    m = regression_oof_metrics(oof_csv)
    ok = verify_round_to_claim(m["r2"], CLAIM_YM_R2, CLAIM_DECIMALS)
    return CheckResult(
        "1_ym_oof_r2",
        ok,
        {
            "recomputed_r2": round(m["r2"], 6),
            "claim": CLAIM_YM_R2,
            "n": m["n"],
            "tolerance": f"rounds to {CLAIM_DECIMALS} decimals "
            f"(retrain budget {RETRAIN_BUDGET_TOL} is a separate tier)",
        },
    )


def check_2_cls_oof(oof_csv: Path = CLS_OOF_CSV) -> CheckResult:
    m = classification_oof_metrics(oof_csv)
    ok = verify_round_to_claim(m["accuracy"], CLAIM_CLS_ACC, CLAIM_DECIMALS) and (
        verify_round_to_claim(m["f1_positive"], CLAIM_CLS_F1, CLAIM_DECIMALS)
    )
    return CheckResult(
        "2_cls_oof_acc_f1",
        ok,
        {
            "recomputed_acc": round(m["accuracy"], 6),
            "recomputed_f1": round(m["f1_positive"], 6),
            "claims": [CLAIM_CLS_ACC, CLAIM_CLS_F1],
            "n": m["n"],
        },
    )


def check_3_consensus() -> CheckResult:
    """Regenerate via the pipeline-declared entry; compare field-exact."""
    declared = PIPELINE_DECLARATION.read_text(encoding="utf-8", errors="replace")
    if CONSENSUS_ENTRY not in declared:
        return CheckResult(
            "3_consensus_regeneration",
            False,
            {"error": f"entry {CONSENSUS_ENTRY!r} not declared in run_pipeline.py"},
        )
    with tempfile.TemporaryDirectory(prefix="vk-consensus-") as tmp:
        snapshot = Path(tmp) / "archived.json"
        shutil.copy2(ARCHIVED_CONSENSUS, snapshot)
        if not _subprocess_ok(
            [sys.executable, "-m", CONSENSUS_ENTRY], cwd=LLM_CONSENSUS
        ):
            return CheckResult(
                "3_consensus_regeneration", False, {"error": "regeneration rc != 0"}
            )
        with open(snapshot, encoding="utf-8") as f:
            archived = json.load(f)
        with open(ARCHIVED_CONSENSUS, encoding="utf-8") as f:
            regenerated = json.load(f)
        diffs = compare_json(regenerated, archived, float_tol=0.0)

        def _has_margin(doc: Dict[str, Any]) -> bool:
            # Top level is a set of model entries (structure derived, not
            # assumed via a wrapper key): every model carries the full
            # analyzer signature.
            entries = [v for v in doc.values() if isinstance(v, dict)]
            return bool(entries) and all("winner_margin" in e for e in entries)

        margin = _has_margin(archived) and _has_margin(regenerated)
    ok = not diffs and margin
    return CheckResult(
        "3_consensus_regeneration",
        ok,
        {
            "n_field_diffs": len(diffs),
            "first_diffs": [
                f"{d.location}: {d.run1!r} != {d.run2!r}" for d in diffs[:5]
            ],
            "winner_margin_present": margin,
            "entry": CONSENSUS_ENTRY,
        },
    )


def check_4_popularity() -> CheckResult:
    """Seeded double run; byte-equal summaries; p-count derived."""
    with tempfile.TemporaryDirectory(prefix="vk-pop-") as tmp:
        if not _subprocess_ok(
            [sys.executable, str(POPULARITY_SCRIPT)], cwd=LLM_CONSENSUS
        ):
            return CheckResult("4_popularity_seeded", False, {"error": "run1 rc != 0"})
        run1_bytes = POPULARITY_SUMMARY.read_bytes()
        run1_json = json.loads(run1_bytes.decode("utf-8"))
        if not _subprocess_ok(
            [sys.executable, str(POPULARITY_SCRIPT)], cwd=LLM_CONSENSUS
        ):
            return CheckResult("4_popularity_seeded", False, {"error": "run2 rc != 0"})
        run2_bytes = POPULARITY_SUMMARY.read_bytes()
        run2_json = json.loads(run2_bytes.decode("utf-8"))
    byte_equal = run1_bytes == run2_bytes
    c1, c2 = count_p_values(run1_json), count_p_values(run2_json)
    ok = byte_equal and c1 == c2 and c1 > 0
    return CheckResult(
        "4_popularity_seeded",
        ok,
        {
            "byte_identical": byte_equal,
            "p_value_count_run1": c1,
            "p_value_count_run2": c2,
        },
    )


CHECKS = {
    "1": check_1_ym_oof,
    "2": check_2_cls_oof,
    "3": check_3_consensus,
    "4": check_4_popularity,
}


def append_log(results: List[CheckResult]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": [{"name": r.name, "ok": r.ok, "detail": r.detail} for r in results],
        "all_ok": all(r.ok for r in results),
        "note": "run evidence only, not acceptance",
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", choices=[*CHECKS, "all"], default="all")
    args = ap.parse_args()

    selected = list(CHECKS.values()) if args.check == "all" else [CHECKS[args.check]]
    results = [fn() for fn in selected]
    for r in results:
        print(r.line())
    append_log(results)
    print(f"[LOG] appended -> {LOG_PATH}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
