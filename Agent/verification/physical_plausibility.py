"""Physical plausibility gate: first-principles range checks for YM and SR.

Task 3 of the minimal-tier work package. Symmetric rule structure over the
two property columns (P4); every threshold is externally anchored (P8) or
explicitly provisional. The gate NEVER modifies the database: Tier A
findings emit an isolation-review recommendation, Tier B findings are
report-only (failure -> more human review, never more automation).

First-run results against the current DB go to the append-only run log
only (P9) -- they are run evidence, not acceptance evidence (P4).

Usage:
    python physical_plausibility.py --csv youngs_modulus.csv --target YM
    python physical_plausibility.py --csv swelling_ratio.csv --target SR
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

try:  # package context (Agent.verification.physical_plausibility)
    from .reproducibility_check import read_table
except ImportError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reproducibility_check import read_table

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
LOG_PATH = REPORTS_DIR / "plausibility_log.jsonl"

TARGET_COLUMNS = {
    "YM": "Young's Modulus (kPa) log10",
    "SR": "Swelling Ratio (times)",
}

# Anchors (P8): externally published ranges, never fitted to our own data.
DOI_KPA_MPA_TUNABLE = "10.1021/acs.iecr.9b06769"  # hydrogel E tuned kPa -> MPa
DOI_UNIT_APP = "10.1002/app.33461"  # unit-mismatch evidence (taskfile)
DOI_UNIT_JBM = "10.1002/jbm.a.37336"  # unit-mismatch evidence (taskfile)
DOI_EQUIL_SWELLING = "10.1007/s42558-020-00032-5"  # equilibrium swelling model
DOI_SUPERABSORBENT = "10.1063/1.4995098"  # superabsorbent swelling capacities

# The YM database is a mixed polymer corpus: plastics/fibers legitimately
# reach GPa in log10(kPa); the gel thresholds apply ONLY to gel-class
# rows, identified by source keywords. Calibration evidence: the four
# rows this scope flags are exactly the four unit-mismatch rows
# documented for collaborators (rows 0/1045/1534/1536, 0-based).
GEL_SOURCE_PATTERN = r"gel|hydrogel|GelMA|SilMA|凝胶|水凝胶"


@dataclass(frozen=True)
class Rule:
    """One externally-anchored plausibility rule over a property column."""

    rule_id: str
    target: str  # "YM" | "SR"
    tier: str  # "domain" | "flag" | "suspicious"
    op: str  # "gt" | "lt"
    threshold: float
    rationale: str
    anchors: Tuple[str, ...] = ()
    provisional: bool = False
    source_scope: Optional[str] = None  # regex over Source; None = all rows

    @property
    def disposition(self) -> str:
        if self.tier in ("domain", "flag"):
            return "TierA-isolation-review"
        return "TierB-report-only"

    def violated_by(self, v: float) -> bool:
        if self.op == "gt":
            return v > self.threshold
        return v < self.threshold


RULES: Tuple[Rule, ...] = (
    Rule(
        rule_id="YM-LOG10KPA-HIGH",
        target="YM",
        tier="flag",
        op="gt",
        threshold=4.0,
        rationale=(
            "Gel-class row with log10(kPa) > 4 means E > 10 MPa -- above "
            "the documented range of typical hydrogels; also the signature "
            "of log10(Pa) values mixed into a log10(kPa) column (+3). "
            "Plastics/fibers in this corpus legitimately reach GPa and "
            "are out of scope."
        ),
        anchors=(DOI_KPA_MPA_TUNABLE, DOI_UNIT_APP, DOI_UNIT_JBM),
        source_scope=GEL_SOURCE_PATTERN,
    ),
    Rule(
        rule_id="YM-LOG10KPA-EXTREME",
        target="YM",
        tier="suspicious",
        op="gt",
        threshold=5.5,
        rationale=(
            "Gel-class row with log10(kPa) > 5.5 means E > ~316 MPa -- "
            "beyond hydrogel literature extremes; verify unit and source "
            "before any use. Plastics out of scope."
        ),
        anchors=(DOI_KPA_MPA_TUNABLE, DOI_UNIT_APP),
        source_scope=GEL_SOURCE_PATTERN,
    ),
    Rule(
        rule_id="SR-RATIO-BELOW-ONE",
        target="SR",
        tier="domain",
        op="lt",
        threshold=1.0,
        rationale=(
            "Swelling ratio is defined as swollen/dry mass (or volume) and "
            "is >= 1 for a swelling measurement; < 1 indicates deswelling "
            "conventions, extraction error, or placeholder zeros."
        ),
        anchors=(DOI_EQUIL_SWELLING,),
    ),
    Rule(
        rule_id="SR-RATIO-SUPERABSORBENT",
        target="SR",
        tier="suspicious",
        op="gt",
        threshold=200.0,
        rationale=(
            "SR > 200 enters superabsorbent territory (documented 100-1000+); "
            "possible for SAP chemistries but atypical for the screened "
            "classes -- verify source."
        ),
        anchors=(DOI_SUPERABSORBENT, DOI_EQUIL_SWELLING),
    ),
)


@dataclass(frozen=True)
class Finding:
    """One row-level plausibility violation."""

    row_index: int
    value: float
    rule_id: str
    tier: str
    disposition: str
    rationale: str


@dataclass
class PlausibilityReport:
    """Findings for one DB scan. Recommendations only -- never edits."""

    csv: str
    target: str
    n_rows: int
    findings: List[Finding] = field(default_factory=list)

    def disposition_of(self, row_index: int) -> Optional[str]:
        hit = [f for f in self.findings if f.row_index == row_index]
        if not hit:
            return None
        if any(f.disposition == "TierA-isolation-review" for f in hit):
            return "TierA-isolation-review"
        return "TierB-report-only"

    def summary(self) -> str:
        a = [f for f in self.findings if f.disposition == "TierA-isolation-review"]
        b = [f for f in self.findings if f.disposition == "TierB-report-only"]
        lines = [
            f"[PLAUSIBILITY] {self.target} n={self.n_rows} "
            f"TierA={len(a)} TierB={len(b)} (gate is read-only)"
        ]
        for label, group in (("TierA", a), ("TierB", b)):
            for f in group[:10]:
                lines.append(
                    f"  {label} row {f.row_index}: value={f.value!r} "
                    f"rule={f.rule_id}"
                )
            if len(group) > 10:
                lines.append(f"  {label} ... {len(group) - 10} more")
        return "\n".join(lines)


def evaluate_value(target: str, v: float) -> List[Finding]:
    """Apply all rules for one value. NaN yields an invalid-value finding."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return [
            Finding(
                row_index=-1,
                value=float("nan"),
                rule_id=f"{target}-INVALID-VALUE",
                tier="invalid",
                disposition="TierA-isolation-review",
                rationale="Non-numeric or missing property value.",
            )
        ]
    out: List[Finding] = []
    for r in RULES:
        if r.target == target and r.violated_by(v):
            out.append(
                Finding(
                    row_index=-1,
                    value=v,
                    rule_id=r.rule_id,
                    tier=r.tier,
                    disposition=r.disposition,
                    rationale=r.rationale,
                )
            )
    return out


def check_db(csv_path: Path, target: str) -> PlausibilityReport:
    """Scan one property DB; findings to console + append-only log (P9)."""
    if target not in TARGET_COLUMNS:
        raise ValueError(f"target must be YM or SR, got {target!r}")
    column = TARGET_COLUMNS[target]
    df = read_table(Path(csv_path))
    if column not in df.columns:
        raise ValueError(f"column {column!r} not in {csv_path}")

    report = PlausibilityReport(csv=str(csv_path), target=target, n_rows=len(df))
    source_col = df["Source"].astype(str) if "Source" in df.columns else None
    for i, raw in enumerate(df[column]):
        value = float(raw) if pd.notna(raw) else math.nan
        row_source = source_col.iloc[i] if source_col is not None else ""
        if math.isnan(value):
            report.findings.append(
                Finding(
                    row_index=i,
                    value=value,
                    rule_id=f"{target}-INVALID-VALUE",
                    tier="invalid",
                    disposition="TierA-isolation-review",
                    rationale="Non-numeric or missing property value.",
                )
            )
            continue
        for rule in RULES:
            if rule.target != target or not rule.violated_by(value):
                continue
            if rule.source_scope is not None and not re.search(
                rule.source_scope, row_source, re.IGNORECASE
            ):
                continue
            report.findings.append(
                Finding(
                    row_index=i,
                    value=value,
                    rule_id=rule.rule_id,
                    tier=rule.tier,
                    disposition=rule.disposition,
                    rationale=rule.rationale,
                )
            )
    append_log(report)
    return report


def append_log(report: PlausibilityReport) -> None:
    """Append one immutable record (P9). First-run evidence, not acceptance."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "csv": report.csv,
        "target": report.target,
        "n_rows": report.n_rows,
        "n_findings": len(report.findings),
        "note": "run evidence only, not acceptance (P4/P9)",
        "findings": [
            {
                "row_index": f.row_index,
                "value": None if (isinstance(f.value, float) and math.isnan(f.value)) else f.value,
                "rule_id": f.rule_id,
                "tier": f.tier,
                "disposition": f.disposition,
            }
            for f in report.findings[:200]
        ],
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--target", choices=("YM", "SR"), required=True)
    args = ap.parse_args()

    report = check_db(args.csv, args.target)
    print(report.summary())
    print(f"[LOG] appended -> {LOG_PATH}")
    return 0  # findings are evidence, not gate failure


if __name__ == "__main__":
    sys.exit(main())
