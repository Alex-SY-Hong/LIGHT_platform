"""Fault-injection regression suite (task 6): mutation testing for the gates.

Each fault family corrupts a copy of clean data using a mechanism anchored
to a real historical prototype (never an invented one), then verifies the
matching gate intercepts it. Clean runs must stay green: 100% interception
with zero false positives is the acceptance bar (taskfile §2.6).

Families and gates:
- value-shift (+3 on a log10 column)   -> physical_plausibility (§3.6 rows)
- value-scale (x100 on a ratio column) -> physical_plausibility (SR=300)
- determinism (unseeded corruption)     -> reproducibility_check compare
- column-shuffle                       -> schema_check fingerprint
- row-shuffle (same count, wrong order)-> schema_check assert_join_alignment

Usage:
    python fault_injection.py --ym-csv ym.csv --sr-csv sr.csv \
        --features-csv feat.csv --workdir tmpdir
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

try:  # package context
    from .physical_plausibility import check_db
    from .reproducibility_check import compare_tables, read_table
    from .schema_check import assert_join_alignment, assert_predict_schema, stamp_schema
except ImportError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from physical_plausibility import check_db
    from reproducibility_check import compare_tables, read_table
    from schema_check import (
        assert_join_alignment,
        assert_predict_schema,
        stamp_schema,
    )

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
LOG_PATH = REPORTS_DIR / "fault_injection_log.jsonl"

YM_COL = "Young's Modulus (kPa) log10"
SR_COL = "Swelling Ratio (times)"


# ---------------------------------------------------------------- generators
def _rewrite(csv_path: Path, df: pd.DataFrame) -> None:
    df.to_csv(csv_path, index=False, encoding="utf-8")


def corrupt_shift_log10(csv_path: Path, column: str, delta: float = 3.0) -> None:
    """Add delta to every log10 value — the mixed-unit Pa/kPa mechanism."""
    df = read_table(Path(csv_path))
    df[column] = df[column].astype(float) + delta
    _rewrite(Path(csv_path), df)


def corrupt_sr_scale(csv_path: Path, factor: float = 100.0) -> None:
    """Multiply every ratio value by factor — scale-corruption mechanism."""
    df = read_table(Path(csv_path))
    df[SR_COL] = df[SR_COL].astype(float) * factor
    _rewrite(Path(csv_path), df)


def corrupt_unseeded(csv_path: Path, column: str) -> None:
    """Add a random.SystemRandom jitter (unseeded by construction)."""
    df = read_table(Path(csv_path))
    jitter = random.SystemRandom().uniform(-0.5, 0.5)
    df[column] = df[column].astype(float) + jitter
    _rewrite(Path(csv_path), df)


def shuffle_columns(csv_path: Path) -> None:
    """Permute column order (values untouched) — order-drift mechanism."""
    df = read_table(Path(csv_path))
    cols = list(df.columns)
    random.Random(1234).shuffle(cols)  # seeded: deterministic corruption
    _rewrite(Path(csv_path), df[cols])


def shuffle_rows(csv_path: Path, key: Optional[str] = None) -> Path:
    """Permute row order into a NEW file (counts preserved) — the
    14/630 misalignment mechanism. Returns the shuffled copy's path."""
    df = read_table(Path(csv_path))
    shuffled = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    out = Path(csv_path).with_name(
        Path(csv_path).stem + "_rowshuffled" + Path(csv_path).suffix
    )
    _rewrite(out, shuffled)
    return out


# ------------------------------------------------------------------- registry
@dataclass(frozen=True)
class FaultSpec:
    """One injectable fault and the gate expected to intercept it."""

    fault_id: str
    family: str
    prototype: str
    expected_gate: str


FAULTS: Tuple[FaultSpec, ...] = (
    FaultSpec(
        "ym-plus3-log10",
        "value-shift",
        "four mixed-unit YM rows (taskfile 3.6)",
        "physical_plausibility",
    ),
    FaultSpec(
        "sr-x100-scale",
        "value-scale",
        "known SR=300 row",
        "physical_plausibility",
    ),
    FaultSpec(
        "unseeded-jitter",
        "determinism",
        "historical double-run drift",
        "reproducibility_check",
    ),
    FaultSpec(
        "column-shuffle",
        "column-shuffle",
        "feature-order drift at predict time",
        "schema_check",
    ),
    FaultSpec(
        "row-shuffle",
        "row-shuffle",
        "kmeans 14/630 SMILES misalignment",
        "schema_check.join_alignment",
    ),
)


@dataclass
class MutationReport:
    """Suite outcome: interception score + clean-corpus false positives."""

    injected: List[str] = field(default_factory=list)
    intercepted: List[str] = field(default_factory=list)
    escaped: List[str] = field(default_factory=list)
    clean_false_positives: int = 0
    details: List[str] = field(default_factory=list)

    @property
    def mutation_score(self) -> float:
        return len(self.intercepted) / len(FAULTS) if FAULTS else 1.0

    def summary(self) -> str:
        lines = [
            f"[MUTATION] score={self.mutation_score:.0%} "
            f"({len(self.intercepted)}/{len(FAULTS)}) "
            f"clean_false_positives={self.clean_false_positives}"
        ]
        lines.extend(f"  escaped: {f}" for f in self.escaped)
        lines.extend(self.details)
        return "\n".join(lines)


def _intercepted(ok: bool, fault: FaultSpec, report: MutationReport, note: str) -> bool:
    report.injected.append(fault.fault_id)
    if ok:
        report.intercepted.append(fault.fault_id)
    else:
        report.escaped.append(fault.fault_id)
    report.details.append(
        f"  {fault.fault_id}: {'intercepted' if ok else 'ESCAPED'} ({note})"
    )
    return ok


def run_suite(
    ym_csv: Path,
    sr_csv: Path,
    features_csv: Path,
    workdir: Path,
) -> MutationReport:
    """Inject every fault family against copies; verify gate interception
    and that the clean corpus triggers nothing."""
    report = MutationReport()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    def _copy(src: Path, name: str) -> Path:
        dst = workdir / name
        shutil.copy2(src, dst)
        return dst

    # --- clean-corpus baseline: all gates silent on pristine copies
    clean_ym, clean_sr, clean_feat = (
        _copy(ym_csv, "clean_ym.csv"),
        _copy(sr_csv, "clean_sr.csv"),
        _copy(features_csv, "clean_feat.csv"),
    )
    schema = stamp_schema(clean_feat, workdir / "clean_schema.json")
    if check_db(clean_ym, "YM").findings:
        report.clean_false_positives += 1
        report.details.append("  clean YM fired (false positive)")
    if check_db(clean_sr, "SR").findings:
        report.clean_false_positives += 1
        report.details.append("  clean SR fired (false positive)")
    if not assert_predict_schema(clean_feat, schema).ok:
        report.clean_false_positives += 1
        report.details.append("  clean features schema-fired (false positive)")
    left = read_table(clean_feat)
    aligned = assert_join_alignment(clean_feat, clean_feat, key_col=left.columns[0])
    if not aligned.ok:
        report.clean_false_positives += 1
        report.details.append("  clean alignment fired (false positive)")

    # --- fault 1: +3 log10 shift on YM (gel rows must trip HIGH rule)
    f1 = FAULTS[0]
    ym_bad = _copy(ym_csv, "shift_ym.csv")
    corrupt_shift_log10(ym_bad, YM_COL, delta=3.0)
    caught = any(
        f.rule_id == "YM-LOG10KPA-HIGH" for f in check_db(ym_bad, "YM").findings
    )
    _intercepted(caught, f1, report, "YM-LOG10KPA-HIGH fired")

    # --- fault 2: SR x100 (mid-range row must cross 200 suspect bar)
    f2 = FAULTS[1]
    sr_bad = _copy(sr_csv, "scale_sr.csv")
    corrupt_sr_scale(sr_bad, factor=100.0)
    caught = any(
        f.rule_id == "SR-RATIO-SUPERABSORBENT"
        for f in check_db(sr_bad, "SR").findings
    )
    _intercepted(caught, f2, report, "SR-RATIO-SUPERABSORBENT fired")

    # --- fault 3: unseeded jitter -> double-run drift must be visible
    f3 = FAULTS[2]
    sr_rand = _copy(sr_csv, "rand_sr.csv")
    corrupt_unseeded(sr_rand, SR_COL)
    first = read_table(sr_rand)
    corrupt_unseeded(sr_rand, SR_COL)
    second = read_table(sr_rand)
    caught = compare_tables(first, second) != []
    _intercepted(caught, f3, report, "compare_tables saw drift")

    # --- fault 4: column shuffle -> schema fingerprint must fail
    f4 = FAULTS[3]
    feat_shuf = _copy(features_csv, "colshuf_feat.csv")
    shuffle_columns(feat_shuf)
    caught = not assert_predict_schema(feat_shuf, schema).ok
    _intercepted(caught, f4, report, "fingerprint mismatch")

    # --- fault 5: row shuffle -> key alignment must fail (counts equal)
    f5 = FAULTS[4]
    left_copy = _copy(features_csv, "align_left.csv")
    right_shuffled = shuffle_rows(left_copy)
    caught = not assert_join_alignment(left_copy, right_shuffled, key_col="Pair_ID").ok
    _intercepted(caught, f5, report, "key sequence mismatch")

    append_log(report)
    return report


def append_log(report: MutationReport) -> None:
    """Append one immutable record (P9): run evidence, not acceptance."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mutation_score": report.mutation_score,
        "injected": report.injected,
        "intercepted": report.intercepted,
        "escaped": report.escaped,
        "clean_false_positives": report.clean_false_positives,
        "note": "run evidence only, not acceptance (P4/P9)",
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ym-csv", type=Path, required=True)
    ap.add_argument("--sr-csv", type=Path, required=True)
    ap.add_argument("--features-csv", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, default=None)
    args = ap.parse_args()
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="fault-injection-"))

    report = run_suite(args.ym_csv, args.sr_csv, args.features_csv, workdir)
    print(report.summary())
    print(f"[LOG] appended -> {LOG_PATH}")
    ok = report.mutation_score == 1.0 and report.clean_false_positives == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
