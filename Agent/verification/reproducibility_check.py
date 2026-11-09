"""Reproducibility gate: run a script twice, compare outputs field by field.

Task 4 of the minimal-tier work package. Contract:
- runs ``cmd`` N times (default 2) in ``cwd``;
- after each run snapshots the declared artifact files;
- compares consecutive runs pairwise (CSV per-cell, JSON recursive,
  anything else byte-exact);
- appends an immutable, timestamped record to reports/run_log.jsonl (P9);
- exits non-zero on any FAIL. The gate never modifies artifacts.

Usage:
    python reproducibility_check.py \
        --cmd "python analyze_llm_reliability.py" \
        --artifact reliability_analysis_results.json --cwd ../LLM_consensus
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
RUN_LOG_PATH = REPORTS_DIR / "run_log.jsonl"
DEFAULT_JSON_FLOAT_TOL = 1e-12
RUN_TIMEOUT_SECONDS = 3600


@dataclass(frozen=True)
class Diff:
    """One field-level difference between two runs."""

    location: str
    run1: Any
    run2: Any
    kind: str = "value"  # "value" | "missing" | "shape"


@dataclass(frozen=True)
class ArtifactResult:
    """Comparison outcome for one artifact across one run pair."""

    artifact: str
    diffs: Tuple[Diff, ...]


@dataclass
class ReproResult:
    """Aggregate outcome of a double-run check."""

    cmd: str
    cwd: Optional[str]
    runs: int
    return_codes: List[int] = field(default_factory=list)
    artifacts: List[ArtifactResult] = field(default_factory=list)

    @property
    def status(self) -> str:
        commands_ok = all(rc == 0 for rc in self.return_codes)
        diffs_ok = all(not a.diffs for a in self.artifacts)
        return "PASS" if commands_ok and diffs_ok else "FAIL"

    def summary(self) -> str:
        n_diffs = sum(len(a.diffs) for a in self.artifacts)
        lines = [f"[{self.status}] cmd={self.cmd!r} runs={self.runs} rc={self.return_codes}"]
        for a in self.artifacts:
            head = "; ".join(
                f"{d.location}: {d.run1!r} != {d.run2!r}" for d in a.diffs[:5]
            )
            suffix = " ..." if len(a.diffs) > 5 else ""
            lines.append(f"  {a.artifact}: {len(a.diffs)} diff(s) {head}{suffix}")
        lines.append(f"  total diffs: {n_diffs}")
        return "\n".join(lines)


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV trying utf-8-sig first, gbk as fallback (repo convention)."""
    for enc in ("utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode CSV with utf-8 or gbk: {path}")


def _numeric_diffs(
    col: str, s1: pd.Series, s2: pd.Series, float_tol: float
) -> List[Diff]:
    """Field diffs for one numeric column (NaN == NaN counts as equal)."""
    both_nan = s1.isna() & s2.isna()
    if float_tol > 0:
        delta = (s1.astype(float) - s2.astype(float)).abs()
        bad = (delta > float_tol) & ~both_nan
    else:
        bad = (s1 != s2) & ~both_nan
    return [
        Diff(f"row {i}, col {col}", s1.iloc[i], s2.iloc[i])
        for i in np.flatnonzero(bad.to_numpy())
    ]


def _object_diffs(col: str, s1: pd.Series, s2: pd.Series) -> List[Diff]:
    """Field diffs for one non-numeric column."""
    bad = (s1 != s2) & ~(s1.isna() & s2.isna())
    return [
        Diff(f"row {i}, col {col}", s1.iloc[i], s2.iloc[i])
        for i in np.flatnonzero(bad.to_numpy())
    ]


def compare_tables(
    a: pd.DataFrame, b: pd.DataFrame, float_tol: float = 0.0
) -> List[Diff]:
    """Cell-by-cell comparison; exact match required by default."""
    if a.shape != b.shape:
        return [Diff("shape", a.shape, b.shape, kind="shape")]
    diffs: List[Diff] = []
    for col in a.columns:
        s1, s2 = a[col], b[col]
        if pd.api.types.is_numeric_dtype(s1.dtype) and pd.api.types.is_numeric_dtype(
            s2.dtype
        ):
            diffs.extend(_numeric_diffs(col, s1, s2, float_tol))
        else:
            diffs.extend(_object_diffs(col, s1, s2))
    return diffs


def compare_json(
    a: Any, b: Any, float_tol: float = DEFAULT_JSON_FLOAT_TOL, path: str = "$"
) -> List[Diff]:
    """Recursive comparison of JSON-like structures (dict/list/scalar)."""
    if isinstance(a, dict) and isinstance(b, dict):
        diffs: List[Diff] = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append(Diff(f"{path}.{k}", a.get(k), b.get(k), kind="missing"))
            else:
                diffs.extend(compare_json(a[k], b[k], float_tol, f"{path}.{k}"))
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [Diff(f"{path}[len]", len(a), len(b), kind="shape")]
        out: List[Diff] = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(compare_json(x, y, float_tol, f"{path}[{i}]"))
        return out
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if a == b else [Diff(path, a, b)]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > float_tol:
            return [Diff(path, a, b)]
        return []
    return [] if a == b else [Diff(path, a, b)]


def _run_command(cmd: str, cwd: Optional[Path]) -> int:
    """Execute one run; return exit code. Output streams are swallowed."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        shell=True,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        print(f"[WARN] rc={proc.returncode} stderr tail: {tail}", file=sys.stderr)
    return proc.returncode


def _compare_artifact(run1: Path, run2: Path, float_tol: float) -> Tuple[Diff, ...]:
    """Compare one artifact file between two run snapshots by suffix."""
    if run2.suffix.lower() == ".csv":
        return tuple(compare_tables(read_table(run1), read_table(run2), float_tol))
    if run2.suffix.lower() == ".json":
        with open(run1, encoding="utf-8") as f1, open(run2, encoding="utf-8") as f2:
            return tuple(
                compare_json(
                    json.load(f1), json.load(f2), max(float_tol, DEFAULT_JSON_FLOAT_TOL)
                )
            )
    return (
        ()
        if run1.read_bytes() == run2.read_bytes()
        else (Diff("bytes", "run1", "run2", kind="value"),)
    )


def check_reproducible(
    cmd: str,
    artifacts: Sequence[Path],
    cwd: Optional[Path] = None,
    runs: int = 2,
    float_tol: float = 0.0,
) -> ReproResult:
    """Run ``cmd`` ``runs`` times and compare artifacts between consecutive runs."""
    result = ReproResult(cmd=cmd, cwd=str(cwd) if cwd else None, runs=runs)
    with tempfile.TemporaryDirectory(prefix="repro-check-") as tmp:
        tmp_dir = Path(tmp)
        prev_snap: Optional[Path] = None
        for n in range(1, runs + 1):
            result.return_codes.append(_run_command(cmd, cwd))
            if n < runs:
                snap = tmp_dir / f"run{n}"
                snap.mkdir()
                for art in artifacts:
                    if Path(art).exists():
                        shutil.copy2(art, snap / Path(art).name)
                prev_snap = snap
                continue
            if prev_snap is None:
                continue
            for art in artifacts:
                a1, a2 = prev_snap / art.name, Path(art)
                if not a2.exists():
                    result.artifacts.append(
                        ArtifactResult(
                            str(art),
                            (Diff(str(art), "present", "missing", kind="missing"),),
                        )
                    )
                    continue
                diffs = _compare_artifact(a1, a2, float_tol)
                result.artifacts.append(ArtifactResult(str(art), diffs))
    append_run_log(result)
    return result


def append_run_log(result: ReproResult) -> None:
    """Append one immutable record (P9: append-only, UTC ISO-8601 timestamp)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cmd": result.cmd,
        "cwd": result.cwd,
        "runs": result.runs,
        "status": result.status,
        "n_diffs": sum(len(a.diffs) for a in result.artifacts),
        "diffs": [
            {"artifact": a.artifact, "items": [d.__dict__ for d in a.diffs[:20]]}
            for a in result.artifacts
            if a.diffs
        ],
    }
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cmd", required=True, help="Command to run twice")
    ap.add_argument(
        "--artifact",
        required=True,
        action="append",
        type=Path,
        help="Artifact file to compare (repeatable)",
    )
    ap.add_argument("--cwd", type=Path, default=None, help="Working directory")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--float-tol", type=float, default=0.0, help="CSV float tolerance")
    args = ap.parse_args()

    result = check_reproducible(
        cmd=args.cmd,
        artifacts=args.artifact,
        cwd=args.cwd,
        runs=args.runs,
        float_tol=args.float_tol,
    )
    print(result.summary())
    print(f"[LOG] appended -> {RUN_LOG_PATH}")
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
