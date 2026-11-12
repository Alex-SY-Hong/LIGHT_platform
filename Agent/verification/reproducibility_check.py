"""Reproducibility gate: run a script repeatedly, compare outputs field by field.

Task 4 of the minimal-tier work package. Contract:
- runs ``cmd`` N times (N >= 2) in ``cwd``;
- snapshots the declared artifact files after EVERY run;
- compares every consecutive run pair (CSV per-cell, JSON recursive,
  anything else byte-exact); any pair difference is a FAIL;
- appends an immutable, timestamped record to reports/run_log.jsonl
  even when the check itself crashes;
- exits non-zero on any FAIL. The gate never modifies artifacts.

Comparison policy: exact match by default. NaN == NaN passes; NaN vs any
value fails; bool vs int fails (type drift); int vs float passes if
numerically equal (documented policy).

Usage:
    python reproducibility_check.py \
        --cmd "python analyze_llm_reliability.py" \
        --artifact reliability_analysis_results.json --cwd ../LLM_consensus
"""

from __future__ import annotations

import argparse
import json
import math
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
    kind: str = "value"  # "value" | "missing" | "shape" | "error"


@dataclass(frozen=True)
class ArtifactResult:
    """Comparison outcome for one artifact (worst pair collapsed)."""

    artifact: str
    diffs: Tuple[Diff, ...]


@dataclass
class ReproResult:
    """Aggregate outcome of a repeat-run check."""

    cmd: str
    cwd: Optional[str]
    runs: int
    float_tol: float = 0.0
    artifacts_resolved: List[str] = field(default_factory=list)
    return_codes: List[int] = field(default_factory=list)
    artifacts: List[ArtifactResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        commands_ok = all(rc == 0 for rc in self.return_codes) and not self.errors
        compared_ok = self.artifacts and not any(a.diffs for a in self.artifacts)
        return "PASS" if commands_ok and compared_ok else "FAIL"

    def summary(self) -> str:
        n_diffs = sum(len(a.diffs) for a in self.artifacts)
        lines = [
            f"[{self.status}] cmd={self.cmd!r} runs={self.runs} "
            f"rc={self.return_codes} errors={self.errors or 'none'}"
        ]
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
    """Field diffs for one numeric column. NaN==NaN equal, one-side-NaN fails."""
    both_nan = s1.isna() & s2.isna()
    one_nan = s1.isna() ^ s2.isna()
    if float_tol > 0:
        delta = (s1.astype(float) - s2.astype(float)).abs()
        bad = one_nan | (~both_nan & (delta > float_tol))
    else:
        bad = (s1 != s2) & ~both_nan  # s1 != s2 already flags one-side NaN
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
    """Cell-by-cell comparison. Column names/order must match exactly."""
    if list(a.columns) != list(b.columns):
        return [Diff("columns", list(a.columns), list(b.columns), kind="shape")]
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
    """Recursive comparison of JSON-like structures (dict/list/scalar).

    NaN==NaN and inf==inf pass; NaN vs value fails; bool vs int fails
    (type-checked); int vs float passes when numerically equal.
    """
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
        ok = type(a) is type(b) and a == b
        return [] if ok else [Diff(path, a, b)]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) or math.isnan(b):
            return [] if (math.isnan(a) and math.isnan(b)) else [Diff(path, a, b)]
        if math.isinf(a) or math.isinf(b):
            return [] if a == b else [Diff(path, a, b)]
        if abs(a - b) > float_tol:
            return [Diff(path, a, b)]
        return []
    return [] if a == b else [Diff(path, a, b)]


def _run_command(cmd: str, cwd: Optional[Path]) -> int:
    """Execute one run; return exit code. Output is discarded (utf-8 safe)."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        shell=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=RUN_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        print(f"[WARN] rc={proc.returncode} stderr tail: {tail}", file=sys.stderr)
    return proc.returncode


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8-sig") as f:  # BOM-tolerant
        return json.load(f)


def _compare_artifact(run1: Path, run2: Path, float_tol: float) -> Tuple[Diff, ...]:
    """Compare one artifact file between two run snapshots by suffix."""
    if run2.suffix.lower() == ".csv":
        return tuple(compare_tables(read_table(run1), read_table(run2), float_tol))
    if run2.suffix.lower() == ".json":
        return tuple(compare_json(_load_json(run1), _load_json(run2), float_tol))
    return (
        ()
        if run1.read_bytes() == run2.read_bytes()
        else (Diff("bytes", "run1", "run2", kind="value"),)
    )


def _resolve(art: Path, cwd: Optional[Path]) -> Path:
    """Artifact paths resolve against --cwd, never the gate's own cwd."""
    return art if art.is_absolute() else (Path(cwd) if cwd else Path.cwd()) / art


def _snapshot(artifacts: Sequence[Path], snap_dir: Path) -> List[Optional[Path]]:
    """Copy artifacts into snap_dir index-keyed; None where absent."""
    copied: List[Optional[Path]] = []
    snap_dir.mkdir(parents=True, exist_ok=True)
    for i, art in enumerate(artifacts):
        if art.exists():
            dest = snap_dir / f"{i:03d}_{art.name}"
            shutil.copy2(art, dest)
            copied.append(dest)
        else:
            copied.append(None)
    return copied


def check_reproducible(
    cmd: str,
    artifacts: Sequence[Path],
    cwd: Optional[Path] = None,
    runs: int = 2,
    float_tol: float = 0.0,
) -> ReproResult:
    """Run ``cmd`` ``runs`` times, compare every consecutive run pair."""
    if runs < 2:
        raise ValueError(f"runs must be >= 2, got {runs}")
    artifacts = [Path(a) for a in artifacts]
    if not artifacts:
        raise ValueError("at least one --artifact is required")
    cwd_path = Path(cwd) if cwd else None
    resolved = [_resolve(a, cwd_path) for a in artifacts]

    result = ReproResult(
        cmd=cmd,
        cwd=str(cwd_path) if cwd_path else None,
        runs=runs,
        float_tol=float_tol,
        artifacts_resolved=[str(p) for p in resolved],
    )
    per_artifact: List[List[Diff]] = [[] for _ in resolved]
    try:
        with tempfile.TemporaryDirectory(prefix="repro-check-") as tmp:
            tmp_dir = Path(tmp)
            prev: Optional[List[Optional[Path]]] = None
            for n in range(1, runs + 1):
                result.return_codes.append(_run_command(cmd, cwd_path))
                snap = _snapshot(resolved, tmp_dir / f"run{n}")
                if prev is not None:
                    for i, (p_old, p_new) in enumerate(zip(prev, snap)):
                        if p_old is None and p_new is None:
                            continue
                        if p_old is None or p_new is None:
                            state_old = "present" if p_old else "missing"
                            state_new = "present" if p_new else "missing"
                            per_artifact[i].append(
                                Diff(
                                    f"run{n-1}->run{n}",
                                    state_old,
                                    state_new,
                                    kind="missing",
                                )
                            )
                            continue
                        per_artifact[i].extend(
                            _compare_artifact(p_old, p_new, float_tol)
                        )
                prev = snap
    except Exception as exc:  # crash cluster guard: never bypass the log
        result.errors.append(f"{type(exc).__name__}: {exc}")

    for art, diffs in zip(resolved, per_artifact):
        result.artifacts.append(ArtifactResult(str(art), tuple(diffs)))
    append_run_log(result)
    return result


def _jsonable(v: Any) -> Any:
    """Normalize scalars for the JSON log (np types -> python)."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (np.floating, float)) and math.isnan(float(v)):
        return "NaN"
    return v


def append_run_log(result: ReproResult) -> None:
    """Append one immutable record (append-only, UTC ISO-8601 timestamp)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cmd": result.cmd,
        "cwd": result.cwd,
        "runs": result.runs,
        "float_tol": result.float_tol,
        "status": result.status,
        "return_codes": result.return_codes,
        "artifacts": result.artifacts_resolved,
        "n_diffs": sum(len(a.diffs) for a in result.artifacts),
        "errors": result.errors,
        "diffs": [
            {
                "artifact": a.artifact,
                "items": [
                    {k: _jsonable(v) for k, v in d.__dict__.items()}
                    for d in a.diffs[:20]
                ],
            }
            for a in result.artifacts
            if a.diffs
        ],
    }
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cmd", required=True, help="Command to run repeatedly")
    ap.add_argument(
        "--artifact",
        required=True,
        action="append",
        type=Path,
        help="Artifact file to compare, resolved against --cwd (repeatable)",
    )
    ap.add_argument("--cwd", type=Path, default=None, help="Working directory")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--float-tol", type=float, default=0.0, help="Float tolerance")
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
