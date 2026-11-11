"""Schema gate: fingerprint the feature columns a model was trained on,
and refuse prediction inputs that do not match (task 5).

Replaces predict.py's silent pad-zero/truncate behaviour with a hard
contract: ordered column names are hashed at training time (``stamp``)
and asserted before every prediction (``check``). Optional join row-count
assertion guards two-stage merges (history: 14/630 SMILES misalignment).

Usage:
    python schema_check.py stamp  --feature-csv train_features.csv \
        [--model-payload best_model.joblib] --out-json schema.json
    python schema_check.py check  --feature-csv candidates.csv \
        --schema-json schema.json
    python schema_check.py check-join --csv a.csv --csv b.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:  # package context (Agent.verification.schema_check)
    from .reproducibility_check import read_table
except ImportError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reproducibility_check import read_table

FEATURE_PATTERNS = ("fp_", "morgan_", "ecfp_", "desc_", "frag_", "idx_", "pair_")


@dataclass(frozen=True)
class SchemaFingerprint:
    """Ordered feature-column contract for one trained model."""

    n_features: int
    columns: Tuple[str, ...]
    columns_sha256: str


@dataclass(frozen=True)
class SchemaCheckResult:
    """Outcome of one assert call; ok=False must fail loudly upstream."""

    ok: bool
    first_mismatch: str = ""
    details: Tuple[str, ...] = ()


def _sha256_of_columns(columns: Sequence[str]) -> str:
    joined = "\n".join(columns)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def compute_schema_fingerprint(df: pd.DataFrame) -> SchemaFingerprint:
    """Hash the ordered feature columns of a dataframe (values ignored)."""
    cols = tuple(str(c) for c in df.columns)
    return SchemaFingerprint(
        n_features=len(cols), columns=cols, columns_sha256=_sha256_of_columns(cols)
    )


def _feature_columns_of(df: pd.DataFrame) -> List[str]:
    """Columns matching the repo's feature prefixes, in file order."""
    return [c for c in df.columns if str(c).startswith(FEATURE_PATTERNS)]


def stamp_schema(
    feature_csv: Optional[Path] = None,
    out_json: Path = Path("schema.json"),
    model_payload: Optional[Path] = None,
) -> Path:
    """Write the training-side schema fingerprint JSON. Returns its path."""
    source: str
    columns: List[str]
    if model_payload is not None:
        import joblib

        payload = joblib.load(model_payload)
        features = payload.get("features") if isinstance(payload, dict) else None
        if features is None:
            raise ValueError(f"Payload has no 'features' list: {model_payload}")
        columns, source = [str(c) for c in features], str(model_payload)
    elif feature_csv is not None:
        df = read_table(Path(feature_csv))
        cols = _feature_columns_of(df) or [str(c) for c in df.columns]
        columns, source = cols, str(feature_csv)
    else:
        raise ValueError("Provide --feature-csv or --model-payload")

    record: Dict[str, Any] = {
        "n_features": len(columns),
        "columns": columns,
        "columns_sha256": _sha256_of_columns(columns),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"[STAMP] {len(columns)} columns -> {out_json}")
    return out_json


def assert_predict_schema(feature_csv: Path, schema_json: Path) -> SchemaCheckResult:
    """Hard gate: incoming feature columns must equal the stamped contract
    in set AND order. Never pads, never truncates."""
    with open(schema_json, encoding="utf-8") as f:
        stamped = json.load(f)
    expected: List[str] = list(stamped["columns"])

    df = read_table(Path(feature_csv))
    actual = _feature_columns_of(df) or [str(c) for c in df.columns]

    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    problems: List[str] = []
    if missing:
        problems.append(f"missing columns vs training schema: {missing[:10]}")
    if extra:
        problems.append(f"extra columns vs training schema: {extra[:10]}")
    if not missing and not extra and actual != expected:
        first_pos = next(
            i for i, (a, e) in enumerate(zip(actual, expected)) if a != e
        )
        problems.append(
            f"column order differs at position {first_pos}: "
            f"{actual[first_pos]!r} vs trained {expected[first_pos]!r}"
        )
    if problems:
        return SchemaCheckResult(False, problems[0], tuple(problems))
    return SchemaCheckResult(True, "schema match (set and order)", ())


def assert_join_rowcounts(*csvs: Path) -> SchemaCheckResult:
    """All tables in a staged join must carry the same row count."""
    counts = [(Path(p).name, len(read_table(Path(p)))) for p in csvs]
    values = {n for _, n in counts}
    if len(values) > 1:
        detail = ", ".join(f"{name}: {n}" for name, n in counts)
        return SchemaCheckResult(
            False, f"row-count mismatch across join ({detail})", (detail,)
        )
    n = counts[0][1] if counts else 0
    return SchemaCheckResult(True, f"row counts equal ({n})", tuple())


def assert_join_alignment(csv_a: Path, csv_b: Path, key_col: str) -> SchemaCheckResult:
    """Key-column SEQUENCE must match position by position.

    Guards the misalignment class where row counts are equal but rows
    are permuted relative to each other (prototype: kmeans 14/630) —
    a row-count check is blind to it by construction.
    """
    a, b = read_table(Path(csv_a)), read_table(Path(csv_b))
    for name, df in ((Path(csv_a).name, a), (Path(csv_b).name, b)):
        if key_col not in df.columns:
            return SchemaCheckResult(
                False, f"key column {key_col!r} missing from {name}", ()
            )
    seq_a = [str(v) for v in a[key_col].tolist()]
    seq_b = [str(v) for v in b[key_col].tolist()]
    if len(seq_a) != len(seq_b):
        return SchemaCheckResult(
            False,
            f"length differs: {len(seq_a)} vs {len(seq_b)} (run row-count check too)",
            (),
        )
    bad = [i for i, (x, y) in enumerate(zip(seq_a, seq_b)) if x != y]
    if bad:
        first = bad[0]
        return SchemaCheckResult(
            False,
            f"key sequence misaligned at position {first}: "
            f"{seq_a[first]!r} vs {seq_b[first]!r} ({len(bad)} positions differ)",
            tuple(str(i) for i in bad[:20]),
        )
    return SchemaCheckResult(True, f"key sequences aligned ({len(seq_a)} rows)", ())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    p_stamp = sub.add_parser("stamp", help="Record training-side schema fingerprint")
    p_stamp.add_argument("--feature-csv", type=Path, default=None)
    p_stamp.add_argument("--model-payload", type=Path, default=None)
    p_stamp.add_argument("--out-json", type=Path, default=Path("schema.json"))

    p_check = sub.add_parser("check", help="Assert predict-side schema match")
    p_check.add_argument("--feature-csv", type=Path, required=True)
    p_check.add_argument("--schema-json", type=Path, required=True)

    p_join = sub.add_parser("check-join", help="Assert equal row counts")
    p_join.add_argument("--csv", type=Path, action="append", required=True)

    args = ap.parse_args()

    if args.mode == "stamp":
        stamp_schema(args.feature_csv, args.out_json, args.model_payload)
        return 0
    if args.mode == "check":
        result = assert_predict_schema(args.feature_csv, args.schema_json)
    else:
        result = assert_join_rowcounts(*args.csv)
    status = "PASS" if result.ok else "FAIL"
    print(f"[{status}] {result.first_mismatch}")
    for line in result.details:
        print(f"  - {line}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
