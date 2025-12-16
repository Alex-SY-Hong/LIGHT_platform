#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline:
1) Count SMILES occurrences from an input CSV (auto-detect SMILES columns).
2) Generate all ordered A–B pairs (A != B) from unique SMILES.
3) Deduplicate A–B pairs as unordered pairs ({A,B}).

Usage:
    python smiles_pipeline.py input.csv

Outputs (in the same folder as input.csv):
    input-counts.csv
    input-AB-ordered.csv
    input-AB-unique.csv
"""

from __future__ import annotations

import sys
import csv
import pandas as pd
from pathlib import Path
from collections import Counter

def read_csv_safely(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "gb18030", "gbk", "utf-8", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Unable to read CSV: {path}\nLast error: {last_err}")

def detect_smiles_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        s = str(c).strip().lower()
        if "smile" in s:
            cols.append(c)
    return cols if cols else [df.columns[0]]

def norm_smiles(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def banner(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def step1_count_smiles(in_csv: Path) -> Path:
    banner("STEP 1 - Count SMILES")

    df = read_csv_safely(str(in_csv))
    smiles_cols = detect_smiles_columns(df)
    print(f"[INFO] Detected SMILES columns: {smiles_cols}")
    print(f"[INFO] Input shape: {df.shape}")

    all_smiles = []
    for col in smiles_cols:
        all_smiles.extend(df[col].dropna().astype(str).map(str.strip).tolist())
    all_smiles = [s for s in all_smiles if s]

    counts = Counter(all_smiles)
    count_df = (
        pd.DataFrame(counts.items(), columns=["SMILES", "Count"])
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    out_csv = in_csv.with_name(f"{in_csv.stem}-counts.csv")
    count_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] Unique SMILES: {len(count_df)}")
    print("[PREVIEW] Top 10:")
    print(count_df.head(10).to_string(index=False))

    if len(count_df) < 2:
        raise ValueError("Not enough SMILES (< 2). Cannot generate pairs.")

    return out_csv

def step2_generate_ordered_pairs(counts_csv: Path) -> Path:
    banner("STEP 2 - Generate ordered A–B pairs")

    df = read_csv_safely(str(counts_csv))
    if "SMILES" in df.columns:
        col = "SMILES"
    else:
        smile_like_cols = [c for c in df.columns if "smile" in str(c).strip().lower()]
        col = smile_like_cols[0] if smile_like_cols else df.columns[0]

    smiles = (
        df[col]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda s: s.str.len() > 0]
        .drop_duplicates()
        .tolist()
    )

    n = len(smiles)
    print(f"[INFO] Valid SMILES: {n}")
    if n < 2:
        raise ValueError("Not enough SMILES (< 2). Cannot generate pairs.")

    out_csv = counts_csv.with_name(f"{counts_csv.stem.replace('-counts','')}-AB-ordered.csv")
    total_pairs = n * (n - 1)
    print(f"[INFO] Total ordered pairs: {total_pairs}")

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Pair_ID", "SMILE_A", "SMILE_B"])
        pair_id = 1
        for a in smiles:
            for b in smiles:
                if a == b:
                    continue
                w.writerow([f"Pair_{pair_id}", a, b])
                pair_id += 1

    print(f"[OK] Ordered pairs generated: {total_pairs}")
    return out_csv

def step3_dedup_unordered_pairs(ordered_csv: Path, reorder_pair: bool = True, chunksize: int = 200_000) -> Path:
    banner("STEP 3 - Deduplicate unordered A–B pairs")

    out_csv = ordered_csv.with_name(f"{ordered_csv.stem.replace('-AB-ordered','')}-AB-unique.csv")

    possible_a = ["SMILE_A", "SMILE A", "A", "smile_a", "smile a"]
    possible_b = ["SMILE_B", "SMILE B", "B", "smile_b", "smile b"]

    head = read_csv_safely(str(ordered_csv)).head(1)
    col_a = next((c for c in possible_a if c in head.columns), None)
    col_b = next((c for c in possible_b if c in head.columns), None)
    if col_a is None or col_b is None:
        raise KeyError(f"Cannot find SMILE_A/SMILE_B columns. Columns: {list(head.columns)}")

    print(f"[INFO] Using columns: {col_a} / {col_b}")

    seen = set()
    kept_rows = 0
    total_rows = 0

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fout:
        writer = None

        for chunk in pd.read_csv(ordered_csv, dtype=str, keep_default_na=False, chunksize=chunksize, encoding="utf-8-sig"):
            total_rows += len(chunk)

            a = chunk[col_a].map(norm_smiles)
            b = chunk[col_b].map(norm_smiles)
            keys = [tuple(sorted((x, y))) for x, y in zip(a, b)]

            mask_keep = []
            for k in keys:
                if k in seen:
                    mask_keep.append(False)
                else:
                    seen.add(k)
                    mask_keep.append(True)

            out_chunk = chunk.loc[mask_keep].copy()

            if reorder_pair and len(out_chunk) > 0:
                out_chunk[[col_a, col_b]] = pd.DataFrame(
                    [tuple(sorted((norm_smiles(x), norm_smiles(y))))
                     for x, y in zip(out_chunk[col_a], out_chunk[col_b])],
                    index=out_chunk.index
                )

            if writer is None:
                writer = csv.DictWriter(fout, fieldnames=list(out_chunk.columns))
                writer.writeheader()

            for _, row in out_chunk.iterrows():
                writer.writerow(row.to_dict())

            kept_rows += len(out_chunk)

    print(f"[OK] Input rows: {total_rows}, kept: {kept_rows}, removed: {total_rows - kept_rows}")
    banner("DONE")
    return out_csv

def main():
    if len(sys.argv) < 2:
        print("Usage: python smiles_pipeline.py <input_csv>")
        sys.exit(1)

    in_csv = Path(sys.argv[1]).expanduser().resolve()
    if not in_csv.exists():
        print("[ERROR] File not found.")
        sys.exit(1)

    counts_csv = step1_count_smiles(in_csv)
    ordered_csv = step2_generate_ordered_pairs(counts_csv)
    _unique_csv = step3_dedup_unordered_pairs(ordered_csv, reorder_pair=True)

if __name__ == "__main__":
    main()
