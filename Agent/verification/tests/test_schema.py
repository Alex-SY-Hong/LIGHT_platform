"""Unit tests for Agent.verification.schema_check (task 5).

Covers: fingerprint stability, order sensitivity, missing/extra column
detection, stamp->check roundtrip, and join row-count assertion.
"""

from pathlib import Path

import pandas as pd

from Agent.verification.schema_check import (
    assert_join_rowcounts,
    assert_predict_schema,
    compute_schema_fingerprint,
    stamp_schema,
)


def _feat_csv(path: Path, cols: list, rows: int = 3) -> Path:
    df = pd.DataFrame({c: [float(i) for i in range(rows)] for c in cols})
    df.to_csv(path, index=False, encoding="utf-8")
    return path


class TestFingerprint:
    def test_same_columns_same_hash(self) -> None:
        # Arrange
        df1 = pd.DataFrame({"fp_0": [1.0], "fp_1": [0.0]})
        df2 = pd.DataFrame({"fp_0": [9.0], "fp_1": [5.0]})

        # Act / Assert — values differ, schema identical
        assert compute_schema_fingerprint(df1) == compute_schema_fingerprint(df2)

    def test_order_change_changes_hash(self) -> None:
        # Arrange
        df1 = pd.DataFrame({"fp_0": [1.0], "fp_1": [0.0]})
        df2 = pd.DataFrame({"fp_1": [0.0], "fp_0": [1.0]})

        # Act / Assert
        assert compute_schema_fingerprint(df1) != compute_schema_fingerprint(df2)


class TestStampAndCheck:
    def test_roundtrip_pass(self, tmp_path: Path) -> None:
        # Arrange
        csv = _feat_csv(tmp_path / "feat.csv", ["fp_0", "fp_1", "fp_2"])
        schema_json = stamp_schema(feature_csv=csv, out_json=tmp_path / "schema.json")

        # Act
        result = assert_predict_schema(feature_csv=csv, schema_json=schema_json)

        # Assert
        assert result.ok, result.first_mismatch

    def test_reordered_columns_fail(self, tmp_path: Path) -> None:
        # Arrange
        train = _feat_csv(tmp_path / "train.csv", ["fp_0", "fp_1"])
        schema_json = stamp_schema(train, tmp_path / "schema.json")
        swapped = _feat_csv(tmp_path / "swap.csv", ["fp_1", "fp_0"])

        # Act
        result = assert_predict_schema(swapped, schema_json)

        # Assert
        assert not result.ok
        assert "order" in result.first_mismatch or "position" in result.first_mismatch

    def test_missing_column_fail(self, tmp_path: Path) -> None:
        # Arrange
        train = _feat_csv(tmp_path / "train.csv", ["fp_0", "fp_1"])
        schema_json = stamp_schema(train, tmp_path / "schema.json")
        short = _feat_csv(tmp_path / "short.csv", ["fp_0"])

        # Act
        result = assert_predict_schema(short, schema_json)

        # Assert
        assert not result.ok
        assert "missing" in result.first_mismatch

    def test_extra_column_fail(self, tmp_path: Path) -> None:
        # Arrange
        train = _feat_csv(tmp_path / "train.csv", ["fp_0", "fp_1"])
        schema_json = stamp_schema(train, tmp_path / "schema.json")
        wide = _feat_csv(tmp_path / "wide.csv", ["fp_0", "fp_1", "fp_extra"])

        # Act
        result = assert_predict_schema(wide, schema_json)

        # Assert
        assert not result.ok
        assert "extra" in result.first_mismatch


class TestJoinRowcounts:
    def test_equal_counts_pass(self, tmp_path: Path) -> None:
        # Arrange
        a = _feat_csv(tmp_path / "a.csv", ["fp_0"])
        b = _feat_csv(tmp_path / "b.csv", ["fp_1"])

        # Act / Assert
        assert assert_join_rowcounts(a, b).ok

    def test_count_mismatch_fails(self, tmp_path: Path) -> None:
        # Arrange
        a = _feat_csv(tmp_path / "a.csv", ["fp_0"], rows=3)
        b = _feat_csv(tmp_path / "b.csv", ["fp_1"], rows=2)

        # Act
        result = assert_join_rowcounts(a, b)

        # Assert
        assert not result.ok
        assert "3" in result.first_mismatch and "2" in result.first_mismatch
