"""Unit tests for Agent.verification.reproducibility_check (task 4).

AAA-structured; run from repo root:
    python -m pytest Agent/verification/tests -q
"""

import sys
from pathlib import Path

import pandas as pd

from Agent.verification.reproducibility_check import (
    check_reproducible,
    compare_json,
    compare_tables,
    read_table,
)


class TestReadTable:
    def test_reads_utf8_csv(self, tmp_path: Path) -> None:
        # Arrange
        p = tmp_path / "u.csv"
        p.write_text("a,b\n1,x\n", encoding="utf-8")

        # Act
        df = read_table(p)

        # Assert
        assert list(df.columns) == ["a", "b"]
        assert df.loc[0, "b"] == "x"

    def test_falls_back_to_gbk(self, tmp_path: Path) -> None:
        # Arrange
        p = tmp_path / "g.csv"
        p.write_text("name\n中文\n", encoding="gbk")

        # Act
        df = read_table(p)

        # Assert
        assert df.loc[0, "name"] == "中文"


class TestCompareTables:
    def test_identical_tables_pass(self) -> None:
        # Arrange
        a = pd.DataFrame({"x": [1, 2], "y": ["p", "q"]})
        b = pd.DataFrame({"x": [1, 2], "y": ["p", "q"]})

        # Act / Assert
        assert compare_tables(a, b) == []

    def test_changed_cell_reported(self) -> None:
        # Arrange
        a = pd.DataFrame({"x": [1.0, 2.0]})
        b = pd.DataFrame({"x": [1.0, 9.0]})

        # Act
        diffs = compare_tables(a, b)

        # Assert
        assert len(diffs) == 1
        assert diffs[0].location == "row 1, col x"
        assert diffs[0].run1 == 2.0 and diffs[0].run2 == 9.0

    def test_float_tolerance_respected(self) -> None:
        # Arrange
        a = pd.DataFrame({"x": [0.1 + 0.2]})
        b = pd.DataFrame({"x": [0.3]})

        # Act / Assert
        assert compare_tables(a, b, float_tol=1e-9) == []
        assert len(compare_tables(a, b, float_tol=0.0)) == 1


class TestCompareJson:
    def test_nested_diff_found(self) -> None:
        # Arrange
        a = {"outer": {"k": 1, "list": [1, 2, 3]}}
        b = {"outer": {"k": 1, "list": [1, 2, 4]}}

        # Act
        diffs = compare_json(a, b)

        # Assert
        assert len(diffs) == 1
        assert diffs[0].location == "$.outer.list[2]"

    def test_float_within_tol_passes(self) -> None:
        # Arrange
        a = {"v": 0.5281}
        b = {"v": 0.5283}

        # Act / Assert
        assert compare_json(a, b, float_tol=0.005) == []
        assert len(compare_json(a, b, float_tol=0.0)) == 1


class TestEndToEnd:
    def _writer_cmd(self, csv_path: Path, seed: int) -> str:
        code = (
            "import csv,sys;"
            "p=sys.argv[1];"
            f"rows=[['i','v'],[0,{seed}],[1,{seed * 2}]];"
            "f=open(p,'w',newline='');"
            "csv.writer(f).writerows(rows);f.close()"
        )
        return f'{sys.executable} -c "{code}" {csv_path}'

    def test_deterministic_script_passes(self, tmp_path: Path) -> None:
        # Arrange
        out = tmp_path / "out.csv"
        cmd = self._writer_cmd(out, seed=7)

        # Act
        result = check_reproducible(cmd=cmd, artifacts=[out], cwd=tmp_path)

        # Assert
        assert result.status == "PASS", result.summary()

    def test_nondeterministic_script_fails(self, tmp_path: Path) -> None:
        # Arrange
        out = tmp_path / "out.csv"
        code = (
            "import csv,random,sys;"
            "p=sys.argv[1];"
            "r=random.SystemRandom().random();"
            "f=open(p,'w',newline='');"
            "csv.writer(f).writerows([['i','v'],[0,r]]);f.close()"
        )
        cmd = f'{sys.executable} -c "{code}" {out}'

        # Act
        result = check_reproducible(cmd=cmd, artifacts=[out], cwd=tmp_path)

        # Assert
        assert result.status == "FAIL"
