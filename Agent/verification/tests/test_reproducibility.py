"""Unit tests for Agent.verification.reproducibility_check (task 4).

AAA-structured; run from repo root:
    python -m pytest Agent/verification/tests -q
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

import Agent.verification.reproducibility_check as rc
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


class TestReviewerFindings:
    """Regression tests for mle-reviewer findings on the first cut."""

    def test_json_nan_vs_value_fails(self) -> None:
        # Arrange — CRITICAL: NaN asymmetry used to pass silently
        a = {"v": math.nan}
        b = {"v": 5.0}

        # Act / Assert
        assert len(compare_json(a, b)) == 1
        assert compare_json({"v": math.nan}, {"v": math.nan}) == []

    def test_json_bool_vs_int_fails(self) -> None:
        # Arrange — True == 1 used to pass
        # Act / Assert
        assert len(compare_json({"v": True}, {"v": 1})) == 1

    def test_column_rename_fails_not_crash(self) -> None:
        # Arrange — same width, renamed column used to KeyError
        a = pd.DataFrame({"x": [1.0], "y": [2.0]})
        b = pd.DataFrame({"x": [1.0], "z": [2.0]})

        # Act
        diffs = compare_tables(a, b)

        # Assert
        assert diffs and diffs[0].kind == "shape"

    def test_runs_below_two_raises(self, tmp_path: Path) -> None:
        # Arrange — runs=1/0 used to vacuously PASS
        out = tmp_path / "o.csv"

        # Act / Assert
        with pytest.raises(ValueError):
            check_reproducible(cmd="echo hi", artifacts=[out], runs=1)
        with pytest.raises(ValueError):
            check_reproducible(cmd="echo hi", artifacts=[], runs=2)

    def test_nonzero_rc_fails(self, tmp_path: Path) -> None:
        # Arrange
        out = tmp_path / "o.csv"
        out.write_text("v\n1\n", encoding="utf-8")
        cmd = f'{sys.executable} -c "import sys; sys.exit(3)"'

        # Act
        result = check_reproducible(cmd=cmd, artifacts=[out], cwd=tmp_path)

        # Assert
        assert result.status == "FAIL"
        assert result.return_codes == [3, 3]

    def test_intermittent_flip_three_runs_caught(self, tmp_path: Path) -> None:
        # Arrange — [B, A, A]: final pair equal, earlier pair differs.
        # Only full pairwise comparison catches this.
        out = tmp_path / "out.csv"
        marker = tmp_path / "marker"
        code = (
            "import csv,os,sys;"
            "p,mi=sys.argv[1],sys.argv[2];"
            "v='A' if os.path.exists(mi) else 'B';"
            "open(mi,'a').close();"
            "f=open(p,'w',newline='');"
            "csv.writer(f).writerows([['v'],[v]]);f.close()"
        )
        cmd = f'{sys.executable} -c "{code}" {out} {marker}'

        # Act
        result = check_reproducible(cmd=cmd, artifacts=[out], cwd=tmp_path, runs=3)

        # Assert
        assert result.status == "FAIL"
        # final pair (run2, run3) is A==A; only run1-vs-run2 yields B->A
        assert any(
            str(d.run1) == "B" and str(d.run2) == "A"
            for a in result.artifacts
            for d in a.diffs
        )

    def test_artifact_missing_second_run_fails(self, tmp_path: Path) -> None:
        # Arrange — toggle create/delete: run1 present, run2 missing
        out = tmp_path / "out.csv"
        code = (
            "import os,sys;"
            "p=sys.argv[1];"
            "os.remove(p) if os.path.exists(p) else open(p,'w').write('v\\n1\\n')"
        )
        cmd = f'{sys.executable} -c "{code}" {out}'

        # Act
        result = check_reproducible(cmd=cmd, artifacts=[out], cwd=tmp_path)

        # Assert
        assert result.status == "FAIL"
        assert any(
            d.kind == "missing" for a in result.artifacts for d in a.diffs
        )

    def test_relative_artifact_resolves_against_cwd(self, tmp_path: Path) -> None:
        # Arrange — gate invoked from an unrelated cwd (this test's process
        # cwd); artifact name must resolve against the --cwd dir
        sub = tmp_path / "sub"
        sub.mkdir()
        out = sub / "out.csv"
        code = (
            "import csv,sys;"
            "p=sys.argv[1];"
            "f=open(p,'w',newline='');"
            "csv.writer(f).writerows([['v'],[7]]);f.close()"
        )
        cmd = f'{sys.executable} -c "{code}" out.csv'

        # Act
        result = check_reproducible(
            cmd=cmd, artifacts=[Path("out.csv")], cwd=sub
        )

        # Assert
        assert result.status == "PASS", result.summary()

    def test_log_record_fields(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange — P9 log contract: timestamp, cmd, status, return_codes
        log = tmp_path / "run_log.jsonl"
        monkeypatch.setattr(rc, "REPORTS_DIR", tmp_path)
        monkeypatch.setattr(rc, "RUN_LOG_PATH", log)
        out = tmp_path / "o.csv"
        out.write_text("v\n1\n", encoding="utf-8")
        cmd = f'{sys.executable} -c "open(r\'{out}\',\'w\').write(\'v\\\\n1\\\\n\')"'

        # Act
        result = check_reproducible(cmd=cmd, artifacts=[out], cwd=tmp_path)

        # Assert
        assert result.status == "PASS"
        record = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert record["status"] == "PASS"
        assert record["return_codes"] == [0, 0]
        assert "timestamp" in record and "T" in record["timestamp"]
