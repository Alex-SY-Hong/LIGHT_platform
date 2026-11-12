"""Fault-injection regression suite (task 6) — mutation testing for the gates.

User journeys:
- As a pipeline maintainer, I want every known corruption family to be
  intercepted by at least one gate, so that silent data damage cannot
  reach prediction.
- As a reviewer, I want proof that the gates never fire spuriously on
  clean data, so that PASS verdicts are trustworthy.

Families (each anchored to a real prototype, not invented):
- value shift +3 on log10 domain  -> the 4 mixed-unit YM rows
- SR x100 scale corruption         -> the known SR=300 row
- unseeded randomness             -> historical double-run drift
- column shuffle                  -> feature-order drift at predict time
- row shuffle                     -> candidate-table row misalignment

Acceptance: 100% interception on injected faults,
zero false positives on a clean synthetic corpus, and the real-DB
plausibility baseline stays exactly the four documented rows.
"""

from pathlib import Path

import pandas as pd
import pytest

from Agent.verification.fault_injection import (
    FAULTS,
    corrupt_sr_scale,
    corrupt_shift_log10,
    corrupt_unseeded,
    run_suite,
    shuffle_columns,
    shuffle_rows,
)
from Agent.verification.physical_plausibility import check_db
from Agent.verification.reproducibility_check import compare_tables
from Agent.verification.schema_check import (
    assert_join_alignment,
    assert_predict_schema,
    stamp_schema,
)

YM_COL = "Young's Modulus (kPa) log10"
SR_COL = "Swelling Ratio (times)"


def _clean_ym(tmp: Path) -> Path:
    df = pd.DataFrame(
        {
            "SMILE A": ["C"] * 3,
            YM_COL: [2.0, 3.2, 4.0],  # 100 kPa .. 10 MPa boundary-included
            "Source": ["水凝胶A.pdf", "GelMA hydrogel study", "plastic DOI"],
        }
    )
    p = tmp / "ym.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    return p


def _clean_sr(tmp: Path) -> Path:
    df = pd.DataFrame({"SMILE A": ["C"] * 3, SR_COL: [1.5, 12.0, 150.0]})
    p = tmp / "sr.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    return p


def _features(tmp: Path) -> Path:
    df = pd.DataFrame(
        {"Pair_ID": [f"P{i}" for i in range(4)], "fp_0": [0, 1, 0, 2], "fp_1": [1, 0, 1, 0]}
    )
    p = tmp / "feat.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    return p


class TestValueCorruption:
    def test_plus3_log10_intercepted(self, tmp_path: Path) -> None:
        # Arrange — mixed-unit prototype: Pa values mixed into kPa-log column
        csv = _clean_ym(tmp_path)
        corrupt_shift_log10(csv, YM_COL, delta=+3.0)

        # Act
        findings = check_db(csv, "YM").findings

        # Assert — every shifted gel row is flagged
        assert {f.row_index for f in findings} >= {0, 1}

    def test_sr_scale_corruption_intercepted(self, tmp_path: Path) -> None:
        # Arrange — x100 on mid-range value crosses the 200 suspect bar
        csv = _clean_sr(tmp_path)
        corrupt_sr_scale(csv, factor=100.0)

        # Act
        findings = check_db(csv, "SR").findings

        # Assert
        assert any(f.rule_id == "SR-RATIO-SUPERABSORBENT" for f in findings)


class TestDeterminismCorruption:
    def test_unseeded_randomness_intercepted(self, tmp_path: Path) -> None:
        # Arrange — two draws from an unseeded corruptor must differ
        csv = _clean_sr(tmp_path)

        # Act
        corrupt_unseeded(csv, SR_COL)
        first = pd.read_csv(csv)
        corrupt_unseeded(csv, SR_COL)
        second = pd.read_csv(csv)

        # Assert — the compare engine the repro gate uses sees the drift
        assert compare_tables(first, second) != []


class TestSchemaCorruption:
    def test_column_shuffle_intercepted(self, tmp_path: Path) -> None:
        # Arrange — stamp the clean contract, then shuffle column order
        csv = _features(tmp_path)
        schema = stamp_schema(csv, tmp_path / "schema.json")
        shuffle_columns(csv)

        # Act
        result = assert_predict_schema(csv, schema)

        # Assert — set matches, order differs: must FAIL, never pad
        assert not result.ok

    def test_row_shuffle_intercepted(self, tmp_path: Path) -> None:
        # Arrange — row-permutation prototype: same rows, same count,
        # wrong order -> row-count join check CANNOT see this; the
        # alignment gate on the key column must.
        left = _features(tmp_path)
        shuffled = shuffle_rows(left, key="Pair_ID")

        # Act
        result = assert_join_alignment(left, shuffled, key_col="Pair_ID")

        # Assert
        assert not result.ok


class TestSuiteRunner:
    def test_registry_has_all_five_families(self) -> None:
        families = {f.family for f in FAULTS}
        assert families >= {
            "value-shift",
            "value-scale",
            "determinism",
            "column-shuffle",
            "row-shuffle",
        }

    def test_clean_corpus_zero_false_positives(self, tmp_path: Path) -> None:
        # Arrange / Act — full suite against clean synthetic corpus
        report = run_suite(
            ym_csv=_clean_ym(tmp_path),
            sr_csv=_clean_sr(tmp_path),
            features_csv=_features(tmp_path),
            workdir=tmp_path,
        )

        # Assert — every fault intercepted, nothing fired on clean data
        assert report.mutation_score == 1.0, report.summary()
        assert report.clean_false_positives == 0

    @pytest.mark.slow
    def test_real_db_baseline_unchanged(self) -> None:
        # Acceptance guard: real YM DB still flags exactly rows
        # 0/1045/1534/1536 (TierA) — no drift from refactor or new rules.
        repo = Path(__file__).resolve().parents[3]
        csv = repo / "Supervised_Learning/DataBase/youngs_modulus.csv"
        if not csv.exists():
            pytest.skip("real DB not present")
        report = check_db(csv, "YM")
        tier_a_rows = {
            f.row_index
            for f in report.findings
            if f.disposition == "TierA-isolation-review"
        }
        assert tier_a_rows == {0, 1045, 1534, 1536}
