"""Unit + integration tests for verify_key_numbers (task 9, checks 1-4).

Fast tests cover the metric math, the rounding-tier anchor policy (paper
values are pinned CLAIMS; structural identities are derived), and the
derived p-value counter. Slow-marked tests run the real artifact checks.
"""

import math
from pathlib import Path

import pandas as pd
import pytest

import Agent.verification.verify_key_numbers as vkn
from Agent.verification.verify_key_numbers import (
    classification_oof_metrics,
    count_p_values,
    regression_oof_metrics,
    verify_round_to_claim,
)


def _oof_csv(path: Path, rows: list) -> Path:
    df = pd.DataFrame(rows, columns=["idx", "y_pred", "y_true", "residual", "fold"])
    df.to_csv(path, index=False, encoding="utf-8")
    return path


class TestMetricMath:
    def test_regression_r2_perfect_and_imperfect(self, tmp_path: Path) -> None:
        # Arrange
        perfect = _oof_csv(tmp_path / "p.csv", [(0, 3.0, 3.0, 0.0, 1)])
        imperfect = _oof_csv(
            tmp_path / "i.csv",
            [(0, 1.0, 1.0, 0.0, 1), (1, 2.0, 2.0, 0.0, 1), (2, 4.0, 3.0, 1.0, 1)],
        )

        # Act
        r_perfect = regression_oof_metrics(perfect)
        r_imperfect = regression_oof_metrics(imperfect)

        # Assert — ss_res=1, ss_tot=2 -> r2 = 0.5
        assert r_perfect["r2"] == 1.0
        assert r_perfect["n"] == 1
        assert math.isclose(r_imperfect["r2"], 0.5, rel_tol=1e-12)
        assert r_imperfect["n"] == 3

    def test_classification_acc_f1(self, tmp_path: Path) -> None:
        # Arrange — tp=1 fp=1 fn=1 tn=1: acc=0.5, precision=recall=f1=0.5
        rows = [(0, 0, 0, 0, 1), (1, 1, 1, 0, 1), (2, 0, 1, 1, 1), (3, 1, 0, -1, 1)]
        csv = _oof_csv(tmp_path / "c.csv", rows)

        # Act
        m = classification_oof_metrics(csv)

        # Assert
        assert m["n"] == 4
        assert math.isclose(m["accuracy"], 0.5, rel_tol=1e-12)
        assert math.isclose(m["f1_positive"], 0.5, rel_tol=1e-12)


class TestAnchorPolicy:
    def test_round_to_claim_accepts_within_half_ulp(self) -> None:
        # Paper states 3 decimals: 0.5283 rounds to 0.528 -> PASS
        assert verify_round_to_claim(0.5283, 0.528, ndigits=3)
        assert verify_round_to_claim(0.5276, 0.528, ndigits=3)

    def test_round_to_claim_rejects_beyond(self) -> None:
        assert not verify_round_to_claim(0.5291, 0.528, ndigits=3)
        assert not verify_round_to_claim(0.526, 0.528, ndigits=3)


class TestDerivedCount:
    def test_p_value_count_derived_not_hardcoded(self) -> None:
        # Arrange — nested structure mirroring the summary json shape
        obj = {
            "results": {
                "m1": {"partial_correlation_p": 0.1},
                "m2": {"p_value": 0.2, "p_val": 0.3, "ci": [0, 1]},
            }
        }

        # Act / Assert — count derived by walking, never a fixed number
        assert count_p_values(obj) == 3
        assert count_p_values({"results": {}}) == 0


class TestRealArtifacts:
    """Slow checks against the tracked artifacts (marked slow)."""

    @pytest.mark.slow
    def test_check1_ym_oof_r2(self) -> None:
        result = vkn.check_1_ym_oof()
        assert result.ok, result.detail
        assert result.detail["n"] == 1629  # paper claim, pinned

    @pytest.mark.slow
    def test_check2_cls_oof_acc_f1(self) -> None:
        result = vkn.check_2_cls_oof()
        assert result.ok, result.detail
        assert result.detail["n"] == 360  # paper claim, pinned

    @pytest.mark.slow
    def test_check3_consensus_regeneration(self) -> None:
        result = vkn.check_3_consensus()
        assert result.ok, result.detail
        assert result.detail["winner_margin_present"] is True

    @pytest.mark.slow
    def test_check4_popularity_seeded_double_run(self) -> None:
        result = vkn.check_4_popularity()
        assert result.ok, result.detail
        assert result.detail["p_value_count_run1"] == result.detail["p_value_count_run2"]
        assert result.detail["p_value_count_run1"] > 0
