"""Unit tests for Agent.verification.physical_plausibility (task 3).

Covers: boundary semantics, tier routing (A=flag/isolation vs B=suspicion),
rule symmetry across YM and SR, zero false positives on clean values,
encoding fallback, and the P9 append-only log record.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

import Agent.verification.physical_plausibility as pp
from Agent.verification.physical_plausibility import (
    RULES,
    check_db,
    evaluate_value,
)


def _ym_csv(path: Path, values: list, sources: list = None) -> Path:
    n = len(values)
    df = pd.DataFrame(
        {
            "SMILE A": ["C"] * n,
            "Young's Modulus (kPa) log10": values,
            "Source": sources if sources is not None else ["plastic DOI x"] * n,
        }
    )
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def _sr_csv(path: Path, values: list) -> Path:
    df = pd.DataFrame(
        {"SMILE A": ["C"] * len(values), "Swelling Ratio (times)": values}
    )
    df.to_csv(path, index=False, encoding="utf-8")
    return path


class TestRuleRegistry:
    def test_symmetric_coverage(self) -> None:
        # Arrange / Act / Assert — P4: YM and SR get the same structure
        for target in ("YM", "SR"):
            tiers = {r.tier for r in RULES if r.target == target}
            assert "flag" in tiers or "domain" in tiers, target
            assert "suspicious" in tiers, target

    def test_all_rules_anchored_or_provisional(self) -> None:
        # P8: every threshold carries an external anchor or is marked provisional
        for r in RULES:
            assert r.anchors or r.provisional, f"{r.rule_id} lacks anchor"
            assert r.rationale, f"{r.rule_id} lacks rationale"


class TestEvaluateValue:
    def test_ym_boundaries(self) -> None:
        # 4.0 exactly passes; 4.001 flags Tier A; 6.0 additionally Tier B
        assert evaluate_value("YM", 4.0) == []
        assert evaluate_value("YM", 4.001)[0].tier == "flag"
        tiers = {f.tier for f in evaluate_value("YM", 6.0)}
        assert tiers == {"flag", "suspicious"}

    def test_sr_domain(self) -> None:
        # 1.0 exactly passes; 0.9 is a domain violation (Tier A)
        assert evaluate_value("SR", 1.0) == []
        assert evaluate_value("SR", 0.9)[0].tier == "domain"
        assert evaluate_value("SR", 0.0)[0].tier == "domain"

    def test_sr_suspicious(self) -> None:
        # 150 clean; 300 suspicious Tier B
        assert evaluate_value("SR", 150.0) == []
        assert evaluate_value("SR", 300.0)[0].tier == "suspicious"

    def test_nan_reported_as_invalid(self) -> None:
        # NaN rows are reported as invalid-value findings, not rule hits
        findings = evaluate_value("YM", math.nan)
        assert findings and findings[0].tier == "invalid"


class TestCheckDb:
    def test_row_indices_reported(self, tmp_path: Path) -> None:
        # Arrange — gel-source row carries the violation
        csv = _ym_csv(
            tmp_path / "ym.csv",
            [1.5, 4.5, 2.0],
            sources=["plastic DOI x", "GelMA 水凝胶论文", "plastic DOI y"],
        )

        # Act
        report = check_db(csv, "YM")

        # Assert — gate never modifies; flags row 1 only
        assert [f.row_index for f in report.findings] == [1]
        assert report.disposition_of(1) == "TierA-isolation-review"
        assert pd.read_csv(csv).shape[0] == 3  # untouched

    def test_plastic_gpa_rows_exempt(self, tmp_path: Path) -> None:
        # Arrange — 260 GPa PBO fiber = log10(kPa) 8.415, legit plastic
        csv = _ym_csv(
            tmp_path / "pl.csv",
            [8.415, 5.9],
            sources=["10.1002/app.33461 PBO fiber", "10.3390 epoxy"],
        )

        # Act / Assert — zero findings outside gel scope
        assert check_db(csv, "YM").findings == []

    def test_clean_db_zero_findings(self, tmp_path: Path) -> None:
        # Arrange — zero false positives on plausible values (P5 acceptance)
        csv = _sr_csv(tmp_path / "sr.csv", [1.2, 5.0, 80.0, 150.0])

        # Act / Assert
        assert check_db(csv, "SR").findings == []

    def test_gbk_csv_read(self, tmp_path: Path) -> None:
        # Arrange
        csv = tmp_path / "g.csv"
        df = pd.DataFrame({"Swelling Ratio (times)": [0.5], "备注": ["中文"]})
        df.to_csv(csv, index=False, encoding="gbk")

        # Act / Assert
        assert check_db(csv, "SR").findings[0].row_index == 0

    def test_log_appended(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — P9: append-only record with timestamp
        log = tmp_path / "plausibility_log.jsonl"
        monkeypatch.setattr(pp, "REPORTS_DIR", tmp_path)
        monkeypatch.setattr(pp, "LOG_PATH", log)
        csv = _ym_csv(tmp_path / "ym.csv", [4.5], sources=["水凝胶 cartilage.pdf"])

        # Act
        check_db(csv, "YM")

        # Assert
        record = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert record["target"] == "YM"
        assert record["n_findings"] == 1
        assert "timestamp" in record
