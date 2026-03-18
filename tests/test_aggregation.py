"""Tests for noethersolve.aggregation — protein aggregation propensity predictor."""

import pytest
from noethersolve.aggregation import (
    predict_aggregation,
    KYTE_DOOLITTLE,
    AGGRESCAN,
)


# ─── Scale exports ──────────────────────────────────────────────────────────

class TestScales:
    def test_kyte_doolittle_has_20(self):
        assert len(KYTE_DOOLITTLE) == 20

    def test_aggrescan_has_20(self):
        assert len(AGGRESCAN) == 20

    def test_isoleucine_most_hydrophobic(self):
        assert KYTE_DOOLITTLE["I"] == max(KYTE_DOOLITTLE.values())

    def test_arginine_least_hydrophobic(self):
        assert KYTE_DOOLITTLE["R"] == min(KYTE_DOOLITTLE.values())


# ─── APR detection ──────────────────────────────────────────────────────────

class TestAPR:
    def test_hydrophobic_stretch_has_aprs(self):
        # All isoleucine — maximally aggregation-prone
        seq = "IIIIIIIIIIIIIIIIIIII"
        report = predict_aggregation(seq)
        assert report.n_aprs > 0
        apr_issues = [i for i in report.issues if i.check_name == "APR"]
        assert any(i.severity == "HIGH" for i in apr_issues)

    def test_charged_sequence_no_aprs(self):
        # All lysine — highly charged, not aggregation-prone
        seq = "KKKKKKKKKKKKKKKKKKKK"
        report = predict_aggregation(seq)
        assert report.n_aprs == 0

    def test_apr_positions_populated(self):
        seq = "ILVFILVFILVFILVFILVF"
        report = predict_aggregation(seq)
        assert len(report.apr_positions) > 0
        for start, end, score in report.apr_positions:
            assert score > 0.0
            assert end >= start


# ─── Hydrophobicity check ──────────────────────────────────────────────────

class TestHydrophobicity:
    def test_hydrophobic_high(self):
        seq = "ILVFILVFILVFILVFILVF"  # very hydrophobic
        report = predict_aggregation(seq)
        hydro = [i for i in report.issues if i.check_name == "HYDROPHOBICITY"]
        assert any(i.severity == "HIGH" for i in hydro)

    def test_hydrophilic_clean(self):
        seq = "DDDEEEKKKRRRDDDEEEK"
        report = predict_aggregation(seq)
        hydro = [i for i in report.issues if i.check_name == "HYDROPHOBICITY"]
        assert len(hydro) == 0

    def test_mean_hydrophobicity_calculated(self):
        seq = "AAAA"  # A = 1.8
        report = predict_aggregation(seq)
        assert abs(report.mean_hydrophobicity - 1.8) < 1e-6


# ─── Hydrophobic patch check ───────────────────────────────────────────────

class TestHydrophobicPatch:
    def test_long_patch_high(self):
        # 12 consecutive hydrophobic residues
        seq = "DDIIIIIIIIIIIIDDDDD"
        report = predict_aggregation(seq)
        patch = [i for i in report.issues if i.check_name == "HYDROPHOBIC_PATCH"]
        assert any(i.severity == "HIGH" for i in patch)

    def test_moderate_patch(self):
        # 8 consecutive hydrophobic
        seq = "DDIIIIIIIIIDDDDDDDD"
        report = predict_aggregation(seq)
        patch = [i for i in report.issues if i.check_name == "HYDROPHOBIC_PATCH"]
        assert any(i.severity == "MODERATE" for i in patch)

    def test_short_patch_clean(self):
        seq = "DDIIIIDDIIIIDDIIIDD"
        report = predict_aggregation(seq)
        patch = [i for i in report.issues if i.check_name == "HYDROPHOBIC_PATCH"]
        assert len(patch) == 0

    def test_longest_patch_metric(self):
        seq = "DDIIIIIIIDDDDDDDDDD"
        report = predict_aggregation(seq)
        assert report.longest_hydrophobic_patch == 7


# ─── Net charge check ──────────────────────────────────────────────────────

class TestNetCharge:
    def test_neutral_flagged(self):
        # Equal K and D → net charge 0
        seq = "KDKDKDKDKDKDKDKDKDKD"
        report = predict_aggregation(seq)
        charge = [i for i in report.issues if i.check_name == "NET_CHARGE"]
        assert any(i.severity == "MODERATE" for i in charge)

    def test_highly_charged_clean(self):
        # All K → strong positive charge
        seq = "KKKKKKKKKKKKKKKKKKKK"
        report = predict_aggregation(seq)
        charge = [i for i in report.issues if i.check_name == "NET_CHARGE"]
        assert len(charge) == 0

    def test_net_charge_metric(self):
        seq = "KKKKDDAA"  # 4 K (+4), 2 D (-2) → net +2
        report = predict_aggregation(seq)
        assert report.net_charge == 2


# ─── Low complexity check ──────────────────────────────────────────────────

class TestLowComplexity:
    def test_single_residue_type_flagged(self):
        seq = "AAAAAAAAAAAAAAAAAAAA"  # 1 unique in 20-window
        report = predict_aggregation(seq)
        lc = [i for i in report.issues if i.check_name == "LOW_COMPLEXITY"]
        assert len(lc) > 0

    def test_diverse_sequence_clean(self):
        seq = "ACDEFGHIKLMNPQRSTVWY"  # 20 unique
        report = predict_aggregation(seq)
        lc = [i for i in report.issues if i.check_name == "LOW_COMPLEXITY"]
        assert len(lc) == 0

    def test_short_sequence_skipped(self):
        seq = "AAAA"  # shorter than 20-residue window
        report = predict_aggregation(seq)
        lc = [i for i in report.issues if i.check_name == "LOW_COMPLEXITY"]
        assert len(lc) == 0


# ─── Verdict logic ──────────────────────────────────────────────────────────

class TestVerdict:
    def test_fail_on_high(self):
        seq = "ILVFILVFILVFILVFILVF"
        report = predict_aggregation(seq)
        assert report.verdict == "FAIL"
        assert not report.passed

    def test_pass_on_clean(self):
        # Charged, diverse, short patches, strong net charge
        seq = "DKKRNHQSDKKRNHQSDKKR"
        report = predict_aggregation(seq)
        assert report.verdict == "PASS"
        assert report.passed


# ─── Input validation ──────────────────────────────────────────────────────

class TestInputValidation:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            predict_aggregation("")

    def test_invalid_chars_raise(self):
        with pytest.raises(ValueError, match="Invalid"):
            predict_aggregation("ACDEFXYZ")

    def test_case_insensitive(self):
        report = predict_aggregation("acdefghik")
        assert report.sequence_length == 9

    def test_whitespace_stripped(self):
        report = predict_aggregation("ACD EFG HIK")
        assert report.sequence_length == 9

    def test_numbers_rejected(self):
        with pytest.raises(ValueError, match="Invalid"):
            predict_aggregation("ACD123")


# ─── Report formatting ─────────────────────────────────────────────────────

class TestReportFormat:
    def test_str_contains_verdict(self):
        report = predict_aggregation("DKERDKERDKERDKERDKER")
        s = str(report)
        assert "PASS" in s or "WARN" in s or "FAIL" in s

    def test_str_contains_hydrophobicity(self):
        report = predict_aggregation("DKERDKERDKERDKERDKER")
        assert "hydrophobicity" in str(report).lower()

    def test_str_contains_length(self):
        report = predict_aggregation("DKERDKERDKERDKERDKER")
        assert "20" in str(report)

    def test_str_contains_issues(self):
        report = predict_aggregation("ILVFILVFILVFILVFILVF")
        assert "Issues found" in str(report)

    def test_str_contains_apr_locations(self):
        report = predict_aggregation("ILVFILVFILVFILVFILVF")
        assert "APR locations" in str(report)
