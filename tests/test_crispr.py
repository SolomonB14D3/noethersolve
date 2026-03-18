"""Tests for noethersolve.crispr — CRISPR guide RNA scorer."""

import pytest
from noethersolve.crispr import (
    score_guide,
    score_guides,
    check_offtarget_pair,
    GuideReport,
)


# ─── Basic scoring ────────────────────────────────────────────────────────────

class TestBasicScoring:
    def test_good_guide_passes(self):
        # 50% GC, no homopolymers, ends with G
        guide = "ATGCATGCATGCATGCATCG"
        report = score_guide(guide)
        assert report.verdict == "PASS"
        assert report.passed
        assert report.activity_score >= 70

    def test_returns_guide_report(self):
        report = score_guide("ATGCATGCATGCATGCATCG")
        assert isinstance(report, GuideReport)
        assert report.length == 20

    def test_gc_content_calculated(self):
        guide = "GCGCGCGCGCGCGCGCGCGC"  # 100% GC
        report = score_guide(guide)
        assert abs(report.gc_content - 1.0) < 1e-9

    def test_seed_gc_calculated(self):
        # First 12 positions
        guide = "GCGCGCGCGCGCATATATAT"
        report = score_guide(guide)
        assert report.seed_gc > 0.8


# ─── GC content checks ───────────────────────────────────────────────────────

class TestGCContent:
    def test_very_low_gc_high(self):
        guide = "AAAAAAAATTTTTTTTTAAA"  # 0% GC
        report = score_guide(guide)
        gc_issues = [i for i in report.issues if i.check_name == "GC_CONTENT"]
        assert any(i.severity == "HIGH" for i in gc_issues)

    def test_very_high_gc_high(self):
        guide = "GCGCGCGCGCGCGCGCGCGC"  # 100% GC
        report = score_guide(guide)
        gc_issues = [i for i in report.issues if i.check_name == "GC_CONTENT"]
        assert len(gc_issues) > 0

    def test_optimal_gc_clean(self):
        guide = "ATGCATGCATGCATGCATCG"  # ~50%
        report = score_guide(guide)
        gc_issues = [i for i in report.issues if i.check_name == "GC_CONTENT"]
        assert len(gc_issues) == 0


# ─── Homopolymer checks ──────────────────────────────────────────────────────

class TestHomopolymer:
    def test_tttt_high(self):
        guide = "ATGCTTTTGCATGCATGCAT"
        report = score_guide(guide)
        homo = [i for i in report.issues if i.check_name == "HOMOPOLYMER"]
        assert any(i.severity == "HIGH" for i in homo)

    def test_cccc_moderate(self):
        guide = "ATGCCCCCGATGCATGCATG"
        report = score_guide(guide)
        homo = [i for i in report.issues if i.check_name == "HOMOPOLYMER"]
        assert any(i.severity == "MODERATE" for i in homo)

    def test_no_homopolymer_clean(self):
        guide = "ATGCATGCATGCATGCATCG"
        report = score_guide(guide)
        homo = [i for i in report.issues if i.check_name == "HOMOPOLYMER"]
        assert len(homo) == 0

    def test_tttt_kills_activity(self):
        guide = "ATGCTTTTGCATGCATGCAT"
        report = score_guide(guide)
        assert report.activity_score < 100  # -30 penalty


# ─── Terminal GC checks ──────────────────────────────────────────────────────

class TestTerminalGC:
    def test_both_at_flagged(self):
        guide = "ATGCATGCATGCATGCATGA"  # starts A, ends A
        report = score_guide(guide)
        term = [i for i in report.issues if i.check_name == "TERMINAL_GC"]
        assert len(term) > 0

    def test_gc_at_ends_clean(self):
        guide = "GATGCATGCATGCATGCATC"  # starts G, ends C
        report = score_guide(guide)
        term = [i for i in report.issues if i.check_name == "TERMINAL_GC"]
        assert len(term) == 0


# ─── Position 20 check ───────────────────────────────────────────────────────

class TestPos20:
    def test_not_g_flagged(self):
        guide = "ATGCATGCATGCATGCATCA"  # ends A
        report = score_guide(guide)
        p20 = [i for i in report.issues if i.check_name == "POS20"]
        assert len(p20) > 0
        assert p20[0].severity == "LOW"

    def test_g_at_end_clean(self):
        guide = "ATGCATGCATGCATGCATCG"  # ends G
        report = score_guide(guide)
        p20 = [i for i in report.issues if i.check_name == "POS20"]
        assert len(p20) == 0


# ─── Seed GC checks ──────────────────────────────────────────────────────────

class TestSeedGC:
    def test_high_seed_gc_moderate(self):
        # First 12 all GC
        guide = "GCGCGCGCGCGCATATATAT"
        report = score_guide(guide)
        seed = [i for i in report.issues if i.check_name == "SEED_GC"]
        assert any(i.severity == "MODERATE" for i in seed)

    def test_low_seed_gc_high(self):
        # First 12 all AT
        guide = "AATATATATATAGATGCGCG"
        report = score_guide(guide)
        seed = [i for i in report.issues if i.check_name == "SEED_GC"]
        assert any(i.severity == "HIGH" for i in seed)


# ─── Off-target risk ─────────────────────────────────────────────────────────

class TestOfftargetRisk:
    def test_high_seed_gc_high_risk(self):
        guide = "GCGCGCGCGCGCATATATAT"
        report = score_guide(guide)
        assert report.offtarget_risk in ("HIGH", "MODERATE")

    def test_balanced_guide_low_risk(self):
        # Non-palindromic, balanced GC, no self-comp
        guide = "AGTCTAGCAGTCTAGCATCG"
        report = score_guide(guide)
        assert report.offtarget_risk == "LOW"


# ─── Activity score ──────────────────────────────────────────────────────────

class TestActivityScore:
    def test_perfect_guide_100(self):
        # 50% GC, ends G, G/C at terminals, balanced seed, no self-comp
        guide = "GAGTCTAGCAGTCTAGCACG"
        report = score_guide(guide)
        assert report.activity_score >= 90

    def test_terrible_guide_low(self):
        # 0% GC, TTTT run, starts A ends A
        guide = "AAAAATTTTTAAATAAAAAA"
        report = score_guide(guide)
        assert report.activity_score < 50
        assert report.verdict == "FAIL"

    def test_score_never_negative(self):
        guide = "TTTTTTTTTTTTTTTTTTTT"
        report = score_guide(guide)
        assert report.activity_score >= 0


# ─── Verdict logic ────────────────────────────────────────────────────────────

class TestVerdict:
    def test_fail_below_50(self):
        guide = "AAAAATTTTTAAATAAAAAA"
        report = score_guide(guide)
        assert report.verdict == "FAIL"

    def test_pass_above_70(self):
        guide = "ATGCATGCATGCATGCATCG"
        report = score_guide(guide)
        assert report.verdict == "PASS"


# ─── Input validation ────────────────────────────────────────────────────────

class TestInputValidation:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            score_guide("")

    def test_invalid_bases_raise(self):
        with pytest.raises(ValueError, match="Invalid"):
            score_guide("ATGCXYZ")

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="length"):
            score_guide("ATGC")

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="length"):
            score_guide("A" * 30)

    def test_unknown_pam_raises(self):
        with pytest.raises(ValueError, match="PAM"):
            score_guide("ATGCATGCATGCATGCATCG", pam="XYZ")

    def test_case_insensitive(self):
        report = score_guide("atgcatgcatgcatgcatcg")
        assert report.length == 20

    def test_17nt_accepted(self):
        report = score_guide("ATGCATGCATGCATGCA")
        assert report.length == 17

    def test_25nt_accepted(self):
        report = score_guide("ATGCATGCATGCATGCATGCATGCA")
        assert report.length == 25


# ─── Batch scoring ────────────────────────────────────────────────────────────

class TestBatchScoring:
    def test_returns_list(self):
        guides = ["ATGCATGCATGCATGCATCG", "GCGCGCGCGCGCGCGCGCGC"]
        reports = score_guides(guides)
        assert len(reports) == 2
        assert all(isinstance(r, GuideReport) for r in reports)

    def test_empty_list(self):
        reports = score_guides([])
        assert reports == []


# ─── Off-target pair comparison ───────────────────────────────────────────────

class TestOfftargetPair:
    def test_identical_zero_mismatches(self):
        guide = "ATGCATGCATGCATGCATCG"
        result = check_offtarget_pair(guide, guide)
        assert result["n_mismatches"] == 0
        assert result["risk_level"] == "HIGH"  # 0 seed mismatches

    def test_one_seed_mismatch(self):
        guide = "ATGCATGCATGCATGCATCG"
        off   = "TTGCATGCATGCATGCATCG"  # pos 1 mismatch (seed)
        result = check_offtarget_pair(guide, off)
        assert result["seed_mismatches"] == 1
        assert result["non_seed_mismatches"] == 0
        assert result["risk_level"] == "HIGH"

    def test_many_seed_mismatches_low_risk(self):
        guide = "ATGCATGCATGCATGCATCG"
        off   = "TACGTACGTACGATGCATCG"  # many seed mismatches
        result = check_offtarget_pair(guide, off)
        assert result["seed_mismatches"] > 4
        assert result["risk_level"] == "LOW"

    def test_non_seed_mismatch(self):
        guide = "ATGCATGCATGCATGCATCG"
        off   = "ATGCATGCATGCATGCATCC"  # pos 20 mismatch (non-seed)
        result = check_offtarget_pair(guide, off)
        assert result["seed_mismatches"] == 0
        assert result["non_seed_mismatches"] == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length"):
            check_offtarget_pair("ATGCATGC", "ATGC")

    def test_invalid_bases_raise(self):
        with pytest.raises(ValueError, match="Invalid"):
            check_offtarget_pair("ATGCXYZ", "ATGCXYZ")

    def test_mismatch_positions_1indexed(self):
        guide = "ATGCATGCATGCATGCATCG"
        off   = "TTGCATGCATGCATGCATCG"
        result = check_offtarget_pair(guide, off)
        assert 1 in result["mismatch_positions"]


# ─── Report formatting ───────────────────────────────────────────────────────

class TestReportFormat:
    def test_str_contains_verdict(self):
        report = score_guide("ATGCATGCATGCATGCATCG")
        s = str(report)
        assert "PASS" in s or "WARN" in s or "FAIL" in s

    def test_str_contains_sequence(self):
        guide = "ATGCATGCATGCATGCATCG"
        report = score_guide(guide)
        assert guide in str(report)

    def test_str_contains_activity_score(self):
        report = score_guide("ATGCATGCATGCATGCATCG")
        assert "Activity score" in str(report)
