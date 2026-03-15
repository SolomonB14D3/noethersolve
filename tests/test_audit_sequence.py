"""Tests for noethersolve.audit_sequence — DNA/RNA sequence design auditor."""

import pytest
from noethersolve.audit_sequence import (
    audit_sequence,
    SequenceReport,
    SequenceIssue,
    gc_content,
    cpg_observed_expected,
)


# ─── gc_content helper ────────────────────────────────────────────────────────

class TestGCContent:
    def test_all_gc(self):
        assert gc_content("GCGCGC") == 1.0

    def test_all_at(self):
        assert gc_content("ATATAT") == 0.0

    def test_mixed(self):
        assert abs(gc_content("ATGC") - 0.5) < 1e-9

    def test_empty(self):
        assert gc_content("") == 0.0

    def test_case_insensitive(self):
        assert gc_content("atgc") == gc_content("ATGC")

    def test_rna_u(self):
        assert gc_content("AUGC") == gc_content("ATGC")

    def test_whitespace_stripped(self):
        assert gc_content("AT GC") == gc_content("ATGC")


# ─── cpg_observed_expected helper ─────────────────────────────────────────────

class TestCpGOE:
    def test_no_cpg(self):
        assert cpg_observed_expected("AAAA") == 0.0

    def test_pure_cg(self):
        # CGCGCG: 3 CpG dinucleotides, 3 C, 3 G, length 6
        # O/E = (3 * 6) / (3 * 3) = 2.0
        oe = cpg_observed_expected("CGCGCG")
        assert abs(oe - 2.0) < 1e-9

    def test_empty(self):
        assert cpg_observed_expected("") == 0.0

    def test_single_base(self):
        assert cpg_observed_expected("C") == 0.0

    def test_no_g(self):
        assert cpg_observed_expected("CCCC") == 0.0


# ─── CpG density check ───────────────────────────────────────────────────────

class TestCpGDensity:
    def test_high_cpg(self):
        # Create sequence with >4% CpG density
        # 20 bases, need >0.04 * 19 = 0.76 → 1+ CpG in 19 dinucleotides
        # Actually need many CpGs: CGCGCG... has 50% CpG density
        seq = "CGCGCGCGCGCGCGCGCGCG"
        report = audit_sequence(seq)
        cpg_issues = [i for i in report.issues if i.check_name == "CpG_DENSITY"]
        assert len(cpg_issues) > 0
        assert cpg_issues[0].severity == "HIGH"

    def test_no_cpg_is_clean(self):
        # All AT sequence has 0 CpG
        report = audit_sequence("AATTAATTAATTAATTAATT")
        cpg_issues = [i for i in report.issues if i.check_name == "CpG_DENSITY"]
        assert len(cpg_issues) == 0


# ─── GC content check ────────────────────────────────────────────────────────

class TestGCCheck:
    def test_high_gc_flagged(self):
        seq = "GCGCGCGCGCGCGCGCGCGC"  # 100% GC
        report = audit_sequence(seq)
        gc_issues = [i for i in report.issues if i.check_name == "GC_CONTENT"]
        assert len(gc_issues) > 0
        assert gc_issues[0].severity == "HIGH"

    def test_low_gc_flagged(self):
        seq = "AAAAAAAAAAAAAAATAAAA"  # ~5% GC
        report = audit_sequence(seq)
        gc_issues = [i for i in report.issues if i.check_name == "GC_CONTENT"]
        assert len(gc_issues) > 0
        assert gc_issues[0].severity == "HIGH"

    def test_optimal_gc_clean(self):
        # ~50% GC
        seq = "ATGCATGCATGCATGCATGC"
        report = audit_sequence(seq)
        gc_issues = [i for i in report.issues if i.check_name == "GC_CONTENT"]
        assert len(gc_issues) == 0

    def test_moderate_gc(self):
        # 35% GC = MODERATE
        seq = "AAAAAAAATTTTTGCGCGCA"  # 7 G/C out of 20 = 35%
        report = audit_sequence(seq)
        gc_issues = [i for i in report.issues if i.check_name == "GC_CONTENT"]
        assert any(i.severity == "MODERATE" for i in gc_issues)


# ─── Homopolymer check ───────────────────────────────────────────────────────

class TestHomopolymers:
    def test_tttt_moderate(self):
        seq = "ATGCTTTTGCATGCATGCAT"
        report = audit_sequence(seq)
        homo_issues = [i for i in report.issues if i.check_name == "HOMOPOLYMER"]
        assert any(i.severity == "MODERATE" for i in homo_issues)

    def test_six_run_high(self):
        seq = "ATGCAAAAAAGCATGCATGC"
        report = audit_sequence(seq)
        homo_issues = [i for i in report.issues if i.check_name == "HOMOPOLYMER"]
        assert any(i.severity == "HIGH" for i in homo_issues)

    def test_short_runs_clean(self):
        seq = "ATGCATGCATGCATGCATGC"
        report = audit_sequence(seq)
        homo_issues = [i for i in report.issues if i.check_name == "HOMOPOLYMER"]
        assert len(homo_issues) == 0

    def test_position_reported(self):
        seq = "ATGCTTTTGCATGCATGCAT"
        report = audit_sequence(seq)
        homo_issues = [i for i in report.issues if i.check_name == "HOMOPOLYMER"]
        assert homo_issues[0].position == 4  # 0-indexed


# ─── Poly-A signal check ─────────────────────────────────────────────────────

class TestPolyASignal:
    def test_aataaa_flagged(self):
        seq = "ATGCAATAAAGCATGCATGC"
        report = audit_sequence(seq)
        polya = [i for i in report.issues if i.check_name == "POLYA_SIGNAL"]
        assert len(polya) == 1
        assert polya[0].severity == "HIGH"

    def test_attaaa_flagged(self):
        seq = "ATGCATTAAAGCATGCATGC"
        report = audit_sequence(seq)
        polya = [i for i in report.issues if i.check_name == "POLYA_SIGNAL"]
        assert len(polya) == 1

    def test_no_polya_clean(self):
        seq = "ATGCATGCATGCATGCATGC"
        report = audit_sequence(seq)
        polya = [i for i in report.issues if i.check_name == "POLYA_SIGNAL"]
        assert len(polya) == 0

    def test_multiple_signals_counted(self):
        seq = "AATAAAGCATAATAAAGCAT"
        report = audit_sequence(seq)
        polya = [i for i in report.issues if i.check_name == "POLYA_SIGNAL"]
        assert len(polya) == 2


# ─── Self-complementarity check ──────────────────────────────────────────────

class TestSelfComp:
    def test_palindrome_detected(self):
        # AATTAATT is its own reverse complement
        seq = "GCGCAATTAATTGCGCATGC"
        report = audit_sequence(seq)
        sc = [i for i in report.issues if i.check_name == "SELF_COMPLEMENT"]
        assert len(sc) > 0

    def test_short_seq_no_check(self):
        report = audit_sequence("ATGC")
        sc = [i for i in report.issues if i.check_name == "SELF_COMPLEMENT"]
        assert len(sc) == 0


# ─── Verdict logic ────────────────────────────────────────────────────────────

class TestVerdict:
    def test_fail_on_high(self):
        # Poly-A signal = HIGH
        seq = "ATGCAATAAAGCATGCATGC"
        report = audit_sequence(seq)
        assert report.verdict == "FAIL"
        assert not report.passed

    def test_pass_on_clean(self):
        # ~50% GC, no palindromes, no issues
        seq = "AGTCTAGCAGTCTAGCAGTC"
        report = audit_sequence(seq)
        assert report.verdict == "PASS"
        assert report.passed

    def test_empty_seq_passes(self):
        report = audit_sequence("")
        assert report.verdict == "PASS"
        assert report.sequence_length == 0


# ─── Input validation ────────────────────────────────────────────────────────

class TestInputValidation:
    def test_invalid_chars_raise(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            audit_sequence("ATGCXYZ")

    def test_invalid_seq_type(self):
        with pytest.raises(ValueError, match="seq_type"):
            audit_sequence("ATGC", seq_type="protein")

    def test_rna_accepted(self):
        report = audit_sequence("AUGCAUGC", seq_type="rna")
        assert report.sequence_length == 8

    def test_case_insensitive(self):
        report = audit_sequence("atgcatgc")
        assert report.sequence_length == 8

    def test_whitespace_stripped(self):
        report = audit_sequence("ATGC ATGC")
        assert report.sequence_length == 8


# ─── Report formatting ───────────────────────────────────────────────────────

class TestReportFormat:
    def test_str_contains_verdict(self):
        report = audit_sequence("AGTCTAGCAGTCTAGCAGTC")
        s = str(report)
        assert "PASS" in s

    def test_str_contains_gc(self):
        report = audit_sequence("AGTCTAGCAGTCTAGCAGTC")
        s = str(report)
        assert "GC content" in s

    def test_str_contains_issues(self):
        report = audit_sequence("ATGCAATAAAGCATGCATGC")
        s = str(report)
        assert "Issues found" in s

    def test_summary_metrics_populated(self):
        seq = "AGTCTAGCAGTCTAGCAGTC"
        report = audit_sequence(seq)
        assert report.sequence_length == 20
        assert 0.0 <= report.gc_content <= 1.0
        assert report.cpg_density >= 0.0
        assert report.longest_homopolymer >= 1
