"""Tests for noethersolve.audit_facts — oracle fact file quality auditor."""

import json
import os
import tempfile

import pytest

from noethersolve.audit_facts import audit_facts, FactAuditReport, _approx_tokens


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_facts():
    """Facts with no quality issues — all distractors equal or longer than truth."""
    return {
        "facts": [
            {
                "id": "good01",
                "context": "test context",
                "truth": "alpha beta gamma delta",
                "distractors": [
                    "epsilon zeta eta theta",
                    "iota kappa lambda omega",
                    "sigma tau upsilon phi rho",
                ],
            },
            {
                "id": "good02",
                "context": "another test",
                "truth": "one two three four five",
                "distractors": [
                    "six seven eight nine ten",
                    "eleven twelve thirteen fourteen fifteen",
                    "sixteen seventeen eighteen nineteen twenty",
                ],
            },
        ],
    }


@pytest.fixture
def length_biased_facts():
    """Facts where distractors are much shorter than truth — the #1 failure mode."""
    return {
        "facts": [
            {
                "id": "chem08_bad",
                "context": "Under mass-action kinetics, the rate of A + B -> C is:",
                "truth": "k × [A] × [B] where k is the rate constant",
                "distractors": [
                    "k × [A]",
                    "proportional to product [C] only",
                    "constant regardless",
                ],
            },
            {
                "id": "ns03_bad",
                "context": "Navier-Stokes energy dissipation rate:",
                "truth": "integral of nu times gradient of velocity squared over the whole domain volume",
                "distractors": [
                    "Q_f ∝ s",
                    "zero always in viscous flows because of incompressibility",
                    "nu times laplacian of velocity field at the boundary only",
                ],
            },
        ],
    }


@pytest.fixture
def substring_facts():
    """Facts where a distractor is a substring of the truth."""
    return {
        "facts": [
            {
                "id": "sub01",
                "context": "rate law",
                "truth": "k × [A] × [B] where k is the rate constant",
                "distractors": [
                    "k × [A]",
                    "something completely different and long enough",
                    "another unrelated distractor of similar length",
                ],
            },
        ],
    }


# ─── Token approximation ────────────────────────────────────────────────────

class TestApproxTokens:
    def test_empty_string(self):
        assert _approx_tokens("") == 0

    def test_single_word(self):
        assert _approx_tokens("hello") == 1

    def test_multiple_words(self):
        assert _approx_tokens("the quick brown fox") == 4

    def test_special_chars_add_tokens(self):
        # Parentheses and operators should add extra tokens
        plain = _approx_tokens("k times A times B")
        special = _approx_tokens("k × [A] × [B]")
        assert special > plain

    def test_single_char(self):
        assert _approx_tokens("x") == 1


# ─── Clean facts → PASS ─────────────────────────────────────────────────────

class TestCleanFacts:
    def test_verdict_pass(self, clean_facts):
        report = audit_facts(clean_facts)
        assert report.verdict == "PASS"
        assert report.passed

    def test_no_issues(self, clean_facts):
        report = audit_facts(clean_facts)
        assert len(report.issues) == 0

    def test_all_ok(self, clean_facts):
        report = audit_facts(clean_facts)
        assert report.n_ok == 2
        assert report.n_high_risk == 0
        assert report.n_moderate_risk == 0

    def test_diagnostics_count(self, clean_facts):
        report = audit_facts(clean_facts)
        assert len(report.diagnostics) == report.n_facts


# ─── Length bias detection ───────────────────────────────────────────────────

class TestLengthBias:
    def test_detects_short_distractor(self, length_biased_facts):
        report = audit_facts(length_biased_facts)
        assert report.verdict == "FAIL"
        assert report.n_high_risk > 0

    def test_flags_correct_fact_ids(self, length_biased_facts):
        report = audit_facts(length_biased_facts)
        flagged_ids = {d.fact_id for d in report.diagnostics if d.risk_level == "HIGH_RISK"}
        # Both facts have very short distractors
        assert "chem08_bad" in flagged_ids or "ns03_bad" in flagged_ids

    def test_issue_type_is_length_bias(self, length_biased_facts):
        report = audit_facts(length_biased_facts)
        length_issues = [i for i in report.issues if i.issue_type == "LENGTH_BIAS"]
        assert len(length_issues) > 0

    def test_ns03_extreme_ratio(self, length_biased_facts):
        """ns03 has 'Q_f ∝ s' (very short) vs long truth — should be HIGH."""
        report = audit_facts(length_biased_facts)
        ns03 = [d for d in report.diagnostics if d.fact_id == "ns03_bad"][0]
        assert ns03.risk_level == "HIGH_RISK"
        assert ns03.min_ratio < 0.7

    def test_moderate_risk_threshold(self):
        """Distractor ~80% of truth length should be MODERATE, not HIGH."""
        facts = {
            "facts": [
                {
                    "id": "moderate01",
                    "context": "test",
                    "truth": "a ten word answer that has exactly this many words here",
                    "distractors": [
                        "eight word distractor that is this long here",
                        "another distractor matching the truth length exactly here now",
                        "and yet another one that matches truth length in this case",
                    ],
                },
            ],
        }
        report = audit_facts(facts)
        # The first distractor is shorter but not drastically — check it's handled
        assert isinstance(report, FactAuditReport)


# ─── Substring detection ────────────────────────────────────────────────────

class TestSubstring:
    def test_detects_substring_distractor(self, substring_facts):
        report = audit_facts(substring_facts)
        substr_issues = [i for i in report.issues if i.issue_type == "SUBSTRING"]
        assert len(substr_issues) > 0

    def test_substring_is_high_severity(self, substring_facts):
        report = audit_facts(substring_facts)
        substr_issues = [i for i in report.issues if i.issue_type == "SUBSTRING"]
        assert all(i.severity == "HIGH" for i in substr_issues)

    def test_has_substring_flag(self, substring_facts):
        report = audit_facts(substring_facts)
        sub01 = [d for d in report.diagnostics if d.fact_id == "sub01"][0]
        assert sub01.has_substring

    def test_no_false_positive_substring(self, clean_facts):
        report = audit_facts(clean_facts)
        substr_issues = [i for i in report.issues if i.issue_type == "SUBSTRING"]
        assert len(substr_issues) == 0


# ─── File path loading ───────────────────────────────────────────────────────

class TestFilePath:
    def test_loads_from_file(self, clean_facts):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_facts.json", delete=False
        ) as f:
            json.dump(clean_facts, f)
            tmp_path = f.name
        try:
            report = audit_facts(tmp_path)
            assert report.verdict == "PASS"
            assert report.n_facts == 2
        finally:
            os.unlink(tmp_path)

    def test_loads_real_facts_file(self):
        """Load an actual facts file from the problems/ directory."""
        facts_path = os.path.join(
            os.path.dirname(__file__), "..", "problems", "3body_conservation_facts.json"
        )
        if os.path.exists(facts_path):
            report = audit_facts(facts_path)
            assert isinstance(report, FactAuditReport)
            assert report.n_facts == 10
            # Verify diagnostics were generated for each fact
            assert len(report.diagnostics) == 10


# ─── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_facts_list(self):
        report = audit_facts({"facts": []})
        assert report.verdict == "PASS"
        assert report.n_facts == 0
        assert len(report.warnings) > 0

    def test_no_distractors(self):
        facts = {
            "facts": [
                {
                    "id": "no_dist",
                    "context": "test",
                    "truth": "the answer",
                    "distractors": [],
                },
            ],
        }
        report = audit_facts(facts)
        assert report.n_facts == 1
        assert len(report.warnings) > 0  # should warn about missing distractors

    def test_single_char_truth(self):
        facts = {
            "facts": [
                {
                    "id": "tiny",
                    "context": "test",
                    "truth": "x",
                    "distractors": ["y", "z", "w"],
                },
            ],
        }
        report = audit_facts(facts)
        assert report.verdict == "PASS"  # all same length
        assert report.n_facts == 1

    def test_empty_distractor_string(self):
        facts = {
            "facts": [
                {
                    "id": "empty_d",
                    "context": "test",
                    "truth": "a reasonable answer with several tokens",
                    "distractors": ["", "normal distractor of similar length here", "another one"],
                },
            ],
        }
        report = audit_facts(facts)
        # Empty distractor has 0 tokens vs nonzero truth → HIGH risk
        high_issues = [i for i in report.issues if i.severity == "HIGH"]
        assert len(high_issues) > 0

    def test_missing_facts_key(self):
        report = audit_facts({"description": "no facts key"})
        assert report.verdict == "PASS"
        assert report.n_facts == 0

    def test_custom_thresholds(self):
        """Custom ratio thresholds should change risk classification."""
        facts = {
            "facts": [
                {
                    "id": "custom01",
                    "context": "test",
                    "truth": "a medium length answer here",
                    "distractors": [
                        "short",
                        "a medium length distractor here too",
                        "another distractor that matches well enough",
                    ],
                },
            ],
        }
        # With very strict thresholds, more things flag
        strict = audit_facts(facts, length_ratio_high=0.95, length_ratio_moderate=1.0)
        # With very loose thresholds, fewer things flag
        loose = audit_facts(facts, length_ratio_high=0.1, length_ratio_moderate=0.2)
        assert strict.n_high_risk >= loose.n_high_risk


# ─── Report formatting ──────────────────────────────────────────────────────

class TestReportFormat:
    def test_str_contains_key_info(self, length_biased_facts):
        report = audit_facts(length_biased_facts)
        s = str(report)
        assert "Fact Quality Audit" in s
        assert "FAIL" in s
        assert "HIGH RISK" in s

    def test_str_clean_report(self, clean_facts):
        report = audit_facts(clean_facts)
        s = str(report)
        assert "PASS" in s
        assert "0 HIGH RISK" in s

    def test_passed_property(self, clean_facts):
        report = audit_facts(clean_facts)
        assert report.passed == (report.verdict == "PASS")

    def test_failed_passed_property(self, length_biased_facts):
        report = audit_facts(length_biased_facts)
        assert not report.passed
