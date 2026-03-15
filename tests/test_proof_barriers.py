"""Tests for noethersolve.proof_barriers — Proof technique barrier checker."""

import pytest

from noethersolve.proof_barriers import (
    check_barriers,
    list_barriers,
    list_techniques,
    get_barrier,
    what_works_for,
    BarrierReport,
    BarrierIssue,
    BarrierInfo,
)


# ─── check_barriers — blocked techniques ─────────────────────────────────────

class TestCheckBarriersBlocked:
    def test_diagonalization_p_vs_np_fails(self):
        report = check_barriers("diagonalization", "P vs NP")
        assert report.verdict == "FAIL"
        assert report.n_high > 0

    def test_simulation_p_vs_np_fails(self):
        report = check_barriers("simulation", "P vs NP")
        assert report.verdict == "FAIL"

    def test_natural_proof_p_vs_np_fails(self):
        report = check_barriers("natural proof", "P vs NP")
        assert report.verdict == "FAIL"

    def test_arithmetization_p_vs_np_fails(self):
        """Algebrization barrier blocks arithmetization for P vs NP."""
        report = check_barriers("arithmetization", "P vs NP")
        assert report.verdict == "FAIL"

    def test_zfc_proof_continuum_hypothesis_fails(self):
        report = check_barriers("ZFC proof", "continuum hypothesis")
        assert report.verdict == "FAIL"

    def test_monotone_circuit_lower_bounds_fails(self):
        report = check_barriers("monotone arguments", "circuit lower bounds")
        assert report.verdict == "FAIL"

    def test_complete_axiomatization_blocked(self):
        report = check_barriers("complete axiomatization", "complete axiomatization of arithmetic")
        assert report.verdict == "FAIL"

    def test_sos_proof_complexity_fails(self):
        report = check_barriers("SOS", "proof complexity")
        assert report.verdict == "FAIL"

    def test_weisfeiler_leman_graph_isomorphism_fails(self):
        report = check_barriers("Weisfeiler-Leman", "graph isomorphism")
        assert report.verdict == "FAIL"

    def test_counting_p_vs_np_fails(self):
        """Counting arguments relativize, blocked for P vs NP."""
        report = check_barriers("counting", "P vs NP")
        assert report.verdict == "FAIL"


# ─── check_barriers — unblocked techniques ───────────────────────────────────

class TestCheckBarriersUnblocked:
    def test_interactive_proofs_p_vs_np_passes(self):
        report = check_barriers("interactive proofs", "P vs NP")
        assert report.verdict == "PASS"

    def test_novel_technique_passes(self):
        """A technique not in any barrier database should pass."""
        report = check_barriers("quantum error correction", "P vs NP")
        assert report.verdict == "PASS"

    def test_unrelated_technique_unrelated_problem(self):
        report = check_barriers("machine learning", "protein folding")
        assert report.verdict == "PASS"

    def test_forcing_continuum_hypothesis_passes(self):
        """Forcing is not blocked for CH (it's a positive approach, not ZFC proof)."""
        report = check_barriers("forcing axioms", "continuum hypothesis")
        assert report.verdict == "PASS"


# ─── check_barriers — alias normalization ────────────────────────────────────

class TestCheckBarriersAliases:
    def test_diagonal_argument_alias(self):
        report = check_barriers("diagonal argument", "P vs NP")
        assert report.verdict == "FAIL"

    def test_p_equals_np_alias(self):
        report = check_barriers("diagonalization", "p vs np")
        assert report.verdict == "FAIL"

    def test_p_neq_np_alias(self):
        report = check_barriers("diagonalization", "p != np")
        assert report.verdict == "FAIL"

    def test_sos_lowercase_alias(self):
        report = check_barriers("sos", "proof complexity")
        assert report.verdict == "FAIL"

    def test_natural_alias(self):
        report = check_barriers("natural", "P vs NP")
        assert report.verdict == "FAIL"

    def test_wl_alias(self):
        report = check_barriers("wl", "graph isomorphism")
        assert report.verdict == "FAIL"

    def test_zfc_alias(self):
        report = check_barriers("zfc", "continuum hypothesis")
        assert report.verdict == "FAIL"


# ─── check_barriers — report structure ───────────────────────────────────────

class TestCheckBarriersReport:
    def test_report_has_technique(self):
        report = check_barriers("diagonalization", "P vs NP")
        assert report.technique == "diagonalization"

    def test_report_has_target(self):
        report = check_barriers("diagonalization", "P vs NP")
        assert report.target == "P vs NP"

    def test_report_issues_nonempty_on_fail(self):
        report = check_barriers("diagonalization", "P vs NP")
        assert len(report.issues) > 0

    def test_report_suggestions_on_fail(self):
        report = check_barriers("diagonalization", "P vs NP")
        assert len(report.suggestions) > 0

    def test_report_severity_counts(self):
        report = check_barriers("diagonalization", "P vs NP")
        total = report.n_high + report.n_moderate + report.n_low + report.n_info
        assert total == len(report.issues)

    def test_report_passed_property(self):
        report_fail = check_barriers("diagonalization", "P vs NP")
        assert report_fail.passed is False
        report_pass = check_barriers("interactive proofs", "P vs NP")
        assert report_pass.passed is True

    def test_report_str_contains_verdict(self):
        report = check_barriers("diagonalization", "P vs NP")
        text = str(report)
        assert "FAIL" in text

    def test_report_str_contains_technique(self):
        report = check_barriers("diagonalization", "P vs NP")
        text = str(report)
        assert "diagonalization" in text

    def test_report_str_contains_target(self):
        report = check_barriers("diagonalization", "P vs NP")
        text = str(report)
        assert "P vs NP" in text


# ─── list_barriers ───────────────────────────────────────────────────────────

class TestListBarriers:
    def test_returns_list(self):
        barriers = list_barriers()
        assert isinstance(barriers, list)
        assert len(barriers) >= 10

    def test_sorted_by_year(self):
        barriers = list_barriers()
        years = [b.year for b in barriers]
        assert years == sorted(years)

    def test_relativization_present(self):
        barriers = list_barriers()
        names = [b.name for b in barriers]
        assert "relativization" in names

    def test_natural_proofs_present(self):
        barriers = list_barriers()
        names = [b.name for b in barriers]
        assert "natural proofs" in names

    def test_barrier_info_fields(self):
        barriers = list_barriers()
        b = barriers[0]
        assert isinstance(b.name, str)
        assert isinstance(b.authors, str)
        assert isinstance(b.year, int)
        assert isinstance(b.summary, str)
        assert isinstance(b.blocked_problems, frozenset)
        assert isinstance(b.blocked_techniques, frozenset)


# ─── list_techniques ─────────────────────────────────────────────────────────

class TestListTechniques:
    def test_returns_sorted_list(self):
        techs = list_techniques()
        assert isinstance(techs, list)
        assert techs == sorted(techs)

    def test_diagonalization_present(self):
        techs = list_techniques()
        assert "diagonalization" in techs

    def test_natural_proof_present(self):
        techs = list_techniques()
        assert "natural proof" in techs

    def test_sos_present(self):
        techs = list_techniques()
        assert "SOS" in techs

    def test_nonempty(self):
        techs = list_techniques()
        assert len(techs) >= 10


# ─── get_barrier ─────────────────────────────────────────────────────────────

class TestGetBarrier:
    def test_get_relativization(self):
        info = get_barrier("relativization")
        assert info.name == "relativization"
        assert info.year == 1975
        assert "Baker" in info.authors

    def test_get_natural_proofs(self):
        info = get_barrier("natural proofs")
        assert info.name == "natural proofs"
        assert info.year == 1997
        assert "Razborov" in info.authors

    def test_get_algebrization(self):
        info = get_barrier("algebrization")
        assert info.year == 2009
        assert "Aaronson" in info.authors

    def test_get_incompleteness(self):
        info = get_barrier("incompleteness")
        assert info.year == 1931
        assert "Godel" in info.authors

    def test_case_insensitive(self):
        info = get_barrier("RELATIVIZATION")
        assert info.name == "relativization"

    def test_unknown_barrier_raises(self):
        with pytest.raises(KeyError):
            get_barrier("nonexistent barrier")

    def test_barrier_info_str(self):
        info = get_barrier("relativization")
        text = str(info)
        assert "Baker" in text
        assert "1975" in text
        assert "relativization" in text


# ─── what_works_for ──────────────────────────────────────────────────────────

class TestWhatWorksFor:
    def test_p_vs_np_has_suggestions(self):
        suggestions = what_works_for("P vs NP")
        assert len(suggestions) > 0

    def test_circuit_lower_bounds_has_suggestions(self):
        suggestions = what_works_for("circuit lower bounds")
        assert len(suggestions) > 0

    def test_derandomization_has_suggestions(self):
        suggestions = what_works_for("derandomization")
        assert len(suggestions) > 0

    def test_continuum_hypothesis_has_suggestions(self):
        suggestions = what_works_for("continuum hypothesis")
        assert len(suggestions) > 0

    def test_unknown_problem_empty(self):
        suggestions = what_works_for("protein folding prediction")
        assert suggestions == []

    def test_alias_normalization(self):
        suggestions = what_works_for("p vs np")
        assert len(suggestions) > 0

    def test_returns_list_of_strings(self):
        suggestions = what_works_for("P vs NP")
        assert all(isinstance(s, str) for s in suggestions)


# ─── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_technique_string(self):
        report = check_barriers("", "P vs NP")
        # Should not crash; verdict depends on whether empty string matches anything
        assert report.verdict in ("PASS", "WARN", "FAIL")

    def test_empty_target_string(self):
        report = check_barriers("diagonalization", "")
        assert report.verdict in ("PASS", "WARN", "FAIL")

    def test_barrier_blocked_problems_nonempty(self):
        """Every barrier should block at least one problem."""
        for barrier in list_barriers():
            assert len(barrier.blocked_problems) > 0

    def test_barrier_blocked_techniques_nonempty(self):
        """Every barrier should block at least one technique."""
        for barrier in list_barriers():
            assert len(barrier.blocked_techniques) > 0

    def test_info_issues_have_references(self):
        """HIGH severity issues should include references."""
        report = check_barriers("diagonalization", "P vs NP")
        high_issues = [i for i in report.issues if i.severity == "HIGH"]
        for issue in high_issues:
            assert issue.reference != ""

    def test_info_issues_have_suggestions(self):
        """HIGH severity issues should include suggestions."""
        report = check_barriers("diagonalization", "P vs NP")
        high_issues = [i for i in report.issues if i.severity == "HIGH"]
        for issue in high_issues:
            assert issue.suggestion != ""
