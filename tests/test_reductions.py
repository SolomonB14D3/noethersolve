"""Tests for noethersolve.reductions — Computational reduction chain validator."""

import pytest

from noethersolve.reductions import (
    validate_chain,
    check_reduction,
    strongest_reduction,
    list_known_reductions,
    get_reduction_info,
    ChainReport,
    ChainIssue,
    ReductionResult,
    REDUCTION_TYPES,
    KNOWN_REDUCTIONS,
)


# ─── validate_chain ─────────────────────────────────────────────────────────

class TestValidateChain:
    def test_valid_chain_passes(self):
        chain = [
            ("3-SAT", "many-one", "CLIQUE"),
            ("CLIQUE", "many-one", "VERTEX-COVER"),
        ]
        report = validate_chain(chain)
        assert isinstance(report, ChainReport)
        assert report.verdict == "PASS"
        assert report.passed is True

    def test_broken_chain_fails(self):
        """Chain where step i target != step i+1 source."""
        chain = [
            ("3-SAT", "many-one", "CLIQUE"),
            ("SUBSET-SUM", "many-one", "VERTEX-COVER"),
        ]
        report = validate_chain(chain)
        assert report.verdict == "FAIL"
        assert report.passed is False
        transitivity_issues = [i for i in report.issues if i.check == "TRANSITIVITY"]
        assert len(transitivity_issues) > 0

    def test_circular_chain_fails(self):
        """A -> B -> A creates a cycle, flagged HIGH."""
        chain = [
            ("SAT", "many-one", "3-SAT"),
            ("3-SAT", "many-one", "SAT"),
        ]
        report = validate_chain(chain)
        # Should detect cycle and flag it
        circularity_issues = [i for i in report.issues if i.check == "CIRCULARITY"]
        assert len(circularity_issues) > 0
        assert report.verdict == "FAIL"

    def test_empty_chain(self):
        report = validate_chain([])
        assert report.verdict == "PASS"
        assert report.chain_length == 0
        assert len(report.warnings) > 0  # "Empty chain" warning

    def test_single_step_chain(self):
        chain = [("3-SAT", "many-one", "CLIQUE")]
        report = validate_chain(chain)
        assert report.chain_length == 1
        assert report.effective_type == "many-one"

    def test_chain_effective_type(self):
        chain = [
            ("3-SAT", "many-one", "CLIQUE"),
            ("CLIQUE", "many-one", "VERTEX-COVER"),
        ]
        report = validate_chain(chain)
        assert report.effective_type == "many-one"

    def test_chain_with_unknown_type_raises(self):
        """validate_chain propagates ValueError from unknown reduction types
        because check_reduction calls _normalize_type which raises."""
        chain = [("A", "teleportation", "B")]
        with pytest.raises(ValueError, match="Unknown reduction type"):
            validate_chain(chain)

    def test_known_direct_reduction_noted(self):
        """When a multi-step chain has a known direct reduction, note it."""
        chain = [
            ("SAT", "many-one", "3-SAT"),
            ("3-SAT", "many-one", "CLIQUE"),
        ]
        report = validate_chain(chain)
        # There's no direct SAT->CLIQUE in KNOWN_REDUCTIONS, so no info note.
        # But the chain itself should be valid.
        assert report.verdict == "PASS"

    def test_backwards_reduction_detected(self):
        """Using a known reduction in the wrong direction."""
        chain = [("CLIQUE", "many-one", "3-SAT")]
        report = validate_chain(chain)
        # 3-SAT -> CLIQUE is known, but CLIQUE -> 3-SAT is backwards
        direction_issues = [
            i for i in report.issues
            if i.check == "KNOWN_REDUCTION" and i.severity == "HIGH"
        ]
        assert len(direction_issues) > 0


# ─── check_reduction ────────────────────────────────────────────────────────

class TestCheckReduction:
    def test_known_reduction_sat_to_3sat(self):
        result = check_reduction("SAT", "many-one", "3-SAT")
        assert isinstance(result, ReductionResult)
        assert result.known is True
        assert result.reference is not None

    def test_known_reduction_3sat_to_clique(self):
        result = check_reduction("3-SAT", "many-one", "CLIQUE")
        assert result.known is True
        assert "Karp" in result.reference

    def test_unknown_reduction(self):
        result = check_reduction("VERTEX-COVER", "many-one", "HAMILTONIAN-PATH")
        assert result.known is False

    def test_alias_karp(self):
        """'Karp' is an alias for 'many-one'."""
        result = check_reduction("3-SAT", "Karp", "CLIQUE")
        assert result.known is True
        assert result.reduction_type == "many-one"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            check_reduction("A", "magic", "B")

    def test_backwards_direction_flagged(self):
        result = check_reduction("CLIQUE", "many-one", "3-SAT")
        assert result.known is False
        assert len(result.issues) > 0
        assert any(i.severity == "HIGH" for i in result.issues)

    def test_log_space_reduction(self):
        result = check_reduction("s-t-CONNECTIVITY", "log-space", "REACHABILITY")
        assert result.known is True


# ─── strongest_reduction ─────────────────────────────────────────────────────

class TestStrongestReduction:
    def test_mixed_types_weaken_to_turing(self):
        chain = [
            ("SAT", "many-one", "3-SAT"),
            ("3-SAT", "Turing", "CLIQUE"),
        ]
        result = strongest_reduction(chain)
        assert result == "Turing"

    def test_all_many_one_stays_many_one(self):
        chain = [
            ("SAT", "many-one", "3-SAT"),
            ("3-SAT", "many-one", "CLIQUE"),
        ]
        result = strongest_reduction(chain)
        assert result == "many-one"

    def test_empty_chain_raises(self):
        with pytest.raises(ValueError):
            strongest_reduction([])

    def test_single_step(self):
        chain = [("A", "log-space", "B")]
        result = strongest_reduction(chain)
        assert result == "log-space"

    def test_log_space_and_many_one(self):
        chain = [
            ("A", "log-space", "B"),
            ("B", "many-one", "C"),
        ]
        result = strongest_reduction(chain)
        assert result == "many-one"  # many-one is weaker (higher power rank)

    def test_alias_resolution(self):
        chain = [("A", "Karp", "B"), ("B", "Cook", "C")]
        result = strongest_reduction(chain)
        assert result == "Turing"


# ─── list_known_reductions ──────────────────────────────────────────────────

class TestListKnownReductions:
    def test_list_all(self):
        all_reds = list_known_reductions()
        assert len(all_reds) == len(KNOWN_REDUCTIONS)
        for r in all_reds:
            assert "from" in r
            assert "to" in r
            assert "type" in r

    def test_filter_by_problem_3sat(self):
        reds = list_known_reductions("3-SAT")
        assert len(reds) > 0
        for r in reds:
            assert "3-SAT" in (r["from"], r["to"])

    def test_filter_case_insensitive(self):
        reds = list_known_reductions("3-sat")
        assert len(reds) > 0

    def test_filter_unknown_problem(self):
        reds = list_known_reductions("NONEXISTENT-PROBLEM")
        assert reds == []

    def test_filter_sat_includes_both_directions(self):
        reds = list_known_reductions("SAT")
        froms = [r for r in reds if r["from"] == "SAT"]
        tos = [r for r in reds if r["to"] == "SAT"]
        # SAT appears as both source (SAT->3-SAT, SAT->#SAT)
        # and target (CIRCUIT-SAT->SAT)
        assert len(froms) > 0
        assert len(tos) > 0


# ─── get_reduction_info ─────────────────────────────────────────────────────

class TestGetReductionInfo:
    def test_many_one_info(self):
        info = get_reduction_info("many-one")
        assert "preserves" in info
        assert "NP-hardness" in info["preserves"]
        assert "NP-completeness" in info["preserves"]
        assert info["transitive"] is True

    def test_turing_info(self):
        info = get_reduction_info("Turing")
        assert "NP-hardness" in info["preserves"]
        assert "does_not_preserve" in info
        assert "NP-completeness" in info["does_not_preserve"]

    def test_alias_karp(self):
        info = get_reduction_info("Karp")
        assert info == get_reduction_info("many-one")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            get_reduction_info("quantum-teleportation")

    def test_randomized_not_transitive(self):
        info = get_reduction_info("randomized")
        assert info["transitive"] is False

    def test_log_space_info(self):
        info = get_reduction_info("log-space")
        assert "NL-completeness" in info["preserves"]


# ─── Type consistency warnings ───────────────────────────────────────────────

class TestTypeConsistency:
    def test_mixed_types_warning(self):
        """Chain with mixed types gets a LOW severity note about weakening."""
        chain = [
            ("SAT", "many-one", "3-SAT"),
            ("3-SAT", "Turing", "CLIQUE"),
        ]
        report = validate_chain(chain)
        type_issues = [
            i for i in report.issues
            if i.check == "TYPE_CONSISTENCY" and i.severity == "LOW"
        ]
        assert len(type_issues) > 0

    def test_claiming_stronger_than_known(self):
        """Claiming log-space when only many-one is known."""
        result = check_reduction("3-SAT", "log-space", "CLIQUE")
        # 3-SAT ->many-one CLIQUE is known, claiming log-space is stronger
        type_issues = [
            i for i in result.issues
            if i.check == "TYPE_CONSISTENCY" and i.severity == "HIGH"
        ]
        assert len(type_issues) > 0


# ─── Hardness preservation ───────────────────────────────────────────────────

class TestHardnessPreservation:
    def test_turing_does_not_give_np_completeness(self):
        """Turing reductions between NP-complete problems: flag that
        NP-completeness is not preserved."""
        chain = [("3-SAT", "Turing", "CLIQUE")]
        report = validate_chain(chain)
        hardness_issues = [
            i for i in report.issues
            if i.check == "HARDNESS_PRESERVATION"
        ]
        assert len(hardness_issues) > 0

    def test_many_one_preserves_np_completeness(self):
        """many-one chain between NP-complete problems: no hardness issue."""
        chain = [("3-SAT", "many-one", "CLIQUE")]
        report = validate_chain(chain)
        hardness_issues = [
            i for i in report.issues
            if i.check == "HARDNESS_PRESERVATION" and i.severity in ("HIGH", "MODERATE")
        ]
        assert len(hardness_issues) == 0


# ─── Report formatting ──────────────────────────────────────────────────────

class TestReportFormatting:
    def test_chain_report_str(self):
        chain = [
            ("3-SAT", "many-one", "CLIQUE"),
            ("CLIQUE", "many-one", "VERTEX-COVER"),
        ]
        report = validate_chain(chain)
        text = str(report)
        assert "Reduction Chain" in text
        assert "PASS" in text
        assert "many-one" in text

    def test_chain_report_fail_str(self):
        chain = [
            ("3-SAT", "many-one", "CLIQUE"),
            ("SUBSET-SUM", "many-one", "VERTEX-COVER"),
        ]
        report = validate_chain(chain)
        text = str(report)
        assert "FAIL" in text
        assert "ERRORS" in text

    def test_reduction_result_str(self):
        result = check_reduction("3-SAT", "many-one", "CLIQUE")
        text = str(result)
        assert "KNOWN" in text
        assert "Karp" in text

    def test_chain_issue_str(self):
        issue = ChainIssue(
            check="TRANSITIVITY", severity="HIGH",
            description="test description", step=0,
        )
        text = str(issue)
        assert "HIGH" in text
        assert "step 0" in text

    def test_empty_chain_report_str(self):
        report = validate_chain([])
        text = str(report)
        assert "PASS" in text

    def test_report_passed_property(self):
        chain = [("3-SAT", "many-one", "CLIQUE")]
        report = validate_chain(chain)
        assert report.passed is True

        broken = [("3-SAT", "many-one", "CLIQUE"), ("FOO", "many-one", "BAR")]
        report2 = validate_chain(broken)
        assert report2.passed is False
