"""Tests for noethersolve.complexity — Computational complexity class relationship auditor."""


from noethersolve.complexity import (
    audit_complexity,
    check_inclusion,
    check_completeness,
    get_class_info,
)


# ─── audit_complexity — known correct claims ─────────────────────────────────

class TestAuditCorrectClaims:
    def test_established_inclusion_passes(self):
        report = audit_complexity(["P ⊆ PSPACE"])
        assert report.passed
        assert report.verdict == "PASS"

    def test_established_separation_passes(self):
        report = audit_complexity(["P ⊊ EXP"])
        assert report.verdict == "PASS"

    def test_known_equality_passes(self):
        report = audit_complexity(["PSPACE = NPSPACE"])
        assert report.verdict == "PASS"

    def test_established_completeness_passes(self):
        report = audit_complexity(["SAT is NP-complete"])
        assert report.verdict == "PASS"

    def test_multiple_correct_claims(self):
        report = audit_complexity([
            "P ⊆ NP",
            "NP ⊆ PSPACE",
            "SAT is NP-complete",
            "TQBF is PSPACE-complete",
        ])
        assert report.passed
        assert report.n_claims == 4

    def test_trivial_self_inclusion(self):
        report = audit_complexity(["NP ⊆ NP"])
        assert report.passed

    def test_transitive_inclusion(self):
        """P ⊆ EXP via P ⊆ NP ⊆ PSPACE ⊆ EXP."""
        report = audit_complexity(["P ⊆ EXP"])
        assert report.passed


# ─── audit_complexity — known wrong claims ───────────────────────────────────

class TestAuditWrongClaims:
    def test_p_equals_np_is_open(self):
        report = audit_complexity(["P = NP"])
        assert not report.passed
        assert report.n_moderate > 0 or report.n_high > 0

    def test_sat_in_p_is_open(self):
        report = audit_complexity(["SAT in P"])
        assert not report.passed

    def test_gi_is_np_complete_fails(self):
        report = audit_complexity(["GI is NP-complete"])
        assert report.verdict == "FAIL"
        assert report.n_high > 0

    def test_exp_equals_p_contradicts_hierarchy(self):
        """EXP = P contradicts the proven P ⊊ EXP separation."""
        report = audit_complexity(["EXP = P"])
        assert report.verdict == "FAIL"

    def test_np_subset_p_implies_collapse(self):
        """NP ⊆ P would collapse PH."""
        report = audit_complexity(["NP ⊆ P"])
        assert not report.passed

    def test_pspace_equals_np_known_separation(self):
        report = audit_complexity(["NPSPACE ≠ PSPACE"])
        assert report.verdict == "FAIL"  # contradicts Savitch's theorem equality


# ─── audit_complexity — report properties ────────────────────────────────────

class TestAuditReportProperties:
    def test_n_claims_counted(self):
        report = audit_complexity(["P ⊆ NP", "NP ⊆ PSPACE", "P = NP"])
        assert report.n_claims == 3

    def test_passed_property_matches_verdict(self):
        report_pass = audit_complexity(["P ⊆ NP"])
        assert report_pass.passed is True
        report_fail = audit_complexity(["GI is NP-complete"])
        assert report_fail.passed is False

    def test_str_contains_verdict(self):
        report = audit_complexity(["P ⊆ NP"])
        text = str(report)
        assert "PASS" in text

    def test_str_contains_verdict_fail(self):
        report = audit_complexity(["GI is NP-complete"])
        text = str(report)
        assert "FAIL" in text

    def test_issues_list_populated_on_failure(self):
        report = audit_complexity(["P = NP"])
        assert len(report.issues) > 0

    def test_warnings_for_unparseable(self):
        report = audit_complexity(["this is gibberish not a claim"])
        assert len(report.warnings) > 0

    def test_empty_claims_list(self):
        report = audit_complexity([])
        assert report.passed
        assert report.n_claims == 0
        assert report.n_issues == 0

    def test_severity_counts_consistent(self):
        report = audit_complexity(["P = NP", "SAT is NP-complete"])
        total = report.n_high + report.n_moderate + report.n_low + report.n_info
        assert total == report.n_issues


# ─── check_inclusion ─────────────────────────────────────────────────────────

class TestCheckInclusion:
    def test_established_inclusion_p_in_np(self):
        result = check_inclusion("P", "NP")
        assert result.status == "ESTABLISHED"

    def test_established_inclusion_np_in_pspace(self):
        result = check_inclusion("NP", "PSPACE")
        assert result.status == "ESTABLISHED"

    def test_transitive_inclusion_p_in_pspace(self):
        result = check_inclusion("P", "PSPACE")
        assert result.status == "ESTABLISHED"

    def test_equality_savitch(self):
        """PSPACE = NPSPACE (Savitch's theorem)."""
        result = check_inclusion("PSPACE", "NPSPACE")
        assert result.status == "ESTABLISHED"
        result_rev = check_inclusion("NPSPACE", "PSPACE")
        assert result_rev.status == "ESTABLISHED"

    def test_trivial_self_inclusion(self):
        result = check_inclusion("NP", "NP")
        assert result.status == "TRIVIAL"

    def test_open_question_p_vs_np_reverse(self):
        """NP ⊆ P is an open question (the P vs NP question)."""
        result = check_inclusion("NP", "P")
        assert result.status in ("OPEN", "CONTRADICTS_SEPARATION")

    def test_contradicts_separation_exp_in_p(self):
        """EXP ⊆ P contradicts the proven P ⊊ EXP."""
        result = check_inclusion("EXP", "P")
        assert result.status == "CONTRADICTS_SEPARATION"

    def test_unknown_class_a(self):
        result = check_inclusion("FOOBAR", "NP")
        assert result.status == "UNKNOWN_CLASS"

    def test_unknown_class_b(self):
        result = check_inclusion("P", "FOOBAR")
        assert result.status == "UNKNOWN_CLASS"

    def test_open_relationship_bqp_np(self):
        result = check_inclusion("BQP", "NP")
        assert result.status == "OPEN"

    def test_inclusion_result_str(self):
        result = check_inclusion("P", "NP")
        text = str(result)
        assert "P" in text and "NP" in text


# ─── check_completeness ──────────────────────────────────────────────────────

class TestCheckCompleteness:
    def test_sat_np_complete_correct(self):
        result = check_completeness("SAT", "NP")
        assert result.status == "CORRECT"

    def test_3sat_np_complete_correct(self):
        result = check_completeness("3SAT", "NP")
        assert result.status == "CORRECT"

    def test_tqbf_pspace_complete_correct(self):
        result = check_completeness("TQBF", "PSPACE")
        assert result.status == "CORRECT"

    def test_gi_np_complete_incorrect(self):
        """GI is NOT known to be NP-complete."""
        result = check_completeness("GI", "NP")
        assert result.status == "INCORRECT"
        assert len(result.issues) > 0

    def test_factoring_np_complete_incorrect(self):
        result = check_completeness("FACTORING", "NP")
        assert result.status == "INCORRECT"

    def test_sat_in_p_open(self):
        result = check_completeness("SAT", "P")
        assert result.status == "OPEN"

    def test_unknown_problem(self):
        result = check_completeness("FIZZBUZZ", "NP")
        assert result.status == "UNKNOWN_PROBLEM"

    def test_alias_graph_isomorphism(self):
        """'graph isomorphism' should resolve to GI."""
        result = check_completeness("graph isomorphism", "NP")
        assert result.status == "INCORRECT"

    def test_alias_travelling_salesman(self):
        result = check_completeness("travelling salesman", "NP")
        assert result.status == "CORRECT"

    def test_completeness_result_str(self):
        result = check_completeness("SAT", "NP")
        text = str(result)
        assert "CORRECT" in text


# ─── get_class_info ──────────────────────────────────────────────────────────

class TestGetClassInfo:
    def test_np_contains_p(self):
        info = get_class_info("NP")
        assert "P" in info["contains"]

    def test_np_contained_in_pspace(self):
        info = get_class_info("NP")
        assert "PSPACE" in info["contained_in"]

    def test_np_complete_problems(self):
        info = get_class_info("NP")
        assert "SAT" in info["complete_problems"]
        assert "3SAT" in info["complete_problems"]

    def test_pspace_equals_npspace(self):
        info = get_class_info("PSPACE")
        assert "NPSPACE" in info["equals"] or "IP" in info["equals"]

    def test_p_open_with_np(self):
        info = get_class_info("P")
        assert "NP" in info["open_with"]

    def test_exp_strict_supersets(self):
        info = get_class_info("P")
        assert "EXP" in info["strict_supersets"]

    def test_name_normalized(self):
        info = get_class_info("np")
        # get_class_info normalizes via _normalize_class; "np" -> "NP"
        assert info["name"] == "NP"

    def test_all_keys_present(self):
        info = get_class_info("P")
        expected_keys = {
            "name", "contains", "contained_in", "equals",
            "strict_subsets", "strict_supersets", "open_with",
            "oracle_separations", "complete_problems",
        }
        assert set(info.keys()) == expected_keys


# ─── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_case_insensitive_class(self):
        result = check_inclusion("p", "np")
        assert result.status == "ESTABLISHED"

    def test_conp_normalization(self):
        result = check_inclusion("P", "coNP")
        assert result.status == "ESTABLISHED"

    def test_p_poly_normalization(self):
        result = check_inclusion("P", "P/poly")
        assert result.status == "ESTABLISHED"

    def test_mixed_correct_and_wrong(self):
        """A mix of correct and wrong claims yields FAIL overall."""
        report = audit_complexity([
            "P ⊆ NP",           # correct
            "GI is NP-complete", # wrong
        ])
        assert report.verdict == "FAIL"

    def test_membership_claim_sat_in_np(self):
        report = audit_complexity(["SAT in NP"])
        assert report.passed

    def test_separation_claim_ac0_nc1(self):
        """AC0 ⊊ NC1 is a proven separation."""
        report = audit_complexity(["AC0 ⊊ NC1"])
        assert report.passed
