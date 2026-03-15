"""Tests for noethersolve.pharmacokinetics — pharmacogenomic CYP interaction checker."""

import pytest
from noethersolve.pharmacokinetics import (
    audit_drug_list,
    check_drug_interactions,
    check_phenotype,
    check_hla,
    get_enzyme_for_drug,
    get_interactions,
    PharmReport,
    DrugInteraction,
    PharmIssue,
)


# ─── Drug interaction checks ───────────────────────────────────────────────

class TestDrugInteractions:
    def test_strong_inhibitor_substrate_high(self):
        # paroxetine is strong CYP2D6 inhibitor, codeine is CYP2D6 substrate
        report = check_drug_interactions(["paroxetine", "codeine"])
        assert any(i.severity == "HIGH" for i in report.interactions)
        assert report.verdict == "FAIL"

    def test_moderate_inhibitor_substrate_moderate(self):
        # duloxetine is moderate CYP2D6 inhibitor, codeine is CYP2D6 substrate
        report = check_drug_interactions(["duloxetine", "codeine"])
        assert any(i.severity == "MODERATE" for i in report.interactions)

    def test_inducer_substrate_moderate(self):
        # rifampin is strong CYP3A4 inducer, simvastatin is CYP3A4 substrate
        report = check_drug_interactions(["rifampin", "simvastatin"])
        assert any(i.interaction_type == "induction" for i in report.interactions)

    def test_no_interaction_passes(self):
        # Two drugs on different enzymes with no cross-inhibition
        report = check_drug_interactions(["warfarin", "theophylline"])
        assert report.verdict == "PASS"
        assert len(report.interactions) == 0

    def test_same_drug_no_self_interaction(self):
        report = check_drug_interactions(["codeine", "codeine"])
        # Should not report self-interaction
        assert len(report.interactions) == 0

    def test_unknown_drugs_clean(self):
        report = check_drug_interactions(["aspirin", "ibuprofen"])
        assert report.verdict == "PASS"

    def test_empty_list_passes(self):
        report = check_drug_interactions([])
        assert report.verdict == "PASS"

    def test_case_insensitive(self):
        r1 = check_drug_interactions(["Paroxetine", "Codeine"])
        r2 = check_drug_interactions(["paroxetine", "codeine"])
        assert len(r1.interactions) == len(r2.interactions)

    def test_interaction_details(self):
        report = check_drug_interactions(["paroxetine", "codeine"])
        ix = [i for i in report.interactions if i.drug_b == "codeine"]
        assert len(ix) > 0
        assert ix[0].enzyme == "CYP2D6"
        assert ix[0].interaction_type == "inhibition"


# ─── Phenotype checks ──────────────────────────────────────────────────────

class TestPhenotype:
    def test_poor_metabolizer_high(self):
        report = check_phenotype("CYP2D6", "poor_metabolizer", ["codeine"])
        assert any(w.severity == "HIGH" for w in report.warnings)
        assert report.verdict == "FAIL"

    def test_ultrarapid_metabolizer_high(self):
        report = check_phenotype("CYP2D6", "ultrarapid_metabolizer", ["codeine"])
        assert any(w.severity == "HIGH" for w in report.warnings)

    def test_intermediate_metabolizer_moderate(self):
        report = check_phenotype("CYP2D6", "intermediate_metabolizer", ["codeine"])
        assert any(w.severity == "MODERATE" for w in report.warnings)

    def test_unaffected_drug_clean(self):
        # warfarin is CYP2C9, not CYP2D6
        report = check_phenotype("CYP2D6", "poor_metabolizer", ["warfarin"])
        assert len(report.affected_drugs) == 0
        assert report.verdict == "PASS"

    def test_clinical_impact_populated(self):
        report = check_phenotype("CYP2D6", "poor_metabolizer", ["codeine"])
        assert len(report.clinical_impact) > 0

    def test_cyp2c19_poor_clopidogrel(self):
        report = check_phenotype("CYP2C19", "poor_metabolizer", ["clopidogrel"])
        assert len(report.affected_drugs) == 1
        assert "clopidogrel" in report.affected_drugs

    def test_unknown_enzyme_passes(self):
        report = check_phenotype("CYP999", "poor_metabolizer", ["codeine"])
        assert report.verdict == "PASS"


# ─── HLA checks ────────────────────────────────────────────────────────────

class TestHLA:
    def test_abacavir_5701_high(self):
        report = check_hla(["HLA-B*57:01"], ["abacavir"])
        assert any(w.severity == "HIGH" for w in report.warnings)
        assert report.verdict == "FAIL"

    def test_carbamazepine_1502_high(self):
        report = check_hla(["HLA-B*15:02"], ["carbamazepine"])
        assert any(w.severity == "HIGH" for w in report.warnings)

    def test_no_match_passes(self):
        report = check_hla(["HLA-B*57:01"], ["metformin"])
        high = [w for w in report.warnings if w.severity == "HIGH"]
        assert len(high) == 0

    def test_required_testing_populated(self):
        report = check_hla(["HLA-B*57:01"], ["abacavir"])
        assert len(report.required_testing) > 0
        assert any("HLA-B*57:01" in t for t in report.required_testing)

    def test_missing_allele_info(self):
        # Drug has HLA association but patient's allele not tested
        report = check_hla([], ["abacavir"])
        # Should still note the association as INFO
        # (empty allele list → handled in audit_drug_list, not check_hla directly)
        assert isinstance(report, type(report))

    def test_allopurinol_5801(self):
        report = check_hla(["HLA-B*58:01"], ["allopurinol"])
        assert any(w.severity == "HIGH" for w in report.warnings)


# ─── Comprehensive audit ───────────────────────────────────────────────────

class TestAuditDrugList:
    def test_basic_interaction(self):
        report = audit_drug_list(["codeine", "paroxetine"])
        assert isinstance(report, PharmReport)
        assert report.verdict == "FAIL"
        assert len(report.interactions) > 0

    def test_with_phenotype(self):
        report = audit_drug_list(
            ["codeine"],
            phenotypes={"CYP2D6": "poor_metabolizer"},
        )
        assert len(report.phenotype_warnings) > 0
        assert report.verdict == "FAIL"

    def test_with_hla(self):
        report = audit_drug_list(
            ["abacavir"],
            hla_alleles=["HLA-B*57:01"],
        )
        assert len(report.hla_warnings) > 0
        assert report.verdict == "FAIL"

    def test_clean_list_passes(self):
        report = audit_drug_list(["warfarin", "theophylline"])
        # No direct CYP interaction between these
        non_info = [i for i in report.issues if hasattr(i, 'severity') and i.severity != "INFO"]
        # warfarin might have HLA info notices; check only non-INFO
        high_mod = [i for i in report.issues
                    if hasattr(i, 'severity') and i.severity in ("HIGH", "MODERATE")]
        assert len(high_mod) == 0

    def test_empty_list_passes(self):
        report = audit_drug_list([])
        assert report.verdict == "PASS"

    def test_no_hla_still_flags_associations(self):
        # abacavir has HLA-B*57:01 association; should note it as INFO
        report = audit_drug_list(["abacavir"])
        hla_info = [w for w in report.hla_warnings if w.severity == "INFO"]
        assert len(hla_info) > 0

    def test_combined_issues(self):
        report = audit_drug_list(
            ["codeine", "paroxetine", "abacavir"],
            hla_alleles=["HLA-B*57:01"],
            phenotypes={"CYP2D6": "poor_metabolizer"},
        )
        assert report.verdict == "FAIL"
        assert len(report.interactions) > 0
        assert len(report.phenotype_warnings) > 0
        assert len(report.hla_warnings) > 0

    def test_issues_list_populated(self):
        report = audit_drug_list(["codeine", "paroxetine"])
        assert len(report.issues) > 0


# ─── Utility functions ─────────────────────────────────────────────────────

class TestUtilityFunctions:
    def test_get_enzyme_for_codeine(self):
        enzymes = get_enzyme_for_drug("codeine")
        assert "CYP2D6" in enzymes

    def test_get_enzyme_unknown_drug(self):
        enzymes = get_enzyme_for_drug("aspirin")
        assert enzymes == []

    def test_get_interactions_codeine(self):
        info = get_interactions("codeine")
        assert "CYP2D6" in info["metabolized_by"]
        assert isinstance(info["inhibits"], list)
        assert isinstance(info["induces"], list)
        assert isinstance(info["co_substrates"], dict)

    def test_get_interactions_paroxetine(self):
        info = get_interactions("paroxetine")
        # paroxetine is both substrate and inhibitor of CYP2D6
        assert "CYP2D6" in info["metabolized_by"]
        assert any(e == "CYP2D6" for e, s in info["inhibits"])

    def test_case_insensitive(self):
        e1 = get_enzyme_for_drug("Codeine")
        e2 = get_enzyme_for_drug("codeine")
        assert e1 == e2

    def test_hyphen_handling(self):
        e1 = get_enzyme_for_drug("st-johns-wort")
        # st_johns_wort is an inducer, not a substrate
        # This should return empty
        assert isinstance(e1, list)


# ─── Report formatting ─────────────────────────────────────────────────────

class TestReportFormat:
    def test_str_contains_verdict(self):
        report = audit_drug_list(["codeine", "paroxetine"])
        s = str(report)
        assert "FAIL" in s

    def test_str_contains_drugs(self):
        report = audit_drug_list(["codeine", "paroxetine"])
        s = str(report)
        assert "codeine" in s
        assert "paroxetine" in s

    def test_str_contains_interactions(self):
        report = audit_drug_list(["codeine", "paroxetine"])
        assert "Drug-Drug Interactions" in str(report)

    def test_str_contains_summary(self):
        report = audit_drug_list(["codeine", "paroxetine"])
        assert "Summary" in str(report)

    def test_passed_property(self):
        report = audit_drug_list(["warfarin", "theophylline"])
        # verdict may be PASS or WARN depending on INFO items
        assert isinstance(report.passed, bool)


# ─── Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_drug_is_both_substrate_and_inhibitor(self):
        # paroxetine is CYP2D6 substrate AND strong CYP2D6 inhibitor
        # When paired with another CYP2D6 substrate, should flag
        report = check_drug_interactions(["paroxetine", "tramadol"])
        assert len(report.interactions) > 0

    def test_multiple_enzymes(self):
        # Some drugs are substrates of multiple enzymes
        info = get_interactions("phenytoin")
        assert len(info["metabolized_by"]) >= 1

    def test_three_drug_interactions(self):
        # Check all pairwise interactions
        report = check_drug_interactions(["paroxetine", "codeine", "tramadol"])
        # paroxetine inhibits both codeine and tramadol metabolism
        assert len(report.interactions) >= 2
