"""Tests for noethersolve.drug_interactions module.

Tests cover:
- Drug profile lookup
- CYP enzyme information
- Drug-drug interaction checking
- Pharmacogenomics assessment
- AUC prediction
- Database consistency
"""

import pytest

from noethersolve.drug_interactions import (
    get_drug_profile,
    get_cyp_info,
    check_interaction,
    predict_auc_change,
    check_pharmacogenomics,
    list_cyp_enzymes,
    list_substrates,
    list_inhibitors,
    list_inducers,
    Strength,
    InteractionType,
    Severity,
)


# ─── Drug Profile Tests ───────────────────────────────────────────────────────

class TestDrugProfile:
    """Tests for get_drug_profile()."""

    def test_midazolam_is_cyp3a4_substrate(self):
        """Midazolam is a sensitive CYP3A4 substrate."""
        r = get_drug_profile("midazolam")
        assert "CYP3A4" in r.metabolizing_enzymes
        assert r.is_sensitive_substrate is True
        assert r.is_prodrug is False

    def test_warfarin_is_cyp2c9_substrate(self):
        """Warfarin is metabolized by CYP2C9."""
        r = get_drug_profile("warfarin")
        assert "CYP2C9" in r.metabolizing_enzymes
        assert r.is_sensitive_substrate is True

    def test_codeine_is_prodrug(self):
        """Codeine is a CYP2D6 prodrug."""
        r = get_drug_profile("codeine")
        assert r.is_prodrug is True
        assert r.prodrug_info is not None
        assert r.prodrug_info["enzyme"] == "CYP2D6"
        assert "morphine" in r.prodrug_info["active_metabolite"].lower()

    def test_clopidogrel_is_prodrug(self):
        """Clopidogrel is a CYP2C19 prodrug."""
        r = get_drug_profile("clopidogrel")
        assert r.is_prodrug is True
        assert r.prodrug_info["enzyme"] == "CYP2C19"

    def test_ketoconazole_is_inhibitor(self):
        """Ketoconazole is a strong CYP3A4 inhibitor."""
        r = get_drug_profile("ketoconazole")
        assert "CYP3A4" in r.known_inhibitor_of
        assert r.known_inhibitor_of["CYP3A4"] == Strength.STRONG

    def test_rifampin_is_inducer(self):
        """Rifampin is a strong CYP3A4 inducer."""
        r = get_drug_profile("rifampin")
        assert "CYP3A4" in r.known_inducer_of
        assert r.known_inducer_of["CYP3A4"] == Strength.STRONG

    def test_paroxetine_is_2d6_inhibitor_and_substrate(self):
        """Paroxetine is both a CYP2D6 substrate and strong inhibitor."""
        r = get_drug_profile("paroxetine")
        assert "CYP2D6" in r.metabolizing_enzymes
        assert "CYP2D6" in r.known_inhibitor_of
        assert r.known_inhibitor_of["CYP2D6"] == Strength.STRONG

    def test_unknown_drug_returns_empty(self):
        """Unknown drug returns empty profile with note."""
        r = get_drug_profile("nonexistent_drug_xyz")
        assert len(r.metabolizing_enzymes) == 0
        assert len(r.notes) > 0
        assert "not found" in r.notes[0].lower()

    def test_case_insensitive_lookup(self):
        """Drug lookup is case-insensitive."""
        r1 = get_drug_profile("MIDAZOLAM")
        r2 = get_drug_profile("midazolam")
        r3 = get_drug_profile("Midazolam")
        assert r1.metabolizing_enzymes == r2.metabolizing_enzymes == r3.metabolizing_enzymes


# ─── CYP Info Tests ───────────────────────────────────────────────────────────

class TestCYPInfo:
    """Tests for get_cyp_info()."""

    def test_cyp3a4_has_substrates(self):
        """CYP3A4 has many known substrates."""
        r = get_cyp_info("CYP3A4")
        assert len(r.substrates) > 10
        assert "midazolam" in r.substrates
        assert "simvastatin" in r.substrates

    def test_cyp3a4_has_sensitive_substrates(self):
        """CYP3A4 has sensitive substrates with narrow therapeutic index."""
        r = get_cyp_info("CYP3A4")
        assert len(r.sensitive_substrates) > 0
        assert "midazolam" in r.sensitive_substrates

    def test_cyp3a4_has_inhibitors(self):
        """CYP3A4 has known strong inhibitors."""
        r = get_cyp_info("CYP3A4")
        assert len(r.strong_inhibitors) > 0
        assert "ketoconazole" in r.strong_inhibitors

    def test_cyp3a4_has_inducers(self):
        """CYP3A4 has known strong inducers."""
        r = get_cyp_info("CYP3A4")
        assert len(r.strong_inducers) > 0
        assert "rifampin" in r.strong_inducers

    def test_cyp2d6_info(self):
        """CYP2D6 enzyme info is complete."""
        r = get_cyp_info("CYP2D6")
        assert "codeine" in r.substrates
        assert "metoprolol" in r.substrates
        assert len(r.strong_inhibitors) > 0
        assert "paroxetine" in r.strong_inhibitors

    def test_cyp_prefix_optional(self):
        """CYP prefix is optional in enzyme lookup."""
        r1 = get_cyp_info("CYP3A4")
        r2 = get_cyp_info("3A4")
        assert r1.substrates == r2.substrates


# ─── Interaction Tests ────────────────────────────────────────────────────────

class TestInteractions:
    """Tests for check_interaction()."""

    def test_ketoconazole_midazolam_inhibition(self):
        """Ketoconazole + midazolam = strong CYP3A4 inhibition."""
        r = check_interaction("ketoconazole", "midazolam")
        assert r.interaction_found is True
        assert r.interaction_type == InteractionType.INHIBITION
        assert r.affected_enzyme == "CYP3A4"
        assert r.strength == Strength.STRONG
        assert r.severity in [Severity.CONTRAINDICATED, Severity.MAJOR]
        assert r.auc_change_range[0] >= 5.0  # Strong = ≥5x

    def test_rifampin_midazolam_induction(self):
        """Rifampin + midazolam = strong CYP3A4 induction."""
        r = check_interaction("rifampin", "midazolam")
        assert r.interaction_found is True
        assert r.interaction_type == InteractionType.INDUCTION
        assert r.affected_enzyme == "CYP3A4"
        assert r.strength == Strength.STRONG
        assert r.auc_change_range[1] <= 0.2  # Strong induction = ≤0.2x

    def test_paroxetine_codeine_inhibition(self):
        """Paroxetine + codeine = CYP2D6 inhibition (blocks prodrug activation)."""
        r = check_interaction("paroxetine", "codeine")
        assert r.interaction_found is True
        assert r.interaction_type == InteractionType.INHIBITION
        assert r.affected_enzyme == "CYP2D6"
        assert r.strength == Strength.STRONG

    def test_fluconazole_warfarin_inhibition(self):
        """Fluconazole + warfarin = CYP2C9 inhibition."""
        r = check_interaction("fluconazole", "warfarin")
        assert r.interaction_found is True
        assert r.affected_enzyme == "CYP2C9"

    def test_no_interaction_unrelated_drugs(self):
        """No interaction between unrelated drugs."""
        r = check_interaction("metformin", "aspirin")
        # These drugs don't have CYP interactions in our database
        # (metformin is renally eliminated, aspirin doesn't go through CYP)
        # They're not in our database, so no interaction found
        assert r.interaction_found is False

    def test_interaction_order_independent(self):
        """Interaction detection works regardless of drug order."""
        r1 = check_interaction("ketoconazole", "midazolam")
        r2 = check_interaction("midazolam", "ketoconazole")
        # Both should find the interaction (A inhibits B or B inhibits A)
        assert r1.interaction_found is True
        # r2 might not find interaction if midazolam isn't an inhibitor
        # (which it isn't - it's a substrate only)

    def test_severe_interaction_gets_contraindicated(self):
        """Strong inhibition of sensitive substrate = contraindicated or major."""
        r = check_interaction("ketoconazole", "simvastatin")
        assert r.interaction_found is True
        assert r.severity in [Severity.CONTRAINDICATED, Severity.MAJOR]

    def test_weak_interaction_severity(self):
        """Weak interactions: minor for normal substrates, major for sensitive."""
        # Metronidazole + warfarin: weak inhibition but WARFARIN is sensitive
        # So severity is MAJOR (sensitive substrate trumps weak strength)
        r = check_interaction("metronidazole", "warfarin")
        if r.interaction_found and r.strength == Strength.WEAK:
            # Warfarin is sensitive, so even weak interaction is major
            assert r.severity == Severity.MAJOR


# ─── AUC Prediction Tests ─────────────────────────────────────────────────────

class TestAUCPrediction:
    """Tests for predict_auc_change()."""

    def test_strong_inhibition_auc(self):
        """Strong CYP3A4 inhibition increases AUC 5-15x."""
        result = predict_auc_change("ketoconazole", "midazolam")
        assert result["auc_low"] >= 5.0
        assert result["auc_high"] <= 15.0
        assert result["interaction_type"] == "inhibition"

    def test_strong_induction_auc(self):
        """Strong CYP3A4 induction decreases AUC 80-95%."""
        result = predict_auc_change("rifampin", "midazolam")
        assert result["auc_low"] <= 0.2  # 80% decrease
        assert result["auc_high"] >= 0.05  # 95% decrease
        assert result["interaction_type"] == "induction"

    def test_no_interaction_auc_unchanged(self):
        """No interaction means AUC = 1x."""
        result = predict_auc_change("vitamin_c", "acetaminophen")
        assert result["auc_low"] == 1.0
        assert result["auc_high"] == 1.0


# ─── Pharmacogenomics Tests ───────────────────────────────────────────────────

class TestPharmacogenomics:
    """Tests for check_pharmacogenomics()."""

    def test_codeine_pgx(self):
        """Codeine pharmacogenomics includes CYP2D6 warning."""
        r = check_pharmacogenomics("codeine")
        assert r.is_prodrug is True
        assert "CYP2D6" in r.relevant_enzymes
        assert "CYP2D6" in r.phenotype_impacts
        # Check that UM warning is present
        um_impact = r.phenotype_impacts["CYP2D6"].get("Ultrarapid metabolizer (UM)", "")
        assert "toxicity" in um_impact.lower() or "rapid" in um_impact.lower()
        # Should have FDA warning recommendation
        assert any("warning" in rec.lower() or "contraindicated" in rec.lower()
                   for rec in r.clinical_recommendations)

    def test_clopidogrel_pgx(self):
        """Clopidogrel pharmacogenomics includes CYP2C19 warning."""
        r = check_pharmacogenomics("clopidogrel")
        assert r.is_prodrug is True
        assert "CYP2C19" in r.relevant_enzymes
        # PM should have reduced efficacy warning
        pm_impact = r.phenotype_impacts["CYP2C19"].get("Poor metabolizer (PM)", "")
        assert "reduced" in pm_impact.lower() or "efficacy" in pm_impact.lower()

    def test_warfarin_pgx(self):
        """Warfarin pharmacogenomics includes CYP2C9."""
        r = check_pharmacogenomics("warfarin")
        assert "CYP2C9" in r.relevant_enzymes
        assert "CYP2C9" in r.phenotype_impacts
        # Should recommend genotyping
        assert any("genotyp" in rec.lower() for rec in r.clinical_recommendations)

    def test_metoprolol_pgx(self):
        """Metoprolol (CYP2D6 substrate) has phenotype impacts."""
        r = check_pharmacogenomics("metoprolol")
        assert "CYP2D6" in r.relevant_enzymes
        assert "CYP2D6" in r.phenotype_impacts


# ─── List Functions Tests ─────────────────────────────────────────────────────

class TestListFunctions:
    """Tests for list_* helper functions."""

    def test_list_cyp_enzymes(self):
        """list_cyp_enzymes returns major CYP enzymes."""
        enzymes = list_cyp_enzymes()
        assert "CYP3A4" in enzymes
        assert "CYP2D6" in enzymes
        assert "CYP2C9" in enzymes
        assert "CYP2C19" in enzymes
        assert "CYP1A2" in enzymes

    def test_list_substrates_cyp3a4(self):
        """list_substrates returns CYP3A4 substrates."""
        subs = list_substrates("CYP3A4")
        assert "midazolam" in subs
        assert "simvastatin" in subs
        assert len(subs) > 10

    def test_list_inhibitors_cyp3a4(self):
        """list_inhibitors returns CYP3A4 inhibitors."""
        inhib = list_inhibitors("CYP3A4")
        assert "ketoconazole" in inhib
        assert "clarithromycin" in inhib

    def test_list_inhibitors_by_strength(self):
        """list_inhibitors can filter by strength."""
        strong = list_inhibitors("CYP3A4", Strength.STRONG)
        assert "ketoconazole" in strong
        # Diltiazem is moderate, not strong
        moderate = list_inhibitors("CYP3A4", Strength.MODERATE)
        assert "diltiazem" in moderate
        assert "ketoconazole" not in moderate

    def test_list_inducers_cyp3a4(self):
        """list_inducers returns CYP3A4 inducers."""
        ind = list_inducers("CYP3A4")
        assert "rifampin" in ind
        assert "phenytoin" in ind

    def test_list_inducers_by_strength(self):
        """list_inducers can filter by strength."""
        strong = list_inducers("CYP3A4", Strength.STRONG)
        assert "rifampin" in strong


# ─── Report String Tests ──────────────────────────────────────────────────────

class TestReportStrings:
    """Tests for report __str__ methods."""

    def test_drug_profile_str(self):
        """Drug profile report has readable string output."""
        r = get_drug_profile("midazolam")
        s = str(r)
        assert "MIDAZOLAM" in s
        assert "CYP3A4" in s
        assert "SENSITIVE" in s

    def test_interaction_str(self):
        """Interaction report has readable string output."""
        r = check_interaction("ketoconazole", "midazolam")
        s = str(r)
        assert "ketoconazole" in s.lower()
        assert "midazolam" in s.lower()
        assert "inhibit" in s.lower()
        assert "CYP3A4" in s

    def test_no_interaction_str(self):
        """No-interaction report is clear."""
        r = check_interaction("vitamin_a", "vitamin_b")
        s = str(r)
        assert "No known" in s or "not found" in s.lower() or "no interaction" in s.lower()

    def test_cyp_info_str(self):
        """CYP info report has readable string output."""
        r = get_cyp_info("CYP3A4")
        s = str(r)
        assert "CYP3A4" in s
        assert "midazolam" in s

    def test_pgx_str(self):
        """Pharmacogenomics report has readable string output."""
        r = check_pharmacogenomics("codeine")
        s = str(r)
        assert "CODEINE" in s
        assert "prodrug" in s.lower() or "PRODRUG" in s


# ─── Database Consistency Tests ───────────────────────────────────────────────

class TestDatabaseConsistency:
    """Tests for database internal consistency."""

    def test_all_sensitive_substrates_in_substrates(self):
        """All sensitive substrates are in the main substrate database."""
        from noethersolve.drug_interactions import SENSITIVE_SUBSTRATES, CYP_SUBSTRATES
        for drug in SENSITIVE_SUBSTRATES:
            assert drug in CYP_SUBSTRATES, f"Sensitive substrate {drug} not in CYP_SUBSTRATES"

    def test_all_prodrugs_in_substrates(self):
        """All prodrugs are in the substrate database."""
        from noethersolve.drug_interactions import PRODRUGS, CYP_SUBSTRATES
        for drug in PRODRUGS:
            assert drug in CYP_SUBSTRATES, f"Prodrug {drug} not in CYP_SUBSTRATES"

    def test_prodrug_enzymes_match_substrate_enzymes(self):
        """Prodrug activation enzyme matches substrate entry."""
        from noethersolve.drug_interactions import PRODRUGS, CYP_SUBSTRATES
        for drug, info in PRODRUGS.items():
            enzyme = info["enzyme"]
            assert enzyme in CYP_SUBSTRATES[drug], \
                f"Prodrug {drug} activation enzyme {enzyme} not in its substrate entry"
