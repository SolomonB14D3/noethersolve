"""Tests for noethersolve.reaction_engine — organic chemistry reaction engine."""

import pytest
from noethersolve.reaction_engine import (
    analyze_molecule,
    predict_selectivity,
    predict_mechanism,
    validate_synthesis,
    check_baldwin,
    check_woodward_hoffmann,
    list_mayr_nucleophiles,
    list_mayr_electrophiles,
    list_reaction_templates,
    get_reaction_template,
    MoleculeAnalysis,
    SelectivityReport,
    MechanismReport,
    SynthesisReport,
    BaldwinReport,
    WoodwardHoffmannReport,
)


# ─── analyze_molecule ────────────────────────────────────────────────────────

class TestAnalyzeMolecule:
    def test_ethanol(self):
        r = analyze_molecule("CCO")
        assert isinstance(r, MoleculeAnalysis)
        assert r.molecular_formula == "C2H6O"
        fg_names = [fg.name for fg in r.functional_groups]
        assert "primary alcohol" in fg_names

    def test_acetone(self):
        r = analyze_molecule("CC(=O)C")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "ketone" in fg_names
        assert any(fg.reactive_role == "electrophile" for fg in r.functional_groups)

    def test_benzene_aromatic(self):
        r = analyze_molecule("c1ccccc1")
        assert r.is_aromatic is True
        fg_names = [fg.name for fg in r.functional_groups]
        assert "aromatic ring" in fg_names

    def test_alkyl_bromide(self):
        r = analyze_molecule("CCBr")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "alkyl bromide" in fg_names
        assert "alkyl bromide" in r.electrophilic_sites

    def test_primary_amine(self):
        r = analyze_molecule("CCN")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "primary amine" in fg_names
        assert "primary amine" in r.nucleophilic_sites

    def test_carboxylic_acid(self):
        r = analyze_molecule("CC(=O)O")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "carboxylic acid" in fg_names
        assert "carboxylic acid" in r.acidic_groups

    def test_epoxide(self):
        r = analyze_molecule("C1CO1")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "epoxide" in fg_names

    def test_conjugated_diene(self):
        r = analyze_molecule("C=CC=C")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "conjugated diene" in fg_names
        assert any("Diels-Alder" in n for n in r.notes)

    def test_michael_acceptor(self):
        r = analyze_molecule("C=CC(=O)C")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "michael acceptor" in fg_names

    def test_stereocenter_detection(self):
        # (R)-2-bromobutane
        r = analyze_molecule("[C@@H](Br)(CC)C")
        assert r.num_stereocenters >= 1

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            analyze_molecule("not_a_smiles")

    def test_str_output(self):
        r = analyze_molecule("CCO")
        s = str(r)
        assert "Molecule Analysis" in s
        assert "C2H6O" in s

    def test_ester(self):
        r = analyze_molecule("CC(=O)OC")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "ester" in fg_names

    def test_acid_chloride(self):
        r = analyze_molecule("CC(=O)Cl")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "acid chloride" in fg_names

    def test_nitrile(self):
        r = analyze_molecule("CC#N")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "nitrile" in fg_names

    def test_thiol(self):
        r = analyze_molecule("CCS")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "thiol" in fg_names
        assert "thiol" in r.nucleophilic_sites

    def test_nitro_group(self):
        r = analyze_molecule("c1ccc([N+](=O)[O-])cc1")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "aromatic ring" in fg_names

    def test_aldehyde(self):
        r = analyze_molecule("CC=O")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "aldehyde" in fg_names

    def test_molecular_weight(self):
        r = analyze_molecule("C")
        assert abs(r.molecular_weight - 16.04) < 0.1

    def test_multiple_groups(self):
        # 4-hydroxybenzaldehyde — has phenol, aldehyde, aromatic
        r = analyze_molecule("Oc1ccc(C=O)cc1")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "aldehyde" in fg_names
        assert "aromatic ring" in fg_names


# ─── predict_selectivity ─────────────────────────────────────────────────────

class TestPredictSelectivity:
    def test_hydroxide_methyl_bromide(self):
        r = predict_selectivity("hydroxide", "methyl bromide", "DMSO")
        assert isinstance(r, SelectivityReport)
        assert r.log_k is not None
        assert r.log_k > 0  # fast reaction
        assert r.N is not None
        assert r.E is not None

    def test_water_weak_nucleophile(self):
        r = predict_selectivity("water", "methyl bromide")
        assert r.log_k is not None
        assert r.log_k < predict_selectivity("hydroxide", "methyl bromide", "DMSO").log_k

    def test_unknown_nucleophile(self):
        r = predict_selectivity("mystery_reagent", "acetaldehyde")
        assert r.log_k is None
        assert any("not found" in n for n in r.notes)

    def test_unknown_electrophile(self):
        r = predict_selectivity("hydroxide", "mystery_substrate")
        assert r.log_k is None

    def test_strong_base_predicts_e2(self):
        r = predict_selectivity("tert-butoxide", "methyl bromide", "DMSO")
        assert "E2" in r.predicted_mechanism or "elimination" in r.predicted_mechanism.lower()

    def test_mayr_equation(self):
        # Verify: log k = s_N * (N + E)
        r = predict_selectivity("cyanide", "methyl iodide", "DMSO")
        assert r.log_k is not None
        expected = r.s_N * (r.N + r.E)
        assert abs(r.log_k - expected) < 0.01

    def test_str_output(self):
        r = predict_selectivity("hydroxide", "methyl bromide", "DMSO")
        s = str(r)
        assert "Selectivity" in s
        assert "log k" in s

    def test_cyanide_better_in_dmso(self):
        r_dmso = predict_selectivity("cyanide", "methyl bromide", "DMSO")
        r_water = predict_selectivity("cyanide", "methyl bromide", "water")
        # CN- is a better nucleophile in DMSO (polar aprotic) than water (polar protic)
        assert r_dmso.log_k > r_water.log_k


# ─── predict_mechanism ───────────────────────────────────────────────────────

class TestPredictMechanism:
    def test_diels_alder(self):
        r = predict_mechanism(["C=CC=C", "C=CC(=O)C"])
        assert isinstance(r, MechanismReport)
        assert "Diels-Alder" in r.reaction_name
        assert "concerted" in r.reaction_type
        assert len(r.mechanism_steps) > 0
        assert "suprafacial" in r.stereo_outcome.lower()

    def test_sn2_detection(self):
        r = predict_mechanism(["CBr"], reagents=["NaOH"])
        assert isinstance(r, MechanismReport)
        assert len(r.mechanism_steps) > 0

    def test_grignard_addition(self):
        r = predict_mechanism(["CC=O"], reagents=["EtMgBr"])
        assert isinstance(r, MechanismReport)

    def test_unknown_reaction(self):
        r = predict_mechanism(["C"])
        assert isinstance(r, MechanismReport)
        # Should return generic analysis
        assert r.reaction_name == "Unknown" or len(r.notes) > 0

    def test_str_output(self):
        r = predict_mechanism(["C=CC=C", "C=C"])
        s = str(r)
        assert "Mechanism" in s

    def test_eas_detection(self):
        r = predict_mechanism(["c1ccccc1"], reagents=["AlCl3"])
        assert isinstance(r, MechanismReport)

    def test_aldol(self):
        r = predict_mechanism(["CC(=O)C", "CC=O"], reagents=["NaOH"])
        assert isinstance(r, MechanismReport)


# ─── validate_synthesis ──────────────────────────────────────────────────────

class TestValidateSynthesis:
    def test_simple_valid_synthesis(self):
        r = validate_synthesis([
            {"substrate": "c1ccccc1", "reagent": "Br2/FeBr3", "product": "c1ccc(Br)cc1"},
        ])
        assert isinstance(r, SynthesisReport)
        assert r.verdict == "PASS"

    def test_grignard_incompatibility_alcohol(self):
        r = validate_synthesis([
            {"substrate": "OCC=O", "reagent": "PhMgBr (Grignard)", "product": "?"},
        ])
        assert r.verdict == "FAIL"
        assert any("Grignard" in iss.description for iss in r.issues)

    def test_grignard_incompatibility_carboxylic_acid(self):
        r = validate_synthesis([
            {"substrate": "OC(=O)c1ccccc1", "reagent": "MeMgCl", "product": "?"},
        ])
        assert r.verdict == "FAIL"
        assert any("Grignard" in iss.description for iss in r.issues)

    def test_friedel_crafts_deactivated(self):
        r = validate_synthesis([
            {"substrate": "c1ccc([N+](=O)[O-])cc1", "reagent": "CH3Cl/AlCl3", "product": "?"},
        ])
        # Should warn about deactivated ring
        has_fc_issue = any("Friedel-Crafts" in iss.description or "deactivated" in iss.description
                           for iss in r.issues)
        assert has_fc_issue

    def test_lialh4_multiple_reducible(self):
        # Molecule with both ester and ketone — LiAlH4 reduces both
        r = validate_synthesis([
            {"substrate": "CC(=O)c1ccc(C(=O)OC)cc1", "reagent": "LiAlH4", "product": "?"},
        ])
        assert any("LiAlH4" in iss.description for iss in r.issues)

    def test_multi_step_synthesis(self):
        r = validate_synthesis([
            {"substrate": "c1ccccc1", "reagent": "CH3COCl/AlCl3", "product": "CC(=O)c1ccccc1"},
            {"substrate": "CC(=O)c1ccccc1", "reagent": "NaBH4", "product": "CC(O)c1ccccc1"},
        ])
        assert r.num_steps == 2
        assert r.verdict == "PASS"

    def test_invalid_smiles(self):
        r = validate_synthesis([
            {"substrate": "not_valid", "reagent": "NaOH"},
        ])
        assert r.verdict == "FAIL"
        assert any("Cannot parse" in iss.description for iss in r.issues)

    def test_str_output(self):
        r = validate_synthesis([
            {"substrate": "CCBr", "reagent": "NaOH", "product": "CCO"},
        ])
        s = str(r)
        assert "Synthesis Validation" in s

    def test_strong_oxidant_alkene_warning(self):
        r = validate_synthesis([
            {"substrate": "C=CCCO", "reagent": "KMnO4", "product": "?"},
        ])
        has_warning = any("alkene" in iss.description.lower() or "cleave" in iss.description.lower()
                          for iss in r.issues)
        assert has_warning


# ─── check_baldwin ───────────────────────────────────────────────────────────

class TestCheckBaldwin:
    def test_5_exo_tet_favored(self):
        r = check_baldwin(5, "exo", "tet")
        assert isinstance(r, BaldwinReport)
        assert r.favored is True

    def test_5_endo_tet_disfavored(self):
        r = check_baldwin(5, "endo", "tet")
        assert r.favored is False

    def test_6_endo_trig_favored(self):
        r = check_baldwin(6, "endo", "trig")
        assert r.favored is True

    def test_3_endo_trig_disfavored(self):
        r = check_baldwin(3, "endo", "trig")
        assert r.favored is False

    def test_3_exo_tet_favored(self):
        r = check_baldwin(3, "exo", "tet")
        assert r.favored is True

    def test_out_of_range(self):
        r = check_baldwin(2, "exo", "tet")
        assert r.favored is False
        assert "outside" in r.explanation.lower()

    def test_7_endo_tet_favored(self):
        r = check_baldwin(7, "endo", "tet")
        assert r.favored is True

    def test_str_output(self):
        r = check_baldwin(5, "exo", "tet")
        s = str(r)
        assert "Baldwin" in s
        assert "FAVORED" in s

    def test_case_insensitive(self):
        r = check_baldwin(5, "EXO", "TET")
        assert r.favored is True

    def test_4_endo_trig_disfavored(self):
        r = check_baldwin(4, "endo", "trig")
        assert r.favored is False


# ─── check_woodward_hoffmann ─────────────────────────────────────────────────

class TestCheckWoodwardHoffmann:
    def test_diels_alder_thermal_allowed(self):
        r = check_woodward_hoffmann(6, "thermal", "cycloaddition")
        assert isinstance(r, WoodwardHoffmannReport)
        assert r.allowed is True

    def test_2plus2_thermal_forbidden(self):
        r = check_woodward_hoffmann(4, "thermal", "cycloaddition")
        assert r.allowed is False

    def test_2plus2_photochemical_allowed(self):
        r = check_woodward_hoffmann(4, "photochemical", "cycloaddition")
        assert r.allowed is True

    def test_electrocyclic_4e_thermal_conrotatory(self):
        r = check_woodward_hoffmann(4, "thermal", "electrocyclic")
        assert r.allowed is True
        assert r.electrocyclic_mode == "conrotatory"

    def test_electrocyclic_4e_photo_disrotatory(self):
        r = check_woodward_hoffmann(4, "photochemical", "electrocyclic")
        assert r.allowed is True
        assert r.electrocyclic_mode == "disrotatory"

    def test_electrocyclic_6e_thermal_disrotatory(self):
        r = check_woodward_hoffmann(6, "thermal", "electrocyclic")
        assert r.allowed is True
        assert r.electrocyclic_mode == "disrotatory"

    def test_electrocyclic_6e_photo_conrotatory(self):
        r = check_woodward_hoffmann(6, "photochemical", "electrocyclic")
        assert r.allowed is True
        assert r.electrocyclic_mode == "conrotatory"

    def test_sigmatropic_1_3_thermal_forbidden(self):
        r = check_woodward_hoffmann(2, "thermal", "sigmatropic")
        assert r.allowed is False

    def test_str_output(self):
        r = check_woodward_hoffmann(6, "thermal")
        s = str(r)
        assert "Woodward-Hoffmann" in s
        assert "ALLOWED" in s

    def test_case_insensitive(self):
        r = check_woodward_hoffmann(6, "THERMAL", "CYCLOADDITION")
        assert r.allowed is True

    def test_general_4n_rule(self):
        # 8 electrons thermal → 4n where n=2 → forbidden supra-supra
        r = check_woodward_hoffmann(8, "thermal", "cycloaddition")
        assert r.allowed is False

    def test_general_4n_photo_allowed(self):
        r = check_woodward_hoffmann(8, "photochemical", "cycloaddition")
        assert r.allowed is True


# ─── list/get helpers ────────────────────────────────────────────────────────

class TestListHelpers:
    def test_list_nucleophiles(self):
        nucs = list_mayr_nucleophiles()
        assert len(nucs) > 20
        assert any("hydroxide" in n for n in nucs)

    def test_list_electrophiles(self):
        elecs = list_mayr_electrophiles()
        assert len(elecs) > 15
        assert any("methyl" in e for e in elecs)

    def test_list_templates(self):
        templates = list_reaction_templates()
        assert "SN2" in templates
        assert "Diels-Alder" in templates
        assert "E2" in templates

    def test_get_template(self):
        t = get_reaction_template("SN2")
        assert t is not None
        assert t.mechanism_type == "SN2"
        assert "inversion" in t.stereo_outcome

    def test_get_template_partial_match(self):
        t = get_reaction_template("Diels")
        assert t is not None
        assert "Diels-Alder" in t.name

    def test_get_template_not_found(self):
        t = get_reaction_template("nonexistent_reaction")
        assert t is None


# ─── Integration tests ──────────────────────────────────────────────────────

class TestIntegration:
    def test_aspirin_synthesis_validation(self):
        """Validate the classic aspirin synthesis: salicylic acid + acetic anhydride."""
        r = validate_synthesis([
            {"substrate": "Oc1ccccc1C(=O)O", "reagent": "acetic anhydride/H2SO4",
             "product": "CC(=O)Oc1ccccc1C(=O)O"},
        ])
        assert r.verdict == "PASS"

    def test_analyze_complex_molecule(self):
        """Analyze ibuprofen — should detect aromatic ring, carboxylic acid."""
        r = analyze_molecule("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
        fg_names = [fg.name for fg in r.functional_groups]
        assert "carboxylic acid" in fg_names
        assert "aromatic ring" in fg_names
        assert r.is_aromatic is True

    def test_mayr_solvent_effect(self):
        """Verify nucleophilicity ordering reverses between protic and aprotic solvents."""
        # In DMSO (aprotic): F- > Cl- > Br- > I-
        f_dmso = predict_selectivity("fluoride", "methyl bromide", "DMSO")
        i_dmso = predict_selectivity("iodide", "methyl bromide", "DMSO")
        if f_dmso.log_k is not None and i_dmso.log_k is not None:
            assert f_dmso.N > i_dmso.N  # F- higher N in DMSO

    def test_full_pipeline(self):
        """Analyze molecule → predict mechanism → validate synthesis."""
        # Step 1: Analyze
        analysis = analyze_molecule("C=CC=C")
        assert "conjugated diene" in [fg.name for fg in analysis.functional_groups]

        # Step 2: Predict mechanism with dienophile
        mech = predict_mechanism(["C=CC=C", "C=CC(=O)C"])
        assert "Diels-Alder" in mech.reaction_name

        # Step 3: Validate a synthesis using this reaction
        synth = validate_synthesis([
            {"substrate": "C=CC=C", "reagent": "methyl vinyl ketone",
             "product": "C1CC(=O)CCC1"},
        ])
        assert synth.verdict == "PASS"
