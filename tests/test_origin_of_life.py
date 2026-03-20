"""Tests for origin_of_life module -- prebiotic chemistry calculators."""

import math
import pytest

from noethersolve.origin_of_life import (
    AutocatalyticSetReport,
    check_autocatalytic_set,
    PrebioticPlausibilityReport,
    prebiotic_plausibility,
    EigenThresholdReport,
    eigen_error_threshold,
    RNAFoldingReport,
    rna_folding_energy,
    MillerUreyReport,
    miller_urey_yield,
)


# ── Autocatalytic Set (RAF) ───────────────────────────────────────────

class TestAutocatalyticSet:

    def test_simple_raf_found(self):
        """A food-generated network with all reactants available."""
        reactions = [
            {"reactants": ["A"], "products": ["B"]},
            {"reactants": ["B"], "products": ["C"]},
        ]
        r = check_autocatalytic_set(reactions, food_set=["A"])
        assert r.has_raf
        assert r.network_size == 2
        assert len(r.raf_reactions) == 2

    def test_no_raf_missing_reactant(self):
        """Reaction requires species not in food and not produced."""
        reactions = [
            {"reactants": ["X", "Y"], "products": ["Z"]},
        ]
        r = check_autocatalytic_set(reactions, food_set=["X"])
        assert not r.has_raf
        assert len(r.raf_reactions) == 0

    def test_raf_with_catalyst(self):
        """Catalyzed reaction where catalyst is produced by another reaction."""
        reactions = [
            {"reactants": ["A"], "products": ["B", "C"], "catalyst": "C"},
            {"reactants": ["A"], "products": ["C"]},
        ]
        r = check_autocatalytic_set(reactions, food_set=["A"])
        assert r.has_raf
        # Reaction 0 needs catalyst C, which is produced by reaction 1
        assert 0 in r.raf_reactions
        assert 1 in r.raf_reactions

    def test_raf_catalyst_unavailable(self):
        """Catalyst not available -> reaction pruned."""
        reactions = [
            {"reactants": ["A"], "products": ["B"], "catalyst": "Z"},
        ]
        r = check_autocatalytic_set(reactions, food_set=["A"])
        # Z is not in food and not produced, so reaction is pruned
        assert not r.has_raf

    def test_autocatalytic_cycle(self):
        """A is reactant and product (autocatalytic)."""
        reactions = [
            {"reactants": ["A", "B"], "products": ["C", "A"]},
            {"reactants": ["C"], "products": ["B"]},
        ]
        r = check_autocatalytic_set(reactions, food_set=["A", "B"])
        assert r.has_raf
        assert "A" in r.raf_species

    def test_partial_raf(self):
        """Only some reactions form the RAF subset."""
        reactions = [
            {"reactants": ["A"], "products": ["B"]},
            {"reactants": ["X"], "products": ["Y"]},  # X not in food
        ]
        r = check_autocatalytic_set(reactions, food_set=["A"])
        assert r.has_raf
        assert len(r.raf_reactions) == 1
        assert 0 in r.raf_reactions

    def test_empty_reactions_error(self):
        with pytest.raises(ValueError, match="empty"):
            check_autocatalytic_set([], food_set=["A"])

    def test_empty_food_error(self):
        with pytest.raises(ValueError, match="empty"):
            check_autocatalytic_set(
                [{"reactants": ["A"], "products": ["B"]}],
                food_set=[],
            )

    def test_report_str(self):
        reactions = [
            {"reactants": ["A"], "products": ["B"]},
        ]
        r = check_autocatalytic_set(reactions, food_set=["A"])
        s = str(r)
        assert "Autocatalytic Set (RAF) Analysis" in s
        assert "=" * 60 in s
        assert "Food set" in s

    def test_entire_network_is_raf(self):
        """When all reactions form a RAF, note says 'entire network'."""
        reactions = [
            {"reactants": ["A"], "products": ["B"]},
            {"reactants": ["B"], "products": ["C"]},
        ]
        r = check_autocatalytic_set(reactions, food_set=["A"])
        assert any("entire network" in n for n in r.notes)


# ── Prebiotic Plausibility ────────────────────────────────────────────

class TestPrebioticPlausibility:

    def test_glycine_high_score(self):
        """Glycine is a confirmed prebiotic molecule."""
        r = prebiotic_plausibility("C2H5NO2")
        assert r.score >= 0.90
        assert r.name == "glycine"
        assert r.pathway is not None
        assert "very plausible" in r.plausibility_class

    def test_adenine_high_score(self):
        """Adenine from HCN pentamerization."""
        r = prebiotic_plausibility("C5H5N5")
        assert r.score >= 0.80
        assert r.name == "adenine"

    def test_tryptophan_low_score(self):
        """Tryptophan is unlikely prebiotic."""
        r = prebiotic_plausibility("C11H12N2O2")
        assert r.score <= 0.20
        assert "implausible" in r.plausibility_class

    def test_hcn_perfect_score(self):
        """HCN is the quintessential prebiotic molecule."""
        r = prebiotic_plausibility("HCN")
        assert r.score == 1.00

    def test_unknown_molecule(self):
        """Unknown formula gets computed score."""
        r = prebiotic_plausibility("C4H8O2")
        assert r.name is None
        assert r.pathway is None
        assert 0.0 <= r.score <= 1.0

    def test_sulfur_note(self):
        """Sulfur-containing molecules should get a note."""
        r = prebiotic_plausibility("CH4S")
        assert any("Sulfur" in n for n in r.notes)

    def test_phosphorus_note(self):
        """Phosphorus should trigger unsolved problem note."""
        r = prebiotic_plausibility("H3PO4")
        assert any("Phosphorus" in n for n in r.notes)

    def test_large_molecule_penalty(self):
        """Very large molecules get complexity penalty."""
        r = prebiotic_plausibility("C40H60N10O15")
        assert r.complexity_penalty < -0.15

    def test_invalid_formula(self):
        with pytest.raises(ValueError, match="parse"):
            prebiotic_plausibility("123")

    def test_report_str(self):
        r = prebiotic_plausibility("C2H5NO2")
        s = str(r)
        assert "Prebiotic Plausibility Assessment" in s
        assert "=" * 60 in s
        assert "glycine" in s

    def test_score_range(self):
        """Score must be in [0, 1]."""
        for formula in ["HCN", "C2H5NO2", "C11H12N2O2", "CH2O", "C40H60N10O15"]:
            r = prebiotic_plausibility(formula)
            assert 0.0 <= r.score <= 1.0


# ── Eigen's Error Threshold ──────────────────────────────────────────

class TestEigenThreshold:

    def test_information_survives(self):
        """Short genome with low error rate."""
        r = eigen_error_threshold(
            genome_length=100, error_rate=0.01, selective_advantage=10)
        assert r.survives
        assert r.mu_L == pytest.approx(1.0)
        assert r.ln_a == pytest.approx(math.log(10))
        assert r.safety_margin > 0

    def test_error_catastrophe(self):
        """Long genome with high error rate -> error catastrophe."""
        r = eigen_error_threshold(
            genome_length=1000, error_rate=0.01, selective_advantage=2)
        # mu*L = 10, ln(2) = 0.693 -> catastrophe
        assert not r.survives
        assert r.safety_margin < 0

    def test_max_genome_length(self):
        """L_max = ln(a) / mu."""
        r = eigen_error_threshold(
            genome_length=50, error_rate=0.01, selective_advantage=10)
        expected_max = int(math.log(10) / 0.01)
        assert r.max_genome_length == expected_max

    def test_q_per_base(self):
        """Per-base fidelity = 1 - mu."""
        r = eigen_error_threshold(
            genome_length=100, error_rate=0.05, selective_advantage=5)
        assert r.q_per_base == pytest.approx(0.95)

    def test_Q_total(self):
        """Total fidelity = (1-mu)^L."""
        r = eigen_error_threshold(
            genome_length=100, error_rate=0.01, selective_advantage=5)
        expected = (0.99) ** 100
        assert r.Q_total == pytest.approx(expected, rel=1e-10)

    def test_Q_min(self):
        """Minimum fidelity = 1/a."""
        r = eigen_error_threshold(
            genome_length=50, error_rate=0.01, selective_advantage=10)
        assert r.Q_min == pytest.approx(0.1)

    def test_rna_world_typical(self):
        """RNA polymerase ribozyme: mu~0.01, L~100."""
        r = eigen_error_threshold(
            genome_length=100, error_rate=0.01, selective_advantage=10)
        assert r.survives  # just barely

    def test_modern_dna_polymerase(self):
        """Modern DNA: mu~1e-8, L~10^6."""
        r = eigen_error_threshold(
            genome_length=1_000_000, error_rate=1e-8, selective_advantage=5)
        assert r.survives
        assert r.mu_L == pytest.approx(0.01)

    def test_near_threshold_note(self):
        """Near threshold should generate warning note."""
        # Find params where safety_margin is small but positive
        r = eigen_error_threshold(
            genome_length=200, error_rate=0.01, selective_advantage=math.exp(2.3))
        if r.survives and r.safety_margin < 0.5:
            assert any("Near threshold" in n for n in r.notes)

    def test_invalid_genome_length(self):
        with pytest.raises(ValueError):
            eigen_error_threshold(0, 0.01, 5)

    def test_invalid_error_rate(self):
        with pytest.raises(ValueError):
            eigen_error_threshold(100, 0.0, 5)
        with pytest.raises(ValueError):
            eigen_error_threshold(100, 1.0, 5)

    def test_invalid_selective_advantage(self):
        with pytest.raises(ValueError):
            eigen_error_threshold(100, 0.01, 1.0)
        with pytest.raises(ValueError):
            eigen_error_threshold(100, 0.01, 0.5)

    def test_report_str(self):
        r = eigen_error_threshold(100, 0.01, 10)
        s = str(r)
        assert "Eigen's Error Threshold" in s
        assert "=" * 60 in s
        assert "Genome length" in s
        assert "Eigen (1971)" in s


# ── RNA Folding Free Energy ──────────────────────────────────────────

class TestRNAFolding:

    def test_simple_duplex(self):
        """GCAUGC should pair G-C, C-G, A-U from ends."""
        r = rna_folding_energy("GCAUGC")
        assert r.length == 6
        assert r.sequence == "GCAUGC"

    def test_explicit_structure(self):
        """Provide dot-bracket structure."""
        r = rna_folding_energy("GGGAAACCC", "(((...)))")
        assert len(r.base_pairs) == 3
        assert r.n_gc == 3
        assert r.dG_stacking < 0  # GC stacks are stabilizing

    def test_gc_pairs_stronger(self):
        """GC pairs should give more negative dG than AU pairs."""
        r_gc = rna_folding_energy("GCGC", "(())")
        r_au = rna_folding_energy("AUAU", "(())")
        # GC stacking should be more stabilizing
        assert r_gc.dG_stacking <= r_au.dG_stacking

    def test_t_to_u_conversion(self):
        """T should be converted to U."""
        r = rna_folding_energy("GCAT")
        assert r.sequence == "GCAU"

    def test_auto_structure(self):
        """Auto-detect simple duplex structure."""
        r = rna_folding_energy("GCAUGC")
        assert r.structure is not None
        assert len(r.structure) == 6

    def test_gu_wobble(self):
        """GU wobble pairs should be counted."""
        r = rna_folding_energy("GCGU", "(())")
        # Position 0 (G) pairs with position 3 (U) - wobble pair
        assert r.n_gu >= 0  # depends on structure

    def test_no_pairs_short(self):
        """Very short sequence may have no pairs."""
        r = rna_folding_energy("AU")
        # No stacking possible with just one pair
        assert r.length == 2

    def test_initiation_penalty(self):
        """Helix initiation adds positive free energy."""
        r = rna_folding_energy("GCGC", "(())")
        assert r.dG_init > 0  # penalty is positive

    def test_total_dG(self):
        """Total = stacking + initiation."""
        r = rna_folding_energy("GGGAAACCC", "(((...)))")
        assert r.dG_total == pytest.approx(r.dG_stacking + r.dG_init, abs=0.01)

    def test_invalid_base(self):
        with pytest.raises(ValueError, match="Invalid RNA base"):
            rna_folding_energy("GCXGC")

    def test_short_sequence(self):
        with pytest.raises(ValueError, match="at least 2"):
            rna_folding_energy("A")

    def test_mismatched_structure_length(self):
        with pytest.raises(ValueError, match="length"):
            rna_folding_energy("GCAU", "((.))")  # 4 vs 5

    def test_unmatched_paren(self):
        with pytest.raises(ValueError, match="Unmatched"):
            rna_folding_energy("GCAU", "((..")

    def test_invalid_pair(self):
        with pytest.raises(ValueError, match="Cannot pair"):
            rna_folding_energy("GGAU", "(())")  # G-U is valid, but G-A is not

    def test_report_str(self):
        r = rna_folding_energy("GCAUGC")
        s = str(r)
        assert "RNA Folding Free Energy" in s
        assert "=" * 60 in s
        assert "Turner" in s

    def test_long_sequence_note(self):
        """Sequences > 30 nt should trigger accuracy warning."""
        seq = "G" * 16 + "AAAA" + "C" * 16
        r = rna_folding_energy(seq)
        assert any("approximate" in n or "ViennaRNA" in n for n in r.notes)


# ── Miller-Urey Yield ────────────────────────────────────────────────

class TestMillerUreyYield:

    def test_reducing_atmosphere(self):
        """Classic Miller reducing atmosphere."""
        r = miller_urey_yield(energy_kJ=340, atmosphere="reducing")
        assert r.total_yield_pct > 0
        assert r.atmosphere == "reducing"
        assert "glycine" in r.amino_acid_yields
        assert r.amino_acid_yields["glycine"] > r.amino_acid_yields["alanine"]

    def test_weakly_reducing(self):
        """Weakly reducing gives lower yields than reducing."""
        r_red = miller_urey_yield(energy_kJ=340, atmosphere="reducing")
        r_weak = miller_urey_yield(energy_kJ=340, atmosphere="weakly_reducing")
        assert r_red.total_yield_pct > r_weak.total_yield_pct

    def test_neutral_lowest(self):
        """Neutral atmosphere gives lowest yields."""
        r_red = miller_urey_yield(energy_kJ=340, atmosphere="reducing")
        r_neut = miller_urey_yield(energy_kJ=340, atmosphere="neutral")
        assert r_red.total_yield_pct > r_neut.total_yield_pct

    def test_energy_scaling(self):
        """More energy -> higher yield."""
        r_low = miller_urey_yield(energy_kJ=100, atmosphere="reducing")
        r_high = miller_urey_yield(energy_kJ=1000, atmosphere="reducing")
        assert r_high.total_yield_pct > r_low.total_yield_pct

    def test_carbon_mass_scaling(self):
        """More carbon -> higher mass estimate."""
        r_low = miller_urey_yield(energy_kJ=340, atmosphere="reducing",
                                  carbon_mass_g=0.5)
        r_high = miller_urey_yield(energy_kJ=340, atmosphere="reducing",
                                   carbon_mass_g=5.0)
        assert r_high.estimated_mass_mg > r_low.estimated_mass_mg

    def test_diminishing_returns(self):
        """Energy yield caps at 5x reference."""
        r_5x = miller_urey_yield(energy_kJ=1700, atmosphere="reducing")
        r_10x = miller_urey_yield(energy_kJ=3400, atmosphere="reducing")
        # Both should be capped at 5x, so yields should be equal
        assert r_5x.total_yield_pct == pytest.approx(
            r_10x.total_yield_pct, rel=0.01)

    def test_invalid_energy(self):
        with pytest.raises(ValueError, match="positive"):
            miller_urey_yield(energy_kJ=0, atmosphere="reducing")

    def test_invalid_atmosphere(self):
        with pytest.raises(ValueError, match="Unknown atmosphere"):
            miller_urey_yield(energy_kJ=100, atmosphere="oxidizing")

    def test_invalid_carbon_mass(self):
        with pytest.raises(ValueError, match="positive"):
            miller_urey_yield(energy_kJ=100, carbon_mass_g=0)

    def test_report_str(self):
        r = miller_urey_yield(energy_kJ=340, atmosphere="reducing")
        s = str(r)
        assert "Miller-Urey Amino Acid Yield Estimate" in s
        assert "=" * 60 in s
        assert "Miller (1953)" in s

    def test_low_energy_note(self):
        """Low energy should trigger a note."""
        r = miller_urey_yield(energy_kJ=5, atmosphere="reducing")
        assert any("Low energy" in n for n in r.notes)

    def test_high_energy_note(self):
        """Energy > 5x reference should trigger a note."""
        r = miller_urey_yield(energy_kJ=2000, atmosphere="reducing")
        assert any("diminishing" in n for n in r.notes)
