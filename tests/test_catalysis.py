"""Tests for catalysis module — Sabatier principle and volcano plots."""

import pytest
import math
from noethersolve.catalysis import (
    calc_bep_activation,
    calc_volcano_position,
    calc_d_band_center,
    get_scaling_relation,
    find_optimal_catalyst,
    BEP_PARAMS,
    VOLCANO_REACTIONS,
    D_BAND_CENTERS,
)


class TestBEPActivation:
    """Tests for BEP correlation calculations."""

    def test_exothermic_lower_barrier(self):
        """More exothermic reactions should have lower barriers."""
        report_endo = calc_bep_activation("H2_dissociation", 0.5)
        report_exo = calc_bep_activation("H2_dissociation", -0.5)

        assert report_exo.Ea < report_endo.Ea

    def test_bep_linear_relation(self):
        """Ea should follow linear BEP relation (clamped to non-negative)."""
        params = BEP_PARAMS["H2_dissociation"]
        alpha = params["alpha"]
        E0 = params["E0"]

        # Use endothermic reaction to ensure Ea > 0
        delta_E = 0.3
        report = calc_bep_activation("H2_dissociation", delta_E)

        expected_Ea = alpha * delta_E + E0
        assert abs(report.Ea - expected_Ea) < 1e-6

    def test_different_reaction_classes(self):
        """Different reactions should have different BEP slopes."""
        report_ch = calc_bep_activation("C-H_activation", 0.0)
        report_h2 = calc_bep_activation("H2_dissociation", 0.0)

        # C-H has higher E0 than H2
        assert report_ch.Ea > report_h2.Ea

    def test_rate_constant_temperature(self):
        """Higher T should give higher rate constant."""
        report_300 = calc_bep_activation("H2_dissociation", 0.0, temperature=300)
        report_500 = calc_bep_activation("H2_dissociation", 0.0, temperature=500)

        assert report_500.rate_constant > report_300.rate_constant

    def test_ea_non_negative(self):
        """Ea should never be negative (physical constraint)."""
        # Very exothermic reaction
        report = calc_bep_activation("H2_dissociation", -5.0)
        assert report.Ea >= 0.0

    def test_invalid_reaction_class(self):
        """Invalid reaction class should raise error."""
        with pytest.raises(ValueError, match="Unknown reaction class"):
            calc_bep_activation("invalid_reaction", 0.0)

    def test_report_str(self):
        """Report should have readable string."""
        report = calc_bep_activation("H2_dissociation", -0.2)
        s = str(report)
        assert "BEP" in s
        assert "slope" in s.lower() or "α" in s


class TestVolcanoPosition:
    """Tests for volcano plot analysis."""

    def test_optimal_is_peak(self):
        """At optimal energy, relative activity should be 1.0."""
        for reaction, params in VOLCANO_REACTIONS.items():
            optimal = params["optimal_dG"]
            report = calc_volcano_position(reaction, optimal)
            assert report.relative_activity >= 0.99, f"{reaction} peak not at 1.0"

    def test_weak_binding_side(self):
        """Energy below optimal should be weak binding limited."""
        report = calc_volcano_position("HER", -1.0)  # Well below 0
        assert "weak" in report.limiting_side.lower()

    def test_strong_binding_side(self):
        """Energy above optimal should be strong binding limited."""
        report = calc_volcano_position("HER", 1.0)  # Well above 0
        assert "strong" in report.limiting_side.lower()

    def test_activity_decreases_away_from_peak(self):
        """Activity should decrease as you move from peak."""
        optimal = VOLCANO_REACTIONS["HER"]["optimal_dG"]

        report_peak = calc_volcano_position("HER", optimal)
        report_off = calc_volcano_position("HER", optimal + 0.5)

        assert report_off.relative_activity < report_peak.relative_activity

    def test_her_thermoneutral_optimal(self):
        """HER should have thermoneutral (ΔG ≈ 0) as optimal."""
        optimal = VOLCANO_REACTIONS["HER"]["optimal_dG"]
        assert abs(optimal) < 0.1, "HER optimal should be near zero"

    def test_invalid_reaction(self):
        """Invalid reaction should raise error."""
        with pytest.raises(ValueError, match="Unknown reaction"):
            calc_volcano_position("invalid_rxn", 0.0)

    def test_report_str(self):
        """Report should mention Sabatier principle."""
        report = calc_volcano_position("HER", 0.0)
        s = str(report)
        assert "Sabatier" in s or "volcano" in s.lower()


class TestDBandCenter:
    """Tests for d-band center analysis."""

    def test_pt_vs_au(self):
        """Pt should bind stronger than Au (higher ε_d)."""
        report_pt = calc_d_band_center("Pt")
        report_au = calc_d_band_center("Au")

        # Higher (less negative) ε_d = stronger binding
        assert report_pt.d_band_center > report_au.d_band_center

    def test_noble_metals_weak(self):
        """Noble metals (Au, Ag) should have weak binding."""
        report_au = calc_d_band_center("Au")
        assert "weak" in report_au.binding_strength.lower()

    def test_reactive_metals_strong(self):
        """Early transition metals should have strong binding."""
        report_ti = calc_d_band_center("Ti")
        assert "strong" in report_ti.binding_strength.lower()

    def test_reference_comparison(self):
        """Should correctly compare to reference metal."""
        report = calc_d_band_center("Pt", reference_metal="Au")
        # Pt has higher ε_d than Au, so should bind stronger
        assert report.delta_binding > 0

    def test_invalid_metal(self):
        """Invalid metal should raise error."""
        with pytest.raises(ValueError, match="Unknown metal"):
            calc_d_band_center("Xx")

    def test_report_str(self):
        """Report should explain d-band model."""
        report = calc_d_band_center("Pt")
        s = str(report)
        assert "d-band" in s.lower()


class TestScalingRelations:
    """Tests for adsorption energy scaling relations."""

    def test_oh_o_scaling(self):
        """OH-O scaling should be known."""
        report = get_scaling_relation("O", "OH")
        assert 0.4 < report.slope < 0.6
        assert "linear" in report.correlation

    def test_inverted_pair(self):
        """Inverted pair should work."""
        report_forward = get_scaling_relation("O", "OH")
        report_inverse = get_scaling_relation("OH", "O")

        # Inverse relation should have slope ≈ 1/forward_slope
        assert abs(report_inverse.slope - 1/report_forward.slope) < 0.01

    def test_unknown_pair(self):
        """Unknown pair should return approximate."""
        report = get_scaling_relation("X", "Y")
        assert "approximate" in report.correlation.lower()

    def test_selectivity_constraint(self):
        """Should mention selectivity constraint."""
        report = get_scaling_relation("O", "OH")
        s = str(report)
        assert "selectivity" in s.lower() or "constraint" in s.lower()


class TestFindOptimalCatalyst:
    """Tests for catalyst ranking."""

    def test_returns_ranking(self):
        """Should return formatted ranking."""
        result = find_optimal_catalyst("HER")
        assert "Rank" in result
        assert "Metal" in result

    def test_includes_metals(self):
        """Should include metal names."""
        result = find_optimal_catalyst("HER", metal_list=["Pt", "Au", "Ni"])
        assert "Pt" in result
        assert "Au" in result

    def test_invalid_reaction(self):
        """Invalid reaction should raise error."""
        with pytest.raises(ValueError, match="Unknown reaction"):
            find_optimal_catalyst("invalid")


class TestPhysicsCorrectness:
    """Tests for physics accuracy."""

    def test_bep_slopes_physical(self):
        """BEP slopes should be between 0 and 1."""
        for reaction, params in BEP_PARAMS.items():
            alpha = params["alpha"]
            assert 0 < alpha < 1, f"{reaction} has unphysical α = {alpha}"

    def test_d_band_ordering(self):
        """d-band centers should follow periodic trends."""
        # Au should be lower than Pt (more inert)
        assert D_BAND_CENTERS["Au"] < D_BAND_CENTERS["Pt"]
        # Ag should be lower than Pd
        assert D_BAND_CENTERS["Ag"] < D_BAND_CENTERS["Pd"]
        # Cu should be lower than Ni
        assert D_BAND_CENTERS["Cu"] < D_BAND_CENTERS["Ni"]

    def test_arrhenius_temperature_dependence(self):
        """Rate constant should follow Arrhenius."""
        # k ∝ exp(-Ea/kT)
        # At higher T, k should increase
        T1, T2 = 300, 600
        report1 = calc_bep_activation("H2_dissociation", 0.0, T1)
        report2 = calc_bep_activation("H2_dissociation", 0.0, T2)

        # Ratio should be approximately exp(Ea/k × (1/T1 - 1/T2))
        Ea = report1.Ea
        k_B = 8.617e-5  # eV/K
        expected_ratio = math.exp(Ea / k_B * (1/T1 - 1/T2))
        actual_ratio = report2.rate_constant / report1.rate_constant

        # Allow 10% tolerance
        assert abs(actual_ratio / expected_ratio - 1) < 0.1
