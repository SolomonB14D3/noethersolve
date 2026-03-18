"""Tests for noethersolve.turbulence module.

Tests cover:
- Kolmogorov 4/5 law (exact third-order structure function)
- Energy spectrum (-5/3 law, approximate)
- Length scales (integral, Taylor, Kolmogorov)
- Structure function exponents with intermittency
- Intermittency models
"""

import pytest

from noethersolve.turbulence import (
    kolmogorov_45_law,
    energy_spectrum,
    length_scales,
    structure_function_exponent,
    intermittency_analysis,
    is_in_inertial_range,
    inertial_range_extent,
    KOLMOGOROV_CONSTANT_45,
)


# ─── Kolmogorov 4/5 Law Tests ───────────────────────────────────────────────

class TestKolmogorov45Law:
    """Tests for kolmogorov_45_law()."""

    def test_exact_coefficient(self):
        """4/5 law coefficient is exactly -4/5."""
        assert KOLMOGOROV_CONSTANT_45 == -4/5

    def test_linear_in_separation(self):
        """S₃(r) is linear in r."""
        epsilon = 0.1
        r1 = kolmogorov_45_law(separation=1.0, energy_dissipation=epsilon)
        r2 = kolmogorov_45_law(separation=2.0, energy_dissipation=epsilon)
        assert abs(r2.third_order_sf - 2 * r1.third_order_sf) < 1e-15

    def test_linear_in_dissipation(self):
        """S₃(r) is linear in ε."""
        r = 1.0
        r1 = kolmogorov_45_law(separation=r, energy_dissipation=0.1)
        r2 = kolmogorov_45_law(separation=r, energy_dissipation=0.2)
        assert abs(r2.third_order_sf - 2 * r1.third_order_sf) < 1e-15

    def test_negative_structure_function(self):
        """S₃ is negative (forward energy cascade)."""
        r = kolmogorov_45_law(separation=1.0, energy_dissipation=0.1)
        assert r.third_order_sf < 0

    def test_exact_formula(self):
        """S₃(r) = -(4/5)εr exactly."""
        r, eps = 2.5, 0.3
        result = kolmogorov_45_law(separation=r, energy_dissipation=eps)
        expected = -(4/5) * eps * r
        assert abs(result.third_order_sf - expected) < 1e-15

    def test_is_exact_flag(self):
        """4/5 law is marked as exact."""
        r = kolmogorov_45_law(separation=1.0, energy_dissipation=0.1)
        assert r.is_exact is True

    def test_invalid_separation(self):
        """Zero or negative separation raises error."""
        with pytest.raises(ValueError):
            kolmogorov_45_law(separation=0, energy_dissipation=0.1)
        with pytest.raises(ValueError):
            kolmogorov_45_law(separation=-1, energy_dissipation=0.1)

    def test_invalid_dissipation(self):
        """Zero or negative dissipation raises error."""
        with pytest.raises(ValueError):
            kolmogorov_45_law(separation=1.0, energy_dissipation=0)


# ─── Energy Spectrum Tests ──────────────────────────────────────────────────

class TestEnergySpectrum:
    """Tests for energy_spectrum()."""

    def test_minus_five_thirds_exponent(self):
        """Default exponent is -5/3."""
        r = energy_spectrum(wavenumber=1.0, energy_dissipation=1.0)
        assert abs(r.spectral_exponent - (-5/3)) < 1e-10

    def test_not_exact(self):
        """Spectrum is NOT exact (unlike 4/5 law)."""
        r = energy_spectrum(wavenumber=1.0, energy_dissipation=1.0)
        assert r.is_exact is False

    def test_scaling_with_wavenumber(self):
        """E(k) scales as k^(-5/3)."""
        eps = 1.0
        r1 = energy_spectrum(wavenumber=1.0, energy_dissipation=eps)
        r2 = energy_spectrum(wavenumber=2.0, energy_dissipation=eps)
        ratio = r2.spectrum / r1.spectrum
        expected_ratio = 2.0 ** (-5/3)
        assert abs(ratio - expected_ratio) < 1e-10

    def test_scaling_with_dissipation(self):
        """E(k) scales as ε^(2/3)."""
        k = 1.0
        r1 = energy_spectrum(wavenumber=k, energy_dissipation=1.0)
        r2 = energy_spectrum(wavenumber=k, energy_dissipation=8.0)
        ratio = r2.spectrum / r1.spectrum
        expected_ratio = 8.0 ** (2/3)
        assert abs(ratio - expected_ratio) < 1e-10

    def test_intermittency_model_she_leveque(self):
        """She-Leveque model modifies exponent."""
        r1 = energy_spectrum(wavenumber=1.0, energy_dissipation=1.0)
        r2 = energy_spectrum(wavenumber=1.0, energy_dissipation=1.0,
                            intermittency_model="she_leveque")
        assert r2.intermittency_correction > 0
        assert r2.spectral_exponent > r1.spectral_exponent  # Less negative

    def test_invalid_model(self):
        """Unknown intermittency model raises error."""
        with pytest.raises(ValueError):
            energy_spectrum(wavenumber=1.0, energy_dissipation=1.0,
                          intermittency_model="unknown")


# ─── Length Scales Tests ────────────────────────────────────────────────────

class TestLengthScales:
    """Tests for length_scales()."""

    def test_scale_ordering(self):
        """η < λ < L always."""
        r = length_scales(integral_scale=1.0, urms=1.0, kinematic_viscosity=1e-5)
        assert r.kolmogorov_scale < r.taylor_scale < r.integral_scale

    def test_reynolds_scaling_eta(self):
        """L/η ~ Re^(3/4)."""
        # Test for two different Re
        r1 = length_scales(integral_scale=1.0, urms=1.0, kinematic_viscosity=1e-4)
        r2 = length_scales(integral_scale=1.0, urms=1.0, kinematic_viscosity=1e-5)
        # Re₂/Re₁ = 10, so (L/η)₂/(L/η)₁ ~ 10^(3/4) = 5.62
        ratio = r2.scale_ratios["L_eta"] / r1.scale_ratios["L_eta"]
        expected = 10 ** 0.75
        assert abs(ratio / expected - 1) < 0.1  # Within 10%

    def test_kolmogorov_scale_formula(self):
        """η = (ν³/ε)^(1/4)."""
        L, u, nu = 1.0, 1.0, 1e-5
        r = length_scales(integral_scale=L, urms=u, kinematic_viscosity=nu)
        eps = u ** 3 / L
        expected_eta = (nu ** 3 / eps) ** 0.25
        assert abs(r.kolmogorov_scale - expected_eta) < 1e-10

    def test_taylor_reynolds(self):
        """Re_λ is computed correctly."""
        r = length_scales(integral_scale=1.0, urms=1.0, kinematic_viscosity=1e-5)
        expected = 1.0 * r.taylor_scale / 1e-5
        assert abs(r.taylor_reynolds - expected) < 1e-10

    def test_invalid_inputs(self):
        """Invalid inputs raise errors."""
        with pytest.raises(ValueError):
            length_scales(integral_scale=0, urms=1, kinematic_viscosity=1e-5)
        with pytest.raises(ValueError):
            length_scales(integral_scale=1, urms=0, kinematic_viscosity=1e-5)


# ─── Structure Function Exponent Tests ──────────────────────────────────────

class TestStructureFunctionExponent:
    """Tests for structure_function_exponent()."""

    def test_k41_linear(self):
        """K41 prediction is ζ_p = p/3."""
        for p in [2, 4, 6]:
            r = structure_function_exponent(order=p, model="k41")
            assert abs(r.zeta_p - p/3) < 1e-15

    def test_zeta_3_exact(self):
        """ζ₃ = 1 exactly, regardless of model."""
        for model in ["k41", "she_leveque", "k62", "beta_model"]:
            r = structure_function_exponent(order=3, model=model)
            assert abs(r.zeta_p - 1.0) < 1e-10

    def test_she_leveque_formula(self):
        """She-Leveque: ζ_p = p/9 + 2(1 - (2/3)^(p/3))."""
        p = 6
        r = structure_function_exponent(order=p, model="she_leveque")
        expected = p/9 + 2 * (1 - (2/3) ** (p/3))
        assert abs(r.zeta_p - expected) < 1e-10

    def test_intermittency_reduces_high_order(self):
        """Intermittency reduces high-order exponents below K41."""
        r_k41 = structure_function_exponent(order=6, model="k41")
        r_sl = structure_function_exponent(order=6, model="she_leveque")
        assert r_sl.zeta_p < r_k41.zeta_p

    def test_invalid_order(self):
        """Order < 1 raises error."""
        with pytest.raises(ValueError):
            structure_function_exponent(order=0)

    def test_invalid_model(self):
        """Unknown model raises error."""
        with pytest.raises(ValueError):
            structure_function_exponent(order=2, model="unknown")


# ─── Intermittency Analysis Tests ───────────────────────────────────────────

class TestIntermittencyAnalysis:
    """Tests for intermittency_analysis()."""

    def test_she_leveque_parameters(self):
        """She-Leveque has correct parameters."""
        r = intermittency_analysis(model="she_leveque")
        assert abs(r.parameters["β"] - 2/3) < 1e-10
        assert abs(r.parameters["Δ"] - 2/3) < 1e-10

    def test_k62_mu(self):
        """K62 has μ ≈ 0.25."""
        r = intermittency_analysis(model="k62")
        assert abs(r.parameters["μ"] - 0.25) < 1e-10

    def test_flatness_scaling_positive(self):
        """Flatness scaling exponent is positive (flatness increases with Re)."""
        r = intermittency_analysis(model="she_leveque")
        assert r.flatness_scaling > 0

    def test_zeta_ordering(self):
        """ζ₂ < ζ₄ < ζ₆ for all models."""
        for model in ["she_leveque", "k62", "beta_model"]:
            r = intermittency_analysis(model=model)
            assert r.zeta_2 < r.zeta_4 < r.zeta_6

    def test_invalid_model(self):
        """Unknown model raises error."""
        with pytest.raises(ValueError):
            intermittency_analysis(model="unknown")


# ─── Utility Function Tests ─────────────────────────────────────────────────

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_in_inertial_range(self):
        """Correct identification of inertial range."""
        eta = 1e-4  # Kolmogorov scale
        L = 1.0     # Integral scale
        # 60η = 0.006, 0.1L = 0.1
        assert is_in_inertial_range(0.01, eta, L) is True  # In range
        assert is_in_inertial_range(1e-5, eta, L) is False  # Too small (< 60η)
        assert is_in_inertial_range(0.5, eta, L) is False   # Too large (> 0.1L)

    def test_inertial_range_extent(self):
        """Inertial range grows with Re."""
        extent_low = inertial_range_extent(1e4)
        extent_high = inertial_range_extent(1e6)
        assert extent_high > extent_low

    def test_inertial_range_low_re(self):
        """Very low Re has no inertial range."""
        extent = inertial_range_extent(10)
        assert extent == 0.0

    def test_invalid_reynolds(self):
        """Invalid Reynolds raises error."""
        with pytest.raises(ValueError):
            inertial_range_extent(-100)


# ─── Report String Tests ────────────────────────────────────────────────────

class TestReportStrings:
    """Tests for report __str__ methods."""

    def test_45_law_str(self):
        """4/5 law report is readable."""
        r = kolmogorov_45_law(separation=0.1, energy_dissipation=0.5)
        s = str(r)
        assert "4/5" in s
        assert "EXACT" in s
        assert "S₃" in s or "Third-Order" in s

    def test_spectrum_str(self):
        """Energy spectrum report is readable."""
        r = energy_spectrum(wavenumber=10, energy_dissipation=0.1)
        s = str(r)
        assert "-5/3" in s
        assert "APPROXIMATE" in s or "NOT exact" in s.lower()

    def test_length_scales_str(self):
        """Length scales report is readable."""
        r = length_scales(integral_scale=1.0, urms=1.0, kinematic_viscosity=1e-5)
        s = str(r)
        assert "Kolmogorov" in s
        assert "Taylor" in s
        assert "Integral" in s

    def test_structure_function_str(self):
        """Structure function report is readable."""
        r = structure_function_exponent(order=3, model="she_leveque")
        s = str(r)
        assert "ζ" in s or "zeta" in s.lower()
        assert "exact" in s.lower()  # For p=3

    def test_intermittency_str(self):
        """Intermittency report is readable."""
        r = intermittency_analysis(model="she_leveque")
        s = str(r)
        assert "She" in s or "she_leveque" in s.lower()
        assert "ζ" in s


# ─── Physical Consistency Tests ─────────────────────────────────────────────

class TestPhysicalConsistency:
    """Tests for physical self-consistency."""

    def test_45_law_matches_zeta_3(self):
        """4/5 law implies ζ₃ = 1."""
        # The 4/5 law gives S₃ ~ r¹, so ζ₃ = 1
        r = structure_function_exponent(order=3, model="she_leveque")
        assert abs(r.zeta_p - 1.0) < 1e-10

    def test_spectrum_consistent_with_epsilon(self):
        """Spectrum integrates to give correct total energy."""
        # E(k) ~ ε^(2/3) k^(-5/3) integrates over inertial range
        # This is a dimensional consistency check
        eps = 0.1
        k = 1.0
        r = energy_spectrum(wavenumber=k, energy_dissipation=eps)
        # Check dimensions: E has units of velocity²/wavenumber
        # ε^(2/3) * k^(-5/3) has correct dimensions
        expected_dimensions = eps ** (2/3) * k ** (-5/3)
        # Ratio should be close to Kolmogorov constant
        assert abs(r.spectrum / expected_dimensions - 1.5) < 0.1

    def test_scale_separation_consistency(self):
        """Scale ratios follow Reynolds scaling."""
        r = length_scales(integral_scale=1.0, urms=1.0, kinematic_viscosity=1e-5)
        Re = r.reynolds_number
        # L/η ~ Re^(3/4)
        expected_ratio = Re ** 0.75
        actual_ratio = r.scale_ratios["L_eta"]
        # Should be within factor of 2 (constants vary)
        assert 0.5 < actual_ratio / expected_ratio < 2.0
