"""Tests for noethersolve.enzyme_kinetics — enzyme kinetics computational engine."""

import math
import pytest

from noethersolve.enzyme_kinetics import (
    michaelis_menten,
    inhibition,
    catalytic_efficiency,
    lineweaver_burk,
    ph_rate_profile,
    cooperativity,
    MMReport,
    InhibitionReport,
    EfficiencyReport,
    LineweaverBurkReport,
    PHProfileReport,
    CooperativityReport,
    DIFFUSION_LIMIT,
)


# ── Michaelis-Menten ─────────────────────────────────────────────────────

class TestMichaelisMenten:
    def test_basic_calculation(self):
        r = michaelis_menten(Vmax=100, Km=5, S=10)
        assert isinstance(r, MMReport)
        assert abs(r.V0 - 100 * 10 / 15) < 1e-10
        assert abs(r.fraction_Vmax - 10 / 15) < 1e-10

    def test_half_saturation(self):
        """At [S] = Km, V0 = Vmax/2."""
        r = michaelis_menten(Vmax=200, Km=10, S=10)
        assert abs(r.V0 - 100.0) < 1e-10
        assert abs(r.fraction_Vmax - 0.5) < 1e-10
        assert "half-saturated" in r.saturation

    def test_saturated(self):
        """At [S] >> Km, V0 ≈ Vmax."""
        r = michaelis_menten(Vmax=100, Km=1, S=1000)
        assert r.fraction_Vmax > 0.99
        assert "saturated" in r.saturation

    def test_unsaturated(self):
        """At [S] << Km, V0 ≈ (Vmax/Km)*[S] (first-order)."""
        r = michaelis_menten(Vmax=100, Km=100, S=1)
        assert r.fraction_Vmax < 0.1
        assert "unsaturated" in r.saturation
        assert any("First-order" in n for n in r.notes)

    def test_zero_substrate(self):
        r = michaelis_menten(Vmax=100, Km=5, S=0)
        assert r.V0 == 0.0
        assert r.fraction_Vmax == 0.0

    def test_zero_order_note(self):
        r = michaelis_menten(Vmax=100, Km=1, S=100)
        assert any("Zero-order" in n for n in r.notes)

    def test_negative_km_raises(self):
        with pytest.raises(ValueError, match="Km must be positive"):
            michaelis_menten(Vmax=100, Km=-5, S=10)

    def test_negative_substrate_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            michaelis_menten(Vmax=100, Km=5, S=-1)

    def test_negative_vmax_raises(self):
        with pytest.raises(ValueError, match="Vmax must be positive"):
            michaelis_menten(Vmax=-100, Km=5, S=10)

    def test_str_output(self):
        r = michaelis_menten(Vmax=100, Km=5, S=10)
        s = str(r)
        assert "Michaelis-Menten" in s
        assert "Vmax" in s


# ── Inhibition ───────────────────────────────────────────────────────────

class TestInhibition:
    def test_competitive_km_increases(self):
        """Competitive: Km_app = Km*(1 + [I]/Ki), Vmax unchanged."""
        r = inhibition(Vmax=100, Km=5, S=10, Ki=2, I=4, mode="competitive")
        assert isinstance(r, InhibitionReport)
        expected_Km_app = 5 * (1 + 4 / 2)  # 15
        assert abs(r.Km_app - expected_Km_app) < 1e-10
        assert abs(r.Vmax_app - 100) < 1e-10

    def test_competitive_vmax_unchanged(self):
        r = inhibition(Vmax=100, Km=5, S=10, Ki=2, I=4, mode="competitive")
        assert r.Vmax_app == 100

    def test_noncompetitive_vmax_decreases(self):
        """Noncompetitive: Vmax_app = Vmax/(1 + [I]/Ki), Km unchanged."""
        r = inhibition(Vmax=100, Km=5, S=10, Ki=2, I=4, mode="noncompetitive")
        expected_Vmax_app = 100 / (1 + 4 / 2)  # 33.33
        assert abs(r.Vmax_app - expected_Vmax_app) < 1e-10
        assert abs(r.Km_app - 5) < 1e-10

    def test_uncompetitive_both_decrease(self):
        """Uncompetitive: both Km and Vmax decrease by same factor."""
        r = inhibition(Vmax=100, Km=10, S=10, Ki=5, I=5, mode="uncompetitive")
        alpha = 1 + 5 / 5  # 2
        assert abs(r.Km_app - 10 / alpha) < 1e-10
        assert abs(r.Vmax_app - 100 / alpha) < 1e-10
        # Ratio Vmax/Km should be preserved
        assert abs(r.Vmax_app / r.Km_app - 100 / 10) < 1e-10

    def test_mixed_inhibition(self):
        r = inhibition(Vmax=100, Km=5, S=10, Ki=2, I=4, mode="mixed", Ki_prime=3)
        alpha = 1 + 4 / 2      # 3
        alpha_p = 1 + 4 / 3    # 2.333
        assert abs(r.Km_app - 5 * alpha / alpha_p) < 1e-10
        assert abs(r.Vmax_app - 100 / alpha_p) < 1e-10

    def test_mixed_requires_ki_prime(self):
        with pytest.raises(ValueError, match="Ki_prime"):
            inhibition(Vmax=100, Km=5, S=10, Ki=2, I=4, mode="mixed")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown inhibition mode"):
            inhibition(Vmax=100, Km=5, S=10, Ki=2, I=4, mode="allosteric")

    def test_inhibition_percentage(self):
        r = inhibition(Vmax=100, Km=5, S=10, Ki=2, I=4, mode="competitive")
        assert r.percent_inhibition > 0
        assert r.percent_inhibition < 100
        assert r.V0_inhibited < r.V0_uninhibited

    def test_no_inhibitor(self):
        """With [I]=0, inhibited velocity equals uninhibited."""
        r = inhibition(Vmax=100, Km=5, S=10, Ki=2, I=0, mode="competitive")
        assert abs(r.V0_inhibited - r.V0_uninhibited) < 1e-10
        assert abs(r.percent_inhibition) < 1e-10

    def test_negative_ki_raises(self):
        with pytest.raises(ValueError, match="Ki must be positive"):
            inhibition(Vmax=100, Km=5, S=10, Ki=-1, I=4, mode="competitive")

    def test_case_insensitive_mode(self):
        r = inhibition(Vmax=100, Km=5, S=10, Ki=2, I=4, mode="Competitive")
        assert r.mode == "competitive"

    def test_str_output(self):
        r = inhibition(Vmax=100, Km=5, S=10, Ki=2, I=4, mode="competitive")
        s = str(r)
        assert "Competitive" in s
        assert "Km" in s

    def test_heavy_inhibition_note(self):
        r = inhibition(Vmax=100, Km=5, S=10, Ki=1, I=100, mode="competitive")
        assert any("heavily inhibited" in n for n in r.notes)


# ── Catalytic Efficiency ─────────────────────────────────────────────────

class TestCatalyticEfficiency:
    def test_diffusion_limited(self):
        """Acetylcholinesterase-like: kcat/Km near diffusion limit."""
        r = catalytic_efficiency(kcat=1.4e4, Km=9e-5)
        assert isinstance(r, EfficiencyReport)
        assert abs(r.kcat_over_Km - 1.4e4 / 9e-5) < 1e3
        assert r.is_diffusion_limited
        assert "perfect" in r.efficiency_class

    def test_moderate_enzyme(self):
        r = catalytic_efficiency(kcat=100, Km=1e-2)
        assert abs(r.kcat_over_Km - 1e4) < 1
        assert not r.is_diffusion_limited
        assert "efficient" in r.efficiency_class

    def test_slow_enzyme(self):
        r = catalytic_efficiency(kcat=1, Km=1e-1)
        assert abs(r.kcat_over_Km - 10) < 1e-10
        assert r.efficiency_class == "slow"

    def test_negative_kcat_raises(self):
        with pytest.raises(ValueError, match="kcat must be positive"):
            catalytic_efficiency(kcat=-1, Km=1e-3)

    def test_negative_km_raises(self):
        with pytest.raises(ValueError, match="Km must be positive"):
            catalytic_efficiency(kcat=100, Km=-1e-3)

    def test_str_output(self):
        r = catalytic_efficiency(kcat=1000, Km=5e-6)
        s = str(r)
        assert "Catalytic Efficiency" in s
        assert "Reference enzymes" in s

    def test_very_efficient_class(self):
        r = catalytic_efficiency(kcat=1e4, Km=1e-2)  # 1e6
        assert "very efficient" in r.efficiency_class


# ── Lineweaver-Burk ──────────────────────────────────────────────────────

class TestLineweaverBurk:
    def test_slope(self):
        r = lineweaver_burk(Vmax=100, Km=10)
        assert abs(r.slope - 10 / 100) < 1e-10

    def test_y_intercept(self):
        r = lineweaver_burk(Vmax=100, Km=10)
        assert abs(r.y_intercept - 1 / 100) < 1e-10

    def test_x_intercept(self):
        r = lineweaver_burk(Vmax=100, Km=10)
        assert abs(r.x_intercept - (-1 / 10)) < 1e-10

    def test_default_data_points(self):
        """Default generates 6 data points at 0.2, 0.5, 1, 2, 5, 10 × Km."""
        r = lineweaver_burk(Vmax=100, Km=10)
        assert len(r.data_points) == 6

    def test_custom_s_values(self):
        r = lineweaver_burk(Vmax=100, Km=10, S_values=[5, 10, 20, 50])
        assert len(r.data_points) == 4

    def test_data_points_correct(self):
        """Each (1/S, 1/V0) should satisfy 1/V0 = slope*(1/S) + y_int."""
        r = lineweaver_burk(Vmax=100, Km=10)
        for inv_s, inv_v in r.data_points:
            expected = r.slope * inv_s + r.y_intercept
            assert abs(inv_v - expected) < 1e-10

    def test_negative_vmax_raises(self):
        with pytest.raises(ValueError, match="Vmax must be positive"):
            lineweaver_burk(Vmax=-100, Km=10)

    def test_str_output(self):
        r = lineweaver_burk(Vmax=100, Km=10)
        s = str(r)
        assert "Lineweaver-Burk" in s
        assert "Double Reciprocal" in s


# ── pH-Rate Profile ──────────────────────────────────────────────────────

class TestPHRateProfile:
    def test_optimal_ph(self):
        """At pH = (pKa1 + pKa2)/2, activity should be near maximum."""
        r = ph_rate_profile(pH=7.0, V_optimal=100, pKa1=6.0, pKa2=8.0)
        assert isinstance(r, PHProfileReport)
        assert r.pH_optimum == 7.0
        # At optimum, fraction should be high
        assert r.fraction_active > 0.8

    def test_extreme_low_ph(self):
        """Far below pKa1, enzyme is largely inactive."""
        r = ph_rate_profile(pH=2.0, V_optimal=100, pKa1=6.0, pKa2=8.0)
        assert r.fraction_active < 0.01
        assert any("below" in n.lower() or "protonated" in n.lower() for n in r.notes)

    def test_extreme_high_ph(self):
        """Far above pKa2, enzyme is largely inactive."""
        r = ph_rate_profile(pH=12.0, V_optimal=100, pKa1=6.0, pKa2=8.0)
        assert r.fraction_active < 0.01
        assert any("above" in n.lower() or "deprotonated" in n.lower() for n in r.notes)

    def test_bell_shaped_symmetry(self):
        """Equidistant from optimum should give similar activity."""
        r_low = ph_rate_profile(pH=6.0, V_optimal=100, pKa1=6.0, pKa2=8.0)
        r_high = ph_rate_profile(pH=8.0, V_optimal=100, pKa1=6.0, pKa2=8.0)
        # With symmetric pKas around 7, these should be identical
        assert abs(r_low.fraction_active - r_high.fraction_active) < 1e-10

    def test_pka_order_validation(self):
        with pytest.raises(ValueError, match="pKa1.*less than pKa2"):
            ph_rate_profile(pH=7.0, V_optimal=100, pKa1=8.0, pKa2=6.0)

    def test_equal_pka_raises(self):
        with pytest.raises(ValueError):
            ph_rate_profile(pH=7.0, V_optimal=100, pKa1=7.0, pKa2=7.0)

    def test_str_output(self):
        r = ph_rate_profile(pH=7.0, V_optimal=100, pKa1=6.0, pKa2=8.0)
        s = str(r)
        assert "pH-Rate Profile" in s
        assert "Bell-shaped" in s


# ── Cooperativity (Hill Equation) ────────────────────────────────────────

class TestCooperativity:
    def test_positive_cooperativity(self):
        """n > 1: sigmoidal, positive cooperativity."""
        r = cooperativity(Vmax=100, K_half=10, n=2.8, S=10)
        assert isinstance(r, CooperativityReport)
        assert "positive" in r.cooperativity_type
        # At S = K_half, should be exactly half-saturated
        assert abs(r.fraction_Vmax - 0.5) < 1e-10

    def test_negative_cooperativity(self):
        r = cooperativity(Vmax=100, K_half=10, n=0.5, S=10)
        assert "negative" in r.cooperativity_type

    def test_non_cooperative_reduces_to_mm(self):
        """n = 1: should match Michaelis-Menten exactly."""
        r_hill = cooperativity(Vmax=100, K_half=5, n=1.0, S=10)
        r_mm = michaelis_menten(Vmax=100, Km=5, S=10)
        assert abs(r_hill.V0 - r_mm.V0) < 1e-10
        assert "non-cooperative" in r_hill.cooperativity_type

    def test_hemoglobin_like(self):
        """Hemoglobin: n ≈ 2.8, 4 subunits."""
        r = cooperativity(Vmax=100, K_half=26, n=2.8, S=26)
        assert abs(r.fraction_Vmax - 0.5) < 1e-10
        assert any("at least 3 binding sites" in n for n in r.notes)

    def test_hill_equation_calculation(self):
        """V0 = Vmax * S^n / (K^n + S^n)."""
        r = cooperativity(Vmax=100, K_half=10, n=3, S=20)
        expected = 100 * 20**3 / (10**3 + 20**3)
        assert abs(r.V0 - expected) < 1e-10

    def test_zero_substrate(self):
        r = cooperativity(Vmax=100, K_half=10, n=2, S=0)
        assert r.V0 == 0.0

    def test_negative_n_raises(self):
        with pytest.raises(ValueError, match="Hill coefficient"):
            cooperativity(Vmax=100, K_half=10, n=-1, S=10)

    def test_negative_k_half_raises(self):
        with pytest.raises(ValueError, match="K_half must be positive"):
            cooperativity(Vmax=100, K_half=-10, n=2, S=10)

    def test_str_output(self):
        r = cooperativity(Vmax=100, K_half=10, n=2.8, S=10)
        s = str(r)
        assert "Hill" in s
        assert "Cooperativity" in s


# ── Integration Tests ────────────────────────────────────────────────────

class TestIntegration:
    def test_inhibition_reduces_mm_velocity(self):
        """Inhibited velocity should always be <= uninhibited."""
        r_mm = michaelis_menten(Vmax=100, Km=5, S=10)
        for mode in ["competitive", "noncompetitive", "uncompetitive"]:
            r_inh = inhibition(Vmax=100, Km=5, S=10, Ki=2, I=4, mode=mode)
            assert r_inh.V0_inhibited <= r_mm.V0 + 1e-10, f"{mode} failed"

    def test_lineweaver_burk_recovers_mm_params(self):
        """LB parameters should recover original Vmax and Km."""
        r = lineweaver_burk(Vmax=50, Km=8)
        assert abs(r.Vmax - 50) < 1e-10
        assert abs(r.Km - 8) < 1e-10

    def test_full_enzyme_characterization(self):
        """Simulate characterizing an enzyme: MM → LB → efficiency → inhibition."""
        # Step 1: Basic kinetics
        r_mm = michaelis_menten(Vmax=500, Km=2e-4, S=1e-3)
        assert r_mm.V0 > 0

        # Step 2: Lineweaver-Burk
        r_lb = lineweaver_burk(Vmax=500, Km=2e-4)
        assert abs(r_lb.Vmax - 500) < 1e-10

        # Step 3: Catalytic efficiency (kcat = Vmax / [E]_total, assume [E] = 1e-6 M)
        kcat = 500 / 1e-6  # 5e8 s^-1 (unrealistic but tests the math)
        r_eff = catalytic_efficiency(kcat=kcat, Km=2e-4)
        assert r_eff.kcat_over_Km > 0

        # Step 4: Test an inhibitor
        r_inh = inhibition(Vmax=500, Km=2e-4, S=1e-3, Ki=1e-5, I=5e-5, mode="competitive")
        assert r_inh.percent_inhibition > 0

    def test_cooperativity_vs_mm_at_khalf(self):
        """At S = K_half, both Hill and MM give V0 = Vmax/2."""
        for n in [0.5, 1.0, 2.0, 2.8, 4.0]:
            r = cooperativity(Vmax=100, K_half=10, n=n, S=10)
            assert abs(r.V0 - 50) < 1e-10, f"Failed for n={n}"
