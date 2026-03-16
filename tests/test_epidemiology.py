"""Tests for noethersolve.epidemiology module.

Tests cover:
- Herd immunity threshold calculations
- Reproduction number (R0, Rt)
- Doubling time
- Attack rate (final epidemic size)
- SIR model parameters
- Vaccine impact
- Generation/serial interval
"""

import math
import pytest

from noethersolve.epidemiology import (
    herd_immunity_threshold,
    reproduction_number,
    doubling_time,
    attack_rate,
    sir_model,
    vaccine_impact,
    generation_interval,
    seir_parameters,
    get_disease_R0,
    list_diseases,
)


# ─── Herd Immunity Tests ──────────────────────────────────────────────────────

class TestHerdImmunity:
    """Tests for herd_immunity_threshold()."""

    def test_measles_r0_15(self):
        """Measles (R0=15) requires ~93% immunity."""
        r = herd_immunity_threshold(R0=15.0)
        assert abs(r.threshold - (1 - 1/15)) < 1e-10
        assert abs(r.threshold_pct - 93.33) < 0.1

    def test_covid_r0_3(self):
        """COVID original (R0=3) requires ~67% immunity."""
        r = herd_immunity_threshold(R0=3.0)
        assert abs(r.threshold - 2/3) < 1e-10
        assert abs(r.threshold_pct - 66.67) < 0.1

    def test_influenza_r0_1_5(self):
        """Seasonal flu (R0=1.5) requires ~33% immunity."""
        r = herd_immunity_threshold(R0=1.5)
        assert abs(r.threshold - 1/3) < 1e-10

    def test_r0_equals_1(self):
        """R0=1 means no herd immunity needed."""
        r = herd_immunity_threshold(R0=1.0)
        assert r.threshold == 0.0
        assert "R0 ≤ 1" in r.notes[0]

    def test_r0_less_than_1(self):
        """R0<1 means no sustained transmission possible."""
        r = herd_immunity_threshold(R0=0.5)
        assert r.threshold == 0.0

    def test_susceptible_at_equilibrium(self):
        """Susceptible at equilibrium = 1/R0."""
        r = herd_immunity_threshold(R0=4.0)
        assert abs(r.susceptible_at_equilibrium - 0.25) < 1e-10

    def test_invalid_r0_raises(self):
        """Negative or zero R0 raises error."""
        with pytest.raises(ValueError):
            herd_immunity_threshold(R0=0)
        with pytest.raises(ValueError):
            herd_immunity_threshold(R0=-2)

    def test_formula_correct(self):
        """Formula is documented correctly."""
        r = herd_immunity_threshold(R0=2.0)
        assert "1 - 1/R₀" in r.formula


# ─── Reproduction Number Tests ────────────────────────────────────────────────

class TestReproductionNumber:
    """Tests for reproduction_number()."""

    def test_r0_from_beta_gamma(self):
        """R0 = beta/gamma."""
        r = reproduction_number(beta=0.3, gamma=0.1)
        assert r.R_type == "basic"
        assert abs(r.R - 3.0) < 1e-10

    def test_rt_from_susceptible_fraction(self):
        """Rt = R0 × S."""
        r = reproduction_number(R0=3.0, susceptible_fraction=0.5)
        assert r.R_type == "effective"
        assert abs(r.R - 1.5) < 1e-10

    def test_epidemic_growing(self):
        """R > 1 means epidemic growing."""
        r = reproduction_number(R0=2.5)
        assert r.R > 1
        assert "GROWING" in r.interpretation

    def test_epidemic_declining(self):
        """R < 1 means epidemic declining."""
        r = reproduction_number(R0=3.0, susceptible_fraction=0.2)
        assert r.R < 1
        assert "DECLINING" in r.interpretation

    def test_growth_rate_per_generation(self):
        """Growth rate = R - 1."""
        r = reproduction_number(R0=2.0)
        assert abs(r.growth_rate - 1.0) < 1e-10


# ─── Doubling Time Tests ──────────────────────────────────────────────────────

class TestDoublingTime:
    """Tests for doubling_time()."""

    def test_growth_rate_to_doubling(self):
        """T_d = ln(2)/r."""
        r = doubling_time(growth_rate=0.1)
        assert abs(r.doubling_time - math.log(2)/0.1) < 1e-10

    def test_r0_and_generation_time(self):
        """Doubling time from R0 and generation time."""
        # r = ln(R0)/T_g, T_d = ln(2)/r
        r = doubling_time(R0=2.0, generation_time=5.0)
        expected_r = math.log(2.0) / 5.0
        expected_Td = math.log(2) / expected_r
        assert abs(r.doubling_time - expected_Td) < 1e-10

    def test_negative_growth_rate(self):
        """Negative growth rate gives halving time."""
        r = doubling_time(growth_rate=-0.1)
        assert r.halving_time is not None
        assert abs(r.halving_time - math.log(2)/0.1) < 1e-10
        assert r.doubling_time == float('inf')

    def test_zero_growth_rate(self):
        """Zero growth rate = endemic/stable."""
        r = doubling_time(growth_rate=0.0)
        assert r.doubling_time == float('inf')
        assert "stable" in r.notes[0].lower() or "endemic" in r.notes[0].lower()


# ─── Attack Rate Tests ────────────────────────────────────────────────────────

class TestAttackRate:
    """Tests for attack_rate()."""

    def test_r0_less_than_1(self):
        """R0 < 1 means no epidemic (0% attack rate)."""
        r = attack_rate(R0=0.8)
        assert r.attack_rate == 0.0
        assert r.final_susceptible == 1.0

    def test_r0_equals_2(self):
        """R0 = 2 gives ~80% attack rate."""
        r = attack_rate(R0=2.0)
        # Known result: ~79.7% for R0=2
        assert 0.79 < r.attack_rate < 0.81

    def test_r0_equals_3(self):
        """R0 = 3 gives ~94% attack rate."""
        r = attack_rate(R0=3.0)
        # Known result: ~94% for R0=3
        assert 0.93 < r.attack_rate < 0.95

    def test_final_size_equation(self):
        """Attack rate satisfies final size equation."""
        R0 = 2.5
        r = attack_rate(R0=R0)
        S_inf = r.final_susceptible
        # S_∞ = exp(-R0 × (1 - S_∞))
        expected = math.exp(-R0 * (1 - S_inf))
        assert abs(S_inf - expected) < 1e-8

    def test_attack_rate_plus_susceptible_equals_1(self):
        """Attack rate + final susceptible = 1."""
        r = attack_rate(R0=4.0)
        assert abs(r.attack_rate + r.final_susceptible - 1.0) < 1e-10


# ─── SIR Model Tests ──────────────────────────────────────────────────────────

class TestSIRModel:
    """Tests for sir_model()."""

    def test_r0_calculation(self):
        """R0 = beta/gamma."""
        r = sir_model(beta=0.4, gamma=0.1)
        assert abs(r.R0 - 4.0) < 1e-10

    def test_generation_time(self):
        """Generation time = 1/gamma."""
        r = sir_model(beta=0.3, gamma=0.2)
        assert abs(r.generation_time - 5.0) < 1e-10

    def test_herd_immunity(self):
        """Herd immunity = 1 - 1/R0."""
        r = sir_model(beta=0.5, gamma=0.1)  # R0 = 5
        assert abs(r.herd_immunity - 0.8) < 1e-10

    def test_peak_infected_exists(self):
        """Peak infected exists when R0 > 1."""
        r = sir_model(beta=0.3, gamma=0.1)
        assert r.peak_infected is not None
        assert r.peak_infected > 0

    def test_no_peak_when_r0_less_than_1(self):
        """No peak when R0 < 1."""
        r = sir_model(beta=0.05, gamma=0.1)  # R0 = 0.5
        assert r.peak_infected is None


# ─── Vaccine Impact Tests ─────────────────────────────────────────────────────

class TestVaccineImpact:
    """Tests for vaccine_impact()."""

    def test_no_vaccine(self):
        """Zero coverage means Rt = R0."""
        r = vaccine_impact(R0=3.0, vaccine_efficacy=0.9, coverage=0.0)
        assert abs(r.Rt_post_vaccine - 3.0) < 1e-10

    def test_perfect_vaccine_full_coverage(self):
        """100% efficacy, 100% coverage means Rt = 0."""
        r = vaccine_impact(R0=3.0, vaccine_efficacy=1.0, coverage=1.0)
        assert abs(r.Rt_post_vaccine - 0.0) < 1e-10

    def test_partial_coverage(self):
        """50% efficacy, 80% coverage reduces Rt."""
        r = vaccine_impact(R0=3.0, vaccine_efficacy=0.5, coverage=0.8)
        # Rt = 3 × (1 - 0.5 × 0.8) = 3 × 0.6 = 1.8
        assert abs(r.Rt_post_vaccine - 1.8) < 1e-10

    def test_herd_immunity_achieved(self):
        """Sufficient coverage achieves herd immunity."""
        r = vaccine_impact(R0=3.0, vaccine_efficacy=0.9, coverage=0.8)
        # Effective immunity = 0.72, HIT = 0.67, so achieved
        assert r.herd_immunity_reached is True

    def test_herd_immunity_not_achieved(self):
        """Insufficient coverage fails to achieve herd immunity."""
        r = vaccine_impact(R0=3.0, vaccine_efficacy=0.5, coverage=0.5)
        # Effective immunity = 0.25, HIT = 0.67, not achieved
        assert r.herd_immunity_reached is False

    def test_critical_coverage(self):
        """Critical coverage = HIT / VE."""
        r = vaccine_impact(R0=3.0, vaccine_efficacy=0.9, coverage=0.5)
        # HIT = 2/3, critical = (2/3)/0.9 ≈ 0.74
        assert abs(r.critical_coverage - (2/3)/0.9) < 1e-10


# ─── Generation Interval Tests ────────────────────────────────────────────────

class TestGenerationInterval:
    """Tests for generation_interval()."""

    def test_basic_generation_interval(self):
        """Generation interval stored correctly."""
        r = generation_interval(mean_generation=5.0)
        assert r.generation_interval == 5.0

    def test_serial_interval_comparison(self):
        """Serial interval compared to generation interval."""
        r = generation_interval(mean_generation=5.0, mean_serial=4.0)
        assert r.difference == -1.0
        assert "presymptomatic" in r.explanation.lower()

    def test_serial_greater_than_generation(self):
        """Serial > generation means post-symptomatic transmission."""
        r = generation_interval(mean_generation=5.0, mean_serial=6.0)
        assert r.difference == 1.0
        assert "after symptom onset" in r.explanation.lower()


# ─── SEIR Parameters Tests ────────────────────────────────────────────────────

class TestSEIRParameters:
    """Tests for seir_parameters()."""

    def test_parameter_calculation(self):
        """SEIR parameters calculated correctly."""
        p = seir_parameters(R0=3.0, latent_period=5.0, infectious_period=10.0)
        assert abs(p["gamma"] - 0.1) < 1e-10
        assert abs(p["sigma"] - 0.2) < 1e-10
        assert abs(p["beta"] - 0.3) < 1e-10  # R0 × gamma = 3 × 0.1

    def test_generation_time_approximate(self):
        """Generation time approximated as latent + infectious/2."""
        p = seir_parameters(R0=2.0, latent_period=4.0, infectious_period=6.0)
        # Generation time ≈ 4 + 6/2 = 7
        assert abs(p["generation_time"] - 7.0) < 1e-10


# ─── Reference Data Tests ─────────────────────────────────────────────────────

class TestReferenceData:
    """Tests for disease reference data."""

    def test_measles_r0(self):
        """Measles has high R0 (12-18)."""
        r0_range = get_disease_R0("measles")
        assert r0_range is not None
        assert r0_range[0] >= 10

    def test_covid_variants(self):
        """COVID variants have different R0."""
        original = get_disease_R0("covid19_original")
        omicron = get_disease_R0("covid19_omicron")
        assert original is not None
        assert omicron is not None
        assert omicron[0] > original[0]  # Omicron more transmissible

    def test_mers_r0_below_1(self):
        """MERS has R0 < 1 (didn't cause pandemic)."""
        r0_range = get_disease_R0("mers")
        assert r0_range is not None
        assert r0_range[1] < 1.0

    def test_list_diseases(self):
        """list_diseases returns known diseases."""
        diseases = list_diseases()
        assert "measles" in diseases
        assert "influenza_seasonal" in diseases
        assert len(diseases) > 10

    def test_unknown_disease(self):
        """Unknown disease returns None."""
        r0 = get_disease_R0("fake_disease_xyz")
        assert r0 is None


# ─── Error Handling Tests ─────────────────────────────────────────────────────

class TestErrorHandling:
    """Tests for input validation."""

    def test_negative_r0_herd_immunity(self):
        """Negative R0 raises error."""
        with pytest.raises(ValueError):
            herd_immunity_threshold(R0=-1)

    def test_invalid_susceptible_fraction(self):
        """Invalid susceptible fraction raises error."""
        with pytest.raises(ValueError):
            reproduction_number(R0=3.0, susceptible_fraction=1.5)

    def test_negative_gamma(self):
        """Negative gamma raises error."""
        with pytest.raises(ValueError):
            sir_model(beta=0.3, gamma=-0.1)

    def test_invalid_vaccine_efficacy(self):
        """VE > 1 raises error."""
        with pytest.raises(ValueError):
            vaccine_impact(R0=3.0, vaccine_efficacy=1.5, coverage=0.5)


# ─── Report String Tests ──────────────────────────────────────────────────────

class TestReportStrings:
    """Tests for report __str__ methods."""

    def test_herd_immunity_str(self):
        """Herd immunity report is readable."""
        r = herd_immunity_threshold(R0=3.0)
        s = str(r)
        assert "R₀ = 3" in s
        assert "66" in s  # ~66.7%
        assert "1 - 1/R₀" in s

    def test_attack_rate_str(self):
        """Attack rate report is readable."""
        r = attack_rate(R0=2.5)
        s = str(r)
        assert "R₀ = 2.5" in s
        assert "Final attack rate" in s

    def test_vaccine_impact_str(self):
        """Vaccine impact report is readable."""
        r = vaccine_impact(R0=3.0, vaccine_efficacy=0.9, coverage=0.75)
        s = str(r)
        assert "Vaccine efficacy VE" in s
        assert "Rt after vaccination" in s
