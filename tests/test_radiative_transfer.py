"""Tests for noethersolve.radiative_transfer module.

Tests cover:
- CO2 radiative forcing (logarithmic relationship)
- Planck response (no-feedback sensitivity)
- Equilibrium climate sensitivity
- Stefan-Boltzmann radiation
- Effective temperature and greenhouse effect
- Climate feedback analysis
"""

import math
import pytest

from noethersolve.radiative_transfer import (
    radiative_forcing,
    planck_response,
    climate_sensitivity,
    stefan_boltzmann,
    effective_temperature,
    analyze_feedback,
    list_feedbacks,
    RF_COEFFICIENT,
    STEFAN_BOLTZMANN,
    CO2_PREINDUSTRIAL,
)


# ─── Radiative Forcing Tests ────────────────────────────────────────────────

class TestRadiativeForcing:
    """Tests for radiative_forcing()."""

    def test_doubling_co2(self):
        """Doubling CO2 gives ~3.7 W/m²."""
        r = radiative_forcing(co2_final=560, co2_initial=280)
        expected = RF_COEFFICIENT * math.log(2)  # ~3.7 W/m²
        assert abs(r.forcing - expected) < 0.01
        assert abs(r.forcing_per_doubling - expected) < 0.01

    def test_quadrupling_co2(self):
        """Quadrupling CO2 gives ~7.4 W/m² (2x doubling)."""
        r = radiative_forcing(co2_final=1120, co2_initial=280)
        expected = RF_COEFFICIENT * math.log(4)  # ~7.4 W/m²
        assert abs(r.forcing - expected) < 0.01
        # Should be exactly 2x forcing per doubling
        assert abs(r.forcing - 2 * r.forcing_per_doubling) < 0.01

    def test_logarithmic_flag(self):
        """Always indicates logarithmic relationship."""
        r = radiative_forcing(co2_final=400, co2_initial=280)
        assert r.is_logarithmic is True

    def test_preindustrial_to_current(self):
        """280 ppm to ~420 ppm gives ~2 W/m²."""
        r = radiative_forcing(co2_final=420, co2_initial=280)
        # Should be about 2 W/m² (observed value)
        assert 1.8 < r.forcing < 2.4

    def test_ratio_calculated(self):
        """Ratio is correctly calculated."""
        r = radiative_forcing(co2_final=560, co2_initial=280)
        assert abs(r.ratio - 2.0) < 1e-10

    def test_invalid_zero_co2(self):
        """Zero CO2 raises error."""
        with pytest.raises(ValueError):
            radiative_forcing(co2_final=0)

    def test_invalid_negative_co2(self):
        """Negative CO2 raises error."""
        with pytest.raises(ValueError):
            radiative_forcing(co2_final=-100)

    def test_formula_correct(self):
        """Formula ΔF = 5.35 × ln(C/C₀) is applied correctly."""
        r = radiative_forcing(co2_final=400, co2_initial=200)
        expected = 5.35 * math.log(2)  # ratio = 2
        assert abs(r.forcing - expected) < 1e-10


# ─── Planck Response Tests ──────────────────────────────────────────────────

class TestPlanckResponse:
    """Tests for planck_response()."""

    def test_default_earth_temperature(self):
        """Default ~255 K gives expected Planck parameter."""
        r = planck_response(emission_temperature=255)
        # Planck feedback is ~3.76 W/(m²·K) (derivative 4σT³)
        assert 3.5 < r.planck_feedback < 4.0
        # Parameter is inverse: ~0.27 K/(W/m²)
        assert 0.25 < r.planck_parameter < 0.30

    def test_stefan_boltzmann_derivative(self):
        """λ = 1/(4σT³) is exact Stefan-Boltzmann derivative."""
        T = 255.0
        r = planck_response(emission_temperature=T)
        expected_feedback = 4 * STEFAN_BOLTZMANN * (T ** 3)
        assert abs(r.planck_feedback - expected_feedback) < 1e-15

    def test_no_feedback_warming(self):
        """No-feedback warming per doubling is ~1.2 K."""
        r = planck_response()
        # Without feedbacks, CO2 doubling gives ~1.0-1.2 K
        assert 0.9 < r.warming_per_doubling_no_feedback < 1.4

    def test_temperature_dependence(self):
        """Higher temperature → stronger feedback (more radiation)."""
        r1 = planck_response(emission_temperature=250)
        r2 = planck_response(emission_temperature=270)
        # Higher T → higher feedback → lower parameter
        assert r2.planck_feedback > r1.planck_feedback
        assert r2.planck_parameter < r1.planck_parameter

    def test_invalid_temperature(self):
        """Zero or negative temperature raises error."""
        with pytest.raises(ValueError):
            planck_response(emission_temperature=0)
        with pytest.raises(ValueError):
            planck_response(emission_temperature=-100)


# ─── Climate Sensitivity Tests ──────────────────────────────────────────────

class TestClimateSensitivity:
    """Tests for climate_sensitivity()."""

    def test_default_ecs(self):
        """Default feedbacks give ECS in reasonable range."""
        r = climate_sensitivity()
        # Should be in IPCC likely range
        assert 2.0 < r.ecs < 5.0

    def test_ecs_from_feedbacks(self):
        """ECS = ΔF₂ₓ / (-Σfeedbacks)."""
        r = climate_sensitivity()
        forcing_2x = RF_COEFFICIENT * math.log(2)
        expected_ecs = forcing_2x / r.feedback_parameter
        assert abs(r.ecs - expected_ecs) < 0.1

    def test_tcr_ratio(self):
        """TCR is ~70% of ECS."""
        r = climate_sensitivity()
        assert abs(r.tcr - 0.7 * r.ecs) < 0.1

    def test_ipcc_likely_range(self):
        """IPCC AR6 likely range is 2.5-4.0 K."""
        r = climate_sensitivity()
        assert r.likely_range == (2.5, 4.0)

    def test_forcing_per_doubling(self):
        """Forcing per doubling is ~3.7 W/m²."""
        r = climate_sensitivity()
        expected = RF_COEFFICIENT * math.log(2)
        assert abs(r.forcing_2x - expected) < 0.01

    def test_feedback_factor(self):
        """Feedback factor > 1 (positive net feedback)."""
        r = climate_sensitivity()
        # With water vapor feedback, should amplify
        assert r.feedback_factor > 1.0

    def test_derive_from_ecs(self):
        """Can derive feedback parameter from specified ECS."""
        r = climate_sensitivity(ecs=3.0)
        # Should reverse-engineer the feedback parameter
        assert abs(r.ecs - 3.0) < 1e-10


# ─── Stefan-Boltzmann Tests ─────────────────────────────────────────────────

class TestStefanBoltzmann:
    """Tests for stefan_boltzmann()."""

    def test_earth_emission(self):
        """Earth at 255 K emits ~240 W/m²."""
        r = stefan_boltzmann(temperature=255)
        # Should be approximately 240 W/m²
        assert 230 < r.power_density < 250

    def test_sun_surface(self):
        """Sun at 5778 K emits ~6.3×10⁷ W/m²."""
        r = stefan_boltzmann(temperature=5778)
        expected = STEFAN_BOLTZMANN * (5778 ** 4)
        assert abs(r.power_density - expected) < 1

    def test_emissivity_scaling(self):
        """Emissivity < 1 reduces power proportionally."""
        r1 = stefan_boltzmann(temperature=300, emissivity=1.0)
        r2 = stefan_boltzmann(temperature=300, emissivity=0.5)
        assert abs(r2.power_density - 0.5 * r1.power_density) < 1e-10

    def test_formula(self):
        """Power = εσT⁴ exactly."""
        T, eps = 300.0, 0.95
        r = stefan_boltzmann(temperature=T, emissivity=eps)
        expected = eps * STEFAN_BOLTZMANN * (T ** 4)
        assert abs(r.power_density - expected) < 1e-10

    def test_invalid_temperature(self):
        """Zero or negative temperature raises error."""
        with pytest.raises(ValueError):
            stefan_boltzmann(temperature=0)

    def test_invalid_emissivity(self):
        """Emissivity outside 0-1 raises error."""
        with pytest.raises(ValueError):
            stefan_boltzmann(temperature=300, emissivity=1.5)
        with pytest.raises(ValueError):
            stefan_boltzmann(temperature=300, emissivity=-0.1)


# ─── Effective Temperature Tests ────────────────────────────────────────────

class TestEffectiveTemperature:
    """Tests for effective_temperature()."""

    def test_earth_default(self):
        """Earth's effective temp is ~255 K."""
        r = effective_temperature()
        # Should be approximately 255 K
        assert 250 < r.effective_temp < 260

    def test_greenhouse_effect(self):
        """Greenhouse effect is ~33 K for Earth."""
        r = effective_temperature()
        # Should be approximately 33 K warming
        assert 30 < r.greenhouse_effect < 36

    def test_energy_balance(self):
        """Absorbed = σT_eff⁴."""
        r = effective_temperature()
        emitted = STEFAN_BOLTZMANN * (r.effective_temp ** 4)
        assert abs(r.absorbed - emitted) < 1

    def test_albedo_effect(self):
        """Higher albedo → lower effective temp."""
        r1 = effective_temperature(albedo=0.20)
        r2 = effective_temperature(albedo=0.40)
        assert r1.effective_temp > r2.effective_temp

    def test_invalid_albedo(self):
        """Albedo outside 0-1 raises error."""
        with pytest.raises(ValueError):
            effective_temperature(albedo=1.0)  # = 1 means no absorption
        with pytest.raises(ValueError):
            effective_temperature(albedo=-0.1)


# ─── Feedback Analysis Tests ────────────────────────────────────────────────

class TestFeedbackAnalysis:
    """Tests for analyze_feedback()."""

    def test_water_vapor_positive(self):
        """Water vapor feedback is positive (amplifying)."""
        r = analyze_feedback("water_vapor")
        assert r.value > 0
        assert "positive" in r.sign

    def test_planck_negative(self):
        """Planck feedback is negative (damping)."""
        r = analyze_feedback("planck")
        assert r.value < 0
        assert "negative" in r.sign

    def test_cloud_uncertainty(self):
        """Cloud feedback has large uncertainty."""
        r = analyze_feedback("cloud")
        uncertainty_range = r.uncertainty_range[1] - r.uncertainty_range[0]
        assert uncertainty_range > 1.0  # Spans more than 1 W/m²/K

    def test_unknown_feedback(self):
        """Unknown feedback name raises error."""
        with pytest.raises(ValueError):
            analyze_feedback("unknown_feedback")

    def test_list_feedbacks(self):
        """list_feedbacks returns expected names."""
        feedbacks = list_feedbacks()
        assert "planck" in feedbacks
        assert "water_vapor" in feedbacks
        assert "cloud" in feedbacks
        assert len(feedbacks) >= 5


# ─── Report String Tests ────────────────────────────────────────────────────

class TestReportStrings:
    """Tests for report __str__ methods."""

    def test_radiative_forcing_str(self):
        """Radiative forcing report is readable."""
        r = radiative_forcing(co2_final=560, co2_initial=280)
        s = str(r)
        assert "LOGARITHMIC" in s
        assert "5.35" in s or "ln" in s.lower()
        assert "W/m²" in s

    def test_planck_response_str(self):
        """Planck response report is readable."""
        r = planck_response()
        s = str(r)
        assert "Planck" in s
        assert "no feedback" in s.lower() or "No-feedback" in s
        assert "255" in s or "K" in s

    def test_climate_sensitivity_str(self):
        """Climate sensitivity report is readable."""
        r = climate_sensitivity()
        s = str(r)
        assert "ECS" in s
        assert "doubling" in s.lower()
        assert "IPCC" in s

    def test_stefan_boltzmann_str(self):
        """Stefan-Boltzmann report is readable."""
        r = stefan_boltzmann(temperature=300)
        s = str(r)
        assert "Stefan" in s or "σT⁴" in s
        assert "W/m²" in s

    def test_effective_temperature_str(self):
        """Effective temperature report is readable."""
        r = effective_temperature()
        s = str(r)
        assert "Greenhouse" in s
        assert "33" in s or "warming" in s.lower()

    def test_feedback_analysis_str(self):
        """Feedback analysis report is readable."""
        r = analyze_feedback("cloud")
        s = str(r)
        assert "cloud" in s.lower()
        assert "uncertainty" in s.lower() or "W/(m²·K)" in s


# ─── Physical Consistency Tests ─────────────────────────────────────────────

class TestPhysicalConsistency:
    """Tests for physical self-consistency."""

    def test_no_feedback_ecs_matches_planck(self):
        """ECS with only Planck feedback matches no-feedback sensitivity."""
        p = planck_response()
        c = climate_sensitivity(include_feedbacks=["planck"])
        # Should be close to no-feedback warming
        assert abs(c.ecs - p.warming_per_doubling_no_feedback) < 0.3

    def test_effective_temp_energy_balance(self):
        """Absorbed solar = emitted radiation at effective temp."""
        r = effective_temperature()
        emitted = STEFAN_BOLTZMANN * (r.effective_temp ** 4)
        # Should balance within numerical precision
        assert abs(r.absorbed - emitted) < 0.1

    def test_feedback_sum_consistency(self):
        """Sum of feedbacks determines ECS correctly."""
        feedbacks = list_feedbacks()
        total = sum(analyze_feedback(f).value for f in feedbacks)
        c = climate_sensitivity()
        # Net feedback = -feedback_parameter (convention)
        assert abs(total + c.feedback_parameter) < 0.1

    def test_doubling_forcing_consistent(self):
        """Forcing per doubling is consistent across functions."""
        rf = radiative_forcing(co2_final=560, co2_initial=280)
        cs = climate_sensitivity()
        assert abs(rf.forcing - cs.forcing_2x) < 0.01
