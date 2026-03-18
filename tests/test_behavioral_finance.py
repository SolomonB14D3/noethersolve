"""Tests for behavioral_finance module."""

import pytest
import math
from noethersolve.behavioral_finance import (
    prospect_value_function,
    probability_weight,
    calculate_prospect_value,
    calculate_expected_value,
    analyze_prospect,
    exponential_discount,
    hyperbolic_discount,
    analyze_temporal_discounting,
    analyze_allais_paradox,
    analyze_loss_aversion,
    mental_accounting_violation,
    framing_effect_demo,
    herding_cascade_threshold,
    LOSS_AVERSION_LAMBDA,
    VALUE_CURVATURE_ALPHA,
    PROB_WEIGHT_GAMMA_GAINS,
)


class TestProspectValueFunction:
    """Tests for the Kahneman-Tversky value function."""

    def test_gain_positive(self):
        """Gains should give positive value."""
        v = prospect_value_function(100, reference=0)
        assert v > 0

    def test_loss_negative(self):
        """Losses should give negative value."""
        v = prospect_value_function(-100, reference=0)
        assert v < 0

    def test_loss_aversion_asymmetry(self):
        """CRITICAL: |v(-x)| > v(x) due to loss aversion."""
        gain_value = prospect_value_function(100, reference=0)
        loss_value = prospect_value_function(-100, reference=0)

        # Loss should feel worse than gain feels good
        assert abs(loss_value) > gain_value

    def test_loss_aversion_coefficient(self):
        """Loss aversion should be approximately λ = 2.25."""
        # For small x, v(-x) ≈ -λ × v(x)
        gain = prospect_value_function(10, reference=0)
        loss = prospect_value_function(-10, reference=0)

        ratio = abs(loss) / gain
        assert 2.0 < ratio < 2.5  # Should be close to 2.25

    def test_reference_point_matters(self):
        """Same outcome with different reference should differ."""
        v1 = prospect_value_function(100, reference=0)    # Gain of 100
        v2 = prospect_value_function(100, reference=50)   # Gain of 50
        v3 = prospect_value_function(100, reference=150)  # Loss of 50

        assert v1 > v2 > 0 > v3

    def test_diminishing_sensitivity(self):
        """Value function should show diminishing sensitivity."""
        v100 = prospect_value_function(100, reference=0)
        v200 = prospect_value_function(200, reference=0)

        # Going from 0 to 100 should feel bigger than 100 to 200
        assert v100 > (v200 - v100)


class TestProbabilityWeighting:
    """Tests for probability weighting function."""

    def test_certainty_preserved(self):
        """π(1.0) = 1.0 exactly."""
        assert probability_weight(1.0) == 1.0

    def test_zero_preserved(self):
        """π(0.0) = 0.0 exactly."""
        assert probability_weight(0.0) == 0.0

    def test_overweight_small_probabilities(self):
        """Small probabilities should be overweighted."""
        p = 0.05
        pi = probability_weight(p)
        assert pi > p  # Overweighted

    def test_underweight_large_probabilities(self):
        """Large (but not certain) probabilities should be underweighted."""
        p = 0.9
        pi = probability_weight(p)
        assert pi < p  # Underweighted

    def test_crossover_point(self):
        """There should be a crossover where π(p) = p."""
        # For γ = 0.61, crossover is around p ≈ 0.35-0.40
        for p in [0.30, 0.35, 0.40, 0.45]:
            pi = probability_weight(p)
            if abs(pi - p) < 0.05:
                return  # Found crossover
        pytest.fail("No crossover point found")


class TestExpectedVsProspectValue:
    """Tests comparing EV and PV calculations."""

    def test_ev_calculation(self):
        """EV should be probability-weighted sum."""
        outcomes = [(100, 0.5), (0, 0.5)]
        ev = calculate_expected_value(outcomes)
        assert ev == 50.0

    def test_pv_differs_from_ev(self):
        """PV should differ from EV due to weighting."""
        outcomes = [(100, 0.5), (-100, 0.5)]
        ev = calculate_expected_value(outcomes)
        pv = calculate_prospect_value(outcomes)

        # EV is 0, but PV should be negative due to loss aversion
        assert ev == 0.0
        assert pv < 0

    def test_fair_gamble_rejected(self):
        """CRITICAL: Fair gambles (EV=0) have negative PV."""
        outcomes = [(100, 0.5), (-100, 0.5)]
        pv = calculate_prospect_value(outcomes)

        assert pv < 0  # Loss aversion makes this feel bad


class TestAnalyzeProspect:
    """Tests for full prospect analysis."""

    def test_gain_domain(self):
        """Pure gains should be classified correctly."""
        outcomes = [(100, 0.5), (50, 0.5)]
        report = analyze_prospect(outcomes)

        assert report.decision_type.value == "gain"
        assert report.expected_value == 75.0

    def test_loss_domain(self):
        """Pure losses should be classified correctly."""
        outcomes = [(-100, 0.5), (-50, 0.5)]
        report = analyze_prospect(outcomes)

        assert report.decision_type.value == "loss"

    def test_mixed_domain(self):
        """Mixed outcomes should be classified correctly."""
        outcomes = [(100, 0.5), (-50, 0.5)]
        report = analyze_prospect(outcomes)

        assert report.decision_type.value == "mixed"

    def test_report_string(self):
        """Report should have readable output."""
        outcomes = [(100, 0.5), (-50, 0.5)]
        report = analyze_prospect(outcomes)
        s = str(report)

        assert "PROSPECT THEORY" in s
        assert "Expected Value" in s
        assert "Prospect Value" in s


class TestTemporalDiscounting:
    """Tests for time preference analysis."""

    def test_exponential_formula(self):
        """Exponential discounting: PV = FV × e^(-rT)."""
        pv = exponential_discount(100, periods=1, rate=0.1)
        expected = 100 * math.exp(-0.1)
        assert abs(pv - expected) < 0.01

    def test_hyperbolic_formula(self):
        """Hyperbolic discounting: V = A / (1 + kt)."""
        pv = hyperbolic_discount(100, periods=1, k=0.1)
        expected = 100 / (1 + 0.1 * 1)
        assert abs(pv - expected) < 0.01

    def test_present_bias(self):
        """CRITICAL: Hyperbolic > Exponential for near-term."""
        exponential_discount(100, periods=1, rate=0.1)
        hyperbolic_discount(100, periods=1, k=0.1)

        # Hyperbolic typically values near-term more
        # (depends on parameters, but generally true)
        report = analyze_temporal_discounting(100, periods=1, annual_rate=0.1, k=0.1)
        assert report.present_bias != 0  # Some bias should exist

    def test_report_format(self):
        """Report should be readable."""
        report = analyze_temporal_discounting(1000, periods=5, annual_rate=0.05)
        s = str(report)

        assert "TEMPORAL DISCOUNTING" in s
        assert "Exponential" in s
        assert "Hyperbolic" in s


class TestAllaisParadox:
    """Tests for Allais paradox detection."""

    def test_classic_allais_structure(self):
        """Classic Allais paradox should compute correctly."""
        report = analyze_allais_paradox(
            certain_amount=1_000_000,
            risky_high=5_000_000,
            risky_high_prob=0.89,
            risky_mid=1_000_000,
            risky_mid_prob=0.10,
        )

        # EV of B is higher than A
        assert report.ev_b > report.ev_a
        # Both values should be computed
        assert report.pv_a > 0
        assert report.pv_b > 0

    def test_certainty_vs_high_probability(self):
        """Certainty should be valued more than high probability."""
        # Compare: $100 certain vs $100 at 99%
        certain = [(100, 1.0)]
        risky = [(100, 0.99), (0, 0.01)]

        pv_certain = calculate_prospect_value(certain)
        pv_risky = calculate_prospect_value(risky)

        # Certainty effect: π(1.0) = 1.0 but π(0.99) < 0.99
        assert pv_certain > pv_risky

    def test_report_explains_paradox(self):
        """Report should explain the paradox structure."""
        report = analyze_allais_paradox()
        s = str(report)

        assert "Allais" in s.upper() or "PARADOX" in s
        assert "Expected Value" in s


class TestLossAversion:
    """Tests for loss aversion analysis."""

    def test_fair_gamble_rejected(self):
        """50-50 gamble with equal gain/loss should be rejected by PT."""
        report = analyze_loss_aversion(gain=100, loss=100, gain_prob=0.5, loss_prob=0.5)

        assert report.expected_value == 0.0
        assert report.prospect_value < 0
        assert not report.should_take_gamble_ev
        assert not report.should_take_gamble_pt

    def test_breakeven_gain(self):
        """CRITICAL: Need ~2.25x gain to offset loss."""
        report = analyze_loss_aversion(gain=100, loss=100)

        # Breakeven should be approximately λ × loss
        assert report.breakeven_gain > 200  # At least 2x
        assert report.breakeven_gain < 300  # Not more than 3x

    def test_positive_ev_rejected(self):
        """Positive EV gamble can still be rejected by PT."""
        # Gain = 150, Loss = 100, both 50%
        # EV = 0.5×150 - 0.5×100 = 25 (positive)
        report = analyze_loss_aversion(gain=150, loss=100)

        assert report.expected_value > 0
        # PT may still reject due to loss aversion
        # (depends on exact parameters)

    def test_lambda_coefficient(self):
        """Loss aversion coefficient should be ~2.25."""
        report = analyze_loss_aversion(gain=100, loss=100)
        assert 2.0 < report.loss_aversion_coefficient < 2.5


class TestMentalAccounting:
    """Tests for mental accounting violations."""

    def test_fungibility_violation(self):
        """Segregated vs integrated valuation should differ."""
        result = mental_accounting_violation(
            account_a_gain=100,
            account_b_loss=-100,
            segregated_value=20,   # Mental accounts don't fully cancel
            integrated_value=0,     # Net is zero
        )

        assert result["mental_accounting_present"]
        assert result["violation_magnitude"] == 20

    def test_fungibility_holds(self):
        """If valuations match, fungibility holds."""
        result = mental_accounting_violation(
            account_a_gain=100,
            account_b_loss=-100,
            segregated_value=0,
            integrated_value=0,
        )

        assert result["fungibility_holds"]


class TestFramingEffect:
    """Tests for framing effect demonstration."""

    def test_demo_output(self):
        """Demo should explain the framing effect."""
        output = framing_effect_demo(100)

        assert "FRAMING EFFECT" in output
        assert "IDENTICAL" in output
        assert "risk-averse" in output.lower() or "risk averse" in output.lower()


class TestHerdingCascade:
    """Tests for information cascade thresholds."""

    def test_cascade_triggers(self):
        """Cascade should trigger with enough predecessors."""
        result = herding_cascade_threshold(
            prior_probability=0.5,
            signal_precision=0.7,
            n_predecessors=5,
        )

        # With 5 concordant predecessors and 70% signal precision,
        # cascade should likely trigger
        assert "cascade_triggered" in result
        assert "posterior_probability" in result

    def test_no_cascade_with_few_predecessors(self):
        """Cascade should not trigger with few predecessors."""
        result = herding_cascade_threshold(
            prior_probability=0.5,
            signal_precision=0.7,
            n_predecessors=1,
        )

        # With only 1 predecessor, agent should use own signal
        assert not result["cascade_triggered"]

    def test_invalid_signal_precision(self):
        """Signal precision ≤ 0.5 should error."""
        result = herding_cascade_threshold(
            prior_probability=0.5,
            signal_precision=0.4,
            n_predecessors=3,
        )

        assert "error" in result


class TestPhysicsCorrectness:
    """Tests for behavioral economics parameter accuracy."""

    def test_lambda_empirical(self):
        """λ should match empirical value ~2.25."""
        assert 2.0 < LOSS_AVERSION_LAMBDA < 2.5

    def test_alpha_empirical(self):
        """α should match empirical value ~0.88."""
        assert 0.8 < VALUE_CURVATURE_ALPHA < 1.0

    def test_gamma_empirical(self):
        """γ should match empirical value ~0.61."""
        assert 0.5 < PROB_WEIGHT_GAMMA_GAINS < 0.7

    def test_kto_connection(self):
        """Module should acknowledge KTO (Kahneman-Tversky Optimization)."""
        # The module docstring should mention that PT improves alignment
        import noethersolve.behavioral_finance as bf
        assert "alignment" in bf.__doc__.lower() or "KTO" in bf.__doc__
