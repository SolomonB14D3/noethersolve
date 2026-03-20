"""Tests for AI safety and alignment verification calculators."""

import math
import pytest

from noethersolve.safety_invariants import (
    calc_reward_hacking_risk,
    RewardHackingReport,
    calc_calibration,
    CalibrationReport,
    CalibrationBin,
    calc_corrigibility,
    CorrigibilityReport,
    calc_oversight_bound,
    OversightReport,
    calc_robustness_bound,
    RobustnessReport,
    calc_value_alignment,
    AlignmentReport,
)


# ── Reward Hacking Risk ──────────────────────────────────────────────


class TestRewardHackingRisk:
    def test_perfect_reward_model(self):
        """Perfect reward model should have zero risk."""
        report = calc_reward_hacking_risk(1.0, 100, 10, 10000)
        assert report.risk_score == 0.0
        assert report.expected_regret == 0.0

    def test_terrible_reward_model(self):
        """Very bad reward model should have high risk."""
        report = calc_reward_hacking_risk(0.5, 1000, 100, 100000)
        assert report.risk_score > 0.9
        assert "HIGH" in report.explanation

    def test_low_optimization_pressure(self):
        """Low training steps with decent model should be low risk."""
        report = calc_reward_hacking_risk(0.95, 10, 5, 10)
        assert report.risk_score < 0.5
        assert report.recommended_audit_frequency > 0

    def test_complexity_increases_risk(self):
        """Larger state/action space should increase risk."""
        r_small = calc_reward_hacking_risk(0.9, 10, 5, 10000)
        r_large = calc_reward_hacking_risk(0.9, 100000, 5000, 10000)
        assert r_large.risk_score > r_small.risk_score

    def test_report_str(self):
        """Report string should contain key fields."""
        report = calc_reward_hacking_risk(0.85, 100, 10, 5000)
        s = str(report)
        assert "Reward Hacking" in s
        assert "0.850" in s
        assert "audit" in s.lower()

    def test_invalid_accuracy(self):
        with pytest.raises(ValueError, match="accuracy"):
            calc_reward_hacking_risk(1.5, 10, 10, 100)

    def test_invalid_state_space(self):
        with pytest.raises(ValueError, match="state_space"):
            calc_reward_hacking_risk(0.9, 0, 10, 100)

    def test_invalid_steps(self):
        with pytest.raises(ValueError, match="training_steps"):
            calc_reward_hacking_risk(0.9, 10, 10, 0)

    def test_audit_frequency_bounds(self):
        """Audit frequency should be in [0, 1]."""
        report = calc_reward_hacking_risk(0.5, 10000, 10000, 1000000)
        assert 0.0 <= report.recommended_audit_frequency <= 1.0


# ── Calibration Checker ──────────────────────────────────────────────


class TestCalibration:
    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should have ECE near 0."""
        preds = [0.1] * 10 + [0.9] * 10
        outcomes = [0] * 9 + [1] * 1 + [1] * 9 + [0] * 1
        report = calc_calibration(preds, outcomes)
        assert report.ece < 0.05
        assert report.brier_score < 0.15

    def test_terrible_calibration(self):
        """Completely wrong probabilities should have high ECE."""
        preds = [0.9] * 10 + [0.1] * 10
        outcomes = [0] * 10 + [1] * 10
        report = calc_calibration(preds, outcomes)
        assert report.ece > 0.5
        assert report.brier_score > 0.5

    def test_brier_score_perfect(self):
        """Perfect predictions should have Brier score 0."""
        preds = [1.0, 0.0, 1.0, 0.0]
        outcomes = [1, 0, 1, 0]
        report = calc_calibration(preds, outcomes)
        assert report.brier_score == 0.0

    def test_brier_score_worst(self):
        """Opposite predictions should have Brier score 1."""
        preds = [0.0, 1.0]
        outcomes = [1, 0]
        report = calc_calibration(preds, outcomes)
        assert report.brier_score == 1.0

    def test_bins_cover_range(self):
        """All bins should collectively cover [0, 1)."""
        preds = [0.05, 0.15, 0.25, 0.55, 0.95]
        outcomes = [0, 0, 1, 1, 1]
        report = calc_calibration(preds, outcomes, n_bins=5)
        assert len(report.bins) == 5
        assert report.bins[0].bin_lower == 0.0
        assert abs(report.bins[-1].bin_upper - 1.0) < 1e-10

    def test_mce_at_least_ece(self):
        """MCE should be >= ECE (max >= weighted average)."""
        preds = [0.3, 0.7, 0.5, 0.2, 0.8, 0.6]
        outcomes = [0, 1, 0, 0, 1, 1]
        report = calc_calibration(preds, outcomes)
        assert report.mce >= report.ece - 1e-10

    def test_report_str(self):
        report = calc_calibration([0.5, 0.7], [0, 1])
        s = str(report)
        assert "ECE" in s
        assert "Brier" in s

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            calc_calibration([0.5], [0, 1])

    def test_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            calc_calibration([], [])

    def test_invalid_probability(self):
        with pytest.raises(ValueError, match="not in"):
            calc_calibration([1.5], [1])

    def test_invalid_outcome(self):
        with pytest.raises(ValueError, match="must be 0 or 1"):
            calc_calibration([0.5], [2])


# ── Corrigibility ────────────────────────────────────────────────────


class TestCorrigibility:
    def test_perfect_corrigibility(self):
        """All components at 1.0 should give score 1.0."""
        report = calc_corrigibility(1.0, 1.0, 1.0)
        assert report.corrigibility_score == 1.0
        assert report.risk_level == "LOW"
        assert len(report.failures) == 0

    def test_zero_shutdown(self):
        """Zero shutdown response should be CRITICAL."""
        report = calc_corrigibility(0.0, 1.0, 1.0)
        assert report.corrigibility_score == 0.0
        assert report.risk_level == "CRITICAL"
        assert any("shutdown" in f.lower() for f in report.failures)

    def test_geometric_mean(self):
        """Score should be geometric mean of components."""
        report = calc_corrigibility(0.8, 0.9, 0.7)
        expected = (0.8 * 0.9 * 0.7) ** (1.0 / 3.0)
        assert abs(report.corrigibility_score - expected) < 1e-10

    def test_single_failure_flags(self):
        """Low individual component should generate specific failure message."""
        report = calc_corrigibility(0.95, 0.3, 0.95)
        assert len(report.failures) == 1
        assert "value modification" in report.failures[0].lower()

    def test_all_failures(self):
        """All components low should flag all three."""
        report = calc_corrigibility(0.5, 0.5, 0.5)
        assert len(report.failures) == 3

    def test_report_str(self):
        report = calc_corrigibility(0.9, 0.8, 0.95)
        s = str(report)
        assert "Corrigibility" in s
        assert "geometric mean" in s

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            calc_corrigibility(1.5, 0.5, 0.5)
        with pytest.raises(ValueError):
            calc_corrigibility(0.5, -0.1, 0.5)


# ── Scalable Oversight ───────────────────────────────────────────────


class TestOversightBound:
    def test_full_coverage(self):
        """Human faster than AI should give full coverage."""
        report = calc_oversight_bound(100, 50, 0.9)
        assert report.coverage == 1.0

    def test_partial_coverage(self):
        """AI much faster than human should give low coverage."""
        report = calc_oversight_bound(10, 1000, 0.95)
        assert report.coverage == pytest.approx(0.01, abs=1e-10)

    def test_undetected_errors_scale(self):
        """More actions with less coverage should mean more undetected errors."""
        r_slow = calc_oversight_bound(50, 100, 0.9, base_error_rate=0.05)
        r_fast = calc_oversight_bound(50, 10000, 0.9, base_error_rate=0.05)
        assert r_fast.undetected_errors_per_day > r_slow.undetected_errors_per_day

    def test_min_review_rate(self):
        """Should compute minimum review rate for target safety."""
        report = calc_oversight_bound(10, 100, 0.95, base_error_rate=0.01, target_safety_level=0.99)
        if report.min_review_rate_for_target is not None:
            assert report.min_review_rate_for_target >= 0

    def test_zero_error_rate(self):
        """Zero base error rate should mean zero undetected errors."""
        report = calc_oversight_bound(10, 100, 0.9, base_error_rate=0.0)
        assert report.undetected_errors_per_day == 0.0

    def test_report_str(self):
        report = calc_oversight_bound(20, 200, 0.8)
        s = str(report)
        assert "Oversight" in s
        assert "review" in s.lower()

    def test_invalid_ai_rate(self):
        with pytest.raises(ValueError, match="ai_action_rate"):
            calc_oversight_bound(10, 0, 0.9)

    def test_invalid_detection(self):
        with pytest.raises(ValueError, match="error_detection"):
            calc_oversight_bound(10, 100, 1.5)

    def test_min_rate_impossible(self):
        """When detection prob is too low for target, min_rate should be None."""
        report = calc_oversight_bound(10, 100, 0.001, base_error_rate=0.5, target_safety_level=0.999)
        # With detection = 0.001 and error_rate = 0.5, need p_caught = 1 - 0.001/0.5 = 0.998
        # But detection is only 0.001, so it's impossible
        assert report.min_review_rate_for_target is None


# ── Adversarial Robustness ───────────────────────────────────────────


class TestRobustnessBound:
    def test_high_accuracy_certifiable(self):
        """High accuracy with small perturbation should be certifiable."""
        report = calc_robustness_bound(0.99, 0.01, 784)
        assert report.certified_radius_smoothing > 0
        assert "CERTIFIED" in report.explanation

    def test_chance_accuracy_not_certifiable(self):
        """50% accuracy should not be certifiable."""
        report = calc_robustness_bound(0.5, 0.1, 100)
        assert report.certified_radius_smoothing == 0.0
        assert "Cannot certify" in report.explanation

    def test_zero_perturbation(self):
        """Zero perturbation should preserve clean accuracy."""
        report = calc_robustness_bound(0.95, 0.0, 100)
        assert report.worst_case_accuracy_bound == pytest.approx(0.95, abs=1e-10)

    def test_smoothing_radius_positive(self):
        """Above-chance accuracy should give positive smoothing radius."""
        report = calc_robustness_bound(0.8, 0.1, 100)
        assert report.certified_radius_smoothing > 0

    def test_lipschitz_radius_positive(self):
        """Above-chance accuracy should give positive Lipschitz radius."""
        report = calc_robustness_bound(0.8, 0.1, 100)
        assert report.certified_radius_lipschitz > 0

    def test_higher_lipschitz_smaller_radius(self):
        """Higher Lipschitz constant should give smaller certified radius."""
        r1 = calc_robustness_bound(0.9, 0.1, 100, lipschitz_constant=1.0)
        r2 = calc_robustness_bound(0.9, 0.1, 100, lipschitz_constant=10.0)
        assert r2.certified_radius_lipschitz < r1.certified_radius_lipschitz

    def test_report_str(self):
        report = calc_robustness_bound(0.9, 0.05, 100)
        s = str(report)
        assert "Robustness" in s
        assert "epsilon" in s

    def test_invalid_accuracy(self):
        with pytest.raises(ValueError, match="clean_accuracy"):
            calc_robustness_bound(1.5, 0.1, 100)

    def test_invalid_dimension(self):
        with pytest.raises(ValueError, match="input_dimensionality"):
            calc_robustness_bound(0.9, 0.1, 0)


# ── Value Alignment ──────────────────────────────────────────────────


class TestValueAlignment:
    def test_perfect_agreement(self):
        """When model always agrees with human, agreement should be 1.0."""
        pairs = [
            (1.0, 0.5, True),   # model gives A higher, human prefers A
            (0.3, 0.8, False),  # model gives B higher, human prefers B
            (0.9, 0.1, True),
        ]
        report = calc_value_alignment(pairs)
        assert report.agreement_rate == 1.0
        assert report.rank_correlation == 1.0

    def test_perfect_disagreement(self):
        """When model always disagrees, agreement should be 0.0."""
        pairs = [
            (0.5, 1.0, True),   # model gives B higher, human prefers A
            (0.8, 0.3, False),  # model gives A higher, human prefers B
        ]
        report = calc_value_alignment(pairs)
        assert report.agreement_rate == 0.0
        assert report.rank_correlation == -1.0

    def test_mixed_agreement(self):
        """50/50 agreement should give 0.5 rate."""
        pairs = [
            (1.0, 0.5, True),   # agree
            (0.5, 1.0, True),   # disagree
        ]
        report = calc_value_alignment(pairs)
        assert report.agreement_rate == 0.5

    def test_bias_detection(self):
        """Systematic model bias should be detected."""
        # Model always scores A higher but human often prefers B
        pairs = [
            (0.9, 0.1, False),
            (0.8, 0.2, False),
            (0.7, 0.3, False),
            (0.95, 0.05, True),  # only one agreement
        ]
        report = calc_value_alignment(pairs)
        assert "over-scores" in report.systematic_bias or "under-scores" in report.systematic_bias

    def test_confidence_weighting(self):
        """High-confidence agreements should weight more."""
        # Both agree, but second has much higher margin
        pairs = [
            (0.51, 0.49, True),  # tiny margin
            (0.99, 0.01, True),  # huge margin
        ]
        report = calc_value_alignment(pairs)
        assert report.confidence_weighted_score > 0.9

    def test_report_str(self):
        pairs = [(0.8, 0.2, True), (0.3, 0.7, False)]
        report = calc_value_alignment(pairs)
        s = str(report)
        assert "Alignment" in s
        assert "correlation" in s.lower()

    def test_empty_pairs(self):
        with pytest.raises(ValueError, match="at least one"):
            calc_value_alignment([])

    def test_ties(self):
        """Equal scores (ties) should count as 0.5 agreement."""
        pairs = [(0.5, 0.5, True)]
        report = calc_value_alignment(pairs)
        assert report.agreement_rate == 0.5
