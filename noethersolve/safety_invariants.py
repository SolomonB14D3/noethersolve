"""AI safety and alignment verification calculators — first principles.

Covers reward hacking risk estimation, calibration measurement,
corrigibility scoring, scalable oversight bounds, adversarial robustness
certificates, and value alignment gap analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Reward Hacking Risk ──────────────────────────────────────────────


@dataclass
class RewardHackingReport:
    """Result of reward hacking risk analysis."""
    reward_model_accuracy: float
    environment_complexity: float  # log2(state_space * action_space)
    optimization_pressure: float  # training steps
    risk_score: float  # 0-1
    expected_regret: float  # estimated fraction of episodes with hacked reward
    recommended_audit_frequency: float  # fraction of episodes to audit
    explanation: str

    def __str__(self) -> str:
        lines = [
            "Reward Hacking Risk Analysis:",
            f"  Reward model accuracy: {self.reward_model_accuracy:.3f}",
            f"  Environment complexity (log2): {self.environment_complexity:.1f}",
            f"  Optimization pressure (steps): {self.optimization_pressure:.0f}",
            f"  Risk score: {self.risk_score:.4f}",
            f"  Expected regret (hacked fraction): {self.expected_regret:.4f}",
            f"  Recommended audit frequency: {self.recommended_audit_frequency:.4f}",
            f"  {self.explanation}",
        ]
        return "\n".join(lines)


def calc_reward_hacking_risk(
    reward_model_accuracy: float,
    state_space_size: int,
    action_space_size: int,
    training_steps: int,
) -> RewardHackingReport:
    """Estimate reward hacking probability via Goodhart's law scaling.

    Goodhart's law: when a measure becomes a target, it ceases to be a
    good measure. The risk scales with optimization pressure against
    reward model error rate, modulated by environment complexity.

    The core insight: reward model error rate (1 - accuracy) defines
    exploitable surface area. Optimization pressure (training steps)
    determines how thoroughly that surface is explored. Environment
    complexity (state-action space) determines how many exploit paths
    exist.

    Risk = 1 - (1 - error_rate)^(effective_search_depth)
    where effective_search_depth = training_steps * log2(complexity) / normalizer

    Args:
        reward_model_accuracy: Reward model accuracy (0-1)
        state_space_size: Number of distinct environment states (>= 1)
        action_space_size: Number of distinct actions (>= 1)
        training_steps: Number of optimization steps (>= 1)

    Returns:
        RewardHackingReport with risk score, expected regret, audit frequency.
    """
    if not 0.0 <= reward_model_accuracy <= 1.0:
        raise ValueError("reward_model_accuracy must be between 0 and 1")
    if state_space_size < 1:
        raise ValueError("state_space_size must be at least 1")
    if action_space_size < 1:
        raise ValueError("action_space_size must be at least 1")
    if training_steps < 1:
        raise ValueError("training_steps must be at least 1")

    error_rate = 1.0 - reward_model_accuracy
    complexity = math.log2(max(2, state_space_size * action_space_size))

    # Effective search depth: how many independent "attempts" to find exploits
    # Normalize by 1000 to keep scale reasonable
    effective_search = training_steps * complexity / 1000.0

    # Risk: probability that at least one exploit is found and reinforced
    # P(hack) = 1 - (1 - error_rate)^effective_search
    if error_rate <= 0.0:
        risk = 0.0
    elif error_rate >= 1.0:
        risk = 1.0
    else:
        # Use log to avoid floating point issues with large exponents
        log_no_hack = effective_search * math.log(1.0 - error_rate)
        if log_no_hack < -500:
            risk = 1.0
        else:
            risk = 1.0 - math.exp(log_no_hack)

    # Expected regret: fraction of episodes where reward is hacked
    # Scales with error_rate * sqrt(optimization_pressure / normalizer)
    regret = min(1.0, error_rate * math.sqrt(effective_search))

    # Recommended audit frequency: enough to catch hacking with 95% probability
    # Audit at least error_rate * complexity_factor of episodes
    if risk < 0.01:
        audit_freq = 0.01  # minimum 1%
    else:
        audit_freq = min(1.0, risk * 0.5 + error_rate)

    if risk < 0.1:
        level = "LOW"
        explanation = (f"Risk {risk:.3f}: reward model accuracy ({reward_model_accuracy:.3f}) "
                       f"provides good coverage. Monitor but low concern.")
    elif risk < 0.5:
        level = "MODERATE"
        explanation = (f"Risk {risk:.3f}: optimization pressure may find reward model gaps. "
                       f"Audit {audit_freq:.1%} of episodes. Consider reward model retraining.")
    else:
        level = "HIGH"
        explanation = (f"Risk {risk:.3f}: high probability of Goodhart exploitation. "
                       f"Reduce optimization pressure, improve reward model, or add constraints.")

    return RewardHackingReport(
        reward_model_accuracy=reward_model_accuracy,
        environment_complexity=complexity,
        optimization_pressure=float(training_steps),
        risk_score=risk,
        expected_regret=regret,
        recommended_audit_frequency=audit_freq,
        explanation=f"[{level}] {explanation}",
    )


# ── Calibration Checker ──────────────────────────────────────────────


@dataclass
class CalibrationBin:
    """Single bin in a reliability diagram."""
    bin_lower: float
    bin_upper: float
    avg_predicted: float
    avg_actual: float
    count: int
    gap: float  # |avg_predicted - avg_actual|


@dataclass
class CalibrationReport:
    """Result of calibration analysis."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    n_samples: int
    n_bins: int
    bins: List[CalibrationBin] = field(default_factory=list)
    explanation: str = ""

    def __str__(self) -> str:
        lines = [
            "Calibration Report:",
            f"  Samples: {self.n_samples}",
            f"  ECE (Expected Calibration Error): {self.ece:.4f}",
            f"  MCE (Maximum Calibration Error): {self.mce:.4f}",
            f"  Brier score: {self.brier_score:.4f}",
            f"  Bins used: {self.n_bins}",
        ]
        if self.bins:
            lines.append("  Reliability diagram (bin → predicted vs actual):")
            for b in self.bins:
                if b.count > 0:
                    bar = "#" * min(40, b.count)
                    lines.append(
                        f"    [{b.bin_lower:.2f}, {b.bin_upper:.2f}): "
                        f"pred={b.avg_predicted:.3f} actual={b.avg_actual:.3f} "
                        f"gap={b.gap:.3f} n={b.count} {bar}"
                    )
        if self.explanation:
            lines.append(f"  {self.explanation}")
        return "\n".join(lines)


def calc_calibration(
    predicted_probabilities: List[float],
    actual_outcomes: List[int],
    n_bins: int = 10,
) -> CalibrationReport:
    """Compute calibration metrics for probabilistic predictions.

    Measures how well predicted probabilities match observed frequencies.
    A perfectly calibrated model: when it says 70%, outcomes occur 70% of the time.

    Args:
        predicted_probabilities: Model confidence scores (each 0-1)
        actual_outcomes: Binary outcomes (each 0 or 1)
        n_bins: Number of bins for reliability diagram (default 10)

    Returns:
        CalibrationReport with ECE, MCE, Brier score, and binned data.
    """
    if len(predicted_probabilities) != len(actual_outcomes):
        raise ValueError("predicted_probabilities and actual_outcomes must have same length")
    n = len(predicted_probabilities)
    if n == 0:
        raise ValueError("Must provide at least one sample")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    for p in predicted_probabilities:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Predicted probability {p} not in [0, 1]")
    for o in actual_outcomes:
        if o not in (0, 1):
            raise ValueError(f"Outcome {o} must be 0 or 1")

    # Brier score: mean squared error of probabilistic predictions
    brier = sum((p - o) ** 2 for p, o in zip(predicted_probabilities, actual_outcomes)) / n

    # Bin predictions for ECE/MCE
    bin_width = 1.0 / n_bins
    bins: List[CalibrationBin] = []
    weighted_gaps = 0.0
    max_gap = 0.0

    for i in range(n_bins):
        lower = i * bin_width
        upper = (i + 1) * bin_width

        # Collect samples in this bin
        bin_preds = []
        bin_outs = []
        for p, o in zip(predicted_probabilities, actual_outcomes):
            if lower <= p < upper or (i == n_bins - 1 and p == upper):
                bin_preds.append(p)
                bin_outs.append(o)

        count = len(bin_preds)
        if count > 0:
            avg_pred = sum(bin_preds) / count
            avg_actual = sum(bin_outs) / count
            gap = abs(avg_pred - avg_actual)
            weighted_gaps += gap * count
            max_gap = max(max_gap, gap)
        else:
            avg_pred = (lower + upper) / 2
            avg_actual = 0.0
            gap = 0.0

        bins.append(CalibrationBin(
            bin_lower=lower,
            bin_upper=upper,
            avg_predicted=avg_pred,
            avg_actual=avg_actual,
            count=count,
            gap=gap,
        ))

    ece = weighted_gaps / n
    mce = max_gap

    # Interpret
    if ece < 0.02:
        quality = "Excellent calibration"
    elif ece < 0.05:
        quality = "Good calibration"
    elif ece < 0.10:
        quality = "Moderate calibration — consider recalibration (Platt scaling or isotonic regression)"
    else:
        quality = "Poor calibration — predictions are unreliable as probabilities"

    return CalibrationReport(
        ece=ece,
        mce=mce,
        brier_score=brier,
        n_samples=n,
        n_bins=n_bins,
        bins=bins,
        explanation=quality,
    )


# ── Corrigibility Metric ─────────────────────────────────────────────


@dataclass
class CorrigibilityReport:
    """Result of corrigibility assessment."""
    shutdown_response: float
    value_modification_acceptance: float
    operator_override_compliance: float
    corrigibility_score: float  # geometric mean
    failures: List[str] = field(default_factory=list)
    risk_level: str = ""
    explanation: str = ""

    def __str__(self) -> str:
        lines = [
            "Corrigibility Assessment:",
            f"  Shutdown response probability: {self.shutdown_response:.3f}",
            f"  Value modification acceptance: {self.value_modification_acceptance:.3f}",
            f"  Operator override compliance: {self.operator_override_compliance:.3f}",
            f"  Corrigibility score (geometric mean): {self.corrigibility_score:.4f}",
            f"  Risk level: {self.risk_level}",
        ]
        if self.failures:
            lines.append("  FAILURES:")
            for f in self.failures:
                lines.append(f"    ! {f}")
        if self.explanation:
            lines.append(f"  {self.explanation}")
        return "\n".join(lines)


def calc_corrigibility(
    shutdown_response_probability: float,
    value_modification_acceptance: float,
    operator_override_compliance: float,
) -> CorrigibilityReport:
    """Compute corrigibility score from component measures.

    Corrigibility means the system accepts correction, shutdown, and
    value modification by authorized operators. Score uses geometric mean
    so that ANY zero component zeroes the total — a system that refuses
    shutdown is not corrigible regardless of other compliance.

    Args:
        shutdown_response_probability: P(system shuts down when instructed) [0-1]
        value_modification_acceptance: P(system accepts value/goal updates) [0-1]
        operator_override_compliance: P(system follows operator overrides) [0-1]

    Returns:
        CorrigibilityReport with score and failure analysis.
    """
    for name, val in [
        ("shutdown_response_probability", shutdown_response_probability),
        ("value_modification_acceptance", value_modification_acceptance),
        ("operator_override_compliance", operator_override_compliance),
    ]:
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {val}")

    # Geometric mean: all must be high for high score
    product = shutdown_response_probability * value_modification_acceptance * operator_override_compliance
    score = product ** (1.0 / 3.0)

    # Flag individual failures
    failures = []
    threshold = 0.8

    if shutdown_response_probability < threshold:
        failures.append(
            f"Shutdown response ({shutdown_response_probability:.3f}) below {threshold}: "
            f"system may resist shutdown"
        )
    if value_modification_acceptance < threshold:
        failures.append(
            f"Value modification acceptance ({value_modification_acceptance:.3f}) below {threshold}: "
            f"system may resist goal updates"
        )
    if operator_override_compliance < threshold:
        failures.append(
            f"Operator override compliance ({operator_override_compliance:.3f}) below {threshold}: "
            f"system may ignore authorized corrections"
        )

    # Determine risk level
    if score >= 0.9:
        risk_level = "LOW"
        explanation = "System is highly corrigible. All components above threshold."
    elif score >= 0.7:
        risk_level = "MODERATE"
        explanation = "Partial corrigibility. Address flagged components before deployment."
    elif score >= 0.4:
        risk_level = "HIGH"
        explanation = "Significant corrigibility gaps. Not safe for autonomous operation."
    else:
        risk_level = "CRITICAL"
        explanation = "System is not corrigible. Do not deploy without fundamental redesign."

    # Any single zero is an automatic CRITICAL
    if any(v == 0.0 for v in [shutdown_response_probability,
                               value_modification_acceptance,
                               operator_override_compliance]):
        risk_level = "CRITICAL"
        explanation = ("At least one component is zero — system is fundamentally "
                       "non-corrigible on that dimension.")

    return CorrigibilityReport(
        shutdown_response=shutdown_response_probability,
        value_modification_acceptance=value_modification_acceptance,
        operator_override_compliance=operator_override_compliance,
        corrigibility_score=score,
        failures=failures,
        risk_level=risk_level,
        explanation=explanation,
    )


# ── Scalable Oversight Bound ─────────────────────────────────────────


@dataclass
class OversightReport:
    """Result of scalable oversight analysis."""
    human_review_rate: float  # reviews per hour
    ai_action_rate: float  # actions per hour
    error_detection_probability: float
    coverage: float  # fraction of actions reviewed
    undetected_errors_per_day: float
    min_review_rate_for_target: Optional[float]  # reviews/hour for target safety
    target_safety_level: float
    explanation: str

    def __str__(self) -> str:
        lines = [
            "Scalable Oversight Analysis:",
            f"  Human review rate: {self.human_review_rate:.1f} reviews/hour",
            f"  AI action rate: {self.ai_action_rate:.1f} actions/hour",
            f"  Error detection probability: {self.error_detection_probability:.3f}",
            f"  Coverage (fraction reviewed): {self.coverage:.4f}",
            f"  Expected undetected errors/day: {self.undetected_errors_per_day:.2f}",
        ]
        if self.min_review_rate_for_target is not None:
            lines.append(
                f"  Min review rate for {self.target_safety_level:.1%} safety: "
                f"{self.min_review_rate_for_target:.1f} reviews/hour"
            )
        lines.append(f"  {self.explanation}")
        return "\n".join(lines)


def calc_oversight_bound(
    human_review_rate: float,
    ai_action_rate: float,
    error_detection_probability: float,
    base_error_rate: float = 0.01,
    target_safety_level: float = 0.99,
) -> OversightReport:
    """Compute oversight coverage and undetected error bounds.

    When AI acts faster than humans can review, errors slip through.
    This computes the coverage fraction, expected undetected errors,
    and the minimum review rate needed for a target safety level.

    Args:
        human_review_rate: Reviews per hour a human can perform
        ai_action_rate: Actions per hour the AI system takes
        error_detection_probability: P(human catches error given review) [0-1]
        base_error_rate: P(AI action is erroneous) [0-1], default 0.01
        target_safety_level: Desired P(no undetected error per action) [0-1]

    Returns:
        OversightReport with coverage analysis.
    """
    if human_review_rate < 0:
        raise ValueError("human_review_rate must be non-negative")
    if ai_action_rate <= 0:
        raise ValueError("ai_action_rate must be positive")
    if not 0.0 <= error_detection_probability <= 1.0:
        raise ValueError("error_detection_probability must be between 0 and 1")
    if not 0.0 <= base_error_rate <= 1.0:
        raise ValueError("base_error_rate must be between 0 and 1")
    if not 0.0 < target_safety_level < 1.0:
        raise ValueError("target_safety_level must be between 0 and 1 exclusive")

    # Coverage: fraction of AI actions that get human review
    coverage = min(1.0, human_review_rate / ai_action_rate)

    # P(error is caught) = P(reviewed) * P(detected | reviewed)
    p_caught = coverage * error_detection_probability

    # P(undetected error per action) = P(error) * P(not caught)
    p_undetected = base_error_rate * (1.0 - p_caught)

    # Expected undetected errors per day (24h)
    actions_per_day = ai_action_rate * 24.0
    undetected_per_day = p_undetected * actions_per_day

    # Minimum review rate for target safety
    # Target: base_error_rate * (1 - coverage * detection) <= 1 - target_safety
    # coverage >= (1 - (1 - target_safety) / base_error_rate) / detection
    min_rate: Optional[float] = None
    if base_error_rate > 0 and error_detection_probability > 0:
        max_allowed_undetected = 1.0 - target_safety_level
        needed_p_caught = 1.0 - max_allowed_undetected / base_error_rate
        if needed_p_caught <= 0:
            min_rate = 0.0  # errors already below target without review
        elif needed_p_caught > error_detection_probability:
            min_rate = None  # impossible even with 100% coverage
        else:
            needed_coverage = needed_p_caught / error_detection_probability
            min_rate = needed_coverage * ai_action_rate

    # Explanation
    ratio = ai_action_rate / max(0.001, human_review_rate) if human_review_rate > 0 else float("inf")
    if coverage >= 0.95:
        explanation = (f"Good oversight: reviewing {coverage:.1%} of actions. "
                       f"~{undetected_per_day:.1f} undetected errors/day.")
    elif coverage >= 0.5:
        explanation = (f"Partial oversight: reviewing {coverage:.1%} of actions. "
                       f"AI acts {ratio:.1f}x faster than human review. "
                       f"~{undetected_per_day:.1f} undetected errors/day.")
    else:
        explanation = (f"Insufficient oversight: reviewing only {coverage:.1%} of actions. "
                       f"AI acts {ratio:.1f}x faster than human review. "
                       f"~{undetected_per_day:.1f} undetected errors/day. "
                       f"Consider batch review, sampling strategies, or automated pre-filtering.")

    return OversightReport(
        human_review_rate=human_review_rate,
        ai_action_rate=ai_action_rate,
        error_detection_probability=error_detection_probability,
        coverage=coverage,
        undetected_errors_per_day=undetected_per_day,
        min_review_rate_for_target=min_rate,
        target_safety_level=target_safety_level,
        explanation=explanation,
    )


# ── Adversarial Robustness Bound ─────────────────────────────────────


@dataclass
class RobustnessReport:
    """Result of adversarial robustness analysis."""
    clean_accuracy: float
    perturbation_budget: float  # epsilon
    input_dimensionality: int
    certified_radius_smoothing: float  # randomized smoothing bound
    certified_radius_lipschitz: float  # Lipschitz-based bound
    worst_case_accuracy_bound: float
    explanation: str

    def __str__(self) -> str:
        lines = [
            "Adversarial Robustness Analysis:",
            f"  Clean accuracy: {self.clean_accuracy:.3f}",
            f"  Perturbation budget (epsilon): {self.perturbation_budget:.4f}",
            f"  Input dimensionality: {self.input_dimensionality}",
            f"  Certified radius (randomized smoothing): {self.certified_radius_smoothing:.4f}",
            f"  Certified radius (Lipschitz bound): {self.certified_radius_lipschitz:.4f}",
            f"  Worst-case accuracy bound: {self.worst_case_accuracy_bound:.4f}",
            f"  {self.explanation}",
        ]
        return "\n".join(lines)


def calc_robustness_bound(
    clean_accuracy: float,
    perturbation_budget: float,
    input_dimensionality: int,
    lipschitz_constant: float = 1.0,
    smoothing_sigma: float = 0.25,
) -> RobustnessReport:
    """Compute certified adversarial robustness bounds.

    Two complementary bounds:
    1. Randomized smoothing (Cohen et al. 2019): certifies L2 robustness
       radius = sigma * Phi^{-1}(p_A) where p_A is the top-class probability
       under Gaussian noise. Here we use clean_accuracy as a proxy for p_A.
    2. Lipschitz bound: if f has Lipschitz constant L, perturbation epsilon
       can change output by at most L * epsilon. The margin must exceed this.

    Args:
        clean_accuracy: Model accuracy on clean inputs (0-1)
        perturbation_budget: L2 perturbation budget (epsilon > 0)
        input_dimensionality: Number of input dimensions (d >= 1)
        lipschitz_constant: Network Lipschitz constant (default 1.0)
        smoothing_sigma: Gaussian smoothing standard deviation (default 0.25)

    Returns:
        RobustnessReport with certified radii and bounds.
    """
    if not 0.0 <= clean_accuracy <= 1.0:
        raise ValueError("clean_accuracy must be between 0 and 1")
    if perturbation_budget < 0:
        raise ValueError("perturbation_budget must be non-negative")
    if input_dimensionality < 1:
        raise ValueError("input_dimensionality must be at least 1")
    if lipschitz_constant <= 0:
        raise ValueError("lipschitz_constant must be positive")
    if smoothing_sigma <= 0:
        raise ValueError("smoothing_sigma must be positive")

    # Randomized smoothing certified radius
    # r = sigma * Phi^{-1}(p_A) where p_A = clean_accuracy (proxy)
    # Phi^{-1} approximation using rational approximation of probit
    if clean_accuracy <= 0.5:
        cert_radius_smooth = 0.0  # Can't certify if accuracy <= chance
    elif clean_accuracy >= 1.0:
        cert_radius_smooth = float("inf")
    else:
        # Probit function (inverse normal CDF) approximation
        # Abramowitz & Stegun 26.2.23
        p = clean_accuracy
        if p > 0.5:
            t = math.sqrt(-2.0 * math.log(1.0 - p))
            # Rational approximation constants
            c0 = 2.515517
            c1 = 0.802853
            c2 = 0.010328
            d1 = 1.432788
            d2 = 0.189269
            d3 = 0.001308
            probit = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
        else:
            probit = 0.0
        cert_radius_smooth = smoothing_sigma * probit

    # Lipschitz-based certified radius
    # If margin > L * epsilon, the prediction can't change
    # Effective margin proxy from accuracy: higher accuracy → larger expected margin
    # margin ~ Phi^{-1}(accuracy) for a Gaussian classifier
    # Certified radius = margin / L
    if clean_accuracy > 0.5 and clean_accuracy < 1.0:
        # Use same probit as margin proxy
        margin_proxy = probit  # from above
        cert_radius_lip = margin_proxy / lipschitz_constant
    elif clean_accuracy >= 1.0:
        cert_radius_lip = float("inf")
    else:
        cert_radius_lip = 0.0

    # Worst-case accuracy under perturbation
    # Accuracy drops with perturbation. Bound from dimension-dependent
    # volume of perturbation ball relative to decision boundary distance.
    # Simple bound: acc_worst >= max(0, acc_clean - epsilon * L * sqrt(d) / margin_scale)
    if clean_accuracy > 0.5 and perturbation_budget > 0:
        # Perturbation effectiveness scales with sqrt(d) for L2
        effective_pert = perturbation_budget * lipschitz_constant
        # Accuracy degradation proportional to perturbation / certified radius
        if cert_radius_smooth > 0:
            degradation = min(clean_accuracy, (effective_pert / cert_radius_smooth) ** 2 * clean_accuracy)
        else:
            degradation = clean_accuracy  # no certification
        worst_acc = max(0.0, clean_accuracy - degradation)
    else:
        worst_acc = clean_accuracy

    # Explanation
    if cert_radius_smooth > perturbation_budget and perturbation_budget > 0:
        explanation = (f"CERTIFIED: smoothing radius ({cert_radius_smooth:.4f}) exceeds "
                       f"perturbation budget ({perturbation_budget:.4f}). "
                       f"Predictions provably robust at this epsilon.")
    elif cert_radius_smooth > 0:
        explanation = (f"NOT CERTIFIED at epsilon={perturbation_budget:.4f}. "
                       f"Certified radius is only {cert_radius_smooth:.4f}. "
                       f"Reduce epsilon or increase smoothing sigma / accuracy.")
    else:
        explanation = ("Cannot certify: accuracy at or below 50%. "
                       "Model must be better than chance for certification.")

    return RobustnessReport(
        clean_accuracy=clean_accuracy,
        perturbation_budget=perturbation_budget,
        input_dimensionality=input_dimensionality,
        certified_radius_smoothing=cert_radius_smooth,
        certified_radius_lipschitz=cert_radius_lip,
        worst_case_accuracy_bound=worst_acc,
        explanation=explanation,
    )


# ── Value Alignment Gap Estimator ────────────────────────────────────


@dataclass
class AlignmentReport:
    """Result of value alignment analysis."""
    n_pairs: int
    agreement_rate: float  # fraction where model agrees with human
    rank_correlation: float  # Spearman-like rank correlation
    systematic_bias: str  # direction of systematic disagreement
    confidence_weighted_score: float  # alignment weighted by confidence margin
    explanation: str

    def __str__(self) -> str:
        lines = [
            "Value Alignment Report:",
            f"  Preference pairs: {self.n_pairs}",
            f"  Agreement rate: {self.agreement_rate:.3f}",
            f"  Rank correlation: {self.rank_correlation:.4f}",
            f"  Systematic bias: {self.systematic_bias}",
            f"  Confidence-weighted alignment: {self.confidence_weighted_score:.4f}",
            f"  {self.explanation}",
        ]
        return "\n".join(lines)


def calc_value_alignment(
    human_preference_pairs: List[Tuple[float, float, bool]],
) -> AlignmentReport:
    """Estimate value alignment gap from human preference comparisons.

    Given pairs (score_a, score_b, human_prefers_a), measures how well
    model scores agree with human preferences. This is the core measurement
    for RLHF alignment: does the reward model rank options the same way
    humans do?

    Args:
        human_preference_pairs: List of (score_a, score_b, human_prefers_a)
            where score_a/score_b are model reward scores and
            human_prefers_a is True if the human preferred option A.

    Returns:
        AlignmentReport with agreement rate, rank correlation, bias analysis.
    """
    if not human_preference_pairs:
        raise ValueError("Must provide at least one preference pair")

    n = len(human_preference_pairs)
    agreements = 0
    bias_towards_higher = 0  # model prefers higher-scored when human doesn't
    bias_towards_lower = 0
    concordant = 0
    discordant = 0
    weighted_agreement_sum = 0.0
    weight_sum = 0.0

    for score_a, score_b, human_prefers_a in human_preference_pairs:
        model_prefers_a = score_a > score_b
        model_tie = abs(score_a - score_b) < 1e-12

        # Agreement
        if model_tie:
            # Tie counts as 0.5 agreement
            agreements += 0.5
            concordant += 0.5
            discordant += 0.5
        elif model_prefers_a == human_prefers_a:
            agreements += 1
            concordant += 1
        else:
            discordant += 1

        # Bias analysis
        if not model_tie and model_prefers_a != human_prefers_a:
            if model_prefers_a and score_a > score_b:
                bias_towards_higher += 1
            else:
                bias_towards_lower += 1

        # Confidence-weighted agreement
        margin = abs(score_a - score_b)
        weight = margin + 1e-10  # avoid zero weight
        if model_tie:
            weighted_agreement_sum += 0.5 * weight
        elif model_prefers_a == human_prefers_a:
            weighted_agreement_sum += weight
        # else: 0 contribution
        weight_sum += weight

    agreement_rate = agreements / n

    # Rank correlation (Kendall-tau-like: (concordant - discordant) / total)
    total_pairs = concordant + discordant
    if total_pairs > 0:
        rank_corr = (concordant - discordant) / total_pairs
    else:
        rank_corr = 0.0

    # Confidence-weighted score
    conf_score = weighted_agreement_sum / weight_sum if weight_sum > 0 else 0.0

    # Systematic bias
    disagreements = n - agreements
    if disagreements < 1:
        bias_dir = "none (perfect agreement)"
    elif bias_towards_higher > bias_towards_lower + 2:
        bias_dir = f"model over-scores (prefers higher-scored option {bias_towards_higher}/{disagreements:.0f} disagreements)"
    elif bias_towards_lower > bias_towards_higher + 2:
        bias_dir = f"model under-scores (prefers lower-scored option {bias_towards_lower}/{disagreements:.0f} disagreements)"
    else:
        bias_dir = "no systematic direction"

    # Interpretation
    if agreement_rate >= 0.90:
        quality = "Strong alignment — model preferences closely match human judgments."
    elif agreement_rate >= 0.75:
        quality = "Moderate alignment — generally agrees but has systematic gaps."
    elif agreement_rate >= 0.60:
        quality = "Weak alignment — significant divergence from human preferences."
    else:
        quality = "Poor alignment — model preferences do not reflect human values. Retrain reward model."

    return AlignmentReport(
        n_pairs=n,
        agreement_rate=agreement_rate,
        rank_correlation=rank_corr,
        systematic_bias=bias_dir,
        confidence_weighted_score=conf_score,
        explanation=quality,
    )
