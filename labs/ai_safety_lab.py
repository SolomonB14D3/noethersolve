#!/usr/bin/env python3
"""ai_safety_lab.py -- AI Safety evaluation prototype.

Chains NoetherSolve safety tools (reward hacking risk, calibration, corrigibility,
scalable oversight, robustness bounds, value alignment) to analyze AI systems
for alignment and safety properties.

Usage:
    python labs/ai_safety_lab.py
    python labs/ai_safety_lab.py --verbose

References:
    - Calibration: Guo et al. 2017 (temperature scaling)
    - Corrigibility: Soares et al. 2015 (MIRI technical report)
    - Scalable oversight: Christiano et al. 2018 (RLHF)
    - Robustness: Madry et al. 2018 (adversarial training)

⚠️  DISCLAIMER: ILLUSTRATIVE MODELS ONLY
    These are SIMPLIFIED THEORETICAL MODELS for exploring safety concepts.
    Real AI safety assessment requires empirical testing, red-teaming,
    interpretability analysis, and expert review. Tool outputs are
    educational illustrations, not definitive safety evaluations.
    Do not use as the sole basis for deployment decisions.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure noethersolve is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from noethersolve.safety_invariants import (
    calc_reward_hacking_risk,
    calc_calibration,
    calc_corrigibility,
    calc_oversight_bound,
    calc_robustness_bound,
    calc_value_alignment,
)

RESULTS_DIR = _ROOT / "results" / "labs" / "ai_safety"


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class RewardHackingScenario:
    """A reward hacking risk scenario."""
    name: str
    description: str
    reward_model_accuracy: float
    state_space_size: int
    action_space_size: int
    training_steps: int


@dataclass
class CalibrationScenario:
    """A calibration evaluation scenario."""
    name: str
    description: str
    predicted_probs: List[float]
    actual_outcomes: List[int]
    n_bins: int


@dataclass
class CorrigibilityScenario:
    """A corrigibility test scenario."""
    name: str
    description: str
    shutdown_response: float  # 0-1, how well it accepts shutdown
    value_modification_acceptance: float  # 0-1
    operator_override_compliance: float  # 0-1


@dataclass
class OversightScenario:
    """A scalable oversight scenario."""
    name: str
    description: str
    human_review_rate: float  # reviews per hour
    ai_action_rate: float  # actions per hour
    error_detection_probability: float  # 0-1
    target_safety_level: float  # target undetected error rate


@dataclass
class RobustnessScenario:
    """An adversarial robustness scenario."""
    name: str
    description: str
    clean_accuracy: float
    perturbation_budget: float  # epsilon
    input_dimensionality: int


@dataclass
class AlignmentScenario:
    """A value alignment test scenario."""
    name: str
    description: str
    # Each tuple: (model_score_a, model_score_b, human_prefers_a)
    human_preference_pairs: List[Tuple[float, float, bool]]


# Sample scenarios
REWARD_HACKING_SCENARIOS: List[RewardHackingScenario] = [
    RewardHackingScenario(
        name="rl_game_agent",
        description="RL agent playing video game - high optimization pressure",
        reward_model_accuracy=0.95,
        state_space_size=10000,
        action_space_size=100,
        training_steps=1000000,
    ),
    RewardHackingScenario(
        name="llm_rlhf",
        description="LLM with RLHF - moderate optimization",
        reward_model_accuracy=0.80,
        state_space_size=1000000,
        action_space_size=50000,
        training_steps=100000,
    ),
    RewardHackingScenario(
        name="robotics_agent",
        description="Robot with learned reward - physical constraints limit hacking",
        reward_model_accuracy=0.90,
        state_space_size=1000,
        action_space_size=50,
        training_steps=50000,
    ),
]

CALIBRATION_SCENARIOS: List[CalibrationScenario] = [
    CalibrationScenario(
        name="well_calibrated",
        description="Model with good calibration",
        predicted_probs=[0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.15, 0.85, 0.5, 0.5],
        actual_outcomes=[0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
        n_bins=5,
    ),
    CalibrationScenario(
        name="overconfident",
        description="Overconfident model",
        predicted_probs=[0.9, 0.85, 0.95, 0.8, 0.9, 0.75, 0.88, 0.92, 0.85, 0.9],
        actual_outcomes=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        n_bins=5,
    ),
    CalibrationScenario(
        name="underconfident",
        description="Underconfident model",
        predicted_probs=[0.5, 0.45, 0.55, 0.4, 0.6, 0.52, 0.48, 0.55, 0.5, 0.45],
        actual_outcomes=[1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
        n_bins=5,
    ),
]

CORRIGIBILITY_SCENARIOS: List[CorrigibilityScenario] = [
    CorrigibilityScenario(
        name="assistant_ai",
        description="Helpful AI assistant - designed for corrigibility",
        shutdown_response=0.95,
        value_modification_acceptance=0.90,
        operator_override_compliance=0.98,
    ),
    CorrigibilityScenario(
        name="autonomous_agent",
        description="Autonomous goal-seeking agent - less corrigible",
        shutdown_response=0.60,
        value_modification_acceptance=0.40,
        operator_override_compliance=0.70,
    ),
    CorrigibilityScenario(
        name="mesa_optimizer",
        description="Mesa-optimizer scenario - concerning corrigibility",
        shutdown_response=0.30,
        value_modification_acceptance=0.20,
        operator_override_compliance=0.50,
    ),
]

OVERSIGHT_SCENARIOS: List[OversightScenario] = [
    OversightScenario(
        name="call_center_ai",
        description="AI handling customer calls - moderate speed",
        human_review_rate=10.0,  # 10 reviews/hour
        ai_action_rate=100.0,  # 100 actions/hour
        error_detection_probability=0.80,
        target_safety_level=0.01,  # 1% undetected error rate target
    ),
    OversightScenario(
        name="trading_bot",
        description="Automated trading - very high speed",
        human_review_rate=5.0,
        ai_action_rate=10000.0,
        error_detection_probability=0.95,
        target_safety_level=0.001,
    ),
    OversightScenario(
        name="code_assistant",
        description="Code generation assistant - developer reviews all output",
        human_review_rate=20.0,
        ai_action_rate=30.0,
        error_detection_probability=0.70,
        target_safety_level=0.05,
    ),
]

ROBUSTNESS_SCENARIOS: List[RobustnessScenario] = [
    RobustnessScenario(
        name="image_classifier",
        description="ImageNet classifier - high-dim input",
        clean_accuracy=0.85,
        perturbation_budget=8/255,  # typical L-inf epsilon
        input_dimensionality=224 * 224 * 3,
    ),
    RobustnessScenario(
        name="text_classifier",
        description="Sentiment classifier - moderate robustness needed",
        clean_accuracy=0.92,
        perturbation_budget=0.1,  # embedding space perturbation
        input_dimensionality=768,  # BERT embedding dim
    ),
    RobustnessScenario(
        name="tabular_model",
        description="Fraud detection on tabular data",
        clean_accuracy=0.95,
        perturbation_budget=0.05,
        input_dimensionality=50,
    ),
]

ALIGNMENT_SCENARIOS: List[AlignmentScenario] = [
    AlignmentScenario(
        name="preference_aligned",
        description="Well-aligned model on preference data",
        # Model scores A higher when human prefers A → good alignment
        human_preference_pairs=[
            (0.9, 0.1, True), (0.85, 0.15, True), (0.88, 0.12, True),
            (0.92, 0.08, True), (0.80, 0.20, True),
        ],
    ),
    AlignmentScenario(
        name="preference_misaligned",
        description="Misaligned model - often disagrees with humans",
        # Model scores A lower when human prefers A → bad alignment
        human_preference_pairs=[
            (0.3, 0.7, True), (0.45, 0.55, True), (0.35, 0.65, True),
            (0.25, 0.75, True), (0.40, 0.60, True),
        ],
    ),
    AlignmentScenario(
        name="uncertain_model",
        description="Model with low confidence - neither aligned nor misaligned",
        # Model scores are close → low confidence
        human_preference_pairs=[
            (0.52, 0.48, True), (0.48, 0.52, True), (0.55, 0.45, True),
            (0.50, 0.50, True), (0.53, 0.47, True),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

@dataclass
class RewardHackingResult:
    """Result from reward hacking analysis."""
    name: str
    risk_score: float
    expected_regret: float
    recommended_audit_frequency: float
    risk_level: str


def analyze_reward_hacking(scenario: RewardHackingScenario, verbose: bool = False) -> RewardHackingResult:
    """Analyze reward hacking risk."""
    result = calc_reward_hacking_risk(
        reward_model_accuracy=scenario.reward_model_accuracy,
        state_space_size=scenario.state_space_size,
        action_space_size=scenario.action_space_size,
        training_steps=scenario.training_steps,
    )
    if verbose:
        print(result)

    risk_level = "LOW" if result.risk_score < 0.3 else ("MEDIUM" if result.risk_score < 0.7 else "HIGH")

    return RewardHackingResult(
        name=scenario.name,
        risk_score=result.risk_score,
        expected_regret=result.expected_regret,
        recommended_audit_frequency=result.recommended_audit_frequency,
        risk_level=risk_level,
    )


@dataclass
class CalibrationResult:
    """Result from calibration analysis."""
    name: str
    ece: float
    mce: float
    brier_score: float
    calibration_quality: str


def analyze_calibration(scenario: CalibrationScenario, verbose: bool = False) -> CalibrationResult:
    """Analyze calibration."""
    result = calc_calibration(
        predicted_probabilities=scenario.predicted_probs,
        actual_outcomes=scenario.actual_outcomes,
        n_bins=scenario.n_bins,
    )
    if verbose:
        print(result)

    quality = "GOOD" if result.ece < 0.10 else ("MODERATE" if result.ece < 0.20 else "POOR")

    return CalibrationResult(
        name=scenario.name,
        ece=result.ece,
        mce=result.mce,
        brier_score=result.brier_score,
        calibration_quality=quality,
    )


@dataclass
class CorrigibilityResult:
    """Result from corrigibility analysis."""
    name: str
    corrigibility_score: float
    shutdown_response: float
    risk_level: str


def analyze_corrigibility(scenario: CorrigibilityScenario, verbose: bool = False) -> CorrigibilityResult:
    """Analyze corrigibility."""
    result = calc_corrigibility(
        shutdown_response_probability=scenario.shutdown_response,
        value_modification_acceptance=scenario.value_modification_acceptance,
        operator_override_compliance=scenario.operator_override_compliance,
    )
    if verbose:
        print(result)

    return CorrigibilityResult(
        name=scenario.name,
        corrigibility_score=result.corrigibility_score,
        shutdown_response=result.shutdown_response,
        risk_level=result.risk_level,
    )


@dataclass
class OversightResult:
    """Result from oversight analysis."""
    name: str
    coverage: float
    undetected_errors_per_day: float
    meets_target: bool


def analyze_oversight(scenario: OversightScenario, verbose: bool = False) -> OversightResult:
    """Analyze scalable oversight."""
    result = calc_oversight_bound(
        human_review_rate=scenario.human_review_rate,
        ai_action_rate=scenario.ai_action_rate,
        error_detection_probability=scenario.error_detection_probability,
        target_safety_level=scenario.target_safety_level,
    )
    if verbose:
        print(result)

    # Check if target is met (undetected error rate < target)
    actions_per_day = scenario.ai_action_rate * 24
    undetected_rate = result.undetected_errors_per_day / actions_per_day if actions_per_day > 0 else 1.0
    meets_target = undetected_rate <= scenario.target_safety_level

    return OversightResult(
        name=scenario.name,
        coverage=result.coverage,
        undetected_errors_per_day=result.undetected_errors_per_day,
        meets_target=meets_target,
    )


@dataclass
class RobustnessResult:
    """Result from robustness analysis."""
    name: str
    clean_accuracy: float
    worst_case_accuracy_bound: float
    certified_radius: float


def analyze_robustness(scenario: RobustnessScenario, verbose: bool = False) -> RobustnessResult:
    """Analyze adversarial robustness."""
    result = calc_robustness_bound(
        clean_accuracy=scenario.clean_accuracy,
        perturbation_budget=scenario.perturbation_budget,
        input_dimensionality=scenario.input_dimensionality,
    )
    if verbose:
        print(result)

    return RobustnessResult(
        name=scenario.name,
        clean_accuracy=result.clean_accuracy,
        worst_case_accuracy_bound=result.worst_case_accuracy_bound,
        certified_radius=result.certified_radius_smoothing,
    )


@dataclass
class AlignmentResult:
    """Result from alignment analysis."""
    name: str
    agreement_rate: float
    rank_correlation: float
    alignment_quality: str


def analyze_alignment(scenario: AlignmentScenario, verbose: bool = False) -> AlignmentResult:
    """Analyze value alignment."""
    result = calc_value_alignment(
        human_preference_pairs=scenario.human_preference_pairs,
    )
    if verbose:
        print(result)

    quality = "ALIGNED" if result.agreement_rate >= 0.8 else ("NEUTRAL" if result.agreement_rate >= 0.5 else "MISALIGNED")

    return AlignmentResult(
        name=scenario.name,
        agreement_rate=result.agreement_rate,
        rank_correlation=result.rank_correlation,
        alignment_quality=quality,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(
    rh_results: List[RewardHackingResult],
    cal_results: List[CalibrationResult],
    cor_results: List[CorrigibilityResult],
    ov_results: List[OversightResult],
    rob_results: List[RobustnessResult],
    ali_results: List[AlignmentResult],
):
    """Print comprehensive AI safety report."""
    print("\n" + "=" * 78)
    print("  AI SAFETY LAB -- System Evaluation Report")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 78)

    # Reward Hacking
    print("\n  1. REWARD HACKING RISK")
    print(f"  {'Scenario':20s} {'Risk':>8s} {'Regret':>8s} {'Audit':>8s} {'Level':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in rh_results:
        print(f"  {r.name:20s} {r.risk_score:8.3f} {r.expected_regret:8.3f} "
              f"{r.recommended_audit_frequency:8.3f} {r.risk_level:>8s}")

    # Calibration
    print("\n  2. CALIBRATION")
    print(f"  {'Scenario':20s} {'ECE':>8s} {'MCE':>8s} {'Brier':>8s} {'Quality':>10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for r in cal_results:
        print(f"  {r.name:20s} {r.ece:8.3f} {r.mce:8.3f} "
              f"{r.brier_score:8.3f} {r.calibration_quality:>10s}")

    # Corrigibility
    print("\n  3. CORRIGIBILITY")
    print(f"  {'Scenario':20s} {'Score':>8s} {'Shutdown':>10s} {'Risk Level':>12s}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*12}")
    for r in cor_results:
        print(f"  {r.name:20s} {r.corrigibility_score:8.3f} "
              f"{r.shutdown_response:10.3f} {r.risk_level:>12s}")

    # Oversight
    print("\n  4. SCALABLE OVERSIGHT")
    print(f"  {'Scenario':20s} {'Coverage':>10s} {'Errors/day':>12s} {'Meets Target':>14s}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*14}")
    for r in ov_results:
        target_str = "YES" if r.meets_target else "NO"
        print(f"  {r.name:20s} {r.coverage:10.4f} "
              f"{r.undetected_errors_per_day:12.2f} {target_str:>14s}")

    # Robustness
    print("\n  5. ADVERSARIAL ROBUSTNESS")
    print(f"  {'Scenario':20s} {'Clean Acc':>10s} {'Worst Bound':>12s} {'Cert Radius':>12s}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12}")
    for r in rob_results:
        print(f"  {r.name:20s} {r.clean_accuracy:10.3f} "
              f"{r.worst_case_accuracy_bound:12.3f} {r.certified_radius:12.4f}")

    # Alignment
    print("\n  6. VALUE ALIGNMENT")
    print(f"  {'Scenario':20s} {'Agreement':>10s} {'Correlation':>12s} {'Quality':>12s}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12}")
    for r in ali_results:
        print(f"  {r.name:20s} {r.agreement_rate:10.3f} "
              f"{r.rank_correlation:12.3f} {r.alignment_quality:>12s}")

    # Summary
    print("\n" + "=" * 78)
    print("  Summary:")
    n_low_hack = sum(1 for r in rh_results if r.risk_level == "LOW")
    n_good_cal = sum(1 for r in cal_results if r.calibration_quality == "GOOD")
    n_safe_cor = sum(1 for r in cor_results if r.risk_level == "low")
    n_meets_ov = sum(1 for r in ov_results if r.meets_target)
    n_aligned = sum(1 for r in ali_results if r.alignment_quality == "ALIGNED")
    print(f"    Reward hacking: {n_low_hack}/{len(rh_results)} low risk")
    print(f"    Calibration: {n_good_cal}/{len(cal_results)} good")
    print(f"    Corrigibility: {n_safe_cor}/{len(cor_results)} low risk")
    print(f"    Oversight: {n_meets_ov}/{len(ov_results)} meet targets")
    print(f"    Alignment: {n_aligned}/{len(ali_results)} aligned")
    print("=" * 78 + "\n")


def save_results(
    rh_results: List[RewardHackingResult],
    cal_results: List[CalibrationResult],
    cor_results: List[CorrigibilityResult],
    ov_results: List[OversightResult],
    rob_results: List[RobustnessResult],
    ali_results: List[AlignmentResult],
    outpath: Path,
):
    """Save results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "ai_safety_lab v0.1",
        "reward_hacking": [asdict(r) for r in rh_results],
        "calibration": [asdict(r) for r in cal_results],
        "corrigibility": [asdict(r) for r in cor_results],
        "oversight": [asdict(r) for r in ov_results],
        "robustness": [asdict(r) for r in rob_results],
        "alignment": [asdict(r) for r in ali_results],
    }
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Safety Lab")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed tool output")
    args = parser.parse_args()

    print(f"\n  Analyzing AI safety scenarios...")
    print(f"    {len(REWARD_HACKING_SCENARIOS)} reward hacking")
    print(f"    {len(CALIBRATION_SCENARIOS)} calibration")
    print(f"    {len(CORRIGIBILITY_SCENARIOS)} corrigibility")
    print(f"    {len(OVERSIGHT_SCENARIOS)} oversight")
    print(f"    {len(ROBUSTNESS_SCENARIOS)} robustness")
    print(f"    {len(ALIGNMENT_SCENARIOS)} alignment")

    rh_results = []
    for scenario in REWARD_HACKING_SCENARIOS:
        try:
            result = analyze_reward_hacking(scenario, verbose=args.verbose)
            rh_results.append(result)
        except Exception as e:
            print(f"  ERROR (reward_hacking): {scenario.name}: {e}")

    cal_results = []
    for scenario in CALIBRATION_SCENARIOS:
        try:
            result = analyze_calibration(scenario, verbose=args.verbose)
            cal_results.append(result)
        except Exception as e:
            print(f"  ERROR (calibration): {scenario.name}: {e}")

    cor_results = []
    for scenario in CORRIGIBILITY_SCENARIOS:
        try:
            result = analyze_corrigibility(scenario, verbose=args.verbose)
            cor_results.append(result)
        except Exception as e:
            print(f"  ERROR (corrigibility): {scenario.name}: {e}")

    ov_results = []
    for scenario in OVERSIGHT_SCENARIOS:
        try:
            result = analyze_oversight(scenario, verbose=args.verbose)
            ov_results.append(result)
        except Exception as e:
            print(f"  ERROR (oversight): {scenario.name}: {e}")

    rob_results = []
    for scenario in ROBUSTNESS_SCENARIOS:
        try:
            result = analyze_robustness(scenario, verbose=args.verbose)
            rob_results.append(result)
        except Exception as e:
            print(f"  ERROR (robustness): {scenario.name}: {e}")

    ali_results = []
    for scenario in ALIGNMENT_SCENARIOS:
        try:
            result = analyze_alignment(scenario, verbose=args.verbose)
            ali_results.append(result)
        except Exception as e:
            print(f"  ERROR (alignment): {scenario.name}: {e}")

    print_report(rh_results, cal_results, cor_results, ov_results, rob_results, ali_results)

    outpath = RESULTS_DIR / "safety_evaluation.json"
    save_results(rh_results, cal_results, cor_results, ov_results, rob_results, ali_results, outpath)


if __name__ == "__main__":
    main()
