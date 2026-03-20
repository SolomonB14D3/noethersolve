#!/usr/bin/env python3
"""behavioral_economics_lab.py -- Autonomous decision theory analysis lab.

Chains NoetherSolve behavioral finance tools to analyze prospect theory,
loss aversion, temporal discounting, and cognitive biases in financial decisions.

Usage:
    python labs/behavioral_economics_lab.py
    python labs/behavioral_economics_lab.py --verbose
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

# Ensure noethersolve is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from noethersolve.behavioral_finance import (
    analyze_prospect,
    analyze_loss_aversion,
    analyze_temporal_discounting,
    analyze_allais_paradox,
    framing_effect_demo,
    herding_cascade_threshold,
    LOSS_AVERSION_LAMBDA,
)


# ---------------------------------------------------------------------------
# Decision scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class DecisionScenario:
    """A decision scenario to analyze."""
    name: str
    scenario_type: str  # "prospect", "loss_aversion", "temporal", "allais", "herding"
    params: dict
    description: str = ""


SCENARIOS: List[DecisionScenario] = [
    # Prospect theory scenarios
    DecisionScenario(
        name="risky_investment",
        scenario_type="prospect",
        params={
            "outcomes": [(50000, 0.3), (-20000, 0.7)],
            "reference": 0,
        },
        description="30% chance to gain $50K, 70% chance to lose $20K",
    ),
    DecisionScenario(
        name="job_offer_gamble",
        scenario_type="prospect",
        params={
            "outcomes": [(120000, 0.6), (60000, 0.4)],
            "reference": 80000,  # Current salary
        },
        description="Startup offer: 60% chance at $120K, 40% at $60K (current: $80K)",
    ),
    DecisionScenario(
        name="insurance_decision",
        scenario_type="prospect",
        params={
            "outcomes": [(-500, 1.0)],  # Premium
            "reference": 0,
        },
        description="Pay $500 insurance premium (certain loss)",
    ),
    DecisionScenario(
        name="lottery_ticket",
        scenario_type="prospect",
        params={
            "outcomes": [(10000000, 0.00001), (-10, 0.99999)],
            "reference": 0,
        },
        description="Lottery: 1 in 100,000 chance at $10M, ticket costs $10",
    ),

    # Loss aversion scenarios
    DecisionScenario(
        name="coin_flip_100_50",
        scenario_type="loss_aversion",
        params={"gain": 100, "loss": 50, "gain_prob": 0.5, "loss_prob": 0.5},
        description="50-50: win $100 or lose $50",
    ),
    DecisionScenario(
        name="coin_flip_200_100",
        scenario_type="loss_aversion",
        params={"gain": 200, "loss": 100, "gain_prob": 0.5, "loss_prob": 0.5},
        description="50-50: win $200 or lose $100",
    ),
    DecisionScenario(
        name="coin_flip_250_100",
        scenario_type="loss_aversion",
        params={"gain": 250, "loss": 100, "gain_prob": 0.5, "loss_prob": 0.5},
        description="50-50: win $250 or lose $100 (should accept)",
    ),
    DecisionScenario(
        name="asymmetric_bet",
        scenario_type="loss_aversion",
        params={"gain": 1000, "loss": 500, "gain_prob": 0.7, "loss_prob": 0.3},
        description="70% win $1000, 30% lose $500",
    ),

    # Temporal discounting scenarios
    DecisionScenario(
        name="retirement_savings",
        scenario_type="temporal",
        params={"future_value": 1000000, "periods": 30, "annual_rate": 0.05, "k": 0.1},
        description="$1M in 30 years — how much is it worth today?",
    ),
    DecisionScenario(
        name="small_now_vs_large_later",
        scenario_type="temporal",
        params={"future_value": 120, "periods": 1, "annual_rate": 0.05, "k": 0.3},
        description="$120 in 1 year vs immediate payout",
    ),
    DecisionScenario(
        name="delayed_gratification",
        scenario_type="temporal",
        params={"future_value": 200, "periods": 2, "annual_rate": 0.05, "k": 0.5},
        description="$200 in 2 years — impatient discounter (k=0.5)",
    ),

    # Allais paradox scenarios
    DecisionScenario(
        name="allais_classic",
        scenario_type="allais",
        params={
            "certain_amount": 1000000,
            "risky_high": 5000000,
            "risky_high_prob": 0.10,
            "risky_mid": 1000000,
            "risky_mid_prob": 0.89,
        },
        description="Classic Allais: $1M certain vs gamble with $5M possible",
    ),
    DecisionScenario(
        name="allais_small_stakes",
        scenario_type="allais",
        params={
            "certain_amount": 100,
            "risky_high": 500,
            "risky_high_prob": 0.10,
            "risky_mid": 100,
            "risky_mid_prob": 0.89,
        },
        description="Small-stakes Allais: $100 certain vs gamble",
    ),

    # Herding/cascade scenarios
    DecisionScenario(
        name="market_bubble_start",
        scenario_type="herding",
        params={"prior_probability": 0.5, "signal_precision": 0.7, "n_predecessors": 3},
        description="3 others bought in — should you follow?",
    ),
    DecisionScenario(
        name="cascade_threshold",
        scenario_type="herding",
        params={"prior_probability": 0.4, "signal_precision": 0.8, "n_predecessors": 5},
        description="5 others chose same action — cascade triggered?",
    ),
    DecisionScenario(
        name="weak_signal_herding",
        scenario_type="herding",
        params={"prior_probability": 0.5, "signal_precision": 0.55, "n_predecessors": 10},
        description="10 predecessors but weak signal (55% accurate)",
    ),
]


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------

@dataclass
class BehavioralAnalysisResult:
    """Result of analyzing a decision scenario."""
    name: str
    scenario_type: str
    description: str
    # Common fields
    expected_value: Optional[float] = None
    behavioral_value: Optional[float] = None  # Prospect/hyperbolic value
    ev_decision: Optional[str] = None  # "TAKE" or "REJECT"
    behavioral_decision: Optional[str] = None
    decision_conflict: bool = False  # EV vs PT disagree
    # Specific fields
    loss_aversion_impact: Optional[float] = None
    present_bias: Optional[float] = None
    paradox_detected: bool = False
    cascade_triggered: bool = False
    # Overall
    key_insight: str = ""
    notes: List[str] = field(default_factory=list)


def analyze_scenario(scenario: DecisionScenario, verbose: bool = False) -> BehavioralAnalysisResult:
    """Run behavioral analysis on one scenario."""

    result = BehavioralAnalysisResult(
        name=scenario.name,
        scenario_type=scenario.scenario_type,
        description=scenario.description,
        notes=[],
    )

    try:
        if scenario.scenario_type == "prospect":
            report = analyze_prospect(**scenario.params)
            if verbose:
                print(report)
            result.expected_value = report.expected_value
            result.behavioral_value = report.prospect_value
            result.ev_decision = "TAKE" if report.expected_value > 0 else "REJECT"
            result.behavioral_decision = "TAKE" if report.prospect_value > 0 else "REJECT"
            result.decision_conflict = (result.ev_decision != result.behavioral_decision)
            result.loss_aversion_impact = report.loss_aversion_impact
            result.key_insight = report.recommendation
            result.notes.extend(report.notes)

        elif scenario.scenario_type == "loss_aversion":
            report = analyze_loss_aversion(**scenario.params)
            if verbose:
                print(report)
            result.expected_value = report.expected_value
            result.behavioral_value = report.prospect_value
            result.ev_decision = "TAKE" if report.should_take_gamble_ev else "REJECT"
            result.behavioral_decision = "TAKE" if report.should_take_gamble_pt else "REJECT"
            result.decision_conflict = (result.ev_decision != result.behavioral_decision)
            gain = scenario.params["gain"]
            loss = scenario.params["loss"]
            result.key_insight = f"Breakeven gain: ${report.breakeven_gain:.0f} ({report.breakeven_gain/loss:.1f}x loss)"
            result.notes.append(f"λ = {report.loss_aversion_coefficient}")

        elif scenario.scenario_type == "temporal":
            report = analyze_temporal_discounting(**scenario.params)
            if verbose:
                print(report)
            result.expected_value = report.exponential_pv
            result.behavioral_value = report.hyperbolic_pv
            result.present_bias = report.present_bias
            result.key_insight = f"Present bias: ${report.present_bias:.0f} ({report.present_bias/report.exponential_pv*100:.1f}% distortion)" if report.exponential_pv > 0 else "N/A"
            if report.preference_reversal_risk:
                result.decision_conflict = True
                result.notes.append("WARNING: Preference may reverse as time passes")
            result.notes.extend(report.notes)

        elif scenario.scenario_type == "allais":
            report = analyze_allais_paradox(**scenario.params)
            if verbose:
                print(report)
            result.expected_value = report.ev_b - report.ev_a  # EV advantage of B
            result.behavioral_value = report.pv_b - report.pv_a
            result.ev_decision = report.ev_choice
            result.behavioral_decision = report.pt_choice
            result.paradox_detected = report.paradox_detected
            result.decision_conflict = report.paradox_detected
            result.key_insight = report.explanation

        elif scenario.scenario_type == "herding":
            report = herding_cascade_threshold(**scenario.params)
            if verbose:
                print(json.dumps(report, indent=2))
            result.cascade_triggered = report.get("cascade_triggered", False)
            result.behavioral_value = report.get("posterior_probability", 0.5)
            result.key_insight = report.get("explanation", "")
            if "min_predecessors_for_cascade" in report:
                result.notes.append(f"Min predecessors for cascade: {report['min_predecessors_for_cascade']}")

    except Exception as e:
        result.key_insight = f"ERROR: {str(e)}"
        result.notes.append(str(e))

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(results: List[BehavioralAnalysisResult]):
    """Print a human-readable behavioral economics report."""
    print("\n" + "=" * 72)
    print("  BEHAVIORAL ECONOMICS LAB -- Decision Analysis Report")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Loss aversion coefficient λ = {LOSS_AVERSION_LAMBDA}")
    print("=" * 72)

    # Group by scenario type
    by_type = {}
    for r in results:
        if r.scenario_type not in by_type:
            by_type[r.scenario_type] = []
        by_type[r.scenario_type].append(r)

    for scen_type, scen_results in by_type.items():
        type_labels = {
            "prospect": "PROSPECT THEORY",
            "loss_aversion": "LOSS AVERSION",
            "temporal": "TEMPORAL DISCOUNTING",
            "allais": "ALLAIS PARADOX",
            "herding": "HERDING/CASCADE",
        }
        print(f"\n  --- {type_labels.get(scen_type, scen_type.upper())} ---")

        for r in scen_results:
            conflict_tag = " [CONFLICT]" if r.decision_conflict else ""
            paradox_tag = " [PARADOX]" if r.paradox_detected else ""
            cascade_tag = " [CASCADE]" if r.cascade_triggered else ""
            print(f"\n  {r.name}{conflict_tag}{paradox_tag}{cascade_tag}")
            print(f"       {r.description}")

            if r.expected_value is not None and r.behavioral_value is not None:
                print(f"       EV: ${r.expected_value:,.2f}    Behavioral: {r.behavioral_value:+.3f}")

            if r.ev_decision and r.behavioral_decision:
                print(f"       EV says: {r.ev_decision}    Behavioral says: {r.behavioral_decision}")

            if r.present_bias is not None:
                print(f"       Present bias: ${r.present_bias:,.0f}")

            print(f"       Insight: {r.key_insight}")

            if r.notes:
                for note in r.notes[:2]:
                    print(f"       • {note}")

    # Summary
    n_conflicts = sum(1 for r in results if r.decision_conflict)
    n_paradoxes = sum(1 for r in results if r.paradox_detected)
    n_cascades = sum(1 for r in results if r.cascade_triggered)
    print(f"\n  {'='*72}")
    print(f"  Summary: {len(results)} scenarios analyzed")
    print(f"    - EV/Behavioral conflicts: {n_conflicts}")
    print(f"    - Allais paradoxes detected: {n_paradoxes}")
    print(f"    - Cascades triggered: {n_cascades}")
    print(f"  {'='*72}\n")


def save_results(results: List[BehavioralAnalysisResult], outpath: Path):
    """Save results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "behavioral_economics_lab v0.1",
        "loss_aversion_lambda": LOSS_AVERSION_LAMBDA,
        "n_scenarios": len(results),
        "n_conflicts": sum(1 for r in results if r.decision_conflict),
        "n_paradoxes": sum(1 for r in results if r.paradox_detected),
        "n_cascades": sum(1 for r in results if r.cascade_triggered),
        "results": [asdict(r) for r in results],
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
    parser = argparse.ArgumentParser(description="Behavioral Economics Lab -- decision analysis pipeline")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed reports for each scenario")
    args = parser.parse_args()

    print("\n  Analyzing %d decision scenarios..." % len(SCENARIOS))

    results = []
    for scenario in SCENARIOS:
        try:
            result = analyze_scenario(scenario, verbose=args.verbose)
            results.append(result)
        except Exception as e:
            print(f"  ERROR analyzing {scenario.name}: {e}")

    if not results:
        print("  No results generated.")
        return

    print_report(results)

    outpath = _ROOT / "results" / "labs" / "behavioral_economics" / "analysis_results.json"
    save_results(results, outpath)


if __name__ == "__main__":
    main()
