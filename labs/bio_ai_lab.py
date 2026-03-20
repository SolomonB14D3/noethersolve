#!/usr/bin/env python3
"""bio_ai_lab.py -- Find convergent solutions between biological and AI systems.

Chains NoetherSolve bio-AI tools to run three test scenarios:
  1. Navigation agent vs bacterial chemotaxis
  2. RL agent vs dopamine reward system
  3. Multi-agent coordination vs swarm consensus

Each scenario exercises multiple tools, scores convergence, and outputs
a unified report to results/labs/bio_ai/convergence_results.json.

Usage:
    python labs/bio_ai_lab.py
    python labs/bio_ai_lab.py --verbose
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from noethersolve.bio_ai_bridge import (
    compare_agent_to_worm,
    identify_convergent_solutions,
    map_behavior_to_architecture,
)
from noethersolve.chemotaxis_model import simulate_chemotaxis, check_perfect_adaptation
from noethersolve.c_elegans_behavior import drift_diffusion_decision
from noethersolve.neural_rl_analogy import (
    compare_hebbian_backprop,
    map_striatum_to_actor_critic,
)
from noethersolve.collective_behavior import swarm_consensus

RESULTS_DIR = _ROOT / "results" / "labs" / "bio_ai"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """Result of running one convergence scenario."""
    name: str
    description: str
    tools_used: List[str]
    convergence_score: float  # 0-1
    verdict: str              # CONVERGENT / DIVERGENT / PARTIAL
    findings: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scenario 1: Navigation agent vs chemotaxis
# ---------------------------------------------------------------------------

def scenario_navigation(verbose: bool = False) -> ScenarioResult:
    """Compare AI navigation to bacterial chemotaxis."""
    print("\n  [Scenario 1] Navigation Agent vs Bacterial Chemotaxis")
    findings: List[str] = []
    details: Dict = {}

    # 1a. Simulate chemotaxis toward a source
    traj = simulate_chemotaxis(
        duration=60.0, dt=0.1,
        source_position=(80.0, 80.0),
        initial_position=(0.0, 0.0),
        base_tumble_rate=1.0,
        adaptation_time=5.0,
        speed=20.0,
        gradient_sensitivity=1.0,
        seed=42,
    )
    if verbose:
        print(traj)

    details["chemotaxis"] = {
        "gradient_following_score": traj.gradient_following_score,
        "effective_velocity": traj.effective_velocity,
        "n_tumbles": len(traj.tumble_events),
        "final_concentration": traj.final_concentration,
    }
    findings.append(
        f"Chemotaxis gradient_score={traj.gradient_following_score:.3f}, "
        f"effective_velocity={traj.effective_velocity:.3f}"
    )

    # 1a2. Test perfect adaptation at different time constants
    # Perfect adaptation = return to baseline after sustained perturbation (integral feedback)
    # Build exponential decay model for bacterial chemotaxis receptor methylation
    adaptation_tests = []
    for tau in [2.0, 5.0, 10.0]:
        # Chemotaxis response: step perturbation at t=10, exponential return to baseline
        perturbation_time = 10.0
        def make_response(tau_val: float, perturb_t: float):
            def response(t: float) -> float:
                if t < perturb_t:
                    return 0.5  # baseline
                else:
                    # Exponential decay to baseline after perturbation
                    return 0.5 + 0.4 * math.exp(-(t - perturb_t) / tau_val)
            return response

        adapt = check_perfect_adaptation(
            system_response=make_response(tau, perturbation_time),
            perturbation_time=perturbation_time,
            measurement_window=100.0,
            baseline_threshold=0.05,
        )
        is_adapted = adapt.adaptation_index > 0.9
        adaptation_tests.append({
            "tau": tau,
            "adapted": is_adapted,
            "adaptation_index": adapt.adaptation_index,
            "steady_state_error": adapt.steady_state_error,
            "recovery_time": adapt.recovery_time,
        })
        if verbose:
            print(f"    Perfect adaptation (τ={tau}s): {is_adapted}, "
                  f"adaptation_index={adapt.adaptation_index:.3f}")

    n_adapted = sum(1 for t in adaptation_tests if t["adapted"])
    avg_index = sum(t["adaptation_index"] for t in adaptation_tests) / len(adaptation_tests)
    details["perfect_adaptation"] = {
        "tests": adaptation_tests,
        "n_adapted": n_adapted,
        "n_total": len(adaptation_tests),
        "avg_adaptation_index": avg_index,
    }
    findings.append(
        f"Perfect adaptation: {n_adapted}/{len(adaptation_tests)} pass "
        f"(avg_index={avg_index:.3f}, integral feedback)"
    )

    # 1b. Compare agent to worm (navigation behavior)
    comparison = compare_agent_to_worm(
        behavior_type="chemotaxis",
        environment_params={"gradient_type": "linear", "noise": 0.1},
    )
    if verbose:
        print(comparison)

    details["agent_vs_worm"] = {
        "verdict": comparison.verdict.value,
        "conservation_score": comparison.conservation_score,
        "bio_efficiency": comparison.bio_efficiency,
        "ai_efficiency": comparison.ai_efficiency,
    }
    findings.append(
        f"Agent-vs-worm verdict={comparison.verdict.value}, "
        f"conservation={comparison.conservation_score:.3f}"
    )

    # 1c. DDM decision model (worm decides whether to turn)
    ddm = drift_diffusion_decision(
        drift_rate=0.3, noise_std=1.0, threshold=1.5,
        max_time=5.0, dt=0.01, seed=42,
    )
    if verbose:
        print(ddm)

    details["ddm"] = {
        "decision": ddm.decision,
        "decision_time": ddm.decision_time,
        "confidence": ddm.confidence,
    }
    findings.append(
        f"DDM decision='{ddm.decision}' in {ddm.decision_time:.3f}s, "
        f"confidence={ddm.confidence:.3f}"
    )

    # 1d. Architecture mapping
    arch = map_behavior_to_architecture(
        biological_system="e. coli",
        behavior_requirements=["gradient_following", "adaptation", "exploration"],
    )
    if verbose:
        print(arch)

    details["architecture"] = {
        "suggested": arch.ai_architecture,
        "confidence": arch.confidence,
    }
    findings.append(f"Suggested architecture: {arch.ai_architecture} "
                    f"(confidence={arch.confidence:.2f})")

    # Score: average of gradient following, adaptation success, and conservation
    # Normalize gradient_following_score from [-1,1] to [0,1]
    grad_norm = (traj.gradient_following_score + 1.0) / 2.0
    adapt_score = n_adapted / len(adaptation_tests) if adaptation_tests else 0.0
    score = (grad_norm + adapt_score + comparison.conservation_score) / 3.0
    verdict = "CONVERGENT" if score > 0.6 else "PARTIAL" if score > 0.3 else "DIVERGENT"

    return ScenarioResult(
        name="navigation_vs_chemotaxis",
        description="AI navigation agent compared to E. coli chemotaxis",
        tools_used=["simulate_chemotaxis", "check_perfect_adaptation",
                     "compare_agent_to_worm", "drift_diffusion_decision",
                     "map_behavior_to_architecture"],
        convergence_score=score,
        verdict=verdict,
        findings=findings,
        details=details,
    )


# ---------------------------------------------------------------------------
# Scenario 2: RL agent vs dopamine reward system
# ---------------------------------------------------------------------------

def scenario_reward_learning(verbose: bool = False) -> ScenarioResult:
    """Compare RL TD learning to dopamine reward prediction error."""
    print("\n  [Scenario 2] RL Agent vs Dopamine Reward System")
    findings: List[str] = []
    details: Dict = {}

    # 2a. Hebbian vs backprop comparison
    # Simulate a simple pattern association task
    pre = [0.8, 0.2, 0.9, 0.1, 0.7]
    post = [0.9, 0.3, 0.8, 0.2, 0.6]
    weights = [0.5, 0.5, 0.5, 0.5, 0.5]
    targets = [1.0, 0.0, 1.0, 0.0, 1.0]

    hebb_bp = compare_hebbian_backprop(
        pre_activities=pre,
        post_activities=post,
        weights=weights,
        target_outputs=targets,
        learning_rate_hebb=0.01,
        learning_rate_bp=0.01,
    )
    if verbose:
        print(hebb_bp)

    details["hebbian_vs_backprop"] = {
        "consistency_score": hebb_bp.consistency_score,
        "weight_update_correlation": hebb_bp.weight_update_correlation,
        "locality_preserved": hebb_bp.locality_preserved,
        "energy_efficiency_ratio": hebb_bp.energy_efficiency_ratio,
    }
    findings.append(
        f"Hebbian-backprop consistency={hebb_bp.consistency_score:.3f}, "
        f"weight_corr={hebb_bp.weight_update_correlation:.3f}"
    )

    # 2b. Striatum to actor-critic mapping
    ac_mappings = map_striatum_to_actor_critic()
    if verbose:
        for m in ac_mappings[:3]:
            print(m)

    avg_confidence = (sum(m.mapping_confidence for m in ac_mappings)
                      / len(ac_mappings)) if ac_mappings else 0.0
    details["striatum_actor_critic"] = {
        "n_mappings": len(ac_mappings),
        "avg_confidence": avg_confidence,
        "regions": [m.striatal_region for m in ac_mappings],
    }
    findings.append(
        f"Striatum-AC mappings: {len(ac_mappings)} regions, "
        f"avg_confidence={avg_confidence:.3f}"
    )

    # 2c. Agent-vs-worm comparison for learning behavior
    comparison = compare_agent_to_worm(
        behavior_type="learning",
        environment_params={"reward_structure": "delayed"},
    )
    if verbose:
        print(comparison)

    details["agent_vs_worm_learning"] = {
        "verdict": comparison.verdict.value,
        "conservation_score": comparison.conservation_score,
    }
    findings.append(
        f"Learning comparison: verdict={comparison.verdict.value}, "
        f"conservation={comparison.conservation_score:.3f}"
    )

    # Score: combine Hebbian-backprop consistency, AC confidence, conservation
    score = (hebb_bp.consistency_score + avg_confidence
             + comparison.conservation_score) / 3.0
    verdict = "CONVERGENT" if score > 0.6 else "PARTIAL" if score > 0.3 else "DIVERGENT"

    return ScenarioResult(
        name="rl_vs_dopamine",
        description="TD learning compared to dopamine RPE in basal ganglia",
        tools_used=["compare_hebbian_backprop", "map_striatum_to_actor_critic",
                     "compare_agent_to_worm"],
        convergence_score=score,
        verdict=verdict,
        findings=findings,
        details=details,
    )


# ---------------------------------------------------------------------------
# Scenario 3: Multi-agent coordination vs swarm consensus
# ---------------------------------------------------------------------------

def scenario_swarm(verbose: bool = False) -> ScenarioResult:
    """Compare multi-agent AI coordination to biological swarm behavior."""
    print("\n  [Scenario 3] Multi-Agent Coordination vs Swarm Consensus")
    findings: List[str] = []
    details: Dict = {}

    # 3a. Swarm consensus with 6 agents on a connected graph
    opinions = [0.1, 0.9, 0.3, 0.7, 0.5, 0.2]
    # Ring + cross links (connected graph)
    adj = [
        [0, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 0, 1],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 1, 0],
    ]
    consensus = swarm_consensus(
        initial_opinions=opinions,
        adjacency=adj,
        iterations=200,
        convergence_threshold=0.001,
    )
    if verbose:
        for k, v in consensus.items():
            if k != "history":
                print(f"    {k}: {v}")

    details["swarm_consensus"] = {
        "consensus_reached": consensus.get("consensus_reached", False),
        "final_opinions": consensus.get("final_opinions", []),
        "iterations": consensus.get("iterations", 0),
        "consensus_value": consensus.get("consensus_value", None),
        "final_variance": consensus.get("final_variance", None),
    }
    converged = consensus.get("consensus_reached", False)
    iters = consensus.get("iterations", 0)
    findings.append(
        f"Swarm consensus: converged={converged} in {iters} iterations"
    )

    # 3b. Compare agent to worm for collective behavior
    comparison = compare_agent_to_worm(
        behavior_type="collective",
        environment_params={"n_agents": 6, "task": "consensus"},
    )
    if verbose:
        print(comparison)

    details["agent_vs_worm_collective"] = {
        "verdict": comparison.verdict.value,
        "conservation_score": comparison.conservation_score,
    }
    findings.append(
        f"Collective comparison: verdict={comparison.verdict.value}, "
        f"conservation={comparison.conservation_score:.3f}"
    )

    # 3c. Architecture mapping for swarm systems
    arch = map_behavior_to_architecture(
        biological_system="ant colony",
        behavior_requirements=["stigmergy", "consensus", "task_allocation"],
    )
    if verbose:
        print(arch)

    details["architecture"] = {
        "suggested": arch.ai_architecture,
        "confidence": arch.confidence,
    }
    findings.append(f"Suggested architecture: {arch.ai_architecture} "
                    f"(confidence={arch.confidence:.2f})")

    # Score: consensus convergence + conservation + architecture confidence
    conv_score = 1.0 if converged else 0.3
    score = (conv_score + comparison.conservation_score + arch.confidence) / 3.0
    verdict = "CONVERGENT" if score > 0.6 else "PARTIAL" if score > 0.3 else "DIVERGENT"

    return ScenarioResult(
        name="multiagent_vs_swarm",
        description="Distributed AI consensus compared to swarm intelligence",
        tools_used=["swarm_consensus", "compare_agent_to_worm",
                     "map_behavior_to_architecture"],
        convergence_score=score,
        verdict=verdict,
        findings=findings,
        details=details,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(scenarios: List[ScenarioResult],
                 convergent_solutions: List[Dict]):
    """Print human-readable convergence report."""
    print("\n" + "=" * 72)
    print("  BIO-AI CONVERGENCE LAB -- Results Report")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)

    for s in scenarios:
        tag = {"CONVERGENT": "[CONVERGENT]",
               "PARTIAL": "[PARTIAL]",
               "DIVERGENT": "[DIVERGENT]"}[s.verdict]
        print(f"\n  {s.name:30s}  Score: {s.convergence_score:.3f}  {tag}")
        print(f"    {s.description}")
        print(f"    Tools: {', '.join(s.tools_used)}")
        for f in s.findings:
            print(f"    - {f}")

    # Known convergent solutions summary
    print(f"\n  {'=' * 72}")
    print(f"  Known Convergent Solutions (from identify_convergent_solutions):")
    print(f"  {'=' * 72}")
    for cs in convergent_solutions[:6]:
        print(f"    [{cs['conservation_score']:.2f}] {cs['domain']}: "
              f"{cs['biological']} <-> {cs['algorithmic']}")

    # Overall summary
    avg_score = (sum(s.convergence_score for s in scenarios)
                 / len(scenarios)) if scenarios else 0.0
    n_conv = sum(1 for s in scenarios if s.verdict == "CONVERGENT")
    n_part = sum(1 for s in scenarios if s.verdict == "PARTIAL")
    n_div = sum(1 for s in scenarios if s.verdict == "DIVERGENT")

    print(f"\n  {'=' * 72}")
    print(f"  Overall: {n_conv} CONVERGENT / {n_part} PARTIAL / {n_div} DIVERGENT")
    print(f"  Mean convergence score: {avg_score:.3f}")
    print(f"  {'=' * 72}\n")


def save_results(scenarios: List[ScenarioResult],
                 convergent_solutions: List[Dict],
                 outpath: Path):
    """Save results to JSON."""
    avg_score = (sum(s.convergence_score for s in scenarios)
                 / len(scenarios)) if scenarios else 0.0
    data = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "bio_ai_lab v0.1",
        "n_scenarios": len(scenarios),
        "mean_convergence_score": avg_score,
        "scenarios": [asdict(s) for s in scenarios],
        "known_convergent_solutions": convergent_solutions,
        "summary": {
            "convergent": sum(1 for s in scenarios if s.verdict == "CONVERGENT"),
            "partial": sum(1 for s in scenarios if s.verdict == "PARTIAL"),
            "divergent": sum(1 for s in scenarios if s.verdict == "DIVERGENT"),
        },
    }
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Results saved to {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Bio-AI Convergence Lab -- find convergent solutions")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed tool outputs")
    args = parser.parse_args()

    print("=" * 72)
    print("  Bio-AI Convergence Lab")
    print("  Chaining: chemotaxis + perfect_adaptation + DDM + Hebbian/backprop")
    print("           + striatum-AC + swarm")
    print("=" * 72)

    # Run all scenarios
    scenarios: List[ScenarioResult] = []
    for runner in [scenario_navigation, scenario_reward_learning, scenario_swarm]:
        try:
            result = runner(verbose=args.verbose)
            scenarios.append(result)
            print(f"    -> {result.verdict} (score={result.convergence_score:.3f})")
        except Exception as e:
            print(f"  ERROR in {runner.__name__}: {e}")

    # Gather known convergent solutions
    convergent_solutions = identify_convergent_solutions()

    # Report
    print_report(scenarios, convergent_solutions)

    outpath = RESULTS_DIR / "convergence_results.json"
    save_results(scenarios, convergent_solutions, outpath)


if __name__ == "__main__":
    main()
