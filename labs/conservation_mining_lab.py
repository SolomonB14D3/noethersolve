#!/usr/bin/env python3
"""
conservation_mining_lab.py — Discover approximate conservation laws in
dynamical systems by chaining NoetherSolve tools.

Chains: known-invariant verification -> invariant discovery (L-BFGS-B) ->
Lyapunov analysis -> ergodic classification -> novelty ranking.

Usage:
    python labs/conservation_mining_lab.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp

# Ensure noethersolve is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from noethersolve.monitor import VortexMonitor, frac_var
from noethersolve.hamiltonian import (
    HamiltonianMonitor, harmonic_oscillator, kepler_2d,
)
from noethersolve.learner import InvariantLearner
from noethersolve.ergodic_theory import (
    lyapunov_analysis, classify_system, poincare_recurrence,
    entropy_analysis,
)

RESULTS_DIR = _ROOT / "results" / "labs" / "conservation_mining"
FRAC_VAR_THRESHOLD = 5e-3  # PASS threshold for approximate invariants


# ---------------------------------------------------------------------------
# Physical systems
# ---------------------------------------------------------------------------

def make_vortex_trajectory(circulations, positions_0, T=20.0, n_steps=500):
    """Integrate 2D point-vortex system and return position trajectory.

    Returns array of shape (n_steps, N, 2).
    """
    G = np.asarray(circulations, dtype=np.float64)
    N = len(G)
    pos0 = np.asarray(positions_0, dtype=np.float64).reshape(N, 2)
    y0 = pos0.flatten()

    def rhs(t, y):
        pos = y.reshape(N, 2)
        dy = np.zeros_like(pos)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dx_ = pos[i, 0] - pos[j, 0]
                dy_ = pos[i, 1] - pos[j, 1]
                r2 = dx_**2 + dy_**2 + 1e-20
                dy[i, 0] += -G[j] * dy_ / (2 * np.pi * r2)
                dy[i, 1] += G[j] * dx_ / (2 * np.pi * r2)
        return dy.flatten()

    t_eval = np.linspace(0, T, n_steps)
    sol = solve_ivp(rhs, (0, T), y0, t_eval=t_eval, rtol=1e-10, atol=1e-12)
    if sol.status != 0:
        raise RuntimeError(f"Vortex integration failed: {sol.message}")
    traj = sol.y.T.reshape(n_steps, N, 2)
    return traj


@dataclass
class SystemSpec:
    name: str
    kind: str  # "vortex" or "hamiltonian"
    description: str
    # Vortex-specific
    circulations: Optional[List[float]] = None
    positions_0: Optional[List[List[float]]] = None
    # Hamiltonian-specific
    monitor_factory: Optional[object] = None
    z0: Optional[List[float]] = None
    T: float = 20.0


SYSTEMS: List[SystemSpec] = [
    # --- Vortex systems ---
    SystemSpec(
        name="vortex_2_equal",
        kind="vortex",
        description="2 vortices, equal circulation (co-rotating)",
        circulations=[1.0, 1.0],
        positions_0=[[0.0, 0.5], [0.0, -0.5]],
        T=30.0,
    ),
    SystemSpec(
        name="vortex_2_opposite",
        kind="vortex",
        description="2 vortices, opposite circulation (dipole translation)",
        circulations=[1.0, -1.0],
        positions_0=[[0.0, 0.3], [0.0, -0.3]],
        T=30.0,
    ),
    SystemSpec(
        name="vortex_2_ratio",
        kind="vortex",
        description="2 vortices, 2:1 circulation ratio",
        circulations=[2.0, 1.0],
        positions_0=[[0.0, 0.5], [0.0, -0.5]],
        T=30.0,
    ),
    SystemSpec(
        name="vortex_3_mixed",
        kind="vortex",
        description="3 vortices, mixed circulations (figure-8 type)",
        circulations=[1.0, -0.5, 0.3],
        positions_0=[[0.5, 0.0], [-0.3, 0.4], [-0.3, -0.4]],
        T=20.0,
    ),
    # --- Hamiltonian systems ---
    SystemSpec(
        name="sho",
        kind="hamiltonian",
        description="Simple harmonic oscillator (omega=1)",
        monitor_factory=harmonic_oscillator,
        z0=[1.0, 0.0],
        T=50.0,
    ),
    SystemSpec(
        name="kepler",
        kind="hamiltonian",
        description="2D Kepler problem (elliptic orbit)",
        monitor_factory=kepler_2d,
        z0=[1.0, 0.0, 0.0, 0.8],
        T=50.0,
    ),
]


# ---------------------------------------------------------------------------
# Candidate invariant
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    system: str
    name: str
    formula: str
    frac_var: float
    classification: str   # "exact", "approximate", "artifact"
    known: bool           # True if it matches a known invariant
    novelty_score: float  # lower frac_var + not-known = higher novelty


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step1_known_invariants(spec: SystemSpec) -> Dict:
    """Verify known conservation laws for the system."""
    print(f"\n  [1] Known invariants for {spec.name}")
    result = {"system": spec.name, "quantities": {}}

    if spec.kind == "vortex":
        traj = make_vortex_trajectory(
            spec.circulations, spec.positions_0, T=spec.T
        )
        monitor = VortexMonitor(spec.circulations)
        monitor.set_initial(traj[0])
        for pos in traj[1:]:
            monitor.check(pos)
        summary = monitor.summary()
        for qname, qdata in summary.items():
            fv = qdata["frac_var"]
            status = "PASS" if fv < FRAC_VAR_THRESHOLD else "FAIL"
            result["quantities"][qname] = {
                "frac_var": fv, "status": status,
                "initial": qdata["initial"], "final": qdata["final"],
            }
            print(f"      {qname:20s}  frac_var={fv:.2e}  {status}")
        result["trajectory"] = traj

    elif spec.kind == "hamiltonian":
        monitor = spec.monitor_factory()
        z0 = np.array(spec.z0)
        report = monitor.validate(
            z0, T=spec.T, check_liouville=False, check_poincare=False,
        )
        for qname, qdata in report.quantities.items():
            fv = qdata["frac_var"]
            status = qdata["verdict"]
            result["quantities"][qname] = {
                "frac_var": fv, "status": status,
                "initial": qdata.get("initial"),
                "final": qdata.get("final"),
            }
            print(f"      {qname:20s}  frac_var={fv:.2e}  {status}")
        # Store the trajectory for discovery step
        sol = monitor._integrate(z0, spec.T)
        result["sol"] = sol

    return result


def step2_discover(spec: SystemSpec, step1_data: Dict) -> Optional[dict]:
    """Use InvariantLearner to find new conserved quantities."""
    if spec.kind != "vortex":
        print(f"  [2] Discovery: skipping (learner is for pairwise systems)")
        return None

    print(f"  [2] Discovering new invariants for {spec.name}")
    traj = step1_data["trajectory"]
    learner = InvariantLearner(maxiter=300)
    report = learner.learn_from_positions(
        [traj], spec.circulations,
    )
    print(f"      Optimal f(r) = {report.formula}")
    print(f"      Loss: {report.initial_loss:.6f} -> {report.final_loss:.6f}"
          f"  ({report.improvement_pct:.1f}% improvement)")
    print(f"      Dominant: {', '.join(report.dominant_terms)}")
    print(f"      Best single basis: {report.best_single_basis}")
    return {
        "formula": report.formula,
        "final_loss": report.final_loss,
        "improvement_pct": report.improvement_pct,
        "dominant_terms": report.dominant_terms,
        "best_single_basis": report.best_single_basis,
        "coefficients": report.coefficients,
        "individual_losses": report.individual_losses,
    }


def step3_lyapunov(spec: SystemSpec, step1_data: Dict) -> Optional[dict]:
    """Estimate Lyapunov exponents and classify chaos."""
    if spec.kind != "hamiltonian":
        print(f"  [3] Lyapunov: skipping (need Hamiltonian ODE)")
        return None

    print(f"  [3] Lyapunov analysis for {spec.name}")
    monitor = spec.monitor_factory()
    z0 = np.array(spec.z0)
    n = len(z0)

    # Compute largest Lyapunov exponent via tangent vector evolution
    delta = 1e-7
    T_lyap = min(spec.T, 30.0)
    sol0 = solve_ivp(monitor._eom, (0, T_lyap), z0,
                     rtol=1e-10, atol=1e-12, dense_output=True)
    # Perturbed trajectory along each axis, measure divergence
    exponents = []
    for axis in range(n):
        z_pert = z0.copy()
        z_pert[axis] += delta
        sol_p = solve_ivp(monitor._eom, (0, T_lyap), z_pert,
                          rtol=1e-10, atol=1e-12)
        if sol_p.status != 0:
            exponents.append(0.0)
            continue
        final_sep = np.linalg.norm(sol_p.y[:, -1] - sol0.y[:, -1])
        if final_sep > 0 and delta > 0:
            lam = np.log(final_sep / delta) / T_lyap
        else:
            lam = 0.0
        exponents.append(float(lam))

    exponents.sort(reverse=True)
    report = lyapunov_analysis(exponents)
    info = {
        "exponents": exponents,
        "lambda_max": exponents[0],
        "chaotic": exponents[0] > 0.01,
        "report_str": str(report),
    }
    print(f"      Exponents: {[f'{e:.4f}' for e in exponents]}")
    print(f"      Chaotic: {info['chaotic']}")
    return info


def step4_classify(spec: SystemSpec) -> Optional[dict]:
    """Classify the system in the ergodic hierarchy."""
    name_map = {
        "sho": "irrational_rotation",
        "kepler": "",
    }
    ergodic_name = name_map.get(spec.name, "")
    if not ergodic_name:
        print(f"  [4] Classification: no ergodic DB entry for {spec.name}")
        return None

    print(f"  [4] Ergodic classification for {spec.name}")
    report = classify_system(name=ergodic_name)
    info = {"name": ergodic_name, "report_str": str(report)}
    print(f"      {str(report)[:120]}")
    return info


def step5_dynamical_analysis(spec: SystemSpec, lyapunov_data: Optional[dict]) -> Optional[dict]:
    """Advanced dynamical analysis: recurrence, entropy, classification."""
    if spec.kind != "hamiltonian":
        print(f"  [5] Dynamical analysis: skipping (need Hamiltonian system)")
        return None

    print(f"  [5] Advanced dynamical analysis for {spec.name}")
    info = {}

    # Poincare recurrence (estimate from phase space volume)
    # For SHO: phase space is a circle of radius sqrt(2*E)
    # For Kepler: phase space is more complex
    if spec.name == "sho":
        # SHO: small set on energy shell
        E = 0.5  # H = (q^2 + p^2)/2 = 0.5 for z0 = [1, 0]
        set_measure = 0.01  # small neighborhood
        phase_volume = 2 * np.pi * E  # circumference of energy shell
        recurrence = poincare_recurrence(set_measure, phase_volume)
        info["poincare_recurrence"] = {
            "estimated_return_time": recurrence.estimated_return_time,
            "set_measure": set_measure,
            "phase_volume": phase_volume,
        }
        print(f"      Poincare recurrence: T ≈ {recurrence.estimated_return_time:.1f} (set/volume = {set_measure/phase_volume:.4f})")

    # Dynamical entropy (Kolmogorov-Sinai from Lyapunov exponents)
    if lyapunov_data and "exponents" in lyapunov_data:
        exps = lyapunov_data["exponents"]
        # KS entropy = sum of positive Lyapunov exponents (Pesin formula)
        positive_sum = sum(max(0, e) for e in exps)
        ks_report = entropy_analysis(
            ks_entropy=positive_sum,
            lyapunov_positive_sum=positive_sum,
        )
        interpretation = "chaotic" if ks_report.is_deterministic else "regular"
        info["ks_entropy"] = {
            "h_ks": ks_report.ks_entropy,
            "positive_exponents": [e for e in exps if e > 0],
            "is_chaotic": ks_report.is_deterministic,
        }
        print(f"      K-S entropy: h_KS = {ks_report.ks_entropy:.4f} ({interpretation})")

    # Full dynamical classification (reuse existing classify_system with system name)
    is_chaotic = lyapunov_data.get("chaotic", False) if lyapunov_data else False
    predictability = "chaotic (short horizon)" if is_chaotic else "integrable (long horizon)"
    info["dynamical_class"] = {
        "chaotic": is_chaotic,
        "predictability": predictability,
        "dimension": len(spec.z0) if spec.z0 else 4,
    }
    print(f"      Predictability: {predictability}")

    return info


# ---------------------------------------------------------------------------
# Candidate ranking
# ---------------------------------------------------------------------------

KNOWN_EXACT = {"H", "Lz", "Px", "Py", "energy", "angular_momentum",
               "LRL_magnitude"}

def build_candidates(spec: SystemSpec, step1_data: Dict,
                     discovery_data: Optional[dict]) -> List[Candidate]:
    """Build candidate list from all quantities found."""
    candidates = []
    for qname, qdata in step1_data["quantities"].items():
        fv = qdata["frac_var"]
        known = qname in KNOWN_EXACT
        if fv < 1e-8:
            classification = "exact"
        elif fv < FRAC_VAR_THRESHOLD:
            classification = "approximate"
        else:
            classification = "artifact"
        # Novelty: low frac_var, not already known
        novelty = (1.0 - min(fv * 1000, 1.0)) * (0.2 if known else 1.0)
        candidates.append(Candidate(
            system=spec.name, name=qname, formula=qname,
            frac_var=fv, classification=classification,
            known=known, novelty_score=novelty,
        ))

    if discovery_data:
        # The learned f(r) is a candidate itself
        fv = discovery_data["final_loss"]
        candidates.append(Candidate(
            system=spec.name,
            name="Q_learned",
            formula=discovery_data["formula"],
            frac_var=fv,
            classification="approximate" if fv < FRAC_VAR_THRESHOLD else "artifact",
            known=False,
            novelty_score=(1.0 - min(fv * 1000, 1.0)) * 1.0,
        ))
        # Also rank individual basis functions
        for bname, bloss in discovery_data["individual_losses"].items():
            if bloss < FRAC_VAR_THRESHOLD:
                candidates.append(Candidate(
                    system=spec.name,
                    name=f"Q_{bname}",
                    formula=f"sum Gi*Gj * {bname}(rij)",
                    frac_var=bloss,
                    classification="approximate",
                    known=(bname == "-ln(r)"),  # -ln(r) gives H
                    novelty_score=(1.0 - min(bloss * 1000, 1.0)) * (0.2 if bname == "-ln(r)" else 1.0),
                ))

    return candidates


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(all_candidates: List[Candidate],
                    system_results: Dict) -> dict:
    """Build the final discovery report."""
    # Rank by novelty
    ranked = sorted(all_candidates, key=lambda c: -c.novelty_score)

    novel_approx = [c for c in ranked
                    if c.classification == "approximate" and not c.known]
    exact_known = [c for c in ranked if c.classification == "exact"]
    artifacts = [c for c in ranked if c.classification == "artifact"]

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_systems": len(system_results),
        "n_candidates_total": len(all_candidates),
        "n_novel_approximate": len(novel_approx),
        "n_exact_known": len(exact_known),
        "n_artifacts": len(artifacts),
        "top_candidates": [asdict(c) for c in ranked[:20]],
        "per_system": {},
    }

    for sname, sdata in system_results.items():
        sys_cands = [c for c in ranked if c.system == sname]
        report["per_system"][sname] = {
            "description": sdata.get("description", ""),
            "n_candidates": len(sys_cands),
            "n_novel": len([c for c in sys_cands
                           if c.classification == "approximate" and not c.known]),
            "lyapunov": sdata.get("lyapunov"),
            "classification": sdata.get("classification"),
            "dynamical": sdata.get("dynamical"),
        }

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  Conservation Law Mining Lab")
    print("  Chaining: VortexMonitor + HamiltonianMonitor + InvariantLearner")
    print("           + Lyapunov + Ergodic + Poincare + K-S Entropy")
    print("=" * 65)

    all_candidates: List[Candidate] = []
    system_results: Dict = {}

    for spec in SYSTEMS:
        print(f"\n{'─' * 65}")
        print(f"  System: {spec.name} — {spec.description}")
        print(f"{'─' * 65}")

        # Step 1: verify known invariants
        step1_data = step1_known_invariants(spec)

        # Step 2: discover new invariants (vortex systems only)
        discovery_data = step2_discover(spec, step1_data)

        # Step 3: Lyapunov analysis (Hamiltonian systems only)
        lyapunov_data = step3_lyapunov(spec, step1_data)

        # Step 4: ergodic classification
        classification_data = step4_classify(spec)

        # Step 5: advanced dynamical analysis (recurrence, entropy)
        dynamical_data = step5_dynamical_analysis(spec, lyapunov_data)

        # Build candidates
        candidates = build_candidates(spec, step1_data, discovery_data)
        all_candidates.extend(candidates)

        system_results[spec.name] = {
            "description": spec.description,
            "discovery": {k: v for k, v in (discovery_data or {}).items()
                         if k != "coefficients"} if discovery_data else None,
            "lyapunov": lyapunov_data,
            "classification": classification_data,
            "dynamical": dynamical_data,
        }

    # Generate and print report
    report = generate_report(all_candidates, system_results)

    print(f"\n{'=' * 65}")
    print("  DISCOVERY REPORT")
    print(f"{'=' * 65}")
    print(f"  Systems tested:        {report['n_systems']}")
    print(f"  Total candidates:      {report['n_candidates_total']}")
    print(f"  Novel approximate:     {report['n_novel_approximate']}")
    print(f"  Exact (known):         {report['n_exact_known']}")
    print(f"  Artifacts (frac_var > {FRAC_VAR_THRESHOLD}): {report['n_artifacts']}")
    print()

    print("  Top candidates by novelty:")
    print(f"  {'Rank':>4}  {'System':15}  {'Name':20}  {'frac_var':>10}  {'Class':12}  {'Known':>5}  {'Score':>6}")
    print(f"  {'─'*4}  {'─'*15}  {'─'*20}  {'─'*10}  {'─'*12}  {'─'*5}  {'─'*6}")
    for i, c in enumerate(report["top_candidates"][:15], 1):
        print(f"  {i:4d}  {c['system']:15}  {c['name']:20}  {c['frac_var']:10.2e}"
              f"  {c['classification']:12}  {str(c['known']):>5}  {c['novelty_score']:6.3f}")

    print(f"\n{'=' * 65}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "discovery_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")
    print(f"{'=' * 65}")

    return report


if __name__ == "__main__":
    main()
