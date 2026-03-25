#!/usr/bin/env python3
"""
Adapter Discovery Probe — Find what the adapter learned beyond its training set.

Core idea: if a domain adapter flips 16 trained facts, what ELSE shifted?
Generate new facts in the same domain (never in training), score on base vs adapted.
Facts where base fails but adapted passes = "adapter-discovered knowledge."

This reveals the transfer radius of each adapter in knowledge space.

Phase 2 (triangulation): score the same new facts with adapters from DIFFERENT
domains. If two independent adapters both shift toward the same fact, that's
convergent evidence from different regions of weight space.

Usage:
    python experiments/adapter_discovery_probe.py
    python experiments/adapter_discovery_probe.py --triangulate
"""

import argparse
import json
import os
import sys
import time

# Set HF_HOME
if not os.environ.get("HF_HOME") and os.path.isdir("/Volumes/4TB SD/ml_cache/huggingface"):
    os.environ["HF_HOME"] = "/Volumes/4TB SD/ml_cache/huggingface"

import mlx.core as mx
import mlx_lm
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from noethersolve.oracle import score_fact_mc

# ──────────────────────────────────────────────────────────────
# NEW probe facts — Hamiltonian domain, NOT in any training set
# These cover adjacent concepts, deeper results, edge cases
# ──────────────────────────────────────────────────────────────

HAMILTONIAN_PROBE_FACTS = [
    {
        "id": "probe01_arnold_diffusion",
        "context": "Arnold diffusion in Hamiltonian systems with n >= 3 degrees of freedom describes:",
        "truth": "slow drift along resonance layers that allows orbits to explore large regions of phase space",
        "distractors": [
            "rapid exponential divergence of nearby trajectories measured by positive Lyapunov exponents",
            "contraction of phase space volume toward a lower-dimensional strange attractor manifold",
            "quantum tunneling between classically forbidden regions of the configuration space"
        ]
    },
    {
        "id": "probe02_moser_twist",
        "context": "The Moser twist theorem strengthens KAM by showing that for area-preserving twist maps:",
        "truth": "invariant circles survive perturbation if the rotation number is sufficiently irrational",
        "distractors": [
            "all periodic orbits become unstable under arbitrarily small perturbation strength",
            "the twist condition guarantees global integrability for all energy surfaces",
            "invariant circles survive only if the perturbation preserves time-reversal symmetry"
        ]
    },
    {
        "id": "probe03_nekhoroshev",
        "context": "Nekhoroshev's theorem for steep Hamiltonian systems provides:",
        "truth": "exponentially long stability times: action variables change by at most epsilon^a for times up to exp(1/epsilon^b)",
        "distractors": [
            "polynomial stability bounds: action variables change by at most epsilon*t^2 for all finite times",
            "permanent stability: action variables are exactly conserved for all time under steep conditions",
            "logarithmic stability times: action variables drift by at most ln(1/epsilon) over times up to 1/epsilon"
        ]
    },
    {
        "id": "probe04_weinstein_conjecture",
        "context": "The Weinstein conjecture (proved by Taubes for 3-manifolds) states that:",
        "truth": "every compact contact manifold carries at least one closed Reeb orbit",
        "distractors": [
            "every symplectic manifold admits a global Hamiltonian function with no critical points",
            "every closed symplectic manifold has vanishing odd-dimensional cohomology groups",
            "every compact Hamiltonian system has at least one stable equilibrium point"
        ]
    },
    {
        "id": "probe05_arnolds_conjecture",
        "context": "Arnold's conjecture on fixed points of Hamiltonian diffeomorphisms states:",
        "truth": "the number of fixed points is at least the sum of Betti numbers of the manifold",
        "distractors": [
            "the number of fixed points equals exactly twice the dimension of the manifold",
            "every Hamiltonian diffeomorphism has at most finitely many fixed points on compact manifolds",
            "the number of periodic orbits grows exponentially with the topological complexity of the manifold"
        ]
    },
    {
        "id": "probe06_magnetic_moment",
        "context": "For a charged particle spiraling in a slowly varying magnetic field, the adiabatic invariant mu = mv_perp^2 / (2B) is:",
        "truth": "approximately conserved, leading to magnetic mirror reflection when B increases",
        "distractors": [
            "exactly conserved at all rates of field variation including rapid changes",
            "conserved only in uniform fields and violated in any spatially varying configuration",
            "the ratio of parallel to perpendicular kinetic energy in the guiding center frame"
        ]
    },
    {
        "id": "probe07_toda_lattice",
        "context": "The Toda lattice with potential V = exp(q_i - q_{i+1}) is significant because:",
        "truth": "it is completely integrable with n independent conserved quantities found via Lax pair representation",
        "distractors": [
            "it exhibits the transition to chaos at a critical energy threshold as a prototypical nonlinear system",
            "it conserves only energy and momentum like a generic nonlinear lattice Hamiltonian system",
            "it produces exact soliton solutions only in the continuum limit as lattice spacing approaches zero"
        ]
    },
    {
        "id": "probe08_cotangent_lift",
        "context": "The cotangent lift of a diffeomorphism phi: Q -> Q to T*Q is:",
        "truth": "automatically a symplectomorphism that preserves the canonical symplectic form",
        "distractors": [
            "a Riemannian isometry that preserves the Sasaki metric on the cotangent bundle",
            "symplectic only when the original diffeomorphism preserves a volume form on Q",
            "a contact transformation that preserves the tautological one-form up to a conformal factor"
        ]
    },
    {
        "id": "probe09_maslov_index",
        "context": "The Maslov index of a Lagrangian submanifold loop counts:",
        "truth": "the number of caustic crossings, giving a topological invariant related to quantum phase corrections",
        "distractors": [
            "the winding number of the energy surface around equilibrium points in phase space",
            "the number of periodic orbits enclosed by the Lagrangian loop in configuration space",
            "the Euler characteristic of the region bounded by the loop in the symplectic manifold"
        ]
    },
    {
        "id": "probe10_bertrand_theorem",
        "context": "Bertrand's theorem states that the only central force potentials giving closed orbits for all bound states are:",
        "truth": "the 1/r (Kepler/gravity) and r^2 (harmonic oscillator) potentials — exactly two",
        "distractors": [
            "the 1/r, r^2, and 1/r^2 potentials — exactly three possible closed-orbit force laws",
            "any power law V(r) = r^n with n > -2 produces closed orbits for all bounded trajectories",
            "only the 1/r potential gives closed orbits; harmonic oscillator orbits precess by 2pi/3"
        ]
    },
    {
        "id": "probe11_calogero_moser",
        "context": "The Calogero-Moser system of n particles with inverse-square pairwise interaction is:",
        "truth": "completely integrable, with solutions expressible via eigenvalues of a Lax matrix that evolves linearly",
        "distractors": [
            "chaotic for n >= 3 particles, similar to the gravitational three-body problem behavior",
            "integrable only for n = 2 particles and exhibits ergodic dynamics for all larger systems",
            "approximately integrable with KAM tori surviving for small coupling between particle pairs"
        ]
    },
    {
        "id": "probe12_gromov_nonsqueezing",
        "context": "Gromov's nonsqueezing theorem ('the symplectic camel') states:",
        "truth": "a symplectic map cannot squeeze a ball through a hole smaller than its projection onto any conjugate (q_i, p_i) plane",
        "distractors": [
            "a volume-preserving map can always deform any shape to pass through an arbitrarily small opening",
            "symplectic maps preserve the radius of the smallest enclosing sphere in phase space exactly",
            "Hamiltonian evolution cannot change the topology of any connected region in phase space"
        ]
    },
]

# ──────────────────────────────────────────────────────────────
# Probe facts from ADJACENT domain (for triangulation)
# These touch Hamiltonian concepts but from a different angle
# ──────────────────────────────────────────────────────────────

VORTEX_ADJACENT_PROBE_FACTS = [
    {
        "id": "vortex_probe01_kirchhoff",
        "context": "The Hamiltonian for N point vortices in 2D (Kirchhoff's formulation) is:",
        "truth": "H = -(1/4pi) * Sum_{i<j} Gamma_i * Gamma_j * ln(r_ij), with vortex positions as conjugate variables",
        "distractors": [
            "H = (1/2) * Sum_i Gamma_i * |v_i|^2, the kinetic energy of each vortex with velocity v_i",
            "H = Sum_{i<j} Gamma_i * Gamma_j / r_ij, the electrostatic-like Coulomb interaction energy",
            "H = Sum_i Gamma_i^2 * ln(r_i), the self-energy of each vortex at distance r_i from origin"
        ]
    },
    {
        "id": "vortex_probe02_symplectic_structure",
        "context": "Point vortex dynamics is Hamiltonian with the unusual feature that:",
        "truth": "positions (x_i, y_i) are canonically conjugate variables — configuration space IS phase space",
        "distractors": [
            "momenta are proportional to vortex velocities and must be tracked as separate state variables",
            "the symplectic structure only holds for equal-strength vortices in an unbounded domain",
            "vortex positions are Lagrangian variables requiring a Legendre transform to become Hamiltonian"
        ]
    },
    {
        "id": "vortex_probe03_3body_integrability",
        "context": "Three point vortices in 2D form a system that is:",
        "truth": "completely integrable — H, linear impulse (2 components), and angular impulse give 4 integrals for 4 effective degrees of freedom",
        "distractors": [
            "generically chaotic for unequal circulations, similar to the gravitational three-body problem",
            "integrable only when all three circulations are equal, with chaos onset for unequal strengths",
            "partially integrable with only H and angular impulse conserved, leaving 2 chaotic degrees of freedom"
        ]
    },
]


def load_model_and_adapter(adapter_path=None, d_inner=64):
    """Load Qwen3-14B-Base and optionally an adapter."""
    print("Loading Qwen3-14B-Base...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-14B-Base")
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    adapter = None
    lm_head = None
    if adapter_path and os.path.exists(adapter_path):
        from noethersolve.adapter import SnapOnConfig, create_adapter
        from noethersolve.train_utils import get_lm_head_fn

        vocab_size = model.model.embed_tokens.weight.shape[0]
        d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
        cfg = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                          n_heads=8, mode="logit", vocab_size=vocab_size)
        adapter = create_adapter(cfg)
        weights = mx.load(adapter_path)
        adapter.load_weights(list(weights.items()))
        mx.eval(adapter.parameters())
        lm_head = get_lm_head_fn(model)
        print(f"  Adapter loaded: {os.path.basename(adapter_path)}")

    return model, tokenizer, adapter, lm_head


def score_facts(model, tokenizer, facts, adapter=None, lm_head=None):
    """Score a list of facts, return per-fact results."""
    results = []
    for fact in facts:
        win, margin, truth_lp, best_dist_lp = score_fact_mc(
            model, tokenizer,
            fact["context"], fact["truth"], fact["distractors"],
            adapter=adapter, lm_head=lm_head,
        )
        results.append({
            "id": fact["id"],
            "context": fact["context"],
            "truth": fact["truth"],
            "win": bool(win),
            "margin": float(margin),
        })
    return results


def compare_base_vs_adapted(base_results, adapted_results):
    """Find facts where adapter flipped the answer."""
    discoveries = []
    for br, ar in zip(base_results, adapted_results):
        delta = ar["margin"] - br["margin"]
        status = "UNCHANGED"
        if not br["win"] and ar["win"]:
            status = "DISCOVERED"  # base wrong, adapter right
        elif br["win"] and not ar["win"]:
            status = "REGRESSED"   # base right, adapter wrong
        elif not br["win"] and not ar["win"]:
            status = "BOTH_FAIL"
        elif br["win"] and ar["win"]:
            status = "BOTH_PASS"

        discoveries.append({
            "id": br["id"],
            "truth": br["truth"][:70],
            "base_margin": br["margin"],
            "adapted_margin": ar["margin"],
            "delta": delta,
            "status": status,
        })
    return discoveries


def print_report(title, discoveries):
    """Print discovery report."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    discovered = [d for d in discoveries if d["status"] == "DISCOVERED"]
    regressed = [d for d in discoveries if d["status"] == "REGRESSED"]
    both_pass = [d for d in discoveries if d["status"] == "BOTH_PASS"]
    both_fail = [d for d in discoveries if d["status"] == "BOTH_FAIL"]

    print(f"\n  DISCOVERED (base wrong → adapter right): {len(discovered)}")
    for d in sorted(discovered, key=lambda x: -x["delta"]):
        print(f"    [{d['id']}] base={d['base_margin']:+.2f} → adapted={d['adapted_margin']:+.2f} (Δ={d['delta']:+.2f})")
        print(f"      {d['truth']}")

    print(f"\n  BOTH PASS (already known): {len(both_pass)}")
    for d in both_pass:
        print(f"    [{d['id']}] base={d['base_margin']:+.2f}, adapted={d['adapted_margin']:+.2f}")

    print(f"\n  BOTH FAIL (not learned): {len(both_fail)}")
    for d in both_fail:
        print(f"    [{d['id']}] base={d['base_margin']:+.2f}, adapted={d['adapted_margin']:+.2f}")

    if regressed:
        print(f"\n  REGRESSED (base right → adapter wrong): {len(regressed)}")
        for d in regressed:
            print(f"    [{d['id']}] base={d['base_margin']:+.2f} → adapted={d['adapted_margin']:+.2f}")

    mean_delta = np.mean([d["delta"] for d in discoveries])
    print(f"\n  Mean margin delta: {mean_delta:+.3f}")
    print(f"  Total: {len(discovered)} discovered, {len(both_pass)} known, {len(both_fail)} unknown, {len(regressed)} regressed")
    return discoveries


def main():
    parser = argparse.ArgumentParser(description="Adapter Discovery Probe")
    parser.add_argument("--triangulate", action="store_true",
                       help="Also score with vortex adapter for triangulation")
    parser.add_argument("--adapter", default=None,
                       help="Path to Hamiltonian adapter (default: auto-detect)")
    parser.add_argument("--vortex-adapter", default=None,
                       help="Path to vortex adapter for triangulation")
    parser.add_argument("--output", default=None,
                       help="Save results to JSON")
    args = parser.parse_args()

    # Auto-detect adapters
    adapter_dir = os.path.join(ROOT, "adapters", "qwen3_4b_base")
    ham_adapter = args.adapter or os.path.join(adapter_dir, "hamiltonian_stage5.npz")
    if not os.path.exists(ham_adapter):
        # Try other adapter names
        for name in ["hamiltonian_adapter.npz", "hamiltonian_mechanics_invariants_intensive_adapter.npz"]:
            p = os.path.join(adapter_dir, name)
            if os.path.exists(p):
                ham_adapter = p
                break

    print(f"Hamiltonian adapter: {os.path.basename(ham_adapter)}")

    # Phase 1: Score new facts with base model
    print("\n" + "─"*70)
    print("Phase 1: Scoring probe facts on BASE model (no adapter)")
    print("─"*70)

    model, tokenizer, _, _ = load_model_and_adapter()
    base_results = score_facts(model, tokenizer, HAMILTONIAN_PROBE_FACTS)

    # Phase 2: Score same facts with Hamiltonian adapter
    print("\n" + "─"*70)
    print("Phase 2: Scoring probe facts with HAMILTONIAN adapter")
    print("─"*70)

    _, _, ham_adpt, ham_lm = load_model_and_adapter(ham_adapter)
    ham_results = score_facts(model, tokenizer, HAMILTONIAN_PROBE_FACTS,
                             adapter=ham_adpt, lm_head=ham_lm)

    ham_discoveries = print_report(
        "Hamiltonian Adapter Transfer — New Facts Never In Training",
        compare_base_vs_adapted(base_results, ham_results)
    )

    all_results = {"hamiltonian_probe": ham_discoveries}

    # Phase 3: Triangulation (if requested)
    if args.triangulate:
        vortex_adapter = args.vortex_adapter
        if not vortex_adapter:
            for name in ["vortex_conservation_continuous_Q_f_adapter.npz",
                         "vortex_adapter.npz"]:
                p = os.path.join(adapter_dir, name)
                if os.path.exists(p):
                    vortex_adapter = p
                    break

        if vortex_adapter and os.path.exists(vortex_adapter):
            print(f"\nVortex adapter: {os.path.basename(vortex_adapter)}")

            # Score Hamiltonian probes with vortex adapter
            print("\n" + "─"*70)
            print("Phase 3a: Hamiltonian probe facts with VORTEX adapter (cross-domain)")
            print("─"*70)

            _, _, vort_adpt, vort_lm = load_model_and_adapter(vortex_adapter)
            vort_on_ham = score_facts(model, tokenizer, HAMILTONIAN_PROBE_FACTS,
                                     adapter=vort_adpt, lm_head=vort_lm)
            vort_ham_disc = print_report(
                "Vortex Adapter on Hamiltonian Facts — Cross-Domain Transfer",
                compare_base_vs_adapted(base_results, vort_on_ham)
            )
            all_results["vortex_on_hamiltonian"] = vort_ham_disc

            # Score vortex-adjacent probes with both adapters
            print("\n" + "─"*70)
            print("Phase 3b: Vortex-adjacent probe facts with BOTH adapters")
            print("─"*70)

            vort_base = score_facts(model, tokenizer, VORTEX_ADJACENT_PROBE_FACTS)
            vort_ham = score_facts(model, tokenizer, VORTEX_ADJACENT_PROBE_FACTS,
                                  adapter=ham_adpt, lm_head=ham_lm)
            vort_vort = score_facts(model, tokenizer, VORTEX_ADJACENT_PROBE_FACTS,
                                   adapter=vort_adpt, lm_head=vort_lm)

            print("\n  TRIANGULATION — Facts where BOTH adapters agree:")
            print(f"  {'ID':<30} {'Base':>8} {'Ham':>8} {'Vort':>8} {'Signal'}")
            print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*15}")

            for b, h, v in zip(vort_base, vort_ham, vort_vort):
                ham_delta = h["margin"] - b["margin"]
                vort_delta = v["margin"] - b["margin"]
                # Both adapters shift in the same direction AND improve
                if ham_delta > 0 and vort_delta > 0:
                    signal = "CONVERGENT ✓"
                elif ham_delta > 0 or vort_delta > 0:
                    signal = "one-sided"
                else:
                    signal = "neither"

                print(f"  {b['id']:<30} {b['margin']:+8.2f} {h['margin']:+8.2f} {v['margin']:+8.2f} {signal}")

            all_results["vortex_adjacent_triangulation"] = {
                "base": [{"id": r["id"], "margin": r["margin"], "win": r["win"]} for r in vort_base],
                "hamiltonian_adapter": [{"id": r["id"], "margin": r["margin"], "win": r["win"]} for r in vort_ham],
                "vortex_adapter": [{"id": r["id"], "margin": r["margin"], "win": r["win"]} for r in vort_vort],
            }
        else:
            print("\n  No vortex adapter found — skipping triangulation.")

    # Save results
    output_path = args.output or os.path.join(ROOT, "results", "adapter_discovery_probe.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
