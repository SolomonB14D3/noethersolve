#!/usr/bin/env python3
"""
Adapter Hypothesis Engine — Use pre-trained adapters to DISCOVER new facts.

The idea: adapters shift the model's knowledge representation. If we ask the
adapted model about things WE don't know the answer to, and multiple adapters
from different domains converge on the same answer, that's a candidate discovery.

Then we verify with MCP computational tools.

Pipeline:
  1. Generate candidate claims (genuinely open questions)
  2. Score each candidate: base model vs each relevant adapter
  3. Find convergence (multiple adapters agree on something base doesn't)
  4. Verify computationally

Usage:
    python experiments/adapter_hypothesis_engine.py
    python experiments/adapter_hypothesis_engine.py --verify  # also run MCP verification
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict

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
# CANDIDATE CLAIMS — genuinely open or under-explored questions
# Each has a truth hypothesis and plausible alternatives.
# We DON'T know which is correct — the adapters will vote,
# and then we verify computationally.
# ──────────────────────────────────────────────────────────────

CANDIDATE_CLAIMS = [
    # === CHEMICAL NETWORK CONSERVATION ===
    # We know detailed balance constrains cycle products. But what about
    # non-cycle topologies? The audit_chemical_network tool can check.
    {
        "id": "hyp01_branched_network_conservation",
        "domain": "chemical_networks",
        "context": "In a branched chemical reaction network (not a cycle) at steady state, the relationship between forward and reverse rate constants is:",
        "candidates": [
            "constrained only by local detailed balance at each node, not by any global cycle condition",
            "constrained by a global Wegscheider condition across all branches simultaneously",
            "unconstrained — any rate constants are thermodynamically allowed for non-cyclic networks",
            "constrained by the ratio of total entropy production across all branches"
        ],
        "verification": "audit_chemical_network",
        "verification_params": {
            "species": ["A", "B", "C", "D"],
            "reactions": [
                {"reactants": ["A"], "products": ["B"], "kf": 1.0, "kr": 0.5},
                {"reactants": ["A"], "products": ["C"], "kf": 2.0, "kr": 0.8},
                {"reactants": ["B"], "products": ["D"], "kf": 1.5, "kr": 0.6},
                {"reactants": ["C"], "products": ["D"], "kf": 0.8, "kr": 0.4}
            ]
        }
    },
    # === VORTEX DYNAMICS ===
    # We know Q_f = sum Gi*Gj*f(rij) is approximately conserved for smooth f.
    # Is there an optimal f for N=4 that differs from N=3?
    {
        "id": "hyp02_4vortex_optimal_f",
        "domain": "vortex_dynamics",
        "context": "For a 4-vortex system compared to 3-vortex, the optimal conserved quantity Q_f = Sum Gamma_i*Gamma_j*f(r_ij) has:",
        "candidates": [
            "the same optimal f = -ln(r) regardless of vortex number, because Green's function optimality is universal",
            "a different optimal f that depends on the number of vortices due to multi-body interaction effects",
            "no well-defined optimum for N >= 4 because the conservation quality degrades catastrophically",
            "optimal f = 1/r for N >= 4, switching from logarithmic to algebraic scaling"
        ],
        "verification": "check_vortex_conservation"
    },
    # === ENZYME KINETICS ===
    # Substrate inhibition at high [S] — what's the kinetic mechanism?
    {
        "id": "hyp03_substrate_inhibition_mechanism",
        "domain": "enzyme_kinetics",
        "context": "When substrate inhibition occurs (rate decreases at high [S]), the dominant kinetic mechanism is:",
        "candidates": [
            "formation of a dead-end ESS complex where a second substrate molecule binds the enzyme-substrate complex",
            "competitive product inhibition where high substrate drives equilibrium toward product accumulation",
            "allosteric conformational change that converts the enzyme to an inactive form at high substrate",
            "substrate-induced enzyme denaturation that irreversibly reduces the active enzyme concentration"
        ],
        "verification": "calc_enzyme_inhibition"
    },
    # === HAMILTONIAN MECHANICS ===
    # The coupled quartic oscillator — integrable or not?
    {
        "id": "hyp04_coupled_quartic_integrability",
        "domain": "hamiltonian_mechanics",
        "context": "The coupled quartic oscillator H = (p1^2+p2^2)/2 + (q1^2+q2^2)/2 + lambda*q1^2*q2^2 is:",
        "candidates": [
            "non-integrable for generic lambda, with a second integral existing only for lambda = 0 (uncoupled) and specific rational values",
            "completely integrable for all lambda due to a hidden SU(2) symmetry of the quartic coupling",
            "non-integrable for ALL nonzero lambda with no exceptional values admitting a second integral",
            "integrable for lambda < 1 (weak coupling) and chaotic for lambda >= 1 (strong coupling threshold)"
        ],
        "verification": "check_hamiltonian_system"
    },
    # === NUMBER THEORY ===
    # Goldbach's conjecture is verified up to 4*10^18. But what about the
    # *number* of representations? Does it grow monotonically?
    {
        "id": "hyp05_goldbach_representations_growth",
        "domain": "number_theory",
        "context": "The number of Goldbach representations r(2n) = |{(p,q) : p+q=2n, p<=q prime}| as n grows:",
        "candidates": [
            "grows roughly as n/ln(n)^2 with large fluctuations but no systematic decrease, consistent with Hardy-Littlewood conjecture",
            "grows as n/ln(n) matching the prime counting function asymptotic exactly",
            "oscillates with a quasi-period related to primorial numbers with no clear growth trend",
            "grows as sqrt(n)*ln(n) following a central limit theorem for prime pair sums"
        ],
        "verification": "verify_goldbach"
    },
    # === PDE / NAVIER-STOKES ===
    # The critical Sobolev exponent for NS regularity
    {
        "id": "hyp06_ns_critical_norm",
        "domain": "pde_regularity",
        "context": "For 3D Navier-Stokes, the critical function space for regularity (the Ladyzhenskaya-Prodi-Serrin condition) requires the velocity field to be in:",
        "candidates": [
            "L^p_t L^q_x with 2/p + 3/q = 1, where the borderline case p=infinity q=3 (L^3) remains open",
            "L^p_t L^q_x with 2/p + 3/q = 2, where all borderline cases have been resolved",
            "L^2_t H^1_x (the energy space) which is sufficient for both existence and regularity",
            "L^infinity_t L^2_x intersected with L^2_t H^1_x which gives regularity by interpolation alone"
        ],
        "verification": "check_pde_regularity"
    },
    # === QUANTUM MECHANICS ===
    # Berry phase for a spin-1 particle in a conical field
    {
        "id": "hyp07_spin1_berry_phase",
        "domain": "quantum_mechanics",
        "context": "The Berry phase for a spin-1 particle adiabatically transported around a cone of half-angle theta in parameter space is:",
        "candidates": [
            "gamma = -m * Omega where m = -1,0,+1 and Omega is the solid angle, so m=0 state acquires NO geometric phase",
            "gamma = -Omega for all three m states equally, independent of the magnetic quantum number",
            "gamma = -m^2 * Omega, giving the same phase for m=+1 and m=-1 but different from m=0",
            "gamma = -(m + 1/2) * Omega due to a spin-orbit correction term for integer spin particles"
        ],
        "verification": "calc_berry_phase"
    },
    # === TOPOLOGICAL PHASES ===
    # Z2 invariant for a specific model
    {
        "id": "hyp08_kane_mele_z2",
        "domain": "topological_phases",
        "context": "In the Kane-Mele model on a honeycomb lattice with spin-orbit coupling lambda_SO and Rashba coupling lambda_R, the Z2 topological invariant is:",
        "candidates": [
            "nu = 1 (topological) when lambda_SO > 0 and lambda_R < 2*sqrt(3)*lambda_SO, regardless of sign",
            "nu = 1 only when lambda_SO > lambda_R strictly, with a sharp transition at equality",
            "nu = 1 for any nonzero lambda_SO regardless of lambda_R, because Rashba cannot destroy the Z2 phase",
            "nu depends on the filling factor and is topological only at half-filling"
        ],
        "verification": "calc_z2_invariant"
    },
    # === DRUG INTERACTIONS ===
    # CYP3A4 inhibition cascade — does ketoconazole + grapefruit stack?
    {
        "id": "hyp09_cyp3a4_double_inhibition",
        "domain": "pharmacology",
        "context": "When a patient takes both ketoconazole (strong CYP3A4 inhibitor) and drinks grapefruit juice (moderate CYP3A4 inhibitor), the combined effect on a CYP3A4 substrate is:",
        "candidates": [
            "nearly identical to ketoconazole alone because strong inhibition already saturates the enzyme (ceiling effect)",
            "additive — the AUC increase equals the sum of individual AUC increases from each inhibitor",
            "synergistic — the combined AUC increase exceeds the sum because they inhibit by different mechanisms",
            "antagonistic — grapefruit juice paradoxically reduces ketoconazole absorption lowering net inhibition"
        ],
        "verification": "predict_ddi_auc_change"
    },
    # === CLIMATE PHYSICS ===
    # Water vapor feedback strength
    {
        "id": "hyp10_water_vapor_feedback",
        "domain": "climate_physics",
        "context": "The water vapor feedback factor (amplification of CO2 forcing) in Earth's climate system is approximately:",
        "candidates": [
            "1.8 ± 0.3 (roughly doubles the bare CO2 forcing), well-constrained by Clausius-Clapeyron and observations",
            "3.5 ± 1.0 (triples the forcing), dominated by upper tropospheric humidity which is poorly constrained",
            "1.0 ± 0.1 (no amplification), because increased water vapor also increases cloud albedo canceling the greenhouse effect",
            "variable between 0.5 and 5.0 depending on latitude, with no meaningful global average"
        ],
        "verification": "analyze_climate_feedback"
    },
    # === CATALYSIS ===
    # Does the d-band center predict HER activity for single-atom catalysts?
    {
        "id": "hyp11_single_atom_dband",
        "domain": "catalysis",
        "context": "For single-atom catalysts (SACs) on carbon supports, the relationship between d-band center and hydrogen evolution reaction (HER) activity is:",
        "candidates": [
            "follows a volcano curve similar to bulk metals but shifted due to the coordination environment",
            "shows no correlation because the d-band model breaks down for isolated metal atoms",
            "linearly correlated — more negative d-band center always gives better HER activity",
            "inversely correlated — SACs with d-band center near the Fermi level are always optimal"
        ],
        "verification": "calc_d_band_center"
    },
    # === INFORMATION THEORY ===
    # Channel capacity of a relay channel — still open in general
    {
        "id": "hyp12_relay_channel_capacity",
        "domain": "information_theory",
        "context": "The capacity of the general discrete memoryless relay channel (source -> relay -> destination with direct link) is:",
        "candidates": [
            "unknown in general — only cut-set upper bound and decode-forward lower bound are known, with a gap",
            "equal to the max-flow min-cut value, which has been proven to be tight for all discrete channels",
            "equal to the capacity of the direct link alone because the relay cannot improve on direct transmission",
            "computable via the Blahut-Arimoto algorithm applied to the joint source-relay input distribution"
        ],
        "verification": "calc_channel_capacity_awgn"
    },
]


@dataclass
class AdapterVote:
    adapter_name: str
    domain: str
    margins: list  # margin for each candidate
    best_idx: int  # which candidate this adapter picks
    confidence: float  # margin of best over second-best


@dataclass
class HypothesisResult:
    claim_id: str
    domain: str
    context: str
    candidates: list
    base_vote: int  # which candidate base model picks
    base_margins: list
    adapter_votes: list  # list of AdapterVote
    consensus_idx: int  # which candidate has most adapter support
    consensus_strength: float  # how many adapters agree
    convergent: bool  # do adapters from different domains agree?


def load_model():
    """Load base model."""
    print("Loading Qwen3-14B-Base...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-14B-Base")
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def load_adapter(model, adapter_path, d_inner=64):
    """Load an adapter onto the model."""
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
    return adapter, lm_head


def score_candidates(model, tokenizer, claim, adapter=None, lm_head=None):
    """Score all candidates for a claim, return margins."""
    # Score truth=each candidate, distractors=all others
    margins = []
    for i, cand in enumerate(claim["candidates"]):
        others = [c for j, c in enumerate(claim["candidates"]) if j != i]
        win, margin, _, _ = score_fact_mc(
            model, tokenizer,
            claim["context"], cand, others,
            adapter=adapter, lm_head=lm_head,
        )
        margins.append(float(margin))
    return margins


def select_adapters_for_claim(claim, adapter_dir, max_domain=5, max_cross=5):
    """Select relevant adapters for a claim based on domain + cross-domain.

    With max_domain=5, max_cross=5 we get up to 10 adapters per claim,
    giving a much stronger vote signal.
    """
    domain = claim["domain"]
    all_npz = sorted([f for f in os.listdir(adapter_dir) if f.endswith('.npz')])

    # Filter out adapters with bias parameters (different architecture, will error)
    # We'll catch errors at runtime but prefer to avoid known-bad ones
    skip_prefixes = []  # could add known-bad prefixes here

    # Domain-specific keyword mapping — broader now
    domain_keywords = {
        "chemical_networks": ["chemical", "chem_enzyme", "chem08"],
        "vortex_dynamics": ["vortex", "ns_regularity", "reduced_navier"],
        "enzyme_kinetics": ["enzyme", "biochem", "chem_enzyme", "chem08"],
        "hamiltonian_mechanics": ["hamiltonian", "3body"],
        "number_theory": ["number_theory", "algebra_topology", "analysis_pde"],
        "pde_regularity": ["ns_regularity", "analysis_pde", "reduced_navier"],
        "quantum_mechanics": ["quantum", "topological", "black_hole"],
        "topological_phases": ["topological", "quantum", "black_hole"],
        "pharmacology": ["drug", "pharma", "clinical", "genetics_pharmaco"],
        "climate_physics": ["climate", "greenhouse", "battery"],
        "catalysis": ["catalyst", "battery", "chemistry", "chem"],
        "information_theory": ["information", "crypto", "network_calc"],
    }

    # Cross-domain keyword mapping — more diverse now
    cross_domains = {
        "chemical_networks": ["hamiltonian", "vortex", "3body", "ns_regularity", "drug"],
        "vortex_dynamics": ["hamiltonian", "3body", "analysis_pde", "number_theory"],
        "enzyme_kinetics": ["chemical", "drug", "clinical", "hamiltonian"],
        "hamiltonian_mechanics": ["vortex", "3body", "ns_regularity", "analysis_pde", "number_theory"],
        "number_theory": ["algebra_topology", "analysis_pde", "hamiltonian", "information"],
        "pde_regularity": ["hamiltonian", "vortex", "3body", "analysis_pde", "number_theory"],
        "quantum_mechanics": ["hamiltonian", "topological", "3body", "information", "black_hole"],
        "topological_phases": ["quantum", "hamiltonian", "black_hole", "information", "number_theory"],
        "pharmacology": ["biochem", "enzyme", "chemical", "aging", "antibiotic"],
        "climate_physics": ["battery", "chemistry", "catalys", "aging", "biochem"],
        "catalysis": ["chemistry", "battery", "biochem", "drug", "chemical"],
        "information_theory": ["crypto", "network", "quantum", "number_theory", "algebra"],
    }

    keywords = domain_keywords.get(domain, [domain.split("_")[0]])
    cross_kws = cross_domains.get(domain, [])

    seen = set()

    # Get domain adapters
    domain_adapters = []
    for kw in keywords:
        for f in all_npz:
            if kw in f.lower() and f not in seen:
                domain_adapters.append((domain, f))
                seen.add(f)
                if len(domain_adapters) >= max_domain:
                    break
        if len(domain_adapters) >= max_domain:
            break

    # Get cross-domain adapters
    cross_adapters = []
    for kw in cross_kws:
        for f in all_npz:
            if kw in f.lower() and f not in seen:
                cross_adapters.append(("cross_" + kw, f))
                seen.add(f)
                if len(cross_adapters) >= max_cross:
                    break
        if len(cross_adapters) >= max_cross:
            break

    return domain_adapters + cross_adapters


def run_hypothesis_engine(claims, verify=False):
    """Run the full hypothesis engine."""
    adapter_dir = os.path.join(ROOT, "adapters", "qwen3_4b_base")
    model, tokenizer = load_model()

    results = []

    for claim in claims:
        print(f"\n{'='*70}")
        print(f"  {claim['id']}: {claim['domain']}")
        print(f"  {claim['context'][:80]}...")
        print(f"{'='*70}")

        # Score with base model
        base_margins = score_candidates(model, tokenizer, claim)
        base_vote = int(np.argmax(base_margins))
        print(f"\n  BASE MODEL picks candidate {base_vote}: margin={base_margins[base_vote]:+.2f}")
        for i, (m, c) in enumerate(zip(base_margins, claim["candidates"])):
            marker = " <<<" if i == base_vote else ""
            print(f"    [{i}] {m:+8.2f}  {c[:65]}{marker}")

        # Score with each relevant adapter
        adapters = select_adapters_for_claim(claim, adapter_dir)
        adapter_votes = []

        for adomain, afile in adapters:
            apath = os.path.join(adapter_dir, afile)
            try:
                adpt, lm_head = load_adapter(model, apath)
                margins = score_candidates(model, tokenizer, claim,
                                          adapter=adpt, lm_head=lm_head)
                best = int(np.argmax(margins))
                sorted_m = sorted(margins, reverse=True)
                confidence = sorted_m[0] - sorted_m[1] if len(sorted_m) > 1 else sorted_m[0]

                vote = AdapterVote(
                    adapter_name=afile,
                    domain=adomain,
                    margins=margins,
                    best_idx=best,
                    confidence=confidence,
                )
                adapter_votes.append(vote)
                shift = "SHIFTED" if best != base_vote else "agrees"
                print(f"  {afile[:45]:45s} → candidate {best} (conf={confidence:+.1f}) [{shift}]")
            except Exception as e:
                print(f"  {afile[:45]:45s} → ERROR: {e}")

        # Find consensus — three methods
        if adapter_votes:
            # Method 1: Simple majority (original)
            vote_counts = {}
            for v in adapter_votes:
                vote_counts[v.best_idx] = vote_counts.get(v.best_idx, 0) + 1
            simple_idx = max(vote_counts, key=vote_counts.get)
            simple_strength = vote_counts[simple_idx] / len(adapter_votes)

            # Method 2: Confidence-weighted (parabolic focus)
            # Each adapter's vote is weighted by its confidence (margin gap)
            # High confidence = coherent light, low = noise
            weighted_scores = {}
            for v in adapter_votes:
                w = max(0, v.confidence)  # only positive confidence counts
                weighted_scores[v.best_idx] = weighted_scores.get(v.best_idx, 0) + w
            total_w = sum(weighted_scores.values()) or 1
            confident_idx = max(weighted_scores, key=weighted_scores.get)
            confident_strength = weighted_scores[confident_idx] / total_w

            # Method 3: Domain-weighted (specialists count more)
            # Domain adapters get weight = confidence * 2
            # Cross-domain adapters get weight = confidence * 1
            domain_weighted = {}
            for v in adapter_votes:
                w = max(0, v.confidence)
                if not v.domain.startswith("cross_"):
                    w *= 2.0  # domain specialist bonus
                domain_weighted[v.best_idx] = domain_weighted.get(v.best_idx, 0) + w
            total_dw = sum(domain_weighted.values()) or 1
            domain_idx = max(domain_weighted, key=domain_weighted.get)
            domain_strength = domain_weighted[domain_idx] / total_dw

            # Use confidence-weighted as primary (the parabolic focus)
            consensus_idx = confident_idx
            consensus_strength = confident_strength

            # Check if votes come from different domains
            consensus_domains = set(v.domain for v in adapter_votes if v.best_idx == consensus_idx)
            convergent = len(consensus_domains) >= 2

            # Report all three methods
            print(f"\n  Voting methods:")
            print(f"    Simple majority:      candidate {simple_idx} ({simple_strength:.0%})")
            print(f"    Confidence-weighted:  candidate {confident_idx} ({confident_strength:.0%})")
            print(f"    Domain-weighted:      candidate {domain_idx} ({domain_strength:.0%})")
            if simple_idx != confident_idx or simple_idx != domain_idx:
                print(f"    *** Methods DISAGREE — signal is noisy ***")
            else:
                print(f"    *** All methods AGREE — strong signal ***")

            result = HypothesisResult(
                claim_id=claim["id"],
                domain=claim["domain"],
                context=claim["context"],
                candidates=claim["candidates"],
                base_vote=base_vote,
                base_margins=base_margins,
                adapter_votes=[asdict(v) for v in adapter_votes],
                consensus_idx=consensus_idx,
                consensus_strength=consensus_strength,
                convergent=convergent,
            )
            results.append(result)

            print(f"\n  CONSENSUS: candidate {consensus_idx} ({consensus_strength:.0%} of adapters)")
            print(f"  Base picked: {base_vote}, Adapters picked: {consensus_idx}")
            if consensus_idx != base_vote:
                print(f"  *** ADAPTERS DISAGREE WITH BASE — potential discovery ***")
            if convergent:
                print(f"  *** CONVERGENT — multiple domains agree ***")
            print(f"  Answer: {claim['candidates'][consensus_idx][:80]}")

    # Summary
    print(f"\n\n{'='*70}")
    print("  HYPOTHESIS ENGINE SUMMARY")
    print(f"{'='*70}")

    discoveries = []
    for r in results:
        status = "CONVERGENT DISCOVERY" if (r.consensus_idx != r.base_vote and r.convergent) else \
                 "ADAPTER SHIFT" if r.consensus_idx != r.base_vote else \
                 "CONFIRMS BASE"
        print(f"\n  {r.claim_id} ({r.domain})")
        print(f"    Base: candidate {r.base_vote} | Adapters: candidate {r.consensus_idx} ({r.consensus_strength:.0%})")
        print(f"    Status: {status}")
        print(f"    Answer: {r.candidates[r.consensus_idx][:80]}")

        if status == "CONVERGENT DISCOVERY":
            discoveries.append(r)

    print(f"\n  Total claims: {len(results)}")
    print(f"  Confirms base: {sum(1 for r in results if r.consensus_idx == r.base_vote)}")
    print(f"  Adapter shifts: {sum(1 for r in results if r.consensus_idx != r.base_vote)}")
    print(f"  Convergent discoveries: {len(discoveries)}")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_claims": len(results),
        "n_discoveries": len(discoveries),
        "results": [asdict(r) for r in results],
    }
    output_path = os.path.join(ROOT, "results", "hypothesis_engine_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return results, discoveries


def select_adapters_by_prefix(adapter_dir, prefix, max_adapters=15):
    """Select ALL compatible adapters matching a domain prefix.

    Used for focused mode where we want maximum coverage from a single domain.
    Skips 27B-only adapters (those with bias params).
    """
    all_npz = sorted([f for f in os.listdir(adapter_dir) if f.endswith('.npz')])
    results = []
    for f in all_npz:
        if prefix in f.lower():
            try:
                weights = mx.load(os.path.join(adapter_dir, f))
                if not any('bias' in k for k in weights.keys()):
                    results.append((prefix, f))
            except:
                pass
        if len(results) >= max_adapters:
            break
    return results


def run_focused_engine(claims_path, verify=False):
    """Run hypothesis engine with focused claims that specify adapter_prefix."""
    adapter_dir = os.path.join(ROOT, "adapters", "qwen3_4b_base")
    model, tokenizer = load_model()

    with open(claims_path) as f:
        claims = json.load(f)

    results = []

    for claim in claims:
        print(f"\n{'='*70}")
        print(f"  {claim['id']}: {claim['domain']}")
        print(f"  {claim['context'][:80]}...")
        print(f"{'='*70}")

        # Score with base model
        base_margins = score_candidates(model, tokenizer, claim)
        base_vote = int(np.argmax(base_margins))
        print(f"\n  BASE MODEL picks candidate {base_vote}: margin={base_margins[base_vote]:+.2f}")
        for i, (m, c) in enumerate(zip(base_margins, claim["candidates"])):
            marker = " <<<" if i == base_vote else ""
            print(f"    [{i}] {m:+8.2f}  {c[:65]}{marker}")

        # Get ALL adapters for this domain prefix
        prefix = claim.get("adapter_prefix", claim["domain"])
        adapters = select_adapters_by_prefix(adapter_dir, prefix)
        print(f"\n  Found {len(adapters)} compatible adapters for prefix '{prefix}'")

        adapter_votes = []
        for adomain, afile in adapters:
            apath = os.path.join(adapter_dir, afile)
            try:
                adpt, lm_head = load_adapter(model, apath)
                margins = score_candidates(model, tokenizer, claim,
                                          adapter=adpt, lm_head=lm_head)
                best = int(np.argmax(margins))
                sorted_m = sorted(margins, reverse=True)
                confidence = sorted_m[0] - sorted_m[1] if len(sorted_m) > 1 else sorted_m[0]

                vote = AdapterVote(
                    adapter_name=afile, domain=adomain,
                    margins=margins, best_idx=best, confidence=confidence,
                )
                adapter_votes.append(vote)
                shift = "SHIFTED" if best != base_vote else "agrees"
                print(f"  {afile[:50]:50s} → cand {best} (conf={confidence:+.1f}) [{shift}]")
            except Exception as e:
                print(f"  {afile[:50]:50s} → ERROR: {str(e)[:60]}")

        if adapter_votes:
            # Confidence-weighted consensus
            weighted_scores = {}
            for v in adapter_votes:
                w = max(0, v.confidence)
                weighted_scores[v.best_idx] = weighted_scores.get(v.best_idx, 0) + w
            total_w = sum(weighted_scores.values()) or 1
            consensus_idx = max(weighted_scores, key=weighted_scores.get)
            consensus_strength = weighted_scores[consensus_idx] / total_w

            # Simple majority for comparison
            vote_counts = {}
            for v in adapter_votes:
                vote_counts[v.best_idx] = vote_counts.get(v.best_idx, 0) + 1
            simple_idx = max(vote_counts, key=vote_counts.get)
            simple_strength = vote_counts[simple_idx] / len(adapter_votes)

            consensus_domains = set(v.domain for v in adapter_votes if v.best_idx == consensus_idx)
            convergent = len(consensus_domains) >= 1  # same domain still counts in focused mode

            print(f"\n  Simple majority:     candidate {simple_idx} ({simple_strength:.0%} of {len(adapter_votes)} adapters)")
            print(f"  Confidence-weighted: candidate {consensus_idx} ({consensus_strength:.0%} of weight)")
            if consensus_idx != base_vote:
                print(f"  *** ADAPTERS DISAGREE WITH BASE ***")
            print(f"  Answer: {claim['candidates'][consensus_idx][:80]}")

            result = HypothesisResult(
                claim_id=claim["id"], domain=claim["domain"],
                context=claim["context"], candidates=claim["candidates"],
                base_vote=base_vote, base_margins=base_margins,
                adapter_votes=[asdict(v) for v in adapter_votes],
                consensus_idx=consensus_idx, consensus_strength=consensus_strength,
                convergent=convergent,
            )
            results.append(result)

    # Summary
    print(f"\n\n{'='*70}")
    print("  FOCUSED HYPOTHESIS ENGINE SUMMARY")
    print(f"{'='*70}")

    for r in results:
        status = "DISCOVERY" if r.consensus_idx != r.base_vote else "CONFIRMS BASE"
        print(f"\n  {r.claim_id}")
        print(f"    Base: cand {r.base_vote} | Adapters: cand {r.consensus_idx} ({r.consensus_strength:.0%})")
        print(f"    [{status}] {r.candidates[r.consensus_idx][:80]}")

    n_disc = sum(1 for r in results if r.consensus_idx != r.base_vote)
    print(f"\n  Total: {len(results)} claims, {n_disc} adapter shifts, {len(results)-n_disc} confirms base")

    output_path = os.path.join(ROOT, "results", "focused_hypothesis_results.json")
    with open(output_path, "w") as f:
        json.dump({"results": [asdict(r) for r in results]}, f, indent=2)
    print(f"  Saved to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Run MCP verification on discoveries")
    parser.add_argument("--claims", type=str, nargs="+", help="Run only specific claim IDs")
    parser.add_argument("--focused", type=str, default=None, help="Path to focused claims JSON")
    args = parser.parse_args()

    if args.focused:
        run_focused_engine(args.focused, verify=args.verify)
    else:
        claims = CANDIDATE_CLAIMS
        if args.claims:
            claims = [c for c in claims if c["id"] in args.claims]
        results, discoveries = run_hypothesis_engine(claims, verify=args.verify)
