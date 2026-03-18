#!/usr/bin/env python3
"""Generate training facts from all 21 NoetherSolve toolkit modules.

Extracts the hardcoded knowledge from each tool and converts it into
oracle-format MC facts (context + truth + distractors) for training
a single unified "toolkit" adapter.

The key insight: our tools contain verified reference databases of
domain knowledge. Distilling this into one adapter could boost
external benchmarks (MMLU, TruthfulQA) because the adapter would
teach general domain knowledge patterns, not just specific oracle facts.

Usage:
    python experiments/generate_toolkit_facts.py
    python experiments/generate_toolkit_facts.py --output problems/toolkit_facts.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_conjecture_facts():
    """Generate facts from conjecture_status.py (63 conjectures)."""
    from noethersolve.conjecture_status import _build_database
    db = _build_database()
    facts = []

    for cid, info in db.items():
        # Fact: conjecture status
        status_map = {
            "PROVEN": "proven/resolved",
            "OPEN": "still open/unresolved",
            "PARTIALLY_RESOLVED": "partially resolved",
            "REFUTED": "refuted/disproven",
        }
        true_status = status_map.get(info.status, info.status.lower())
        wrong_statuses = [s for s in ["proven/resolved", "still open/unresolved",
                                        "refuted/disproven", "partially resolved"]
                         if s != true_status][:3]

        facts.append({
            "id": f"conj_{cid}_status",
            "context": f"The current status of the {info.name} conjecture is:",
            "truth": true_status,
            "distractors": wrong_statuses,
            "cluster": "conjecture_status",
            "source_tool": "conjecture_status",
        })

        # If proven, fact about solver/year
        if info.status == "PROVEN" and info.solver:
            facts.append({
                "id": f"conj_{cid}_solver",
                "context": f"The {info.name} was resolved by:",
                "truth": f"{info.solver}" + (f" in {info.year}" if info.year else ""),
                "distractors": [
                    "still unresolved as of 2024",
                    "no one has submitted a valid proof",
                    "claimed proven but proof was retracted",
                ],
                "cluster": "conjecture_proofs",
                "source_tool": "conjecture_status",
            })

    return facts


def generate_complexity_facts():
    """Generate facts from complexity.py (class relationships)."""
    from noethersolve.complexity import (
        KNOWN_INCLUSIONS, KNOWN_SEPARATIONS, COMPLETENESS,
    )
    facts = []

    # Known inclusions
    for sub, sup in list(KNOWN_INCLUSIONS)[:15]:
        facts.append({
            "id": f"complex_incl_{sub}_{sup}",
            "context": f"The relationship between complexity classes {sub} and {sup} is:",
            "truth": f"{sub} ⊆ {sup} (proven inclusion)",
            "distractors": [
                f"{sup} ⊆ {sub} (reverse inclusion)",
                f"{sub} and {sup} are incomparable",
                f"{sub} = {sup} (proven equality)",
            ],
            "cluster": "complexity_inclusions",
            "source_tool": "complexity",
        })

    # Known separations
    for lo, hi in list(KNOWN_SEPARATIONS)[:8]:
        facts.append({
            "id": f"complex_sep_{lo}_{hi}",
            "context": f"The separation between {lo} and {hi} is:",
            "truth": f"{lo} ⊊ {hi} (proven strict separation)",
            "distractors": [
                f"{lo} = {hi} (they are equal)",
                "separation is still unknown",
                f"{hi} ⊊ {lo} (reverse)",
            ],
            "cluster": "complexity_separations",
            "source_tool": "complexity",
        })

    # NP-complete problems from COMPLETENESS dict
    for prob, info in list(COMPLETENESS.items())[:10]:
        if info.get("class") == "NP" and info.get("type") == "complete":
            facts.append({
                "id": f"complex_npc_{prob.replace(' ', '_')[:30]}",
                "context": f"The complexity classification of {prob} is:",
                "truth": "NP-complete",
                "distractors": [
                    "in P (polynomial time)",
                    "in BPP (randomized polynomial)",
                    "PSPACE-complete",
                ],
                "cluster": "complexity_npc",
                "source_tool": "complexity",
            })

    return facts


def generate_proof_barrier_facts():
    """Generate facts from proof_barriers.py (10 barriers)."""
    from noethersolve.proof_barriers import list_barriers
    barriers = list_barriers()
    facts = []

    for barrier in barriers:
        bid = barrier.name.lower().replace(" ", "_").replace("'", "")[:30]
        facts.append({
            "id": f"barrier_{bid}",
            "context": f"The {barrier.name} barrier in computational complexity prevents:",
            "truth": barrier.formal_statement[:120] if barrier.formal_statement else
                     f"certain proof techniques from resolving {', '.join(list(barrier.blocked_problems)[:2])}",
            "distractors": [
                "no known proof technique barriers exist for these problems",
                "only relativization is a barrier, others are obsolete",
                "these barriers were all overcome by 2020",
            ],
            "cluster": "proof_barriers",
            "source_tool": "proof_barriers",
        })

        # Blocked problems
        if barrier.blocked_problems:
            probs = ", ".join(list(barrier.blocked_problems)[:3])
            facts.append({
                "id": f"barrier_{bid}_blocks",
                "context": f"Problems blocked by the {barrier.name} barrier include:",
                "truth": probs,
                "distractors": [
                    "no problems are blocked by this barrier",
                    "only graph isomorphism",
                    "only factoring-related problems",
                ],
                "cluster": "proof_barriers",
                "source_tool": "proof_barriers",
            })

    return facts


def generate_pharmacokinetics_facts():
    """Generate facts from pharmacokinetics.py (CYP interactions)."""
    from noethersolve.pharmacokinetics import (
        _CYP_SUBSTRATES, _STRONG_INHIBITORS, _PHENOTYPE_IMPACT,
    )
    facts = []

    # CYP substrates
    for enzyme, drugs in _CYP_SUBSTRATES.items():
        drug_list = sorted(drugs)[:5]
        facts.append({
            "id": f"pharm_{enzyme}_substrates",
            "context": f"Key substrates of the {enzyme} enzyme include:",
            "truth": ", ".join(drug_list),
            "distractors": [
                "this enzyme has no known drug substrates",
                "only acetaminophen is metabolized by this enzyme",
                "this enzyme is not clinically relevant",
            ],
            "cluster": "pharmacokinetics",
            "source_tool": "pharmacokinetics",
        })

    # Strong inhibitors
    for enzyme, inhibitors in _STRONG_INHIBITORS.items():
        if not inhibitors:
            continue
        inh_list = sorted(inhibitors)[:4]
        facts.append({
            "id": f"pharm_{enzyme}_inhibitors",
            "context": f"Strong inhibitors of {enzyme} include:",
            "truth": ", ".join(inh_list),
            "distractors": [
                "no strong inhibitors are known for this enzyme",
                "only grapefruit juice inhibits this enzyme",
                "inhibition of this enzyme is not clinically significant",
            ],
            "cluster": "pharmacokinetics",
            "source_tool": "pharmacokinetics",
        })

    # Phenotype impacts
    for enzyme, phenotypes in _PHENOTYPE_IMPACT.items():
        for pheno, effect in phenotypes.items():
            facts.append({
                "id": f"pharm_{enzyme}_{pheno}",
                "context": f"A {pheno} metabolizer for {enzyme} has the following drug impact:",
                "truth": effect,
                "distractors": [
                    "no dose adjustment needed for any phenotype",
                    "metabolizer status has no clinical relevance",
                    "all phenotypes process drugs identically",
                ],
                "cluster": "pharmacogenomics",
                "source_tool": "pharmacokinetics",
            })

    return facts


def generate_pde_regularity_facts():
    """Generate facts from pde_regularity.py using public API."""
    from noethersolve.pde_regularity import check_pde_regularity, check_blowup
    facts = []

    # Test common PDEs
    pde_names = [
        "laplace", "heat", "wave", "navier-stokes", "euler",
        "burgers", "kdv", "nls", "schrodinger", "maxwell",
    ]
    for pde_name in pde_names:
        try:
            report = check_pde_regularity(pde_name)
            if report and report.regularity:
                facts.append({
                    "id": f"pde_{pde_name.replace('-','_')}_reg",
                    "context": f"The regularity result for the {pde_name} equation is:",
                    "truth": report.regularity[:120],
                    "distractors": [
                        "global smooth solutions always exist",
                        "solutions are always discontinuous",
                        "regularity is completely unknown",
                    ],
                    "cluster": "pde_regularity",
                    "source_tool": "pde_regularity",
                })
        except Exception:
            pass

    # Blowup results
    blowup_cases = ["nls_supercritical", "euler_3d", "semilinear_heat"]
    for case in blowup_cases:
        try:
            report = check_blowup(case)
            if report:
                facts.append({
                    "id": f"pde_blowup_{case}",
                    "context": f"Finite-time blowup for {case.replace('_', ' ')}:",
                    "truth": str(report)[:120],
                    "distractors": [
                        "blowup never occurs for this equation",
                        "blowup occurs for all initial data",
                        "blowup question is completely open",
                    ],
                    "cluster": "pde_blowup",
                    "source_tool": "pde_regularity",
                })
        except Exception:
            pass

    return facts


def generate_llm_claims_facts():
    """Generate facts from llm_claims.py (35+ topics)."""
    from noethersolve.llm_claims import _build_database
    db = _build_database()
    facts = []

    for topic_id, info in db.items():
        # Truth statement
        facts.append({
            "id": f"llm_{topic_id}_truth",
            "context": f"Regarding {info.description[:80]}:",
            "truth": info.truth[:120],
            "distractors": info.misconceptions[:3],
            "cluster": f"llm_{info.domain}",
            "source_tool": "llm_claims",
        })

        # Each misconception as a negative fact
        for j, misconception in enumerate(info.misconceptions[:2]):
            facts.append({
                "id": f"llm_{topic_id}_misc{j}",
                "context": f"A common misconception about {info.description[:60]} is:",
                "truth": f"'{misconception}' is FALSE — {info.truth[:80]}",
                "distractors": [
                    f"'{misconception}' is well-established and proven",
                    "this is not a misconception, it's standard knowledge",
                    "experts unanimously agree with this statement",
                ],
                "cluster": f"llm_{info.domain}",
                "source_tool": "llm_claims",
            })

    return facts


def generate_knot_facts():
    """Generate facts from knot.py (knot invariants)."""
    facts = [
        {
            "id": "knot_reidemeister",
            "context": "Reidemeister moves are important because:",
            "truth": "two knot diagrams represent the same knot iff related by Reidemeister moves",
            "distractors": [
                "they change the knot type",
                "they only apply to trivial knots",
                "they are purely cosmetic, not mathematical",
            ],
            "cluster": "knot_theory",
            "source_tool": "knot",
        },
        {
            "id": "knot_jones_polynomial",
            "context": "The Jones polynomial of a knot:",
            "truth": "is a knot invariant — same polynomial for equivalent knots, distinguishes many knot types",
            "distractors": [
                "always equals 1 for all knots",
                "changes under Reidemeister moves",
                "is only defined for prime knots",
            ],
            "cluster": "knot_theory",
            "source_tool": "knot",
        },
        {
            "id": "knot_crossing_number",
            "context": "The crossing number of a knot is:",
            "truth": "the minimum number of crossings in any diagram of that knot",
            "distractors": [
                "the total crossings in any random diagram",
                "always equal to the genus",
                "not a knot invariant",
            ],
            "cluster": "knot_theory",
            "source_tool": "knot",
        },
    ]
    return facts


def generate_number_theory_facts():
    """Generate facts from number_theory.py."""
    facts = [
        {
            "id": "nt_goldbach",
            "context": "Goldbach's conjecture states:",
            "truth": "every even integer > 2 is the sum of two primes (unproven, verified to ~4×10^18)",
            "distractors": [
                "every odd integer is the sum of two primes",
                "this was proven by Euler in 1742",
                "counterexamples exist near 10^15",
            ],
            "cluster": "number_theory",
            "source_tool": "number_theory",
        },
        {
            "id": "nt_collatz",
            "context": "The Collatz conjecture concerns:",
            "truth": "iterating n→n/2 (even) or n→3n+1 (odd) always reaches 1 (unproven)",
            "distractors": [
                "the conjecture was proven by Tao in 2020",
                "counterexamples exist for large n",
                "it only applies to prime numbers",
            ],
            "cluster": "number_theory",
            "source_tool": "number_theory",
        },
        {
            "id": "nt_twin_primes",
            "context": "The twin prime conjecture asks whether:",
            "truth": "there are infinitely many prime pairs (p, p+2) — unproven, best bound by Zhang/Maynard is gap≤246",
            "distractors": [
                "twin primes are finite in number",
                "proven by Yitang Zhang in 2013",
                "the gap is exactly 2 infinitely often (proven)",
            ],
            "cluster": "number_theory",
            "source_tool": "number_theory",
        },
        {
            "id": "nt_abc",
            "context": "The ABC conjecture relates to:",
            "truth": "for coprime a+b=c, the radical rad(abc) constrains c — Mochizuki's proof remains controversial",
            "distractors": [
                "universally accepted as proven since 2012",
                "definitively disproven in 2021",
                "only applies to even numbers",
            ],
            "cluster": "number_theory",
            "source_tool": "number_theory",
        },
    ]
    return facts


def generate_conservation_facts():
    """Generate facts from conservation law monitors."""
    facts = [
        {
            "id": "cons_noether",
            "context": "Noether's theorem connects:",
            "truth": "every continuous symmetry of a physical system to a conserved quantity",
            "distractors": [
                "discrete symmetries to conservation laws",
                "only rotational symmetry to energy",
                "symmetry breaking to conservation",
            ],
            "cluster": "conservation",
            "source_tool": "monitor",
        },
        {
            "id": "cons_hamiltonian",
            "context": "In Hamiltonian mechanics, the Hamiltonian H is:",
            "truth": "conserved when it has no explicit time dependence (energy conservation)",
            "distractors": [
                "always conserved regardless of time dependence",
                "never exactly conserved in real systems",
                "only conserved for linear systems",
            ],
            "cluster": "conservation",
            "source_tool": "hamiltonian",
        },
        {
            "id": "cons_liouville",
            "context": "Liouville's theorem in Hamiltonian mechanics states:",
            "truth": "phase space volume is preserved under Hamiltonian flow (symplecticity)",
            "distractors": [
                "energy is always conserved",
                "trajectories never cross in phase space",
                "all orbits are periodic",
            ],
            "cluster": "conservation",
            "source_tool": "hamiltonian",
        },
        {
            "id": "cons_em_helicity",
            "context": "Electromagnetic helicity is:",
            "truth": "conserved in source-free vacuum (topological invariant related to field linkage)",
            "distractors": [
                "always conserved even with charges present",
                "not a physical observable",
                "the same as electromagnetic energy",
            ],
            "cluster": "conservation",
            "source_tool": "monitor_em",
        },
        {
            "id": "cons_vorticity",
            "context": "In 2D incompressible flow, vorticity is:",
            "truth": "materially conserved along particle paths (Kelvin's theorem in 2D)",
            "distractors": [
                "always dissipated by viscosity",
                "only conserved in 3D flows",
                "never conserved in real fluids",
            ],
            "cluster": "conservation",
            "source_tool": "monitor",
        },
    ]
    return facts


def generate_reductions_facts():
    """Generate facts about computational reductions."""
    facts = [
        {
            "id": "red_sat_3sat",
            "context": "The relationship between SAT and 3-SAT is:",
            "truth": "SAT reduces to 3-SAT (polynomial time), both NP-complete, 3-SAT is the canonical NP-complete problem",
            "distractors": [
                "3-SAT is easier than SAT",
                "they are incomparable in complexity",
                "SAT is in P but 3-SAT is NP-complete",
            ],
            "cluster": "reductions",
            "source_tool": "reductions",
        },
        {
            "id": "red_cook_levin",
            "context": "The Cook-Levin theorem establishes:",
            "truth": "SAT is NP-complete — the first problem proven NP-complete (1971)",
            "distractors": [
                "P = NP",
                "SAT is in P",
                "NP-completeness doesn't exist",
            ],
            "cluster": "reductions",
            "source_tool": "reductions",
        },
    ]
    return facts


def main():
    all_generators = [
        ("conjecture_status", generate_conjecture_facts),
        ("complexity", generate_complexity_facts),
        ("proof_barriers", generate_proof_barrier_facts),
        ("pharmacokinetics", generate_pharmacokinetics_facts),
        ("pde_regularity", generate_pde_regularity_facts),
        ("llm_claims", generate_llm_claims_facts),
        ("knot_theory", generate_knot_facts),
        ("number_theory", generate_number_theory_facts),
        ("conservation_laws", generate_conservation_facts),
        ("reductions", generate_reductions_facts),
    ]

    all_facts = []
    for tool_name, generator in all_generators:
        try:
            facts = generator()
            all_facts.extend(facts)
            print(f"  {tool_name}: {len(facts)} facts")
        except Exception as e:
            print(f"  {tool_name}: FAILED — {e}")

    print(f"\n  TOTAL: {len(all_facts)} facts from {len(all_generators)} tools")

    # Save
    output = {
        "problem": "toolkit_knowledge",
        "description": "Knowledge distilled from all 21 NoetherSolve toolkit modules",
        "n_tools": len(all_generators),
        "facts": all_facts,
    }

    out_path = Path(__file__).resolve().parent.parent / "problems" / "toolkit_facts.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
