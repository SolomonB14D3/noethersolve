#!/usr/bin/env python3
"""
Analyze distractor patterns across all domains to find what makes facts learnable.

Hypothesis: Physics domains have worse gaps than LLM domains because:
1. Physics truths require qualifiers ("approximately", "under conditions")
2. Physics distractors are short absolutes ("exactly conserved", "not conserved")
3. The length ratio may be worse than for LLM facts
"""

import json
from pathlib import Path

def analyze_fact(fact: dict) -> dict:
    """Analyze a single fact for distractor attractiveness patterns."""
    truth = fact.get("truth", "")
    distractors = fact.get("distractors", [])

    if not distractors:
        return None

    # Length analysis
    truth_len = len(truth)
    distractor_lens = [len(d) for d in distractors]
    min_distractor_len = min(distractor_lens)
    max_distractor_len = max(distractor_lens)
    sum(distractor_lens) / len(distractor_lens)

    # Length ratio (truth / shortest distractor) - >1 means truth is longer
    length_ratio = truth_len / min_distractor_len if min_distractor_len > 0 else float('inf')

    # Hedging markers in truth
    hedging_words = ["approximately", "may", "might", "under", "but", "however",
                     "partially", "sometimes", "often", "typically", "generally",
                     "in some cases", "can be", "tends to", "likely"]
    truth_lower = truth.lower()
    hedge_count = sum(1 for h in hedging_words if h in truth_lower)

    # Absolute markers in distractors
    absolute_words = ["exactly", "always", "never", "guaranteed", "impossible",
                      "perfectly", "completely", "all", "none", "only", "must",
                      "is solved", "eliminates", "not conserved", "undefined"]
    absolute_distractors = 0
    for d in distractors:
        d_lower = d.lower()
        if any(a in d_lower for a in absolute_words):
            absolute_distractors += 1

    return {
        "fact_id": fact.get("id", "unknown"),
        "truth_len": truth_len,
        "min_distractor_len": min_distractor_len,
        "max_distractor_len": max_distractor_len,
        "length_ratio": length_ratio,
        "hedge_count": hedge_count,
        "absolute_distractors": absolute_distractors,
        "truth": truth[:80] + "..." if len(truth) > 80 else truth,
        "shortest_distractor": min(distractors, key=len) if distractors else ""
    }


def analyze_domain(facts_file: Path) -> dict:
    """Analyze all facts in a domain."""
    with open(facts_file) as f:
        data = json.load(f)

    facts = data.get("facts", data.get("examples", []))
    analyses = []

    for fact in facts:
        analysis = analyze_fact(fact)
        if analysis:
            analyses.append(analysis)

    if not analyses:
        return None

    # Domain-level statistics
    avg_length_ratio = sum(a["length_ratio"] for a in analyses) / len(analyses)
    avg_hedge_count = sum(a["hedge_count"] for a in analyses) / len(analyses)
    avg_absolute = sum(a["absolute_distractors"] for a in analyses) / len(analyses)

    # Worst facts (highest length ratio)
    worst_by_length = sorted(analyses, key=lambda x: x["length_ratio"], reverse=True)[:3]

    return {
        "domain": facts_file.stem.replace("_facts", ""),
        "num_facts": len(analyses),
        "avg_length_ratio": avg_length_ratio,
        "avg_hedge_count": avg_hedge_count,
        "avg_absolute_distractors": avg_absolute,
        "worst_facts": worst_by_length,
        "all_facts": analyses
    }


def main():
    problems_dir = Path(__file__).parent.parent / "problems"
    facts_files = sorted(problems_dir.glob("*_facts.json"))

    results = []
    for f in facts_files:
        try:
            analysis = analyze_domain(f)
            if analysis:
                results.append(analysis)
        except Exception as e:
            print(f"Error analyzing {f}: {e}")

    # Sort by average length ratio (highest = worst)
    results.sort(key=lambda x: x["avg_length_ratio"], reverse=True)

    print("=" * 80)
    print("DOMAIN ANALYSIS: DISTRACTOR PATTERNS")
    print("=" * 80)
    print()
    print("Sorted by length_ratio (truth_len / shortest_distractor_len):")
    print("Higher ratio = truth is longer = harder to learn")
    print()
    print(f"{'Domain':<35} {'N':>3} {'LenRatio':>8} {'Hedges':>6} {'Absolutes':>8}")
    print("-" * 65)

    for r in results:
        print(f"{r['domain']:<35} {r['num_facts']:>3} {r['avg_length_ratio']:>8.2f} "
              f"{r['avg_hedge_count']:>6.1f} {r['avg_absolute_distractors']:>8.1f}")

    print()
    print("=" * 80)
    print("WORST INDIVIDUAL FACTS (by length ratio)")
    print("=" * 80)

    all_facts = []
    for r in results:
        for fact in r["all_facts"]:
            fact["domain"] = r["domain"]
            all_facts.append(fact)

    worst_facts = sorted(all_facts, key=lambda x: x["length_ratio"], reverse=True)[:20]

    for i, f in enumerate(worst_facts, 1):
        print(f"\n{i}. {f['domain']} / {f['fact_id']} (ratio={f['length_ratio']:.1f})")
        print(f"   Truth ({f['truth_len']}): {f['truth']}")
        print(f"   Shortest ({f['min_distractor_len']}): {f['shortest_distractor']}")

    # Group domains by type to see patterns
    print()
    print("=" * 80)
    print("DOMAIN GROUPING")
    print("=" * 80)

    categories = {
        "LLM": ["llm_hallucination", "llm_reasoning", "llm_alignment", "llm_training",
                "llm_evaluation", "llm_context_memory"],
        "Physics_Novel": ["kinetic_k", "continuous_qf", "qf_ratio", "vortex_pair",
                          "3body_conservation", "ns_regularity"],
        "Physics_Standard": ["hamiltonian", "em_zilch", "quantum_mechanics"],
        "Math": ["millennium_problems", "number_theory_conjectures", "algebra_topology_conjectures",
                 "proof_techniques", "knot_invariants"],
        "Biology": ["genetics_therapeutics", "disease_targets", "protein_structure",
                    "origin_of_life", "protein_folding", "aging_biology"],
        "CS_Systems": ["distributed_systems", "networking", "operating_systems",
                       "database_internals"],
        "PL": ["pl_type_systems", "pl_memory", "pl_concurrency", "pl_paradigms",
               "pl_compilers", "pl_pitfalls"]
    }

    for cat_name, domains in categories.items():
        cat_results = [r for r in results if r["domain"] in domains]
        if cat_results:
            avg_ratio = sum(r["avg_length_ratio"] for r in cat_results) / len(cat_results)
            avg_hedges = sum(r["avg_hedge_count"] for r in cat_results) / len(cat_results)
            avg_absolutes = sum(r["avg_absolute_distractors"] for r in cat_results) / len(cat_results)
            print(f"\n{cat_name}:")
            print(f"  Avg length ratio: {avg_ratio:.2f}")
            print(f"  Avg hedges in truths: {avg_hedges:.2f}")
            print(f"  Avg absolutes in distractors: {avg_absolutes:.2f}")


if __name__ == "__main__":
    main()
