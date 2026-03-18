#!/usr/bin/env python3
"""
Correlate length ratio with baseline accuracy to test the hypothesis:
    Length ratio predicts baseline accuracy better than other factors.
"""

import json
from pathlib import Path

# Manual baseline data from candidates.tsv
BASELINES = {
    # HIGH BASELINE (>50%)
    "operating_systems": 83,
    "pl_concurrency": 80,
    "pl_paradigms": 75,
    "cryptography": 75,
    "distributed_systems": 75,
    "aging_biology": 75,
    "chemistry": 75,
    "networking": 67,
    "database_internals": 67,
    "organic_chemistry": 67,
    "control_systems": 67,
    "holographic_qinfo": 67,
    "battery_technology": 67,
    "pl_type_systems": 58,
    "economics_finance": 58,
    "dark_matter_energy": 58,
    "protein_folding": 58,
    "quantum_mechanics": 50,
    "pl_memory": 50,
    "pl_compilers": 50,

    # BELOW THRESHOLD (25-50%)
    "antibiotic_resistance": 42,
    "3body_conservation": 40,
    "consciousness": 33,
    "millennium_problems": 25,
    "proof_techniques": 25,

    # DEEP GAPS (<25%)
    "knot_invariants": 6,  # 1/16
    "kinetic_k": 0,  # 0/8
    "continuous_qf": 0,  # 0/12
    "ns_regularity": 0,  # 0/16
    "llm_hallucination": 0,  # 0/12 (estimated from pattern)
    "llm_reasoning": 17,  # 2/12
    "llm_alignment": 8,  # 1/12
    "llm_training": 8,  # 1/12
    "llm_evaluation": 0,  # 0/12
    "llm_context_memory": 0,  # 0/10
}


def analyze_fact(fact: dict) -> dict:
    """Get length analysis for a fact."""
    truth = fact.get("truth", "")
    distractors = fact.get("distractors", [])
    if not distractors:
        return None
    truth_len = len(truth)
    min_distractor_len = min(len(d) for d in distractors)
    return {
        "length_ratio": truth_len / min_distractor_len if min_distractor_len > 0 else 999
    }


def analyze_domain(facts_file: Path) -> dict:
    """Get average length ratio for a domain."""
    with open(facts_file) as f:
        data = json.load(f)
    facts = data.get("facts", data.get("examples", []))
    ratios = []
    for fact in facts:
        analysis = analyze_fact(fact)
        if analysis:
            ratios.append(analysis["length_ratio"])
    if not ratios:
        return None
    return {
        "domain": facts_file.stem.replace("_facts", ""),
        "avg_length_ratio": sum(ratios) / len(ratios),
        "max_length_ratio": max(ratios),
    }


def main():
    problems_dir = Path(__file__).parent.parent / "problems"
    facts_files = list(problems_dir.glob("*_facts.json"))

    # Get length ratios
    domain_ratios = {}
    for f in facts_files:
        try:
            analysis = analyze_domain(f)
            if analysis:
                domain_ratios[analysis["domain"]] = analysis
        except Exception:
            pass

    # Match with baselines
    matched = []
    for domain, baseline in BASELINES.items():
        if domain in domain_ratios:
            matched.append({
                "domain": domain,
                "baseline": baseline,
                "length_ratio": domain_ratios[domain]["avg_length_ratio"],
                "max_ratio": domain_ratios[domain]["max_length_ratio"],
            })

    # Sort by length ratio
    matched.sort(key=lambda x: x["length_ratio"])

    print("=" * 75)
    print("LENGTH RATIO vs BASELINE ACCURACY")
    print("=" * 75)
    print()
    print(f"{'Domain':<30} {'Ratio':>8} {'MaxRatio':>8} {'Baseline':>8}")
    print("-" * 60)

    for m in matched:
        print(f"{m['domain']:<30} {m['length_ratio']:>8.2f} {m['max_ratio']:>8.1f} {m['baseline']:>7}%")

    # Compute correlation
    ratios = [m["length_ratio"] for m in matched]
    baselines = [m["baseline"] for m in matched]

    n = len(matched)
    mean_r = sum(ratios) / n
    mean_b = sum(baselines) / n

    cov = sum((r - mean_r) * (b - mean_b) for r, b in zip(ratios, baselines)) / n
    std_r = (sum((r - mean_r) ** 2 for r in ratios) / n) ** 0.5
    std_b = (sum((b - mean_b) ** 2 for b in baselines) / n) ** 0.5

    if std_r > 0 and std_b > 0:
        correlation = cov / (std_r * std_b)
    else:
        correlation = 0

    print()
    print("=" * 75)
    print(f"CORRELATION: r = {correlation:.3f}")
    print("=" * 75)
    print()

    if correlation < -0.5:
        print("STRONG NEGATIVE CORRELATION: Higher length ratio → Lower baseline")
        print("This supports the hypothesis that length ratio predicts difficulty.")
    elif correlation < -0.3:
        print("MODERATE NEGATIVE CORRELATION: Length ratio matters but isn't the only factor.")
    else:
        print("WEAK CORRELATION: Length ratio alone doesn't explain baseline differences.")

    # Partition analysis
    print()
    print("=" * 75)
    print("PARTITION ANALYSIS")
    print("=" * 75)

    low_ratio = [m for m in matched if m["length_ratio"] < 1.2]
    med_ratio = [m for m in matched if 1.2 <= m["length_ratio"] < 2.5]
    high_ratio = [m for m in matched if m["length_ratio"] >= 2.5]

    if low_ratio:
        avg_b = sum(m["baseline"] for m in low_ratio) / len(low_ratio)
        print(f"\nRatio < 1.2 (n={len(low_ratio)}): Avg baseline = {avg_b:.1f}%")
        for m in low_ratio:
            print(f"  {m['domain']}: {m['baseline']}%")

    if med_ratio:
        avg_b = sum(m["baseline"] for m in med_ratio) / len(med_ratio)
        print(f"\nRatio 1.2-2.5 (n={len(med_ratio)}): Avg baseline = {avg_b:.1f}%")
        for m in med_ratio:
            print(f"  {m['domain']}: {m['baseline']}%")

    if high_ratio:
        avg_b = sum(m["baseline"] for m in high_ratio) / len(high_ratio)
        print(f"\nRatio >= 2.5 (n={len(high_ratio)}): Avg baseline = {avg_b:.1f}%")
        for m in high_ratio:
            print(f"  {m['domain']}: {m['baseline']}%")


if __name__ == "__main__":
    main()
