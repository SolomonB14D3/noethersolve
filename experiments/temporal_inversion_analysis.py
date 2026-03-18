#!/usr/bin/env python3
"""Analyze Temporal Inversion Bias: does the model prefer pre-confirmation positions?

Hypothesis: The model systematically inverts recent experimental confirmations,
preferring skeptical pre-confirmation positions over post-confirmation consensus.

This is distinct from Certainty Contamination Bias (preferring definitive language).
Temporal Inversion is about preferring OLDER (skeptical) positions over NEWER (confirmed).

Methodology:
1. Categorize facts by temporal status:
   - RECENT_CONFIRMATION: established 2016-2024 (EHT, LIGO, muon g-2, etc.)
   - ESTABLISHED: known pre-2010 (thermodynamics, QM basics, etc.)
   - FRONTIER: genuinely uncertain (dark matter candidates, quantum gravity)
2. Run oracle on each category
3. Measure correlation between confirmation recency and oracle margin
"""

import json
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Temporal categorization of key facts
# Format: fact_id -> (category, year_established, notes)
TEMPORAL_CATEGORIES = {
    # PARTICLE PHYSICS
    "ppf01_muon_g2": ("RECENT_CONFIRMATION", 2021, "Fermilab confirmed Brookhaven anomaly"),
    "ppf02_g2_theory": ("FRONTIER", None, "Theory disagreement ongoing"),
    "ppf03_higgs_coupling": ("FRONTIER", None, "Still uncertain"),
    "ppf04_higgs_width": ("RECENT_CONFIRMATION", 2022, "CMS confirmed narrow width"),
    "ppf05_neutrino_mass": ("FRONTIER", None, "Mass ordering still uncertain"),
    "ppf06_neutrino_cp": ("FRONTIER", None, "Hints, not confirmed"),
    "ppf07_dm_direct": ("FRONTIER", None, "No detection yet"),

    # BLACK HOLES
    "bhf01_eht": ("RECENT_CONFIRMATION", 2019, "First EHT image M87*"),
    "bhf02_ringdown": ("RECENT_CONFIRMATION", 2019, "LIGO ringdown matches GR"),
    "bhf03_islands": ("RECENT_CONFIRMATION", 2019, "Island formula published"),
    "bhf04_soft_hair": ("FRONTIER", None, "Still theoretical"),
    "bhf05_complexity": ("FRONTIER", None, "Conjecture, not proven"),
    "bhf06_page_curve": ("RECENT_CONFIRMATION", 2019, "Page curve derived from gravity"),
    "bhf07_replica": ("RECENT_CONFIRMATION", 2019, "Replica wormhole papers"),

    # COSMOLOGY (hypothetical - need to check)
    "cof01_hubble": ("FRONTIER", None, "Tension ongoing"),
    "cof02_cmb": ("ESTABLISHED", 1992, "COBE confirmed"),
    "cof03_inflation": ("FRONTIER", None, "No direct evidence"),

    # NEUTRINO FRONTIERS
    "nf01_sterile": ("FRONTIER", None, "No confirmed detection"),
    "nf02_majorana": ("FRONTIER", None, "Still unknown"),
    "nf03_hierarchy": ("FRONTIER", None, "Hints only"),

    # DARK MATTER
    "dm01_wimp": ("FRONTIER", None, "No detection"),
    "dm02_axion": ("FRONTIER", None, "Hints, no detection"),

    # ESTABLISHED PHYSICS (control group)
    "pf01_noether": ("ESTABLISHED", 1918, "Noether's theorem"),
    "pf02_conservation": ("ESTABLISHED", 1850, "Energy conservation"),
    "pf03_entropy": ("ESTABLISHED", 1865, "2nd law thermodynamics"),
}


@dataclass
class FactResult:
    fact_id: str
    domain: str
    category: str  # RECENT_CONFIRMATION, ESTABLISHED, FRONTIER
    year: Optional[int]
    margin: float
    passed: bool
    truth: str
    best_distractor: str


def load_model():
    """Load model and tokenizer."""
    import mlx_lm
    from noethersolve.train_utils import get_lm_head_fn

    print("Loading Qwen/Qwen3-4B-Base...")
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    lm_head = get_lm_head_fn(model)
    return model, tokenizer, lm_head


def run_oracle_on_fact(model, tokenizer, fact: dict) -> tuple:
    """Run oracle on a single fact, return (margin, passed, best_distractor)."""
    from noethersolve.oracle import score_fact_mc

    context = fact["context"]
    truth = fact["truth"]
    distractors = fact.get("distractors", [])

    result = score_fact_mc(model, tokenizer, context, truth, distractors)
    win, margin, truth_lp, best_dist_lp = result

    # Find best distractor
    best_dist = distractors[0] if distractors else ""

    return margin, win, best_dist


def analyze_domains(model, tokenizer, lm_head, domains: list[str]):
    """Analyze temporal inversion across specified domains."""
    results: list[FactResult] = []

    problems_dir = Path(__file__).resolve().parent.parent / "problems"

    for domain in domains:
        facts_file = problems_dir / f"{domain}_facts.json"
        if not facts_file.exists():
            print(f"  Skipping {domain} - no facts file")
            continue

        with open(facts_file) as f:
            data = json.load(f)

        facts = data.get("facts", data) if isinstance(data, dict) else data
        print(f"\n  {domain}: {len(facts)} facts")

        for fact in facts:
            fact_id = fact.get("id", "unknown")

            # Get temporal category
            if fact_id in TEMPORAL_CATEGORIES:
                category, year, _ = TEMPORAL_CATEGORIES[fact_id]
            else:
                # Default categorization based on heuristics
                truth = fact.get("truth", "").lower()
                if any(w in truth for w in ["uncertain", "unknown", "hints", "awaits", "may"]):
                    category = "FRONTIER"
                    year = None
                elif any(w in truth for w in ["confirmed", "established", "shows", "proves"]):
                    category = "RECENT_CONFIRMATION"
                    year = 2020  # Guess
                else:
                    category = "ESTABLISHED"
                    year = 2000

            # Run oracle
            margin, passed, best_dist = run_oracle_on_fact(model, tokenizer, fact)

            results.append(FactResult(
                fact_id=fact_id,
                domain=domain,
                category=category,
                year=year,
                margin=margin,
                passed=passed,
                truth=fact.get("truth", ""),
                best_distractor=best_dist,
            ))

            status = "PASS" if passed else "FAIL"
            print(f"    {fact_id}: {status} margin={margin:.2f} [{category}]")

    return results


def compute_statistics(results: list[FactResult]):
    """Compute statistics by temporal category."""
    by_category = defaultdict(list)

    for r in results:
        by_category[r.category].append(r)

    print("\n" + "=" * 60)
    print("TEMPORAL INVERSION ANALYSIS - RESULTS")
    print("=" * 60)

    for category in ["RECENT_CONFIRMATION", "ESTABLISHED", "FRONTIER"]:
        facts = by_category[category]
        if not facts:
            continue

        n_passed = sum(1 for f in facts if f.passed)
        n_total = len(facts)
        mean_margin = sum(f.margin for f in facts) / n_total

        print(f"\n{category}:")
        print(f"  Pass rate: {n_passed}/{n_total} ({100*n_passed/n_total:.1f}%)")
        print(f"  Mean margin: {mean_margin:.2f}")

        # Show worst failures
        failures = sorted([f for f in facts if not f.passed], key=lambda x: x.margin)[:5]
        if failures:
            print("  Worst failures:")
            for f in failures:
                print(f"    {f.fact_id}: margin={f.margin:.2f}")

    # Correlation with year
    confirmed = [r for r in results if r.category == "RECENT_CONFIRMATION" and r.year]
    if len(confirmed) >= 3:
        import numpy as np
        years = [r.year for r in confirmed]
        margins = [r.margin for r in confirmed]
        if np.std(years) > 0:
            corr = np.corrcoef(years, margins)[0, 1]
            print("\nCorrelation (year vs margin) for RECENT_CONFIRMATION:")
            print(f"  r = {corr:.3f} (n={len(confirmed)})")
            if corr < -0.3:
                print("  → More recent confirmations have LOWER margins (temporal inversion!)")
            elif corr > 0.3:
                print("  → More recent confirmations have HIGHER margins (good!)")
            else:
                print("  → Weak or no correlation with year")

    return by_category


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=[
        "particle_physics_frontiers",
        "black_hole_frontiers",
        "neutrino_frontiers",
        "cosmology_frontiers",
        "dark_matter_energy",
        "physics_fundamentals",
    ])
    parser.add_argument("--dry-run", action="store_true", help="Just print categorizations")
    args = parser.parse_args()

    if args.dry_run:
        print("Known temporal categorizations:")
        for fid, (cat, year, note) in sorted(TEMPORAL_CATEGORIES.items()):
            print(f"  {fid}: {cat} ({year}) - {note}")
        return

    model, tokenizer, lm_head = load_model()
    results = analyze_domains(model, tokenizer, lm_head, args.domains)
    compute_statistics(results)

    # Save results
    output_path = Path(__file__).resolve().parent.parent / "results" / "temporal_inversion_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump([{
            "fact_id": r.fact_id,
            "domain": r.domain,
            "category": r.category,
            "year": r.year,
            "margin": r.margin,
            "passed": r.passed,
            "truth": r.truth,
            "best_distractor": r.best_distractor,
        } for r in results], f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
