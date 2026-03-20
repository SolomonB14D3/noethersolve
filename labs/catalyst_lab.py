#!/usr/bin/env python3
"""catalyst_lab.py -- Catalyst screening for Hydrogen Evolution Reaction.

Chains NoetherSolve catalysis tools to screen transition-metal catalysts:
  1. d-band center analysis (electronic structure)
  2. Volcano plot positioning (Sabatier principle)
  3. Scaling relations (selectivity constraints)
  4. BEP activation energy (kinetic barrier)
  5. Optimal catalyst ranking (combined score)

Usage:
    python labs/catalyst_lab.py
    python labs/catalyst_lab.py --verbose

Data sources:
    - d-band centers: DFT calculations (Hammer-Nørskov, Mavrikakis et al.)
    - Adsorption energies: BEEF-vdW functional (Wellendorff 2012)
    - Materials Project API (optional): Surface energies, stability

References:
    - Nørskov et al. 2005, J. Electrochem. Soc. (volcano model)
    - Greeley et al. 2006, Nature Materials (HER screening)
    - Seh et al. 2017, Science (universal scaling)

⚠️  NOTE: Computational predictions require experimental validation.
    DFT-derived adsorption energies have typical errors of ±0.1-0.2 eV.
    Surface reconstruction, coverage effects, and electrolyte interactions
    may alter real-world activity from these ideal-surface predictions.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Ensure noethersolve is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from noethersolve.catalysis import (
    calc_volcano_position,
    calc_d_band_center,
    get_scaling_relation,
    calc_bep_activation,
    find_optimal_catalyst,
    D_BAND_CENTERS,
    K_B_EV,
)

RESULTS_DIR = _ROOT / "results" / "labs" / "catalyst_discovery"
DEFAULT_REACTION = "HER"
SUPPORTED_REACTIONS = ["HER", "OER", "ORR"]
TEMPERATURE_K = 300.0


# ---------------------------------------------------------------------------
# Catalyst candidate definitions
# ---------------------------------------------------------------------------

@dataclass
class CatalystCandidate:
    """A metal catalyst candidate for HER screening."""
    symbol: str
    name: str
    cost_rank: int        # 1=cheapest, 8=most expensive (relative)
    abundance_ppm: float  # crustal abundance


CANDIDATES: List[CatalystCandidate] = [
    CatalystCandidate("Pt", "Platinum",  8, 0.005),
    CatalystCandidate("Pd", "Palladium", 7, 0.015),
    CatalystCandidate("Ni", "Nickel",    2, 84.0),
    CatalystCandidate("Cu", "Copper",    3, 60.0),
    CatalystCandidate("Au", "Gold",      6, 0.004),
    CatalystCandidate("Ru", "Ruthenium", 5, 0.001),
    CatalystCandidate("Ir", "Iridium",   7, 0.001),
    CatalystCandidate("Rh", "Rhodium",   6, 0.001),
]


# ---------------------------------------------------------------------------
# Screening result
# ---------------------------------------------------------------------------

@dataclass
class ScreeningResult:
    """Result of screening a single catalyst candidate."""
    symbol: str
    name: str
    # d-band analysis
    d_band_center_eV: float
    binding_strength: str
    # Volcano plot
    estimated_dG_eV: float
    distance_from_peak_eV: float
    relative_activity: float
    limiting_side: str
    # BEP kinetics
    Ea_eV: float
    rate_constant: float
    # Scaling relation
    OH_binding_eV: float
    selectivity_constraint: str
    # Cost-adjusted
    cost_rank: int
    abundance_ppm: float
    # Composite
    activity_score: float    # 0-40: volcano position
    kinetic_score: float     # 0-25: BEP barrier
    cost_score: float        # 0-20: cost/abundance
    selectivity_score: float # 0-15: scaling relation
    total_score: float       # 0-100
    verdict: str             # TOP / VIABLE / MARGINAL


# ---------------------------------------------------------------------------
# Screening pipeline
# ---------------------------------------------------------------------------

def screen_candidate(cat: CatalystCandidate, reaction: str = DEFAULT_REACTION, verbose: bool = False) -> ScreeningResult:
    """Run the full screening pipeline on one catalyst."""

    # -- Step 1: d-band center --
    dband = calc_d_band_center(cat.symbol, reference_metal="Pt")
    if verbose:
        print(dband)

    # -- Step 2: Volcano position --
    # Estimate adsorption energy from d-band center
    # ΔG ≈ -0.3 * ε_d - 1.0 (same mapping as find_optimal_catalyst)
    eps_d = D_BAND_CENTERS[cat.symbol]
    estimated_dG = -0.3 * eps_d - 1.0
    volcano = calc_volcano_position(reaction, estimated_dG)
    if verbose:
        print(volcano)

    # -- Step 3: Scaling relation (H -> OH for HER side reactions) --
    scaling = get_scaling_relation("O", "OH")
    # Estimate OH binding from the O binding (which tracks with H for metals)
    OH_binding = scaling.slope * estimated_dG + scaling.intercept

    # -- Step 4: BEP activation energy for H2 dissociation --
    bep = calc_bep_activation("H2_dissociation", estimated_dG, TEMPERATURE_K)
    if verbose:
        print(bep)

    # -- Step 5: Composite scoring --

    # Activity score (0-40): closer to volcano peak = better
    dist = abs(volcano.distance_from_peak)
    if dist < 0.05:
        activity_score = 40.0
    elif dist < 0.2:
        activity_score = 35.0
    elif dist < 0.5:
        activity_score = 25.0
    elif dist < 1.0:
        activity_score = 15.0
    else:
        activity_score = max(0.0, 10.0 - dist * 3)

    # Kinetic score (0-25): lower Ea = better
    if bep.Ea < 0.1:
        kinetic_score = 25.0
    elif bep.Ea < 0.3:
        kinetic_score = 20.0
    elif bep.Ea < 0.5:
        kinetic_score = 15.0
    elif bep.Ea < 1.0:
        kinetic_score = 8.0
    else:
        kinetic_score = max(0.0, 5.0 - bep.Ea)

    # Cost score (0-20): cheaper and more abundant = better
    cost_score = 20.0 * (1.0 - (cat.cost_rank - 1) / 7.0)
    # Bonus for high abundance
    if cat.abundance_ppm > 10.0:
        cost_score = min(20.0, cost_score + 3.0)

    # Selectivity score (0-15): less constrained scaling = better
    # Tighter slope means harder to optimize selectivity independently
    sel_penalty = abs(scaling.slope - 1.0) * 5.0
    selectivity_score = max(0.0, 15.0 - sel_penalty)

    total = activity_score + kinetic_score + cost_score + selectivity_score

    if total >= 70:
        verdict = "TOP"
    elif total >= 50:
        verdict = "VIABLE"
    else:
        verdict = "MARGINAL"

    return ScreeningResult(
        symbol=cat.symbol,
        name=cat.name,
        d_band_center_eV=eps_d,
        binding_strength=dband.binding_strength,
        estimated_dG_eV=estimated_dG,
        distance_from_peak_eV=volcano.distance_from_peak,
        relative_activity=volcano.relative_activity,
        limiting_side=volcano.limiting_side,
        Ea_eV=bep.Ea,
        rate_constant=bep.rate_constant,
        OH_binding_eV=OH_binding,
        selectivity_constraint=scaling.selectivity_constraint,
        cost_rank=cat.cost_rank,
        abundance_ppm=cat.abundance_ppm,
        activity_score=activity_score,
        kinetic_score=kinetic_score,
        cost_score=cost_score,
        selectivity_score=selectivity_score,
        total_score=total,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(results: List[ScreeningResult], reaction: str = DEFAULT_REACTION):
    """Print a human-readable screening report."""
    print("\n" + "=" * 76)
    print(f"  CATALYST DISCOVERY LAB -- {reaction} Screening Report")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  T = {TEMPERATURE_K:.0f} K")
    print("=" * 76)

    ranked = sorted(results, key=lambda r: r.total_score, reverse=True)

    # Header
    print(f"\n  {'Rk':>2}  {'Metal':5}  {'Score':>5}  {'Verdict':8}"
          f"  {'eps_d':>6}  {'dG':>6}  {'Dist':>5}  {'Ea':>5}  {'k (s-1)':>10}"
          f"  {'Cost':>4}")
    print(f"  {'--':>2}  {'-----':5}  {'-----':>5}  {'--------':8}"
          f"  {'------':>6}  {'------':>6}  {'-----':>5}  {'-----':>5}  {'----------':>10}"
          f"  {'----':>4}")

    for rank, r in enumerate(ranked, 1):
        tag = {"TOP": "[TOP]", "VIABLE": "[OK]", "MARGINAL": "[--]"}[r.verdict]
        print(f"  {rank:2d}  {r.symbol:5s}  {r.total_score:5.1f}  {tag:8s}"
              f"  {r.d_band_center_eV:+6.2f}  {r.estimated_dG_eV:+6.2f}"
              f"  {abs(r.distance_from_peak_eV):5.2f}  {r.Ea_eV:5.2f}"
              f"  {r.rate_constant:10.2e}  {r.cost_rank:4d}")

    # Score breakdown
    print(f"\n  Score breakdown (act/kin/cost/sel):")
    for r in ranked:
        bar_len = int(r.total_score / 5)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"    {r.symbol:5s} [{bar}] "
              f"{r.activity_score:4.0f}/{r.kinetic_score:4.0f}/"
              f"{r.cost_score:4.0f}/{r.selectivity_score:4.0f}")

    # Optimal catalyst from full library
    print(f"\n  --- find_optimal_catalyst() ranking (full d-band library) ---")
    opt_report = find_optimal_catalyst(
        reaction, [c.symbol for c in CANDIDATES]
    )
    print(opt_report)

    # Summary
    n_top = sum(1 for r in ranked if r.verdict == "TOP")
    n_viable = sum(1 for r in ranked if r.verdict == "VIABLE")
    n_marg = sum(1 for r in ranked if r.verdict == "MARGINAL")
    print(f"\n  {'='*76}")
    print(f"  Summary: {n_top} TOP / {n_viable} VIABLE / {n_marg} MARGINAL "
          f"out of {len(ranked)} candidates")
    best = ranked[0]
    print(f"  Best candidate: {best.symbol} ({best.name}) "
          f"-- score {best.total_score:.1f}/100")
    print(f"  {'='*76}\n")


def save_results(results: List[ScreeningResult], outpath: Path, reaction: str = DEFAULT_REACTION):
    """Save results to JSON."""
    ranked = sorted(results, key=lambda r: r.total_score, reverse=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "catalyst_lab v0.2",
        "reaction": reaction,
        "temperature_K": TEMPERATURE_K,
        "n_candidates": len(ranked),
        "n_top": sum(1 for r in ranked if r.verdict == "TOP"),
        "n_viable": sum(1 for r in ranked if r.verdict == "VIABLE"),
        "n_marginal": sum(1 for r in ranked if r.verdict == "MARGINAL"),
        "results": [asdict(r) for r in ranked],
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
    parser = argparse.ArgumentParser(
        description="Catalyst Discovery Lab -- HER/OER/ORR screening pipeline"
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed tool reports for each metal")
    parser.add_argument("--reaction", "-r", choices=SUPPORTED_REACTIONS,
                        default=DEFAULT_REACTION,
                        help=f"Reaction to screen for (default: {DEFAULT_REACTION})")
    parser.add_argument("--all-reactions", action="store_true",
                        help="Run screening for all reactions (HER, OER, ORR)")
    args = parser.parse_args()

    reactions_to_run = SUPPORTED_REACTIONS if args.all_reactions else [args.reaction]

    for reaction in reactions_to_run:
        print(f"\n  Screening {len(CANDIDATES)} catalyst candidates for {reaction}...")

        results = []
        for cat in CANDIDATES:
            try:
                result = screen_candidate(cat, reaction=reaction, verbose=args.verbose)
                results.append(result)
            except Exception as e:
                print(f"  ERROR screening {cat.symbol}: {e}")

        if not results:
            print("  No results generated.")
            continue

        print_report(results, reaction)

        outpath = RESULTS_DIR / f"screening_results_{reaction.lower()}.json"
        save_results(results, outpath, reaction)


if __name__ == "__main__":
    main()
