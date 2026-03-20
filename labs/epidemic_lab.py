#!/usr/bin/env python3
"""epidemic_lab.py -- Epidemic scenario modeling across five diseases.

Chains NoetherSolve epidemiology tools to model COVID-19 (Omicron +
original), measles, influenza, and Ebola: R0 lookup, SIR dynamics,
herd immunity, vaccine impact, doubling time, and final attack rate.

Usage:
    python labs/epidemic_lab.py
    python labs/epidemic_lab.py --verbose

Data sources:
    - R0 estimates: WHO/CDC published literature, peer-reviewed studies
    - Vaccine efficacy: Clinical trial data (Polack 2020, Baden 2021, etc.)
    - Generation times: Serial interval studies (Nishiura, Leung, etc.)

⚠️  DISCLAIMER: FOR EDUCATIONAL AND PLANNING PURPOSES ONLY
    SIR models are SIMPLIFIED APPROXIMATIONS. Real epidemics involve
    heterogeneous population mixing, spatial structure, age stratification,
    behavioral changes, and data uncertainty. Do not use for actual public
    health decisions without epidemiological expertise and validated models.
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

from noethersolve.epidemiology import (
    herd_immunity_threshold,
    sir_model,
    vaccine_impact,
    doubling_time,
    attack_rate,
    get_disease_R0,
)


# ---------------------------------------------------------------------------
# Disease scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class DiseaseScenario:
    """A disease with epidemiological and vaccination parameters."""
    name: str
    display_name: str
    r0_key: str               # key into DISEASE_R0
    infectious_period_days: float
    vaccine_efficacy: float   # VE (0-1)
    vaccine_coverage: float   # current realistic coverage (0-1)
    generation_time_days: float
    notes: str


SCENARIOS: List[DiseaseScenario] = [
    DiseaseScenario(
        name="covid19_omicron",
        display_name="COVID-19 (Omicron)",
        r0_key="covid19_omicron",
        infectious_period_days=5.0,
        vaccine_efficacy=0.50,
        vaccine_coverage=0.65,
        generation_time_days=3.0,
        notes="High transmissibility, partial vaccine escape",
    ),
    DiseaseScenario(
        name="measles",
        display_name="Measles",
        r0_key="measles",
        infectious_period_days=8.0,
        vaccine_efficacy=0.97,
        vaccine_coverage=0.85,
        generation_time_days=12.0,
        notes="Highest R0 of common diseases, excellent vaccine",
    ),
    DiseaseScenario(
        name="influenza_seasonal",
        display_name="Seasonal Influenza",
        r0_key="influenza_seasonal",
        infectious_period_days=5.0,
        vaccine_efficacy=0.45,
        vaccine_coverage=0.50,
        generation_time_days=3.0,
        notes="Low R0 but poor vaccine match; annual drift",
    ),
    DiseaseScenario(
        name="ebola",
        display_name="Ebola",
        r0_key="ebola",
        infectious_period_days=10.0,
        vaccine_efficacy=0.975,
        vaccine_coverage=0.40,
        generation_time_days=15.0,
        notes="Low R0, high CFR, ring vaccination strategy",
    ),
    DiseaseScenario(
        name="covid19_original",
        display_name="COVID-19 (Wuhan)",
        r0_key="covid19_original",
        infectious_period_days=7.0,
        vaccine_efficacy=0.95,
        vaccine_coverage=0.70,
        generation_time_days=5.0,
        notes="Original strain, high mRNA vaccine efficacy",
    ),
]

# Additional diseases for extended analysis
EXTENDED_SCENARIOS: List[DiseaseScenario] = [
    DiseaseScenario(
        name="pertussis",
        display_name="Pertussis (Whooping Cough)",
        r0_key="pertussis",
        infectious_period_days=21.0,
        vaccine_efficacy=0.85,
        vaccine_coverage=0.80,
        generation_time_days=14.0,
        notes="High R0, waning vaccine immunity, resurgence risk",
    ),
    DiseaseScenario(
        name="smallpox",
        display_name="Smallpox (Historical)",
        r0_key="smallpox",
        infectious_period_days=14.0,
        vaccine_efficacy=0.95,
        vaccine_coverage=0.0,  # Eradicated, no active vaccination
        generation_time_days=17.0,
        notes="Eradicated 1980 via ring vaccination; bioterror concern",
    ),
    DiseaseScenario(
        name="polio",
        display_name="Polio",
        r0_key="polio",
        infectious_period_days=14.0,
        vaccine_efficacy=0.99,
        vaccine_coverage=0.90,
        generation_time_days=7.0,
        notes="Near eradication; vaccine-derived strains emerging",
    ),
    DiseaseScenario(
        name="influenza_1918",
        display_name="Spanish Flu (1918)",
        r0_key="influenza_1918",
        infectious_period_days=7.0,
        vaccine_efficacy=0.0,
        vaccine_coverage=0.0,
        generation_time_days=3.0,
        notes="50M deaths; no vaccine at time; NPI-only control",
    ),
    DiseaseScenario(
        name="sars_2003",
        display_name="SARS (2003)",
        r0_key="sars_2003",
        infectious_period_days=10.0,
        vaccine_efficacy=0.0,
        vaccine_coverage=0.0,
        generation_time_days=8.5,
        notes="Contained via contact tracing; symptomatic-only spread",
    ),
]


# ---------------------------------------------------------------------------
# Screening pipeline
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """Full epidemiological profile for one disease scenario."""
    name: str
    display_name: str
    R0: float
    R0_range: str
    # SIR dynamics
    beta: float
    gamma: float
    generation_time_days: float
    peak_infected_pct: Optional[float]
    # Herd immunity
    herd_immunity_pct: float
    # Doubling time
    doubling_time_days: float
    growth_rate_per_day: float
    # Final attack rate
    attack_rate_pct: float
    final_susceptible_pct: float
    # Vaccine impact
    Rt_post_vaccine: float
    herd_achieved: bool
    critical_coverage_pct: float
    vaccine_efficacy_pct: float
    vaccine_coverage_pct: float
    # INTERVENTION OPTIMIZATION (new)
    coverage_gap_pct: float           # How far from herd immunity (neg = achieved)
    max_vaccine_Rt: float             # Rt if 100% coverage achieved
    npi_reduction_needed_pct: float   # % transmission reduction needed via NPI if vax alone fails
    controllable: bool                # Can be controlled with realistic interventions
    control_strategy: str             # "vaccination_alone" | "vaccination_plus_npi" | "npi_only" | "not_controllable"
    # Meta
    notes: str


def analyze_scenario(
    scenario: DiseaseScenario, verbose: bool = False,
) -> ScenarioResult:
    """Run the full epidemiological analysis chain on one disease."""

    # -- Step 1: R0 lookup --
    r0_range = get_disease_R0(scenario.r0_key)
    if r0_range is None:
        raise ValueError(f"Unknown disease key: {scenario.r0_key}")
    R0_mid = (r0_range[0] + r0_range[1]) / 2.0
    range_str = f"{r0_range[0]:.1f}-{r0_range[1]:.1f}"

    if verbose:
        print(f"    R0 lookup: {scenario.display_name} -> {range_str} (using {R0_mid:.1f})")

    # -- Step 2: SIR model parameters --
    gamma = 1.0 / scenario.infectious_period_days
    beta = R0_mid * gamma
    sir = sir_model(beta=beta, gamma=gamma)
    if verbose:
        print(sir)

    # -- Step 3: Herd immunity threshold --
    hit = herd_immunity_threshold(R0=R0_mid)
    if verbose:
        print(hit)

    # -- Step 4: Doubling time --
    dt = doubling_time(R0=R0_mid, generation_time=scenario.generation_time_days)
    if verbose:
        print(dt)

    # -- Step 5: Final attack rate --
    ar = attack_rate(R0=R0_mid)
    if verbose:
        print(ar)

    # -- Step 6: Vaccine impact --
    vi = vaccine_impact(
        R0=R0_mid,
        vaccine_efficacy=scenario.vaccine_efficacy,
        coverage=scenario.vaccine_coverage,
    )
    if verbose:
        print(vi)

    # -- Step 7: Intervention optimization (new) --
    # Coverage gap: how far from critical coverage
    coverage_gap = vi.critical_coverage - scenario.vaccine_coverage

    # What Rt would we get with 100% coverage?
    max_vax_Rt = R0_mid * (1 - scenario.vaccine_efficacy)

    # Can vaccination alone control it?
    if max_vax_Rt < 1.0:
        # Yes, vaccination alone is sufficient if coverage is high enough
        if vi.herd_immunity_reached:
            control_strategy = "vaccination_achieved"
            npi_needed = 0.0
            controllable = True
        else:
            control_strategy = "vaccination_alone"
            npi_needed = 0.0
            controllable = True
    else:
        # Even 100% vaccination won't get Rt < 1
        # Calculate NPI reduction needed: we need (1 - npi_reduction) * max_vax_Rt < 1
        # So npi_reduction > 1 - 1/max_vax_Rt
        npi_needed = max(0, (1 - 1.0 / max_vax_Rt) * 100) if max_vax_Rt > 0 else 100.0

        if npi_needed < 50:
            # Realistic NPI (masks, distancing) can achieve ~30-50% reduction
            control_strategy = "vaccination_plus_npi"
            controllable = True
        elif npi_needed < 80:
            # Requires aggressive NPI (lockdown-level)
            control_strategy = "aggressive_npi_required"
            controllable = True  # Possible but very difficult
        else:
            # Essentially uncontrollable with current tools
            control_strategy = "not_controllable"
            controllable = False

    if verbose:
        print(f"    Intervention: gap={coverage_gap*100:.1f}%, max_vax_Rt={max_vax_Rt:.2f}, "
              f"NPI_needed={npi_needed:.1f}%, strategy={control_strategy}")

    return ScenarioResult(
        name=scenario.name,
        display_name=scenario.display_name,
        R0=R0_mid,
        R0_range=range_str,
        beta=round(beta, 4),
        gamma=round(gamma, 4),
        generation_time_days=scenario.generation_time_days,
        peak_infected_pct=round(sir.peak_infected * 100, 1) if sir.peak_infected else None,
        herd_immunity_pct=round(hit.threshold_pct, 1),
        doubling_time_days=round(dt.doubling_time, 2),
        growth_rate_per_day=round(dt.growth_rate, 4),
        attack_rate_pct=round(ar.attack_rate_pct, 1),
        final_susceptible_pct=round(ar.final_susceptible * 100, 1),
        Rt_post_vaccine=round(vi.Rt_post_vaccine, 3),
        herd_achieved=vi.herd_immunity_reached,
        critical_coverage_pct=round(vi.critical_coverage * 100, 1),
        vaccine_efficacy_pct=round(scenario.vaccine_efficacy * 100, 1),
        vaccine_coverage_pct=round(scenario.vaccine_coverage * 100, 1),
        coverage_gap_pct=round(coverage_gap * 100, 1),
        max_vaccine_Rt=round(max_vax_Rt, 2),
        npi_reduction_needed_pct=round(npi_needed, 1),
        controllable=controllable,
        control_strategy=control_strategy,
        notes=scenario.notes,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(results: List[ScenarioResult]):
    """Print a comparison table across all disease scenarios."""
    print("\n" + "=" * 90)
    print("  EPIDEMIC LAB -- Disease Scenario Comparison")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 90)

    # Header
    hdr = (f"  {'Disease':24s} {'R0':>6s} {'HIT%':>6s} {'Td(d)':>7s} "
           f"{'AR%':>6s} {'Peak%':>6s} {'Rt_vax':>7s} {'Herd?':>6s}")
    print(f"\n{hdr}")
    print(f"  {'─'*24} {'─'*6} {'─'*6} {'─'*7} {'─'*6} {'─'*6} {'─'*7} {'─'*6}")

    for r in results:
        peak_str = f"{r.peak_infected_pct:5.1f}" if r.peak_infected_pct else "  N/A"
        herd_str = "  YES" if r.herd_achieved else "   NO"
        td_str = f"{r.doubling_time_days:6.1f}" if r.doubling_time_days < 1e6 else "   inf"
        print(f"  {r.display_name:24s} {r.R0:6.1f} {r.herd_immunity_pct:5.1f}% "
              f"{td_str}  {r.attack_rate_pct:5.1f}% {peak_str}% "
              f"{r.Rt_post_vaccine:6.2f}  {herd_str}")

    # Detailed cards
    print(f"\n{'─'*90}")
    print("  DETAILED PROFILES")
    print(f"{'─'*90}")

    for r in results:
        print(f"\n  {r.display_name} (R0 = {r.R0_range})")
        print(f"    SIR:  beta={r.beta:.4f}  gamma={r.gamma:.4f}  "
              f"T_gen={r.generation_time_days:.0f}d")
        print(f"    Growth: r={r.growth_rate_per_day:.4f}/d  "
              f"doubling={r.doubling_time_days:.1f}d")
        print(f"    Unmitigated: {r.attack_rate_pct:.1f}% infected, "
              f"{r.final_susceptible_pct:.1f}% escape")
        if r.peak_infected_pct:
            print(f"    Peak simultaneous infected: {r.peak_infected_pct:.1f}%")
        print(f"    Herd immunity threshold: {r.herd_immunity_pct:.1f}%")
        print(f"    Vaccine: VE={r.vaccine_efficacy_pct:.0f}% "
              f"coverage={r.vaccine_coverage_pct:.0f}% -> "
              f"Rt={r.Rt_post_vaccine:.2f} "
              f"({'CONTROLLED' if r.herd_achieved else 'NOT controlled'})")
        if not r.herd_achieved:
            print(f"    Need {r.critical_coverage_pct:.0f}% coverage for herd immunity")
        print(f"    Note: {r.notes}")

    # Intervention recommendations
    print(f"\n{'─'*90}")
    print("  INTERVENTION RECOMMENDATIONS")
    print(f"{'─'*90}")

    for r in results:
        strategy_desc = {
            "vaccination_achieved": "✓ CONTROLLED — current vaccination sufficient",
            "vaccination_alone": f"↑ Need {r.critical_coverage_pct:.0f}% coverage (currently {r.vaccine_coverage_pct:.0f}%)",
            "vaccination_plus_npi": f"↑ Max coverage + {r.npi_reduction_needed_pct:.0f}% NPI reduction (masks/distancing)",
            "aggressive_npi_required": f"⚠ Requires lockdown-level NPIs ({r.npi_reduction_needed_pct:.0f}% reduction)",
            "not_controllable": "✗ Not controllable with current tools",
        }
        print(f"  {r.display_name:24s} {strategy_desc.get(r.control_strategy, '?')}")

    # Summary
    n_controlled_now = sum(1 for r in results if r.herd_achieved)
    n_controllable = sum(1 for r in results if r.controllable)
    n_vax_only = sum(1 for r in results if r.control_strategy in ("vaccination_achieved", "vaccination_alone"))
    n_need_npi = sum(1 for r in results if "npi" in r.control_strategy.lower())

    print(f"\n{'='*90}")
    print(f"  Summary:")
    print(f"    Currently controlled:     {n_controlled_now}/{len(results)}")
    print(f"    Controllable with effort: {n_controllable}/{len(results)}")
    print(f"    Vaccination alone:        {n_vax_only}/{len(results)}")
    print(f"    Need NPI support:         {n_need_npi}/{len(results)}")
    print(f"{'='*90}\n")


def save_results(results: List[ScenarioResult], outpath: Path):
    """Save results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "epidemic_lab v0.2",
        "n_scenarios": len(results),
        "n_controlled_now": sum(1 for r in results if r.herd_achieved),
        "n_controllable": sum(1 for r in results if r.controllable),
        "n_vaccination_only": sum(1 for r in results if r.control_strategy in ("vaccination_achieved", "vaccination_alone")),
        "n_need_npi": sum(1 for r in results if "npi" in r.control_strategy.lower()),
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
    parser = argparse.ArgumentParser(description="Epidemic Lab -- disease scenario modeling")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed tool outputs for each scenario")
    parser.add_argument("--extended", "-e", action="store_true",
                        help="Include extended scenarios (pertussis, smallpox, polio, 1918 flu, SARS)")
    args = parser.parse_args()

    scenarios = SCENARIOS.copy()
    if args.extended:
        scenarios.extend(EXTENDED_SCENARIOS)

    print("\n  Analyzing %d disease scenarios..." % len(scenarios))

    results = []
    for scenario in scenarios:
        try:
            result = analyze_scenario(scenario, verbose=args.verbose)
            results.append(result)
        except Exception as e:
            print(f"  ERROR analyzing {scenario.display_name}: {e}")

    if not results:
        print("  No results generated.")
        return

    print_report(results)

    suffix = "_extended" if args.extended else ""
    outpath = _ROOT / "results" / "labs" / "epidemiology" / f"scenario_results{suffix}.json"
    save_results(results, outpath)


if __name__ == "__main__":
    main()
