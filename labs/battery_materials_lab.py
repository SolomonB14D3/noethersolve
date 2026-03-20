#!/usr/bin/env python3
"""battery_materials_lab.py -- Autonomous battery degradation analysis lab.

Chains NoetherSolve battery degradation tools to model capacity fade under
various usage patterns and compare chemistry options for different applications.

Usage:
    python labs/battery_materials_lab.py
    python labs/battery_materials_lab.py --verbose
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Literal

# Ensure noethersolve is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from noethersolve.battery_degradation import (
    calc_calendar_aging,
    calc_cycle_aging,
    calc_combined_aging,
    compare_chemistries,
    CHEMISTRY_PARAMS,
)


# ---------------------------------------------------------------------------
# Use case scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class BatteryUseCase:
    """A battery usage scenario to analyze."""
    name: str
    description: str
    # Usage parameters
    time_days: float
    cycles: int
    temperature_C: float = 25.0
    dod: float = 0.8  # Depth of discharge
    soc_storage: float = 0.5  # SOC when not cycling
    c_rate: float = 1.0
    # Application context
    application: str = "general"


USE_CASES: List[BatteryUseCase] = [
    # Electric vehicle scenarios
    BatteryUseCase(
        name="ev_commuter_moderate",
        description="EV daily commuter, moderate climate, 2 years",
        time_days=730,  # 2 years
        cycles=500,     # ~40 miles/day, 13k miles/year
        temperature_C=25.0,
        dod=0.6,        # Moderate DOD
        c_rate=0.5,
        application="EV",
    ),
    BatteryUseCase(
        name="ev_heavy_user",
        description="EV heavy user, 2 years with frequent DC fast charging",
        time_days=730,
        cycles=800,
        temperature_C=30.0,  # Slightly warmer from DCFC
        dod=0.85,
        c_rate=2.0,          # DC fast charging
        application="EV",
    ),
    BatteryUseCase(
        name="ev_hot_climate",
        description="EV in hot climate (Arizona/Texas), 3 years",
        time_days=1095,
        cycles=600,
        temperature_C=40.0,  # Hot climate
        dod=0.7,
        c_rate=1.0,
        application="EV",
    ),

    # Grid storage scenarios
    BatteryUseCase(
        name="grid_daily_cycling",
        description="Grid storage, 1 cycle/day, 5 years",
        time_days=1825,  # 5 years
        cycles=1825,     # Daily cycling
        temperature_C=25.0,
        dod=0.9,         # Deep cycles for max utilization
        c_rate=0.5,
        application="Grid",
    ),
    BatteryUseCase(
        name="grid_peaker",
        description="Grid peaker plant, 2 cycles/day, 3 years",
        time_days=1095,
        cycles=2190,
        temperature_C=30.0,
        dod=0.8,
        c_rate=1.5,
        application="Grid",
    ),

    # Consumer electronics
    BatteryUseCase(
        name="phone_typical",
        description="Smartphone, 1 charge/day, 2 years",
        time_days=730,
        cycles=730,
        temperature_C=28.0,  # Slightly warm in pocket
        dod=0.8,
        soc_storage=0.4,
        c_rate=1.5,          # Fast charging common
        application="Consumer",
    ),
    BatteryUseCase(
        name="laptop_heavy",
        description="Laptop, heavy user, 3 years",
        time_days=1095,
        cycles=600,
        temperature_C=35.0,  # Often hot
        dod=0.7,
        soc_storage=0.8,     # Often plugged in at high SOC
        c_rate=1.0,
        application="Consumer",
    ),

    # Solar storage
    BatteryUseCase(
        name="home_solar_storage",
        description="Home solar battery, daily use, 10 years",
        time_days=3650,
        cycles=3650,
        temperature_C=25.0,  # Indoor installation
        dod=0.7,
        soc_storage=0.5,
        c_rate=0.3,
        application="Solar",
    ),

    # Extreme conditions
    BatteryUseCase(
        name="cold_storage",
        description="Cold climate storage, minimal use, 5 years",
        time_days=1825,
        cycles=100,
        temperature_C=5.0,   # Cold storage
        dod=0.5,
        soc_storage=0.4,     # Optimal storage SOC
        c_rate=0.5,
        application="Storage",
    ),
    BatteryUseCase(
        name="high_temp_abuse",
        description="High temperature abuse test, 1 year",
        time_days=365,
        cycles=300,
        temperature_C=50.0,  # Extreme heat
        dod=0.95,
        c_rate=2.0,
        application="Test",
    ),
]


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------

@dataclass
class BatteryAnalysisResult:
    """Result of analyzing a battery use case."""
    name: str
    description: str
    application: str
    chemistry: str
    # Degradation results
    calendar_loss_pct: float
    cycle_loss_pct: float
    total_loss_pct: float
    remaining_capacity_pct: float
    # Analysis
    dominant_mechanism: str
    calendar_fraction: float
    time_to_eol_years: float
    # Verdict
    verdict: str  # EXCELLENT, GOOD, ACCEPTABLE, POOR, FAILED
    notes: List[str] = field(default_factory=list)


def analyze_use_case(
    use_case: BatteryUseCase,
    chemistry: Literal["NMC", "LFP", "NCA"],
    verbose: bool = False,
) -> BatteryAnalysisResult:
    """Run battery degradation analysis for one use case and chemistry."""

    report = calc_combined_aging(
        chemistry=chemistry,
        time_days=use_case.time_days,
        cycles=use_case.cycles,
        temperature_C=use_case.temperature_C,
        dod=use_case.dod,
        soc_storage=use_case.soc_storage,
        c_rate=use_case.c_rate,
    )

    if verbose:
        print(report)

    # Determine verdict based on remaining capacity
    remaining = report.remaining_capacity_percent
    if remaining >= 90:
        verdict = "EXCELLENT"
    elif remaining >= 80:
        verdict = "GOOD"
    elif remaining >= 70:
        verdict = "ACCEPTABLE"
    elif remaining >= 60:
        verdict = "POOR"
    else:
        verdict = "FAILED"

    # Time to EOL in years
    time_to_eol_years = report.time_to_eol / 365.25 if report.time_to_eol < float('inf') else 99

    notes = list(report.notes)
    params = CHEMISTRY_PARAMS[chemistry]
    notes.append(f"Chemistry dominant mechanism: {params['dominant_mechanism']}")

    return BatteryAnalysisResult(
        name=use_case.name,
        description=use_case.description,
        application=use_case.application,
        chemistry=chemistry,
        calendar_loss_pct=report.calendar_loss_percent,
        cycle_loss_pct=report.cycle_loss_percent,
        total_loss_pct=report.total_loss_percent,
        remaining_capacity_pct=report.remaining_capacity_percent,
        dominant_mechanism=report.dominant_mechanism,
        calendar_fraction=report.calendar_fraction,
        time_to_eol_years=time_to_eol_years,
        verdict=verdict,
        notes=notes,
    )


def find_best_chemistry(use_case: BatteryUseCase, verbose: bool = False) -> tuple:
    """Find the best chemistry for a use case, return (best_chem, all_results)."""
    results = []
    for chem in ["NMC", "LFP", "NCA"]:
        result = analyze_use_case(use_case, chem, verbose=verbose)
        results.append(result)

    # Sort by remaining capacity (best first)
    results.sort(key=lambda r: r.remaining_capacity_pct, reverse=True)
    best = results[0]

    return best.chemistry, results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(all_results: List[tuple]):
    """Print a human-readable battery analysis report."""
    print("\n" + "=" * 72)
    print("  BATTERY MATERIALS LAB -- Degradation Analysis Report")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)

    # Group by application
    by_app = {}
    for use_case, best_chem, results in all_results:
        if use_case.application not in by_app:
            by_app[use_case.application] = []
        by_app[use_case.application].append((use_case, best_chem, results))

    for app, app_results in by_app.items():
        print(f"\n  --- {app.upper()} APPLICATIONS ---")

        for use_case, best_chem, results in app_results:
            best_result = results[0]  # Already sorted by remaining capacity
            verdict_tag = {
                "EXCELLENT": "[EXCELLENT]",
                "GOOD": "[GOOD]",
                "ACCEPTABLE": "[OK]",
                "POOR": "[POOR]",
                "FAILED": "[FAILED]",
            }[best_result.verdict]

            print(f"\n  {use_case.name} {verdict_tag}")
            print(f"       {use_case.description}")
            print(f"       Best chemistry: {best_chem}")

            # Show comparison
            print(f"       Chemistry comparison:")
            for r in results:
                print(f"         {r.chemistry}: {r.remaining_capacity_pct:.1f}% remaining "
                      f"(cal {r.calendar_loss_pct:.1f}% + cyc {r.cycle_loss_pct:.1f}%)")

            print(f"       Dominant: {best_result.dominant_mechanism}")
            print(f"       EOL estimate: {best_result.time_to_eol_years:.1f} years")

            if best_result.notes:
                for note in best_result.notes[:2]:
                    print(f"       • {note}")

    # Summary statistics
    total_cases = len(all_results)
    by_best_chem = {"NMC": 0, "LFP": 0, "NCA": 0}
    for _, best_chem, _ in all_results:
        by_best_chem[best_chem] += 1

    print(f"\n  {'='*72}")
    print(f"  Summary: {total_cases} use cases analyzed")
    print(f"    Best chemistry by use case:")
    for chem, count in sorted(by_best_chem.items(), key=lambda x: -x[1]):
        print(f"      {chem}: {count} ({count/total_cases*100:.0f}%)")
    print(f"  {'='*72}\n")


def save_results(all_results: List[tuple], outpath: Path):
    """Save results to JSON."""
    results_data = []
    for use_case, best_chem, results in all_results:
        results_data.append({
            "use_case": use_case.name,
            "description": use_case.description,
            "application": use_case.application,
            "best_chemistry": best_chem,
            "chemistry_results": [asdict(r) for r in results],
        })

    data = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "battery_materials_lab v0.1",
        "n_use_cases": len(all_results),
        "chemistry_wins": {
            "NMC": sum(1 for _, c, _ in all_results if c == "NMC"),
            "LFP": sum(1 for _, c, _ in all_results if c == "LFP"),
            "NCA": sum(1 for _, c, _ in all_results if c == "NCA"),
        },
        "results": results_data,
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
    parser = argparse.ArgumentParser(description="Battery Materials Lab -- degradation analysis pipeline")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed reports for each analysis")
    args = parser.parse_args()

    print("\n  Analyzing %d battery use cases across 3 chemistries..." % len(USE_CASES))

    all_results = []
    for use_case in USE_CASES:
        try:
            best_chem, results = find_best_chemistry(use_case, verbose=args.verbose)
            all_results.append((use_case, best_chem, results))
        except Exception as e:
            print(f"  ERROR analyzing {use_case.name}: {e}")

    if not all_results:
        print("  No results generated.")
        return

    print_report(all_results)

    outpath = _ROOT / "results" / "labs" / "battery_materials" / "analysis_results.json"
    save_results(all_results, outpath)


if __name__ == "__main__":
    main()
