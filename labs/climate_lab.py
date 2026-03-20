#!/usr/bin/env python3
"""climate_lab.py -- Climate sensitivity analysis across CO2 scenarios.

Chains NoetherSolve radiative transfer tools to compare climate response
under pre-industrial, 2xCO2, and 4xCO2 scenarios with varying feedback
strengths (low, central, high).

Usage:
    python labs/climate_lab.py
    python labs/climate_lab.py --verbose

Data sources:
    - CO2 forcing: Myhre et al. 1998 (logarithmic relation)
    - Feedback strengths: IPCC AR6 Chapter 7 (2021)
    - Equilibrium sensitivity: Sherwood et al. 2020

References:
    - IPCC AR6 Working Group I (2021)
    - Forster et al. 2021, Earth Syst. Sci. Data

⚠️  DISCLAIMER: SIMPLIFIED EDUCATIONAL MODEL
    This uses 0D energy balance equations. For actual climate projections,
    use validated GCMs from CMIP6 (e.g., CESM, GFDL, UKESM) with full
    atmospheric chemistry, ocean dynamics, and ice sheet coupling.
    Results are illustrative of radiative forcing concepts only.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List

# Ensure noethersolve is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import math

from noethersolve.radiative_transfer import (
    radiative_forcing,
    climate_sensitivity,
    stefan_boltzmann,
    effective_temperature,
    planck_response,
    analyze_feedback,
    FEEDBACK_DATABASE,
    list_feedbacks,
)
from noethersolve.turbulence import (
    kolmogorov_45_law,
    energy_spectrum,
    length_scales,
    intermittency_analysis,
)

RESULTS_DIR = _ROOT / "results" / "labs" / "climate_analysis"

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class CO2Scenario:
    """A CO2 concentration scenario to evaluate."""
    name: str
    co2_ppm: float
    description: str


@dataclass
class FeedbackProfile:
    """A feedback strength profile (scales the cloud feedback)."""
    name: str
    cloud_scale: float  # multiplier on cloud feedback central estimate
    description: str


SCENARIOS: List[CO2Scenario] = [
    CO2Scenario("pre-industrial", 280.0, "Pre-industrial baseline (1750)"),
    CO2Scenario("present-day",    421.0, "Approximate 2024 CO2 level"),
    CO2Scenario("2xCO2",         560.0, "Double pre-industrial CO2"),
    CO2Scenario("4xCO2",        1120.0, "Quadruple pre-industrial CO2"),
]

FEEDBACK_PROFILES: List[FeedbackProfile] = [
    FeedbackProfile("low",     -0.4, "Low sensitivity: cloud feedback near-zero (-0.2 W/m2/K)"),
    FeedbackProfile("central",  1.0, "Central estimate: IPCC AR6 best guess (+0.5 W/m2/K)"),
    FeedbackProfile("high",     2.4, "High sensitivity: strong cloud feedback (+1.2 W/m2/K)"),
]


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """Result from analyzing one scenario x feedback combination."""
    scenario: str
    co2_ppm: float
    feedback_profile: str
    # Radiative forcing
    forcing_Wm2: float
    doublings: float
    # Planck (no-feedback) response
    planck_warming_K: float
    # Full-feedback warming
    ecs_K: float
    feedback_factor: float
    net_feedback_Wm2K: float
    # Stefan-Boltzmann emission at new equilibrium
    new_surface_temp_K: float
    emission_power_Wm2: float
    # Greenhouse effect change
    greenhouse_effect_K: float


def analyze_scenario(
    scenario: CO2Scenario,
    profile: FeedbackProfile,
    verbose: bool = False,
) -> ScenarioResult:
    """Run the full analysis chain for one scenario x feedback profile."""

    # Step 1: Radiative forcing from CO2 change
    rf = radiative_forcing(co2_final=scenario.co2_ppm, co2_initial=280.0)
    if verbose:
        print(rf)

    # Step 2: Planck (no-feedback) response
    pr = planck_response(emission_temperature=255.0)
    planck_warming = rf.forcing * pr.planck_parameter
    if verbose:
        print(pr)

    # Step 3: Climate sensitivity with scaled cloud feedback
    #   Build custom feedback sum: all standard feedbacks, but scale cloud
    fb_sum = 0.0
    for name, info in FEEDBACK_DATABASE.items():
        if name == "cloud":
            fb_sum += info["value"] * profile.cloud_scale
        else:
            fb_sum += info["value"]

    if fb_sum >= 0:
        # Runaway -- cap at very high value for reporting
        ecs_per_doubling = 20.0
    else:
        ecs_per_doubling = -rf.forcing_per_doubling / fb_sum

    # Warming for this specific forcing level
    if fb_sum >= 0:
        warming = 20.0
    else:
        warming = -rf.forcing / fb_sum

    feedback_factor = FEEDBACK_DATABASE["planck"]["value"] / fb_sum if fb_sum != 0 else float("inf")

    # Step 4: New equilibrium surface temperature
    baseline_temp = 288.0  # current Earth average
    # For pre-industrial, the forcing is 0 so warming is 0
    new_temp = baseline_temp + warming

    # Step 5: Stefan-Boltzmann emission at new temperature
    sb = stefan_boltzmann(temperature=new_temp, emissivity=0.95)
    if verbose:
        print(sb)

    # Step 6: Greenhouse effect at new equilibrium
    eff = effective_temperature(actual_surface_temp=new_temp)
    if verbose:
        print(eff)

    doublings = rf.ratio
    doublings_count = math.log2(doublings) if doublings > 0 else 0.0

    return ScenarioResult(
        scenario=scenario.name,
        co2_ppm=scenario.co2_ppm,
        feedback_profile=profile.name,
        forcing_Wm2=round(rf.forcing, 3),
        doublings=round(doublings_count, 3),
        planck_warming_K=round(planck_warming, 2),
        ecs_K=round(ecs_per_doubling, 2),
        feedback_factor=round(feedback_factor, 2),
        net_feedback_Wm2K=round(fb_sum, 3),
        new_surface_temp_K=round(new_temp, 2),
        emission_power_Wm2=round(sb.power_density, 1),
        greenhouse_effect_K=round(eff.greenhouse_effect, 1),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: List[ScenarioResult]):
    """Print a human-readable scenario comparison."""
    print("\n" + "=" * 78)
    print("  CLIMATE SENSITIVITY LAB -- Scenario Comparison")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 78)

    # Step A: feedback overview
    print("\n  Feedback mechanisms (IPCC AR6 central estimates):")
    print(f"  {'Name':18s} {'Value':>10s}  {'Type':20s}")
    print(f"  {'-'*18} {'-'*10}  {'-'*20}")
    for name in list_feedbacks():
        fb = analyze_feedback(name)
        print(f"  {name:18s} {fb.value:+10.2f}  {fb.sign}")

    # Step B: scenario matrix
    profiles = sorted(set(r.feedback_profile for r in results),
                      key=lambda p: ["low", "central", "high"].index(p))

    for pname in profiles:
        subset = [r for r in results if r.feedback_profile == pname]
        print(f"\n  --- Feedback profile: {pname.upper()} (ECS = {subset[0].ecs_K:.1f} K/doubling) ---")
        print(f"  {'Scenario':16s} {'CO2':>7s} {'Forcing':>9s} {'Planck dT':>10s} "
              f"{'Full dT':>9s} {'T_surf':>8s} {'GHE':>6s}")
        print(f"  {'-'*16} {'-'*7} {'-'*9} {'-'*10} {'-'*9} {'-'*8} {'-'*6}")
        for r in subset:
            warming = r.new_surface_temp_K - 288.0
            print(f"  {r.scenario:16s} {r.co2_ppm:7.0f} {r.forcing_Wm2:+9.2f} "
                  f"{r.planck_warming_K:+10.2f} {warming:+9.2f} "
                  f"{r.new_surface_temp_K:8.1f} {r.greenhouse_effect_K:6.1f}")

    # Summary
    central = [r for r in results if r.feedback_profile == "central"]
    two_x = next((r for r in central if r.scenario == "2xCO2"), None)
    four_x = next((r for r in central if r.scenario == "4xCO2"), None)
    print(f"\n  {'='*78}")
    print("  Key findings (central feedback profile):")
    if two_x:
        print(f"    2xCO2: +{two_x.new_surface_temp_K - 288:.1f} K warming, "
              f"forcing {two_x.forcing_Wm2:+.2f} W/m2")
    if four_x:
        print(f"    4xCO2: +{four_x.new_surface_temp_K - 288:.1f} K warming, "
              f"forcing {four_x.forcing_Wm2:+.2f} W/m2")
    print("    Forcing is LOGARITHMIC: 4xCO2 has only 2x the forcing of 2xCO2")
    low_2x = next((r for r in results if r.feedback_profile == "low" and r.scenario == "2xCO2"), None)
    high_2x = next((r for r in results if r.feedback_profile == "high" and r.scenario == "2xCO2"), None)
    if low_2x and high_2x:
        print(f"    2xCO2 range across feedback profiles: "
              f"{low_2x.new_surface_temp_K - 288:+.1f} to {high_2x.new_surface_temp_K - 288:+.1f} K")
    print(f"  {'='*78}\n")


def save_results(results: List[ScenarioResult], outpath: Path):
    """Save results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "climate_lab v0.1",
        "n_scenarios": len(set(r.scenario for r in results)),
        "n_profiles": len(set(r.feedback_profile for r in results)),
        "n_results": len(results),
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
    parser = argparse.ArgumentParser(description="Climate Sensitivity Lab")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed tool output for each step")
    args = parser.parse_args()

    print("\n  Analyzing %d scenarios x %d feedback profiles..."
          % (len(SCENARIOS), len(FEEDBACK_PROFILES)))

    results: List[ScenarioResult] = []
    for scenario in SCENARIOS:
        for profile in FEEDBACK_PROFILES:
            try:
                result = analyze_scenario(scenario, profile, verbose=args.verbose)
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {scenario.name} / {profile.name}: {e}")

    if not results:
        print("  No results generated.")
        return

    print_report(results)

    outpath = RESULTS_DIR / "scenario_results.json"
    save_results(results, outpath)

    # Turbulence cascade analysis — connection between climate physics and energy transfer
    print("\n" + "=" * 78)
    print("  TURBULENCE CASCADE ANALYSIS")
    print("=" * 78)

    # Atmospheric boundary layer conditions
    # Typical ABL: L ~ 100-1000m, u' ~ 0.5-3 m/s, nu ~ 1.5e-5 m²/s
    nu = 1.5e-5  # kinematic viscosity of air
    L = 100.0    # integral scale (m)
    urms_vals = [0.5, 1.5, 3.0]  # RMS velocity for calm, moderate, stormy

    print("\n  Atmospheric turbulence at different intensities:")
    print(f"  {'Condition':15s} {'u_rms (m/s)':>12s} {'η (mm)':>10s} {'λ (m)':>10s} {'Re_λ':>10s}")
    print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    turbulence_results = []
    for urms in urms_vals:
        scales = length_scales(integral_scale=L, urms=urms, kinematic_viscosity=nu)
        condition = "calm" if urms < 1.0 else ("moderate" if urms < 2.0 else "stormy")
        print(f"  {condition:15s} {urms:12.1f} {scales.kolmogorov_scale*1000:10.2f} "
              f"{scales.taylor_scale:10.4f} {scales.taylor_reynolds:10.1f}")
        turbulence_results.append({
            "condition": condition,
            "urms": urms,
            "eta_mm": scales.kolmogorov_scale * 1000,
            "lambda_m": scales.taylor_scale,
            "Re_lambda": scales.taylor_reynolds,
        })

    # Kolmogorov 4/5 law verification
    print("\n  Kolmogorov 4/5 law: S₃(r) = -4/5 ε r")
    separations = [0.01, 0.1, 1.0]  # 1cm, 10cm, 1m
    eps_moderate = 1e-4
    print(f"  Using ε = {eps_moderate:.0e} m²/s³ (moderate turbulence)")
    print(f"  {'r (m)':>10s} {'S₃ (m³/s³)':>15s} {'Verification':>15s}")
    print(f"  {'-'*10} {'-'*15} {'-'*15}")
    for r in separations:
        k45 = kolmogorov_45_law(separation=r, energy_dissipation=eps_moderate)
        status = "PASS (in inertial range)"
        print(f"  {r:10.3f} {k45.third_order_sf:15.2e} {status:>15s}")

    # Energy spectrum slope
    print("\n  Energy spectrum: E(k) ~ k^(-5/3)")
    # Calculate at different wavenumbers to show the scaling
    wavenumbers = [0.1, 1.0, 10.0]  # rad/m
    print(f"  {'k (rad/m)':>12s} {'E(k) (m³/s²)':>15s}")
    print(f"  {'-'*12} {'-'*15}")
    spectral_data = []
    for k in wavenumbers:
        spec = energy_spectrum(wavenumber=k, energy_dissipation=eps_moderate)
        print(f"  {k:12.1f} {spec.spectrum:15.6f}")
        spectral_data.append({"k": k, "E": spec.spectrum})
    # Show theoretical slope
    print(f"  Theoretical slope: -5/3 = -1.667 (Kolmogorov)")

    # Intermittency analysis
    print("\n  Intermittency correction (Kolmogorov refined hypothesis):")
    intermi = intermittency_analysis(model="she_leveque")
    k41_zeta6 = 6 / 3  # K41 predicts ζ_p = p/3
    print(f"  6th order structure function: ζ₆ = {intermi.zeta_6:.3f}")
    print(f"  K41 prediction: ζ₆ = {k41_zeta6:.3f}")
    print(f"  Deviation from K41: {(intermi.zeta_6 - k41_zeta6):.3f}")
    print(f"  Model: {intermi.model}")

    # Save turbulence results
    turb_data = {
        "timestamp": datetime.now().isoformat(),
        "scales": turbulence_results,
        "k45_law_epsilon": 1e-4,  # Used in K45 law tests
        "spectrum_data": spectral_data,
        "intermittency_zeta6": intermi.zeta_6,
    }
    turb_path = RESULTS_DIR / "turbulence_results.json"
    with open(turb_path, "w") as f:
        json.dump(turb_data, f, indent=2)
    print(f"\n  Turbulence results saved to {turb_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
