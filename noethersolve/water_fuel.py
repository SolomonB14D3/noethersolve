#!/usr/bin/env python3
"""
Water Fuel / HHO / Electrolysis Fact-Checker

Analyzes claims about running vehicles on water through electrolysis.

THE PHYSICS:
- Electrolysis: 2H₂O + energy → 2H₂ + O₂
- Combustion: 2H₂ + O₂ → 2H₂O + energy
- These are INVERSE reactions - you cannot get more energy out than you put in

THERMODYNAMICS:
- Electrolysis requires: 286 kJ/mol H₂O (at 25°C)
- Hydrogen combustion releases: 286 kJ/mol H₂O
- Efficiency of electrolysis: ~60-80%
- Efficiency of combustion: ~25-40% (engine)
- Net efficiency: 15-32% (ALWAYS < 100%)

Common claims debunked:
- "HHO gas" / "Brown's gas" - just hydrogen + oxygen mixture
- "Water as fuel" - water is the ASH, not the fuel
- "Over-unity water splitter" - violates thermodynamics
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math

# Physical constants
WATER_MOLAR_MASS = 18.015  # g/mol
H2_MOLAR_MASS = 2.016  # g/mol
WATER_DENSITY = 1000  # g/L

# Thermodynamic data (at 25°C, 1 atm)
ELECTROLYSIS_ENTHALPY = 286  # kJ/mol H₂O (minimum energy required)
ELECTROLYSIS_GIBBS = 237  # kJ/mol H₂O (reversible work)
H2_HHV = 286  # kJ/mol (higher heating value)
H2_LHV = 242  # kJ/mol (lower heating value)

# Standard electrode potential
E_CELL_STANDARD = 1.23  # V (theoretical minimum for electrolysis)
E_CELL_PRACTICAL = 1.8  # V (practical with overpotential)

# Energy content comparisons
GASOLINE_ENERGY_DENSITY = 34.2  # MJ/L
H2_ENERGY_DENSITY_GAS = 0.01  # MJ/L at STP (very low!)
H2_ENERGY_DENSITY_LIQUID = 8.5  # MJ/L (requires -253°C)


class Verdict(Enum):
    IMPOSSIBLE = "impossible"
    IMPLAUSIBLE = "implausible"
    MISLEADING = "misleading"
    CORRECT = "correct"


@dataclass
class WaterFuelClaim:
    """A water fuel claim to be analyzed."""
    name: str
    description: str
    claimed_water_input_L: float
    claimed_electricity_input_kWh: float
    claimed_energy_output_kWh: float
    claimed_efficiency: Optional[float]  # output/input
    mechanism: str


def calculate_electrolysis_energy(water_liters: float) -> Dict:
    """
    Calculate energy required for electrolysis of given water volume.
    """
    water_mass_g = water_liters * WATER_DENSITY
    water_moles = water_mass_g / WATER_MOLAR_MASS

    # Theoretical minimum energy (Gibbs free energy)
    min_energy_kJ = water_moles * ELECTROLYSIS_GIBBS
    min_energy_kWh = min_energy_kJ / 3600

    # Practical energy (with overpotential losses)
    practical_energy_kJ = water_moles * ELECTROLYSIS_ENTHALPY / 0.7  # 70% efficiency
    practical_energy_kWh = practical_energy_kJ / 3600

    # Hydrogen produced
    h2_moles = water_moles  # 1 mol H₂O → 1 mol H₂
    h2_mass_g = h2_moles * H2_MOLAR_MASS
    h2_volume_L_stp = h2_moles * 22.4  # Ideal gas at STP

    # Energy content of produced hydrogen
    h2_energy_hhv_kJ = h2_moles * H2_HHV
    h2_energy_hhv_kWh = h2_energy_hhv_kJ / 3600

    return {
        "water_liters": water_liters,
        "water_moles": water_moles,
        "min_electrolysis_energy_kWh": min_energy_kWh,
        "practical_electrolysis_energy_kWh": practical_energy_kWh,
        "h2_produced_moles": h2_moles,
        "h2_produced_mass_g": h2_mass_g,
        "h2_produced_volume_L_stp": h2_volume_L_stp,
        "h2_energy_content_kWh": h2_energy_hhv_kWh,
    }


def calculate_round_trip_efficiency(
    electrolysis_efficiency: float = 0.70,
    fuel_cell_efficiency: float = 0.50,
    engine_efficiency: float = 0.25
) -> Dict:
    """
    Calculate round-trip efficiency for water → H₂ → energy.
    """
    # Fuel cell path
    fuel_cell_round_trip = electrolysis_efficiency * fuel_cell_efficiency

    # Combustion engine path
    engine_round_trip = electrolysis_efficiency * engine_efficiency

    return {
        "electrolysis_efficiency": electrolysis_efficiency,
        "fuel_cell_efficiency": fuel_cell_efficiency,
        "engine_efficiency": engine_efficiency,
        "fuel_cell_round_trip": fuel_cell_round_trip,
        "engine_round_trip": engine_round_trip,
        "best_case_round_trip": fuel_cell_round_trip,
        "worst_case_round_trip": engine_round_trip,
    }


def analyze_water_fuel_claim(claim: WaterFuelClaim) -> Dict:
    """
    Analyze a water fuel claim against thermodynamics.
    """
    results = {
        "claim": claim.name,
        "checks": [],
        "verdict": Verdict.CORRECT,
        "physics": {},
        "notes": []
    }

    # Calculate the physics
    if claim.claimed_water_input_L > 0:
        physics = calculate_electrolysis_energy(claim.claimed_water_input_L)
        results["physics"] = physics

        # Check 1: Energy input vs minimum required
        # Only check if electrolysis is claimed (energy output > 0)
        if (claim.claimed_energy_output_kWh > 0 and
            claim.claimed_electricity_input_kWh < physics["min_electrolysis_energy_kWh"] * 0.9):
            results["checks"].append({
                "name": "Electrolysis energy",
                "passed": False,
                "message": (f"Claimed input {claim.claimed_electricity_input_kWh:.3f} kWh "
                           f"< minimum required {physics['min_electrolysis_energy_kWh']:.3f} kWh")
            })
            results["verdict"] = Verdict.IMPOSSIBLE
            results["notes"].append("Violates First Law of Thermodynamics")

        # Check 2: Energy output vs hydrogen energy content
        if claim.claimed_energy_output_kWh > physics["h2_energy_content_kWh"] * 1.1:
            results["checks"].append({
                "name": "Energy output",
                "passed": False,
                "message": (f"Claimed output {claim.claimed_energy_output_kWh:.3f} kWh "
                           f"> H₂ energy content {physics['h2_energy_content_kWh']:.3f} kWh")
            })
            results["verdict"] = Verdict.IMPOSSIBLE
            results["notes"].append("Cannot extract more energy than hydrogen contains")

    # Check 3: Over-unity claims
    if claim.claimed_efficiency is not None:
        if claim.claimed_efficiency > 1.0:
            results["checks"].append({
                "name": "Over-unity efficiency",
                "passed": False,
                "message": f"Claimed efficiency {claim.claimed_efficiency:.1%} > 100%"
            })
            results["verdict"] = Verdict.IMPOSSIBLE
            results["notes"].append("Over-unity violates thermodynamics")
        elif claim.claimed_efficiency > 0.5:
            results["checks"].append({
                "name": "High efficiency",
                "passed": False,
                "message": f"Claimed efficiency {claim.claimed_efficiency:.1%} exceeds practical limits (~35%)"
            })
            if results["verdict"] != Verdict.IMPOSSIBLE:
                results["verdict"] = Verdict.IMPLAUSIBLE

    # Check 4: Mechanism claims
    mechanism = claim.mechanism.lower()
    if "free energy" in mechanism or "over unity" in mechanism:
        results["checks"].append({
            "name": "Free energy claim",
            "passed": False,
            "message": "Claims free energy from water - thermodynamically impossible"
        })
        results["verdict"] = Verdict.IMPOSSIBLE

    if "resonance" in mechanism or "frequency" in mechanism:
        results["checks"].append({
            "name": "Resonance claim",
            "passed": False,
            "message": "Resonant frequencies do not reduce electrolysis energy requirements"
        })
        if results["verdict"] != Verdict.IMPOSSIBLE:
            results["verdict"] = Verdict.MISLEADING

    if "hho" in mechanism or "brown" in mechanism:
        results["checks"].append({
            "name": "HHO/Brown's gas",
            "passed": True,
            "message": "HHO is just H₂+O₂ mixture - no special properties"
        })
        results["notes"].append("'Brown's gas' has no special properties beyond H₂+O₂")

    # Check 5: Net energy output
    if claim.claimed_energy_output_kWh > claim.claimed_electricity_input_kWh:
        results["checks"].append({
            "name": "Net energy",
            "passed": False,
            "message": (f"Output ({claim.claimed_energy_output_kWh:.3f} kWh) > "
                       f"Input ({claim.claimed_electricity_input_kWh:.3f} kWh)")
        })
        results["verdict"] = Verdict.IMPOSSIBLE
        results["notes"].append("Net energy gain violates First Law")

    # If all checks pass
    if not results["checks"]:
        results["checks"].append({
            "name": "Basic physics",
            "passed": True,
            "message": "No obvious thermodynamic violations"
        })

    return results


def print_analysis(results: Dict):
    """Pretty-print analysis results."""
    print("=" * 70)
    print(f"CLAIM: {results['claim']}")
    print("=" * 70)

    if results["physics"]:
        p = results["physics"]
        print(f"\nPHYSICS (for {p['water_liters']:.2f} L water):")
        print(f"  Minimum electrolysis energy: {p['min_electrolysis_energy_kWh']:.4f} kWh")
        print(f"  Practical electrolysis energy: {p['practical_electrolysis_energy_kWh']:.4f} kWh")
        print(f"  H₂ produced: {p['h2_produced_mass_g']:.1f} g ({p['h2_produced_volume_L_stp']:.1f} L at STP)")
        print(f"  H₂ energy content: {p['h2_energy_content_kWh']:.4f} kWh")

    print(f"\nCHECKS:")
    for check in results["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['name']}: {check['message']}")

    print(f"\nVERDICT: {results['verdict'].value.upper()}")

    if results["notes"]:
        print(f"\nNOTES:")
        for note in results["notes"]:
            print(f"  - {note}")

    print("=" * 70)


# ============================================================================
# Standard Water Fuel Claims
# ============================================================================

WATER_FUEL_CLAIMS = {
    "stanley_meyer": WaterFuelClaim(
        name="Stanley Meyer Water Fuel Cell",
        description="Claimed to run car on water using special electrolysis",
        claimed_water_input_L=1.0,
        claimed_electricity_input_kWh=0.1,  # Claimed low input
        claimed_energy_output_kWh=5.0,      # Claimed high output
        claimed_efficiency=50.0,            # 5000%
        mechanism="Resonant frequency electrolysis produces more energy than input"
    ),

    "hho_generator": WaterFuelClaim(
        name="HHO Generator / Brown's Gas",
        description="Electrolysis to produce 'HHO' for fuel supplement",
        claimed_water_input_L=0.5,
        claimed_electricity_input_kWh=0.5,
        claimed_energy_output_kWh=1.0,      # Claims 200% efficiency
        claimed_efficiency=2.0,
        mechanism="HHO gas has special combustion properties"
    ),

    "legitimate_electrolysis": WaterFuelClaim(
        name="Standard Industrial Electrolysis",
        description="Real hydrogen production via electrolysis",
        claimed_water_input_L=1.0,
        claimed_electricity_input_kWh=5.0,  # Realistic input
        claimed_energy_output_kWh=3.5,      # Realistic output (70% efficiency)
        claimed_efficiency=0.70,
        mechanism="Standard PEM or alkaline electrolysis"
    ),

    "fuel_cell_vehicle": WaterFuelClaim(
        name="Hydrogen Fuel Cell Vehicle",
        description="Legitimate fuel cell technology",
        claimed_water_input_L=1.0,          # Water input for electrolysis
        claimed_electricity_input_kWh=5.0,  # Grid electricity
        claimed_energy_output_kWh=1.75,     # After round-trip losses
        claimed_efficiency=0.35,            # Realistic round-trip
        mechanism="Electrolysis → H₂ storage → Fuel cell → Electric motor"
    ),

    "water_injection": WaterFuelClaim(
        name="Water Injection (Legitimate)",
        description="Water injection for engine cooling/detonation suppression",
        claimed_water_input_L=0.1,
        claimed_electricity_input_kWh=0.0,  # No electrolysis
        claimed_energy_output_kWh=0.0,      # No energy from water
        claimed_efficiency=None,
        mechanism="Water cools intake charge, allows higher compression - does NOT provide energy"
    ),

    "joe_cell": WaterFuelClaim(
        name="Joe Cell (Orgone Generator)",
        description="Claims to run engine on 'orgone energy' from water",
        claimed_water_input_L=1.0,
        claimed_electricity_input_kWh=0.01,
        claimed_energy_output_kWh=10.0,
        claimed_efficiency=1000.0,
        mechanism="Free energy from orgone/life force in water"
    ),
}


def explain_why_water_cannot_be_fuel():
    """Print educational explanation."""
    print("\n" + "=" * 70)
    print("WHY WATER CANNOT BE A FUEL")
    print("=" * 70)
    print("""
CHEMISTRY:
  Fuel + Oxidizer → Combustion products + Energy

  For hydrogen:
    2H₂ + O₂ → 2H₂O + 286 kJ/mol

  Water (H₂O) is the PRODUCT of hydrogen combustion.
  It is the "ash" - already fully oxidized.

ANALOGY:
  Asking "can we burn water?" is like asking "can we burn wood ash?"
  The energy has already been released!

WHAT ABOUT ELECTROLYSIS?
  2H₂O + 286 kJ/mol → 2H₂ + O₂

  This is the REVERSE reaction - it requires energy INPUT.

  Energy cycle:
    Electricity → Electrolysis → H₂ + O₂ → Combustion → Back to H₂O

  Each step has losses. You ALWAYS get less energy out than you put in.

EFFICIENCY:
  - Electrolysis: 60-80% efficient
  - Fuel cell: 40-60% efficient
  - Engine: 20-40% efficient

  Best case (electrolysis + fuel cell): 0.8 × 0.6 = 48%
  Worst case (electrolysis + engine): 0.6 × 0.25 = 15%

  You get back 15-48% of the energy you put in. NEVER more than 100%.

THE CLAIMS:
  "Special frequencies" - Do not change thermodynamics
  "Brown's gas" - Just H₂ + O₂, no special properties
  "Over-unity" - Violates conservation of energy
  "Free energy from water" - Water has no extractable energy

LEGITIMATE USES OF HYDROGEN:
  - Energy storage (renewable electricity → H₂ → fuel cells)
  - Industrial processes (steel, ammonia)
  - Rocket fuel (with separate oxygen supply)

  But in ALL cases, hydrogen must be produced with external energy.
""")


def main():
    """Analyze all water fuel claims."""
    print("\nWATER FUEL / HHO FACT-CHECKER")
    print("Based on thermodynamics and electrochemistry\n")

    for name, claim in WATER_FUEL_CLAIMS.items():
        results = analyze_water_fuel_claim(claim)
        print_analysis(results)
        print()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Claim':<35} {'Verdict':<15}")
    print("-" * 50)

    verdicts = {}
    for name, claim in WATER_FUEL_CLAIMS.items():
        results = analyze_water_fuel_claim(claim)
        verdicts[name] = results["verdict"]
        print(f"{name:<35} {results['verdict'].value.upper():<15}")

    impossible = sum(1 for v in verdicts.values() if v == Verdict.IMPOSSIBLE)
    print(f"\nIMPOSSIBLE: {impossible}  |  Others: {len(verdicts) - impossible}")

    # Educational explanation
    explain_why_water_cannot_be_fuel()

    # Round-trip efficiency calculation
    print("\n" + "=" * 70)
    print("ROUND-TRIP EFFICIENCY CALCULATION")
    print("=" * 70)

    efficiencies = calculate_round_trip_efficiency()
    print(f"""
Starting with 100 kWh of electricity:

Path 1: Electrolysis → Storage → Fuel Cell
  Electrolysis (70%): 100 kWh → 70 kWh in H₂
  Fuel cell (50%): 70 kWh → 35 kWh electricity
  Round-trip: {efficiencies['fuel_cell_round_trip']:.0%}

Path 2: Electrolysis → Storage → Engine
  Electrolysis (70%): 100 kWh → 70 kWh in H₂
  Engine (25%): 70 kWh → 17.5 kWh mechanical
  Round-trip: {efficiencies['engine_round_trip']:.0%}

CONCLUSION: You get back 17-35% of input energy. Never more than 100%.
""")


if __name__ == "__main__":
    main()
