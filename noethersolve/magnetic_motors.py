#!/usr/bin/env python3
"""
Magnetic Motor / Permanent Magnet Motor Fact-Checker

Analyzes claims about motors that run on permanent magnets alone.

THE PHYSICS:
Permanent magnets do NOT contain energy that can be extracted.
They are analogous to springs - they store potential energy that
came from the magnetization process, not a source of free energy.

KEY PRINCIPLES:
1. Magnets exert FORCES, not ENERGY
2. Any work extracted in one direction requires work input in another
3. Magnetic fields are conservative - closed-loop work = 0
4. Thermal fluctuations (Curie point) can demagnetize, not power

COMMON FALLACIES:
- "Magnets push forever" - Yes, but work = force × displacement
- "Rotating magnets" - Force averages to zero over full rotation
- "Magnetic shielding" - Cannot be done asymmetrically
- "Spin coupling" - Does not violate conservation laws

This module demonstrates WHY magnetic motors cannot work using
fundamental physics from Maxwell's equations and thermodynamics.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math

# Physical constants
MU_0 = 4 * math.pi * 1e-7  # Vacuum permeability (T·m/A)
NEODYMIUM_BR = 1.4  # Remanence of NdFeB magnets (T)
NEODYMIUM_BH_MAX = 450e3  # Max energy product (J/m³)
FERRITE_BR = 0.4  # Remanence of ferrite magnets (T)

# Curie temperatures
NEODYMIUM_CURIE = 310  # °C (loses magnetism above this)
FERRITE_CURIE = 450  # °C


class Verdict(Enum):
    IMPOSSIBLE = "impossible"
    IMPLAUSIBLE = "implausible"
    MISLEADING = "misleading"
    CORRECT = "correct"


class FallacyType(Enum):
    CONSERVATIVE_FIELD = "conservative_field"  # ∮ B·dl = 0 for static field
    ENERGY_STORAGE = "energy_storage"  # Magnets store, don't generate
    SYMMETRIC_FORCES = "symmetric_forces"  # Forces cancel over cycle
    MAGNETIC_SHIELDING = "magnetic_shielding"  # Cannot shield asymmetrically
    PERPETUAL_MOTION = "perpetual_motion"  # Second Law violation


@dataclass
class MagneticMotorClaim:
    """A magnetic motor claim to be analyzed."""
    name: str
    description: str
    magnet_type: str  # "neodymium", "ferrite", "electromagnet"
    claimed_output_watts: Optional[float]
    claimed_efficiency: Optional[float]  # > 100% = over-unity
    mechanism: str
    has_external_input: bool  # True if has any power input


@dataclass
class AnalysisResult:
    """Result of analyzing a magnetic motor claim."""
    claim_name: str
    verdict: Verdict
    fallacies: List[FallacyType]
    explanations: List[str]
    physics_notes: str


def analyze_magnetic_field_work():
    """
    Demonstrate that static magnetic fields do zero net work.

    Maxwell's equations: ∇ × E = -∂B/∂t
    For static fields: ∂B/∂t = 0, so ∇ × E = 0

    This means the electric field is conservative, and no work
    can be extracted from a closed loop.
    """
    return {
        "principle": "Conservative Field",
        "equation": "∮ F·dl = 0 for any closed path",
        "consequence": "No net work can be extracted over a complete cycle",
        "analogy": "Like trying to extract energy from gravity by rolling a ball around a hill - it returns to its starting point with zero net energy gain"
    }


def calculate_magnet_energy(volume_cm3: float, magnet_type: str = "neodymium") -> Dict:
    """
    Calculate the total energy stored in a permanent magnet.

    Energy density = B²/(2μ₀) or equivalently BH_max

    This is the TOTAL energy - once extracted, the magnet is demagnetized.
    """
    volume_m3 = volume_cm3 * 1e-6

    if magnet_type == "neodymium":
        bh_max = NEODYMIUM_BH_MAX
        br = NEODYMIUM_BR
    else:
        bh_max = 40e3  # Ferrite
        br = FERRITE_BR

    # Energy stored in the magnet
    energy_j = bh_max * volume_m3

    # Energy density from field
    field_energy_density = (br ** 2) / (2 * MU_0)

    return {
        "volume_cm3": volume_cm3,
        "magnet_type": magnet_type,
        "stored_energy_J": energy_j,
        "stored_energy_Wh": energy_j / 3600,
        "energy_density_J_m3": bh_max,
        "field_energy_density_J_m3": field_energy_density,
        "note": "This is TOTAL stored energy - extraction would demagnetize"
    }


def check_conservative_field(claim: MagneticMotorClaim) -> Tuple[bool, str, FallacyType]:
    """
    Check if claim violates conservative field principle.

    Static magnetic fields are conservative: ∮ B·dl = 0
    This means no net work can be done over a closed path.
    """
    mechanism = claim.mechanism.lower()

    # Any claim of continuous rotation without input violates this
    if not claim.has_external_input and claim.claimed_output_watts:
        if claim.claimed_output_watts > 0:
            return (False,
                    "Magnetic fields are conservative - no net work over closed loop",
                    FallacyType.CONSERVATIVE_FIELD)

    # Claims about "special arrangements" don't bypass physics
    if "arrangement" in mechanism or "configuration" in mechanism:
        if not claim.has_external_input:
            return (False,
                    "No magnet arrangement can produce net work - field is conservative",
                    FallacyType.CONSERVATIVE_FIELD)

    return (True, "Conservative field not obviously violated", FallacyType.CONSERVATIVE_FIELD)


def check_energy_storage(claim: MagneticMotorClaim) -> Tuple[bool, str, FallacyType]:
    """
    Check if claim confuses energy storage with energy generation.

    Permanent magnets store energy from the magnetization process.
    This energy can be released ONCE by demagnetization.
    """
    mechanism = claim.mechanism.lower()

    # Claims about "magnetic energy" as a source
    if "magnetic energy" in mechanism or "magnet power" in mechanism:
        return (False,
                "Magnets store energy, they don't generate it - like a spring, not a battery",
                FallacyType.ENERGY_STORAGE)

    # Claims about eternal operation
    if "forever" in mechanism or "perpetual" in mechanism or "infinite" in mechanism:
        return (False,
                "Magnets have finite stored energy - continuous extraction would demagnetize them",
                FallacyType.ENERGY_STORAGE)

    return (True, "Energy storage principle not obviously violated", FallacyType.ENERGY_STORAGE)


def check_force_symmetry(claim: MagneticMotorClaim) -> Tuple[bool, str, FallacyType]:
    """
    Check if claim relies on asymmetric forces that don't exist.

    For any arrangement of permanent magnets:
    - Attractive forces in one direction are balanced by repulsive forces
    - Net force over a complete rotation = 0
    - This is Newton's Third Law + conservative fields
    """
    mechanism = claim.mechanism.lower()

    # Claims about special force asymmetry
    asymmetry_words = ["asymmetric", "unbalanced", "net force", "one-way"]
    if any(word in mechanism for word in asymmetry_words):
        return (False,
                "Forces between permanent magnets are symmetric - no net torque over full rotation",
                FallacyType.SYMMETRIC_FORCES)

    # Claims about magnetic "gates" or "shields"
    if "gate" in mechanism or "shield" in mechanism or "block" in mechanism:
        return (False,
                "Magnetic shielding cannot create asymmetric forces - it just redirects them",
                FallacyType.MAGNETIC_SHIELDING)

    return (True, "Force symmetry not obviously violated", FallacyType.SYMMETRIC_FORCES)


def check_thermodynamic_limits(claim: MagneticMotorClaim) -> Tuple[bool, str, FallacyType]:
    """
    Check if claim violates thermodynamic limits.

    Second Law: No process can produce more work than input energy
    Efficiency ≤ 100% for any real device
    """
    if claim.claimed_efficiency is not None:
        if claim.claimed_efficiency > 1.0:
            return (False,
                    f"Over-unity efficiency ({claim.claimed_efficiency:.0%}) violates Second Law",
                    FallacyType.PERPETUAL_MOTION)

    if not claim.has_external_input and claim.claimed_output_watts:
        if claim.claimed_output_watts > 0:
            return (False,
                    "Output without input violates First Law (energy conservation)",
                    FallacyType.PERPETUAL_MOTION)

    return (True, "Thermodynamic limits not obviously violated", FallacyType.PERPETUAL_MOTION)


def analyze_magnetic_motor_claim(claim: MagneticMotorClaim) -> AnalysisResult:
    """
    Comprehensive analysis of a magnetic motor claim.
    """
    fallacies = []
    explanations = []

    # Run all checks
    checks = [
        check_conservative_field,
        check_energy_storage,
        check_force_symmetry,
        check_thermodynamic_limits,
    ]

    for check_func in checks:
        passed, explanation, fallacy_type = check_func(claim)
        if not passed:
            fallacies.append(fallacy_type)
            explanations.append(explanation)

    # Determine verdict
    if FallacyType.PERPETUAL_MOTION in fallacies:
        verdict = Verdict.IMPOSSIBLE
        physics_notes = "Violates laws of thermodynamics"
    elif FallacyType.CONSERVATIVE_FIELD in fallacies:
        verdict = Verdict.IMPOSSIBLE
        physics_notes = "Magnetic fields are conservative - no net work over cycles"
    elif fallacies:
        verdict = Verdict.IMPLAUSIBLE
        physics_notes = f"Contains fallacies: {', '.join(f.value for f in fallacies)}"
    else:
        verdict = Verdict.CORRECT
        physics_notes = "No obvious physics violations"

    return AnalysisResult(
        claim_name=claim.name,
        verdict=verdict,
        fallacies=fallacies,
        explanations=explanations,
        physics_notes=physics_notes
    )


def print_analysis(result: AnalysisResult):
    """Pretty-print analysis result."""
    print("=" * 70)
    print(f"CLAIM: {result.claim_name}")
    print("=" * 70)
    print(f"VERDICT: {result.verdict.value.upper()}")

    if result.fallacies:
        print(f"\nFALLACIES DETECTED:")
        for fallacy in result.fallacies:
            print(f"  - {fallacy.value}")

    if result.explanations:
        print(f"\nEXPLANATIONS:")
        for i, exp in enumerate(result.explanations, 1):
            print(f"  {i}. {exp}")

    print(f"\nPHYSICS: {result.physics_notes}")
    print("=" * 70)


# ============================================================================
# Standard Magnetic Motor Claims
# ============================================================================

MAGNETIC_MOTOR_CLAIMS = {
    "perendev": MagneticMotorClaim(
        name="Perendev Motor (Mike Brady)",
        description="Claimed self-running motor using permanent magnets",
        magnet_type="neodymium",
        claimed_output_watts=20000,  # 20 kW claimed
        claimed_efficiency=float('inf'),  # No input
        mechanism="Special arrangement of magnets creates continuous rotation",
        has_external_input=False
    ),

    "yildiz": MagneticMotorClaim(
        name="Yildiz Motor",
        description="Turkish inventor claimed self-running magnetic motor",
        magnet_type="neodymium",
        claimed_output_watts=1000,
        claimed_efficiency=float('inf'),
        mechanism="Rotating magnetic fields create asymmetric forces",
        has_external_input=False
    ),

    "howard_johnson": MagneticMotorClaim(
        name="Howard Johnson Motor",
        description="US patent 4,151,431 - claimed permanent magnet motor",
        magnet_type="neodymium",
        claimed_output_watts=500,
        claimed_efficiency=float('inf'),
        mechanism="Arcuate magnets create unbalanced magnetic gate",
        has_external_input=False
    ),

    "v_gate": MagneticMotorClaim(
        name="V-Gate Magnet Motor",
        description="Common YouTube 'free energy' demonstration",
        magnet_type="neodymium",
        claimed_output_watts=10,
        claimed_efficiency=float('inf'),
        mechanism="V-shaped magnet arrangement allows continuous motion",
        has_external_input=False
    ),

    "muammer_yildiz": MagneticMotorClaim(
        name="Muammer Yildiz All-Magnet Motor",
        description="Demonstrated at university, claimed 10+ minutes run",
        magnet_type="neodymium",
        claimed_output_watts=1500,
        claimed_efficiency=float('inf'),
        mechanism="Multiple magnetic layers create magnetic energy extraction",
        has_external_input=False
    ),

    "steorn_orbo": MagneticMotorClaim(
        name="Steorn Orbo",
        description="Irish company claimed over-unity magnetic effect",
        magnet_type="neodymium",
        claimed_output_watts=100,
        claimed_efficiency=3.0,  # 300% claimed
        mechanism="Time-varying magnetic fields produce extra energy",
        has_external_input=True  # Had electronic control
    ),

    "minato_wheel": MagneticMotorClaim(
        name="Minato Wheel",
        description="Japanese magnetic motor claimed 330% efficiency",
        magnet_type="neodymium",
        claimed_output_watts=500,
        claimed_efficiency=3.3,
        mechanism="Sticky point effect in magnetic interaction",
        has_external_input=True  # Electric pulse input
    ),

    "bedini_motor": MagneticMotorClaim(
        name="Bedini Motor/Energizer",
        description="John Bedini's claimed over-unity motor",
        magnet_type="neodymium",
        claimed_output_watts=100,
        claimed_efficiency=2.0,  # 200% claimed
        mechanism="Battery charging extracts more energy than input",
        has_external_input=True  # Battery input
    ),

    "bldc_motor": MagneticMotorClaim(
        name="BLDC Motor (Legitimate)",
        description="Standard brushless DC motor with permanent magnets",
        magnet_type="neodymium",
        claimed_output_watts=1000,
        claimed_efficiency=0.90,  # 90% efficiency
        mechanism="Electromagnetic interaction with energized coils",
        has_external_input=True
    ),

    "electric_vehicle": MagneticMotorClaim(
        name="EV Motor (Legitimate)",
        description="Modern electric vehicle motor",
        magnet_type="neodymium",
        claimed_output_watts=200000,  # 200 kW
        claimed_efficiency=0.95,  # 95% efficiency
        mechanism="Permanent magnet synchronous motor with electronic control",
        has_external_input=True
    ),
}


def explain_why_magnetic_motors_cannot_work():
    """Print educational explanation."""
    print("\n" + "=" * 70)
    print("WHY 'FREE ENERGY' MAGNETIC MOTORS CANNOT WORK")
    print("=" * 70)
    print("""
MISCONCEPTION 1: "Magnets push forever, so they have infinite energy"

REALITY: Magnets exert FORCE, not ENERGY.
  - Work = Force × Displacement
  - A magnet pushing on a stationary object does ZERO work
  - Like your table pushing up against a book - no energy transferred

MISCONCEPTION 2: "I've seen videos of them spinning"

REALITY: The videos are misleading:
  - Hidden batteries or motors
  - Initial push provides kinetic energy that slowly dissipates
  - Camera tricks (time-lapse, selective filming)
  - Friction is minimized but will eventually stop it

MISCONCEPTION 3: "Special arrangements create net force"

REALITY: Magnetic fields are CONSERVATIVE.
  - ∮ F·dl = 0 for any closed path
  - Whatever push you get going one way, you get equal resistance going back
  - This is fundamental to Maxwell's equations
  - No arrangement of magnets can change this

THE MATH:

  For a rotating system with angle θ:

  Torque τ(θ) from magnets
  Work over full rotation = ∫₀²π τ(θ) dθ = 0

  This is ALWAYS zero for permanent magnets because:
  τ(θ) = -dU/dθ  (torque is gradient of potential energy)
  ∮ τ dθ = -∮ dU = 0  (potential returns to starting value)

WHAT MAGNETS ARE GOOD FOR:
  - Converting electrical energy to mechanical (motors)
  - Converting mechanical energy to electrical (generators)
  - Holding things in place (magnetic latches)
  - Sensors and switches

  In ALL cases, energy comes from EXTERNAL source (electricity, motion).

THE BOTTOM LINE:
  Permanent magnet motors with no input = Perpetual motion
  Perpetual motion = Impossible (violates thermodynamics)
  Therefore: "Free energy" magnetic motors are impossible
""")


def main():
    """Analyze all magnetic motor claims."""
    print("\nMAGNETIC MOTOR FACT-CHECKER")
    print("Based on Maxwell's equations and thermodynamics\n")

    results_summary = []

    for name, claim in MAGNETIC_MOTOR_CLAIMS.items():
        result = analyze_magnetic_motor_claim(claim)
        print_analysis(result)
        print()
        results_summary.append((name, result.verdict, len(result.fallacies)))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Claim':<25} {'Verdict':<15} {'Fallacies':<10}")
    print("-" * 50)

    for name, verdict, n_fallacies in results_summary:
        print(f"{name:<25} {verdict.value.upper():<15} {n_fallacies:<10}")

    impossible = sum(1 for _, v, _ in results_summary if v == Verdict.IMPOSSIBLE)
    print(f"\nIMPOSSIBLE: {impossible}  |  Others: {len(results_summary) - impossible}")

    # Calculate stored energy in a typical neodymium magnet
    print("\n" + "=" * 70)
    print("MAGNET ENERGY CALCULATION")
    print("=" * 70)

    magnet = calculate_magnet_energy(100, "neodymium")  # 100 cm³ = 10x10x1 cm
    print(f"""
A 100 cm³ neodymium magnet (about 10×10×1 cm):

  Total stored energy: {magnet['stored_energy_J']:.1f} J = {magnet['stored_energy_Wh']:.4f} Wh
  Energy density: {magnet['energy_density_J_m3']/1000:.0f} kJ/m³

  Compare to:
  - AA battery: ~10,000 J (about 200× more)
  - 1 gram of gasoline: ~47,000 J (about 1000× more)

  This energy can only be extracted ONCE by demagnetizing the magnet.
  It cannot be continuously "harvested".
""")

    # Educational explanation
    explain_why_magnetic_motors_cannot_work()


if __name__ == "__main__":
    main()
