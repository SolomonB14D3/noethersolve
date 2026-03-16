#!/usr/bin/env python3
"""
Fringe Physics Fact-Checker

Verifies claims from controversial physics areas against fundamental laws:
- Perpetual motion machines
- Free energy devices
- Anti-gravity claims
- Faster-than-light travel
- Over-unity devices

Based on conservation laws from Noether's theorem.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import math


class FringeCategory(Enum):
    """Categories of fringe physics claims."""
    PERPETUAL_MOTION = "perpetual_motion"
    FREE_ENERGY = "free_energy"
    ANTI_GRAVITY = "anti_gravity"
    FTL = "faster_than_light"
    OVER_UNITY = "over_unity"
    COLD_FUSION = "cold_fusion"
    ZERO_POINT = "zero_point_energy"


class ViolationType(Enum):
    """Type of physics violation."""
    FIRST_LAW = "first_law_thermodynamics"  # Energy conservation
    SECOND_LAW = "second_law_thermodynamics"  # Entropy increase
    MOMENTUM = "momentum_conservation"
    ANGULAR_MOMENTUM = "angular_momentum_conservation"
    CAUSALITY = "causality"
    RELATIVITY = "special_relativity"
    GENERAL_RELATIVITY = "general_relativity"  # Gravity/spacetime
    QUANTUM_MECHANICS = "quantum_mechanics"
    NONE = "none"


@dataclass
class FringeClaim:
    """A fringe physics claim to be evaluated."""
    name: str
    description: str
    category: FringeCategory
    claimed_input_energy: Optional[float]  # Joules
    claimed_output_energy: Optional[float]  # Joules
    claimed_efficiency: Optional[float]  # ratio (>1 = over-unity)
    requires_fuel: bool
    claimed_mechanism: str


@dataclass
class FactCheckResult:
    """Result of fact-checking a fringe claim."""
    claim_name: str
    verdict: str  # "IMPOSSIBLE", "IMPLAUSIBLE", "UNVERIFIED", "PLAUSIBLE"
    violations: List[ViolationType]
    explanations: List[str]
    physics_notes: str


# ============================================================================
# Conservation Law Checks
# ============================================================================

def check_energy_conservation(claim: FringeClaim) -> Tuple[bool, str]:
    """
    First Law of Thermodynamics: Energy cannot be created or destroyed.

    ΔE_system + ΔE_surroundings = 0

    For any device: E_out ≤ E_in (equality for ideal reversible)
    """
    if claim.claimed_efficiency is not None and claim.claimed_efficiency > 1.0:
        return False, (f"Over-unity efficiency ({claim.claimed_efficiency:.1%}) "
                      "violates energy conservation. Energy cannot be created.")

    if (claim.claimed_output_energy is not None and
        claim.claimed_input_energy is not None):
        if claim.claimed_output_energy > claim.claimed_input_energy:
            if claim.claimed_input_energy > 0:
                ratio = claim.claimed_output_energy / claim.claimed_input_energy
                return False, (f"Output > Input (ratio {ratio:.2f}) violates "
                              "First Law of Thermodynamics.")
            else:
                return False, ("Output energy with zero input violates "
                              "First Law of Thermodynamics.")

    if not claim.requires_fuel and claim.claimed_output_energy:
        if claim.claimed_output_energy > 0:
            return False, "Perpetual output without fuel input violates energy conservation."

    return True, "Energy conservation not obviously violated."


def check_entropy_increase(claim: FringeClaim) -> Tuple[bool, str]:
    """
    Second Law of Thermodynamics: Entropy of isolated system never decreases.

    ΔS_universe ≥ 0

    Consequences:
    - No perfect heat engine (η < 1)
    - No spontaneous heat flow from cold to hot
    - No perpetual motion of second kind
    """
    if claim.category == FringeCategory.PERPETUAL_MOTION:
        return False, ("Perpetual motion violates Second Law: any real process "
                      "increases entropy, eventually stopping the machine.")

    if claim.claimed_efficiency is not None and claim.claimed_efficiency >= 1.0:
        if "heat" in claim.claimed_mechanism.lower():
            return False, ("Heat engine with efficiency ≥ 100% violates Second Law. "
                          "Carnot limit: η ≤ 1 - T_cold/T_hot.")

    return True, "Second Law not obviously violated."


def check_momentum_conservation(claim: FringeClaim) -> Tuple[bool, str]:
    """
    Momentum conservation: Σp = constant for isolated system.

    Consequences:
    - No "reactionless drive"
    - No thrust without exhaust
    - No EM drive (if isolated)
    """
    mechanism = claim.claimed_mechanism.lower()

    if "reactionless" in mechanism or "no exhaust" in mechanism:
        return False, ("Reactionless drives violate momentum conservation. "
                      "Newton's Third Law: every action has equal opposite reaction.")

    if "em drive" in mechanism or "emdrive" in mechanism:
        return False, ("EM Drive claims violate momentum conservation. "
                      "Closed cavity cannot produce net thrust.")

    return True, "Momentum conservation not obviously violated."


def check_causality(claim: FringeClaim) -> Tuple[bool, str]:
    """
    Causality: Effects cannot precede causes.

    In special relativity: no superluminal signaling.
    Tachyons would allow backward-in-time communication.
    """
    if claim.category == FringeCategory.FTL:
        return False, ("FTL travel/communication violates causality. "
                      "In special relativity, v > c implies time travel paradoxes.")

    mechanism = claim.claimed_mechanism.lower()
    if "tachyon" in mechanism:
        return False, "Tachyons would violate causality by enabling backward signaling."

    return True, "Causality not obviously violated."


def check_relativity(claim: FringeClaim) -> Tuple[bool, str]:
    """
    Special Relativity: c is invariant maximum speed for information/energy.

    E = γmc² means infinite energy needed to reach c for massive particles.
    """
    if claim.category == FringeCategory.FTL:
        return False, ("FTL for massive objects requires infinite energy. "
                      "E → ∞ as v → c (Lorentz factor γ → ∞).")

    mechanism = claim.claimed_mechanism.lower()
    if "warp" in mechanism or "alcubierre" in mechanism:
        # Alcubierre drive requires exotic matter
        return True, ("Alcubierre warp drive is theoretically consistent but "
                     "requires exotic matter with negative energy density.")

    return True, "Special relativity not obviously violated."


def check_zero_point_energy_extraction(claim: FringeClaim) -> Tuple[bool, str]:
    """
    Zero-point energy is real but cannot be extracted for work.

    The quantum vacuum is the ground state - by definition, no lower state exists.
    Casimir effect confirms ZPE but doesn't extract usable energy.
    """
    if claim.category == FringeCategory.ZERO_POINT:
        return False, ("Zero-point energy cannot be extracted for useful work. "
                      "Vacuum is already the ground state - no lower state to transition to. "
                      "Casimir effect is a force, not an energy source.")

    mechanism = claim.claimed_mechanism.lower()
    if "zero point" in mechanism or "vacuum energy" in mechanism:
        if "extract" in mechanism or "harvest" in mechanism:
            return False, ("Cannot extract energy from vacuum ground state. "
                          "This would violate quantum mechanics.")

    return True, "Zero-point energy claims not made."


def check_anti_gravity(claim: FringeClaim) -> Tuple[bool, str, bool]:
    """
    Anti-gravity / gravitational shielding has no theoretical basis.

    In GR, gravity is spacetime curvature - cannot be "shielded".
    Negative mass would be needed for repulsive gravity.

    Returns (passed, message, is_violation) - third value indicates if this
    should count as a physics violation.
    """
    if claim.category == FringeCategory.ANTI_GRAVITY:
        mechanism = claim.claimed_mechanism.lower()

        if "shield" in mechanism:
            return False, ("Gravitational shielding is impossible. Gravity is "
                          "spacetime curvature, not a field that can be blocked."), True

        if "negative mass" not in mechanism and "exotic matter" not in mechanism:
            return False, ("Anti-gravity requires negative mass/energy. "
                          "No such matter has ever been observed."), True

    return True, "Anti-gravity claims not made.", False


# ============================================================================
# Main Fact-Checker
# ============================================================================

def fact_check_fringe_claim(claim: FringeClaim) -> FactCheckResult:
    """
    Comprehensive fact-check of a fringe physics claim.
    """
    violations = []
    explanations = []

    # Run standard checks (return passed, explanation)
    standard_checks = [
        (check_energy_conservation, ViolationType.FIRST_LAW),
        (check_entropy_increase, ViolationType.SECOND_LAW),
        (check_momentum_conservation, ViolationType.MOMENTUM),
        (check_causality, ViolationType.CAUSALITY),
        (check_relativity, ViolationType.RELATIVITY),
        (check_zero_point_energy_extraction, ViolationType.QUANTUM_MECHANICS),
    ]

    for check_func, violation_type in standard_checks:
        passed, explanation = check_func(claim)
        explanations.append(explanation)
        if not passed:
            violations.append(violation_type)

    # Anti-gravity check returns three values
    ag_passed, ag_explanation, ag_is_violation = check_anti_gravity(claim)
    explanations.append(ag_explanation)
    if not ag_passed and ag_is_violation:
        violations.append(ViolationType.GENERAL_RELATIVITY)

    # Determine verdict
    if ViolationType.FIRST_LAW in violations:
        verdict = "IMPOSSIBLE"
        physics_notes = "Violates energy conservation (most fundamental)."
    elif ViolationType.SECOND_LAW in violations:
        verdict = "IMPOSSIBLE"
        physics_notes = "Violates entropy increase (Second Law)."
    elif ViolationType.MOMENTUM in violations:
        verdict = "IMPOSSIBLE"
        physics_notes = "Violates momentum conservation."
    elif ViolationType.CAUSALITY in violations:
        verdict = "IMPOSSIBLE"
        physics_notes = "Violates causality (would enable time travel paradoxes)."
    elif ViolationType.GENERAL_RELATIVITY in violations:
        verdict = "IMPOSSIBLE"
        physics_notes = "Violates general relativity (gravity is spacetime curvature)."
    elif violations:
        verdict = "IMPLAUSIBLE"
        physics_notes = f"Violates: {', '.join(v.value for v in violations)}"
    else:
        verdict = "UNVERIFIED"
        physics_notes = "No obvious conservation law violations, but requires experimental verification."

    return FactCheckResult(
        claim_name=claim.name,
        verdict=verdict,
        violations=violations,
        explanations=[e for e in explanations if "not obviously violated" not in e.lower()],
        physics_notes=physics_notes
    )


def print_fact_check(result: FactCheckResult):
    """Pretty-print fact-check result."""
    print("=" * 70)
    print(f"CLAIM: {result.claim_name}")
    print("=" * 70)
    print(f"VERDICT: {result.verdict}")
    print()
    if result.violations:
        print(f"VIOLATIONS: {', '.join(v.value for v in result.violations)}")
    print()
    print("ANALYSIS:")
    for i, exp in enumerate(result.explanations, 1):
        print(f"  {i}. {exp}")
    print()
    print(f"PHYSICS NOTES: {result.physics_notes}")
    print("=" * 70)


# ============================================================================
# Standard Fringe Claims Database
# ============================================================================

FRINGE_CLAIMS = {
    "perpetual_motion_wheel": FringeClaim(
        name="Overbalanced Wheel (Perpetual Motion)",
        description="Self-rotating wheel using shifting weights",
        category=FringeCategory.PERPETUAL_MOTION,
        claimed_input_energy=0,
        claimed_output_energy=100,
        claimed_efficiency=float('inf'),
        requires_fuel=False,
        claimed_mechanism="Weights shift to maintain imbalance"
    ),

    "free_energy_magnet_motor": FringeClaim(
        name="Magnet Motor Free Energy",
        description="Motor that runs on permanent magnets alone",
        category=FringeCategory.FREE_ENERGY,
        claimed_input_energy=0,
        claimed_output_energy=1000,
        claimed_efficiency=float('inf'),
        requires_fuel=False,
        claimed_mechanism="Permanent magnets provide endless repulsion"
    ),

    "em_drive": FringeClaim(
        name="EM Drive / RF Resonant Cavity Thruster",
        description="Propellantless thruster using microwave cavity",
        category=FringeCategory.ANTI_GRAVITY,
        claimed_input_energy=1000,  # 1 kW
        claimed_output_energy=None,
        claimed_efficiency=None,
        requires_fuel=False,  # No propellant
        claimed_mechanism="EM drive - microwave resonance in closed cavity produces thrust with no exhaust"
    ),

    "zero_point_generator": FringeClaim(
        name="Zero-Point Energy Generator",
        description="Device that extracts energy from quantum vacuum",
        category=FringeCategory.ZERO_POINT,
        claimed_input_energy=0,
        claimed_output_energy=10000,
        claimed_efficiency=float('inf'),
        requires_fuel=False,
        claimed_mechanism="Extract zero point vacuum energy through Casimir-like effect"
    ),

    "ftl_neutrino": FringeClaim(
        name="Faster-Than-Light Neutrinos (OPERA, retracted)",
        description="2011 OPERA experiment claimed FTL neutrinos",
        category=FringeCategory.FTL,
        claimed_input_energy=None,
        claimed_output_energy=None,
        claimed_efficiency=None,
        requires_fuel=False,
        claimed_mechanism="Neutrinos traveling faster than light speed"
    ),

    "podkletnov_gravity_shield": FringeClaim(
        name="Podkletnov Gravity Shielding",
        description="Rotating superconductor reduces gravity above it",
        category=FringeCategory.ANTI_GRAVITY,
        claimed_input_energy=1000,
        claimed_output_energy=None,
        claimed_efficiency=None,
        requires_fuel=True,  # Requires power for superconductor
        claimed_mechanism="Rotating superconductor shields gravitational field"
    ),

    "water_fuel_cell": FringeClaim(
        name="Water Fuel Cell (Stanley Meyer)",
        description="Car runs on water through 'special' electrolysis",
        category=FringeCategory.FREE_ENERGY,
        claimed_input_energy=100,  # Battery to start
        claimed_output_energy=10000,  # Claimed output
        claimed_efficiency=100.0,  # 10000%
        requires_fuel=True,  # Water
        claimed_mechanism="Special electrolysis produces more energy than input"
    ),

    "dean_drive": FringeClaim(
        name="Dean Drive (Reactionless Thruster)",
        description="Mechanical device produces thrust without exhaust",
        category=FringeCategory.ANTI_GRAVITY,
        claimed_input_energy=500,
        claimed_output_energy=None,
        claimed_efficiency=None,
        requires_fuel=False,
        claimed_mechanism="Reactionless drive using counter-rotating masses"
    ),

    "hydrino": FringeClaim(
        name="Hydrino / BlackLight Power",
        description="Hydrogen drops below ground state releasing energy",
        category=FringeCategory.FREE_ENERGY,
        claimed_input_energy=100,
        claimed_output_energy=1000,
        claimed_efficiency=10.0,
        requires_fuel=True,  # Hydrogen + catalyst
        claimed_mechanism="Hydrogen atom drops to fractional quantum states"
    ),

    "n_machine": FringeClaim(
        name="N-Machine / Faraday Homopolar Generator",
        description="Claims over-unity from rotating magnet disk",
        category=FringeCategory.OVER_UNITY,
        claimed_input_energy=1000,
        claimed_output_energy=1200,
        claimed_efficiency=1.2,
        requires_fuel=True,  # Requires spinning
        claimed_mechanism="Homopolar generator extracts energy from space itself"
    ),

    "alcubierre_warp": FringeClaim(
        name="Alcubierre Warp Drive",
        description="FTL via spacetime contraction/expansion",
        category=FringeCategory.FTL,
        claimed_input_energy=None,  # Unknown but huge
        claimed_output_energy=None,
        claimed_efficiency=None,
        requires_fuel=True,  # Exotic matter
        claimed_mechanism="Warp bubble contracts space ahead, expands behind - requires exotic matter with negative energy density"
    ),
}


def main():
    """Run fact-checks on all standard fringe claims."""
    print()
    print("FRINGE PHYSICS FACT-CHECKER")
    print("Based on Conservation Laws from Noether's Theorem")
    print()

    results_summary = []

    for name, claim in FRINGE_CLAIMS.items():
        result = fact_check_fringe_claim(claim)
        print_fact_check(result)
        print()
        results_summary.append((name, result.verdict, len(result.violations)))

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Claim':<40} {'Verdict':<15} {'Violations':<10}")
    print("-" * 70)

    for name, verdict, n_violations in results_summary:
        print(f"{name:<40} {verdict:<15} {n_violations:<10}")

    impossible = sum(1 for _, v, _ in results_summary if v == "IMPOSSIBLE")
    implausible = sum(1 for _, v, _ in results_summary if v == "IMPLAUSIBLE")
    unverified = sum(1 for _, v, _ in results_summary if v == "UNVERIFIED")

    print()
    print(f"IMPOSSIBLE: {impossible}  |  IMPLAUSIBLE: {implausible}  |  UNVERIFIED: {unverified}")


if __name__ == "__main__":
    main()
