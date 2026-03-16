#!/usr/bin/env python3
"""
Unified Physics Fact-Checker

Routes claims to the appropriate specialized module based on category.
All analysis is grounded in conservation laws derived from Noether's theorem.

CONSERVATION LAWS (from symmetries):
- Energy conservation ← time translation symmetry
- Momentum conservation ← space translation symmetry
- Angular momentum conservation ← rotational symmetry
- Charge conservation ← gauge symmetry (U(1))
- Baryon number ← approximate symmetry

MODULES:
- cold_fusion: LENR/cold fusion claims
- fringe_physics: Perpetual motion, free energy, FTL, anti-gravity
- atmospheric_electricity: Atmospheric energy extraction
- water_fuel: Water fuel / HHO / electrolysis
- magnetic_motors: Permanent magnet motors

Usage:
    from noethersolve.fact_checker import fact_check
    result = fact_check("Stanley Meyer's water car runs on water alone")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import re

# Import all specialized modules
from noethersolve.cold_fusion import (
    fact_check_claim as check_cold_fusion,
    STANDARD_CLAIMS as COLD_FUSION_CLAIMS
)
from noethersolve.fringe_physics import (
    fact_check_fringe_claim,
    FringeClaim, FringeCategory,
    FRINGE_CLAIMS
)
from noethersolve.atmospheric_electricity import (
    analyze_atmospheric_claim,
    AtmosphericEnergyDevice,
    ATMOSPHERIC_DEVICES
)
from noethersolve.water_fuel import (
    analyze_water_fuel_claim,
    WaterFuelClaim,
    WATER_FUEL_CLAIMS
)
from noethersolve.magnetic_motors import (
    analyze_magnetic_motor_claim,
    MagneticMotorClaim,
    MAGNETIC_MOTOR_CLAIMS
)


class ClaimCategory(Enum):
    """Categories of physics claims."""
    COLD_FUSION = "cold_fusion"
    PERPETUAL_MOTION = "perpetual_motion"
    FREE_ENERGY = "free_energy"
    ATMOSPHERIC = "atmospheric"
    WATER_FUEL = "water_fuel"
    MAGNETIC_MOTOR = "magnetic_motor"
    FTL = "faster_than_light"
    ANTI_GRAVITY = "anti_gravity"
    UNKNOWN = "unknown"


@dataclass
class UnifiedResult:
    """Unified result format for all fact-checks."""
    claim_text: str
    category: ClaimCategory
    verdict: str
    confidence: float  # 0-1
    violations: List[str]
    explanations: List[str]
    physics_basis: str
    references: List[str]


# Keyword mappings for automatic categorization
CATEGORY_KEYWORDS = {
    ClaimCategory.COLD_FUSION: [
        "cold fusion", "lenr", "low energy nuclear",
        "fleischmann", "pons", "palladium", "deuterium",
        "e-cat", "rossi", "nuclear transmutation"
    ],
    ClaimCategory.WATER_FUEL: [
        "water fuel", "hho", "brown's gas", "browns gas",
        "electrolysis", "stanley meyer", "water car",
        "hydrogen on demand", "water4gas"
    ],
    ClaimCategory.MAGNETIC_MOTOR: [
        "magnetic motor", "magnet motor", "permanent magnet",
        "perendev", "yildiz", "howard johnson", "bedini",
        "steorn", "orbo", "free energy motor"
    ],
    ClaimCategory.ATMOSPHERIC: [
        "atmospheric electricity", "atmospheric energy",
        "tesla tower", "wardenclyffe", "ionosphere",
        "fair weather", "earth battery", "radiant energy"
    ],
    ClaimCategory.PERPETUAL_MOTION: [
        "perpetual motion", "overbalanced wheel",
        "self-running", "runs forever", "no input"
    ],
    ClaimCategory.FREE_ENERGY: [
        "free energy", "over unity", "over-unity",
        "cop >", "zero point", "vacuum energy",
        "unlimited energy", "infinite energy"
    ],
    ClaimCategory.FTL: [
        "faster than light", "ftl", "warp drive",
        "hyperspace", "superluminal", "tachyon"
    ],
    ClaimCategory.ANTI_GRAVITY: [
        "anti-gravity", "antigravity", "gravity shield",
        "em drive", "emdrive", "reactionless",
        "podkletnov", "lifter"
    ]
}


def categorize_claim(claim_text: str) -> ClaimCategory:
    """
    Automatically categorize a claim based on keywords.
    """
    text_lower = claim_text.lower()

    # Check each category's keywords
    matches = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            matches[category] = count

    if not matches:
        return ClaimCategory.UNKNOWN

    # Return category with most keyword matches
    return max(matches, key=matches.get)


def fact_check(claim_text: str, category: Optional[ClaimCategory] = None) -> UnifiedResult:
    """
    Fact-check a physics claim using the appropriate module.

    Args:
        claim_text: The claim to check (natural language)
        category: Optional category override (auto-detected if not provided)

    Returns:
        UnifiedResult with verdict and explanations
    """
    # Auto-categorize if not specified
    if category is None:
        category = categorize_claim(claim_text)

    # Route to appropriate handler
    if category == ClaimCategory.COLD_FUSION:
        return _handle_cold_fusion(claim_text)
    elif category == ClaimCategory.WATER_FUEL:
        return _handle_water_fuel(claim_text)
    elif category == ClaimCategory.MAGNETIC_MOTOR:
        return _handle_magnetic_motor(claim_text)
    elif category == ClaimCategory.ATMOSPHERIC:
        return _handle_atmospheric(claim_text)
    elif category in [ClaimCategory.PERPETUAL_MOTION, ClaimCategory.FREE_ENERGY,
                      ClaimCategory.FTL, ClaimCategory.ANTI_GRAVITY]:
        return _handle_fringe_physics(claim_text, category)
    else:
        return _handle_unknown(claim_text)


def _handle_cold_fusion(claim_text: str) -> UnifiedResult:
    """Handle cold fusion/LENR claims."""
    # Find closest matching known claim
    for name, claim in COLD_FUSION_CLAIMS.items():
        if name.replace("_", " ") in claim_text.lower() or \
           any(word in claim_text.lower() for word in name.split("_")):
            result = check_cold_fusion(claim)
            return UnifiedResult(
                claim_text=claim_text,
                category=ClaimCategory.COLD_FUSION,
                verdict=result["overall_verdict"].value.upper(),
                confidence=0.9,
                violations=[c["name"] for c in result["checks"] if not c["passed"]],
                explanations=[c["message"] for c in result["checks"]],
                physics_basis="Conservation of charge (Z), baryon number (A), energy (Q-value)",
                references=["Noether's theorem", "Nuclear physics constraints"]
            )

    # Generic cold fusion analysis
    return UnifiedResult(
        claim_text=claim_text,
        category=ClaimCategory.COLD_FUSION,
        verdict="IMPLAUSIBLE",
        confidence=0.7,
        violations=["Coulomb barrier", "Missing radiation"],
        explanations=[
            "Cold fusion claims typically fail due to ~400 keV Coulomb barrier",
            "Expected radiation signatures (neutrons, gammas) not observed",
            "Energy claims often violate nuclear physics constraints"
        ],
        physics_basis="Conservation of charge, baryon number, and energy",
        references=["DOE 1989 review", "DOE 2004 review"]
    )


def _handle_water_fuel(claim_text: str) -> UnifiedResult:
    """Handle water fuel / HHO claims."""
    for name, claim in WATER_FUEL_CLAIMS.items():
        if name.replace("_", " ") in claim_text.lower():
            result = analyze_water_fuel_claim(claim)
            return UnifiedResult(
                claim_text=claim_text,
                category=ClaimCategory.WATER_FUEL,
                verdict=result["verdict"].value.upper(),
                confidence=0.95,
                violations=[c["name"] for c in result["checks"] if not c["passed"]],
                explanations=[c["message"] for c in result["checks"]],
                physics_basis="Thermodynamics: electrolysis requires 286 kJ/mol, same as H₂ releases",
                references=["First Law of Thermodynamics", "Hess's Law"]
            )

    return UnifiedResult(
        claim_text=claim_text,
        category=ClaimCategory.WATER_FUEL,
        verdict="IMPOSSIBLE",
        confidence=0.95,
        violations=["Energy conservation"],
        explanations=[
            "Water (H₂O) is the product of hydrogen combustion - the 'ash'",
            "Electrolysis requires exactly as much energy as combustion releases",
            "Round-trip efficiency is 15-48%, never over 100%"
        ],
        physics_basis="First Law of Thermodynamics (energy conservation)",
        references=["Thermochemistry", "Electrochemistry"]
    )


def _handle_magnetic_motor(claim_text: str) -> UnifiedResult:
    """Handle magnetic motor claims."""
    for name, claim in MAGNETIC_MOTOR_CLAIMS.items():
        if name.replace("_", " ") in claim_text.lower():
            result = analyze_magnetic_motor_claim(claim)
            return UnifiedResult(
                claim_text=claim_text,
                category=ClaimCategory.MAGNETIC_MOTOR,
                verdict=result.verdict.value.upper(),
                confidence=0.95,
                violations=[f.value for f in result.fallacies],
                explanations=result.explanations,
                physics_basis="Maxwell's equations: magnetic fields are conservative",
                references=["Conservation laws", "Maxwell's equations"]
            )

    return UnifiedResult(
        claim_text=claim_text,
        category=ClaimCategory.MAGNETIC_MOTOR,
        verdict="IMPOSSIBLE",
        confidence=0.95,
        violations=["Conservative field", "Energy conservation"],
        explanations=[
            "Magnetic fields are conservative: ∮F·dl = 0",
            "No net work can be done over a complete rotation cycle",
            "Magnets store energy like springs - they don't generate it"
        ],
        physics_basis="Conservative field principle, thermodynamics",
        references=["Maxwell's equations", "Noether's theorem"]
    )


def _handle_atmospheric(claim_text: str) -> UnifiedResult:
    """Handle atmospheric electricity claims."""
    for name, device in ATMOSPHERIC_DEVICES.items():
        if name.replace("_", " ") in claim_text.lower():
            result = analyze_atmospheric_claim(device)
            return UnifiedResult(
                claim_text=claim_text,
                category=ClaimCategory.ATMOSPHERIC,
                verdict=result["verdict"],
                confidence=0.85,
                violations=[c["name"] for c in result["checks"] if not c["passed"]],
                explanations=[c["message"] for c in result["checks"]],
                physics_basis="Atmospheric electricity: 130 V/m field, 2 pA/m² current",
                references=result["notes"]
            )

    return UnifiedResult(
        claim_text=claim_text,
        category=ClaimCategory.ATMOSPHERIC,
        verdict="MISLEADING",
        confidence=0.85,
        violations=["Power density"],
        explanations=[
            "Atmospheric electricity IS real (~130 V/m at ground)",
            "But power density is ~10 billion times weaker than solar",
            "Global circuit total: only ~400 kW from ~2000 thunderstorms"
        ],
        physics_basis="Atmospheric physics, global circuit",
        references=["Fair-weather electric field measurements"]
    )


def _handle_fringe_physics(claim_text: str, category: ClaimCategory) -> UnifiedResult:
    """Handle general fringe physics claims."""
    # Map to FringeCategory
    category_map = {
        ClaimCategory.PERPETUAL_MOTION: FringeCategory.PERPETUAL_MOTION,
        ClaimCategory.FREE_ENERGY: FringeCategory.FREE_ENERGY,
        ClaimCategory.FTL: FringeCategory.FTL,
        ClaimCategory.ANTI_GRAVITY: FringeCategory.ANTI_GRAVITY,
    }

    fringe_cat = category_map.get(category, FringeCategory.FREE_ENERGY)

    # Check known claims
    for name, claim in FRINGE_CLAIMS.items():
        if name.replace("_", " ") in claim_text.lower():
            result = fact_check_fringe_claim(claim)
            return UnifiedResult(
                claim_text=claim_text,
                category=category,
                verdict=result.verdict,
                confidence=0.9,
                violations=[v.value for v in result.violations],
                explanations=result.explanations,
                physics_basis=result.physics_notes,
                references=["Conservation laws from Noether's theorem"]
            )

    # Generic analysis based on category
    if category == ClaimCategory.PERPETUAL_MOTION:
        return UnifiedResult(
            claim_text=claim_text,
            category=category,
            verdict="IMPOSSIBLE",
            confidence=0.99,
            violations=["Second Law of Thermodynamics"],
            explanations=["Perpetual motion violates entropy increase (Second Law)"],
            physics_basis="ΔS_universe ≥ 0 for any real process",
            references=["Second Law of Thermodynamics"]
        )

    elif category == ClaimCategory.FREE_ENERGY:
        return UnifiedResult(
            claim_text=claim_text,
            category=category,
            verdict="IMPOSSIBLE",
            confidence=0.99,
            violations=["First Law of Thermodynamics"],
            explanations=["Energy cannot be created, only converted"],
            physics_basis="Energy conservation from time translation symmetry",
            references=["Noether's theorem", "First Law of Thermodynamics"]
        )

    elif category == ClaimCategory.FTL:
        return UnifiedResult(
            claim_text=claim_text,
            category=category,
            verdict="IMPOSSIBLE",
            confidence=0.95,
            violations=["Special Relativity", "Causality"],
            explanations=[
                "FTL for massive objects requires infinite energy",
                "FTL communication would violate causality"
            ],
            physics_basis="Lorentz invariance, c as maximum information speed",
            references=["Special Relativity", "Light cone structure"]
        )

    elif category == ClaimCategory.ANTI_GRAVITY:
        return UnifiedResult(
            claim_text=claim_text,
            category=category,
            verdict="IMPOSSIBLE",
            confidence=0.9,
            violations=["General Relativity"],
            explanations=[
                "Gravity is spacetime curvature, not a field that can be shielded",
                "No observed matter has negative mass"
            ],
            physics_basis="General Relativity: gravity = curved spacetime",
            references=["Einstein field equations"]
        )

    return _handle_unknown(claim_text)


def _handle_unknown(claim_text: str) -> UnifiedResult:
    """Handle unrecognized claims."""
    return UnifiedResult(
        claim_text=claim_text,
        category=ClaimCategory.UNKNOWN,
        verdict="UNVERIFIED",
        confidence=0.3,
        violations=[],
        explanations=["Could not categorize claim for analysis"],
        physics_basis="Unable to determine applicable physics",
        references=[]
    )


def print_result(result: UnifiedResult):
    """Pretty-print a unified result."""
    print("=" * 70)
    print(f"CLAIM: {result.claim_text[:60]}...")
    print("=" * 70)
    print(f"Category: {result.category.value}")
    print(f"Verdict: {result.verdict} (confidence: {result.confidence:.0%})")

    if result.violations:
        print(f"\nViolations:")
        for v in result.violations:
            print(f"  - {v}")

    if result.explanations:
        print(f"\nExplanations:")
        for e in result.explanations:
            print(f"  - {e}")

    print(f"\nPhysics basis: {result.physics_basis}")

    if result.references:
        print(f"\nReferences: {', '.join(result.references)}")

    print("=" * 70)


def main():
    """Demonstrate the unified fact-checker."""
    print("\nUNIFIED PHYSICS FACT-CHECKER")
    print("Based on conservation laws from Noether's theorem\n")

    # Test claims from different categories
    test_claims = [
        "Stanley Meyer claimed his car could run on water using special electrolysis",
        "The Perendev motor runs on permanent magnets alone with no input",
        "Cold fusion produces excess heat in palladium-deuterium systems",
        "A Tesla tower can extract unlimited energy from the atmosphere",
        "The EM drive produces thrust without propellant",
        "Zero point energy can be extracted from the quantum vacuum",
        "This machine achieves over-unity and runs perpetually",
    ]

    for claim in test_claims:
        result = fact_check(claim)
        print_result(result)
        print()

    # Summary statistics
    print("\n" + "=" * 70)
    print("FACT-CHECKING TOOLKIT SUMMARY")
    print("=" * 70)
    print(f"""
Modules available:
  - cold_fusion: {len(COLD_FUSION_CLAIMS)} known claims
  - water_fuel: {len(WATER_FUEL_CLAIMS)} known claims
  - magnetic_motors: {len(MAGNETIC_MOTOR_CLAIMS)} known claims
  - atmospheric: {len(ATMOSPHERIC_DEVICES)} known devices
  - fringe_physics: {len(FRINGE_CLAIMS)} known claims

Total: {sum([len(COLD_FUSION_CLAIMS), len(WATER_FUEL_CLAIMS),
             len(MAGNETIC_MOTOR_CLAIMS), len(ATMOSPHERIC_DEVICES),
             len(FRINGE_CLAIMS)])} analyzed claims/devices

All analysis grounded in:
  - Conservation laws (energy, momentum, charge, baryon number)
  - Noether's theorem (symmetries → conservation)
  - Maxwell's equations (electromagnetism)
  - Thermodynamics (entropy, efficiency limits)
  - Nuclear physics (Coulomb barrier, Q-values)
""")


if __name__ == "__main__":
    main()
