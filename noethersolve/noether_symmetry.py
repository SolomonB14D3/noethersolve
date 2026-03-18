"""Noether's Theorem — Bidirectional Symmetry ↔ Conservation Law Mapping.

Emmy Noether proved (1915): Every continuous symmetry corresponds to
a conserved quantity. This module provides a verified bidirectional
mapping between symmetries and conservation laws.

Models often get the direction wrong:
- Time translation → Energy (✓ models know this)
- Spatial translation → Momentum (✗ models often confuse with energy)
- Rotation → Angular momentum (✓ mostly correct)
- Gauge → Charge (✗ models struggle with this)

This tool provides verified mappings in both directions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SymmetryType(Enum):
    SPACETIME = "spacetime"
    INTERNAL = "internal"
    DISCRETE = "discrete"


class ConservationType(Enum):
    EXTENSIVE = "extensive"  # Scales with system size
    INTENSIVE = "intensive"  # Independent of size
    TOPOLOGICAL = "topological"  # Integer-valued


@dataclass
class NoetherPair:
    """A symmetry-conservation law pair from Noether's theorem."""
    symmetry: str
    symmetry_description: str
    symmetry_type: SymmetryType
    conservation_law: str
    conserved_quantity: str
    quantity_symbol: str
    conservation_type: ConservationType
    generator: str  # Mathematical generator of the symmetry
    common_errors: List[str]
    examples: List[str]
    notes: str


# The verified Noether pairs database
NOETHER_PAIRS: Dict[str, NoetherPair] = {
    "time_translation": NoetherPair(
        symmetry="Time translation",
        symmetry_description="Physics unchanged under t → t + Δt",
        symmetry_type=SymmetryType.SPACETIME,
        conservation_law="Energy conservation",
        conserved_quantity="Total energy",
        quantity_symbol="E or H",
        conservation_type=ConservationType.EXTENSIVE,
        generator="∂/∂t (Hamiltonian)",
        common_errors=[
            "Confusing with momentum conservation",
            "Assuming always valid (fails in cosmology with expansion)",
        ],
        examples=[
            "Pendulum: E = ½mv² + mgh = const",
            "Particle in static potential: H = T + V = const",
        ],
        notes="Breaks in time-dependent potentials or expanding spacetime.",
    ),
    "spatial_translation": NoetherPair(
        symmetry="Spatial translation",
        symmetry_description="Physics unchanged under x → x + Δx",
        symmetry_type=SymmetryType.SPACETIME,
        conservation_law="Momentum conservation",
        conserved_quantity="Total linear momentum",
        quantity_symbol="p or P",
        conservation_type=ConservationType.EXTENSIVE,
        generator="∂/∂x (momentum operator)",
        common_errors=[
            "Confusing with energy conservation",
            "Models often say 'translation → energy'",
            "Forgetting to specify which component if symmetry is partial",
        ],
        examples=[
            "Free particle: p = mv = const",
            "Collision: p_before = p_after",
            "Translating x but not y: p_x conserved but not p_y",
        ],
        notes="Applies per direction. Broken by external potentials.",
    ),
    "rotation": NoetherPair(
        symmetry="Rotational symmetry",
        symmetry_description="Physics unchanged under rotation",
        symmetry_type=SymmetryType.SPACETIME,
        conservation_law="Angular momentum conservation",
        conserved_quantity="Total angular momentum",
        quantity_symbol="L or J",
        conservation_type=ConservationType.EXTENSIVE,
        generator="r × ∇ (angular momentum operator)",
        common_errors=[
            "Forgetting to specify axis for partial symmetry",
            "Confusing orbital (L) and spin (S) contributions",
        ],
        examples=[
            "Central force: L = r × p = const",
            "Kepler orbits: L preserved, defines orbital plane",
        ],
        notes="Full SO(3) symmetry → all components conserved.",
    ),
    "boost": NoetherPair(
        symmetry="Lorentz boost",
        symmetry_description="Physics unchanged under velocity change",
        symmetry_type=SymmetryType.SPACETIME,
        conservation_law="Center of mass motion",
        conserved_quantity="Center of mass velocity × time - position",
        quantity_symbol="K or N",
        conservation_type=ConservationType.EXTENSIVE,
        generator="t∂/∂x - x∂/∂t",
        common_errors=[
            "Thinking this gives a new conserved quantity",
            "Confusing with momentum",
        ],
        examples=[
            "Free particle: x_cm - v_cm × t = const",
        ],
        notes="Less commonly mentioned than other conservation laws.",
    ),
    "u1_gauge": NoetherPair(
        symmetry="U(1) gauge symmetry",
        symmetry_description="Physics unchanged under ψ → e^{iα}ψ",
        symmetry_type=SymmetryType.INTERNAL,
        conservation_law="Electric charge conservation",
        conserved_quantity="Total electric charge",
        quantity_symbol="Q",
        conservation_type=ConservationType.EXTENSIVE,
        generator="i (phase rotation)",
        common_errors=[
            "Confusing U(1) gauge with U(1) global",
            "Thinking charge is 'quantized' by Noether (it's from topology)",
        ],
        examples=[
            "QED: ∂_μ j^μ = 0 (current conservation)",
            "Electron-positron: e^- + e^+ → γγ preserves Q=0",
        ],
        notes="Global U(1) → charge. Local U(1) → gauge field (photon).",
    ),
    "baryon_number": NoetherPair(
        symmetry="Baryon number U(1)",
        symmetry_description="Global phase rotation of quarks",
        symmetry_type=SymmetryType.INTERNAL,
        conservation_law="Baryon number conservation",
        conserved_quantity="Total baryon number",
        quantity_symbol="B",
        conservation_type=ConservationType.EXTENSIVE,
        generator="i (baryon phase)",
        common_errors=[
            "Thinking B is exact (violated by sphalerons)",
            "Confusing with quark number",
        ],
        examples=[
            "Proton decay p → π⁰ + e⁺ would violate B",
            "Nuclear reactions: B_initial = B_final",
        ],
        notes="Approximately conserved; violated by anomalies.",
    ),
    "lepton_number": NoetherPair(
        symmetry="Lepton number U(1)",
        symmetry_description="Global phase rotation of leptons",
        symmetry_type=SymmetryType.INTERNAL,
        conservation_law="Lepton number conservation",
        conserved_quantity="Total lepton number",
        quantity_symbol="L",
        conservation_type=ConservationType.EXTENSIVE,
        generator="i (lepton phase)",
        common_errors=[
            "Confusing L (lepton) with L (angular momentum)",
            "Thinking each flavor is separately conserved",
        ],
        examples=[
            "Beta decay: n → p + e^- + ν̄_e (L=0→0+1-1=0)",
            "Neutrino oscillations violate flavor but not total L",
        ],
        notes="Flavor-specific L violated by neutrino mixing.",
    ),
    "su3_color": NoetherPair(
        symmetry="SU(3) color symmetry",
        symmetry_description="QCD color rotations",
        symmetry_type=SymmetryType.INTERNAL,
        conservation_law="Color charge conservation",
        conserved_quantity="Color charges (8 generators)",
        quantity_symbol="T^a",
        conservation_type=ConservationType.EXTENSIVE,
        generator="Gell-Mann matrices λ^a",
        common_errors=[
            "Thinking color is like electric charge",
            "Missing that observable states must be color singlets",
        ],
        examples=[
            "Quark color changes via gluon emission",
            "Mesons: q̄q (color singlet), Baryons: qqq (singlet)",
        ],
        notes="Confined — only singlets observed.",
    ),
    "cpt": NoetherPair(
        symmetry="CPT symmetry",
        symmetry_description="Charge × Parity × Time reversal",
        symmetry_type=SymmetryType.DISCRETE,
        conservation_law="CPT invariance",
        conserved_quantity="CPT parity",
        quantity_symbol="CPT",
        conservation_type=ConservationType.TOPOLOGICAL,
        generator="Discrete: C·P·T operators",
        common_errors=[
            "Confusing with individual C, P, T symmetries",
            "Thinking CPT is derived from Noether (it's from Lorentz + QFT)",
        ],
        examples=[
            "Particle-antiparticle: same mass, lifetime, |charge|",
        ],
        notes="CPT is exact in Lorentz-invariant QFT.",
    ),
    "parity": NoetherPair(
        symmetry="Parity (P)",
        symmetry_description="Spatial reflection x → -x",
        symmetry_type=SymmetryType.DISCRETE,
        conservation_law="Parity conservation",
        conserved_quantity="Parity eigenvalue",
        quantity_symbol="P = ±1",
        conservation_type=ConservationType.TOPOLOGICAL,
        generator="P operator",
        common_errors=[
            "Assuming parity is always conserved (violated in weak)",
            "Confusing parity with chirality",
        ],
        examples=[
            "Strong/EM: conserved",
            "Weak: maximally violated (Wu experiment)",
        ],
        notes="Not a Noether symmetry (discrete), but gives quantum number.",
    ),
}


# Reverse lookup: conserved quantity → symmetry
CONSERVATION_TO_SYMMETRY: Dict[str, str] = {
    "energy": "time_translation",
    "momentum": "spatial_translation",
    "linear momentum": "spatial_translation",
    "angular momentum": "rotation",
    "charge": "u1_gauge",
    "electric charge": "u1_gauge",
    "baryon number": "baryon_number",
    "lepton number": "lepton_number",
    "color": "su3_color",
    "parity": "parity",
}


@dataclass
class NoetherReport:
    """Report on Noether symmetry-conservation mapping."""
    input_query: str
    direction: str  # "symmetry→conservation" or "conservation→symmetry"
    pair: NoetherPair
    model_likely_error: Optional[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "NOETHER'S THEOREM MAPPING",
            "=" * 60,
            f"Query: {self.input_query}",
            f"Direction: {self.direction}",
            "",
            f"SYMMETRY: {self.pair.symmetry}",
            f"  {self.pair.symmetry_description}",
            f"  Type: {self.pair.symmetry_type.value}",
            f"  Generator: {self.pair.generator}",
            "",
            f"CONSERVATION LAW: {self.pair.conservation_law}",
            f"  Conserved: {self.pair.conserved_quantity} ({self.pair.quantity_symbol})",
            f"  Type: {self.pair.conservation_type.value}",
            "",
        ]

        if self.model_likely_error:
            lines.append(f"⚠️  COMMON MODEL ERROR: {self.model_likely_error}")
            lines.append("")

        lines.append("Common errors:")
        for err in self.pair.common_errors:
            lines.append(f"  • {err}")
        lines.append("")

        lines.append("Examples:")
        for ex in self.pair.examples:
            lines.append(f"  • {ex}")
        lines.append("")

        lines.append(f"Note: {self.pair.notes}")

        return "\n".join(lines)


def symmetry_to_conservation(symmetry: str) -> NoetherReport:
    """Map a symmetry to its conserved quantity.

    Args:
        symmetry: Symmetry name or description, e.g.:
            - "time translation", "temporal", "time shift"
            - "spatial translation", "space translation", "translational"
            - "rotation", "rotational", "SO(3)"
            - "U(1)", "gauge", "phase"

    Returns:
        NoetherReport with the mapping and common errors.
    """
    sym_lower = symmetry.lower()

    # Match to known symmetries
    matched_key = None

    if any(x in sym_lower for x in ["time", "temporal", "energy"]):
        matched_key = "time_translation"
    elif any(x in sym_lower for x in ["space", "spatial", "translation", "position"]):
        if "time" not in sym_lower:
            matched_key = "spatial_translation"
    elif any(x in sym_lower for x in ["rotation", "angular", "so(3)", "spherical"]):
        matched_key = "rotation"
    elif any(x in sym_lower for x in ["boost", "lorentz", "velocity"]):
        matched_key = "boost"
    elif any(x in sym_lower for x in ["u(1)", "phase", "gauge", "charge"]):
        if any(x in sym_lower for x in ["baryon", "quark"]):
            matched_key = "baryon_number"
        elif any(x in sym_lower for x in ["lepton"]):
            matched_key = "lepton_number"
        else:
            matched_key = "u1_gauge"
    elif any(x in sym_lower for x in ["su(3)", "color", "qcd"]):
        matched_key = "su3_color"
    elif any(x in sym_lower for x in ["parity", "reflection"]):
        matched_key = "parity"
    elif any(x in sym_lower for x in ["cpt"]):
        matched_key = "cpt"

    if matched_key is None:
        # Try direct key match
        matched_key = sym_lower.replace(" ", "_")

    if matched_key not in NOETHER_PAIRS:
        raise ValueError(
            f"Unknown symmetry: {symmetry}. "
            f"Available: {list(NOETHER_PAIRS.keys())}"
        )

    pair = NOETHER_PAIRS[matched_key]

    # Check for likely model error
    model_error = None
    if matched_key == "spatial_translation":
        model_error = "Models often say 'translation → energy' (WRONG: translation → momentum)"

    return NoetherReport(
        input_query=symmetry,
        direction="symmetry → conservation",
        pair=pair,
        model_likely_error=model_error,
    )


def conservation_to_symmetry(conserved: str) -> NoetherReport:
    """Map a conserved quantity to its underlying symmetry.

    Args:
        conserved: Conservation law or quantity, e.g.:
            - "energy", "total energy", "Hamiltonian"
            - "momentum", "linear momentum", "p"
            - "angular momentum", "L", "spin"
            - "charge", "electric charge", "Q"

    Returns:
        NoetherReport with the mapping and common errors.
    """
    cons_lower = conserved.lower()

    # Match to known conserved quantities
    matched_key = None

    if any(x in cons_lower for x in ["energy", "hamiltonian", "kinetic", "potential"]):
        matched_key = "time_translation"
    elif any(x in cons_lower for x in ["angular", "spin", " l ", " j "]):
        # Match angular momentum BEFORE linear momentum
        matched_key = "rotation"
    elif any(x in cons_lower for x in ["momentum", "linear momentum", " p "]):
        # Only match linear momentum if not angular
        matched_key = "spatial_translation"
    elif any(x in cons_lower for x in ["charge", " q ", "electric"]):
        if "color" not in cons_lower:
            matched_key = "u1_gauge"
    elif any(x in cons_lower for x in ["baryon"]):
        matched_key = "baryon_number"
    elif any(x in cons_lower for x in ["lepton"]):
        matched_key = "lepton_number"
    elif any(x in cons_lower for x in ["color"]):
        matched_key = "su3_color"
    elif any(x in cons_lower for x in ["parity"]):
        matched_key = "parity"

    # Try reverse lookup
    if matched_key is None:
        for key, sym_key in CONSERVATION_TO_SYMMETRY.items():
            if key in cons_lower:
                matched_key = sym_key
                break

    if matched_key is None:
        raise ValueError(
            f"Unknown conserved quantity: {conserved}. "
            f"Try: energy, momentum, angular momentum, charge, baryon/lepton number"
        )

    pair = NOETHER_PAIRS[matched_key]

    # Check for likely model error
    model_error = None
    if matched_key == "spatial_translation":
        model_error = "Models often confuse momentum ↔ energy symmetries"

    return NoetherReport(
        input_query=conserved,
        direction="conservation → symmetry",
        pair=pair,
        model_likely_error=model_error,
    )


def verify_noether_claim(
    symmetry: str,
    conserved: str,
) -> Tuple[bool, str]:
    """Verify if a symmetry-conservation claim is correct.

    Args:
        symmetry: Claimed symmetry
        conserved: Claimed conserved quantity

    Returns:
        (is_correct, explanation)
    """
    try:
        report = symmetry_to_conservation(symmetry)
    except ValueError as e:
        return False, str(e)

    actual_conservation = report.pair.conservation_law.lower()
    claimed_conservation = conserved.lower()

    # Check if they match
    if (
        actual_conservation in claimed_conservation
        or claimed_conservation in actual_conservation
        or report.pair.conserved_quantity.lower() in claimed_conservation
        or claimed_conservation in report.pair.conserved_quantity.lower()
    ):
        return True, f"CORRECT: {symmetry} → {report.pair.conservation_law}"
    else:
        return False, (
            f"INCORRECT: {symmetry} → {conserved}\n"
            f"The correct mapping is: {symmetry} → {report.pair.conservation_law}\n"
            f"({report.pair.symmetry} → {report.pair.conserved_quantity})"
        )


def list_all_pairs() -> List[str]:
    """List all known Noether symmetry-conservation pairs."""
    return list(NOETHER_PAIRS.keys())


def get_pair(key: str) -> Optional[NoetherPair]:
    """Get a specific Noether pair by key."""
    return NOETHER_PAIRS.get(key)


# Quick test
if __name__ == "__main__":
    print("=== Noether's Theorem Bidirectional Tool ===\n")

    # Test symmetry → conservation
    print("--- Symmetry → Conservation ---")
    report = symmetry_to_conservation("spatial translation")
    print(report)

    print("\n--- Conservation → Symmetry ---")
    report = conservation_to_symmetry("energy")
    print(report)

    print("\n--- Verify Claim ---")
    correct, explanation = verify_noether_claim("time translation", "energy")
    print(f"'time translation → energy': {correct}")
    print(explanation)

    correct, explanation = verify_noether_claim("spatial translation", "energy")
    print(f"\n'spatial translation → energy': {correct}")
    print(explanation)
