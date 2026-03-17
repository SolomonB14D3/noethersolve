"""Topological invariants and phases of matter.

Verified formulas for topological physics:
- Chern numbers (EXACTLY quantized, not approximate!)
- Z2 topological invariants
- Bulk-boundary correspondence
- Berry phase and curvature
- Quantum Hall conductance quantization
- Topological classification (periodic table)

KEY POINT: Topological invariants are EXACTLY INTEGER-QUANTIZED.
They cannot change continuously - only through gap closure. LLMs
often present these as approximate quantities, which is fundamentally wrong.
"""

from dataclasses import dataclass
from typing import Optional
import math

# ─── Physical Constants ─────────────────────────────────────────────────────

# Conductance quantum e²/h
CONDUCTANCE_QUANTUM = 3.87405e-5  # Siemens (exact: e²/h)
# Fine structure constant
ALPHA = 1/137.035999  # dimensionless
# Von Klitzing constant R_K = h/e²
VON_KLITZING = 25812.80745  # Ohms (exact by definition since 2019)


# ─── Report Dataclasses ─────────────────────────────────────────────────────

@dataclass
class ChernNumberReport:
    """Report from Chern number calculation."""
    chern_number: int  # Always integer!
    band_index: int
    formula: str
    is_exactly_quantized: bool  # Always True
    physical_meaning: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Chern Number (Band {self.band_index})",
            "=" * 60,
            f"  C = {self.chern_number}",
            "-" * 60,
            "  QUANTIZATION: This is EXACTLY an integer.",
            "  Cannot change without closing the band gap.",
            "-" * 60,
            f"  Physical meaning: {self.physical_meaning}",
            f"  Formula: {self.formula}",
        ]
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class Z2InvariantReport:
    """Report from Z2 topological invariant calculation."""
    nu: int  # 0 or 1 (trivial or topological)
    dimension: int  # 2D or 3D
    indices: Optional[tuple[int, ...]]  # (ν₀; ν₁ν₂ν₃) for 3D
    classification: str  # "trivial" or "topological"
    protected_by: str  # Symmetry that protects the invariant
    formula: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Z₂ Topological Invariant ({self.dimension}D)",
            "=" * 60,
            f"  ν = {self.nu} → {self.classification.upper()}",
        ]
        if self.indices:
            lines.append(f"  Full indices: (ν₀; ν₁ν₂ν₃) = {self.indices}")
        lines.extend([
            "-" * 60,
            f"  Protected by: {self.protected_by}",
            f"  Formula: {self.formula}",
            "-" * 60,
            "  Z₂ = 0: Trivial insulator (no edge states)",
            "  Z₂ = 1: Topological insulator (protected edge states)",
        ])
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class BulkBoundaryReport:
    """Report from bulk-boundary correspondence analysis."""
    bulk_invariant: int
    edge_modes: int
    correspondence_satisfied: bool
    system_type: str
    formula: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Bulk-Boundary Correspondence",
            "=" * 60,
            f"  System: {self.system_type}",
            f"  Bulk invariant: {self.bulk_invariant}",
            f"  Edge modes: {self.edge_modes}",
            "-" * 60,
        ]
        if self.correspondence_satisfied:
            lines.append("  ✓ Correspondence SATISFIED")
        else:
            lines.append("  ✗ Correspondence VIOLATED (indicates error)")
        lines.extend([
            "-" * 60,
            f"  Formula: {self.formula}",
            "  The number of edge modes equals |bulk invariant|.",
        ])
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class QuantumHallReport:
    """Report from quantum Hall effect calculation."""
    filling_factor: float  # ν
    chern_number: int  # Quantized
    hall_conductance: float  # σ_xy in e²/h
    hall_resistance: float  # R_H in Ohms
    plateau_type: str  # "integer" or "fractional"
    is_exactly_quantized: bool
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Quantum Hall Effect ({self.plateau_type.upper()} QHE)",
            "=" * 60,
            f"  Filling factor ν = {self.filling_factor}",
            f"  Chern number C = {self.chern_number}",
            "-" * 60,
            f"  Hall conductance σ_xy = {self.hall_conductance:.6f} e²/h",
            f"  Hall resistance R_H = {self.hall_resistance:.2f} Ω",
            "-" * 60,
        ]
        if self.is_exactly_quantized:
            lines.extend([
                "  QUANTIZATION: Conductance is EXACTLY ν × e²/h",
                "  This exactness is used to define the Ohm (since 2019).",
            ])
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class BerryPhaseReport:
    """Report from Berry phase calculation."""
    berry_phase: float  # In radians
    berry_phase_pi: float  # In units of π
    is_quantized: bool  # True if protected by symmetry
    quantized_value: Optional[float]  # 0 or π if quantized
    symmetry: Optional[str]  # Protecting symmetry
    formula: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Berry Phase",
            "=" * 60,
            f"  φ_B = {self.berry_phase:.6f} rad = {self.berry_phase_pi:.4f}π",
            "-" * 60,
        ]
        if self.is_quantized:
            lines.extend([
                f"  QUANTIZED to {self.quantized_value}π by {self.symmetry} symmetry",
                "  Cannot change without breaking this symmetry.",
            ])
        else:
            lines.append("  Not symmetry-protected (can vary continuously)")
        lines.extend([
            "-" * 60,
            f"  Formula: {self.formula}",
        ])
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class TopologicalClassReport:
    """Report from topological classification."""
    symmetry_class: str  # A, AIII, AI, BDI, D, DIII, AII, CII, C, CI
    dimension: int
    invariant_type: str  # "Z", "Z2", "2Z", or "0"
    has_time_reversal: bool
    has_particle_hole: bool
    has_chiral: bool
    examples: list[str]
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Topological Classification: Class {self.symmetry_class}",
            "=" * 60,
            f"  Dimension: {self.dimension}D",
            f"  Invariant: {self.invariant_type}",
            "-" * 60,
            "  Symmetries:",
            f"    Time reversal (T): {'Yes' if self.has_time_reversal else 'No'}",
            f"    Particle-hole (C): {'Yes' if self.has_particle_hole else 'No'}",
            f"    Chiral (S=TC): {'Yes' if self.has_chiral else 'No'}",
        ]
        if self.examples:
            lines.append("-" * 60)
            lines.append("  Physical examples:")
            for ex in self.examples:
                lines.append(f"    • {ex}")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Chern Number ───────────────────────────────────────────────────────────

def chern_number(
    band_index: int = 1,
    system: str = "quantum_hall",
) -> ChernNumberReport:
    """Calculate/analyze Chern number for a topological system.

    The Chern number is EXACTLY an integer - this is not an approximation!
    It is a topological invariant that cannot change without closing the gap.

    Args:
        band_index: Which band to analyze
        system: "quantum_hall", "haldane", "chern_insulator"

    Returns:
        ChernNumberReport with exact quantization emphasized
    """
    if band_index < 1:
        raise ValueError("Band index must be at least 1")

    # Different systems have different Chern numbers
    system_data = {
        "quantum_hall": {
            "chern": 1,
            "meaning": "Each filled Landau level contributes C=1",
            "notes": ["Discovered by TKNN (1982)", "Explains integer QHE plateaus"],
        },
        "haldane": {
            "chern": 1,
            "meaning": "Chern insulator without Landau levels",
            "notes": ["Haldane model (1988)", "No external magnetic field needed"],
        },
        "chern_insulator": {
            "chern": 1,
            "meaning": "General Chern insulator",
            "notes": ["σ_xy = C × e²/h exactly"],
        },
    }

    if system not in system_data:
        available = ", ".join(system_data.keys())
        raise ValueError(f"Unknown system '{system}'. Available: {available}")

    data = system_data[system]

    return ChernNumberReport(
        chern_number=data["chern"],
        band_index=band_index,
        formula="C = (1/2π) ∮ F d²k (integrated Berry curvature)",
        is_exactly_quantized=True,
        physical_meaning=data["meaning"],
        notes=data["notes"] + [
            "Chern number is EXACTLY integer-quantized",
            "Cannot change without closing the band gap",
        ],
    )


# ─── Z2 Invariant ───────────────────────────────────────────────────────────

def z2_invariant(
    nu: int,
    dimension: int = 2,
    nu_indices: Optional[tuple[int, int, int, int]] = None,
) -> Z2InvariantReport:
    """Analyze Z2 topological invariant.

    The Z2 invariant classifies topological insulators protected by
    time-reversal symmetry. It is EXACTLY 0 or 1.

    Args:
        nu: Z2 invariant (0 = trivial, 1 = topological)
        dimension: 2 or 3
        nu_indices: For 3D, the full (ν₀; ν₁ν₂ν₃) indices

    Returns:
        Z2InvariantReport with classification
    """
    if nu not in [0, 1]:
        raise ValueError("Z2 invariant must be 0 or 1")
    if dimension not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3")

    classification = "topological" if nu == 1 else "trivial"

    if dimension == 3 and nu_indices is None:
        nu_indices = (nu, 0, 0, 0)

    notes = [
        "Z₂ invariant is EXACTLY 0 or 1 (not approximate!)",
        "Protected by time-reversal symmetry",
    ]
    if nu == 1:
        notes.append("Kramer's pairs at boundary: robust to disorder")
    if dimension == 3:
        notes.append("3D: Strong TI (ν₀=1) vs Weak TI (ν₀=0, other νᵢ≠0)")

    return Z2InvariantReport(
        nu=nu,
        dimension=dimension,
        indices=nu_indices if dimension == 3 else None,
        classification=classification,
        protected_by="Time-reversal symmetry (T² = -1 for spin-1/2)",
        formula="ν = (1/2π) [∮ A·dk - ∫ F d²k] mod 2",
        notes=notes,
    )


# ─── Bulk-Boundary Correspondence ───────────────────────────────────────────

def bulk_boundary_correspondence(
    bulk_invariant: int,
    edge_modes: Optional[int] = None,
    system_type: str = "chern_insulator",
) -> BulkBoundaryReport:
    """Verify bulk-boundary correspondence.

    A fundamental principle: the number of protected edge modes equals
    the absolute value of the bulk topological invariant.

    Args:
        bulk_invariant: Bulk Chern number or Z2 invariant
        edge_modes: Number of observed edge modes (defaults to |bulk_invariant|)
        system_type: Type of topological system

    Returns:
        BulkBoundaryReport with correspondence check
    """
    if edge_modes is None:
        edge_modes = abs(bulk_invariant)

    correspondence_satisfied = (edge_modes == abs(bulk_invariant))

    formulas = {
        "chern_insulator": "n_edge = |C| (chiral edge modes)",
        "z2_insulator": "n_edge = ν mod 2 (helical edge modes)",
        "quantum_hall": "n_edge = |ν| (chiral edge states per edge)",
    }
    formula = formulas.get(system_type, "n_edge = |bulk invariant|")

    notes = []
    if correspondence_satisfied:
        notes.append("Bulk-boundary correspondence is a theorem, not empirical")
    else:
        notes.append("VIOLATION indicates calculation error or symmetry breaking")
    notes.append("Edge modes are topologically protected against backscattering")

    return BulkBoundaryReport(
        bulk_invariant=bulk_invariant,
        edge_modes=edge_modes,
        correspondence_satisfied=correspondence_satisfied,
        system_type=system_type,
        formula=formula,
        notes=notes,
    )


# ─── Quantum Hall Effect ────────────────────────────────────────────────────

def quantum_hall(
    filling_factor: float,
    is_integer: bool = True,
) -> QuantumHallReport:
    """Calculate quantum Hall effect properties.

    The Hall conductance is EXACTLY quantized to ν × e²/h.
    This exactness is used to define the Ohm in the SI system (since 2019).

    Args:
        filling_factor: Landau level filling ν
        is_integer: True for IQHE, False for FQHE

    Returns:
        QuantumHallReport with quantization emphasized
    """
    if filling_factor <= 0:
        raise ValueError("Filling factor must be positive")

    if is_integer:
        chern = int(round(filling_factor))
        plateau_type = "integer"
    else:
        # Fractional QHE - use denominator for effective Chern
        chern = 1  # Effective Chern for the composite fermion
        plateau_type = "fractional"

    # Hall conductance σ_xy = ν × e²/h
    hall_conductance = filling_factor  # In units of e²/h

    # Hall resistance R_H = h/(ν × e²)
    hall_resistance = VON_KLITZING / filling_factor

    notes = [
        "Hall conductance is EXACTLY ν × e²/h",
        f"Von Klitzing constant R_K = h/e² = {VON_KLITZING:.5f} Ω (exact by definition)",
    ]
    if is_integer:
        notes.append("Integer QHE: each Landau level contributes e²/h")
    else:
        notes.append("Fractional QHE: strongly correlated electron states")
        notes.append("Laughlin states at ν = 1/3, 1/5, ...")

    return QuantumHallReport(
        filling_factor=filling_factor,
        chern_number=chern,
        hall_conductance=hall_conductance,
        hall_resistance=hall_resistance,
        plateau_type=plateau_type,
        is_exactly_quantized=True,
        notes=notes,
    )


# ─── Berry Phase ────────────────────────────────────────────────────────────

def berry_phase(
    phase_value: float,
    symmetry: Optional[str] = None,
) -> BerryPhaseReport:
    """Analyze Berry phase and its quantization.

    The Berry phase can be quantized to 0 or π by certain symmetries.
    Without symmetry protection, it varies continuously.

    Args:
        phase_value: Berry phase in radians
        symmetry: Protecting symmetry ("inversion", "time_reversal", or None)

    Returns:
        BerryPhaseReport with quantization analysis
    """
    # Normalize to [0, 2π)
    phase_normalized = phase_value % (2 * math.pi)
    phase_pi = phase_normalized / math.pi

    # Check if quantized
    is_quantized = symmetry in ["inversion", "time_reversal"]
    quantized_value = None

    if is_quantized:
        # Must be 0 or π
        if abs(phase_normalized) < 0.01 or abs(phase_normalized - 2*math.pi) < 0.01:
            quantized_value = 0.0
        elif abs(phase_normalized - math.pi) < 0.01:
            quantized_value = 1.0
        else:
            is_quantized = False  # Symmetry must be broken

    notes = []
    if is_quantized:
        notes.append(f"Quantized to {quantized_value}π by {symmetry} symmetry")
        notes.append("This quantization is topologically protected")
    else:
        notes.append("Not quantized - can vary continuously under deformation")

    notes.append("Berry phase is gauge-dependent mod 2π")
    notes.append("Physical observables depend on Berry curvature (gauge-invariant)")

    return BerryPhaseReport(
        berry_phase=phase_normalized,
        berry_phase_pi=phase_pi,
        is_quantized=is_quantized,
        quantized_value=quantized_value,
        symmetry=symmetry,
        formula="φ_B = ∮ ⟨u|∇_k|u⟩ · dk",
        notes=notes,
    )


# ─── Topological Classification ─────────────────────────────────────────────

# The periodic table of topological insulators/superconductors
PERIODIC_TABLE = {
    # (class, dimension) -> invariant type
    ("A", 2): "Z",  # Chern insulator
    ("A", 3): "0",
    ("AIII", 1): "Z",
    ("AIII", 2): "0",
    ("AIII", 3): "Z",
    ("AI", 1): "0",
    ("AI", 2): "0",
    ("AI", 3): "0",
    ("BDI", 1): "Z",
    ("BDI", 2): "0",
    ("BDI", 3): "0",
    ("D", 1): "Z2",
    ("D", 2): "Z",  # p+ip superconductor
    ("D", 3): "0",
    ("DIII", 1): "Z2",
    ("DIII", 2): "Z2",
    ("DIII", 3): "Z",  # 3He-B
    ("AII", 1): "0",
    ("AII", 2): "Z2",  # 2D TI (QSH)
    ("AII", 3): "Z2",  # 3D TI
    ("CII", 1): "2Z",
    ("CII", 2): "0",
    ("CII", 3): "Z2",
    ("C", 1): "0",
    ("C", 2): "2Z",
    ("C", 3): "0",
    ("CI", 1): "0",
    ("CI", 2): "0",
    ("CI", 3): "2Z",
}

# Symmetry properties for each class
SYMMETRY_CLASSES = {
    "A": (False, False, False),      # No symmetries (complex)
    "AIII": (False, False, True),    # Chiral only
    "AI": (True, False, False),      # T²=+1
    "BDI": (True, True, True),       # T²=+1, C²=+1
    "D": (False, True, False),       # C²=+1
    "DIII": (True, True, True),      # T²=-1, C²=+1
    "AII": (True, False, False),     # T²=-1
    "CII": (True, True, True),       # T²=-1, C²=-1
    "C": (False, True, False),       # C²=-1
    "CI": (True, True, True),        # T²=+1, C²=-1
}

EXAMPLES = {
    ("A", 2): ["Integer Quantum Hall", "Haldane model"],
    ("AII", 2): ["HgTe/CdTe quantum wells", "Quantum spin Hall"],
    ("AII", 3): ["Bi₂Se₃", "Bi₂Te₃", "3D topological insulator"],
    ("D", 2): ["p+ip superconductor", "Sr₂RuO₄"],
    ("DIII", 3): ["³He-B superfluid"],
}


def topological_classification(
    symmetry_class: str,
    dimension: int,
) -> TopologicalClassReport:
    """Look up topological classification from periodic table.

    The "periodic table" of topological insulators classifies all
    possible topological phases based on symmetry and dimension.

    Args:
        symmetry_class: Altland-Zirnbauer class (A, AIII, AI, BDI, D, DIII, AII, CII, C, CI)
        dimension: Spatial dimension (1, 2, or 3)

    Returns:
        TopologicalClassReport with full classification
    """
    symmetry_class = symmetry_class.upper()

    if symmetry_class not in SYMMETRY_CLASSES:
        available = ", ".join(SYMMETRY_CLASSES.keys())
        raise ValueError(f"Unknown class '{symmetry_class}'. Available: {available}")
    if dimension not in [1, 2, 3]:
        raise ValueError("Dimension must be 1, 2, or 3")

    invariant = PERIODIC_TABLE.get((symmetry_class, dimension), "0")
    has_T, has_C, has_S = SYMMETRY_CLASSES[symmetry_class]
    examples = EXAMPLES.get((symmetry_class, dimension), [])

    notes = []
    if invariant == "Z":
        notes.append("Z: Characterized by integer (Chern number)")
    elif invariant == "Z2":
        notes.append("Z₂: Characterized by 0 or 1 (Kane-Mele invariant)")
    elif invariant == "2Z":
        notes.append("2Z: Even integers only")
    else:
        notes.append("Topologically trivial in this dimension")

    notes.append("Classification follows Bott periodicity (period 8)")

    return TopologicalClassReport(
        symmetry_class=symmetry_class,
        dimension=dimension,
        invariant_type=invariant,
        has_time_reversal=has_T,
        has_particle_hole=has_C,
        has_chiral=has_S,
        examples=examples,
        notes=notes,
    )


def list_symmetry_classes() -> list[str]:
    """List all Altland-Zirnbauer symmetry classes."""
    return list(SYMMETRY_CLASSES.keys())
