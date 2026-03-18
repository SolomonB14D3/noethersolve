"""General Relativity Constraint Equations and Mass Definitions.

Verified analysis of GR constraint equations and conserved quantities:
- ADM (Arnowitt-Deser-Misner) formalism
- Hamiltonian and momentum constraints
- Different mass definitions (ADM, Bondi, Komar)
- Constraint propagation and violation

KEY POINTS LLMs GET WRONG:
1. ADM mass is conserved (constant), Bondi mass DECREASES with gravitational radiation
2. Komar mass only applies to stationary spacetimes (requires Killing vector)
3. Hamiltonian constraint H ≈ 0 is NOT an evolution equation — it's a CONSTRAINT
4. Constraints propagate automatically in exact GR but need monitoring numerically
5. All three masses agree for stationary, asymptotically flat spacetimes

CRITICAL DISTINCTIONS:
- ADM mass: measured at spatial infinity, includes all energy
- Bondi mass: measured at null infinity, decreases as waves radiate
- Komar mass: integral of Killing vector, only for stationary spacetimes
- Quasi-local masses: various definitions for finite regions
"""

from dataclasses import dataclass
from typing import Optional


# ─── Physical Constants ─────────────────────────────────────────────────────

G = 6.674e-11  # Gravitational constant (m³/kg/s²)
C = 2.998e8    # Speed of light (m/s)


# ─── Report Dataclasses ─────────────────────────────────────────────────────

@dataclass
class ConstraintReport:
    """Report on constraint satisfaction."""
    constraint_type: str  # "Hamiltonian" or "momentum"
    value: float
    tolerance: float
    is_satisfied: bool
    physical_meaning: str
    violation_consequence: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  GR Constraint: {self.constraint_type}",
            "=" * 60,
            f"  Value: {self.value:.4e}",
            f"  Tolerance: {self.tolerance:.4e}",
            "-" * 60,
        ]
        if self.is_satisfied:
            lines.append("  ✓ SATISFIED")
        else:
            lines.append("  ✗ VIOLATED")
            lines.append(f"    Consequence: {self.violation_consequence}")
        lines.append("-" * 60)
        lines.append(f"  Physical meaning: {self.physical_meaning}")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class MassReport:
    """Report on GR mass definition analysis."""
    mass_type: str  # "ADM", "Bondi", "Komar"
    value: Optional[float]  # In solar masses or kg
    is_applicable: bool
    applicability_condition: str
    conservation_property: str
    related_to_radiation: bool
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  GR Mass Definition: {self.mass_type}",
            "=" * 60,
        ]
        if self.is_applicable:
            lines.append("  ✓ Applicable to this spacetime")
            if self.value is not None:
                lines.append(f"  Mass value: {self.value:.4e}")
        else:
            lines.append("  ✗ NOT applicable")
            lines.append(f"    Requires: {self.applicability_condition}")
        lines.append("-" * 60)
        lines.append(f"  Conservation: {self.conservation_property}")
        if self.related_to_radiation:
            lines.append("  ⚠ Changes with gravitational radiation")
        else:
            lines.append("  Constant (no radiation dependence)")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class ADMReport:
    """Report on ADM formalism analysis."""
    is_valid_foliation: bool
    lapse_positive: bool
    constraint_status: str
    evolution_type: str  # "hyperbolic", "elliptic constraints + hyperbolic evolution"
    gauge_choice: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  ADM Formalism Analysis",
            "=" * 60,
            f"  Valid foliation: {'✓' if self.is_valid_foliation else '✗'}",
            f"  Lapse positive: {'✓' if self.lapse_positive else '✗'}",
            f"  Constraint status: {self.constraint_status}",
            "-" * 60,
            f"  Evolution type: {self.evolution_type}",
            f"  Gauge choice: {self.gauge_choice}",
        ]
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class MassComparisonReport:
    """Report comparing different mass definitions."""
    spacetime_type: str
    adm_applicable: bool
    bondi_applicable: bool
    komar_applicable: bool
    masses_agree: bool
    disagreement_reason: Optional[str]
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  GR Mass Comparison",
            "=" * 60,
            f"  Spacetime: {self.spacetime_type}",
            "-" * 60,
            f"  ADM mass applicable: {'✓' if self.adm_applicable else '✗'}",
            f"  Bondi mass applicable: {'✓' if self.bondi_applicable else '✗'}",
            f"  Komar mass applicable: {'✓' if self.komar_applicable else '✗'}",
            "-" * 60,
        ]
        if self.masses_agree:
            lines.append("  ✓ All applicable masses AGREE")
        else:
            lines.append("  ✗ Masses DISAGREE")
            if self.disagreement_reason:
                lines.append(f"    Reason: {self.disagreement_reason}")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Constraint Functions ───────────────────────────────────────────────────

def check_hamiltonian_constraint(
    value: float,
    tolerance: float = 1e-10,
) -> ConstraintReport:
    """Check Hamiltonian constraint satisfaction.

    The Hamiltonian constraint H ≈ 0 is one of two constraint equations in GR.
    In the ADM formalism: H = R + K² - K_ij K^ij - 16πρ = 0

    This is NOT an evolution equation — it must be satisfied on each time slice.

    CRITICAL: Constraint violation indicates:
    - Initial data is unphysical
    - Numerical errors have accumulated
    - Gauge/coordinate problems

    Args:
        value: Computed value of Hamiltonian constraint H
        tolerance: Acceptable tolerance for H ≈ 0

    Returns:
        ConstraintReport with analysis
    """
    is_satisfied = abs(value) < tolerance

    meaning = "Energy constraint: total energy density on hypersurface"
    violation = "Initial data or evolution is unphysical"

    notes = [
        "H = R + K² - K_ij K^ij - 16πρ ≈ 0",
        "R = 3D scalar curvature of spatial slice",
        "K_ij = extrinsic curvature, ρ = energy density",
    ]

    if is_satisfied:
        notes.append("Constraint satisfied — data is physically consistent")
    else:
        notes.append("Constraint violated — check initial data or numerical scheme")
        notes.append("Violation can grow exponentially if not controlled")

    return ConstraintReport(
        constraint_type="Hamiltonian",
        value=value,
        tolerance=tolerance,
        is_satisfied=is_satisfied,
        physical_meaning=meaning,
        violation_consequence=violation,
        notes=notes,
    )


def check_momentum_constraint(
    value: float,
    component: str = "all",
    tolerance: float = 1e-10,
) -> ConstraintReport:
    """Check momentum constraint satisfaction.

    The momentum constraint M_i ≈ 0 is the second constraint in GR.
    In the ADM formalism: M_i = D_j(K^j_i - δ^j_i K) - 8π j_i = 0

    Three components (one per spatial direction) must all vanish.

    CRITICAL: Momentum constraint violation indicates:
    - Unphysical momentum distribution
    - Violation of local momentum conservation

    Args:
        value: Computed value of momentum constraint |M|
        component: Which component ("x", "y", "z", or "all" for norm)
        tolerance: Acceptable tolerance

    Returns:
        ConstraintReport with analysis
    """
    is_satisfied = abs(value) < tolerance

    meaning = f"Momentum constraint ({component}): local momentum conservation"
    violation = "Unphysical momentum distribution in initial data"

    notes = [
        "M_i = D_j(K^j_i - δ^j_i K) - 8π j_i ≈ 0",
        "D_j = covariant derivative on 3-slice",
        "j_i = momentum density, three components",
    ]

    if is_satisfied:
        notes.append("Momentum constraint satisfied")
    else:
        notes.append("Momentum constraint violated — check momentum sources")

    return ConstraintReport(
        constraint_type=f"Momentum ({component})",
        value=value,
        tolerance=tolerance,
        is_satisfied=is_satisfied,
        physical_meaning=meaning,
        violation_consequence=violation,
        notes=notes,
    )


# ─── Mass Definition Functions ──────────────────────────────────────────────

def check_adm_mass(
    is_asymptotically_flat: bool = True,
    is_isolated: bool = True,
    mass_value: Optional[float] = None,
) -> MassReport:
    """Analyze ADM mass applicability and properties.

    ADM mass is measured at SPATIAL infinity and includes ALL energy.
    It is CONSTANT in time — does not change even with gravitational radiation.

    REQUIRES:
    - Asymptotically flat spacetime
    - Well-defined spatial infinity

    Formula: M_ADM = (1/16π) lim_{r→∞} ∮ (∂_j γ_ij - ∂_i γ_jj) dS^i

    Args:
        is_asymptotically_flat: Does spacetime approach flat at spatial infinity?
        is_isolated: Is the system isolated (no matter at infinity)?
        mass_value: Computed ADM mass value (optional)

    Returns:
        MassReport with analysis
    """
    is_applicable = is_asymptotically_flat and is_isolated

    notes = [
        "M_ADM measures total mass-energy at spatial infinity i⁰",
        "Includes ALL contributions: matter, binding energy, radiation",
        "CONSTANT in time — conserved even as waves radiate",
    ]

    if is_applicable:
        conservation = "Exactly conserved (constant in time)"
        notes.append("Well-defined because spacetime is asymptotically flat")
    else:
        conservation = "Not applicable"
        if not is_asymptotically_flat:
            notes.append("Not applicable: spacetime is not asymptotically flat")
        if not is_isolated:
            notes.append("Not applicable: matter extends to infinity")

    return MassReport(
        mass_type="ADM",
        value=mass_value,
        is_applicable=is_applicable,
        applicability_condition="Asymptotically flat and isolated",
        conservation_property=conservation,
        related_to_radiation=False,  # ADM mass does NOT change with radiation
        notes=notes,
    )


def check_bondi_mass(
    is_asymptotically_flat: bool = True,
    has_null_infinity: bool = True,
    has_radiation: bool = False,
    mass_value: Optional[float] = None,
    news_function: Optional[float] = None,
) -> MassReport:
    """Analyze Bondi mass applicability and properties.

    Bondi mass is measured at NULL infinity and DECREASES with radiation.
    This is the mass "seen" by distant observers at late times.

    CRITICAL: M_Bondi ≤ M_ADM always, with equality only when no radiation.

    Bondi mass loss: dM_Bondi/du = -(1/4π) ∮ |N|² dS
    where N is the news function (gravitational wave amplitude).

    Args:
        is_asymptotically_flat: Does spacetime approach flat?
        has_null_infinity: Does null infinity (scri+) exist?
        has_radiation: Is there gravitational radiation?
        mass_value: Computed Bondi mass value (optional)
        news_function: News function magnitude (for mass loss rate)

    Returns:
        MassReport with analysis
    """
    is_applicable = is_asymptotically_flat and has_null_infinity

    notes = [
        "M_Bondi measures mass at null infinity ℐ⁺ (scri+)",
        "DECREASES as gravitational waves radiate energy",
        "M_Bondi ≤ M_ADM always",
    ]

    if is_applicable:
        if has_radiation:
            conservation = "Decreases monotonically with radiation"
            notes.append("dM/du = -(1/4π) ∮ |N|² dS (Bondi mass loss formula)")
            if news_function is not None:
                notes.append(f"News function |N| = {news_function:.4e}")
        else:
            conservation = "Constant (no radiation)"
            notes.append("Without radiation, M_Bondi = M_ADM")
    else:
        conservation = "Not applicable"
        if not has_null_infinity:
            notes.append("Not applicable: null infinity ℐ⁺ not well-defined")

    return MassReport(
        mass_type="Bondi",
        value=mass_value,
        is_applicable=is_applicable,
        applicability_condition="Asymptotically flat with null infinity",
        conservation_property=conservation,
        related_to_radiation=has_radiation,
        notes=notes,
    )


def check_komar_mass(
    is_stationary: bool = True,
    has_killing_vector: bool = True,
    killing_type: str = "timelike",
    mass_value: Optional[float] = None,
) -> MassReport:
    """Analyze Komar mass applicability and properties.

    Komar mass REQUIRES a timelike Killing vector — only exists for
    STATIONARY spacetimes. Does not apply to dynamical spacetimes!

    Formula: M_Komar = (1/4π) ∮ ∇^μ ξ^ν dS_μν
    where ξ is the timelike Killing vector.

    CRITICAL LIMITATION: Cannot be used for:
    - Binary black hole mergers (non-stationary)
    - Gravitational wave sources (non-stationary)
    - Any dynamical evolution

    Args:
        is_stationary: Is spacetime stationary (time-independent)?
        has_killing_vector: Does a Killing vector exist?
        killing_type: Type of Killing vector ("timelike", "null", "spacelike")
        mass_value: Computed Komar mass value (optional)

    Returns:
        MassReport with analysis
    """
    is_applicable = is_stationary and has_killing_vector and killing_type == "timelike"

    notes = [
        "Komar mass requires timelike Killing vector ξ^μ",
        "M_K = (1/4π) ∮ ∇^μ ξ^ν dS_μν",
        "Only defined for STATIONARY spacetimes",
    ]

    if is_applicable:
        conservation = "Constant (stationary by definition)"
        notes.append("For Schwarzschild: M_Komar = M_ADM = M_Bondi = M")
        notes.append("For Kerr: includes rotational energy")
    else:
        conservation = "Not applicable"
        if not is_stationary:
            notes.append("Not applicable: spacetime is dynamical")
            notes.append("Use ADM or Bondi mass instead")
        if not has_killing_vector:
            notes.append("Not applicable: no Killing vector exists")
        if killing_type != "timelike":
            notes.append(f"Not applicable: Killing vector is {killing_type}, not timelike")

    return MassReport(
        mass_type="Komar",
        value=mass_value,
        is_applicable=is_applicable,
        applicability_condition="Stationary with timelike Killing vector",
        conservation_property=conservation,
        related_to_radiation=False,
        notes=notes,
    )


# ─── Mass Comparison ────────────────────────────────────────────────────────

def compare_mass_definitions(
    spacetime_type: str = "schwarzschild",
    has_radiation: bool = False,
) -> MassComparisonReport:
    """Compare ADM, Bondi, and Komar mass for a spacetime type.

    For stationary, asymptotically flat spacetimes WITHOUT radiation:
    M_ADM = M_Bondi = M_Komar (all agree)

    For radiating spacetimes:
    M_Bondi < M_ADM (Bondi decreases)
    M_Komar is NOT defined (not stationary)

    Args:
        spacetime_type: Type of spacetime
        has_radiation: Is there gravitational radiation?

    Returns:
        MassComparisonReport with comparison
    """
    st = spacetime_type.lower()

    if st in ["schwarzschild", "kerr", "reissner_nordstrom", "kerr_newman"]:
        # Stationary black holes
        adm = True
        bondi = True
        komar = True
        if has_radiation:
            # Can't have radiation from isolated stationary BH
            has_radiation = False
        masses_agree = True
        disagreement = None
        notes = [
            f"{spacetime_type.title()} is stationary and asymptotically flat",
            "All three mass definitions are applicable and agree",
            "M_ADM = M_Bondi = M_Komar = M (total mass)",
        ]
    elif st in ["binary_merger", "bbh", "bns", "dynamical"]:
        # Dynamical spacetime
        adm = True
        bondi = True
        komar = False
        masses_agree = not has_radiation
        if has_radiation:
            disagreement = "M_Bondi < M_ADM due to radiated energy"
        else:
            disagreement = None
        notes = [
            "Dynamical spacetime — not stationary",
            "Komar mass NOT defined (no timelike Killing vector)",
            "ADM mass is constant, Bondi mass decreases with radiation",
        ]
    elif st in ["cosmological", "flrw", "de_sitter"]:
        # Cosmological spacetime
        adm = False
        bondi = False
        komar = st == "de_sitter"  # de Sitter has Killing vector
        masses_agree = False
        disagreement = "Cosmological spacetimes are not asymptotically flat"
        notes = [
            "Cosmological spacetime — not asymptotically flat",
            "ADM and Bondi masses not defined",
            "Different mass concepts needed (quasi-local)",
        ]
    else:
        # Generic
        adm = True
        bondi = True
        komar = False
        masses_agree = not has_radiation
        disagreement = "Bondi decreases with radiation" if has_radiation else None
        notes = [f"Generic spacetime: {spacetime_type}"]

    return MassComparisonReport(
        spacetime_type=spacetime_type,
        adm_applicable=adm,
        bondi_applicable=bondi,
        komar_applicable=komar,
        masses_agree=masses_agree,
        disagreement_reason=disagreement,
        notes=notes,
    )


# ─── ADM Formalism ──────────────────────────────────────────────────────────

def analyze_adm_formalism(
    lapse_value: float = 1.0,
    shift_norm: float = 0.0,
    hamiltonian_constraint: float = 0.0,
    momentum_constraint: float = 0.0,
    gauge: str = "geodesic",
) -> ADMReport:
    """Analyze ADM formalism setup.

    The ADM formalism decomposes spacetime into space + time:
    ds² = -α² dt² + γ_ij (dx^i + β^i dt)(dx^j + β^j dt)

    α = lapse function (how much proper time passes)
    β^i = shift vector (how coordinates shift between slices)

    EVOLUTION: hyperbolic for metric, CONSTRAINTS must be monitored.

    Args:
        lapse_value: Value of lapse function α (must be > 0)
        shift_norm: Norm of shift vector |β|
        hamiltonian_constraint: Value of Hamiltonian constraint
        momentum_constraint: Value of momentum constraint |M|
        gauge: Gauge choice ("geodesic", "harmonic", "BSSN", "puncture")

    Returns:
        ADMReport with analysis
    """
    lapse_positive = lapse_value > 0
    is_valid_foliation = lapse_positive  # Minimum requirement

    # Constraint status
    h_ok = abs(hamiltonian_constraint) < 1e-10
    m_ok = abs(momentum_constraint) < 1e-10

    if h_ok and m_ok:
        constraint_status = "All constraints satisfied"
    elif h_ok:
        constraint_status = "Hamiltonian OK, momentum violated"
    elif m_ok:
        constraint_status = "Momentum OK, Hamiltonian violated"
    else:
        constraint_status = "Both constraints violated"

    notes = [
        f"Lapse α = {lapse_value:.4f} {'✓' if lapse_positive else '✗ INVALID'}",
        f"Shift |β| = {shift_norm:.4f}",
    ]

    # Gauge-specific notes
    gauge_lower = gauge.lower()
    if gauge_lower == "geodesic":
        gauge_name = "Geodesic slicing (α=1, β=0)"
        notes.append("Simple but can develop coordinate singularities")
    elif gauge_lower == "harmonic":
        gauge_name = "Harmonic gauge (□x^μ = 0)"
        notes.append("Well-posed, used in many numerical codes")
    elif gauge_lower in ["bssn", "baumgarte_shapiro"]:
        gauge_name = "BSSN formulation"
        notes.append("Conformal decomposition, better stability")
    elif gauge_lower in ["puncture", "moving_puncture"]:
        gauge_name = "Moving puncture gauge"
        notes.append("Standard for binary black hole simulations")
    else:
        gauge_name = gauge

    evolution_type = "Elliptic constraints + hyperbolic evolution equations"
    notes.append("Constraints propagate automatically in exact GR")
    notes.append("Must monitor constraints numerically (can grow)")

    return ADMReport(
        is_valid_foliation=is_valid_foliation,
        lapse_positive=lapse_positive,
        constraint_status=constraint_status,
        evolution_type=evolution_type,
        gauge_choice=gauge_name,
        notes=notes,
    )


def list_gr_concepts() -> list[str]:
    """List available GR concepts."""
    return [
        "hamiltonian_constraint",
        "momentum_constraint",
        "adm_mass",
        "bondi_mass",
        "komar_mass",
        "adm_formalism",
    ]
