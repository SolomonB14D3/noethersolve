"""MHD (Magnetohydrodynamics) Conservation Laws.

Verified conservation law analysis for ideal and resistive MHD:
- Magnetic helicity (topological invariant)
- Cross helicity (correlation of v and B)
- Energy (magnetic + kinetic + thermal)
- Mass, momentum, magnetic flux

KEY POINTS LLMs GET WRONG:
1. Magnetic helicity is EXACTLY conserved in ideal MHD, but DECAYS in resistive MHD
2. Cross helicity is conserved in ideal MHD, broken by resistivity AND compressibility
3. Total energy is conserved in ideal MHD, but Ohmic dissipation exists in resistive MHD
4. The ∇·B = 0 constraint must be preserved numerically — NOT automatic!
5. Reconnection REQUIRES resistivity (or numerical diffusion) — ideal MHD cannot reconnect

CRITICAL DISTINCTIONS:
- Ideal MHD: η = 0, all invariants conserved, frozen-in flux
- Resistive MHD: η > 0, helicity decays, reconnection possible
- Hall MHD: preserves helicity but transfers between scales
- Relativistic MHD: different conservation forms
"""

from dataclasses import dataclass
from typing import Optional
import math


# ─── Physical Constants ─────────────────────────────────────────────────────

MU_0 = 4 * math.pi * 1e-7  # Vacuum permeability (H/m)


# ─── Report Dataclasses ─────────────────────────────────────────────────────

@dataclass
class HelicityReport:
    """Report on magnetic helicity conservation."""
    helicity_type: str  # "magnetic", "cross", "kinetic"
    is_conserved: bool
    decay_rate: Optional[float]  # For resistive MHD
    regime: str  # "ideal", "resistive", "hall"
    resistivity: float
    topological_meaning: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  {self.helicity_type.title()} Helicity Conservation",
            "=" * 60,
            f"  Regime: {self.regime}",
            f"  Resistivity η: {self.resistivity:.2e}",
            "-" * 60,
        ]
        if self.is_conserved:
            lines.append("  ✓ CONSERVED (exactly in this regime)")
        else:
            lines.append("  ✗ NOT CONSERVED")
            if self.decay_rate is not None:
                lines.append(f"    Decay rate: dH/dt ∝ -η × {self.decay_rate:.2e}")
        lines.append("-" * 60)
        lines.append(f"  Meaning: {self.topological_meaning}")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class MHDEnergyReport:
    """Report on MHD energy conservation."""
    magnetic_energy: float
    kinetic_energy: float
    thermal_energy: Optional[float]
    total_energy: float
    is_conserved: bool
    dissipation_rate: Optional[float]
    dissipation_sources: list[str]
    regime: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  MHD Energy Conservation",
            "=" * 60,
            f"  Magnetic energy: {self.magnetic_energy:.4e}",
            f"  Kinetic energy: {self.kinetic_energy:.4e}",
        ]
        if self.thermal_energy is not None:
            lines.append(f"  Thermal energy: {self.thermal_energy:.4e}")
        lines.append(f"  Total energy: {self.total_energy:.4e}")
        lines.append("-" * 60)
        lines.append(f"  Regime: {self.regime}")
        if self.is_conserved:
            lines.append("  ✓ CONSERVED")
        else:
            lines.append("  ✗ NOT CONSERVED")
            if self.dissipation_rate is not None:
                lines.append(f"    Dissipation rate: {self.dissipation_rate:.4e}")
            if self.dissipation_sources:
                lines.append(f"    Sources: {', '.join(self.dissipation_sources)}")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class FrozenFluxReport:
    """Report on frozen-in flux theorem."""
    is_frozen: bool
    regime: str
    magnetic_reynolds: Optional[float]
    diffusion_time: Optional[float]
    reconnection_possible: bool
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Frozen-in Flux Theorem",
            "=" * 60,
            f"  Regime: {self.regime}",
        ]
        if self.magnetic_reynolds is not None:
            lines.append(f"  Magnetic Reynolds Rm: {self.magnetic_reynolds:.2e}")
        lines.append("-" * 60)
        if self.is_frozen:
            lines.append("  ✓ Flux is FROZEN into plasma")
            lines.append("    Field lines move with fluid")
        else:
            lines.append("  ✗ Flux NOT frozen")
            lines.append("    Field lines can slip through plasma")
        lines.append("-" * 60)
        if self.reconnection_possible:
            lines.append("  ⚠ Magnetic reconnection IS possible")
        else:
            lines.append("  Magnetic reconnection NOT possible")
        if self.diffusion_time is not None:
            lines.append(f"  Diffusion time: {self.diffusion_time:.2e} s")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class DivBReport:
    """Report on ∇·B = 0 constraint."""
    max_div_B: float
    tolerance: float
    is_satisfied: bool
    cleaning_method: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  ∇·B = 0 Constraint",
            "=" * 60,
            f"  max|∇·B|: {self.max_div_B:.2e}",
            f"  Tolerance: {self.tolerance:.2e}",
            "-" * 60,
        ]
        if self.is_satisfied:
            lines.append("  ✓ SATISFIED (constraint preserved)")
        else:
            lines.append("  ✗ VIOLATED (monopoles present)")
            lines.append(f"    Recommended cleaning: {self.cleaning_method}")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class InvariantReport:
    """Report on MHD invariant conservation."""
    invariant: str
    value: float
    is_conserved: bool
    regime: str
    conservation_condition: str
    breaking_mechanism: Optional[str]
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  MHD Invariant: {self.invariant}",
            "=" * 60,
            f"  Value: {self.value:.6e}",
            f"  Regime: {self.regime}",
            "-" * 60,
        ]
        if self.is_conserved:
            lines.append("  ✓ CONSERVED")
            lines.append(f"    Condition: {self.conservation_condition}")
        else:
            lines.append("  ✗ NOT CONSERVED")
            if self.breaking_mechanism:
                lines.append(f"    Breaking mechanism: {self.breaking_mechanism}")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Helicity Functions ─────────────────────────────────────────────────────

def check_magnetic_helicity(
    resistivity: float = 0.0,
    regime: str = "",
) -> HelicityReport:
    """Check magnetic helicity conservation.

    Magnetic helicity H_m = ∫ A·B dV measures the linkage and twist of
    magnetic field lines. It is a TOPOLOGICAL invariant in ideal MHD.

    CRITICAL FACTS:
    - Ideal MHD (η = 0): H_m is EXACTLY conserved
    - Resistive MHD (η > 0): H_m decays at rate dH_m/dt = -2η ∫ J·B dV
    - Hall MHD: H_m is conserved but transfers between scales
    - Reconnection REQUIRES helicity dissipation

    Args:
        resistivity: Magnetic diffusivity η (= 1/μ₀σ)
        regime: Override regime detection ("ideal", "resistive", "hall")

    Returns:
        HelicityReport with conservation analysis
    """
    # Determine regime
    if regime:
        reg = regime.lower()
    elif resistivity == 0.0:
        reg = "ideal"
    elif resistivity > 0:
        reg = "resistive"
    else:
        raise ValueError("Resistivity must be non-negative")

    notes = []

    if reg == "ideal":
        is_conserved = True
        decay_rate = None
        notes.append("H_m = ∫ A·B dV is EXACTLY conserved")
        notes.append("Topological invariant: measures field line linkage")
        notes.append("Cannot change without resistivity")
    elif reg == "resistive":
        is_conserved = False
        decay_rate = 1.0  # Proportional constant
        notes.append("dH_m/dt = -2η ∫ J·B dV")
        notes.append("Helicity decays where current and field are aligned")
        notes.append("Reconnection sites have high |J·B|")
    elif reg == "hall":
        is_conserved = True
        decay_rate = None
        notes.append("Hall MHD conserves total helicity")
        notes.append("But transfers helicity between scales (inverse cascade)")
        notes.append("Generalized helicity H_g = H_m + H_ion is conserved")
    else:
        raise ValueError(f"Unknown regime '{regime}'")

    meaning = "Measures magnetic field line linkage, twist, and writhe"

    return HelicityReport(
        helicity_type="magnetic",
        is_conserved=is_conserved,
        decay_rate=decay_rate,
        regime=reg,
        resistivity=resistivity,
        topological_meaning=meaning,
        notes=notes,
    )


def check_cross_helicity(
    resistivity: float = 0.0,
    viscosity: float = 0.0,
    compressible: bool = False,
) -> HelicityReport:
    """Check cross helicity conservation.

    Cross helicity H_c = ∫ v·B dV measures the correlation between
    velocity and magnetic field. Conserved in ideal incompressible MHD.

    BREAKING MECHANISMS:
    - Resistivity: ∂H_c/∂t includes -η ∫ (∇×B)·v dV
    - Viscosity: ∂H_c/∂t includes -ν ∫ (∇²v)·B dV
    - Compressibility: ∂H_c/∂t includes terms with ∇·v

    Args:
        resistivity: Magnetic diffusivity η
        viscosity: Kinematic viscosity ν
        compressible: Whether flow is compressible

    Returns:
        HelicityReport with conservation analysis
    """
    notes = []
    breaking_reasons = []

    # Check conservation conditions
    is_conserved = True
    decay_rate = None

    if resistivity > 0:
        is_conserved = False
        breaking_reasons.append("resistivity")
        notes.append(f"Resistive term: -η ∫ J·v dV (η = {resistivity:.2e})")

    if viscosity > 0:
        is_conserved = False
        breaking_reasons.append("viscosity")
        notes.append(f"Viscous term: -ν ∫ ω·B dV (ν = {viscosity:.2e})")

    if compressible:
        is_conserved = False
        breaking_reasons.append("compressibility")
        notes.append("Compressible term: ∫ (∇·v)(v·B) dV")

    if is_conserved:
        regime = "ideal incompressible"
        notes.append("H_c = ∫ v·B dV is EXACTLY conserved")
        notes.append("Measures alignment of flow and magnetic field")
    else:
        regime = f"dissipative ({', '.join(breaking_reasons)})"
        decay_rate = resistivity + viscosity  # Simplified

    meaning = "Measures correlation/alignment between velocity and magnetic field"

    return HelicityReport(
        helicity_type="cross",
        is_conserved=is_conserved,
        decay_rate=decay_rate,
        regime=regime,
        resistivity=resistivity,
        topological_meaning=meaning,
        notes=notes,
    )


# ─── Energy Functions ───────────────────────────────────────────────────────

def check_mhd_energy(
    B_rms: float,
    v_rms: float,
    density: float,
    resistivity: float = 0.0,
    viscosity: float = 0.0,
    J_rms: float = 0.0,
    omega_rms: float = 0.0,
    temperature: Optional[float] = None,
    volume: float = 1.0,
) -> MHDEnergyReport:
    """Check MHD energy conservation.

    Total MHD energy: E = E_mag + E_kin + E_thermal
    where E_mag = ∫ B²/(2μ₀) dV, E_kin = ∫ ρv²/2 dV

    DISSIPATION:
    - Ohmic: η J² (resistive heating)
    - Viscous: ν ρ ω² (viscous heating)
    - Ideal MHD: E is EXACTLY conserved

    Args:
        B_rms: RMS magnetic field (T)
        v_rms: RMS velocity (m/s)
        density: Mass density (kg/m³)
        resistivity: Magnetic diffusivity η
        viscosity: Kinematic viscosity ν
        J_rms: RMS current density (for dissipation estimate)
        omega_rms: RMS vorticity (for viscous dissipation)
        temperature: Temperature (K) for thermal energy
        volume: Volume (m³)

    Returns:
        MHDEnergyReport with energy analysis
    """
    # Compute energies
    E_mag = (B_rms ** 2 / (2 * MU_0)) * volume
    E_kin = (0.5 * density * v_rms ** 2) * volume

    if temperature is not None:
        # Simple ideal gas: E_thermal ~ (3/2) n k_B T
        # Using P = ρ R T / μ, assume μ = 1 amu
        k_B = 1.38e-23
        E_thermal = (3 / 2) * (density / 1.67e-27) * k_B * temperature * volume
    else:
        E_thermal = None

    E_total = E_mag + E_kin
    if E_thermal is not None:
        E_total += E_thermal

    # Check conservation
    dissipation_sources = []
    dissipation_rate = 0.0

    if resistivity > 0 and J_rms > 0:
        ohmic = resistivity * J_rms ** 2 * volume
        dissipation_rate += ohmic
        dissipation_sources.append("Ohmic (η J²)")

    if viscosity > 0 and omega_rms > 0:
        viscous = viscosity * density * omega_rms ** 2 * volume
        dissipation_rate += viscous
        dissipation_sources.append("Viscous (ν ρ ω²)")

    is_conserved = len(dissipation_sources) == 0

    if is_conserved:
        regime = "ideal"
        notes = [
            "Total energy E = E_mag + E_kin is EXACTLY conserved",
            "Energy transfers between magnetic and kinetic forms",
        ]
    else:
        regime = "dissipative"
        notes = [
            "Energy dissipates via Ohmic and/or viscous heating",
            "Energy converts to heat (increases thermal energy)",
        ]

    return MHDEnergyReport(
        magnetic_energy=E_mag,
        kinetic_energy=E_kin,
        thermal_energy=E_thermal,
        total_energy=E_total,
        is_conserved=is_conserved,
        dissipation_rate=dissipation_rate if dissipation_rate > 0 else None,
        dissipation_sources=dissipation_sources,
        regime=regime,
        notes=notes,
    )


# ─── Frozen Flux ────────────────────────────────────────────────────────────

def check_frozen_flux(
    resistivity: float = 0.0,
    length_scale: float = 1.0,
    velocity: float = 1.0,
) -> FrozenFluxReport:
    """Check frozen-in flux theorem validity.

    In ideal MHD, magnetic field lines are "frozen" into the plasma:
    they move with the fluid and cannot slip through it.

    FROZEN FLUX BREAKS when:
    - Resistivity η > 0 (field diffuses through plasma)
    - Magnetic Reynolds number Rm = vL/η < ~10 (diffusion dominates)

    RECONNECTION requires breaking frozen flux!

    Args:
        resistivity: Magnetic diffusivity η (m²/s)
        length_scale: Characteristic length L (m)
        velocity: Characteristic velocity v (m/s)

    Returns:
        FrozenFluxReport with frozen flux analysis
    """
    notes = []

    if resistivity == 0.0:
        is_frozen = True
        regime = "ideal"
        Rm = float('inf')
        tau_diff = float('inf')
        reconnection = False
        notes.append("Flux is EXACTLY frozen (Alfvén theorem)")
        notes.append("Field lines move with fluid velocity")
        notes.append("No magnetic reconnection possible")
    else:
        # Magnetic Reynolds number
        Rm = velocity * length_scale / resistivity
        tau_diff = length_scale ** 2 / resistivity

        if Rm > 100:
            is_frozen = True  # Approximately frozen
            regime = "weakly resistive"
            reconnection = True  # But possible at small scales
            notes.append(f"Rm = {Rm:.1e} >> 1: flux approximately frozen")
            notes.append("Reconnection possible at small scales (current sheets)")
        elif Rm > 1:
            is_frozen = False
            regime = "resistive"
            reconnection = True
            notes.append(f"Rm = {Rm:.1e} ~ O(1): flux not frozen")
            notes.append("Significant field line slippage")
        else:
            is_frozen = False
            regime = "diffusion-dominated"
            reconnection = True
            notes.append(f"Rm = {Rm:.1e} < 1: diffusion dominates")
            notes.append("Field structure determined by boundaries")

        notes.append(f"Diffusion time τ_η = L²/η = {tau_diff:.2e} s")

    return FrozenFluxReport(
        is_frozen=is_frozen,
        regime=regime,
        magnetic_reynolds=Rm if resistivity > 0 else None,
        diffusion_time=tau_diff if resistivity > 0 else None,
        reconnection_possible=reconnection,
        notes=notes,
    )


# ─── Div B Constraint ───────────────────────────────────────────────────────

def check_div_B(
    max_div_B: float,
    B_scale: float = 1.0,
    dx: float = 1.0,
    tolerance_factor: float = 1e-8,
) -> DivBReport:
    """Check ∇·B = 0 constraint satisfaction.

    Maxwell's equations require ∇·B = 0 (no magnetic monopoles).
    Numerical MHD must PRESERVE this constraint — it's not automatic!

    COMMON CLEANING METHODS:
    - Projection method: solve ∇²φ = ∇·B, then B → B - ∇φ
    - Constrained transport: preserve ∇·B by construction
    - Powell 8-wave: source terms to damp ∇·B
    - Hyperbolic divergence cleaning: wave equation for ψ

    Args:
        max_div_B: Maximum |∇·B| in domain
        B_scale: Characteristic magnetic field strength
        dx: Grid spacing (for normalizing)
        tolerance_factor: Acceptable normalized divergence

    Returns:
        DivBReport with constraint analysis
    """
    # Normalized divergence
    normalized_div = max_div_B * dx / B_scale
    tolerance = tolerance_factor * B_scale / dx

    is_satisfied = max_div_B < tolerance

    notes = []
    if is_satisfied:
        notes.append("Constraint satisfied to specified tolerance")
        cleaning = "none needed"
    else:
        notes.append("Monopole error detected — cleaning recommended")
        notes.append(f"Normalized error: {normalized_div:.2e}")
        if normalized_div < 1e-4:
            cleaning = "projection method"
            notes.append("Small error — projection cleaning sufficient")
        elif normalized_div < 1e-2:
            cleaning = "hyperbolic divergence cleaning"
            notes.append("Moderate error — use hyperbolic cleaning (Dedner et al.)")
        else:
            cleaning = "constrained transport"
            notes.append("Large error — switch to constrained transport scheme")

    notes.append("∇·B = 0 must be preserved for physical solutions")
    notes.append("Monopole errors can cause spurious forces and instabilities")

    return DivBReport(
        max_div_B=max_div_B,
        tolerance=tolerance,
        is_satisfied=is_satisfied,
        cleaning_method=cleaning,
        notes=notes,
    )


# ─── General Invariant Check ────────────────────────────────────────────────

def check_mhd_invariant(
    invariant: str,
    resistivity: float = 0.0,
    viscosity: float = 0.0,
    compressible: bool = False,
) -> InvariantReport:
    """Check conservation of a specific MHD invariant.

    INVARIANTS AND THEIR CONSERVATION:
    - Mass: Always conserved (continuity equation)
    - Momentum: Conserved (no external forces)
    - Magnetic flux: Conserved in ideal MHD (frozen flux)
    - Magnetic helicity: Conserved in ideal MHD, decays with η
    - Cross helicity: Conserved in ideal incompressible MHD
    - Energy: Conserved in ideal MHD, dissipates with η and ν
    - Alfvén wave action: Conserved in uniform background

    Args:
        invariant: Name of invariant to check
        resistivity: Magnetic diffusivity η
        viscosity: Kinematic viscosity ν
        compressible: Whether flow is compressible

    Returns:
        InvariantReport with conservation analysis
    """
    inv_lower = invariant.lower().replace(" ", "_").replace("-", "_")

    is_ideal = resistivity == 0 and viscosity == 0

    if inv_lower in ["mass", "density"]:
        return InvariantReport(
            invariant="Mass",
            value=0.0,  # Placeholder
            is_conserved=True,
            regime="all",
            conservation_condition="Always (continuity equation ∂ρ/∂t + ∇·(ρv) = 0)",
            breaking_mechanism=None,
            notes=["Mass is ALWAYS conserved in MHD", "Fundamental conservation law"],
        )

    elif inv_lower == "momentum":
        return InvariantReport(
            invariant="Momentum",
            value=0.0,
            is_conserved=True,  # Assuming no external forces
            regime="no external forces",
            conservation_condition="No external forces (gravity, etc.)",
            breaking_mechanism="External forces" if False else None,
            notes=[
                "Momentum conserved for closed system",
                "Magnetic tension/pressure redistribute momentum",
            ],
        )

    elif inv_lower in ["magnetic_flux", "flux"]:
        is_conserved = is_ideal
        return InvariantReport(
            invariant="Magnetic Flux",
            value=0.0,
            is_conserved=is_conserved,
            regime="ideal" if is_ideal else "resistive",
            conservation_condition="η = 0 (frozen-in flux)",
            breaking_mechanism="Resistive diffusion" if not is_conserved else None,
            notes=[
                "Flux through any surface moving with fluid is conserved" if is_conserved
                else "Flux can diffuse through plasma with η > 0",
            ],
        )

    elif inv_lower in ["magnetic_helicity", "helicity"]:
        is_conserved = is_ideal
        return InvariantReport(
            invariant="Magnetic Helicity",
            value=0.0,
            is_conserved=is_conserved,
            regime="ideal" if is_ideal else "resistive",
            conservation_condition="η = 0",
            breaking_mechanism="Resistive dissipation dH/dt = -2η∫J·B dV" if not is_conserved else None,
            notes=[
                "Topological invariant measuring field line linkage",
                "EXACTLY conserved in ideal MHD",
                "Decays slowly even with resistivity (robust invariant)",
            ],
        )

    elif inv_lower == "cross_helicity":
        is_conserved = is_ideal and not compressible
        breaking = []
        if resistivity > 0:
            breaking.append("resistivity")
        if viscosity > 0:
            breaking.append("viscosity")
        if compressible:
            breaking.append("compressibility")
        return InvariantReport(
            invariant="Cross Helicity",
            value=0.0,
            is_conserved=is_conserved,
            regime="ideal incompressible" if is_conserved else "dissipative",
            conservation_condition="η = 0, ν = 0, incompressible",
            breaking_mechanism=", ".join(breaking) if breaking else None,
            notes=[
                "Measures v·B alignment/correlation",
                "Requires BOTH zero dissipation AND incompressibility",
            ],
        )

    elif inv_lower == "energy":
        is_conserved = is_ideal
        return InvariantReport(
            invariant="Total Energy",
            value=0.0,
            is_conserved=is_conserved,
            regime="ideal" if is_ideal else "dissipative",
            conservation_condition="η = 0, ν = 0",
            breaking_mechanism="Ohmic (ηJ²) and viscous (νρω²) dissipation" if not is_conserved else None,
            notes=[
                "E = E_mag + E_kin (+ E_thermal)",
                "Energy converts between forms but total conserved in ideal MHD",
            ],
        )

    else:
        raise ValueError(f"Unknown invariant '{invariant}'. Known: mass, momentum, "
                        "magnetic_flux, magnetic_helicity, cross_helicity, energy")


def list_mhd_invariants() -> list[str]:
    """List available MHD invariants."""
    return [
        "mass",
        "momentum",
        "magnetic_flux",
        "magnetic_helicity",
        "cross_helicity",
        "energy",
    ]
