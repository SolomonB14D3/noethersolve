"""
Plasma physics adiabatic invariants calculator.

The three adiabatic invariants for charged particle motion in magnetic fields:
1. Magnetic moment μ = m*v⊥²/(2*B) — first (fastest) invariant
2. Longitudinal invariant J = ∮ v∥ ds — second (bounce) invariant
3. Flux invariant Φ = ∮ A·dl — third (drift) invariant

CRITICAL: Each invariant breaks under different conditions that LLMs confuse.

BREAKING CONDITIONS (must be VIOLATED for invariant to hold):
- μ breaks when: ω_timescale ~ ω_cyclotron (changes too fast)
- J breaks when: ω_timescale ~ ω_bounce (trap changes during bounce)
- Φ breaks when: ω_timescale ~ ω_drift (field changes during drift orbit)

Hierarchy: ω_cyclotron >> ω_bounce >> ω_drift
So: μ is most robust, Φ is most fragile.

Common LLM errors:
1. Treating all three as equally robust
2. Getting the hierarchy backwards
3. Not recognizing μ conservation in magnetic mirrors
4. Confusing μ with total magnetic flux
"""

from dataclasses import dataclass, field
from typing import Optional, List
import math

# Physical constants
ELECTRON_MASS = 9.109e-31  # kg
PROTON_MASS = 1.673e-27  # kg
ELECTRON_CHARGE = 1.602e-19  # C


@dataclass
class MagneticMomentReport:
    """Report for first adiabatic invariant μ."""

    mu: float  # Magnetic moment J/T
    v_perp: float  # Perpendicular velocity m/s
    B: float  # Magnetic field T
    mass: float  # Particle mass kg
    kinetic_energy_perp: float  # m*v⊥²/2 in Joules
    kinetic_energy_perp_eV: float  # In electron volts

    # Cyclotron frequency
    omega_cyclotron: float  # rad/s
    cyclotron_period: float  # seconds
    cyclotron_radius: float  # Larmor radius, meters

    # Conservation status
    is_conserved: bool
    breaking_condition: str
    field_timescale: Optional[float]  # Timescale of field variation

    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "First Adiabatic Invariant (μ) Report",
            "=" * 50,
            f"Magnetic moment μ: {self.mu:.6e} J/T",
            f"Perpendicular velocity v⊥: {self.v_perp:.4e} m/s",
            f"Magnetic field B: {self.B:.4e} T",
            f"Particle mass: {self.mass:.4e} kg",
            "",
            "Perpendicular energy:",
            f"  E⊥ = {self.kinetic_energy_perp:.4e} J",
            f"  E⊥ = {self.kinetic_energy_perp_eV:.4e} eV",
            "",
            "Cyclotron motion:",
            f"  ω_c = {self.omega_cyclotron:.4e} rad/s",
            f"  Period τ_c = {self.cyclotron_period:.4e} s",
            f"  Larmor radius r_L = {self.cyclotron_radius:.4e} m",
            "",
            f"Conserved: {self.is_conserved}",
        ]
        if not self.is_conserved:
            lines.append(f"Breaking condition: {self.breaking_condition}")
        if self.field_timescale:
            lines.append(f"Field timescale: {self.field_timescale:.4e} s")
            ratio = self.field_timescale / self.cyclotron_period
            lines.append(f"Timescale ratio (τ_field/τ_c): {ratio:.2f}")
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


@dataclass
class BounceInvariantReport:
    """Report for second adiabatic invariant J."""

    J: float  # Longitudinal invariant, m*m/s (momentum units)
    v_parallel: float  # Parallel velocity at reference point
    bounce_length: float  # Distance between mirror points
    B_min: float  # Minimum B (at equator)
    B_max: float  # Maximum B (at mirror points)
    mirror_ratio: float  # B_max / B_min

    # Bounce frequency
    omega_bounce: float  # rad/s
    bounce_period: float  # seconds

    # Conservation status
    is_conserved: bool
    breaking_condition: str
    field_timescale: Optional[float]

    # Mirror point info
    pitch_angle: float  # degrees, at B_min
    loss_cone_angle: float  # degrees, particles with α < this are lost

    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "Second Adiabatic Invariant (J) Report",
            "=" * 50,
            f"Longitudinal invariant J: {self.J:.6e} m²/s",
            f"Parallel velocity v∥: {self.v_parallel:.4e} m/s",
            f"Bounce length: {self.bounce_length:.4e} m",
            "",
            "Magnetic configuration:",
            f"  B_min: {self.B_min:.4e} T",
            f"  B_max: {self.B_max:.4e} T",
            f"  Mirror ratio R = B_max/B_min: {self.mirror_ratio:.4f}",
            "",
            "Bounce motion:",
            f"  ω_bounce = {self.omega_bounce:.4e} rad/s",
            f"  Bounce period τ_b = {self.bounce_period:.4e} s",
            "",
            "Pitch angle (at B_min):",
            f"  α = {self.pitch_angle:.2f}°",
            f"  Loss cone angle: {self.loss_cone_angle:.2f}°",
            "",
            f"Conserved: {self.is_conserved}",
        ]
        if not self.is_conserved:
            lines.append(f"Breaking condition: {self.breaking_condition}")
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"Note: {note}")
        return "\n".join(lines)


@dataclass
class FluxInvariantReport:
    """Report for third adiabatic invariant Φ."""

    Phi: float  # Flux enclosed by drift orbit, Weber (Tm²)
    drift_radius: float  # Average drift orbit radius, m
    B_average: float  # Average field over drift orbit, T

    # Drift frequency
    omega_drift: float  # rad/s
    drift_period: float  # seconds

    # Conservation status
    is_conserved: bool
    breaking_condition: str
    field_timescale: Optional[float]

    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "Third Adiabatic Invariant (Φ) Report",
            "=" * 50,
            f"Flux invariant Φ: {self.Phi:.6e} Wb",
            f"Drift orbit radius: {self.drift_radius:.4e} m",
            f"Average field B: {self.B_average:.4e} T",
            "",
            "Drift motion:",
            f"  ω_drift = {self.omega_drift:.4e} rad/s",
            f"  Drift period τ_d = {self.drift_period:.4e} s",
            "",
            f"Conserved: {self.is_conserved}",
        ]
        if not self.is_conserved:
            lines.append(f"Breaking condition: {self.breaking_condition}")
        if self.field_timescale:
            lines.append(f"Field timescale: {self.field_timescale:.4e} s")
            ratio = self.field_timescale / self.drift_period
            lines.append(f"Timescale ratio (τ_field/τ_d): {ratio:.2f}")
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"Note: {note}")
        return "\n".join(lines)


@dataclass
class AdiabaticHierarchyReport:
    """Report comparing all three adiabatic invariants."""

    # Frequencies
    omega_cyclotron: float
    omega_bounce: float
    omega_drift: float

    # Periods
    tau_cyclotron: float
    tau_bounce: float
    tau_drift: float

    # Which are conserved?
    mu_conserved: bool
    J_conserved: bool
    Phi_conserved: bool

    field_timescale: Optional[float]

    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "Adiabatic Invariant Hierarchy",
            "=" * 50,
            "",
            "Frequency hierarchy (should be ω_c >> ω_b >> ω_d):",
            f"  ω_cyclotron = {self.omega_cyclotron:.4e} rad/s",
            f"  ω_bounce    = {self.omega_bounce:.4e} rad/s",
            f"  ω_drift     = {self.omega_drift:.4e} rad/s",
            "",
            "Period hierarchy (τ_c << τ_b << τ_d):",
            f"  τ_cyclotron = {self.tau_cyclotron:.4e} s",
            f"  τ_bounce    = {self.tau_bounce:.4e} s",
            f"  τ_drift     = {self.tau_drift:.4e} s",
            "",
            "Ratios:",
            f"  τ_b / τ_c = {self.tau_bounce / self.tau_cyclotron:.2f}",
            f"  τ_d / τ_b = {self.tau_drift / self.tau_bounce:.2f}",
            "",
            "Conservation status:",
            f"  μ (first):  {'CONSERVED' if self.mu_conserved else 'BROKEN'}",
            f"  J (second): {'CONSERVED' if self.J_conserved else 'BROKEN'}",
            f"  Φ (third):  {'CONSERVED' if self.Phi_conserved else 'BROKEN'}",
        ]
        if self.field_timescale:
            lines.append("")
            lines.append(f"Field variation timescale: {self.field_timescale:.4e} s")
            lines.append("")
            lines.append("Adiabaticity check (τ_field vs periods):")
            lines.append(f"  vs τ_c: {self.field_timescale/self.tau_cyclotron:.2f}× (μ {'OK' if self.mu_conserved else 'BREAKS'})")
            lines.append(f"  vs τ_b: {self.field_timescale/self.tau_bounce:.2f}× (J {'OK' if self.J_conserved else 'BREAKS'})")
            lines.append(f"  vs τ_d: {self.field_timescale/self.tau_drift:.2f}× (Φ {'OK' if self.Phi_conserved else 'BREAKS'})")
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"Note: {note}")
        return "\n".join(lines)


def calc_magnetic_moment(
    v_perp: float,
    B: float,
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
    field_timescale: Optional[float] = None,
) -> MagneticMomentReport:
    """
    Calculate the first adiabatic invariant μ = m*v⊥²/(2*B).

    The magnetic moment is conserved when the magnetic field changes
    slowly compared to the cyclotron period.

    CONSERVATION CONDITION: τ_field >> τ_cyclotron

    Args:
        v_perp: Perpendicular velocity (m/s)
        B: Magnetic field strength (Tesla)
        mass: Particle mass (kg, default: electron)
        charge: Particle charge magnitude (C, default: e)
        field_timescale: Timescale of field variation (s, optional)

    Returns:
        MagneticMomentReport with μ, cyclotron parameters, conservation status

    Example:
        # 1 keV electron in 1 T field
        v_perp = sqrt(2*1e3*1.6e-19/9.1e-31) ≈ 1.9e7 m/s
        result = calc_magnetic_moment(1.9e7, 1.0)
    """
    if B <= 0:
        raise ValueError(f"Magnetic field must be positive, got {B}")
    if v_perp < 0:
        raise ValueError(f"Velocity must be non-negative, got {v_perp}")
    if mass <= 0:
        raise ValueError(f"Mass must be positive, got {mass}")
    if charge <= 0:
        raise ValueError(f"Charge must be positive, got {charge}")

    # Magnetic moment μ = m*v⊥²/(2*B)
    mu = mass * v_perp**2 / (2 * B)

    # Perpendicular kinetic energy
    E_perp = 0.5 * mass * v_perp**2
    E_perp_eV = E_perp / ELECTRON_CHARGE

    # Cyclotron frequency ω_c = |q|B/m
    omega_c = charge * B / mass
    tau_c = 2 * math.pi / omega_c

    # Larmor radius r_L = v⊥ / ω_c
    r_L = v_perp / omega_c if omega_c > 0 else float('inf')

    # Check conservation
    notes = []
    if field_timescale is not None:
        ratio = field_timescale / tau_c
        is_conserved = ratio > 10  # Rule of thumb: need >> 1, use 10 as threshold
        if is_conserved:
            breaking_condition = "N/A - field varies slowly"
            notes.append(f"τ_field/τ_c = {ratio:.1f} >> 1: Adiabatic condition satisfied")
        else:
            breaking_condition = f"τ_field ~ τ_c (ratio = {ratio:.2f})"
            notes.append(f"τ_field/τ_c = {ratio:.2f} < 10: Field changes too fast")
    else:
        is_conserved = True  # Assume conserved if no timescale given
        breaking_condition = "N/A"
        notes.append("No field timescale provided - assuming slow variation")

    return MagneticMomentReport(
        mu=mu,
        v_perp=v_perp,
        B=B,
        mass=mass,
        kinetic_energy_perp=E_perp,
        kinetic_energy_perp_eV=E_perp_eV,
        omega_cyclotron=omega_c,
        cyclotron_period=tau_c,
        cyclotron_radius=r_L,
        is_conserved=is_conserved,
        breaking_condition=breaking_condition,
        field_timescale=field_timescale,
        notes=notes,
    )


def calc_bounce_invariant(
    v_parallel: float,
    bounce_length: float,
    B_min: float,
    B_max: float,
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
    field_timescale: Optional[float] = None,
) -> BounceInvariantReport:
    """
    Calculate the second adiabatic invariant J = ∮ v∥ ds.

    For a magnetic mirror trap, J ≈ 2 * v∥_avg * L where L is
    the distance between mirror points.

    CONSERVATION CONDITION: τ_field >> τ_bounce

    Pitch angle at B_min: sin²(α) = B_min/B_max for mirror point
    Loss cone: particles with sin²(α₀) < B_min/B_max escape

    Args:
        v_parallel: Parallel velocity at B_min (m/s)
        bounce_length: Distance between mirror points (m)
        B_min: Minimum field at trap center (T)
        B_max: Maximum field at mirror points (T)
        mass: Particle mass (kg)
        charge: Particle charge magnitude (C)
        field_timescale: Timescale of field variation (s, optional)

    Returns:
        BounceInvariantReport with J, bounce frequency, mirror geometry
    """
    if B_min <= 0 or B_max <= 0:
        raise ValueError(f"Magnetic fields must be positive")
    if B_max <= B_min:
        raise ValueError(f"B_max ({B_max}) must be greater than B_min ({B_min})")
    if bounce_length <= 0:
        raise ValueError(f"Bounce length must be positive")
    if v_parallel < 0:
        raise ValueError(f"Velocity must be non-negative")

    # Mirror ratio
    R = B_max / B_min

    # Approximate J ≈ 2 * v∥ * L (simplified for constant v∥)
    # More precisely, J = ∮ m*v∥ ds, integrated over bounce orbit
    J = 2 * v_parallel * bounce_length

    # Bounce period τ_b ≈ 2L / v∥ (order of magnitude)
    tau_b = 2 * bounce_length / v_parallel if v_parallel > 0 else float('inf')
    omega_b = 2 * math.pi / tau_b if tau_b != float('inf') else 0

    # Pitch angle at B_min (for particle that mirrors at B_max)
    # sin²(α) = B_min / B_max at mirror point
    sin2_alpha = 1 / R  # At equator, this is the minimum sin²α
    pitch_angle = math.degrees(math.asin(math.sqrt(sin2_alpha)))

    # Loss cone angle - particles with smaller pitch angle escape
    loss_cone = math.degrees(math.asin(math.sqrt(1/R)))

    # Check conservation
    notes = []
    if field_timescale is not None:
        ratio = field_timescale / tau_b
        is_conserved = ratio > 10
        if is_conserved:
            breaking_condition = "N/A - field varies slowly compared to bounce"
        else:
            breaking_condition = f"τ_field ~ τ_bounce (ratio = {ratio:.2f})"
            notes.append("Trap configuration changes during bounce motion")
    else:
        is_conserved = True
        breaking_condition = "N/A"

    notes.append(f"Mirror ratio R = {R:.2f}: particles mirror when B reaches {B_max:.2e} T")
    if loss_cone < 10:
        notes.append(f"Small loss cone ({loss_cone:.1f}°): good particle confinement")

    return BounceInvariantReport(
        J=J,
        v_parallel=v_parallel,
        bounce_length=bounce_length,
        B_min=B_min,
        B_max=B_max,
        mirror_ratio=R,
        omega_bounce=omega_b,
        bounce_period=tau_b,
        is_conserved=is_conserved,
        breaking_condition=breaking_condition,
        field_timescale=field_timescale,
        pitch_angle=pitch_angle,
        loss_cone_angle=loss_cone,
        notes=notes,
    )


def calc_flux_invariant(
    drift_radius: float,
    B_average: float,
    energy: float,
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
    field_timescale: Optional[float] = None,
) -> FluxInvariantReport:
    """
    Calculate the third adiabatic invariant Φ = ∮ A·dl.

    The flux enclosed by the drift orbit. This is the most fragile
    invariant - breaks when field changes on drift timescale.

    CONSERVATION CONDITION: τ_field >> τ_drift

    Args:
        drift_radius: Average radius of drift orbit (m)
        B_average: Average magnetic field over drift orbit (T)
        energy: Particle kinetic energy (eV)
        mass: Particle mass (kg)
        charge: Particle charge magnitude (C)
        field_timescale: Timescale of field variation (s, optional)

    Returns:
        FluxInvariantReport with Φ, drift frequency, conservation status
    """
    if drift_radius <= 0:
        raise ValueError(f"Drift radius must be positive")
    if B_average <= 0:
        raise ValueError(f"Magnetic field must be positive")
    if energy < 0:
        raise ValueError(f"Energy must be non-negative")

    # Flux enclosed by circular drift orbit
    # Φ ≈ π * r² * B for circular orbit
    Phi = math.pi * drift_radius**2 * B_average

    # Energy in Joules
    E_joules = energy * ELECTRON_CHARGE

    # Drift velocity (gradient + curvature, order of magnitude)
    # v_d ~ E / (q * B * R) for gradient drift
    v_drift = E_joules / (charge * B_average * drift_radius)

    # Drift period
    circumference = 2 * math.pi * drift_radius
    tau_d = circumference / v_drift if v_drift > 0 else float('inf')
    omega_d = 2 * math.pi / tau_d if tau_d != float('inf') else 0

    # Check conservation
    notes = []
    if field_timescale is not None:
        ratio = field_timescale / tau_d
        is_conserved = ratio > 10
        if is_conserved:
            breaking_condition = "N/A - field varies slowly compared to drift"
        else:
            breaking_condition = f"τ_field ~ τ_drift (ratio = {ratio:.2f})"
            notes.append("Field reconfigures during drift orbit - Φ not conserved")
    else:
        is_conserved = True
        breaking_condition = "N/A"

    notes.append("Third invariant is most fragile - breaks for rapid field changes")
    notes.append(f"Typical breaking: magnetic storms, substorms (τ ~ minutes)")

    return FluxInvariantReport(
        Phi=Phi,
        drift_radius=drift_radius,
        B_average=B_average,
        omega_drift=omega_d,
        drift_period=tau_d,
        is_conserved=is_conserved,
        breaking_condition=breaking_condition,
        field_timescale=field_timescale,
        notes=notes,
    )


def check_adiabatic_hierarchy(
    B: float,
    v_total: float,
    pitch_angle_deg: float,
    bounce_length: float,
    drift_radius: float,
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
    field_timescale: Optional[float] = None,
) -> AdiabaticHierarchyReport:
    """
    Check all three adiabatic invariants and their hierarchy.

    The correct hierarchy is: ω_cyclotron >> ω_bounce >> ω_drift
    Equivalently: τ_cyclotron << τ_bounce << τ_drift

    This means μ is most robust, Φ is most fragile.

    Args:
        B: Magnetic field at reference point (T)
        v_total: Total particle velocity (m/s)
        pitch_angle_deg: Pitch angle in degrees at reference point
        bounce_length: Distance between mirror points (m)
        drift_radius: Average drift orbit radius (m)
        mass: Particle mass (kg)
        charge: Particle charge magnitude (C)
        field_timescale: Timescale of field variation (s, optional)

    Returns:
        AdiabaticHierarchyReport with all frequencies and conservation status
    """
    if B <= 0:
        raise ValueError("Magnetic field must be positive")
    if v_total <= 0:
        raise ValueError("Total velocity must be positive")
    if not 0 < pitch_angle_deg < 90:
        raise ValueError("Pitch angle must be between 0 and 90 degrees")

    # Decompose velocity
    alpha_rad = math.radians(pitch_angle_deg)
    v_perp = v_total * math.sin(alpha_rad)
    v_parallel = v_total * math.cos(alpha_rad)

    # Cyclotron frequency and period
    omega_c = charge * B / mass
    tau_c = 2 * math.pi / omega_c

    # Bounce frequency and period
    tau_b = 2 * bounce_length / v_parallel if v_parallel > 0 else float('inf')
    omega_b = 2 * math.pi / tau_b if tau_b != float('inf') else 0

    # Drift frequency and period
    # Using gradient drift scaling: v_d ~ E / (q*B*R)
    E_joules = 0.5 * mass * v_total**2
    v_drift = E_joules / (charge * B * drift_radius)
    circumference = 2 * math.pi * drift_radius
    tau_d = circumference / v_drift if v_drift > 0 else float('inf')
    omega_d = 2 * math.pi / tau_d if tau_d != float('inf') else 0

    # Check hierarchy
    notes = []
    if omega_c > omega_b > omega_d:
        notes.append("Correct hierarchy: ω_c >> ω_b >> ω_d")
    else:
        notes.append("WARNING: Hierarchy violated - adiabatic theory may not apply")

    # Check conservation based on field timescale
    if field_timescale is not None:
        mu_conserved = field_timescale > 10 * tau_c
        J_conserved = field_timescale > 10 * tau_b
        Phi_conserved = field_timescale > 10 * tau_d

        if mu_conserved and J_conserved and Phi_conserved:
            notes.append("All invariants conserved - truly adiabatic regime")
        elif mu_conserved and J_conserved and not Phi_conserved:
            notes.append("μ and J conserved, Φ broken - intermediate regime")
            notes.append("Typical of: geomagnetic storms, substorm injection")
        elif mu_conserved and not J_conserved:
            notes.append("Only μ conserved - fast parallel transport")
            notes.append("Typical of: wave-particle resonance, field line reconnection")
        else:
            notes.append("μ broken - non-adiabatic regime")
            notes.append("Typical of: strong turbulence, shock acceleration")
    else:
        # Assume all conserved if no timescale given
        mu_conserved = True
        J_conserved = True
        Phi_conserved = True
        notes.append("No field timescale given - assuming slow variation")

    return AdiabaticHierarchyReport(
        omega_cyclotron=omega_c,
        omega_bounce=omega_b,
        omega_drift=omega_d,
        tau_cyclotron=tau_c,
        tau_bounce=tau_b,
        tau_drift=tau_d,
        mu_conserved=mu_conserved,
        J_conserved=J_conserved,
        Phi_conserved=Phi_conserved,
        field_timescale=field_timescale,
        notes=notes,
    )


def mirror_force(
    mu: float,
    dB_ds: float,
) -> float:
    """
    Calculate the mirror force on a particle.

    F_mirror = -μ * (dB/ds)

    This force acts along the field line, pushing particles
    away from regions of high field (magnetic mirrors).

    Args:
        mu: Magnetic moment (J/T)
        dB_ds: Field gradient along field line (T/m)

    Returns:
        Mirror force in Newtons (negative = toward lower B)
    """
    return -mu * dB_ds


def loss_cone_angle(
    B_min: float,
    B_max: float,
) -> float:
    """
    Calculate the loss cone angle for a magnetic mirror.

    Particles with pitch angle α < α_loss escape the trap.

    sin²(α_loss) = B_min / B_max

    Args:
        B_min: Minimum field at trap center (T)
        B_max: Maximum field at mirror points (T)

    Returns:
        Loss cone half-angle in degrees
    """
    if B_min <= 0 or B_max <= 0:
        raise ValueError("Fields must be positive")
    if B_max <= B_min:
        raise ValueError("B_max must be greater than B_min")

    sin2_alpha = B_min / B_max
    alpha = math.asin(math.sqrt(sin2_alpha))
    return math.degrees(alpha)


def cyclotron_frequency(
    B: float,
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
) -> float:
    """
    Calculate cyclotron (gyro) frequency.

    ω_c = |q|B/m

    Args:
        B: Magnetic field (T)
        mass: Particle mass (kg)
        charge: Particle charge magnitude (C)

    Returns:
        Cyclotron frequency in rad/s
    """
    if B <= 0:
        raise ValueError("Field must be positive")
    return charge * B / mass


def larmor_radius(
    v_perp: float,
    B: float,
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
) -> float:
    """
    Calculate Larmor (gyro) radius.

    r_L = m*v⊥ / (|q|B) = v⊥ / ω_c

    Args:
        v_perp: Perpendicular velocity (m/s)
        B: Magnetic field (T)
        mass: Particle mass (kg)
        charge: Particle charge magnitude (C)

    Returns:
        Larmor radius in meters
    """
    if B <= 0:
        raise ValueError("Field must be positive")
    omega_c = charge * B / mass
    return v_perp / omega_c


def get_particle_mass(particle: str) -> float:
    """
    Get mass of common particles.

    Args:
        particle: "electron", "proton", "alpha", "oxygen" (O+), etc.

    Returns:
        Mass in kg
    """
    masses = {
        "electron": ELECTRON_MASS,
        "e": ELECTRON_MASS,
        "e-": ELECTRON_MASS,
        "proton": PROTON_MASS,
        "p": PROTON_MASS,
        "p+": PROTON_MASS,
        "H+": PROTON_MASS,
        "alpha": 4 * PROTON_MASS,
        "He++": 4 * PROTON_MASS,
        "oxygen": 16 * PROTON_MASS,
        "O+": 16 * PROTON_MASS,
    }
    if particle.lower() in masses:
        return masses[particle.lower()]
    # Case-sensitive fallback
    if particle in masses:
        return masses[particle]
    raise ValueError(f"Unknown particle: {particle}. Available: {list(masses.keys())}")
