"""noethersolve.qm_calculator — Quantum mechanics computational engine.

Computes quantum mechanical quantities from first principles: energy levels,
tunneling probabilities, uncertainty relations, and angular momentum coupling.
Replaces static QM fact-checking with actual calculation.

Conservation law philosophy: quantum mechanics IS conservation laws.
The uncertainty principle is the conservation of information content in
conjugate bases (Fourier duality). Energy quantization is a boundary
condition constraint. Angular momentum addition rules are Clebsch-Gordan
coefficients from group theory (SU(2) representation decomposition).

Usage:
    from noethersolve.qm_calculator import (
        particle_in_box, hydrogen_energy, uncertainty_check,
        tunneling_probability, harmonic_oscillator, angular_momentum_addition,
    )

    # Particle in a box
    r = particle_in_box(n=3, L=1e-9, m=9.109e-31)
    print(r)  # E_3 = 3.39 eV

    # Hydrogen atom
    r = hydrogen_energy(n=2)
    print(r)  # E_2 = -3.4 eV, radius = 2.12 Å

    # Tunneling through a barrier
    r = tunneling_probability(E=5.0, V=10.0, L=1e-10, m=9.109e-31)
    print(r)  # T ≈ 0.68

    # Check uncertainty principle
    r = uncertainty_check(delta_x=1e-10, delta_p=1e-24)
    print(r)  # SATISFIED: Δx·Δp = 1e-34 ≥ ℏ/2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─── Physical Constants ──────────────────────────────────────────────────────

HBAR = 1.054571817e-34    # J·s (reduced Planck constant)
H_PLANCK = 6.62607015e-34 # J·s (Planck constant)
ME = 9.1093837015e-31     # kg (electron mass)
MP = 1.67262192369e-27    # kg (proton mass)
E_CHARGE = 1.602176634e-19  # C (elementary charge)
K_COULOMB = 8.9875517873681764e9  # N·m²/C² (Coulomb constant)
A0 = 5.29177210903e-11    # m (Bohr radius)
EV_TO_J = 1.602176634e-19  # J per eV
ALPHA = 7.2973525693e-3   # fine-structure constant
C_LIGHT = 299792458.0     # m/s


# ─── Report Dataclasses ──────────────────────────────────────────────────────

@dataclass
class ParticleInBoxReport:
    """Energy levels and wavefunctions for particle in an infinite square well."""
    n: int               # quantum number
    L: float             # box length (m)
    m: float             # particle mass (kg)
    E_n_J: float         # energy in Joules
    E_n_eV: float        # energy in eV
    wavelength: float    # de Broglie wavelength (m)
    nodes: int           # number of interior nodes (n-1)
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Particle in a Box (Infinite Square Well)", "=" * 60]
        lines.append(f"  n = {self.n}    L = {self.L:.4g} m    m = {self.m:.4g} kg")
        lines.append(f"  E_n = n²π²ℏ² / (2mL²) = {self.E_n_eV:.4g} eV ({self.E_n_J:.4g} J)")
        lines.append(f"  de Broglie wavelength = 2L/n = {self.wavelength:.4g} m")
        lines.append(f"  Interior nodes: {self.nodes}")
        lines.append("-" * 60)
        lines.append(f"  ψ_n(x) = √(2/L) sin(nπx/L)")
        lines.append(f"  Energy scales as n² — quadratic spacing")
        if self.notes:
            for note in self.notes:
                lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class HydrogenEnergyReport:
    """Hydrogen atom energy levels and orbital properties."""
    n: int               # principal quantum number
    E_n_eV: float        # energy in eV
    E_n_J: float         # energy in Joules
    radius_m: float      # expected orbital radius (m)
    radius_A: float      # expected orbital radius (Ångström)
    ionization_eV: float # ionization energy from this level
    degeneracy: int      # total degeneracy (2n²)
    wavelength_nm: Optional[float] = None  # transition wavelength from n→1
    Z: int = 1           # nuclear charge
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Hydrogen Atom Energy Levels", "=" * 60]
        atom = "Hydrogen" if self.Z == 1 else f"Hydrogen-like (Z={self.Z})"
        lines.append(f"  {atom}, n = {self.n}")
        lines.append(f"  E_n = -13.6 × Z²/n² = {self.E_n_eV:.4g} eV")
        lines.append(f"  Orbital radius = a₀ × n²/Z = {self.radius_A:.4g} Å ({self.radius_m:.4g} m)")
        lines.append(f"  Ionization energy = {self.ionization_eV:.4g} eV")
        lines.append(f"  Degeneracy = 2n² = {self.degeneracy}")
        if self.wavelength_nm is not None and self.n > 1:
            lines.append(f"  Transition n→1 wavelength = {self.wavelength_nm:.2f} nm")
        lines.append("-" * 60)
        lines.append(f"  Fine structure correction: ΔE/E ~ α² ≈ {ALPHA**2:.2e}")
        lines.append(f"  Lamb shift (2S-2P): ~1057 MHz (QED effect)")
        if self.notes:
            for note in self.notes:
                lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class UncertaintyReport:
    """Heisenberg uncertainty principle check."""
    delta_x: float        # position uncertainty (m)
    delta_p: float        # momentum uncertainty (kg·m/s)
    product: float        # Δx·Δp (J·s)
    hbar_over_2: float    # ℏ/2 (J·s)
    ratio: float          # product / (ℏ/2)
    satisfied: bool
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Heisenberg Uncertainty Principle Check", "=" * 60]
        lines.append(f"  Δx = {self.delta_x:.4g} m")
        lines.append(f"  Δp = {self.delta_p:.4g} kg·m/s")
        lines.append(f"  Δx·Δp = {self.product:.4g} J·s")
        lines.append(f"  ℏ/2 = {self.hbar_over_2:.4g} J·s")
        lines.append(f"  Ratio = Δx·Δp / (ℏ/2) = {self.ratio:.4g}")
        status = "SATISFIED" if self.satisfied else "VIOLATED"
        lines.append(f"  Status: {status}")
        lines.append("-" * 60)
        if self.satisfied:
            lines.append("  The proposed uncertainties are consistent with quantum mechanics.")
            if self.ratio < 2:
                lines.append("  Near the minimum uncertainty — this is a coherent state.")
        else:
            lines.append("  VIOLATION: these uncertainties cannot coexist in quantum mechanics.")
            lines.append("  Either Δx is too small for this Δp, or vice versa.")
        if self.notes:
            for note in self.notes:
                lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class TunnelingReport:
    """Quantum tunneling probability through a rectangular barrier."""
    E: float              # particle energy (eV)
    V: float              # barrier height (eV)
    L: float              # barrier width (m)
    m: float              # particle mass (kg)
    T: float              # transmission coefficient
    R: float              # reflection coefficient (1 - T)
    kappa: float          # decay constant (1/m)
    regime: str           # "tunneling" or "above_barrier"
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Quantum Tunneling (Rectangular Barrier)", "=" * 60]
        lines.append(f"  E = {self.E:.4g} eV    V₀ = {self.V:.4g} eV    L = {self.L:.4g} m")
        lines.append(f"  m = {self.m:.4g} kg")
        lines.append(f"  Regime: {self.regime}")
        lines.append("-" * 60)
        if self.regime == "tunneling":
            lines.append(f"  κ = √(2m(V-E))/ℏ = {self.kappa:.4g} m⁻¹")
            lines.append(f"  κL = {self.kappa * self.L:.4g}")
            lines.append(f"  T = 1/(1 + V²sinh²(κL)/(4E(V-E))) = {self.T:.6g}")
        else:
            lines.append(f"  k = √(2m(E-V))/ℏ = {self.kappa:.4g} m⁻¹")
            lines.append(f"  T = 1/(1 + V²sin²(kL)/(4E(E-V))) = {self.T:.6g}")
        lines.append(f"  R = 1 - T = {self.R:.6g}")
        if self.notes:
            for note in self.notes:
                lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class HarmonicOscillatorReport:
    """Quantum harmonic oscillator energy levels."""
    n: int               # quantum number (0, 1, 2, ...)
    omega: float         # angular frequency (rad/s)
    m: float             # particle mass (kg)
    E_n_J: float         # energy in Joules
    E_n_eV: float        # energy in eV
    zero_point_J: float  # zero-point energy in Joules
    zero_point_eV: float # zero-point energy in eV
    classical_amplitude: float  # classical turning point
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Quantum Harmonic Oscillator", "=" * 60]
        lines.append(f"  n = {self.n}    ω = {self.omega:.4g} rad/s    m = {self.m:.4g} kg")
        lines.append(f"  E_n = (n + ½)ℏω = {self.E_n_eV:.4g} eV ({self.E_n_J:.4g} J)")
        lines.append(f"  Zero-point energy = ½ℏω = {self.zero_point_eV:.4g} eV")
        lines.append(f"  Classical turning point = {self.classical_amplitude:.4g} m")
        lines.append("-" * 60)
        lines.append(f"  Energy spacing = ℏω = {HBAR * self.omega / EV_TO_J:.4g} eV (equally spaced)")
        lines.append(f"  ψ_n(x) = H_n(αx) × exp(-α²x²/2), α = √(mω/ℏ)")
        if self.notes:
            for note in self.notes:
                lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class AngularMomentumReport:
    """Angular momentum addition (Clebsch-Gordan decomposition)."""
    j1: float            # first angular momentum quantum number
    j2: float            # second angular momentum quantum number
    j_min: float         # minimum total J
    j_max: float         # maximum total J
    allowed_j: List[float]  # all allowed J values
    total_states: int    # (2j1+1)(2j2+1) — must equal sum of (2J+1) for each J
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Angular Momentum Addition", "=" * 60]
        lines.append(f"  j₁ = {self.j1}    j₂ = {self.j2}")
        lines.append(f"  |j₁ - j₂| ≤ J ≤ j₁ + j₂")
        lines.append(f"  J_min = {self.j_min}    J_max = {self.j_max}")
        lines.append(f"  Allowed J values: {self.allowed_j}")
        lines.append(f"  Total states: (2j₁+1)(2j₂+1) = {self.total_states}")
        lines.append("-" * 60)
        lines.append("  Decomposition:")
        check_sum = 0
        for j in self.allowed_j:
            deg = int(2 * j + 1)
            check_sum += deg
            lines.append(f"    J = {j}: {deg} states (m = {-j} to {j})")
        lines.append(f"  Check: Σ(2J+1) = {check_sum} = {self.total_states} ✓" if check_sum == self.total_states
                      else f"  Check: Σ(2J+1) = {check_sum} ≠ {self.total_states} ✗")
        if self.notes:
            for note in self.notes:
                lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Public API ──────────────────────────────────────────────────────────────

def particle_in_box(
    n: int,
    L: float,
    m: float = ME,
) -> ParticleInBoxReport:
    """Compute energy levels for a particle in an infinite square well.

    E_n = n²π²ℏ² / (2mL²)

    Args:
        n: quantum number (1, 2, 3, ...)
        L: box length in meters
        m: particle mass in kg (default: electron mass)

    Returns:
        ParticleInBoxReport with energy, wavelength, and node count.
    """
    if n < 1:
        raise ValueError(f"Quantum number n must be ≥ 1, got {n}")
    if L <= 0:
        raise ValueError(f"Box length L must be positive, got {L}")
    if m <= 0:
        raise ValueError(f"Mass must be positive, got {m}")

    E_J = (n ** 2) * (math.pi ** 2) * (HBAR ** 2) / (2 * m * L ** 2)
    E_eV = E_J / EV_TO_J
    wavelength = 2 * L / n
    nodes = n - 1

    notes = []
    if E_eV > 1e6:
        notes.append("Very high energy — relativistic corrections may be needed")
    if n == 1:
        notes.append("Ground state — minimum energy from confinement (zero-point energy)")

    return ParticleInBoxReport(
        n=n, L=L, m=m, E_n_J=E_J, E_n_eV=E_eV,
        wavelength=wavelength, nodes=nodes, notes=notes,
    )


def hydrogen_energy(
    n: int,
    Z: int = 1,
) -> HydrogenEnergyReport:
    """Compute hydrogen atom (or hydrogen-like ion) energy levels.

    E_n = -13.6 × Z² / n² eV  (non-relativistic, Bohr model exact)

    Args:
        n: principal quantum number (1, 2, 3, ...)
        Z: nuclear charge (1 for hydrogen, 2 for He+, etc.)

    Returns:
        HydrogenEnergyReport with energy, radius, degeneracy, and transition wavelength.
    """
    if n < 1:
        raise ValueError(f"Principal quantum number n must be ≥ 1, got {n}")
    if Z < 1:
        raise ValueError(f"Nuclear charge Z must be ≥ 1, got {Z}")

    E_eV = -13.6 * Z ** 2 / n ** 2
    E_J = E_eV * EV_TO_J
    radius_m = A0 * n ** 2 / Z
    radius_A = radius_m * 1e10
    ionization_eV = -E_eV  # energy needed to ionize from this level
    degeneracy = 2 * n ** 2  # including spin

    # Transition wavelength n → 1 (Lyman series for hydrogen)
    wavelength_nm = None
    if n > 1:
        delta_E_eV = 13.6 * Z ** 2 * (1 - 1 / n ** 2)
        delta_E_J = delta_E_eV * EV_TO_J
        wavelength_m = H_PLANCK * C_LIGHT / delta_E_J
        wavelength_nm = wavelength_m * 1e9

    notes = []
    if n == 1:
        notes.append("Ground state — Lyman series terminates here")
    if n == 2:
        notes.append("First excited state — Balmer series starts from here (visible light)")
    if Z > 1:
        notes.append(f"Hydrogen-like ion: energy scales as Z², radius scales as 1/Z")

    return HydrogenEnergyReport(
        n=n, E_n_eV=E_eV, E_n_J=E_J, radius_m=radius_m,
        radius_A=radius_A, ionization_eV=ionization_eV,
        degeneracy=degeneracy, wavelength_nm=wavelength_nm,
        Z=Z, notes=notes,
    )


def uncertainty_check(
    delta_x: float,
    delta_p: float,
) -> UncertaintyReport:
    """Check whether proposed position and momentum uncertainties satisfy
    the Heisenberg uncertainty principle.

    Δx · Δp ≥ ℏ/2

    Args:
        delta_x: position uncertainty in meters
        delta_p: momentum uncertainty in kg·m/s

    Returns:
        UncertaintyReport with product, ratio, and satisfaction status.
    """
    if delta_x <= 0:
        raise ValueError(f"Δx must be positive, got {delta_x}")
    if delta_p <= 0:
        raise ValueError(f"Δp must be positive, got {delta_p}")

    product = delta_x * delta_p
    hbar_2 = HBAR / 2
    ratio = product / hbar_2
    satisfied = product >= hbar_2 * (1 - 1e-10)  # tiny tolerance for float

    notes = []
    if satisfied and ratio < 1.01:
        notes.append("Minimum uncertainty state (coherent state / Gaussian wavepacket)")
    if not satisfied:
        notes.append("This is NOT a measurement apparatus limitation — it is a fundamental "
                     "property of wave mechanics (Fourier duality of conjugate variables)")

    return UncertaintyReport(
        delta_x=delta_x, delta_p=delta_p, product=product,
        hbar_over_2=hbar_2, ratio=ratio, satisfied=satisfied,
        notes=notes,
    )


def tunneling_probability(
    E: float,
    V: float,
    L: float,
    m: float = ME,
) -> TunnelingReport:
    """Compute quantum tunneling probability through a rectangular barrier.

    For E < V (tunneling regime):
        T = 1 / (1 + V₀²sinh²(κL) / (4E(V-E)))
        where κ = √(2m(V-E)) / ℏ

    For E > V (above barrier):
        T = 1 / (1 + V₀²sin²(kL) / (4E(E-V)))
        where k = √(2m(E-V)) / ℏ

    Args:
        E: particle energy in eV
        V: barrier height in eV
        L: barrier width in meters
        m: particle mass in kg (default: electron mass)

    Returns:
        TunnelingReport with transmission and reflection coefficients.
    """
    if E <= 0:
        raise ValueError(f"Energy E must be positive, got {E}")
    if V <= 0:
        raise ValueError(f"Barrier height V must be positive, got {V}")
    if L <= 0:
        raise ValueError(f"Barrier width L must be positive, got {L}")
    if m <= 0:
        raise ValueError(f"Mass must be positive, got {m}")

    E_J = E * EV_TO_J
    V_J = V * EV_TO_J

    notes = []

    if abs(E - V) < 1e-10 * V:
        # E ≈ V: limiting case
        # T = 1 / (1 + mVL²/(2ℏ²))
        factor = m * V_J * L ** 2 / (2 * HBAR ** 2)
        T = 1.0 / (1.0 + factor)
        kappa = 0.0
        regime = "resonance"
        notes.append("E ≈ V: at barrier top, transmission depends on barrier width")
    elif E < V:
        # Tunneling regime
        kappa = math.sqrt(2 * m * (V_J - E_J)) / HBAR
        kappa_L = kappa * L
        if kappa_L > 500:
            T = 0.0
            notes.append(f"κL = {kappa_L:.1f} >> 1: tunneling probability is effectively zero")
        else:
            sinh_val = math.sinh(kappa_L)
            denom = 1 + (V ** 2 * sinh_val ** 2) / (4 * E * (V - E))
            T = 1.0 / denom
        regime = "tunneling"
        if kappa_L > 1:
            T_approx = (16 * E * (V - E) / V ** 2) * math.exp(-2 * kappa_L) if kappa_L < 500 else 0.0
            notes.append(f"WKB approximation: T ≈ {T_approx:.4g} (good for κL >> 1)")
    else:
        # Above barrier — oscillatory transmission
        k = math.sqrt(2 * m * (E_J - V_J)) / HBAR
        kappa = k
        k_L = k * L
        sin_val = math.sin(k_L)
        if abs(E * (E - V)) > 1e-30:
            denom = 1 + (V ** 2 * sin_val ** 2) / (4 * E * (E - V))
            T = 1.0 / denom
        else:
            T = 1.0
        regime = "above_barrier"
        if abs(sin_val) < 1e-10:
            notes.append("Resonance condition: kL = nπ, perfect transmission (T = 1)")

    R = 1.0 - T

    return TunnelingReport(
        E=E, V=V, L=L, m=m, T=T, R=R,
        kappa=kappa, regime=regime, notes=notes,
    )


def harmonic_oscillator(
    n: int,
    omega: float,
    m: float = ME,
) -> HarmonicOscillatorReport:
    """Compute quantum harmonic oscillator energy levels.

    E_n = (n + 1/2) ℏω

    Args:
        n: quantum number (0, 1, 2, ...)
        omega: angular frequency in rad/s
        m: particle mass in kg (default: electron mass)

    Returns:
        HarmonicOscillatorReport with energy, zero-point energy, and turning point.
    """
    if n < 0:
        raise ValueError(f"Quantum number n must be ≥ 0, got {n}")
    if omega <= 0:
        raise ValueError(f"Angular frequency ω must be positive, got {omega}")
    if m <= 0:
        raise ValueError(f"Mass must be positive, got {m}")

    E_J = (n + 0.5) * HBAR * omega
    E_eV = E_J / EV_TO_J
    zpe_J = 0.5 * HBAR * omega
    zpe_eV = zpe_J / EV_TO_J

    # Classical turning point: E = ½mω²x² → x = √(2E/(mω²))
    classical_amp = math.sqrt(2 * E_J / (m * omega ** 2))

    notes = []
    if n == 0:
        notes.append("Ground state: zero-point energy ½ℏω is nonzero — "
                     "direct consequence of the uncertainty principle")
    if n > 20:
        notes.append("High quantum number: approaching classical limit (correspondence principle)")

    return HarmonicOscillatorReport(
        n=n, omega=omega, m=m, E_n_J=E_J, E_n_eV=E_eV,
        zero_point_J=zpe_J, zero_point_eV=zpe_eV,
        classical_amplitude=classical_amp, notes=notes,
    )


def angular_momentum_addition(
    j1: float,
    j2: float,
) -> AngularMomentumReport:
    """Compute allowed total angular momentum values from coupling j1 and j2.

    J ranges from |j1 - j2| to j1 + j2 in integer steps.
    Total states: (2j1+1)(2j2+1) = Σ(2J+1) for each allowed J.

    Args:
        j1: first angular momentum quantum number (integer or half-integer ≥ 0)
        j2: second angular momentum quantum number (integer or half-integer ≥ 0)

    Returns:
        AngularMomentumReport with allowed J values and state counting.
    """
    if j1 < 0:
        raise ValueError(f"j1 must be non-negative, got {j1}")
    if j2 < 0:
        raise ValueError(f"j2 must be non-negative, got {j2}")
    # Check that j1, j2 are integer or half-integer
    if abs(2 * j1 - round(2 * j1)) > 1e-10:
        raise ValueError(f"j1 must be integer or half-integer, got {j1}")
    if abs(2 * j2 - round(2 * j2)) > 1e-10:
        raise ValueError(f"j2 must be integer or half-integer, got {j2}")

    j_max = j1 + j2
    j_min = abs(j1 - j2)

    # Build list of allowed J values
    allowed = []
    j = j_min
    while j <= j_max + 1e-10:
        allowed.append(round(2 * j) / 2)  # clean up float
        j += 1

    total_states = int((2 * j1 + 1) * (2 * j2 + 1))

    notes = []
    if j1 == 0.5 and j2 == 0.5:
        notes.append("Two spin-½: singlet (J=0, antisymmetric) + triplet (J=1, symmetric)")
    if j1 == j2:
        notes.append(f"Equal angular momenta: J ranges from 0 {'or ½' if j1 % 1 else ''} to {int(2*j1)}")

    return AngularMomentumReport(
        j1=j1, j2=j2, j_min=j_min, j_max=j_max,
        allowed_j=allowed, total_states=total_states,
        notes=notes,
    )
