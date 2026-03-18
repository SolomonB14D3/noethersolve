"""Chemistry calculator — derives answers from first principles.

Covers electrochemistry (Nernst equation), acid-base (Henderson-Hasselbalch),
crystal field theory (d-orbital splitting), and semiconductor physics (band gap).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


# Constants
R = 8.314462618  # J/(mol·K)
F = 96485.33212  # C/mol (Faraday constant)
K_W = 1e-14  # water autoionization constant at 25°C
K_B = 1.380649e-23  # Boltzmann constant J/K
E_CHARGE = 1.602176634e-19  # electron charge in C


@dataclass
class NernstReport:
    """Result of Nernst equation calculation."""
    E_standard: float  # V
    E_cell: float  # V
    n_electrons: int
    Q_reaction: float
    temperature: float  # K
    spontaneous: bool
    delta_G: float  # J/mol

    def __str__(self) -> str:
        lines = [
            "Nernst Equation Result:",
            f"  E° = {self.E_standard:.4f} V",
            f"  E  = {self.E_cell:.4f} V",
            f"  n  = {self.n_electrons} electrons transferred",
            f"  Q  = {self.Q_reaction:.4e}",
            f"  T  = {self.temperature:.1f} K",
            f"  ΔG = {self.delta_G:.1f} J/mol",
            f"  Spontaneous: {self.spontaneous}",
        ]
        return "\n".join(lines)


@dataclass
class BufferReport:
    """Result of Henderson-Hasselbalch calculation."""
    pH: float
    pKa: float
    acid_conc: float  # M
    base_conc: float  # M
    buffer_capacity: float  # mol/L per pH unit (approximate)
    effective_range: tuple  # (pH_low, pH_high)

    def __str__(self) -> str:
        lines = [
            "Henderson-Hasselbalch Buffer Calculation:",
            f"  pH  = {self.pH:.3f}",
            f"  pKa = {self.pKa:.3f}",
            f"  [HA] = {self.acid_conc:.4f} M",
            f"  [A⁻] = {self.base_conc:.4f} M",
            f"  Buffer capacity ≈ {self.buffer_capacity:.4f} mol/(L·pH)",
            f"  Effective range: pH {self.effective_range[0]:.1f} – {self.effective_range[1]:.1f}",
        ]
        return "\n".join(lines)


@dataclass
class CrystalFieldReport:
    """Result of crystal field splitting calculation."""
    geometry: str  # octahedral, tetrahedral, square_planar
    d_electrons: int
    delta_o: float  # crystal field splitting parameter (in Dq units)
    cfse: float  # crystal field stabilization energy (in Dq units)
    spin_state: str  # high_spin or low_spin
    unpaired_electrons: int
    configuration: str  # e.g., "t2g^3 eg^0"

    def __str__(self) -> str:
        lines = [
            "Crystal Field Theory Result:",
            f"  Geometry: {self.geometry}",
            f"  d-electrons: {self.d_electrons}",
            f"  Δ = {self.delta_o:.1f} Dq",
            f"  CFSE = {self.cfse:.1f} Dq",
            f"  Spin state: {self.spin_state}",
            f"  Unpaired electrons: {self.unpaired_electrons}",
            f"  Configuration: {self.configuration}",
        ]
        return "\n".join(lines)


@dataclass
class BandGapReport:
    """Result of semiconductor band gap calculation."""
    band_gap_eV: float
    band_gap_J: float
    wavelength_nm: float  # absorption edge wavelength
    conductor_type: str  # conductor, semiconductor, insulator
    intrinsic_carrier_conc: float  # carriers/cm³ at given temperature
    temperature: float  # K

    def __str__(self) -> str:
        lines = [
            "Semiconductor Band Gap Analysis:",
            f"  Band gap = {self.band_gap_eV:.3f} eV ({self.band_gap_J:.3e} J)",
            f"  Absorption edge = {self.wavelength_nm:.1f} nm",
            f"  Type: {self.conductor_type}",
            f"  Intrinsic carrier concentration = {self.intrinsic_carrier_conc:.3e} /cm³",
            f"  Temperature = {self.temperature:.1f} K",
        ]
        return "\n".join(lines)


@dataclass
class ChemCalcReport:
    """Combined chemistry calculation report."""
    calculation: str
    details: str
    issues: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Chemistry Calculator — {self.calculation}", self.details]
        if self.issues:
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  ⚠ {issue}")
        return "\n".join(lines)


def nernst_equation(
    E_standard: float,
    n_electrons: int,
    Q: float,
    temperature: float = 298.15,
) -> NernstReport:
    """Calculate cell potential using the Nernst equation.

    E = E° - (RT/nF) ln(Q)

    Args:
        E_standard: Standard cell potential (V)
        n_electrons: Number of electrons transferred
        Q: Reaction quotient
        temperature: Temperature in Kelvin (default 298.15 K)

    Returns:
        NernstReport with cell potential and thermodynamic data.
    """
    if n_electrons <= 0:
        raise ValueError("n_electrons must be positive")
    if Q <= 0:
        raise ValueError("Reaction quotient Q must be positive")
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    E_cell = E_standard - (R * temperature / (n_electrons * F)) * math.log(Q)
    delta_G = -n_electrons * F * E_cell
    spontaneous = E_cell > 0

    return NernstReport(
        E_standard=E_standard,
        E_cell=E_cell,
        n_electrons=n_electrons,
        Q_reaction=Q,
        temperature=temperature,
        spontaneous=spontaneous,
        delta_G=delta_G,
    )


def henderson_hasselbalch(
    pKa: float,
    acid_conc: float,
    base_conc: float,
) -> BufferReport:
    """Calculate buffer pH using Henderson-Hasselbalch equation.

    pH = pKa + log10([A⁻]/[HA])

    Args:
        pKa: Acid dissociation constant (-log10 Ka)
        acid_conc: Concentration of weak acid [HA] in M
        base_conc: Concentration of conjugate base [A⁻] in M

    Returns:
        BufferReport with pH and buffer properties.
    """
    if acid_conc <= 0 or base_conc <= 0:
        raise ValueError("Concentrations must be positive")

    pH = pKa + math.log10(base_conc / acid_conc)

    # Buffer capacity (Van Slyke equation approximation)
    C_total = acid_conc + base_conc
    Ka = 10 ** (-pKa)
    H = 10 ** (-pH)
    beta = 2.303 * C_total * Ka * H / (Ka + H) ** 2

    # Effective range: pKa ± 1
    effective_range = (pKa - 1.0, pKa + 1.0)

    return BufferReport(
        pH=pH,
        pKa=pKa,
        acid_conc=acid_conc,
        base_conc=base_conc,
        buffer_capacity=beta,
        effective_range=effective_range,
    )


def crystal_field_splitting(
    d_electrons: int,
    geometry: str = "octahedral",
    strong_field: bool = False,
) -> CrystalFieldReport:
    """Calculate crystal field splitting for transition metal complexes.

    Args:
        d_electrons: Number of d-electrons (1-10)
        geometry: "octahedral", "tetrahedral", or "square_planar"
        strong_field: If True, use low-spin configuration; if False, high-spin.

    Returns:
        CrystalFieldReport with CFSE and electron configuration.
    """
    if not 1 <= d_electrons <= 10:
        raise ValueError("d_electrons must be between 1 and 10")
    geometry = geometry.lower()
    if geometry not in ("octahedral", "tetrahedral", "square_planar"):
        raise ValueError("geometry must be octahedral, tetrahedral, or square_planar")

    if geometry == "octahedral":
        delta = 10.0  # Dq units (Δo = 10 Dq by definition)
        # t2g (lower, -4Dq each) and eg (upper, +6Dq each)
        # High spin: fill singly first, then pair
        # Low spin: fill t2g completely before eg
        if strong_field:  # low spin
            configs = {
                1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (4, 0), 5: (5, 0),
                6: (6, 0), 7: (6, 1), 8: (6, 2), 9: (6, 3), 10: (6, 4),
            }
        else:  # high spin
            configs = {
                1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (3, 1), 5: (3, 2),
                6: (4, 2), 7: (5, 2), 8: (6, 2), 9: (6, 3), 10: (6, 4),
            }
        t2g, eg = configs[d_electrons]
        # CFSE = -0.4 * t2g + 0.6 * eg (in Δo units)
        cfse = -0.4 * t2g + 0.6 * eg
        cfse_dq = cfse * 10  # convert to Dq
        config_str = f"t2g^{t2g} eg^{eg}"

        # Count unpaired electrons
        if strong_field:
            # t2g: 3 orbitals, fill pairs first
            t2g_unpaired = max(0, t2g - 3) if t2g > 3 else t2g if t2g <= 3 else 6 - t2g
            if t2g <= 3:
                t2g_unpaired = t2g
            else:
                t2g_unpaired = 6 - t2g
            eg_unpaired = eg if eg <= 2 else 4 - eg
        else:
            if t2g <= 3:
                t2g_unpaired = t2g
            else:
                t2g_unpaired = 6 - t2g
            if eg <= 2:
                eg_unpaired = eg
            else:
                eg_unpaired = 4 - eg
        unpaired = t2g_unpaired + eg_unpaired
        spin_state = "low_spin" if strong_field else "high_spin"

    elif geometry == "tetrahedral":
        delta = 4.44  # Δt ≈ 4/9 Δo
        # e (lower, -6Dq each) and t2 (upper, +4Dq each) — inverted from octahedral
        # Tetrahedral is almost always high spin (Δt too small for pairing)
        configs = {
            1: (1, 0), 2: (2, 0), 3: (2, 1), 4: (2, 2), 5: (2, 3),
            6: (3, 3), 7: (4, 3), 8: (4, 4), 9: (4, 5), 10: (4, 6),
        }
        e, t2 = configs[d_electrons]
        cfse = -0.6 * e + 0.4 * t2
        cfse_dq = cfse * delta
        config_str = f"e^{e} t2^{t2}"
        spin_state = "high_spin"  # tetrahedral is almost always high spin

        if e <= 2:
            e_unpaired = e
        else:
            e_unpaired = 4 - e
        if t2 <= 3:
            t2_unpaired = t2
        else:
            t2_unpaired = 6 - t2
        unpaired = e_unpaired + t2_unpaired

    else:  # square_planar
        delta = 10.0
        # Energy ordering (low to high): dxy, dxz/dyz, dz2, dx2-y2
        # Square planar strongly favors low spin for d8
        # Simplified: fill from bottom
        capacities = [2, 4, 2, 2]
        energies_dq = [-5.14, -2.28, -0.86, 12.28]  # approximate in Dq

        remaining = d_electrons
        filling = []
        for cap in capacities:
            n = min(remaining, cap)
            filling.append(n)
            remaining -= n

        cfse_dq = sum(f * e for f, e in zip(filling, energies_dq))
        config_str = f"dxy^{filling[0]} dxz/dyz^{filling[1]} dz2^{filling[2]} dx2-y2^{filling[3]}"
        spin_state = "low_spin"

        unpaired = 0
        for f, cap in zip(filling, capacities):
            half = cap // 2
            if f <= half:
                unpaired += f
            else:
                unpaired += cap - f

    return CrystalFieldReport(
        geometry=geometry,
        d_electrons=d_electrons,
        delta_o=delta,
        cfse=cfse_dq,
        spin_state=spin_state,
        unpaired_electrons=unpaired,
        configuration=config_str,
    )


def band_gap_analysis(
    band_gap_eV: float,
    temperature: float = 300.0,
) -> BandGapReport:
    """Analyze semiconductor properties from band gap energy.

    Calculates absorption edge wavelength, material classification,
    and intrinsic carrier concentration.

    Args:
        band_gap_eV: Band gap energy in electron volts
        temperature: Temperature in Kelvin (default 300 K)

    Returns:
        BandGapReport with derived semiconductor properties.
    """
    if band_gap_eV < 0:
        raise ValueError("Band gap must be non-negative")
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    band_gap_J = band_gap_eV * E_CHARGE

    # Absorption edge wavelength: λ = hc/Eg
    h = 6.62607015e-34  # Planck constant
    c = 2.998e8  # speed of light
    if band_gap_eV > 0:
        wavelength_nm = (h * c / band_gap_J) * 1e9
    else:
        wavelength_nm = float("inf")

    # Classification
    if band_gap_eV < 0.1:
        conductor_type = "conductor"
    elif band_gap_eV <= 4.0:
        conductor_type = "semiconductor"
    else:
        conductor_type = "insulator"

    # Intrinsic carrier concentration (simplified model)
    # ni ≈ A * T^(3/2) * exp(-Eg/(2kT))
    # A ≈ 4.83e15 cm^-3 K^(-3/2) for typical semiconductors
    A = 4.83e15
    kT = K_B * temperature
    if band_gap_eV > 0 and kT > 0:
        ni = A * temperature ** 1.5 * math.exp(-band_gap_J / (2 * kT))
    else:
        ni = A * temperature ** 1.5  # metallic

    return BandGapReport(
        band_gap_eV=band_gap_eV,
        band_gap_J=band_gap_J,
        wavelength_nm=wavelength_nm,
        conductor_type=conductor_type,
        intrinsic_carrier_conc=ni,
        temperature=temperature,
    )


def balance_redox(
    E_cathode: float,
    E_anode: float,
    n_electrons: int,
    temperature: float = 298.15,
) -> NernstReport:
    """Calculate standard cell potential and spontaneity for a redox reaction.

    Args:
        E_cathode: Standard reduction potential of cathode (V)
        E_anode: Standard reduction potential of anode (V)
        n_electrons: Electrons transferred
        temperature: Temperature in K

    Returns:
        NernstReport at standard conditions (Q=1).
    """
    E_standard = E_cathode - E_anode
    return nernst_equation(E_standard, n_electrons, Q=1.0, temperature=temperature)
