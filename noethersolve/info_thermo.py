"""Information-thermodynamics calculator — the Landauer-Shannon connection.

This module bridges information theory and thermodynamics, implementing tools
for the fundamental relationship: erasing information has a minimum energy cost.

Key insight: Both Huffman coding and Landauer's principle optimize the same
objective: Σ p_i × cost_i where cost ∝ -log(p_i). Shannon entropy bounds
both the bits needed for compression AND the work extractable from a system.

Physical constants:
    k_B = 1.380649e-23 J/K (Boltzmann constant)
    ln(2) ≈ 0.693 (conversion factor bits ↔ nats)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


# Constants
K_B = 1.380649e-23  # Boltzmann constant J/K
LN_2 = math.log(2)  # ≈ 0.693


@dataclass
class LandauerReport:
    """Result of Landauer bound calculation."""
    bits_erased: float
    temperature: float  # K
    min_energy_joules: float  # J
    min_energy_eV: float  # eV
    min_energy_kT: float  # in units of kT
    entropy_increase: float  # J/K (environment)

    def __str__(self) -> str:
        lines = [
            "Landauer Bound:",
            f"  Bits erased: {self.bits_erased:.3g}",
            f"  Temperature: {self.temperature:.1f} K",
            f"  Minimum energy: {self.min_energy_joules:.3e} J",
            f"                  {self.min_energy_eV:.3e} eV",
            f"                  {self.min_energy_kT:.3f} kT per bit",
            f"  Environment ΔS: {self.entropy_increase:.3e} J/K",
            "",
            "  Physics: Erasing 1 bit requires dissipating kT·ln(2) of heat.",
            "           This increases environment entropy by k·ln(2).",
        ]
        return "\n".join(lines)


def calc_landauer_bound(
    bits_erased: float,
    temperature: float = 300.0,
) -> LandauerReport:
    """Calculate the minimum energy required to erase information.

    Landauer's principle (1961): Erasing one bit of information requires
    dissipating at least kT·ln(2) of energy as heat, where k is Boltzmann's
    constant and T is the temperature of the heat bath.

    This is a fundamental limit connecting information and thermodynamics.
    It explains why Maxwell's demon cannot violate the second law: the demon
    must erase its memory, which costs exactly the energy it extracted.

    Args:
        bits_erased: Number of bits to erase (can be fractional for
            probabilistic erasure or partial reset operations)
        temperature: Temperature of heat bath in Kelvin (default: 300 K)

    Returns:
        LandauerReport with minimum energy in various units

    Example:
        >>> report = calc_landauer_bound(1.0, 300)
        >>> print(f"{report.min_energy_joules:.3e} J")
        2.867e-21 J
        >>> print(f"{report.min_energy_kT:.3f} kT")
        0.693 kT

    Note:
        At room temperature (300 K), erasing 1 bit costs ~2.87 × 10⁻²¹ J.
        Modern computers dissipate ~10⁴ times this limit per bit operation.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    if bits_erased < 0:
        raise ValueError(f"bits_erased must be non-negative, got {bits_erased}")

    # E_min = n_bits × kT × ln(2)
    min_energy = bits_erased * K_B * temperature * LN_2

    # Convert to other units
    eV_per_joule = 6.241509074e18
    min_energy_eV = min_energy * eV_per_joule

    # Energy per bit in units of kT
    min_energy_kT = LN_2  # Always ln(2) ≈ 0.693 kT per bit

    # Entropy increase in environment = Q/T = E/T
    entropy_increase = min_energy / temperature  # = n_bits × k × ln(2)

    return LandauerReport(
        bits_erased=bits_erased,
        temperature=temperature,
        min_energy_joules=min_energy,
        min_energy_eV=min_energy_eV,
        min_energy_kT=min_energy_kT,
        entropy_increase=entropy_increase,
    )


@dataclass
class ShannonEntropyReport:
    """Result of Shannon entropy calculation."""
    entropy_bits: float  # H in bits
    entropy_nats: float  # H in nats (natural units)
    min_bits_per_symbol: float  # lower bound for compression
    probabilities: List[float]
    n_symbols: int
    max_entropy: float  # log2(n_symbols)
    efficiency: float  # H / H_max

    def __str__(self) -> str:
        lines = [
            "Shannon Entropy:",
            f"  H = {self.entropy_bits:.4f} bits",
            f"    = {self.entropy_nats:.4f} nats",
            f"  Minimum bits/symbol: {self.min_bits_per_symbol:.4f}",
            f"  Maximum possible: {self.max_entropy:.4f} bits (uniform)",
            f"  Efficiency: {self.efficiency:.1%}",
            f"  Symbols: {self.n_symbols}",
        ]
        return "\n".join(lines)


def calc_shannon_entropy(
    probabilities: List[float],
    base: str = "bits",
) -> ShannonEntropyReport:
    """Calculate Shannon entropy of a probability distribution.

    H = -Σ p_i × log(p_i)

    This is the fundamental measure of information content and uncertainty.
    It gives the minimum average number of bits needed to encode symbols
    drawn from this distribution (Shannon's source coding theorem).

    The connection to thermodynamics: Shannon entropy in nats equals
    Gibbs entropy divided by Boltzmann's constant. Erasing H bits of
    information requires dissipating at least H × kT × ln(2) of energy.

    Args:
        probabilities: List of probabilities (must sum to 1)
        base: "bits" (log base 2) or "nats" (natural log)

    Returns:
        ShannonEntropyReport with entropy in both units

    Example:
        >>> report = calc_shannon_entropy([0.5, 0.5])
        >>> print(f"H = {report.entropy_bits:.3f} bits")
        H = 1.000 bits

        >>> report = calc_shannon_entropy([0.9, 0.1])
        >>> print(f"H = {report.entropy_bits:.3f} bits")
        H = 0.469 bits
    """
    probs = [p for p in probabilities if p > 0]  # Filter zeros

    if not probs:
        raise ValueError("At least one non-zero probability required")

    total = sum(probs)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Probabilities must sum to 1, got {total}")

    # H = -Σ p_i × log(p_i)
    entropy_nats = -sum(p * math.log(p) for p in probs)
    entropy_bits = entropy_nats / LN_2

    n_symbols = len(probabilities)
    max_entropy = math.log2(n_symbols)
    efficiency = entropy_bits / max_entropy if max_entropy > 0 else 1.0

    return ShannonEntropyReport(
        entropy_bits=entropy_bits,
        entropy_nats=entropy_nats,
        min_bits_per_symbol=entropy_bits,
        probabilities=list(probabilities),
        n_symbols=n_symbols,
        max_entropy=max_entropy,
        efficiency=efficiency,
    )


@dataclass
class InfoThermoBridgeReport:
    """Unified report connecting information and thermodynamics."""
    probabilities: List[float]
    temperature: float
    shannon_entropy_bits: float
    landauer_energy_joules: float
    gibbs_entropy_JK: float  # S = k × H_nats
    free_energy_joules: float  # F = -kT × ln(Z) where Z = Σ exp(-E_i/kT)

    def __str__(self) -> str:
        lines = [
            "Information-Thermodynamics Bridge:",
            f"  Shannon entropy: {self.shannon_entropy_bits:.4f} bits",
            f"  Gibbs entropy: {self.gibbs_entropy_JK:.3e} J/K",
            f"  Landauer cost to erase: {self.landauer_energy_joules:.3e} J",
            "",
            "  Key identity:",
            "    S_Gibbs = k_B × H_nats",
            "    E_erase = T × S_Gibbs = kT × H_nats = kT × ln(2) × H_bits",
        ]
        return "\n".join(lines)


def calc_info_thermo_bridge(
    probabilities: List[float],
    temperature: float = 300.0,
) -> InfoThermoBridgeReport:
    """Unify information-theoretic and thermodynamic perspectives.

    Shows the deep connection between Shannon entropy (information theory)
    and Gibbs entropy (thermodynamics):

        S_Gibbs = k_B × H_nats

    And the energy cost of information erasure:

        E_erase = T × S_Gibbs = k_B × T × H_nats = k_B × T × ln(2) × H_bits

    Args:
        probabilities: Probability distribution over states
        temperature: Temperature in Kelvin

    Returns:
        InfoThermoBridgeReport unifying both perspectives
    """
    shannon = calc_shannon_entropy(probabilities)
    landauer = calc_landauer_bound(shannon.entropy_bits, temperature)

    # Gibbs entropy S = k × H_nats
    gibbs_entropy = K_B * shannon.entropy_nats

    # Free energy F = -kT ln(Z) where for uniform distribution Z = n
    # For general distribution, F = -kT × Σ p_i × ln(p_i) - kT × ln(n)
    # But the relevant quantity here is the entropy contribution
    free_energy = -temperature * gibbs_entropy

    return InfoThermoBridgeReport(
        probabilities=list(probabilities),
        temperature=temperature,
        shannon_entropy_bits=shannon.entropy_bits,
        landauer_energy_joules=landauer.min_energy_joules,
        gibbs_entropy_JK=gibbs_entropy,
        free_energy_joules=free_energy,
    )


@dataclass
class HuffmanLandauerReport:
    """Shows the parallel between Huffman coding and Landauer erasure."""
    probabilities: List[float]
    temperature: float

    # Huffman perspective
    shannon_entropy: float  # bits - optimal code length
    huffman_bound: float  # minimum average bits per symbol

    # Landauer perspective
    min_erase_energy: float  # J - minimum energy to erase state
    energy_per_bit: float  # J/bit at this temperature

    # The connection
    objective: str  # What both optimize

    def __str__(self) -> str:
        lines = [
            "Huffman-Landauer Parallel:",
            "",
            "  HUFFMAN (Information):",
            f"    Shannon entropy H = {self.shannon_entropy:.4f} bits",
            f"    Minimum avg code length = {self.huffman_bound:.4f} bits/symbol",
            "    Optimizes: Σ p_i × len(code_i)",
            "",
            "  LANDAUER (Thermodynamics):",
            f"    Energy per bit = {self.energy_per_bit:.3e} J",
            f"    Min erasure energy = {self.min_erase_energy:.3e} J",
            "    Optimizes: Σ p_i × E_i where E_i ∝ -log(p_i)",
            "",
            "  THE CONNECTION:",
            "    Both optimize Σ p_i × cost_i where cost ∝ -log(p_i)",
            "    Huffman: cost = code length (bits)",
            "    Landauer: cost = erasure energy (kT × ln(2) per bit)",
            "",
            "    Shannon entropy bounds BOTH:",
            "    - Minimum bits for lossless compression",
            "    - Minimum energy for irreversible erasure",
        ]
        return "\n".join(lines)


def calc_huffman_landauer_parallel(
    probabilities: List[float],
    temperature: float = 300.0,
) -> HuffmanLandauerReport:
    """Show the structural parallel between Huffman coding and Landauer erasure.

    Both Huffman coding (optimal prefix-free codes) and Landauer's principle
    (minimum erasure energy) optimize the same objective function:

        minimize Σ p_i × cost_i

    where cost_i ∝ -log(p_i).

    For Huffman: cost_i = length of codeword for symbol i
    For Landauer: cost_i = energy to erase state i

    Shannon entropy H = -Σ p_i × log(p_i) is the optimal value of both,
    which is why it bounds both compression and thermodynamic work.

    Args:
        probabilities: Probability distribution
        temperature: Temperature in Kelvin

    Returns:
        HuffmanLandauerReport showing the parallel
    """
    shannon = calc_shannon_entropy(probabilities)
    landauer = calc_landauer_bound(shannon.entropy_bits, temperature)

    return HuffmanLandauerReport(
        probabilities=list(probabilities),
        temperature=temperature,
        shannon_entropy=shannon.entropy_bits,
        huffman_bound=shannon.entropy_bits,  # H is the bound
        min_erase_energy=landauer.min_energy_joules,
        energy_per_bit=K_B * temperature * LN_2,
        objective="Σ p_i × cost_i where cost ∝ -log(p_i)",
    )
