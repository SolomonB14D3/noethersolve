"""Information theory calculations — exact formulas, not approximations.

Verified computational tools for:
- Channel capacity (BSC, BEC, AWGN, Z-channel)
- Rate-distortion theory
- Source coding bounds
- Multiple access channel regions
- Entropy calculations

Key insight: LLMs often confuse achievable rates with converse bounds,
or present exact results as approximations. These tools compute exactly.
"""

from dataclasses import dataclass
from math import log2, log
from typing import Optional, Tuple, List
import numpy as np


# ============================================================
# Entropy Functions
# ============================================================

def binary_entropy(p: float) -> float:
    """Binary entropy function H(p) = -p log p - (1-p) log(1-p).

    Returns entropy in bits.
    """
    if p <= 0 or p >= 1:
        return 0.0
    return -p * log2(p) - (1 - p) * log2(1 - p)


def entropy(probs: List[float]) -> float:
    """Shannon entropy H(X) = -Σ p_i log p_i.

    Args:
        probs: Probability distribution (must sum to 1)

    Returns:
        Entropy in bits
    """
    return -sum(p * log2(p) for p in probs if p > 0)


def relative_entropy(p: List[float], q: List[float]) -> float:
    """Kullback-Leibler divergence D(P||Q) = Σ p_i log(p_i/q_i).

    Returns infinity if Q has zeros where P is nonzero.
    """
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * log2(pi / qi)
    return total


def mutual_information(joint: List[List[float]]) -> float:
    """Mutual information I(X;Y) from joint distribution P(X,Y).

    Args:
        joint: 2D array of joint probabilities P(x,y)

    Returns:
        I(X;Y) in bits
    """
    joint = np.array(joint)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for i, pi in enumerate(px):
        for j, pj in enumerate(py):
            pxy = joint[i, j]
            if pxy > 0 and pi > 0 and pj > 0:
                mi += pxy * log2(pxy / (pi * pj))
    return mi


# ============================================================
# Channel Capacity
# ============================================================

@dataclass
class ChannelCapacityReport:
    """Result of channel capacity calculation."""
    channel_type: str
    capacity: float
    capacity_bits: str
    parameters: dict
    achieving_input: str
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        return (f"Channel: {self.channel_type}\n"
                f"  Capacity C = {self.capacity:.6f} bits/use\n"
                f"  Formula: {self.capacity_bits}\n"
                f"  Parameters: {self.parameters}\n"
                f"  Achieving input: {self.achieving_input}\n"
                f"  {self.explanation}")


def capacity_bsc(p: float) -> ChannelCapacityReport:
    """Capacity of Binary Symmetric Channel.

    BSC flips each bit independently with probability p.
    Capacity C = 1 - H(p) where H is binary entropy.

    Args:
        p: Crossover probability (0 to 0.5)

    Returns:
        ChannelCapacityReport with exact capacity
    """
    if p < 0 or p > 1:
        return ChannelCapacityReport(
            "BSC", 0, "invalid", {"p": p}, "N/A",
            f"Invalid crossover probability: {p}", False
        )

    H_p = binary_entropy(p)
    C = 1 - H_p

    return ChannelCapacityReport(
        "Binary Symmetric Channel (BSC)",
        C,
        f"C = 1 - H({p}) = 1 - {H_p:.6f} = {C:.6f}",
        {"crossover_prob": p, "H(p)": H_p},
        "Uniform: P(X=0) = P(X=1) = 0.5",
        "Exact result. Capacity achieved by uniform input distribution."
    )


def capacity_bec(epsilon: float) -> ChannelCapacityReport:
    """Capacity of Binary Erasure Channel.

    BEC erases each bit with probability ε, else transmits perfectly.
    Capacity C = 1 - ε (exact, simpler than BSC).

    Args:
        epsilon: Erasure probability

    Returns:
        ChannelCapacityReport with exact capacity
    """
    if epsilon < 0 or epsilon > 1:
        return ChannelCapacityReport(
            "BEC", 0, "invalid", {"epsilon": epsilon}, "N/A",
            f"Invalid erasure probability: {epsilon}", False
        )

    C = 1 - epsilon

    return ChannelCapacityReport(
        "Binary Erasure Channel (BEC)",
        C,
        f"C = 1 - ε = 1 - {epsilon} = {C:.6f}",
        {"erasure_prob": epsilon},
        "Uniform: P(X=0) = P(X=1) = 0.5",
        "Exact result. BEC capacity is linear in (1-ε), unlike BSC."
    )


def capacity_awgn(snr: float, bandwidth: float = 1.0) -> ChannelCapacityReport:
    """Capacity of Additive White Gaussian Noise channel.

    Shannon's formula: C = B × log₂(1 + SNR)

    Args:
        snr: Signal-to-noise ratio (linear, not dB)
        bandwidth: Channel bandwidth in Hz (default 1 for normalized)

    Returns:
        ChannelCapacityReport with exact capacity
    """
    if snr < 0:
        return ChannelCapacityReport(
            "AWGN", 0, "invalid", {"SNR": snr}, "N/A",
            f"Invalid SNR: {snr} (must be non-negative)", False
        )

    C = bandwidth * log2(1 + snr)
    snr_db = 10 * log(snr, 10) if snr > 0 else float('-inf')

    return ChannelCapacityReport(
        "Additive White Gaussian Noise (AWGN)",
        C,
        f"C = B × log₂(1 + SNR) = {bandwidth} × log₂(1 + {snr}) = {C:.6f}",
        {"SNR_linear": snr, "SNR_dB": snr_db, "bandwidth_Hz": bandwidth},
        "Gaussian: X ~ N(0, P) where P is signal power",
        "Shannon's exact formula (1948). Achieving distribution is Gaussian."
    )


def capacity_z_channel(p: float) -> ChannelCapacityReport:
    """Capacity of Z-channel (asymmetric binary channel).

    Z-channel: 0→0 always, 1→0 with probability p, 1→1 with probability 1-p.
    Capacity requires optimization over input distribution.

    Args:
        p: Probability of 1→0 transition

    Returns:
        ChannelCapacityReport with exact capacity
    """
    if p < 0 or p > 1:
        return ChannelCapacityReport(
            "Z-channel", 0, "invalid", {"p": p}, "N/A",
            f"Invalid transition probability: {p}", False
        )

    if p == 0:
        return ChannelCapacityReport(
            "Z-channel", 1.0, "C = 1 (noiseless)",
            {"p": 0}, "Any", "No errors, perfect channel."
        )
    if p == 1:
        return ChannelCapacityReport(
            "Z-channel", 0.0, "C = 0 (useless)",
            {"p": 1}, "N/A", "All 1s become 0s, no information transfer."
        )

    # Z-channel capacity formula (exact):
    # C = log2(1 + (1-p) * p^(p/(1-p)))
    # Optimal P(X=1) biases toward 0 (the reliable input)
    q = 1 - p
    term = q * (p ** (p / q))
    C = log2(1 + term)

    # Optimal input: P(X=1) = p^(p/(1-p)) / (1 + p^(p/(1-p)))
    # This is < 0.5 for all p > 0 (biased toward reliable input 0)
    exp_term = p ** (p / q)
    p1_opt = exp_term / (1 + exp_term)

    return ChannelCapacityReport(
        "Z-channel",
        C,
        f"C = log₂(1 + (1-p) × p^(p/(1-p))) = {C:.6f}",
        {"transition_prob": p, "optimal_P1": p1_opt},
        f"P(X=1) = {p1_opt:.6f} < 0.5 (biased toward reliable 0)",
        "Z-channel is asymmetric; optimal input biases toward noiseless input 0."
    )


# ============================================================
# Rate-Distortion Theory
# ============================================================

@dataclass
class RateDistortionReport:
    """Result of rate-distortion calculation."""
    source_type: str
    distortion_measure: str
    rate: float
    distortion: float
    formula: str
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        return (f"Rate-Distortion: {self.source_type}\n"
                f"  Distortion measure: {self.distortion_measure}\n"
                f"  R(D) = {self.rate:.6f} bits/symbol\n"
                f"  D = {self.distortion}\n"
                f"  Formula: {self.formula}\n"
                f"  {self.explanation}")


def rate_distortion_binary(D: float) -> RateDistortionReport:
    """Rate-distortion function for binary source with Hamming distortion.

    R(D) = H(p) - H(D) for D ≤ p
    R(D) = 0 for D > p

    For symmetric source (p=0.5): R(D) = 1 - H(D)

    Args:
        D: Target distortion (Hamming)

    Returns:
        RateDistortionReport
    """
    if D < 0 or D > 0.5:
        return RateDistortionReport(
            "Binary", "Hamming", 0, D,
            "invalid", f"Distortion must be in [0, 0.5], got {D}", False
        )

    R = 1 - binary_entropy(D) if D < 0.5 else 0

    return RateDistortionReport(
        "Binary symmetric (p=0.5)",
        "Hamming distance",
        R,
        D,
        f"R(D) = 1 - H(D) = 1 - H({D}) = {R:.6f}",
        "Exact result. At D=0, R=1 bit. At D=0.5, R=0 bits."
    )


def rate_distortion_gaussian(D: float, variance: float = 1.0) -> RateDistortionReport:
    """Rate-distortion function for Gaussian source with MSE distortion.

    R(D) = (1/2) log₂(σ²/D) for D ≤ σ²
    R(D) = 0 for D > σ²

    Args:
        D: Target distortion (MSE)
        variance: Source variance σ²

    Returns:
        RateDistortionReport
    """
    if D <= 0:
        return RateDistortionReport(
            "Gaussian", "MSE", float('inf'), D,
            "invalid", "Distortion must be positive", False
        )

    if D >= variance:
        R = 0
        formula = f"R(D) = 0 (D ≥ σ² = {variance})"
    else:
        R = 0.5 * log2(variance / D)
        formula = f"R(D) = (1/2) log₂(σ²/D) = (1/2) log₂({variance}/{D}) = {R:.6f}"

    return RateDistortionReport(
        "Gaussian N(0, σ²)",
        "Mean squared error",
        R,
        D,
        formula,
        f"Exact result. Source variance σ² = {variance}."
    )


# ============================================================
# Source Coding
# ============================================================

@dataclass
class SourceCodingReport:
    """Result of source coding bound calculation."""
    bound_type: str
    entropy: float
    rate: float
    efficiency: float
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        return (f"Source Coding: {self.bound_type}\n"
                f"  Source entropy H(X) = {self.entropy:.6f} bits/symbol\n"
                f"  Code rate = {self.rate:.6f} bits/symbol\n"
                f"  Efficiency = {self.efficiency:.2%}\n"
                f"  {self.explanation}")


def source_coding_bound(probs: List[float], code_lengths: Optional[List[int]] = None) -> SourceCodingReport:
    """Check source coding theorem bounds.

    For lossless compression:
    - Average code length L ≥ H(X) (lower bound)
    - L < H(X) + 1 is achievable (Shannon-Fano)
    - Huffman achieves L ≤ H(X) + 1

    Args:
        probs: Source probability distribution
        code_lengths: Optional actual code lengths to evaluate

    Returns:
        SourceCodingReport
    """
    H = entropy(probs)

    if code_lengths is None:
        return SourceCodingReport(
            "Shannon's source coding theorem",
            H,
            H,  # Optimal rate equals entropy
            1.0,
            f"Minimum achievable rate is H(X) = {H:.6f} bits/symbol. "
            "Any uniquely decodable code has L ≥ H(X)."
        )

    # Check Kraft inequality
    kraft_sum = sum(2**(-l) for l in code_lengths)
    kraft_satisfied = kraft_sum <= 1

    # Average length
    L = sum(p * l for p, l in zip(probs, code_lengths))
    efficiency = H / L if L > 0 else 0

    if not kraft_satisfied:
        return SourceCodingReport(
            "Invalid code",
            H, L, efficiency,
            f"Kraft inequality violated: Σ2^(-l_i) = {kraft_sum:.4f} > 1. "
            "No prefix-free code exists with these lengths.",
            False
        )

    return SourceCodingReport(
        "Prefix-free code analysis",
        H, L, efficiency,
        f"Kraft sum = {kraft_sum:.4f} ≤ 1. "
        f"Average length L = {L:.4f} ≥ H = {H:.4f}. "
        f"Redundancy = {L - H:.4f} bits/symbol."
    )


# ============================================================
# Multiple Access Channel
# ============================================================

@dataclass
class MACRegionReport:
    """Result of MAC capacity region calculation."""
    num_users: int
    rate_bounds: List[str]
    corner_points: List[Tuple[float, ...]]
    sum_rate: float
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        bounds_str = "\n    ".join(self.rate_bounds)
        corners_str = ", ".join(str(c) for c in self.corner_points)
        return (f"Multiple Access Channel ({self.num_users} users)\n"
                f"  Rate constraints:\n    {bounds_str}\n"
                f"  Sum-rate capacity: {self.sum_rate:.6f}\n"
                f"  Corner points: {corners_str}\n"
                f"  {self.explanation}")


def mac_capacity_region_2user(
    I_X1_Y: float,
    I_X2_Y: float,
    I_X1X2_Y: float,
) -> MACRegionReport:
    """Capacity region for 2-user Multiple Access Channel.

    The capacity region is the pentagon bounded by:
    - R1 ≤ I(X1; Y | X2)
    - R2 ≤ I(X2; Y | X1)
    - R1 + R2 ≤ I(X1, X2; Y)

    Args:
        I_X1_Y: I(X1; Y | X2) — user 1's rate treating user 2 as known
        I_X2_Y: I(X2; Y | X1) — user 2's rate treating user 1 as known
        I_X1X2_Y: I(X1, X2; Y) — sum rate capacity

    Returns:
        MACRegionReport describing the capacity region
    """
    # Sanity check: I(X1,X2;Y) ≤ I(X1;Y|X2) + I(X2;Y|X1) by chain rule
    if I_X1X2_Y > I_X1_Y + I_X2_Y + 1e-10:
        return MACRegionReport(
            2, [], [], 0,
            f"Invalid: I(X1,X2;Y) = {I_X1X2_Y} > I(X1;Y|X2) + I(X2;Y|X1) = {I_X1_Y + I_X2_Y}",
            False
        )

    bounds = [
        f"R₁ ≤ {I_X1_Y:.4f}",
        f"R₂ ≤ {I_X2_Y:.4f}",
        f"R₁ + R₂ ≤ {I_X1X2_Y:.4f}",
    ]

    # Corner points of the pentagon
    corners = [
        (0, 0),
        (I_X1_Y, 0),
        (I_X1X2_Y - I_X2_Y, I_X2_Y),  # Where sum-rate meets R2 bound
        (I_X1_Y, I_X1X2_Y - I_X1_Y),  # Where sum-rate meets R1 bound
        (0, I_X2_Y),
    ]

    return MACRegionReport(
        2,
        bounds,
        corners,
        I_X1X2_Y,
        "Capacity region is a PENTAGON (common error: models say it's rectangular). "
        "Corner points correspond to successive decoding orders."
    )


# ============================================================
# Data Processing Inequality
# ============================================================

@dataclass
class DataProcessingReport:
    """Result of data processing inequality check."""
    I_XY: float
    I_XZ: float
    satisfies_dpi: bool
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        status = "✓ SATISFIED" if self.satisfies_dpi else "✗ VIOLATED"
        return (f"Data Processing Inequality: {status}\n"
                f"  I(X; Y) = {self.I_XY:.6f}\n"
                f"  I(X; Z) = {self.I_XZ:.6f}\n"
                f"  {self.explanation}")


def check_data_processing(I_XY: float, I_XZ: float, chain: str = "X → Y → Z") -> DataProcessingReport:
    """Check data processing inequality.

    For Markov chain X → Y → Z:
        I(X; Z) ≤ I(X; Y)

    Processing cannot increase information.

    Args:
        I_XY: Mutual information I(X; Y)
        I_XZ: Mutual information I(X; Z)
        chain: Description of Markov chain

    Returns:
        DataProcessingReport
    """
    satisfies = I_XZ <= I_XY + 1e-10  # Small tolerance for numerical

    return DataProcessingReport(
        I_XY,
        I_XZ,
        satisfies,
        f"For {chain}: I(X;Z) ≤ I(X;Y). "
        f"{'Holds' if satisfies else 'VIOLATED - not a valid Markov chain'}. "
        "Equality iff Z is a sufficient statistic for X given Y."
    )


# ============================================================
# Fano's Inequality
# ============================================================

@dataclass
class FanoReport:
    """Result of Fano's inequality calculation."""
    H_X: float
    P_error: float
    lower_bound: float
    upper_bound_Pe: float
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        return (f"Fano's Inequality\n"
                f"  H(X) = {self.H_X:.6f} bits\n"
                f"  P(error) = {self.P_error:.6f}\n"
                f"  H(X|Y) lower bound: {self.lower_bound:.6f}\n"
                f"  P_e upper bound from H(X|Y): {self.upper_bound_Pe:.6f}\n"
                f"  {self.explanation}")


def _inverse_binary_entropy(h: float, tol: float = 1e-10) -> float:
    """Find p such that H(p) = h using binary search."""
    if h <= 0:
        return 0.0
    if h >= 1:
        return 0.5

    # Binary search in [0, 0.5] since H is symmetric and monotonic on [0, 0.5]
    lo, hi = 0.0, 0.5
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if binary_entropy(mid) < h:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fano_inequality(H_X: float, H_X_given_Y: float, alphabet_size: int) -> FanoReport:
    """Apply Fano's inequality.

    H(X|Y) ≤ H(P_e) + P_e × log(|X| - 1)

    Gives lower bound on error probability from conditional entropy.

    Args:
        H_X: Entropy of X
        H_X_given_Y: Conditional entropy H(X|Y)
        alphabet_size: Size of X alphabet |X|

    Returns:
        FanoReport with bounds
    """
    if alphabet_size < 2:
        return FanoReport(H_X, 0, 0, 0, "Alphabet size must be ≥ 2", False)

    log_M_minus_1 = log2(alphabet_size - 1) if alphabet_size > 2 else 0

    if alphabet_size == 2:
        # Binary case: H(X|Y) ≤ H(P_e), so P_e ≥ H^{-1}(H(X|Y))
        if H_X_given_Y >= 1:
            Pe_lower = 0.5
        else:
            Pe_lower = _inverse_binary_entropy(H_X_given_Y)
    else:
        # General case: solve H(X|Y) ≤ H(P_e) + P_e × log(|X|-1)
        # Use binary search
        def fano_rhs(pe):
            return binary_entropy(pe) + pe * log_M_minus_1

        lo, hi = 0.0, 1.0
        while hi - lo > 1e-10:
            mid = (lo + hi) / 2
            if fano_rhs(mid) < H_X_given_Y:
                lo = mid
            else:
                hi = mid
        Pe_lower = lo

    Pe_lower = max(0, min(1, Pe_lower))

    # Given P_e, compute H(X|Y) lower bound
    H_lower = binary_entropy(Pe_lower) + Pe_lower * log_M_minus_1

    return FanoReport(
        H_X,
        Pe_lower,
        H_lower,
        Pe_lower,
        f"Fano: H(X|Y) ≤ H(P_e) + P_e × log(|X|-1). "
        f"For |X|={alphabet_size}, H(X|Y)={H_X_given_Y:.4f} implies P_e ≥ {Pe_lower:.4f}."
    )


# ============================================================
# AEP and Typical Sets
# ============================================================

@dataclass
class TypicalSetReport:
    """Result of typical set calculation."""
    n: int
    epsilon: float
    entropy: float
    typical_set_size: float
    atypical_prob: float
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        return (f"Typical Set (AEP)\n"
                f"  Sequence length n = {self.n}\n"
                f"  Epsilon = {self.epsilon}\n"
                f"  Source entropy H = {self.entropy:.6f} bits\n"
                f"  Typical set size ≈ 2^(nH) = 2^{self.n * self.entropy:.1f} ≈ {self.typical_set_size:.2e}\n"
                f"  P(atypical) ≤ {self.atypical_prob:.6f}\n"
                f"  {self.explanation}")


def typical_set_bounds(probs: List[float], n: int, epsilon: float = 0.1) -> TypicalSetReport:
    """Compute typical set properties via AEP.

    For n large enough:
    - |A_ε^n| ≤ 2^{n(H + ε)}
    - |A_ε^n| ≥ (1-ε) × 2^{n(H - ε)} for n large
    - P(A_ε^n) ≥ 1 - ε

    Args:
        probs: Source distribution
        n: Sequence length
        epsilon: Tolerance parameter

    Returns:
        TypicalSetReport
    """
    H = entropy(probs)

    # Typical set size is approximately 2^(nH)
    typical_size = 2 ** (n * H)

    # Upper bound on atypical probability (by Chebyshev/AEP)
    # For i.i.d., P(atypical) → 0 as n → ∞
    # Rough bound: ε (by definition of typical set)
    atypical_prob = epsilon

    return TypicalSetReport(
        n,
        epsilon,
        H,
        typical_size,
        atypical_prob,
        f"AEP: -(1/n)log P(x^n) → H in probability. "
        f"Typical set contains ~2^(nH) sequences, total space has 2^(n×{log2(len(probs)):.2f}) = "
        f"2^{n * log2(len(probs)):.1f} sequences."
    )
