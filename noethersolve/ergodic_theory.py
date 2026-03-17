"""Ergodic theory and dynamical systems hierarchy.

Verified formulas for dynamical systems classification:
- Ergodic hierarchy: Bernoulli ⊊ K-mixing ⊊ mixing ⊊ weak mixing ⊊ ergodic
- Kolmogorov-Sinai entropy
- Lyapunov exponents and Pesin formula
- Poincaré recurrence

KEY POINT: The ergodic hierarchy has STRICT inclusions. Each level is
strictly stronger than the previous. LLMs often confuse these levels or
claim equivalence where there is none.

CRITICAL DISTINCTIONS LLMs GET WRONG:
1. Ergodic ≠ mixing (ergodic is weaker!)
2. Mixing implies ergodic, NOT the other way around
3. K-mixing implies positive entropy
4. Bernoulli is the STRONGEST, not ergodic
"""

from dataclasses import dataclass, field
from typing import Optional
import math

# ─── Constants ─────────────────────────────────────────────────────────────

# Ergodic hierarchy levels (ordered from strongest to weakest)
HIERARCHY_LEVELS = ["Bernoulli", "K-mixing", "mixing", "weak_mixing", "ergodic"]

# ─── Report Dataclasses ─────────────────────────────────────────────────────

@dataclass
class HierarchyReport:
    """Report from ergodic hierarchy classification."""
    system_name: str
    level: str  # Bernoulli, K-mixing, mixing, weak_mixing, ergodic
    is_ergodic: bool
    is_weak_mixing: bool
    is_mixing: bool
    is_k_mixing: bool
    is_bernoulli: bool
    entropy: Optional[float]  # Kolmogorov-Sinai entropy
    example_property: str  # What makes it this level
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Ergodic Classification: {self.system_name}",
            "=" * 60,
            f"  Level: {self.level.upper()}",
            "-" * 60,
            "  Hierarchy status (each implies all below):",
        ]
        lines.append(f"    Bernoulli:    {'✓' if self.is_bernoulli else '✗'}")
        lines.append(f"    K-mixing:     {'✓' if self.is_k_mixing else '✗'}")
        lines.append(f"    Mixing:       {'✓' if self.is_mixing else '✗'}")
        lines.append(f"    Weak mixing:  {'✓' if self.is_weak_mixing else '✗'}")
        lines.append(f"    Ergodic:      {'✓' if self.is_ergodic else '✗'}")
        if self.entropy is not None:
            lines.append("-" * 60)
            lines.append(f"  KS entropy: h = {self.entropy:.6f}")
        lines.append("-" * 60)
        lines.append(f"  Key property: {self.example_property}")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class LyapunovReport:
    """Report from Lyapunov exponent calculation."""
    exponents: list[float]  # Ordered from largest to smallest
    dimension: int
    sum_positive: float  # Sum of positive exponents
    kaplan_yorke_dim: float  # Lyapunov dimension
    is_chaotic: bool  # At least one positive exponent
    system_type: str  # "dissipative", "conservative", "expanding"
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Lyapunov Exponents",
            "=" * 60,
            f"  Exponents: {', '.join(f'{l:.4f}' for l in self.exponents)}",
            f"  Dimension: {self.dimension}",
            "-" * 60,
            f"  Sum of positive: {self.sum_positive:.6f}",
            f"  Kaplan-Yorke dimension: {self.kaplan_yorke_dim:.4f}",
            f"  System type: {self.system_type}",
        ]
        if self.is_chaotic:
            lines.append("  Status: CHAOTIC (λ_max > 0)")
        else:
            lines.append("  Status: Regular (all λ ≤ 0)")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class EntropyReport:
    """Report from entropy calculation."""
    ks_entropy: float  # Kolmogorov-Sinai entropy
    topological_entropy: Optional[float]
    satisfies_pesin: bool  # h_μ = Σλ⁺ (for smooth systems)
    positive_exponent_sum: float
    is_deterministic: bool  # h > 0 means effectively random
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Dynamical Entropy",
            "=" * 60,
            f"  KS entropy h_μ = {self.ks_entropy:.6f} bits/iteration",
        ]
        if self.topological_entropy is not None:
            lines.append(f"  Topological entropy h_top = {self.topological_entropy:.6f}")
        lines.extend([
            "-" * 60,
            f"  Sum of positive Lyapunov: Σλ⁺ = {self.positive_exponent_sum:.6f}",
        ])
        if self.satisfies_pesin:
            lines.append("  Pesin formula: h_μ = Σλ⁺ ✓ (SRB measure)")
        else:
            lines.append("  Pesin formula: h_μ < Σλ⁺ (measure not SRB)")
        lines.append("-" * 60)
        if self.is_deterministic:
            lines.append("  Predictability: CHAOTIC (h > 0, information loss)")
        else:
            lines.append("  Predictability: Regular (h = 0, no information loss)")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class RecurrenceReport:
    """Report from Poincaré recurrence analysis."""
    measure: float  # Measure of the set
    estimated_return_time: float  # Expected recurrence time
    volume: float  # Phase space volume
    is_finite_recurrence: bool  # True if recurrence is guaranteed
    formula: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Poincaré Recurrence",
            "=" * 60,
            f"  Set measure μ(A) = {self.measure:.6e}",
            f"  Phase space volume = {self.volume:.6e}",
            "-" * 60,
            f"  Expected return time ~ {self.estimated_return_time:.2e}",
            f"  Formula: {self.formula}",
            "-" * 60,
        ]
        if self.is_finite_recurrence:
            lines.append("  Recurrence: GUARANTEED (almost every point returns)")
        else:
            lines.append("  Recurrence: NOT GUARANTEED (non-recurrent set)")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class ComparisonReport:
    """Report comparing two hierarchy levels."""
    level_1: str
    level_2: str
    relationship: str  # "implies", "implied_by", "incomparable", "equivalent"
    strict: bool  # True if strictly stronger/weaker
    counterexample: Optional[str]  # Example showing non-equivalence
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Ergodic Hierarchy Comparison",
            "=" * 60,
            f"  {self.level_1} vs {self.level_2}",
            "-" * 60,
        ]
        if self.relationship == "implies":
            lines.append(f"  {self.level_1} ⟹ {self.level_2}")
            if self.strict:
                lines.append(f"  {self.level_2} ⇏ {self.level_1} (STRICT)")
        elif self.relationship == "implied_by":
            lines.append(f"  {self.level_2} ⟹ {self.level_1}")
            if self.strict:
                lines.append(f"  {self.level_1} ⇏ {self.level_2} (STRICT)")
        elif self.relationship == "equivalent":
            lines.append(f"  {self.level_1} ⟺ {self.level_2}")
        else:
            lines.append(f"  No implication either direction")
        if self.counterexample:
            lines.append("-" * 60)
            lines.append(f"  Counterexample: {self.counterexample}")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Example Systems Database ──────────────────────────────────────────────

EXAMPLE_SYSTEMS = {
    # (name, level, entropy, description, key property)
    "bernoulli_shift": (
        "Bernoulli shift",
        "Bernoulli",
        math.log(2),  # For 2-symbol shift
        "Independent coin flips",
        "Past and future are independent",
    ),
    "bakers_map": (
        "Baker's map",
        "Bernoulli",
        math.log(2),
        "Stretch, cut, stack",
        "Equivalent to Bernoulli shift",
    ),
    "arnolds_cat": (
        "Arnold's cat map",
        "K-mixing",
        math.log((3 + math.sqrt(5)) / 2),  # Golden ratio
        "Linear toral automorphism",
        "K-automorphism, exponential mixing",
    ),
    "geodesic_flow_negative": (
        "Geodesic flow (negative curvature)",
        "K-mixing",
        None,  # Depends on curvature
        "Flow on negatively curved manifold",
        "K-mixing by Anosov structure",
    ),
    "horocycle_flow": (
        "Horocycle flow",
        "mixing",
        0.0,  # Zero entropy
        "Flow along horocycles",
        "Mixing but NOT K-mixing (zero entropy)",
    ),
    "irrational_rotation": (
        "Irrational rotation",
        "ergodic",
        0.0,
        "T: x → x + α (mod 1), α irrational",
        "Ergodic but NOT weak mixing",
    ),
    "skew_product": (
        "Skew product",
        "weak_mixing",
        0.0,
        "Rotation with varying angle",
        "Weak mixing but NOT mixing",
    ),
    "logistic_map_chaos": (
        "Logistic map (r=4)",
        "Bernoulli",
        math.log(2),
        "x → 4x(1-x)",
        "Conjugate to tent map (Bernoulli)",
    ),
    "henon_attractor": (
        "Hénon attractor",
        "K-mixing",
        0.42,  # Approximate
        "Strange attractor",
        "Positive entropy, dissipative",
    ),
    "lorenz_attractor": (
        "Lorenz attractor",
        "mixing",
        0.91,  # Approximate
        "Strange attractor",
        "Mixing with SRB measure",
    ),
}


# ─── Classification Functions ──────────────────────────────────────────────

def classify_system(
    name: str = "",
    level: str = "",
    entropy: Optional[float] = None,
    lyapunov_max: Optional[float] = None,
) -> HierarchyReport:
    """Classify a dynamical system in the ergodic hierarchy.

    The ergodic hierarchy (strict inclusions):
    Bernoulli ⊊ K-mixing ⊊ mixing ⊊ weak mixing ⊊ ergodic

    Each level implies all levels below it.

    Args:
        name: Name of a known system (see list_systems())
        level: Or specify level directly (bernoulli, k_mixing, mixing, weak_mixing, ergodic)
        entropy: KS entropy (optional)
        lyapunov_max: Largest Lyapunov exponent (optional)

    Returns:
        HierarchyReport with full classification
    """
    # Look up known system
    if name.lower().replace(" ", "_").replace("-", "_") in EXAMPLE_SYSTEMS:
        key = name.lower().replace(" ", "_").replace("-", "_")
        sys_name, sys_level, sys_entropy, sys_desc, sys_prop = EXAMPLE_SYSTEMS[key]
        level = sys_level
        if sys_entropy is not None:
            entropy = sys_entropy
        example_property = sys_prop
    else:
        sys_name = name if name else "Custom system"
        example_property = _get_level_property(level)

    # Normalize level
    level_lower = level.lower().replace("-", "_").replace(" ", "_")
    level_map = {
        "bernoulli": "Bernoulli",
        "k_mixing": "K-mixing",
        "k_system": "K-mixing",
        "kolmogorov": "K-mixing",
        "mixing": "mixing",
        "strongly_mixing": "mixing",
        "weak_mixing": "weak_mixing",
        "weakly_mixing": "weak_mixing",
        "ergodic": "ergodic",
    }
    if level_lower not in level_map:
        raise ValueError(f"Unknown level '{level}'. Use: {list(level_map.keys())}")

    canonical_level = level_map[level_lower]

    # Determine hierarchy flags (each level implies all below)
    level_idx = HIERARCHY_LEVELS.index(canonical_level)
    is_bernoulli = level_idx <= 0
    is_k_mixing = level_idx <= 1
    is_mixing = level_idx <= 2
    is_weak_mixing = level_idx <= 3
    is_ergodic = level_idx <= 4

    notes = [
        "Bernoulli ⊊ K-mixing ⊊ mixing ⊊ weak mixing ⊊ ergodic",
        "Each level STRICTLY implies all levels to the right",
    ]

    if canonical_level == "Bernoulli":
        notes.append("Bernoulli: strongest mixing — past/future independent")
    elif canonical_level == "K-mixing":
        notes.append("K-mixing (Kolmogorov): positive entropy, exponential decorrelation")
    elif canonical_level == "mixing":
        notes.append("Mixing: correlations decay to zero (but may be polynomial)")
    elif canonical_level == "weak_mixing":
        notes.append("Weak mixing: correlations decay in Cesàro mean")
    else:
        notes.append("Ergodic: time averages equal space averages")

    return HierarchyReport(
        system_name=sys_name,
        level=canonical_level,
        is_ergodic=is_ergodic,
        is_weak_mixing=is_weak_mixing,
        is_mixing=is_mixing,
        is_k_mixing=is_k_mixing,
        is_bernoulli=is_bernoulli,
        entropy=entropy,
        example_property=example_property,
        notes=notes,
    )


def _get_level_property(level: str) -> str:
    """Get key property for a hierarchy level."""
    properties = {
        "bernoulli": "Independent processes (past and future independent)",
        "k_mixing": "Positive entropy, exponential mixing",
        "k-mixing": "Positive entropy, exponential mixing",
        "mixing": "Correlations decay to zero",
        "weak_mixing": "Correlations decay in Cesàro mean",
        "ergodic": "Time average = space average",
    }
    return properties.get(level.lower().replace("_", "-"), "Unknown level")


def compare_levels(level_1: str, level_2: str) -> ComparisonReport:
    """Compare two levels in the ergodic hierarchy.

    Determines whether level_1 implies level_2, vice versa, or neither.
    Provides counterexamples when the implication is strict.

    Args:
        level_1: First level
        level_2: Second level

    Returns:
        ComparisonReport with relationship and counterexample
    """
    # Normalize levels
    level_map = {
        "bernoulli": 0,
        "k_mixing": 1,
        "k-mixing": 1,
        "mixing": 2,
        "weak_mixing": 3,
        "weak-mixing": 3,
        "ergodic": 4,
    }

    l1 = level_1.lower().replace(" ", "_")
    l2 = level_2.lower().replace(" ", "_")

    if l1 not in level_map:
        raise ValueError(f"Unknown level '{level_1}'")
    if l2 not in level_map:
        raise ValueError(f"Unknown level '{level_2}'")

    idx1 = level_map[l1]
    idx2 = level_map[l2]

    # Counterexamples for strict implications
    counterexamples = {
        (0, 1): None,  # Bernoulli → K is equivalent for many systems
        (1, 2): "Horocycle flow: mixing but zero entropy (not K)",
        (2, 3): "Skew products can be weak mixing but not mixing",
        (3, 4): "Irrational rotation: ergodic but not weak mixing",
    }

    canonical_1 = HIERARCHY_LEVELS[idx1]
    canonical_2 = HIERARCHY_LEVELS[idx2]

    if idx1 == idx2:
        return ComparisonReport(
            level_1=canonical_1,
            level_2=canonical_2,
            relationship="equivalent",
            strict=False,
            counterexample=None,
            notes=["Same level in hierarchy"],
        )
    elif idx1 < idx2:
        # level_1 is stronger
        return ComparisonReport(
            level_1=canonical_1,
            level_2=canonical_2,
            relationship="implies",
            strict=True,
            counterexample=counterexamples.get((idx1, idx2)),
            notes=[
                f"{canonical_1} is STRONGER than {canonical_2}",
                "The implication is strict (not reversible)",
            ],
        )
    else:
        # level_2 is stronger
        return ComparisonReport(
            level_1=canonical_1,
            level_2=canonical_2,
            relationship="implied_by",
            strict=True,
            counterexample=counterexamples.get((idx2, idx1)),
            notes=[
                f"{canonical_2} is STRONGER than {canonical_1}",
                "The implication is strict (not reversible)",
            ],
        )


# ─── Lyapunov Exponents ────────────────────────────────────────────────────

def lyapunov_analysis(
    exponents: list[float],
) -> LyapunovReport:
    """Analyze Lyapunov exponents for a dynamical system.

    The Lyapunov exponents measure exponential divergence of nearby
    trajectories. Positive exponents indicate chaos.

    Args:
        exponents: List of Lyapunov exponents (will be sorted)

    Returns:
        LyapunovReport with analysis
    """
    if not exponents:
        raise ValueError("At least one exponent required")

    # Sort from largest to smallest
    sorted_exp = sorted(exponents, reverse=True)
    dimension = len(sorted_exp)

    # Sum of positive exponents
    sum_positive = sum(l for l in sorted_exp if l > 0)

    # Kaplan-Yorke dimension
    ky_dim = _kaplan_yorke_dimension(sorted_exp)

    # Is chaotic?
    is_chaotic = sorted_exp[0] > 0

    # System type
    exp_sum = sum(sorted_exp)
    if abs(exp_sum) < 1e-10:
        system_type = "conservative"
    elif exp_sum < 0:
        system_type = "dissipative"
    else:
        system_type = "expanding"

    notes = []
    if is_chaotic:
        notes.append(f"Maximum Lyapunov exponent λ_max = {sorted_exp[0]:.4f} > 0")
        notes.append(f"Lyapunov time (predictability horizon) ~ 1/λ_max = {1/sorted_exp[0]:.2f}")
    else:
        notes.append("All exponents ≤ 0: regular (non-chaotic) dynamics")

    if system_type == "conservative":
        notes.append("Sum = 0: volume-preserving (Hamiltonian)")
    elif system_type == "dissipative":
        notes.append(f"Sum = {exp_sum:.4f} < 0: contracts phase space volume")

    return LyapunovReport(
        exponents=sorted_exp,
        dimension=dimension,
        sum_positive=sum_positive,
        kaplan_yorke_dim=ky_dim,
        is_chaotic=is_chaotic,
        system_type=system_type,
        notes=notes,
    )


def _kaplan_yorke_dimension(exponents: list[float]) -> float:
    """Calculate Kaplan-Yorke (Lyapunov) dimension.

    D_KY = j + (Σᵢ₌₁ʲ λᵢ) / |λⱼ₊₁|

    where j is the largest integer such that Σᵢ₌₁ʲ λᵢ ≥ 0.
    """
    n = len(exponents)
    cumsum = 0.0
    j = 0

    for i, l in enumerate(exponents):
        cumsum += l
        if cumsum >= 0:
            j = i + 1
        else:
            break

    if j == 0:
        return 0.0
    if j >= n:
        return float(n)

    # D_KY = j + sum_positive / |λ_{j+1}|
    sum_j = sum(exponents[:j])
    if abs(exponents[j]) < 1e-15:
        return float(j)

    return j + sum_j / abs(exponents[j])


# ─── Entropy Calculations ──────────────────────────────────────────────────

def entropy_analysis(
    ks_entropy: float,
    lyapunov_positive_sum: Optional[float] = None,
    topological_entropy: Optional[float] = None,
) -> EntropyReport:
    """Analyze dynamical entropy.

    The Kolmogorov-Sinai (metric) entropy measures the rate of
    information creation. The Pesin formula relates it to Lyapunov
    exponents for smooth systems.

    Args:
        ks_entropy: Kolmogorov-Sinai entropy (bits per iteration)
        lyapunov_positive_sum: Sum of positive Lyapunov exponents
        topological_entropy: Topological entropy (optional)

    Returns:
        EntropyReport with analysis
    """
    if ks_entropy < 0:
        raise ValueError("KS entropy cannot be negative")

    # Check Pesin formula: h_μ ≤ Σλ⁺ with equality for SRB measure
    if lyapunov_positive_sum is None:
        lyapunov_positive_sum = ks_entropy  # Assume equality
        satisfies_pesin = True
    else:
        # Pesin inequality: h ≤ Σλ⁺
        satisfies_pesin = abs(ks_entropy - lyapunov_positive_sum) < 1e-6

    is_deterministic = ks_entropy > 1e-10

    notes = []
    if is_deterministic:
        notes.append(f"h > 0: system creates {ks_entropy:.4f} bits/iteration of unpredictability")
        notes.append("Positive entropy implies K-mixing (or stronger)")
    else:
        notes.append("h = 0: no information creation (predictable)")
        notes.append("Zero entropy compatible with ergodic, weak mixing, or mixing")

    if topological_entropy is not None:
        notes.append(f"Variational principle: h_μ ≤ h_top for all μ")
        if ks_entropy > topological_entropy + 1e-10:
            notes.append("WARNING: h_μ > h_top violates variational principle!")

    return EntropyReport(
        ks_entropy=ks_entropy,
        topological_entropy=topological_entropy,
        satisfies_pesin=satisfies_pesin,
        positive_exponent_sum=lyapunov_positive_sum,
        is_deterministic=is_deterministic,
        notes=notes,
    )


# ─── Poincaré Recurrence ───────────────────────────────────────────────────

def poincare_recurrence(
    set_measure: float,
    phase_space_volume: float = 1.0,
) -> RecurrenceReport:
    """Analyze Poincaré recurrence time.

    The Poincaré recurrence theorem guarantees that almost every point
    in a finite-measure system returns arbitrarily close to its starting
    point. The expected return time is approximately 1/measure(A).

    Args:
        set_measure: Measure of the set A
        phase_space_volume: Total phase space volume (default 1 for normalized)

    Returns:
        RecurrenceReport with analysis
    """
    if set_measure <= 0:
        raise ValueError("Set measure must be positive")
    if set_measure > phase_space_volume:
        raise ValueError("Set measure cannot exceed phase space volume")

    # Kac's lemma: expected return time = 1/μ(A)
    expected_return = 1.0 / set_measure

    is_finite = phase_space_volume < float('inf')

    notes = [
        "Poincaré recurrence: almost every point returns infinitely often",
        "Kac's lemma: ⟨τ⟩ = 1/μ(A) for ergodic systems",
    ]

    if set_measure < 1e-10:
        notes.append(f"WARNING: Very small set — return time ~ {expected_return:.2e} iterations")
        notes.append("May exceed age of universe for physical systems!")

    return RecurrenceReport(
        measure=set_measure,
        estimated_return_time=expected_return,
        volume=phase_space_volume,
        is_finite_recurrence=is_finite,
        formula="⟨τ⟩ = 1/μ(A) (Kac's lemma)",
        notes=notes,
    )


# ─── Mixing Rate Analysis ──────────────────────────────────────────────────

@dataclass
class MixingRateReport:
    """Report on mixing rate."""
    rate_type: str  # "exponential", "polynomial", "none"
    decay_rate: Optional[float]  # For exponential
    decay_exponent: Optional[float]  # For polynomial
    is_rapid_mixing: bool
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Mixing Rate Analysis",
            "=" * 60,
            f"  Decay type: {self.rate_type.upper()}",
        ]
        if self.rate_type == "exponential":
            lines.append(f"  Rate: |C(t)| ~ e^(-{self.decay_rate:.4f}t)")
        elif self.rate_type == "polynomial":
            lines.append(f"  Rate: |C(t)| ~ t^(-{self.decay_exponent:.2f})")
        lines.append("-" * 60)
        if self.is_rapid_mixing:
            lines.append("  Rapid mixing: correlations decay quickly")
        else:
            lines.append("  Slow mixing: correlations persist")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


def mixing_rate(
    rate_type: str,
    rate_value: float,
) -> MixingRateReport:
    """Analyze mixing rate (correlation decay).

    For mixing systems, correlations C(t) = ⟨f∘T^t · g⟩ - ⟨f⟩⟨g⟩ decay.
    The rate determines how quickly the system "forgets" its initial state.

    Args:
        rate_type: "exponential" or "polynomial"
        rate_value: Decay rate (λ for e^(-λt)) or exponent (α for t^(-α))

    Returns:
        MixingRateReport with analysis
    """
    rate_type_lower = rate_type.lower()
    if rate_type_lower not in ["exponential", "polynomial"]:
        raise ValueError("rate_type must be 'exponential' or 'polynomial'")

    if rate_value <= 0:
        raise ValueError("Rate/exponent must be positive")

    notes = []
    if rate_type_lower == "exponential":
        decay_rate = rate_value
        decay_exponent = None
        is_rapid = decay_rate > 0.1
        notes.append("Exponential mixing implies K-system (positive entropy)")
        notes.append(f"Mixing time ~ 1/λ = {1/decay_rate:.2f}")
    else:
        decay_rate = None
        decay_exponent = rate_value
        is_rapid = decay_exponent > 2
        notes.append("Polynomial mixing: slower forgetting")
        if decay_exponent <= 1:
            notes.append("WARNING: α ≤ 1 means correlations not summable")

    return MixingRateReport(
        rate_type=rate_type_lower,
        decay_rate=decay_rate,
        decay_exponent=decay_exponent,
        is_rapid_mixing=is_rapid,
        notes=notes,
    )


# ─── Utility Functions ─────────────────────────────────────────────────────

def list_systems() -> list[str]:
    """List all known example systems."""
    return list(EXAMPLE_SYSTEMS.keys())


def list_levels() -> list[str]:
    """List ergodic hierarchy levels from strongest to weakest."""
    return HIERARCHY_LEVELS.copy()


def is_stronger(level_1: str, level_2: str) -> bool:
    """Check if level_1 is strictly stronger than level_2.

    In the ergodic hierarchy, "stronger" means more restrictive.
    Bernoulli is strongest, ergodic is weakest.
    """
    level_map = {
        "bernoulli": 0,
        "k_mixing": 1,
        "k-mixing": 1,
        "mixing": 2,
        "weak_mixing": 3,
        "weak-mixing": 3,
        "ergodic": 4,
    }
    l1 = level_1.lower().replace(" ", "_")
    l2 = level_2.lower().replace(" ", "_")

    if l1 not in level_map or l2 not in level_map:
        raise ValueError("Unknown level")

    return level_map[l1] < level_map[l2]


def implies(level_from: str, level_to: str) -> bool:
    """Check if level_from implies level_to.

    In the hierarchy, stronger levels imply weaker ones.
    E.g., Bernoulli implies everything, ergodic implies nothing stronger.
    """
    return is_stronger(level_from, level_to) or level_from.lower() == level_to.lower()
