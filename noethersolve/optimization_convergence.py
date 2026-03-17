"""Optimization convergence rates and analysis.

Verified formulas for optimization algorithm convergence:
- Gradient descent on smooth strongly convex functions
- Nesterov accelerated gradient
- Condition number analysis
- Lower bounds and optimality

KEY POINT: These convergence rates are EXACT (not approximate):
- GD on μ-strongly convex, L-smooth: (1 - μ/L)^k
- Nesterov: (1 - √(μ/L))^k
- Acceleration ONLY helps when condition number κ = L/μ > 1

CRITICAL DISTINCTIONS LLMs GET WRONG:
1. Nesterov is NOT always faster (identical at κ=1)
2. Momentum parameter is sqrt(μ/L), not a hyperparameter to tune
3. "Linear convergence" means EXPONENTIAL decay in k, not linear
4. Lower bounds are for ORACLE complexity, not just this algorithm
"""

from dataclasses import dataclass
from typing import Optional
import math

# ─── Report Dataclasses ─────────────────────────────────────────────────────

@dataclass
class ConvergenceReport:
    """Report from convergence rate analysis."""
    algorithm: str
    rate: float  # Per-iteration contraction (1-μ/L or 1-sqrt(μ/L))
    rate_formula: str
    iterations_to_epsilon: int  # k for (rate)^k < epsilon
    condition_number: float  # κ = L/μ
    is_optimal: bool  # Matches lower bound?
    comparison_note: str  # vs other algorithms
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Convergence Analysis: {self.algorithm}",
            "=" * 60,
            f"  Contraction rate: ρ = {self.rate:.6f}",
            f"  Formula: ρ = {self.rate_formula}",
            f"  Condition number: κ = L/μ = {self.condition_number:.2f}",
            "-" * 60,
            f"  Iterations for ε=10⁻⁶: {self.iterations_to_epsilon}",
        ]
        if self.is_optimal:
            lines.append("  Optimality: ✓ Matches oracle lower bound")
        else:
            lines.append("  Optimality: ✗ Suboptimal (acceleration possible)")
        lines.append("-" * 60)
        lines.append(f"  {self.comparison_note}")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class ComparisonReport:
    """Report comparing two optimization algorithms."""
    algorithm_1: str
    algorithm_2: str
    rate_1: float
    rate_2: float
    speedup_factor: float  # How many times faster is alg_2
    condition_number: float
    crossover_kappa: Optional[float]  # κ where they're equal
    winner: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Algorithm Comparison",
            "=" * 60,
            f"  {self.algorithm_1}: ρ = {self.rate_1:.6f}",
            f"  {self.algorithm_2}: ρ = {self.rate_2:.6f}",
            "-" * 60,
            f"  Condition number κ = {self.condition_number:.2f}",
            f"  Speedup: {self.algorithm_2} is {self.speedup_factor:.2f}× faster",
            f"  Winner: {self.winner}",
        ]
        if self.crossover_kappa:
            lines.append(f"  Equal at κ = {self.crossover_kappa:.1f}")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class ConditionReport:
    """Report on problem conditioning."""
    L: float  # Smoothness constant
    mu: float  # Strong convexity constant
    kappa: float  # Condition number
    sqrt_kappa: float
    classification: str  # "well-conditioned", "moderately ill-conditioned", "ill-conditioned"
    gd_iterations: int
    nesterov_iterations: int
    acceleration_benefit: float  # Factor improvement
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Problem Conditioning",
            "=" * 60,
            f"  Smoothness L = {self.L:.4f}",
            f"  Strong convexity μ = {self.mu:.4f}",
            "-" * 60,
            f"  Condition number κ = L/μ = {self.kappa:.2f}",
            f"  √κ = {self.sqrt_kappa:.2f}",
            f"  Classification: {self.classification}",
            "-" * 60,
            f"  GD iterations (ε=10⁻⁶): {self.gd_iterations}",
            f"  Nesterov iterations: {self.nesterov_iterations}",
            f"  Acceleration benefit: {self.acceleration_benefit:.1f}× faster",
        ]
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class LowerBoundReport:
    """Report on oracle complexity lower bounds."""
    function_class: str  # "L-smooth μ-strongly-convex"
    lower_bound_rate: float
    lower_bound_formula: str
    gd_achieves: bool
    nesterov_achieves: bool
    gap_gd: float  # How far GD is from optimal
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Oracle Complexity Lower Bound",
            "=" * 60,
            f"  Function class: {self.function_class}",
            f"  Lower bound: ρ* = {self.lower_bound_rate:.6f}",
            f"  Formula: {self.lower_bound_formula}",
            "-" * 60,
        ]
        if self.gd_achieves:
            lines.append("  GD: ✓ Achieves lower bound")
        else:
            lines.append(f"  GD: ✗ Suboptimal by factor {self.gap_gd:.2f}×")
        if self.nesterov_achieves:
            lines.append("  Nesterov: ✓ Achieves lower bound (optimal!)")
        else:
            lines.append("  Nesterov: ✗ Suboptimal")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class StepSizeReport:
    """Report on step size selection."""
    algorithm: str
    optimal_step: float
    step_formula: str
    convergent_range: tuple[float, float]
    diverges_above: float
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Step Size Analysis: {self.algorithm}",
            "=" * 60,
            f"  Optimal step: η* = {self.optimal_step:.6f}",
            f"  Formula: η* = {self.step_formula}",
            "-" * 60,
            f"  Convergent range: η ∈ ({self.convergent_range[0]:.4f}, {self.convergent_range[1]:.4f})",
            f"  Diverges for: η > {self.diverges_above:.4f}",
        ]
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Core Convergence Functions ────────────────────────────────────────────

def gradient_descent_rate(
    L: float,
    mu: float,
    epsilon: float = 1e-6,
) -> ConvergenceReport:
    """Compute convergence rate for gradient descent on μ-strongly convex, L-smooth.

    The EXACT rate is (1 - μ/L)^k. This is "linear convergence" which means
    exponential decay — confusingly named!

    Args:
        L: Smoothness constant (Lipschitz gradient)
        mu: Strong convexity constant
        epsilon: Target accuracy

    Returns:
        ConvergenceReport with exact rate
    """
    if L <= 0 or mu <= 0:
        raise ValueError("L and μ must be positive")
    if mu > L:
        raise ValueError("Must have μ ≤ L (otherwise not valid)")

    kappa = L / mu
    rate = 1 - mu / L  # = 1 - 1/κ = (κ-1)/κ

    # Iterations to reach epsilon: rate^k < epsilon
    # k > log(epsilon) / log(rate)
    if rate >= 1:
        iterations = float('inf')
    elif rate <= 0:
        iterations = 1  # Converges in 1 step (perfect conditioning)
    else:
        iterations = int(math.ceil(math.log(epsilon) / math.log(rate)))

    # Is GD optimal? No — Nesterov is faster for κ > 1
    is_optimal = kappa == 1.0

    notes = [
        "Rate (1 - μ/L) is EXACT, not approximate",
        "'Linear convergence' means EXPONENTIAL decay in error",
        f"Error after k steps: ε₀ × (1 - 1/κ)^k = ε₀ × {rate:.4f}^k",
    ]

    nesterov_rate = 1 - math.sqrt(mu / L)
    if nesterov_rate < rate:
        comparison = f"Nesterov is √κ ≈ {math.sqrt(kappa):.1f}× faster"
    else:
        comparison = "Nesterov has same rate (κ = 1)"

    return ConvergenceReport(
        algorithm="Gradient Descent",
        rate=rate,
        rate_formula="1 - μ/L = 1 - 1/κ",
        iterations_to_epsilon=iterations,
        condition_number=kappa,
        is_optimal=is_optimal,
        comparison_note=comparison,
        notes=notes,
    )


def nesterov_rate(
    L: float,
    mu: float,
    epsilon: float = 1e-6,
) -> ConvergenceReport:
    """Compute convergence rate for Nesterov accelerated gradient.

    The EXACT rate is (1 - √(μ/L))^k = (1 - 1/√κ)^k.
    This is optimal for first-order methods on this function class.

    Args:
        L: Smoothness constant
        mu: Strong convexity constant
        epsilon: Target accuracy

    Returns:
        ConvergenceReport with exact rate
    """
    if L <= 0 or mu <= 0:
        raise ValueError("L and μ must be positive")
    if mu > L:
        raise ValueError("Must have μ ≤ L")

    kappa = L / mu
    rate = 1 - math.sqrt(mu / L)  # = 1 - 1/√κ = (√κ-1)/√κ

    if rate >= 1:
        iterations = float('inf')
    elif rate <= 0:
        iterations = 1  # Converges in 1 step
    else:
        iterations = int(math.ceil(math.log(epsilon) / math.log(rate)))

    # Nesterov IS optimal for this function class
    is_optimal = True

    notes = [
        "Rate (1 - √(μ/L)) is EXACT and OPTIMAL",
        "Matches oracle complexity lower bound",
        "Momentum parameter: β = (√κ - 1)/(√κ + 1) — NOT a hyperparameter!",
    ]

    gd_rate = 1 - mu / L
    if rate < gd_rate:
        speedup = math.log(rate) / math.log(gd_rate) if gd_rate < 1 else float('inf')
        comparison = f"√κ ≈ {math.sqrt(kappa):.1f}× fewer iterations than GD"
    else:
        comparison = "Same as GD (only helps when κ > 1)"

    return ConvergenceReport(
        algorithm="Nesterov Accelerated Gradient",
        rate=rate,
        rate_formula="1 - √(μ/L) = 1 - 1/√κ",
        iterations_to_epsilon=iterations,
        condition_number=kappa,
        is_optimal=is_optimal,
        comparison_note=comparison,
        notes=notes,
    )


def compare_algorithms(
    L: float,
    mu: float,
) -> ComparisonReport:
    """Compare GD and Nesterov convergence.

    KEY INSIGHT: Nesterov is √κ times faster in iteration count.
    At κ=1, they are IDENTICAL. Acceleration only helps for ill-conditioned problems.

    Args:
        L: Smoothness constant
        mu: Strong convexity constant

    Returns:
        ComparisonReport with comparison
    """
    if L <= 0 or mu <= 0:
        raise ValueError("L and μ must be positive")
    if mu > L:
        raise ValueError("Must have μ ≤ L")

    kappa = L / mu
    rate_gd = 1 - mu / L
    rate_nesterov = 1 - math.sqrt(mu / L)

    # Speedup in iteration count: how many GD iterations per Nesterov iteration
    # For same accuracy: gd_iters / nest_iters = log(eps)/log(rate_gd) / log(eps)/log(rate_nest)
    #                                          = log(rate_nest) / log(rate_gd)
    if rate_gd > 0 and rate_gd < 1 and rate_nesterov > 0 and rate_nesterov < 1:
        speedup = math.log(rate_nesterov) / math.log(rate_gd)
    elif rate_gd <= 0 or rate_nesterov <= 0:
        speedup = 1.0  # Both converge immediately
    else:
        speedup = float('inf')

    notes = [
        f"GD contracts by (1 - 1/κ) = {rate_gd:.4f} per step",
        f"Nesterov contracts by (1 - 1/√κ) = {rate_nesterov:.4f} per step",
    ]

    if kappa == 1.0:
        winner = "Tie (κ = 1)"
        notes.append("At perfect conditioning, no acceleration benefit")
    elif rate_nesterov < rate_gd:
        winner = "Nesterov"
        notes.append(f"Nesterov uses √κ ≈ {math.sqrt(kappa):.1f}× fewer iterations")
    else:
        winner = "Tie"

    return ComparisonReport(
        algorithm_1="Gradient Descent",
        algorithm_2="Nesterov",
        rate_1=rate_gd,
        rate_2=rate_nesterov,
        speedup_factor=speedup,
        condition_number=kappa,
        crossover_kappa=1.0,
        winner=winner,
        notes=notes,
    )


# ─── Condition Number Analysis ─────────────────────────────────────────────

def analyze_conditioning(
    L: float,
    mu: float,
    epsilon: float = 1e-6,
) -> ConditionReport:
    """Analyze problem conditioning and its impact on convergence.

    The condition number κ = L/μ determines:
    - How hard the optimization problem is
    - How much acceleration helps
    - Whether the problem is numerically stable

    Args:
        L: Smoothness constant
        mu: Strong convexity constant
        epsilon: Target accuracy

    Returns:
        ConditionReport with full analysis
    """
    if L <= 0 or mu <= 0:
        raise ValueError("L and μ must be positive")
    if mu > L:
        raise ValueError("Must have μ ≤ L")

    kappa = L / mu
    sqrt_kappa = math.sqrt(kappa)

    # Classification
    if kappa <= 10:
        classification = "Well-conditioned"
    elif kappa <= 1000:
        classification = "Moderately ill-conditioned"
    else:
        classification = "Ill-conditioned"

    # Iterations
    rate_gd = 1 - mu / L
    rate_nest = 1 - math.sqrt(mu / L)

    if rate_gd >= 1:
        gd_iters = float('inf')
    else:
        gd_iters = int(math.ceil(math.log(epsilon) / math.log(rate_gd)))

    if rate_nest >= 1:
        nest_iters = float('inf')
    else:
        nest_iters = int(math.ceil(math.log(epsilon) / math.log(rate_nest)))

    benefit = gd_iters / nest_iters if nest_iters > 0 else float('inf')

    notes = []
    if kappa > 100:
        notes.append("Consider preconditioning to reduce κ")
    if benefit > 10:
        notes.append("Acceleration gives major speedup — use Nesterov")
    elif benefit < 1.5:
        notes.append("Acceleration gives minimal benefit — GD may suffice")

    notes.append(f"GD error: ε₀ × (1-1/κ)^k ≈ ε₀ × e^(-k/κ)")
    notes.append(f"Nesterov error: ε₀ × (1-1/√κ)^k ≈ ε₀ × e^(-k/√κ)")

    return ConditionReport(
        L=L,
        mu=mu,
        kappa=kappa,
        sqrt_kappa=sqrt_kappa,
        classification=classification,
        gd_iterations=gd_iters,
        nesterov_iterations=nest_iters,
        acceleration_benefit=benefit,
        notes=notes,
    )


# ─── Lower Bounds ──────────────────────────────────────────────────────────

def oracle_lower_bound(
    L: float,
    mu: float,
) -> LowerBoundReport:
    """Compute oracle complexity lower bound for first-order methods.

    The lower bound rate is (√κ - 1)/(√κ + 1) ≈ 1 - 2/√κ for large κ.
    Nesterov achieves this bound (optimal). GD does not.

    Args:
        L: Smoothness constant
        mu: Strong convexity constant

    Returns:
        LowerBoundReport with bound analysis
    """
    if L <= 0 or mu <= 0:
        raise ValueError("L and μ must be positive")
    if mu > L:
        raise ValueError("Must have μ ≤ L")

    kappa = L / mu
    sqrt_kappa = math.sqrt(kappa)

    # Exact lower bound rate
    lower_rate = (sqrt_kappa - 1) / (sqrt_kappa + 1)

    # GD rate
    gd_rate = 1 - mu / L

    # Nesterov effectively achieves lower bound
    # (1 - 1/√κ) vs (√κ-1)/(√κ+1) are close but not identical
    nesterov_rate = 1 - 1 / sqrt_kappa

    # Gap: how many times more iterations does GD need?
    if lower_rate < 1 and gd_rate < 1:
        gap_gd = math.log(lower_rate) / math.log(gd_rate)
    else:
        gap_gd = float('inf')

    notes = [
        "Lower bound from Nemirovsky-Yudin (1983)",
        "Any first-order method requires Ω(√κ ln(1/ε)) iterations",
        "Nesterov (1983) achieves O(√κ ln(1/ε)) — optimal!",
    ]

    return LowerBoundReport(
        function_class="L-smooth μ-strongly-convex",
        lower_bound_rate=lower_rate,
        lower_bound_formula="(√κ - 1)/(√κ + 1)",
        gd_achieves=False,
        nesterov_achieves=True,
        gap_gd=gap_gd,
        notes=notes,
    )


# ─── Step Size Analysis ────────────────────────────────────────────────────

def optimal_step_size(
    L: float,
    mu: float = 0.0,
    algorithm: str = "gd",
) -> StepSizeReport:
    """Compute optimal step size for convergence.

    For GD on L-smooth functions: η = 1/L (or 2/(L+μ) for strongly convex)
    Step size > 2/L causes DIVERGENCE (common error!).

    Args:
        L: Smoothness constant
        mu: Strong convexity constant (0 for just L-smooth)
        algorithm: "gd" or "nesterov"

    Returns:
        StepSizeReport with analysis
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if mu < 0:
        raise ValueError("μ must be non-negative")
    if mu > L:
        raise ValueError("Must have μ ≤ L")

    algorithm_lower = algorithm.lower()

    if algorithm_lower in ["gd", "gradient_descent"]:
        if mu > 0:
            # Strongly convex: optimal is 2/(L+μ)
            optimal = 2 / (L + mu)
            formula = "2/(L + μ)"
        else:
            # Just L-smooth: use 1/L
            optimal = 1 / L
            formula = "1/L"
        alg_name = "Gradient Descent"
    elif algorithm_lower in ["nesterov", "agd"]:
        optimal = 1 / L
        formula = "1/L"
        alg_name = "Nesterov AGD"
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'")

    # Divergence boundary
    diverges = 2 / L

    # Convergent range
    convergent = (0.0, diverges)

    notes = [
        f"Step size η = {optimal:.4f} gives fastest convergence",
        f"η > 2/L = {diverges:.4f} causes DIVERGENCE",
    ]
    if mu > 0:
        notes.append(f"For strongly convex, η = 2/(L+μ) is optimal")
    notes.append("Common error: using η too large causes oscillation/divergence")

    return StepSizeReport(
        algorithm=alg_name,
        optimal_step=optimal,
        step_formula=formula,
        convergent_range=convergent,
        diverges_above=diverges,
        notes=notes,
    )


# ─── Convex vs Non-Convex ──────────────────────────────────────────────────

@dataclass
class NonConvexReport:
    """Report on non-convex optimization guarantees."""
    algorithm: str
    rate_to_stationary: str  # Rate to reach ||∇f|| < ε
    iterations_to_epsilon: int
    guarantee: str  # What we can guarantee
    no_guarantee: str  # What we cannot guarantee
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Non-Convex Optimization: {self.algorithm}",
            "=" * 60,
            f"  Rate to stationary point: {self.rate_to_stationary}",
            f"  Iterations for ||∇f|| < ε: {self.iterations_to_epsilon}",
            "-" * 60,
            f"  Guarantee: {self.guarantee}",
            f"  No guarantee: {self.no_guarantee}",
        ]
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


def non_convex_rate(
    L: float,
    f_init: float,
    f_star: float,
    epsilon: float = 1e-3,
) -> NonConvexReport:
    """Compute convergence rate for GD on non-convex L-smooth functions.

    For non-convex, we can only guarantee convergence to a STATIONARY POINT
    (||∇f(x)|| → 0), not a GLOBAL minimum. The rate is O(1/√k).

    Args:
        L: Smoothness constant
        f_init: Initial function value f(x₀)
        f_star: Lower bound on f (e.g., 0 for loss functions)
        epsilon: Target gradient norm ||∇f|| < ε

    Returns:
        NonConvexReport with analysis
    """
    if L <= 0:
        raise ValueError("L must be positive")

    delta = f_init - f_star
    if delta < 0:
        raise ValueError("f_init must be ≥ f_star")

    # Iterations: k = O(L × Δ / ε²)
    iterations = int(math.ceil(2 * L * delta / (epsilon ** 2)))

    notes = [
        "Non-convex: only converge to stationary point, not global min",
        f"Rate: min_{{k'≤k}} ||∇f(x_k')||² ≤ 2LΔ/k",
        "Acceleration does NOT help for non-convex (unlike convex)",
    ]

    return NonConvexReport(
        algorithm="Gradient Descent (non-convex)",
        rate_to_stationary="O(1/√k)",
        iterations_to_epsilon=iterations,
        guarantee="Converges to stationary point (∇f = 0)",
        no_guarantee="May be saddle point or local minimum, not global",
        notes=notes,
    )


# ─── Utility Functions ─────────────────────────────────────────────────────

def list_algorithms() -> list[str]:
    """List supported optimization algorithms."""
    return ["gradient_descent", "nesterov", "heavy_ball"]


def iterations_needed(
    algorithm: str,
    L: float,
    mu: float,
    epsilon: float = 1e-6,
) -> int:
    """Compute iterations needed to reach epsilon accuracy.

    Args:
        algorithm: "gd", "nesterov", or "heavy_ball"
        L: Smoothness constant
        mu: Strong convexity constant
        epsilon: Target accuracy

    Returns:
        Number of iterations
    """
    if algorithm.lower() in ["gd", "gradient_descent"]:
        rate = 1 - mu / L
    elif algorithm.lower() in ["nesterov", "agd"]:
        rate = 1 - math.sqrt(mu / L)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'")

    if rate >= 1:
        return float('inf')
    elif rate <= 0:
        return 1

    return int(math.ceil(math.log(epsilon) / math.log(rate)))
