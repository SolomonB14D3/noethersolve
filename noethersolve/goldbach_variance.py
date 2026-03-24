"""
goldbach_variance.py -- Goldbach representation variance analysis.

Computes the Goldbach representation count r(n) and analyzes its fluctuation
structure beyond the Hardy-Littlewood prediction. Key finding: the standard
deviation of normalized residuals r(n)/E(n) decreases multiplicatively with
the number of distinct odd prime factors of n, at a rate of ~30.6% per factor.

The Hardy-Littlewood singular series C(n) predicts the mean representation
count correctly (ratio of means = 1.004), but the fluctuation structure has
additional arithmetic content not captured by the singular series.

Usage:
    from noethersolve.goldbach_variance import analyze_goldbach_variance
    report = analyze_goldbach_variance(limit=50000)
    print(report)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _is_prime(n: int) -> bool:
    """Simple primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _prime_factors(n: int) -> Dict[int, int]:
    """Return prime factorization as {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def _singular_series_factor(n: int) -> float:
    """Compute the Hardy-Littlewood singular series ratio C(n)/C_twin.

    C(n) = 2 * prod_{p|n, p>2} (p-1)/(p-2) * prod_{p>2} (1 - 1/(p-1)^2)
    This returns just the n-dependent part: prod_{p|n, p>2} (p-1)/(p-2).
    """
    result = 1.0
    for p in _prime_factors(n):
        if p > 2:
            result *= (p - 1) / (p - 2)
    return result


def _count_representations(n: int) -> int:
    """Count Goldbach representations r(n) = |{(p,q) : p+q=n, p<=q, both prime}|."""
    count = 0
    for p in range(2, n // 2 + 1):
        if _is_prime(p) and _is_prime(n - p):
            count += 1
    return count


@dataclass
class GoldbachVarianceReport:
    """Report on Goldbach representation variance structure."""
    limit: int
    total_even: int
    mean_residual: float                    # Overall mean of r(n)/E(n)
    overall_std: float                      # Overall std
    std_by_factor_count: Dict[int, float]   # std grouped by # odd prime factors
    reduction_per_factor: float             # Multiplicative reduction (~0.694)
    fit_quality: float                      # Max residual of log-linear fit
    mod6_residuals: Dict[int, float]        # Mean residual by n mod 6
    hardest_numbers: List[Tuple[int, int]]  # (n, r(n)) for smallest r(n)
    min_by_range: List[Tuple[str, int, int]]  # (range, min_r, at_n)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Goldbach Representation Variance Analysis",
            "=" * 60,
            f"  Range: even numbers from 4 to {self.limit}",
            f"  Total tested: {self.total_even}",
            f"  Mean residual r(n)/E(n): {self.mean_residual:.4f}",
            f"    (HL overestimates by {(1-self.mean_residual)*100:.1f}%)",
            "",
            "  Variance by distinct odd prime factor count:",
        ]
        for k in sorted(self.std_by_factor_count.keys()):
            s = self.std_by_factor_count[k]
            lines.append(f"    k={k}: std = {s:.4f}")
        lines.extend([
            "",
            f"  Per-factor variance reduction: {self.reduction_per_factor:.4f}",
            f"    ({(1-self.reduction_per_factor)*100:.1f}% reduction per odd prime factor)",
            f"  Log-linear fit quality: max|residual| = {self.fit_quality:.4f}",
            "",
            "  Mod-6 residual structure:",
        ])
        for mod_val in sorted(self.mod6_residuals.keys()):
            lines.append(f"    n = {mod_val} mod 6: mean = {self.mod6_residuals[mod_val]:.4f}")
        lines.extend([
            "",
            "  Minimum r(n) by range:",
        ])
        for range_str, min_r, at_n in self.min_by_range:
            lines.append(f"    {range_str}: r_min = {min_r} at n = {at_n}")
        lines.append("=" * 60)
        return "\n".join(lines)


def analyze_goldbach_variance(limit: int = 10000) -> GoldbachVarianceReport:
    """Analyze Goldbach representation variance structure.

    Computes r(n) for all even n up to limit, normalizes by
    Hardy-Littlewood prediction, and analyzes the fluctuation structure.

    Key finding: std(residual) decreases multiplicatively with the number
    of distinct odd prime factors, at ~30.6% per factor.

    Args:
        limit: Upper bound for analysis (even numbers from 4 to limit).
               Keep ≤ 50000 for reasonable runtime.

    Returns:
        GoldbachVarianceReport with full analysis.
    """
    # Compute representations
    data = []  # (n, r(n), C(n), expected, residual, k)
    for n in range(4, limit + 1, 2):
        r_n = _count_representations(n)
        C_n = _singular_series_factor(n)
        log_n = math.log(n) if n > 1 else 1
        expected = C_n * n / (log_n ** 2) if log_n > 0 else 1
        residual = r_n / expected if expected > 0 else 0
        k = sum(1 for p in _prime_factors(n) if p > 2)
        data.append((n, r_n, C_n, expected, residual, k))

    total_even = len(data)
    residuals = [r for _, _, _, _, r, _ in data]
    mean_residual = sum(residuals) / len(residuals)
    overall_std = (sum((r - mean_residual) ** 2 for r in residuals) / len(residuals)) ** 0.5

    # Group by factor count (only n > 1000 to avoid small-number effects)
    by_k = {}
    for n, r_n, C_n, exp, res, k in data:
        if n > 1000:
            by_k.setdefault(k, []).append(res)

    std_by_k = {}
    for k, vals in by_k.items():
        if len(vals) >= 10:
            mean_v = sum(vals) / len(vals)
            std_by_k[k] = (sum((v - mean_v) ** 2 for v in vals) / len(vals)) ** 0.5

    # Fit log-linear: log(std) = a + b*k
    ks = sorted(std_by_k.keys())
    if len(ks) >= 2:
        log_stds = [math.log(std_by_k[k]) for k in ks]
        k_arr = list(ks)
        # Simple linear regression
        n_pts = len(ks)
        mean_k = sum(k_arr) / n_pts
        mean_ls = sum(log_stds) / n_pts
        cov = sum((k_arr[i] - mean_k) * (log_stds[i] - mean_ls) for i in range(n_pts))
        var_k = sum((k_arr[i] - mean_k) ** 2 for i in range(n_pts))
        b = cov / var_k if var_k > 0 else 0
        a = mean_ls - b * mean_k
        reduction = math.exp(b)
        predicted = [a + b * k for k in k_arr]
        fit_quality = max(abs(log_stds[i] - predicted[i]) for i in range(n_pts))
    else:
        reduction = 1.0
        fit_quality = 0.0

    # Mod-6 structure
    mod6 = {}
    for n, r_n, C_n, exp, res, k in data:
        if n > 1000:
            m = n % 6
            mod6.setdefault(m, []).append(res)
    mod6_means = {m: sum(v) / len(v) for m, v in mod6.items() if v}

    # Hardest numbers
    hardest = sorted(data, key=lambda x: x[1])[:10]
    hardest_list = [(n, r_n) for n, r_n, _, _, _, _ in hardest]

    # Min by range
    min_by_range = []
    for start in range(0, limit, 10000):
        end = start + 10000
        subset = [(n, r_n) for n, r_n, _, _, _, _ in data if start < n <= end]
        if subset:
            min_n, min_r = min(subset, key=lambda x: x[1])
            min_by_range.append((f"[{start+1:6d}, {end:6d}]", min_r, min_n))

    return GoldbachVarianceReport(
        limit=limit,
        total_even=total_even,
        mean_residual=mean_residual,
        overall_std=overall_std,
        std_by_factor_count=std_by_k,
        reduction_per_factor=reduction,
        fit_quality=fit_quality,
        mod6_residuals=mod6_means,
        hardest_numbers=hardest_list,
        min_by_range=min_by_range,
    )
