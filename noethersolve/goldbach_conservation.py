"""
goldbach_conservation.py -- Goldbach conservation law analysis.

Verifies the double-counting conservation identity for Goldbach representations
within a window [N, N+W], measures anti-bunching (negative lag-1 autocorrelation
of normalized residuals), computes the sub-Poisson factor, and analyzes prime
pairing capacity distribution.

Key identity: sum_n r(n) = (1/2) * sum_q capacity(q)
where capacity(q) = number of primes p such that q+p is in the window and
both q, p are prime. This is exact (each pair (p, q) with p+q=n is counted
once on the left and once each by p and q on the right).

Usage:
    from noethersolve.goldbach_conservation import GoldbachConservationReport
    report = GoldbachConservationReport(N=10000, W=2000)
    print(report)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


def _sieve(limit: int) -> List[bool]:
    """Sieve of Eratosthenes. Returns is_prime array of size limit+1."""
    if limit < 2:
        return [False] * (limit + 1)
    is_p = [True] * (limit + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            for j in range(i * i, limit + 1, i):
                is_p[j] = False
    return is_p


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
    """Compute C(n) = product_{p|n, p odd} (p-1)/(p-2).

    This is the n-dependent part of the Hardy-Littlewood singular series.
    """
    result = 1.0
    for p in _prime_factors(n):
        if p > 2:
            result *= (p - 1) / (p - 2)
    return result


@dataclass
class GoldbachConservationReport:
    """Goldbach conservation law analysis for a window [N, N+W].

    Computes:
    1. Conservation identity verification (sum r(n) vs (1/2) sum capacity(q))
    2. Anti-bunching: lag-1 autocorrelation of C(n)-normalized residuals
    3. Sub-Poisson factor: Var(normalized) / (1/Mean(r))
    4. Capacity distribution statistics
    5. Windowed conservation: CV of sub-window sums vs independence prediction

    Args:
        N: Start of the analysis window (must be even, >= 4).
        W: Window width (default 2000). Even numbers in [N, N+W] are analyzed.
    """

    N: int
    W: int = 2000

    # Conservation
    sum_representations: int = 0
    sum_half_capacities: float = 0.0
    conservation_error: float = 0.0
    conservation_exact: bool = False

    # Anti-bunching
    lag1_autocorrelation: float = 0.0
    is_anti_bunched: bool = False

    # Sub-Poisson
    normalized_variance: float = 0.0
    normalized_mean: float = 0.0
    sub_poisson_factor: float = 0.0
    is_sub_poisson: bool = False

    # Capacity distribution
    capacity_mean: float = 0.0
    capacity_std: float = 0.0
    capacity_cv: float = 0.0
    capacity_min: int = 0
    capacity_max: int = 0

    # Windowed conservation
    num_subwindows: int = 0
    subwindow_cv: float = 0.0
    independence_cv: float = 0.0
    windowed_ratio: float = 0.0

    # Metadata
    num_even: int = 0
    num_primes_in_window: int = 0

    def __init__(self, N: int, W: int = 2000):
        self.N = N
        self.W = W
        self._compute()

    def _compute(self):
        """Run all analyses."""
        N, W = self.N, self.W
        upper = N + W

        # Build sieve up to upper bound
        is_prime = _sieve(upper)

        # Collect even numbers in the window
        start = N if N % 2 == 0 else N + 1
        evens = list(range(max(start, 4), upper + 1, 2))
        self.num_even = len(evens)
        if self.num_even == 0:
            return

        # Primes in the window (for capacity computation)
        primes_in_window = [p for p in range(2, upper + 1) if is_prime[p]]
        window_primes = [p for p in primes_in_window if N <= p <= upper]
        self.num_primes_in_window = len(window_primes)

        # --- 1. Conservation identity ---
        # r(n) = count of primes q in [2, n/2] where n-q is also prime (unordered)
        # R(n) = count of ordered pairs (p, q) both prime with p+q=n
        # R(n) = 2*r(n) when n/2 is not prime, R(n) = 2*r(n)-1 when n/2 is prime
        r_values = {}
        R_values = {}
        total_r = 0
        total_R = 0
        for n in evens:
            count = 0
            for q in range(2, n // 2 + 1):
                if is_prime[q] and is_prime[n - q]:
                    count += 1
            r_values[n] = count
            total_r += count
            # Ordered count: each unordered pair {p,q} with p<q gives 2 ordered
            # pairs; the pair {p,p} (when n=2p) gives 1 ordered pair.
            half_n = n // 2
            R_n = 2 * count - (1 if is_prime[half_n] else 0)
            R_values[n] = R_n
            total_R += R_n
        self.sum_representations = total_r

        # capacity(q) = number of primes p such that q+p is an even number
        # in the window (ordered pairs). By construction:
        #   sum_q capacity(q) = sum_n R(n)  (exact double-counting identity)
        # Each ordered pair (p,q) with p+q=n contributes 1 to capacity(q).
        all_primes = [p for p in range(2, upper + 1) if is_prime[p]]
        even_set = set(evens)
        total_capacity = 0
        capacities = []
        for q in all_primes:
            cap = 0
            for p in all_primes:
                n_val = q + p
                if n_val in even_set:
                    cap += 1
            capacities.append(cap)
            total_capacity += cap

        self.sum_half_capacities = total_capacity / 2.0
        # Exact identity: sum_q capacity(q) = sum_n R(n) (ordered pair counting)
        # We report the unordered version: sum r(n) vs (1/2) sum capacity(q)
        # with correction for self-pairs (n=2p).
        # The exact check: total_capacity == total_R
        self.conservation_exact = (total_capacity == total_R)
        if total_R > 0:
            self.conservation_error = abs(total_capacity - total_R) / total_R
        else:
            self.conservation_error = 0.0

        # --- 2. Anti-bunching (lag-1 autocorrelation of normalized residuals) ---
        normalized = []
        for n in evens:
            C_n = _singular_series_factor(n)
            log_n = math.log(n) if n > 1 else 1.0
            expected = C_n * n / (2.0 * log_n ** 2) if log_n > 0 else 1.0
            norm_r = r_values[n] / expected if expected > 0 else 0.0
            normalized.append(norm_r)

        if len(normalized) > 1:
            mean_norm = sum(normalized) / len(normalized)
            self.normalized_mean = mean_norm
            var_norm = sum((x - mean_norm) ** 2 for x in normalized) / len(normalized)
            self.normalized_variance = var_norm

            # Lag-1 autocorrelation
            if var_norm > 0:
                cov1 = sum(
                    (normalized[i] - mean_norm) * (normalized[i + 1] - mean_norm)
                    for i in range(len(normalized) - 1)
                ) / (len(normalized) - 1)
                self.lag1_autocorrelation = cov1 / var_norm
            else:
                self.lag1_autocorrelation = 0.0
            self.is_anti_bunched = self.lag1_autocorrelation < 0

            # --- 3. Sub-Poisson factor ---
            # Poisson prediction: Var = Mean for count data
            # For normalized residuals: Var(normalized) vs 1/Mean(r)
            r_vals = [r_values[n] for n in evens]
            mean_r = sum(r_vals) / len(r_vals) if r_vals else 1.0
            poisson_var = 1.0 / mean_r if mean_r > 0 else 1.0
            self.sub_poisson_factor = var_norm / poisson_var if poisson_var > 0 else 0.0
            self.is_sub_poisson = self.sub_poisson_factor < 1.0
        else:
            self.normalized_mean = normalized[0] if normalized else 0.0

        # --- 4. Capacity distribution ---
        if capacities:
            self.capacity_mean = sum(capacities) / len(capacities)
            var_cap = sum((c - self.capacity_mean) ** 2 for c in capacities) / len(capacities)
            self.capacity_std = var_cap ** 0.5
            self.capacity_cv = self.capacity_std / self.capacity_mean if self.capacity_mean > 0 else 0.0
            self.capacity_min = min(capacities)
            self.capacity_max = max(capacities)

        # --- 5. Windowed conservation ---
        # Break into sub-windows of ~200 even numbers each
        subwin_size = min(200, self.num_even)
        if subwin_size > 0 and self.num_even > subwin_size:
            subwindow_sums = []
            for i in range(0, self.num_even, subwin_size):
                chunk = evens[i:i + subwin_size]
                if len(chunk) < subwin_size // 2:
                    break  # skip tiny remainder
                s = sum(r_values[n] for n in chunk)
                subwindow_sums.append(s)

            self.num_subwindows = len(subwindow_sums)
            if self.num_subwindows >= 2:
                mean_sw = sum(subwindow_sums) / len(subwindow_sums)
                var_sw = sum((s - mean_sw) ** 2 for s in subwindow_sums) / len(subwindow_sums)
                self.subwindow_cv = (var_sw ** 0.5) / mean_sw if mean_sw > 0 else 0.0

                # Independence prediction: CV = 1/sqrt(count_per_window)
                self.independence_cv = 1.0 / (subwin_size ** 0.5)
                self.windowed_ratio = (
                    self.subwindow_cv / self.independence_cv
                    if self.independence_cv > 0 else 0.0
                )
            else:
                self.num_subwindows = 1
        else:
            self.num_subwindows = 1 if self.num_even > 0 else 0

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Goldbach Conservation Law Analysis",
            "=" * 60,
            f"  Window: [{self.N}, {self.N + self.W}]",
            f"  Even numbers analyzed: {self.num_even}",
            f"  Primes in window: {self.num_primes_in_window}",
            "",
            "  1. Conservation Identity (sum_q cap(q) = sum_n R(n))",
            "  " + "-" * 40,
            f"     sum r(n) [unordered]  = {self.sum_representations}",
            f"     (1/2) sum capacity(q) = {self.sum_half_capacities:.1f}",
            f"     Exact match (ordered) = {'YES' if self.conservation_exact else 'NO'}",
            f"     Relative error        = {self.conservation_error:.2e}",
            "",
            "  2. Anti-Bunching (lag-1 autocorrelation)",
            "  " + "-" * 40,
            f"     Lag-1 autocorrelation = {self.lag1_autocorrelation:+.4f}",
            f"     Anti-bunched: {'YES' if self.is_anti_bunched else 'NO'}",
        ]
        if self.is_anti_bunched:
            lines.append(
                "     (Negative = consecutive residuals repel, not cluster)"
            )
        lines.extend([
            "",
            "  3. Sub-Poisson Factor",
            "  " + "-" * 40,
            f"     Normalized mean       = {self.normalized_mean:.4f}",
            f"     Normalized variance   = {self.normalized_variance:.6f}",
            f"     Sub-Poisson factor    = {self.sub_poisson_factor:.4f}",
            f"     Sub-Poisson: {'YES' if self.is_sub_poisson else 'NO'}",
        ])
        if self.is_sub_poisson:
            lines.append(
                f"     ({(1 - self.sub_poisson_factor) * 100:.1f}% below Poisson variance)"
            )
        lines.extend([
            "",
            "  4. Prime Pairing Capacity",
            "  " + "-" * 40,
            f"     Mean capacity  = {self.capacity_mean:.2f}",
            f"     Std capacity   = {self.capacity_std:.2f}",
            f"     CV             = {self.capacity_cv:.4f}",
            f"     Range          = [{self.capacity_min}, {self.capacity_max}]",
            "",
            "  5. Windowed Conservation",
            "  " + "-" * 40,
            f"     Sub-windows         = {self.num_subwindows}",
            f"     Sub-window CV       = {self.subwindow_cv:.4f}",
            f"     Independence CV     = {self.independence_cv:.4f}",
            f"     Ratio (actual/indep)= {self.windowed_ratio:.4f}",
        ])
        if self.windowed_ratio > 0:
            if self.windowed_ratio < 0.8:
                lines.append("     Tighter than independence (conservation constrains)")
            elif self.windowed_ratio > 1.2:
                lines.append("     Looser than independence (clustering present)")
            else:
                lines.append("     Consistent with independence")
        lines.append("=" * 60)
        return "\n".join(lines)


def analyze_goldbach_conservation(N: int = 10000, W: int = 2000) -> GoldbachConservationReport:
    """Analyze Goldbach conservation laws within a window.

    Computes conservation identity, anti-bunching, sub-Poisson factor,
    capacity distribution, and windowed conservation statistics.

    Args:
        N: Start of window (even, >= 4). Default 10000.
        W: Window width. Default 2000.

    Returns:
        GoldbachConservationReport with all metrics.
    """
    return GoldbachConservationReport(N=N, W=W)
