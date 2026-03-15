"""
noethersolve.number_theory — Number theory conjecture numerical verifier.

Pure-computation verification of famous number theory conjectures for given
inputs. No external data required -- all results from deterministic arithmetic.

Usage:
    from noethersolve.number_theory import (
        verify_goldbach, verify_collatz, verify_twin_primes,
        check_abc_triple, verify_legendre, prime_gap_analysis,
        is_prime, prime_sieve, radical,
    )

    # Goldbach: every even n > 2 is the sum of two primes
    report = verify_goldbach(100)
    print(report)
    # Shows: 6 decompositions, smallest (3, 97), largest (47, 53)

    # Collatz: sequence always reaches 1
    report = verify_collatz(27)
    print(report)
    # Shows: 111 steps, max value 9232, reached 1

    # Twin primes up to 1000
    report = verify_twin_primes(1000)
    print(report)
    # Shows: 35 pairs, density 0.209, largest (881, 883)

    # ABC triple quality
    report = check_abc_triple(5, 27, 32)
    print(report)
    # Shows: radical=30, quality=1.0283, exceptional (q > 1)

    # Legendre: prime between n^2 and (n+1)^2
    report = verify_legendre(10)
    print(report)
    # Shows: primes in [100, 121], count=5, verified

    # Prime gap analysis up to 10000
    report = prime_gap_analysis(10000)
    print(report)
    # Shows: max gap, Cramer ratio, gap distribution
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# Deterministic Miller-Rabin witnesses for n < 3.317e24 (Sorenson & Webster 2016).
_MR_WITNESSES_SMALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    """Deterministic primality test.

    Uses trial division for n < 1000, then deterministic Miller-Rabin with
    witnesses sufficient for all n < 3.317 * 10^24.

    Args:
        n: Integer to test.

    Returns:
        True if n is prime.
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Trial division up to sqrt for small n
    if n < 1000:
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    # Miller-Rabin with deterministic witnesses
    return _miller_rabin(n, _MR_WITNESSES_SMALL)


def _miller_rabin(n: int, witnesses: Tuple[int, ...]) -> bool:
    """Miller-Rabin primality test with given witnesses."""
    # Write n-1 as 2^r * d with d odd
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def prime_sieve(limit: int) -> List[int]:
    """Sieve of Eratosthenes returning all primes up to *limit* (inclusive).

    Args:
        limit: Upper bound (inclusive).

    Returns:
        Sorted list of primes <= limit.
    """
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i, v in enumerate(sieve) if v]


def _factorize(n: int) -> List[int]:
    """Return the list of distinct prime factors of n.

    Uses trial division. Practical for n up to ~10^12.
    """
    if n <= 1:
        return []
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append(n)
    return factors


def radical(n: int) -> int:
    """Product of distinct prime factors of n.

    rad(n) = product of p for each prime p dividing n.
    For example, rad(12) = rad(2^2 * 3) = 2 * 3 = 6.

    Args:
        n: Positive integer.

    Returns:
        The radical of n.

    Raises:
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError(f"radical requires n >= 1, got {n}")
    result = 1
    for p in _factorize(n):
        result *= p
    return result


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@dataclass
class GoldbachReport:
    """Result of verify_goldbach().

    Attributes:
        n: The even integer tested.
        is_verified: True if at least one (p1, p2) decomposition exists.
        decomposition_count: Number of distinct unordered pairs {p1, p2}.
        smallest_pair: Pair with the smallest p1.
        largest_pair: Pair with the largest p1 (closest to n/2).
        severity: HIGH if not verified, INFO otherwise.
        verdict: Human-readable summary.
    """
    n: int
    is_verified: bool
    decomposition_count: int
    smallest_pair: Optional[Tuple[int, int]]
    largest_pair: Optional[Tuple[int, int]]
    severity: str
    verdict: str

    @property
    def passed(self) -> bool:
        return self.is_verified

    def __str__(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  Goldbach Verification: {self.verdict}",
            f"{'=' * 60}",
            f"  n = {self.n}",
            f"  Verified: {self.is_verified}",
            f"  Decompositions: {self.decomposition_count}",
        ]
        if self.smallest_pair:
            lines.append(f"  Smallest pair: {self.smallest_pair[0]} + {self.smallest_pair[1]}")
        if self.largest_pair:
            lines.append(f"  Largest pair:  {self.largest_pair[0]} + {self.largest_pair[1]}")
        lines.append(f"  Severity: {self.severity}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


@dataclass
class CollatzReport:
    """Result of verify_collatz().

    Attributes:
        n: Starting integer.
        reached_one: True if the sequence reached 1 within max_steps.
        steps: Number of steps to reach 1 (or max_steps if unresolved).
        max_value: Peak value in the trajectory.
        trajectory_length: Total number of values in the trajectory.
        trajectory: Full trajectory list (only populated for n <= 1000).
        severity: HIGH if unresolved, INFO if reached 1.
        verdict: Human-readable summary.
    """
    n: int
    reached_one: bool
    steps: int
    max_value: int
    trajectory_length: int
    trajectory: Optional[List[int]]
    severity: str
    verdict: str

    @property
    def passed(self) -> bool:
        return self.reached_one

    def __str__(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  Collatz Verification: {self.verdict}",
            f"{'=' * 60}",
            f"  n = {self.n}",
            f"  Reached 1: {self.reached_one}",
            f"  Steps: {self.steps}",
            f"  Max value: {self.max_value}",
            f"  Trajectory length: {self.trajectory_length}",
        ]
        if self.trajectory and len(self.trajectory) <= 30:
            lines.append(f"  Trajectory: {self.trajectory}")
        elif self.trajectory:
            lines.append(f"  Trajectory: [{self.trajectory[0]}, ..., {self.trajectory[-1]}] ({len(self.trajectory)} values)")
        lines.append(f"  Severity: {self.severity}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


@dataclass
class TwinPrimeReport:
    """Result of verify_twin_primes().

    Attributes:
        limit: Upper bound searched.
        count: Number of twin prime pairs found.
        pairs: List of pairs (truncated for large limits; first/last 10).
        largest_pair: The largest twin prime pair found.
        density: twin_prime_count / prime_count.
        prime_count: Total primes up to limit.
        severity: INFO (this is observational, not pass/fail).
        verdict: Human-readable summary.
    """
    limit: int
    count: int
    pairs: List[Tuple[int, int]]
    largest_pair: Optional[Tuple[int, int]]
    density: float
    prime_count: int
    severity: str
    verdict: str

    @property
    def passed(self) -> bool:
        return self.count > 0

    def __str__(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  Twin Prime Search: {self.verdict}",
            f"{'=' * 60}",
            f"  Limit: {self.limit}",
            f"  Twin prime pairs: {self.count}",
            f"  Total primes: {self.prime_count}",
            f"  Density (twin/prime): {self.density:.6f}",
        ]
        if self.largest_pair:
            lines.append(f"  Largest pair: ({self.largest_pair[0]}, {self.largest_pair[1]})")
        if len(self.pairs) <= 20:
            lines.append(f"  All pairs: {self.pairs}")
        else:
            lines.append(f"  First 5: {self.pairs[:5]}")
            lines.append(f"  Last 5:  {self.pairs[-5:]}")
        lines.append(f"  Severity: {self.severity}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


@dataclass
class ABCReport:
    """Result of check_abc_triple().

    Attributes:
        a: First value.
        b: Second value.
        c: Third value (should equal a + b).
        is_valid_triple: True if a+b=c, gcd(a,b)=1, a,b,c > 0.
        radical: rad(abc) = product of distinct prime factors of a*b*c.
        quality: log(c) / log(rad(abc)). ABC conjecture: q < 1+eps.
        is_exceptional: True if quality > 1 (finitely many for each eps).
        severity: HIGH if q > 1.4 (rare), MODERATE if q > 1, LOW otherwise.
        verdict: Human-readable summary.
    """
    a: int
    b: int
    c: int
    is_valid_triple: bool
    radical: int
    quality: float
    is_exceptional: bool
    severity: str
    verdict: str

    @property
    def passed(self) -> bool:
        return self.is_valid_triple

    def __str__(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  ABC Triple Check: {self.verdict}",
            f"{'=' * 60}",
            f"  (a, b, c) = ({self.a}, {self.b}, {self.c})",
            f"  Valid triple: {self.is_valid_triple}",
        ]
        if self.is_valid_triple:
            lines.append(f"  rad(abc) = {self.radical}")
            lines.append(f"  Quality q = log(c)/log(rad) = {self.quality:.6f}")
            lines.append(f"  Exceptional (q > 1): {self.is_exceptional}")
        lines.append(f"  Severity: {self.severity}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


@dataclass
class LegendreReport:
    """Result of verify_legendre().

    Attributes:
        n: The input integer.
        n_squared: n^2.
        n_plus_1_squared: (n+1)^2.
        primes_found: List of primes in the open interval (n^2, (n+1)^2).
        count: Number of primes found.
        is_verified: True if at least one prime exists in the interval.
        severity: HIGH if no prime found (would be a counterexample), INFO otherwise.
        verdict: Human-readable summary.
    """
    n: int
    n_squared: int
    n_plus_1_squared: int
    primes_found: List[int]
    count: int
    is_verified: bool
    severity: str
    verdict: str

    @property
    def passed(self) -> bool:
        return self.is_verified

    def __str__(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  Legendre Verification: {self.verdict}",
            f"{'=' * 60}",
            f"  n = {self.n}",
            f"  Interval: ({self.n_squared}, {self.n_plus_1_squared})",
            f"  Primes found: {self.count}",
        ]
        if self.count <= 20:
            lines.append(f"  Primes: {self.primes_found}")
        else:
            lines.append(f"  First 5: {self.primes_found[:5]}")
            lines.append(f"  Last 5:  {self.primes_found[-5:]}")
        lines.append(f"  Verified: {self.is_verified}")
        lines.append(f"  Severity: {self.severity}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


@dataclass
class PrimeGapReport:
    """Result of prime_gap_analysis().

    Attributes:
        limit: Upper bound searched.
        prime_count: Number of primes up to limit.
        max_gap: Largest gap between consecutive primes.
        max_gap_location: (p, p+gap) where the max gap occurs.
        avg_gap: Mean gap between consecutive primes.
        cramer_ratio: max_gap / (log(p))^2 where p is the prime before max gap.
            Cramer's conjecture: this ratio < 1 for all sufficiently large p.
        severity: INFO (observational). MODERATE if cramer_ratio > 0.9.
        verdict: Human-readable summary.
    """
    limit: int
    prime_count: int
    max_gap: int
    max_gap_location: Tuple[int, int]
    avg_gap: float
    cramer_ratio: float
    severity: str
    verdict: str

    @property
    def passed(self) -> bool:
        return self.cramer_ratio < 1.0

    def __str__(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  Prime Gap Analysis: {self.verdict}",
            f"{'=' * 60}",
            f"  Limit: {self.limit}",
            f"  Primes found: {self.prime_count}",
            f"  Max gap: {self.max_gap} (between {self.max_gap_location[0]} and {self.max_gap_location[1]})",
            f"  Average gap: {self.avg_gap:.4f}",
            f"  Cramer ratio (max_gap / (log p)^2): {self.cramer_ratio:.6f}",
        ]
        if self.cramer_ratio < 1.0:
            lines.append(f"  Cramer's conjecture: consistent (ratio < 1)")
        else:
            lines.append(f"  Cramer's conjecture: EXCEEDED (ratio >= 1)")
        lines.append(f"  Severity: {self.severity}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verification functions
# ---------------------------------------------------------------------------

def verify_goldbach(n: int) -> GoldbachReport:
    """Verify Goldbach's conjecture for a specific even integer.

    For even n > 2, finds all decompositions n = p1 + p2 where both p1 and p2
    are prime, with p1 <= p2.

    Args:
        n: An even integer greater than 2.

    Returns:
        GoldbachReport with decomposition details.

    Raises:
        ValueError: If n is odd or n <= 2.
    """
    if n <= 2:
        raise ValueError(f"Goldbach's conjecture applies to even n > 2, got {n}")
    if n % 2 != 0:
        raise ValueError(f"Goldbach's conjecture applies to even numbers, got {n} (odd)")

    # Use sieve for efficiency
    primes_set = set(prime_sieve(n))
    pairs: List[Tuple[int, int]] = []

    for p in sorted(primes_set):
        if p > n // 2:
            break
        complement = n - p
        if complement in primes_set:
            pairs.append((p, complement))

    is_verified = len(pairs) > 0
    smallest = pairs[0] if pairs else None
    largest = pairs[-1] if pairs else None

    if is_verified:
        severity = "INFO"
        verdict = f"VERIFIED: {n} has {len(pairs)} Goldbach decomposition(s)"
    else:
        severity = "HIGH"
        verdict = f"NOT VERIFIED: no prime pair sums to {n} (potential counterexample!)"

    return GoldbachReport(
        n=n,
        is_verified=is_verified,
        decomposition_count=len(pairs),
        smallest_pair=smallest,
        largest_pair=largest,
        severity=severity,
        verdict=verdict,
    )


def verify_collatz(n: int, max_steps: int = 10_000) -> CollatzReport:
    """Run the Collatz sequence from n and check if it reaches 1.

    The Collatz rule: if even, divide by 2; if odd, compute 3n + 1. Repeat.
    The conjecture states that every positive integer eventually reaches 1.

    Args:
        n: Starting positive integer.
        max_steps: Maximum iterations before declaring UNRESOLVED.

    Returns:
        CollatzReport with trajectory details.

    Raises:
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError(f"Collatz sequence requires n >= 1, got {n}")

    store_trajectory = n <= 1000
    trajectory = [n] if store_trajectory else None
    current = n
    max_value = n
    steps = 0

    while current != 1 and steps < max_steps:
        if current % 2 == 0:
            current = current // 2
        else:
            current = 3 * current + 1
        steps += 1
        if current > max_value:
            max_value = current
        if store_trajectory:
            trajectory.append(current)

    reached_one = current == 1
    traj_len = steps + 1  # includes starting value

    if reached_one:
        severity = "INFO"
        verdict = f"REACHED 1 in {steps} steps (max value {max_value})"
    else:
        severity = "HIGH"
        verdict = f"UNRESOLVED after {max_steps} steps (current={current}, max={max_value})"

    return CollatzReport(
        n=n,
        reached_one=reached_one,
        steps=steps,
        max_value=max_value,
        trajectory_length=traj_len,
        trajectory=trajectory,
        severity=severity,
        verdict=verdict,
    )


def verify_twin_primes(limit: int) -> TwinPrimeReport:
    """Find all twin prime pairs (p, p+2) up to limit.

    A twin prime pair consists of two primes that differ by 2. The twin prime
    conjecture states that there are infinitely many such pairs.

    Args:
        limit: Upper bound for the search (inclusive).

    Returns:
        TwinPrimeReport with pair count, density, and pair list.

    Raises:
        ValueError: If limit < 2.
    """
    if limit < 2:
        raise ValueError(f"Need limit >= 2, got {limit}")

    primes = prime_sieve(limit)
    prime_count = len(primes)
    primes_set = set(primes)

    pairs: List[Tuple[int, int]] = []
    for p in primes:
        if p + 2 <= limit and (p + 2) in primes_set:
            pairs.append((p, p + 2))

    count = len(pairs)
    largest = pairs[-1] if pairs else None
    density = count / prime_count if prime_count > 0 else 0.0

    severity = "INFO"
    if count == 0:
        verdict = f"No twin primes found up to {limit}"
    else:
        verdict = f"Found {count} twin prime pairs up to {limit} (density {density:.4f})"

    return TwinPrimeReport(
        limit=limit,
        count=count,
        pairs=pairs,
        largest_pair=largest,
        density=density,
        prime_count=prime_count,
        severity=severity,
        verdict=verdict,
    )


def check_abc_triple(a: int, b: int, c: int) -> ABCReport:
    """Check whether (a, b, c) is an ABC triple and compute its quality.

    An ABC triple satisfies: a + b = c, gcd(a, b) = 1, and a, b, c > 0.
    The quality is q = log(c) / log(rad(abc)). The ABC conjecture asserts
    that for every epsilon > 0, there are only finitely many triples with
    q > 1 + epsilon.

    Triples with q > 1 are called "exceptional". Triples with q > 1.4 are
    very rare and mathematically interesting.

    Args:
        a: First positive integer.
        b: Second positive integer.
        c: Third positive integer.

    Returns:
        ABCReport with validity, radical, quality, and severity.
    """
    is_valid = True
    reasons = []

    if a < 1 or b < 1 or c < 1:
        is_valid = False
        reasons.append("a, b, c must all be positive")
    if a + b != c:
        is_valid = False
        reasons.append(f"a + b = {a + b} != {c} = c")
    if is_valid and math.gcd(a, b) != 1:
        is_valid = False
        reasons.append(f"gcd(a, b) = {math.gcd(a, b)} != 1")

    if not is_valid:
        return ABCReport(
            a=a, b=b, c=c,
            is_valid_triple=False,
            radical=0,
            quality=0.0,
            is_exceptional=False,
            severity="LOW",
            verdict=f"INVALID: {'; '.join(reasons)}",
        )

    rad = radical(a * b * c)
    # quality = log(c) / log(rad(abc))
    if rad <= 1:
        # rad(abc) = 1 only if abc = 1, but a,b,c > 0 and a+b=c means c >= 2
        quality = float('inf')
    else:
        quality = math.log(c) / math.log(rad)

    is_exceptional = quality > 1.0

    if quality > 1.4:
        severity = "HIGH"
        verdict = f"EXCEPTIONAL (q={quality:.4f} > 1.4): rare high-quality ABC triple"
    elif quality > 1.0:
        severity = "MODERATE"
        verdict = f"EXCEPTIONAL (q={quality:.4f} > 1): quality ABC triple"
    else:
        severity = "LOW"
        verdict = f"Ordinary triple (q={quality:.4f} <= 1)"

    return ABCReport(
        a=a, b=b, c=c,
        is_valid_triple=True,
        radical=rad,
        quality=quality,
        is_exceptional=is_exceptional,
        severity=severity,
        verdict=verdict,
    )


def verify_legendre(n: int) -> LegendreReport:
    """Check Legendre's conjecture: there exists a prime between n^2 and (n+1)^2.

    Args:
        n: Non-negative integer.

    Returns:
        LegendreReport with the primes found in the open interval (n^2, (n+1)^2).

    Raises:
        ValueError: If n < 0.
    """
    if n < 0:
        raise ValueError(f"Legendre's conjecture requires n >= 0, got {n}")

    n_sq = n * n
    n1_sq = (n + 1) * (n + 1)

    # For small ranges, test each candidate; for large ranges, use sieve
    primes_found: List[int] = []
    if n1_sq <= 10_000_000:
        # Sieve up to (n+1)^2 and filter
        all_primes = prime_sieve(n1_sq)
        primes_found = [p for p in all_primes if n_sq < p < n1_sq]
    else:
        # Test each odd number in range (for very large n)
        lo = n_sq + 1
        if lo % 2 == 0:
            lo += 1
        # Also check 2 if in range
        if n_sq < 2 < n1_sq:
            primes_found.append(2)
        for candidate in range(lo, n1_sq, 2):
            if is_prime(candidate):
                primes_found.append(candidate)

    count = len(primes_found)
    is_verified = count > 0

    if is_verified:
        severity = "INFO"
        verdict = f"VERIFIED: {count} prime(s) between {n_sq} and {n1_sq}"
    else:
        severity = "HIGH"
        verdict = f"NOT VERIFIED: no prime between {n_sq} and {n1_sq} (potential counterexample!)"

    return LegendreReport(
        n=n,
        n_squared=n_sq,
        n_plus_1_squared=n1_sq,
        primes_found=primes_found,
        count=count,
        is_verified=is_verified,
        severity=severity,
        verdict=verdict,
    )


def prime_gap_analysis(limit: int) -> PrimeGapReport:
    """Analyze gaps between consecutive primes up to limit.

    Computes max gap, average gap, and the Cramer ratio (max_gap / (log p)^2).
    Cramer's conjecture predicts this ratio stays below 1 for all primes.

    Args:
        limit: Upper bound for the prime search.

    Returns:
        PrimeGapReport with gap statistics and Cramer ratio.

    Raises:
        ValueError: If limit < 3 (need at least two primes).
    """
    if limit < 3:
        raise ValueError(f"Need limit >= 3 for gap analysis, got {limit}")

    primes = prime_sieve(limit)
    prime_count = len(primes)

    if prime_count < 2:
        raise ValueError(f"Only {prime_count} prime(s) up to {limit}; need at least 2")

    max_gap = 0
    max_gap_start = primes[0]
    total_gap = 0

    for i in range(1, prime_count):
        gap = primes[i] - primes[i - 1]
        total_gap += gap
        if gap > max_gap:
            max_gap = gap
            max_gap_start = primes[i - 1]

    avg_gap = total_gap / (prime_count - 1)
    max_gap_end = max_gap_start + max_gap

    # Cramer ratio: max_gap / (log p)^2 where p is the prime before the gap
    log_p = math.log(max_gap_start) if max_gap_start > 1 else 1.0
    cramer_ratio = max_gap / (log_p ** 2)

    if cramer_ratio > 0.9:
        severity = "MODERATE"
    else:
        severity = "INFO"

    if cramer_ratio < 1.0:
        verdict = f"Consistent with Cramer's conjecture (ratio={cramer_ratio:.4f})"
    else:
        verdict = f"Cramer ratio {cramer_ratio:.4f} >= 1 (exceeds conjecture bound at p={max_gap_start})"

    return PrimeGapReport(
        limit=limit,
        prime_count=prime_count,
        max_gap=max_gap,
        max_gap_location=(max_gap_start, max_gap_end),
        avg_gap=avg_gap,
        cramer_ratio=cramer_ratio,
        severity=severity,
        verdict=verdict,
    )
