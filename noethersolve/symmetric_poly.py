"""
noethersolve.symmetric_poly — Symmetric polynomial calculator.

Computes elementary symmetric polynomials (e_k), power sums (p_k), and
verifies Newton's identities connecting them. Built for the c10_repair
domain where models consistently fail on Newton's identity p2 = e1^2 - 2*e2
and related symmetric function relationships.

Newton's identities connect power sums p_k = sum(x_i^k) to elementary
symmetric polynomials e_k:
    p1 = e1
    p2 = e1*p1 - 2*e2
    p3 = e1*p2 - e2*p1 + 3*e3
    General: p_k = sum_{i=1}^{k-1} (-1)^{i-1} e_i * p_{k-i} + (-1)^{k-1} * k * e_k

Usage:
    from noethersolve.symmetric_poly import (
        calc_elementary_symmetric, calc_power_sum,
        verify_newton_identities, calc_all_symmetric,
    )

    values = [1.0, 2.0, 3.0]

    # Elementary symmetric polynomials
    e1 = calc_elementary_symmetric(values, 1)  # 6.0 (sum)
    e2 = calc_elementary_symmetric(values, 2)  # 11.0 (sum of products of pairs)
    e3 = calc_elementary_symmetric(values, 3)  # 6.0 (product of all)

    # Power sums
    p1 = calc_power_sum(values, 1)  # 6.0
    p2 = calc_power_sum(values, 2)  # 14.0

    # Verify Newton's identities
    result = verify_newton_identities(values, max_k=3)
    print(result)

    # Full report
    report = calc_all_symmetric(values)
    print(report)
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import List


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def calc_elementary_symmetric(values: List[float], k: int) -> float:
    """Compute the k-th elementary symmetric polynomial e_k.

    e_k(x_1, ..., x_n) = sum of all products of k distinct elements.

    Examples:
        e_1(1,2,3) = 1+2+3 = 6
        e_2(1,2,3) = 1*2 + 1*3 + 2*3 = 11
        e_3(1,2,3) = 1*2*3 = 6

    Args:
        values: List of numeric values.
        k: Order of the elementary symmetric polynomial (1 <= k <= len(values)).

    Returns:
        The value of e_k.

    Raises:
        ValueError: If k < 0 or k > len(values).
    """
    n = len(values)
    if k < 0 or k > n:
        raise ValueError(f"k={k} out of range for {n} values (need 0 <= k <= {n})")
    if k == 0:
        return 1.0
    total = 0.0
    for combo in combinations(values, k):
        prod = 1.0
        for x in combo:
            prod *= x
        total += prod
    return total


def calc_power_sum(values: List[float], k: int) -> float:
    """Compute the k-th power sum p_k = sum(x_i^k).

    Args:
        values: List of numeric values.
        k: Power to raise each value to (k >= 1).

    Returns:
        The value of p_k.

    Raises:
        ValueError: If k < 1.
    """
    if k < 1:
        raise ValueError(f"k={k} must be >= 1")
    return sum(x ** k for x in values)


# ---------------------------------------------------------------------------
# Newton's identity verification
# ---------------------------------------------------------------------------


@dataclass
class NewtonIdentityCheck:
    """Result of checking a single Newton's identity at order k."""

    k: int
    lhs: float  # p_k (direct power sum)
    rhs: float  # Newton's identity reconstruction from e_i and p_j
    residual: float  # |lhs - rhs|
    passed: bool  # residual < tolerance


@dataclass
class NewtonIdentityResult:
    """Result of verifying Newton's identities up to max_k."""

    values: List[float]
    max_k: int
    checks: List[NewtonIdentityCheck] = field(default_factory=list)
    all_passed: bool = False
    tolerance: float = 1e-10

    def __str__(self) -> str:
        lines = [
            f"Newton's Identity Verification (n={len(self.values)}, max_k={self.max_k})",
            f"  Values: {self.values}",
            f"  Tolerance: {self.tolerance}",
            "",
        ]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(
                f"  k={c.k}: p_{c.k}={c.lhs:.6g}, "
                f"Newton RHS={c.rhs:.6g}, "
                f"residual={c.residual:.2e} [{status}]"
            )
        lines.append("")
        verdict = "ALL PASSED" if self.all_passed else "SOME FAILED"
        lines.append(f"  Result: {verdict}")
        return "\n".join(lines)


def _newton_rhs(e_vals: List[float], p_vals: List[float], k: int) -> float:
    """Compute the RHS of Newton's identity for order k.

    Newton's identity:
        p_k = sum_{i=1}^{k-1} (-1)^{i-1} * e_i * p_{k-i} + (-1)^{k-1} * k * e_k

    Args:
        e_vals: List where e_vals[i] = e_{i+1} (0-indexed).
        p_vals: List where p_vals[i] = p_{i+1} (0-indexed), must have p_1..p_{k-1}.
        k: The order to compute.

    Returns:
        The RHS value that should equal p_k.
    """
    rhs = 0.0
    for i in range(1, k):
        # e_i is e_vals[i-1], p_{k-i} is p_vals[k-i-1]
        sign = (-1) ** (i - 1)
        rhs += sign * e_vals[i - 1] * p_vals[k - i - 1]
    # Last term: (-1)^{k-1} * k * e_k
    if k <= len(e_vals):
        rhs += ((-1) ** (k - 1)) * k * e_vals[k - 1]
    return rhs


def verify_newton_identities(
    values: List[float], max_k: int = 5, tolerance: float = 1e-10
) -> NewtonIdentityResult:
    """Verify Newton's identities connecting power sums and elementary symmetric polys.

    Newton's identities:
        p_1 = e_1
        p_2 = e_1*p_1 - 2*e_2
        p_3 = e_1*p_2 - e_2*p_1 + 3*e_3
        ...

    Args:
        values: List of numeric values.
        max_k: Maximum order to check (capped at len(values)).
        tolerance: Maximum allowed residual for a PASS.

    Returns:
        NewtonIdentityResult with per-order checks.
    """
    n = len(values)
    if n == 0:
        raise ValueError("Need at least one value")
    max_k = min(max_k, n)

    # Precompute all needed e_k and p_k
    e_vals = [calc_elementary_symmetric(values, k) for k in range(1, max_k + 1)]
    p_vals = [calc_power_sum(values, k) for k in range(1, max_k + 1)]

    checks = []
    for k in range(1, max_k + 1):
        lhs = p_vals[k - 1]
        rhs = _newton_rhs(e_vals, p_vals, k)
        residual = abs(lhs - rhs)
        checks.append(
            NewtonIdentityCheck(
                k=k,
                lhs=lhs,
                rhs=rhs,
                residual=residual,
                passed=residual < tolerance,
            )
        )

    result = NewtonIdentityResult(
        values=list(values),
        max_k=max_k,
        checks=checks,
        all_passed=all(c.passed for c in checks),
        tolerance=tolerance,
    )
    return result


# ---------------------------------------------------------------------------
# Full symmetric polynomial report
# ---------------------------------------------------------------------------


@dataclass
class SymmetricPolyResult:
    """Complete symmetric polynomial analysis of a set of values."""

    values: List[float]
    n: int
    elementary: List[float]  # e_1, e_2, ..., e_n
    power_sums: List[float]  # p_1, p_2, ..., p_n
    sigma_2: float  # sum of r_ij^2 for i < j = e1^2 - 2*e2
    newton_result: NewtonIdentityResult = None

    def __str__(self) -> str:
        lines = [
            f"Symmetric Polynomial Report (n={self.n})",
            f"  Values: {self.values}",
            "",
            "  Elementary symmetric polynomials:",
        ]
        for k, ek in enumerate(self.elementary, 1):
            lines.append(f"    e_{k} = {ek:.6g}")

        lines.append("")
        lines.append("  Power sums:")
        for k, pk in enumerate(self.power_sums, 1):
            lines.append(f"    p_{k} = {pk:.6g}")

        lines.append("")
        lines.append(f"  Sigma_2 = sum(x_i*x_j for i<j) = e_2 = {self.elementary[1] if self.n >= 2 else 'N/A'}")
        lines.append(f"  sum(x_i^2) = p_2 = e_1^2 - 2*e_2 = {self.sigma_2:.6g}")

        if self.newton_result is not None:
            lines.append("")
            verdict = "ALL PASSED" if self.newton_result.all_passed else "SOME FAILED"
            lines.append(f"  Newton's identities: {verdict}")
            for c in self.newton_result.checks:
                status = "PASS" if c.passed else "FAIL"
                lines.append(f"    k={c.k}: residual={c.residual:.2e} [{status}]")

        return "\n".join(lines)


def calc_all_symmetric(values: List[float]) -> SymmetricPolyResult:
    """Compute all elementary symmetric polynomials, power sums, and verify Newton's identities.

    Also computes Sigma_2 = p_2 = e_1^2 - 2*e_2 (the identity LLMs most
    commonly get wrong in the c10_repair domain).

    Args:
        values: List of numeric values.

    Returns:
        SymmetricPolyResult with complete analysis.

    Raises:
        ValueError: If values is empty.
    """
    n = len(values)
    if n == 0:
        raise ValueError("Need at least one value")

    elementary = [calc_elementary_symmetric(values, k) for k in range(1, n + 1)]
    power_sums = [calc_power_sum(values, k) for k in range(1, n + 1)]

    # Sigma_2 = p_2 = e_1^2 - 2*e_2 (for n >= 2)
    if n >= 2:
        sigma_2 = elementary[0] ** 2 - 2 * elementary[1]
    else:
        sigma_2 = values[0] ** 2

    newton_result = verify_newton_identities(values, max_k=n)

    return SymmetricPolyResult(
        values=list(values),
        n=n,
        elementary=elementary,
        power_sums=power_sums,
        sigma_2=sigma_2,
        newton_result=newton_result,
    )
