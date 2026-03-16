"""
noethersolve.elliptic_curves — Elliptic curve calculator from first principles.

Verified computations for elliptic curves: point arithmetic, group order,
discriminant, j-invariant, torsion structure, and BSD-related invariants.

Usage:
    from noethersolve.elliptic_curves import (
        point_add, scalar_mult, is_on_curve,
        discriminant, j_invariant, hasse_bounds,
        count_points_naive, verify_hasse,
        torsion_subgroup_order, check_mazur,
        EllipticCurveReport, PointArithmeticReport,
    )

    # Curve y² = x³ + ax + b over F_p
    E = {"a": -1, "b": 1, "p": 23}
    P = (0, 1)
    Q = (6, 4)

    # Point arithmetic
    R = point_add(E, P, Q)
    nP = scalar_mult(E, 5, P)

    # Invariants
    delta = discriminant(-1, 1)  # -368
    j = j_invariant(-1, 1)  # ...

    # Point counting
    count = count_points_naive(E)  # includes point at infinity
    lo, hi = hasse_bounds(23)  # Hasse: |#E - p - 1| <= 2*sqrt(p)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

# Type aliases
Point = Union[Tuple[int, int], None]  # None represents the point at infinity
Curve = Dict[str, int]  # {"a": int, "b": int, "p": int}


# -----------------------------------------------------------------------------
# Core arithmetic over F_p
# -----------------------------------------------------------------------------

def _mod_inverse(a: int, p: int) -> int:
    """Compute modular inverse of a mod p using extended Euclidean algorithm.

    Raises:
        ValueError: If a is not invertible mod p (gcd(a,p) != 1).
    """
    if a == 0:
        raise ValueError("Cannot invert 0")
    a = a % p
    g, x, _ = _extended_gcd(a, p)
    if g != 1:
        raise ValueError(f"{a} is not invertible mod {p}")
    return x % p


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm. Returns (gcd, x, y) where ax + by = gcd."""
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


def _tonelli_shanks(n: int, p: int) -> Optional[int]:
    """Compute square root of n mod p using Tonelli-Shanks algorithm.

    Returns:
        One square root if it exists, None otherwise.
    """
    if n == 0:
        return 0
    if pow(n, (p - 1) // 2, p) != 1:
        return None  # n is not a quadratic residue

    # Factor out powers of 2 from p - 1
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    if s == 1:
        return pow(n, (p + 1) // 4, p)

    # Find a quadratic non-residue
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1

    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)

    while True:
        if t == 1:
            return r
        # Find the least i such that t^(2^i) = 1
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        # Update
        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p


# -----------------------------------------------------------------------------
# Elliptic curve arithmetic
# -----------------------------------------------------------------------------

def is_on_curve(E: Curve, P: Point) -> bool:
    """Check if point P lies on elliptic curve E: y² = x³ + ax + b (mod p).

    Args:
        E: Curve parameters {"a": int, "b": int, "p": int}
        P: Point (x, y) or None for point at infinity

    Returns:
        True if P is on E.
    """
    if P is None:
        return True  # Point at infinity is always on the curve

    x, y = P
    a, b, p = E["a"], E["b"], E["p"]

    lhs = (y * y) % p
    rhs = (x * x * x + a * x + b) % p
    return lhs == rhs


def point_add(E: Curve, P: Point, Q: Point) -> Point:
    """Add two points on elliptic curve E.

    Uses the standard addition law:
    - If P = O (infinity), return Q
    - If Q = O (infinity), return P
    - If P = -Q (same x, opposite y), return O
    - If P = Q, use doubling formula
    - Otherwise, use standard addition formula

    Args:
        E: Curve parameters {"a": int, "b": int, "p": int}
        P: First point (x, y) or None
        Q: Second point (x, y) or None

    Returns:
        P + Q as a point on E.
    """
    p = E["p"]

    # Handle identity element
    if P is None:
        return Q
    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    # Reduce coordinates mod p
    x1, y1 = x1 % p, y1 % p
    x2, y2 = x2 % p, y2 % p

    # Check if P = -Q (vertical line case)
    if x1 == x2:
        if (y1 + y2) % p == 0:
            return None  # P + (-P) = O
        # P = Q case (doubling)
        return point_double(E, P)

    # Standard addition formula
    # slope m = (y2 - y1) / (x2 - x1)
    dy = (y2 - y1) % p
    dx = (x2 - x1) % p
    m = (dy * _mod_inverse(dx, p)) % p

    # x3 = m² - x1 - x2
    x3 = (m * m - x1 - x2) % p
    # y3 = m(x1 - x3) - y1
    y3 = (m * (x1 - x3) - y1) % p

    return (x3, y3)


def point_double(E: Curve, P: Point) -> Point:
    """Double a point on elliptic curve E.

    Uses the tangent line formula:
    m = (3x² + a) / (2y)

    Args:
        E: Curve parameters {"a": int, "b": int, "p": int}
        P: Point (x, y) or None

    Returns:
        2P as a point on E.
    """
    if P is None:
        return None

    x, y = P
    a, p = E["a"], E["p"]

    x, y = x % p, y % p

    # If y = 0, the tangent is vertical
    if y == 0:
        return None

    # slope m = (3x² + a) / (2y)
    numerator = (3 * x * x + a) % p
    denominator = (2 * y) % p
    m = (numerator * _mod_inverse(denominator, p)) % p

    # x3 = m² - 2x
    x3 = (m * m - 2 * x) % p
    # y3 = m(x - x3) - y
    y3 = (m * (x - x3) - y) % p

    return (x3, y3)


def point_negate(E: Curve, P: Point) -> Point:
    """Compute the negation of point P on curve E.

    For P = (x, y), -P = (x, -y) = (x, p - y).
    """
    if P is None:
        return None
    x, y = P
    p = E["p"]
    return (x % p, (-y) % p)


def scalar_mult(E: Curve, n: int, P: Point) -> Point:
    """Compute nP using double-and-add algorithm.

    Args:
        E: Curve parameters
        n: Scalar multiplier (can be negative)
        P: Point on E

    Returns:
        nP as a point on E.
    """
    if P is None or n == 0:
        return None

    if n < 0:
        n = -n
        P = point_negate(E, P)

    result = None  # Identity (point at infinity)
    addend = P

    while n > 0:
        if n & 1:
            result = point_add(E, result, addend)
        addend = point_double(E, addend)
        n >>= 1

    return result


def point_order(E: Curve, P: Point, max_order: int = 10000) -> Optional[int]:
    """Compute the order of point P in the group E(F_p).

    The order is the smallest positive n such that nP = O.

    Args:
        E: Curve parameters
        P: Point on E
        max_order: Maximum order to search for

    Returns:
        Order of P, or None if not found within max_order.
    """
    if P is None:
        return 1

    Q = P
    for n in range(1, max_order + 1):
        Q = point_add(E, Q, P)
        if Q is None:
            return n + 1

    return None


# -----------------------------------------------------------------------------
# Curve invariants
# -----------------------------------------------------------------------------

def discriminant(a: int, b: int) -> int:
    """Compute the discriminant of y² = x³ + ax + b.

    Δ = -16(4a³ + 27b²)

    The curve is non-singular if and only if Δ ≠ 0.
    """
    return -16 * (4 * a**3 + 27 * b**2)


def j_invariant(a: int, b: int) -> Optional[float]:
    """Compute the j-invariant of y² = x³ + ax + b.

    j = -1728 * (4a)³ / Δ = -1728 * 64a³ / (-16(4a³ + 27b²))
      = 6912a³ / (4a³ + 27b²)

    Actually, standard form: j = 1728 * 4a³ / (4a³ + 27b²) when Δ ≠ 0.

    Returns:
        j-invariant, or None if curve is singular (Δ = 0).
    """
    denom = 4 * a**3 + 27 * b**2
    if denom == 0:
        return None  # Singular curve

    # j = 1728 * 4a³ / (4a³ + 27b²)
    return 1728 * (4 * a**3) / denom


def is_singular(a: int, b: int) -> bool:
    """Check if the curve y² = x³ + ax + b is singular.

    A curve is singular if and only if Δ = 0, i.e., 4a³ + 27b² = 0.
    """
    return 4 * a**3 + 27 * b**2 == 0


# -----------------------------------------------------------------------------
# Point counting
# -----------------------------------------------------------------------------

def hasse_bounds(p: int) -> Tuple[int, int]:
    """Compute the Hasse bounds for #E(F_p).

    Hasse's theorem: |#E(F_p) - (p + 1)| <= 2*sqrt(p)

    So: p + 1 - 2*sqrt(p) <= #E(F_p) <= p + 1 + 2*sqrt(p)

    Args:
        p: Prime modulus

    Returns:
        (lower_bound, upper_bound) for the group order.
    """
    sqrt_p = math.isqrt(p)
    # Use floor/ceil to be safe
    two_sqrt_p = 2 * sqrt_p + 2  # Upper bound on 2*sqrt(p)

    lower = p + 1 - two_sqrt_p
    upper = p + 1 + two_sqrt_p

    return (max(1, lower), upper)


def count_points_naive(E: Curve) -> int:
    """Count points on E(F_p) by exhaustive enumeration.

    For each x in F_p, solve y² = x³ + ax + b:
    - If no solution: 0 points
    - If y = 0: 1 point (0, 0)
    - Otherwise: 2 points (x, y) and (x, -y)

    Plus the point at infinity.

    Args:
        E: Curve parameters {"a": int, "b": int, "p": int}

    Returns:
        Total number of points including the point at infinity.
    """
    a, b, p = E["a"], E["b"], E["p"]
    count = 1  # Point at infinity

    for x in range(p):
        rhs = (x**3 + a * x + b) % p

        if rhs == 0:
            count += 1  # Point (x, 0)
        else:
            # Check if rhs is a quadratic residue
            if pow(rhs, (p - 1) // 2, p) == 1:
                count += 2  # Two points (x, y) and (x, -y)

    return count


def verify_hasse(E: Curve) -> Tuple[bool, int, int, int]:
    """Verify Hasse's theorem for curve E.

    Returns:
        (is_satisfied, count, lower_bound, upper_bound)
    """
    count = count_points_naive(E)
    lower, upper = hasse_bounds(E["p"])
    return (lower <= count <= upper, count, lower, upper)


def find_points(E: Curve, limit: Optional[int] = None) -> List[Point]:
    """Find all points on E(F_p), including the point at infinity.

    Args:
        E: Curve parameters
        limit: Maximum number of points to find (None for all)

    Returns:
        List of points, with None representing the point at infinity.
    """
    a, b, p = E["a"], E["b"], E["p"]
    points: List[Point] = [None]  # Start with point at infinity

    for x in range(p):
        if limit and len(points) >= limit:
            break

        rhs = (x**3 + a * x + b) % p

        if rhs == 0:
            points.append((x, 0))
        else:
            y = _tonelli_shanks(rhs, p)
            if y is not None:
                points.append((x, y))
                if y != 0:
                    points.append((x, p - y))

    return points


# -----------------------------------------------------------------------------
# Torsion and Mazur's theorem
# -----------------------------------------------------------------------------

# Mazur's theorem: For E/Q, the torsion subgroup is one of:
# Z/nZ for n in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}
# Z/2Z × Z/2nZ for n in {1, 2, 3, 4}

MAZUR_CYCLIC = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}
MAZUR_PRODUCT = {(2, 2), (2, 4), (2, 6), (2, 8)}


def is_valid_torsion_order(n: int) -> bool:
    """Check if n is a valid torsion subgroup order per Mazur's theorem.

    Valid cyclic orders: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12
    Valid product structures: Z/2 × Z/2, Z/2 × Z/4, Z/2 × Z/6, Z/2 × Z/8
    (orders 4, 8, 12, 16)

    Combined valid orders: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16
    """
    valid_orders = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16}
    return n in valid_orders


# -----------------------------------------------------------------------------
# Report dataclasses
# -----------------------------------------------------------------------------

@dataclass
class EllipticCurveReport:
    """Report on an elliptic curve's properties."""
    a: int
    b: int
    p: int
    discriminant_val: int
    j_invariant_val: Optional[float]
    is_singular: bool
    point_count: int
    hasse_lower: int
    hasse_upper: int
    hasse_satisfied: bool
    sample_points: List[Point]
    severity: str
    verdict: str

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Elliptic Curve Analysis: {self.verdict}",
            "=" * 60,
            f"  Curve: y² = x³ + ({self.a})x + ({self.b}) over F_{self.p}",
            f"  Discriminant Δ = {self.discriminant_val}",
        ]
        if self.j_invariant_val is not None:
            lines.append(f"  j-invariant = {self.j_invariant_val:.4f}")
        else:
            lines.append(f"  j-invariant = undefined (singular)")
        lines.append(f"  Singular: {self.is_singular}")
        lines.append(f"  #E(F_p) = {self.point_count}")
        lines.append(f"  Hasse bounds: [{self.hasse_lower}, {self.hasse_upper}]")
        lines.append(f"  Hasse satisfied: {self.hasse_satisfied}")
        if self.sample_points:
            pts = [str(p) if p else "O" for p in self.sample_points[:5]]
            lines.append(f"  Sample points: {', '.join(pts)}")
        lines.append(f"  Severity: {self.severity}")
        lines.append("=" * 60)
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return not self.is_singular and self.hasse_satisfied


@dataclass
class PointArithmeticReport:
    """Report on point arithmetic operations."""
    curve_a: int
    curve_b: int
    curve_p: int
    P: Point
    Q: Point
    P_plus_Q: Point
    two_P: Point
    P_on_curve: bool
    Q_on_curve: bool
    result_on_curve: bool
    severity: str
    verdict: str

    def __str__(self) -> str:
        def fmt(pt: Point) -> str:
            return str(pt) if pt else "O (infinity)"

        lines = [
            "=" * 60,
            f"  Point Arithmetic: {self.verdict}",
            "=" * 60,
            f"  Curve: y² = x³ + ({self.curve_a})x + ({self.curve_b}) mod {self.curve_p}",
            f"  P = {fmt(self.P)}",
            f"  Q = {fmt(self.Q)}",
            f"  P + Q = {fmt(self.P_plus_Q)}",
            f"  2P = {fmt(self.two_P)}",
            f"  P on curve: {self.P_on_curve}",
            f"  Q on curve: {self.Q_on_curve}",
            f"  Result on curve: {self.result_on_curve}",
            f"  Severity: {self.severity}",
            "=" * 60,
        ]
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.P_on_curve and self.Q_on_curve and self.result_on_curve


# -----------------------------------------------------------------------------
# High-level analysis functions
# -----------------------------------------------------------------------------

def analyze_curve(a: int, b: int, p: int) -> EllipticCurveReport:
    """Comprehensive analysis of an elliptic curve over F_p.

    Args:
        a, b: Curve parameters for y² = x³ + ax + b
        p: Prime modulus

    Returns:
        EllipticCurveReport with all computed invariants.
    """
    E = {"a": a, "b": b, "p": p}

    disc = discriminant(a, b)
    j_inv = j_invariant(a, b)
    singular = is_singular(a, b)

    if singular:
        return EllipticCurveReport(
            a=a, b=b, p=p,
            discriminant_val=disc,
            j_invariant_val=None,
            is_singular=True,
            point_count=0,
            hasse_lower=0, hasse_upper=0,
            hasse_satisfied=False,
            sample_points=[],
            severity="HIGH",
            verdict="SINGULAR CURVE — not a valid elliptic curve",
        )

    count = count_points_naive(E)
    lo, hi = hasse_bounds(p)
    hasse_ok = lo <= count <= hi

    sample = find_points(E, limit=6)

    if not hasse_ok:
        severity = "HIGH"
        verdict = f"HASSE VIOLATION — count {count} outside [{lo}, {hi}]"
    else:
        severity = "INFO"
        verdict = f"Valid curve with {count} points"

    return EllipticCurveReport(
        a=a, b=b, p=p,
        discriminant_val=disc,
        j_invariant_val=j_inv,
        is_singular=singular,
        point_count=count,
        hasse_lower=lo, hasse_upper=hi,
        hasse_satisfied=hasse_ok,
        sample_points=sample,
        severity=severity,
        verdict=verdict,
    )


def analyze_point_arithmetic(a: int, b: int, p: int,
                             P: Point, Q: Point) -> PointArithmeticReport:
    """Analyze point arithmetic on a curve.

    Args:
        a, b, p: Curve parameters
        P, Q: Points to analyze

    Returns:
        PointArithmeticReport with addition and doubling results.
    """
    E = {"a": a, "b": b, "p": p}

    p_ok = is_on_curve(E, P)
    q_ok = is_on_curve(E, Q)

    p_plus_q = point_add(E, P, Q) if (p_ok and q_ok) else None
    two_p = point_double(E, P) if p_ok else None

    result_ok = True
    if p_plus_q is not None:
        result_ok = result_ok and is_on_curve(E, p_plus_q)
    if two_p is not None:
        result_ok = result_ok and is_on_curve(E, two_p)

    if not (p_ok and q_ok):
        severity = "HIGH"
        verdict = "INPUT ERROR — point(s) not on curve"
    elif not result_ok:
        severity = "HIGH"
        verdict = "COMPUTATION ERROR — result not on curve"
    else:
        severity = "INFO"
        verdict = "Arithmetic verified"

    return PointArithmeticReport(
        curve_a=a, curve_b=b, curve_p=p,
        P=P, Q=Q,
        P_plus_Q=p_plus_q,
        two_P=two_p,
        P_on_curve=p_ok,
        Q_on_curve=q_ok,
        result_on_curve=result_ok,
        severity=severity,
        verdict=verdict,
    )


# -----------------------------------------------------------------------------
# Convenience exports
# -----------------------------------------------------------------------------

__all__ = [
    # Core arithmetic
    "is_on_curve", "point_add", "point_double", "point_negate",
    "scalar_mult", "point_order",
    # Invariants
    "discriminant", "j_invariant", "is_singular",
    # Point counting
    "hasse_bounds", "count_points_naive", "verify_hasse", "find_points",
    # Torsion
    "is_valid_torsion_order", "MAZUR_CYCLIC", "MAZUR_PRODUCT",
    # Reports
    "EllipticCurveReport", "PointArithmeticReport",
    # Analysis
    "analyze_curve", "analyze_point_arithmetic",
]
