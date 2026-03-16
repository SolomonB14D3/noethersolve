"""Intersection theory calculations for algebraic geometry.

Verified computational tools for:
- Bezout theorem intersection counts
- Self-intersection numbers on surfaces
- Genus-degree formula for plane curves
- Canonical divisor calculations
- Chern number relations (Noether formula)
- Classical enumerative results (27 lines, 28 bitangents)
"""

from dataclasses import dataclass
from math import comb, factorial
from typing import Optional


@dataclass
class BezoutReport:
    """Result of Bezout intersection computation."""
    degree_1: int
    degree_2: int
    intersection_count: int
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        return (f"Bezout: deg {self.degree_1} × deg {self.degree_2} = "
                f"{self.intersection_count} points (with multiplicity)\n"
                f"  {self.explanation}")


@dataclass
class GenusReport:
    """Result of genus calculation."""
    degree: int
    genus: int
    formula: str
    curve_type: str
    passed: bool = True

    def __str__(self) -> str:
        return (f"Genus of degree {self.degree} smooth plane curve: {self.genus}\n"
                f"  Formula: {self.formula}\n"
                f"  Type: {self.curve_type}")


@dataclass
class SelfIntersectionReport:
    """Result of self-intersection calculation."""
    surface: str
    divisor: str
    self_intersection: int
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        return (f"Self-intersection on {self.surface}:\n"
                f"  {self.divisor}·{self.divisor} = {self.self_intersection}\n"
                f"  {self.explanation}")


@dataclass
class CanonicalReport:
    """Result of canonical divisor calculation."""
    surface: str
    K_degree: int
    K_squared: int
    explanation: str
    is_fano: bool
    passed: bool = True

    def __str__(self) -> str:
        fano_str = "Fano variety (-K ample)" if self.is_fano else "not Fano"
        return (f"Canonical divisor on {self.surface}:\n"
                f"  K degree: {self.K_degree}\n"
                f"  K² = {self.K_squared}\n"
                f"  {fano_str}\n"
                f"  {self.explanation}")


@dataclass
class NoetherReport:
    """Result of Noether formula calculation."""
    c1_squared: int
    c2: int
    chi: float
    formula_check: bool
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        check = "✓" if self.formula_check else "✗"
        return (f"Noether formula check: c₁² + c₂ = 12χ\n"
                f"  c₁² = {self.c1_squared}, c₂ = {self.c2}, χ = {self.chi}\n"
                f"  {self.c1_squared} + {self.c2} = {self.c1_squared + self.c2}, "
                f"12χ = {12 * self.chi} {check}\n"
                f"  {self.explanation}")


@dataclass
class EnumerativeReport:
    """Result of classical enumerative calculation."""
    problem: str
    count: int
    symmetry_group: str
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        return (f"{self.problem}: {self.count}\n"
                f"  Symmetry: {self.symmetry_group}\n"
                f"  {self.explanation}")


# ============================================================
# Core Calculations
# ============================================================

def bezout_intersection(d1: int, d2: int) -> BezoutReport:
    """Compute intersection count by Bezout's theorem.

    Two plane curves of degrees d1 and d2 with no common component
    intersect in EXACTLY d1 * d2 points, counted with multiplicity.

    Args:
        d1: Degree of first curve
        d2: Degree of second curve

    Returns:
        BezoutReport with exact intersection count

    Example:
        >>> bezout_intersection(2, 3)  # conic and cubic
        BezoutReport with intersection_count=6
    """
    if d1 < 1 or d2 < 1:
        return BezoutReport(d1, d2, 0, "Invalid: degrees must be positive", False)

    count = d1 * d2
    explanation = (f"By Bezout's theorem, curves of degrees {d1} and {d2} "
                   f"intersect in exactly {d1} × {d2} = {count} points "
                   f"(counted with multiplicity in projective plane).")
    return BezoutReport(d1, d2, count, explanation)


def genus_degree_formula(d: int) -> GenusReport:
    """Compute genus of smooth plane curve by genus-degree formula.

    For a smooth plane curve of degree d:
        g = (d-1)(d-2)/2

    Args:
        d: Degree of the smooth plane curve

    Returns:
        GenusReport with genus and curve type
    """
    if d < 1:
        return GenusReport(d, 0, "invalid", "invalid degree", False)

    g = (d - 1) * (d - 2) // 2
    formula = f"g = (d-1)(d-2)/2 = ({d}-1)({d}-2)/2 = {g}"

    if g == 0:
        curve_type = "rational curve (genus 0)"
    elif g == 1:
        curve_type = "elliptic curve (genus 1)"
    else:
        curve_type = f"curve of genus {g}"

    return GenusReport(d, g, formula, curve_type)


def self_intersection_Pn(n: int, k: int) -> SelfIntersectionReport:
    """Self-intersection of a degree k hypersurface in P^n.

    In P^n, a hypersurface H of degree k has H^n = k^n (top intersection).
    For a divisor D of degree k, D·D in the intersection ring equals k²
    when n=2 (surface case).

    Args:
        n: Dimension of projective space
        k: Degree of the hypersurface/divisor

    Returns:
        SelfIntersectionReport
    """
    if n == 2:
        # In P², a degree k curve has self-intersection k² in the Chow ring
        # But geometrically, two generic curves of degree k meet in k² points
        self_int = k * k
        explanation = (f"In P², a degree {k} curve has self-intersection {k}² = {self_int}. "
                       f"Two generic degree {k} curves meet in {self_int} points.")
    else:
        self_int = k ** n
        explanation = (f"In P^{n}, a degree {k} hypersurface has H^{n} = {k}^{n} = {self_int}.")

    return SelfIntersectionReport(f"P^{n}", f"degree {k} hypersurface", self_int, explanation)


def self_intersection_line_P2() -> SelfIntersectionReport:
    """Self-intersection of a line in P².

    A line L in P² has L·L = 1 because two distinct lines meet at exactly
    one point (in projective plane).
    """
    return SelfIntersectionReport(
        "P²", "line L",
        1,
        "Two distinct lines in P² meet at exactly 1 point, so L·L = 1."
    )


def self_intersection_exceptional() -> SelfIntersectionReport:
    """Self-intersection of exceptional divisor on blow-up.

    When blowing up P² at a point, the exceptional divisor E has E·E = -1.
    This is the defining property of a (-1)-curve.
    """
    return SelfIntersectionReport(
        "Bl_p(P²)", "exceptional divisor E",
        -1,
        "The exceptional divisor E from blowing up a smooth point has E² = -1. "
        "(-1)-curves can be contracted to smooth points."
    )


def canonical_P2() -> CanonicalReport:
    """Canonical divisor of P².

    K_P² = -3H where H is a hyperplane (line) class.
    deg(K) = -3, making P² a Fano variety.
    """
    return CanonicalReport(
        "P²",
        K_degree=-3,
        K_squared=9,  # K·K = (-3H)·(-3H) = 9H² = 9
        explanation="K_P² = -3H (H = line class). Since -K is ample, P² is Fano.",
        is_fano=True
    )


def canonical_cubic_surface() -> CanonicalReport:
    """Canonical divisor of a smooth cubic surface in P³.

    A cubic surface is a del Pezzo surface of degree 3.
    It's the blow-up of P² at 6 points, so K² = 9 - 6 = 3.
    """
    return CanonicalReport(
        "smooth cubic surface in P³",
        K_degree=-1,  # K = -H|_S where H is hyperplane
        K_squared=3,  # del Pezzo degree 3
        explanation="Cubic surface = del Pezzo of degree 3 = Bl_6(P²). "
                    "K² = 9 - 6 = 3. Contains exactly 27 lines.",
        is_fano=True
    )


def del_pezzo_degree(n_blowups: int) -> CanonicalReport:
    """Canonical divisor of del Pezzo surface (blow-up of P² at n points).

    For 0 ≤ n ≤ 8 points in general position:
    - K² = 9 - n (the degree of the del Pezzo)
    - For n ≤ 8, the surface is Fano (-K ample)

    Args:
        n_blowups: Number of points blown up (0 to 8)
    """
    if n_blowups < 0 or n_blowups > 8:
        return CanonicalReport(
            f"Bl_{n_blowups}(P²)", 0, 0,
            f"Invalid: del Pezzo requires 0 ≤ n ≤ 8, got {n_blowups}",
            False, False
        )

    K_sq = 9 - n_blowups
    return CanonicalReport(
        f"Bl_{n_blowups}(P²)",
        K_degree=-1,
        K_squared=K_sq,
        explanation=f"Del Pezzo of degree {K_sq}. K² = 9 - {n_blowups} = {K_sq}. "
                    f"Picard rank = 1 + {n_blowups} = {1 + n_blowups}.",
        is_fano=True
    )


def noether_formula(c1_sq: int, c2: int) -> NoetherReport:
    """Check the Noether formula: c₁² + c₂ = 12χ(O_S).

    For a smooth complex surface S:
    - c₁² = K² (self-intersection of canonical)
    - c₂ = e(S) (topological Euler characteristic)
    - χ(O_S) = 1 - q + p_g (holomorphic Euler characteristic)

    Args:
        c1_sq: c₁² = K² for the surface
        c2: c₂ = topological Euler characteristic

    Returns:
        NoetherReport with consistency check
    """
    # Noether formula: c1² + c2 = 12χ
    lhs = c1_sq + c2
    chi = lhs / 12.0
    is_integer_chi = abs(chi - round(chi)) < 1e-10

    if is_integer_chi:
        explanation = f"Noether formula satisfied: {c1_sq} + {c2} = {lhs} = 12 × {int(round(chi))}"
        return NoetherReport(c1_sq, c2, round(chi), True, explanation)
    else:
        explanation = f"Noether formula: {c1_sq} + {c2} = {lhs}, but {lhs}/12 = {chi:.4f} (not integer χ)"
        return NoetherReport(c1_sq, c2, chi, False, explanation)


# ============================================================
# Classical Enumerative Results
# ============================================================

def lines_on_cubic_surface() -> EnumerativeReport:
    """The number of lines on a smooth cubic surface.

    Every smooth cubic surface in P³ contains exactly 27 lines.
    This is a famous result of Cayley and Salmon (1849).

    The configuration is related to the E6 root system.
    """
    return EnumerativeReport(
        "Lines on smooth cubic surface in P³",
        27,
        "W(E6) of order 51840",
        "Classical result (Cayley-Salmon 1849). Each line meets exactly 10 others. "
        "Configuration encoded by E6 root system."
    )


def bitangents_to_quartic() -> EnumerativeReport:
    """The number of bitangent lines to a smooth plane quartic.

    Every smooth quartic curve in P² has exactly 28 bitangent lines.
    Related to theta characteristics.
    """
    return EnumerativeReport(
        "Bitangents to smooth plane quartic",
        28,
        "Sp(6, F₂)",
        "Each bitangent touches the quartic at 2 points with multiplicity 2. "
        "Related to 28 odd theta characteristics."
    )


def conics_through_5_points() -> EnumerativeReport:
    """The number of conics through 5 general points in P².

    Through 5 general points in P², there passes exactly 1 conic.
    (A conic is determined by 5 parameters, so 5 points give a unique conic.)
    """
    return EnumerativeReport(
        "Conics through 5 general points in P²",
        1,
        "trivial",
        "A conic has 5 degrees of freedom (up to scaling). "
        "5 points in general position determine a unique conic."
    )


def lines_meeting_4_general_lines_P3() -> EnumerativeReport:
    """The number of lines meeting 4 general lines in P³.

    There are exactly 2 lines meeting 4 general lines in P³.
    """
    return EnumerativeReport(
        "Lines meeting 4 general lines in P³",
        2,
        "Z/2Z",
        "Classical result. The two lines are exchanged by a symmetry of the configuration."
    )


def plane_cubics_through_9_points() -> EnumerativeReport:
    """The number of plane cubic curves through 9 general points.

    Through 9 general points in P², there passes exactly 1 cubic curve.
    (A cubic has 10 coefficients, minus scaling = 9 parameters.)
    """
    return EnumerativeReport(
        "Cubics through 9 general points in P²",
        1,
        "trivial",
        "A plane cubic has 9 degrees of freedom. "
        "9 points in general position determine a unique cubic."
    )


def rational_curves_on_quintic_threefold(d: int = 1) -> EnumerativeReport:
    """The number of degree d rational curves on a general quintic threefold.

    For d=1: 2875 lines (classical result)
    For d=2: 609250 conics (Katz)
    For d=3: 317206375 twisted cubics

    These are foundational results in enumerative geometry and mirror symmetry.
    """
    counts = {
        1: 2875,
        2: 609250,
        3: 317206375,
    }
    if d not in counts:
        return EnumerativeReport(
            f"Degree {d} rational curves on quintic 3-fold",
            0,
            "unknown",
            f"Result for degree {d} not in database. Known: d=1,2,3.",
            False
        )

    return EnumerativeReport(
        f"Degree {d} rational curves on general quintic 3-fold in P⁴",
        counts[d],
        "various",
        f"Foundational in mirror symmetry. d=1: 2875 lines (classical). "
        f"d=2: 609250 (Katz). d=3: 317206375."
    )


# ============================================================
# Intersection Multiplicity
# ============================================================

def intersection_multiplicity_formula() -> str:
    """Return the definition of intersection multiplicity.

    At a point P where curves C and D meet:
        mult_P(C, D) = dim_k(O_P / (f, g))
    where f, g are local equations for C, D at P.
    """
    return ("Intersection multiplicity at P:\n"
            "  mult_P(C, D) = dim_k(O_P / (f, g))\n"
            "where f, g are local equations.\n"
            "Transverse intersection: mult = 1\n"
            "Tangent intersection: mult ≥ 2")


def compute_multiplicity_smooth_transverse() -> int:
    """Intersection multiplicity for smooth curves meeting transversely."""
    return 1


def compute_multiplicity_tangent(order: int) -> int:
    """Intersection multiplicity when curves are tangent with given contact order."""
    return order


# ============================================================
# Chow Ring Operations
# ============================================================

def chow_ring_Pn(n: int) -> str:
    """Describe the Chow ring A*(P^n).

    A*(P^n) = Z[h] / (h^{n+1})
    where h is the hyperplane class.
    """
    return (f"A*(P^{n}) = Z[h] / (h^{{{n+1}}})\n"
            f"where h = [hyperplane]. Degrees: h^k represents codimension k cycles.\n"
            f"Top intersection: h^{n} = [point] = 1.")


def segre_embedding_degree(m: int, n: int) -> int:
    """Degree of the Segre embedding P^m × P^n → P^{mn+m+n}.

    The degree is C(m+n, m) = (m+n)! / (m! n!).
    """
    return comb(m + n, m)
