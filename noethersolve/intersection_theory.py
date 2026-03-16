"""Intersection theory calculations for algebraic geometry.

Verified computational tools for:
- Bezout theorem intersection counts
- Self-intersection numbers on surfaces
- Genus-degree formula for plane curves
- Canonical divisor calculations
- Chern number relations (Noether formula)
- Classical enumerative results (27 lines, 28 bitangents)
- Adjunction formula for divisors
- Blow-up formulas for canonical class
- Ruled surface intersection theory
- Toric variety canonical classes

CRITICAL FACTS LLMs GET WRONG:
1. Blow-up formula: K² DECREASES by 1 per point blown up (K²_X̃ = K²_X - 1)
2. Exceptional divisor: E² = -1 ALWAYS for smooth blowups
3. Adjunction: genus formula requires SMOOTH curve (singularities need correction)
4. Ruled surfaces: e = -deg(E) can be NEGATIVE for non-trivial bundles
5. Toric varieties: K is ALWAYS anti-effective (no effective canonical)
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


@dataclass
class AdjunctionReport:
    """Result of adjunction formula calculation."""
    ambient: str
    divisor: str
    K_ambient: str
    K_divisor: str
    genus: Optional[int]
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        genus_str = f", genus = {self.genus}" if self.genus is not None else ""
        return (f"Adjunction formula on {self.ambient}:\n"
                f"  K_D = (K_X + D)|_D\n"
                f"  K_{self.divisor} = {self.K_divisor}{genus_str}\n"
                f"  {self.explanation}")


@dataclass
class BlowupReport:
    """Result of blow-up formula calculation."""
    original: str
    n_points: int
    K_original_squared: int
    K_blowup_squared: int
    picard_rank: int
    exceptional_curves: int
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        return (f"Blow-up of {self.original} at {self.n_points} point(s):\n"
                f"  K²_original = {self.K_original_squared}\n"
                f"  K²_blowup = {self.K_blowup_squared} = {self.K_original_squared} - {self.n_points}\n"
                f"  Picard rank: 1 + {self.n_points} = {self.picard_rank}\n"
                f"  Exceptional (-1)-curves: {self.exceptional_curves}\n"
                f"  {self.explanation}")


@dataclass
class RuledSurfaceReport:
    """Result of ruled surface calculation."""
    base_genus: int
    invariant_e: int
    K_squared: int
    is_hirzebruch: bool
    hirzebruch_n: Optional[int]
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        type_str = f"Hirzebruch surface F_{self.hirzebruch_n}" if self.is_hirzebruch else f"ruled over genus {self.base_genus} curve"
        return (f"Ruled surface ({type_str}):\n"
                f"  Base genus g = {self.base_genus}\n"
                f"  Invariant e = {self.invariant_e}\n"
                f"  K² = 8(1 - g) = {self.K_squared}\n"
                f"  {self.explanation}")


@dataclass
class ToricCanonicalReport:
    """Result of toric variety canonical class calculation."""
    variety: str
    dimension: int
    K_description: str
    is_fano: bool
    anti_canonical_degree: Optional[int]
    explanation: str
    passed: bool = True

    def __str__(self) -> str:
        fano_str = "Fano (-K ample)" if self.is_fano else "not Fano"
        deg_str = f", (-K)^n = {self.anti_canonical_degree}" if self.anti_canonical_degree else ""
        return (f"Toric variety: {self.variety} (dim {self.dimension})\n"
                f"  K = {self.K_description}\n"
                f"  {fano_str}{deg_str}\n"
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
# Adjunction Formula
# ============================================================

def adjunction_formula(
    ambient_dim: int,
    divisor_degree: int,
    ambient_type: str = "Pn"
) -> AdjunctionReport:
    """Apply the adjunction formula to compute K_D for a divisor D in X.

    For a smooth divisor D in smooth variety X:
        K_D = (K_X + D)|_D

    For a smooth degree d curve in P²:
        K_C = (K_P² + C)|_C = (-3H + dH)|_C = (d-3)H|_C
        deg K_C = d(d-3), genus g = (d-1)(d-2)/2

    Args:
        ambient_dim: Dimension of ambient variety (e.g., 2 for P²)
        divisor_degree: Degree of the divisor
        ambient_type: "Pn" for projective space, "complete_intersection" for CI

    Returns:
        AdjunctionReport with K_D calculation
    """
    if ambient_type == "Pn" and ambient_dim == 2:
        # Plane curve case
        d = divisor_degree
        K_ambient = f"-3H (K_P² = -(n+1)H = -3H)"
        deg_K_C = d * (d - 3)
        K_divisor = f"(d-3)H|_C = {d-3}H|_C, degree {deg_K_C}"
        genus = (d - 1) * (d - 2) // 2
        explanation = (f"For smooth degree {d} curve: K_C = (K_P² + C)|_C = (-3H + {d}H)|_C. "
                       f"By Riemann-Roch, 2g - 2 = deg K_C = {deg_K_C}, giving g = {genus}.")
        return AdjunctionReport(
            f"P²", f"degree {d} curve", K_ambient, K_divisor, genus, explanation
        )
    elif ambient_type == "Pn":
        n = ambient_dim
        d = divisor_degree
        K_ambient = f"-{n+1}H"
        K_divisor = f"({d} - {n+1})H|_D = {d - n - 1}H|_D"
        genus = None  # Higher dimensional divisors don't have genus
        explanation = (f"Smooth hypersurface of degree {d} in P^{n}: "
                       f"K_D = (K_P^{n} + D)|_D = ({d - n - 1})H|_D. "
                       f"Fano if d < {n+1}, general type if d > {n+1}.")
        return AdjunctionReport(
            f"P^{n}", f"degree {d} hypersurface", K_ambient, K_divisor, genus, explanation
        )
    else:
        return AdjunctionReport(
            ambient_type, f"degree {divisor_degree}", "unknown", "unknown", None,
            f"Ambient type '{ambient_type}' not supported.", False
        )


def adjunction_complete_intersection(degrees: list[int], ambient_dim: int) -> AdjunctionReport:
    """Adjunction for a complete intersection in P^n.

    For V = V(f_1, ..., f_k) ⊂ P^n with deg(f_i) = d_i:
        K_V = (Σd_i - n - 1)H|_V

    If dim(V) = 1 (curve), genus by adjunction:
        g = 1 + (d₁...d_k)/2 * (Σd_i - n - 1)

    Args:
        degrees: List of hypersurface degrees [d_1, ..., d_k]
        ambient_dim: n for P^n
    """
    n = ambient_dim
    k = len(degrees)
    if k > n:
        return AdjunctionReport(
            f"P^{n}", f"CI({degrees})", f"-{n+1}H", "undefined",
            None, f"Error: {k} equations in P^{n} gives empty or excess intersection.", False
        )

    prod_deg = 1
    for d in degrees:
        prod_deg *= d
    sum_deg = sum(degrees)

    K_coeff = sum_deg - n - 1
    dim_V = n - k

    K_divisor = f"({K_coeff})H|_V" if K_coeff != 0 else "0 (Calabi-Yau)"

    if dim_V == 1:
        # Curve case: compute genus
        # g = 1 + d₁...d_k/2 * (Σd_i - n - 1)
        genus_from_adj = 1 + (prod_deg * K_coeff) // 2
        explanation = (f"CI of degrees {degrees} in P^{n} is a curve. "
                       f"deg(V) = {prod_deg}. K_V = {K_coeff}H. "
                       f"Genus g = 1 + {prod_deg}·{K_coeff}/2 = {genus_from_adj}.")
        return AdjunctionReport(
            f"P^{n}", f"CI{tuple(degrees)} curve", f"-{n+1}H", K_divisor,
            genus_from_adj, explanation
        )
    else:
        explanation = (f"CI of degrees {degrees} in P^{n} has dim {dim_V}. "
                       f"deg(V) = {prod_deg}. K_V = ({K_coeff})H|_V. "
                       f"{'Fano' if K_coeff < 0 else 'Calabi-Yau' if K_coeff == 0 else 'General type'}.")
        return AdjunctionReport(
            f"P^{n}", f"CI{tuple(degrees)}", f"-{n+1}H", K_divisor,
            None, explanation
        )


# ============================================================
# Blow-up Formulas
# ============================================================

def blowup_K_squared(original_K_sq: int, n_points: int) -> BlowupReport:
    """Compute K² after blowing up n points.

    CRITICAL FORMULA:
        K²_X̃ = K²_X - n

    Each blow-up:
    - Decreases K² by 1
    - Increases Picard rank by 1
    - Adds one (-1)-curve (exceptional divisor E with E² = -1)

    Args:
        original_K_sq: K² of the original surface
        n_points: Number of points being blown up (assumed general position)

    Returns:
        BlowupReport with K² computation
    """
    new_K_sq = original_K_sq - n_points
    picard_rank = 1 + n_points

    if new_K_sq > 0:
        type_str = "general type"
    elif new_K_sq == 0:
        type_str = "log Calabi-Yau (K² = 0)"
    else:
        type_str = f"Fano (K² < 0, degree {-new_K_sq} del Pezzo if applicable)"

    explanation = (f"K_X̃ = π*K_X + Σ E_i. Since E_i² = -1 and E_i·π*K_X = 0, "
                   f"K²_X̃ = K²_X + ΣE_i² = {original_K_sq} + {n_points}(-1) = {new_K_sq}. "
                   f"Surface is {type_str}.")

    return BlowupReport(
        original="surface", n_points=n_points,
        K_original_squared=original_K_sq, K_blowup_squared=new_K_sq,
        picard_rank=picard_rank, exceptional_curves=n_points,
        explanation=explanation
    )


def blowup_P2(n_points: int) -> BlowupReport:
    """Blow up P² at n points in general position.

    P² has K² = 9. After blowing up n ≤ 8 points:
    - K² = 9 - n
    - Picard rank = 1 + n
    - For n ≤ 8: del Pezzo surface of degree 9-n

    For n = 9: K² = 0 (rational elliptic surface)
    For n > 9: K² < 0 (big anti-canonical, but not Fano)

    Args:
        n_points: Number of points to blow up
    """
    K_sq_P2 = 9
    new_K_sq = K_sq_P2 - n_points
    picard_rank = 1 + n_points

    if n_points <= 8:
        explanation = (f"Bl_{n_points}(P²) is a del Pezzo surface of degree {new_K_sq}. "
                       f"K² = 9 - {n_points} = {new_K_sq}. -K is ample (Fano).")
    elif n_points == 9:
        explanation = ("Bl_9(P²) is a rational elliptic surface. K² = 0. "
                       "The anti-canonical system gives an elliptic fibration.")
    else:
        explanation = (f"Bl_{n_points}(P²): K² = {new_K_sq} < 0. "
                       f"-K is big but not ample (not del Pezzo).")

    return BlowupReport(
        original="P²", n_points=n_points,
        K_original_squared=K_sq_P2, K_blowup_squared=new_K_sq,
        picard_rank=picard_rank, exceptional_curves=n_points,
        explanation=explanation
    )


def blowup_transform_divisor(
    original_mult: int,
    curve_mult_at_point: int,
    intersection_original: int
) -> tuple[int, int, str]:
    """Compute how a divisor transforms under blow-up.

    For D on X, if D passes through p with multiplicity m:
        D̃ = π*D - m·E
        D̃² = D² - m²
        D̃·Ẽ = m

    Args:
        original_mult: Multiplicity of D at the blown-up point
        curve_mult_at_point: Not used (legacy)
        intersection_original: D² on original surface

    Returns:
        (new_self_intersection, intersection_with_E, explanation)
    """
    m = original_mult
    new_self_int = intersection_original - m * m
    int_with_E = m

    explanation = (f"D̃ = π*D - {m}E. "
                   f"D̃² = D² - m² = {intersection_original} - {m}² = {new_self_int}. "
                   f"D̃·E = {int_with_E}.")

    return new_self_int, int_with_E, explanation


# ============================================================
# Ruled Surface Theory
# ============================================================

def ruled_surface(base_genus: int, invariant_e: int = 0) -> RuledSurfaceReport:
    """Compute intersection theory on a ruled surface.

    A ruled surface S = P(E) over curve C of genus g:
    - Pic(S) = Zh ⊕ Zf (h = section class, f = fiber class)
    - f² = 0, h·f = 1, h² = -e where e = -deg(E) + 2(1-g)
    - K_S = -2h + (2g - 2 + e)f
    - K²_S = 8(1 - g) (always, regardless of e)

    Hirzebruch surfaces F_n = P(O ⊕ O(-n)) over P¹ have g=0, e=n.

    Args:
        base_genus: Genus g of the base curve C
        invariant_e: The invariant e = -h² (related to splitting type)
    """
    g = base_genus
    e = invariant_e

    K_squared = 8 * (1 - g)

    is_hirzebruch = (g == 0)
    hirz_n = e if is_hirzebruch else None

    if is_hirzebruch:
        explanation = (f"Hirzebruch surface F_{e} = P(O ⊕ O(-{e})) over P¹. "
                       f"h² = -{e}, f² = 0, h·f = 1. K = -2h + ({e} - 2)f. "
                       f"K² = 8. {'Minimal' if e != 1 else 'Bl_p(P²) (not minimal)'}.")
    else:
        explanation = (f"Ruled surface P(E) over genus {g} curve. "
                       f"e = {e}. K = -2h + (2·{g} - 2 + {e})f = -2h + {2*g - 2 + e}f. "
                       f"K² = 8(1 - {g}) = {K_squared}.")

    return RuledSurfaceReport(
        base_genus=g, invariant_e=e, K_squared=K_squared,
        is_hirzebruch=is_hirzebruch, hirzebruch_n=hirz_n,
        explanation=explanation
    )


def hirzebruch_surface(n: int) -> RuledSurfaceReport:
    """The Hirzebruch surface F_n = P(O ⊕ O(-n)) over P¹.

    F_0 = P¹ × P¹
    F_1 = Bl_p(P²) (P² blown up at one point)
    F_2 has a (-2)-curve (the negative section)

    Args:
        n: The invariant (n ≥ 0)
    """
    if n < 0:
        return RuledSurfaceReport(
            base_genus=0, invariant_e=n, K_squared=8,
            is_hirzebruch=True, hirzebruch_n=n,
            explanation=f"Error: n must be ≥ 0, got {n}",
            passed=False
        )

    explanation_parts = [f"F_{n} = P(O ⊕ O(-{n})) over P¹."]
    if n == 0:
        explanation_parts.append("F_0 ≅ P¹ × P¹.")
    elif n == 1:
        explanation_parts.append("F_1 ≅ Bl_p(P²), not minimal.")
    elif n >= 2:
        explanation_parts.append(f"Has a section with self-intersection -{n} (negative section).")

    explanation_parts.append(f"K² = 8. Picard rank 2.")

    return RuledSurfaceReport(
        base_genus=0, invariant_e=n, K_squared=8,
        is_hirzebruch=True, hirzebruch_n=n,
        explanation=" ".join(explanation_parts)
    )


# ============================================================
# Toric Variety Canonical Classes
# ============================================================

def toric_canonical(variety_name: str) -> ToricCanonicalReport:
    """Compute canonical class for common toric varieties.

    For a complete smooth toric variety X with fan Σ:
        K_X = -Σ_ρ D_ρ (sum over rays)

    Each ray ρ corresponds to a torus-invariant divisor D_ρ.
    K is always anti-effective (never effective) on complete toric varieties.

    Args:
        variety_name: One of "Pn", "P1xP1", "Fn", "weighted_Pn", "hirzebruch_n"
    """
    name_lower = variety_name.lower()

    if name_lower.startswith("p") and name_lower[1:].isdigit():
        # Projective space P^n
        n = int(name_lower[1:])
        K_desc = f"-{n+1}H (H = hyperplane class)"
        antiK_deg = (n + 1) ** n  # Actually (-K)^n for Fano
        explanation = (f"P^{n} is toric with {n+1} rays. K = -Σ D_ρ = -{n+1}H. "
                       f"Fano variety of index {n+1}. (-K)^{n} = {(n+1)**n} (volume).")
        return ToricCanonicalReport(
            variety=f"P^{n}", dimension=n, K_description=K_desc,
            is_fano=True, anti_canonical_degree=(n+1)**n if n <= 3 else None,
            explanation=explanation
        )

    elif name_lower == "p1xp1":
        # P¹ × P¹ = F_0
        explanation = ("P¹×P¹ is toric with 4 rays. K = -2(h₁ + h₂) where h_i are the pullbacks. "
                       "K² = 8. Fano (del Pezzo of degree 8). Same as Hirzebruch F_0.")
        return ToricCanonicalReport(
            variety="P¹×P¹", dimension=2, K_description="-2(h₁ + h₂)",
            is_fano=True, anti_canonical_degree=8,
            explanation=explanation
        )

    elif name_lower.startswith("f") and name_lower[1:].isdigit():
        # Hirzebruch surface F_n
        n = int(name_lower[1:])
        explanation = (f"Hirzebruch F_{n} is toric with 4 rays. "
                       f"K = -2h + ({n} - 2)f (using h² = -{n}, f² = 0, h·f = 1). "
                       f"K² = 8. Fano iff n ≤ 2.")
        return ToricCanonicalReport(
            variety=f"F_{n}", dimension=2, K_description=f"-2h + ({n}-2)f",
            is_fano=(n <= 2), anti_canonical_degree=8,
            explanation=explanation
        )

    elif name_lower.startswith("weighted_p"):
        # Weighted projective space
        explanation = ("Weighted projective space P(w₀,...,wₙ) is toric. "
                       "K = -(Σwᵢ)H where H is the fundamental class. "
                       "Fano iff Σwᵢ > lcm(wᵢ). May be singular (orbifold points).")
        return ToricCanonicalReport(
            variety="P(w₀,...,wₙ)", dimension=0, K_description="-(Σwᵢ)H",
            is_fano=True, anti_canonical_degree=None,
            explanation=explanation
        )

    elif name_lower.startswith("hirzebruch_"):
        n = int(name_lower.split("_")[1])
        return toric_canonical(f"F{n}")

    else:
        explanation = (f"Unknown toric variety '{variety_name}'. "
                       f"Try: P1, P2, P3, P1xP1, F0, F1, F2, weighted_P.")
        return ToricCanonicalReport(
            variety=variety_name, dimension=0, K_description="unknown",
            is_fano=False, anti_canonical_degree=None,
            explanation=explanation, passed=False
        )


def toric_Pn_canonical(n: int) -> ToricCanonicalReport:
    """Canonical class of P^n as a toric variety.

    P^n has n+1 torus-invariant divisors (coordinate hyperplanes).
    K = -D₀ - D₁ - ... - D_n = -(n+1)H.
    """
    return toric_canonical(f"P{n}")


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
