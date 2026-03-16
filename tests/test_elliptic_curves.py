"""Tests for noethersolve.elliptic_curves module."""

import pytest
from noethersolve.elliptic_curves import (
    is_on_curve, point_add, point_double, point_negate,
    scalar_mult, point_order,
    discriminant, j_invariant, is_singular,
    hasse_bounds, count_points_naive, verify_hasse, find_points,
    is_valid_torsion_order,
    analyze_curve, analyze_point_arithmetic,
)


# Standard test curve: y² = x³ - x + 1 over F_23
E23 = {"a": -1, "b": 1, "p": 23}


class TestCoreArithmetic:
    """Test point arithmetic on elliptic curves."""

    def test_point_on_curve(self):
        """Verify point membership."""
        # (0, 1) should be on y² = x³ - x + 1: 1 = 0 - 0 + 1 ✓
        assert is_on_curve(E23, (0, 1))
        assert is_on_curve(E23, (0, 22))  # -1 mod 23 = 22
        # Point at infinity is always on curve
        assert is_on_curve(E23, None)
        # (1, 2) is NOT on curve: 4 ≠ 1 - 1 + 1 = 1
        assert not is_on_curve(E23, (1, 2))

    def test_point_addition_identity(self):
        """P + O = P and O + P = P."""
        P = (0, 1)
        assert point_add(E23, P, None) == P
        assert point_add(E23, None, P) == P
        assert point_add(E23, None, None) is None

    def test_point_addition_inverse(self):
        """P + (-P) = O."""
        P = (0, 1)
        neg_P = (0, 22)  # -1 mod 23
        result = point_add(E23, P, neg_P)
        assert result is None  # Point at infinity

    def test_point_addition_standard(self):
        """Standard point addition with two different points."""
        P = (0, 1)
        Q = (1, 1)
        # Verify both are on curve
        assert is_on_curve(E23, P)
        assert is_on_curve(E23, Q)
        # Compute P + Q
        R = point_add(E23, P, Q)
        # Result should be on curve
        assert is_on_curve(E23, R)
        assert R is not None

    def test_point_doubling(self):
        """Test doubling formula."""
        P = (0, 1)
        two_P = point_double(E23, P)
        assert two_P is not None
        assert is_on_curve(E23, two_P)
        # Should equal P + P
        assert two_P == point_add(E23, P, P)

    def test_point_doubling_at_infinity(self):
        """2 * O = O."""
        assert point_double(E23, None) is None

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        P = (0, 1)
        # 1P = P
        assert scalar_mult(E23, 1, P) == P
        # 2P should equal doubling
        assert scalar_mult(E23, 2, P) == point_double(E23, P)
        # 3P = 2P + P
        two_P = point_double(E23, P)
        three_P = point_add(E23, two_P, P)
        assert scalar_mult(E23, 3, P) == three_P
        # 0P = O
        assert scalar_mult(E23, 0, P) is None

    def test_scalar_multiplication_negative(self):
        """Test negative scalar multiplication."""
        P = (0, 1)
        neg_P = point_negate(E23, P)
        assert scalar_mult(E23, -1, P) == neg_P
        assert scalar_mult(E23, -2, P) == point_double(E23, neg_P)

    def test_point_order_generator(self):
        """Test finding the order of a point."""
        P = (0, 1)
        order = point_order(E23, P)
        assert order is not None
        # nP should be O
        assert scalar_mult(E23, order, P) is None
        # (n-1)P should not be O
        assert scalar_mult(E23, order - 1, P) is not None


class TestInvariants:
    """Test curve invariant calculations."""

    def test_discriminant_nonsingular(self):
        """Non-singular curve has non-zero discriminant."""
        disc = discriminant(-1, 1)
        assert disc != 0
        assert disc == -16 * (4 * (-1)**3 + 27 * 1**2)
        assert disc == -16 * (-4 + 27)
        assert disc == -368

    def test_discriminant_singular(self):
        """Singular curve has zero discriminant."""
        # 4a³ + 27b² = 0 when a = -3, b = 2 (since 4*(-27) + 27*4 = -108 + 108 = 0)
        disc = discriminant(-3, 2)
        assert disc == 0
        assert is_singular(-3, 2)

    def test_j_invariant_nonsingular(self):
        """j-invariant for non-singular curve."""
        j = j_invariant(-1, 1)
        assert j is not None
        # j = 1728 * 4a³ / (4a³ + 27b²) = 1728 * (-4) / 23 = -300.52...
        expected = 1728 * 4 * (-1)**3 / (4 * (-1)**3 + 27 * 1**2)
        assert abs(j - expected) < 0.01

    def test_j_invariant_singular(self):
        """j-invariant undefined for singular curve."""
        j = j_invariant(-3, 2)
        assert j is None

    def test_j_invariant_special_values(self):
        """Test j-invariant for curves with special values."""
        # j = 0 when a = 0 (supersingular in char 3)
        j_zero = j_invariant(0, 1)
        assert j_zero == 0
        # j = 1728 when b = 0
        j_1728 = j_invariant(1, 0)
        assert j_1728 == 1728


class TestPointCounting:
    """Test point counting and Hasse bounds."""

    def test_hasse_bounds(self):
        """Hasse bounds are correct."""
        lo, hi = hasse_bounds(23)
        # p + 1 = 24, 2*sqrt(23) ≈ 9.6
        assert lo > 0
        assert lo < 24
        assert hi > 24

    def test_count_points_includes_infinity(self):
        """Point count includes point at infinity."""
        count = count_points_naive(E23)
        # Count includes O, so should be >= 1
        assert count >= 1

    def test_verify_hasse(self):
        """Hasse's theorem is satisfied."""
        ok, count, lo, hi = verify_hasse(E23)
        assert ok
        assert lo <= count <= hi

    def test_hasse_many_curves(self):
        """Test Hasse bounds on multiple curves."""
        primes = [7, 11, 13, 17, 19, 23, 29, 31]
        for p in primes:
            E = {"a": 1, "b": 1, "p": p}
            if not is_singular(1, 1):
                ok, count, lo, hi = verify_hasse(E)
                assert ok, f"Hasse violated for p={p}: count={count}, bounds=[{lo},{hi}]"

    def test_find_points(self):
        """Find all points on a curve."""
        pts = find_points(E23)
        # Should include point at infinity
        assert None in pts
        # All points should be on curve
        for P in pts:
            assert is_on_curve(E23, P)
        # Count should match naive count
        count = count_points_naive(E23)
        assert len(pts) == count


class TestTorsion:
    """Test torsion structure validation."""

    def test_valid_cyclic_orders(self):
        """Valid cyclic torsion orders per Mazur."""
        valid = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}
        for n in valid:
            assert is_valid_torsion_order(n), f"{n} should be valid"

    def test_invalid_orders(self):
        """Invalid torsion orders per Mazur."""
        invalid = {11, 13, 14, 15, 17, 18, 19, 20}
        for n in invalid:
            assert not is_valid_torsion_order(n), f"{n} should be invalid"

    def test_product_orders(self):
        """Valid product structure orders."""
        # Z/2 × Z/2 has order 4
        # Z/2 × Z/4 has order 8
        # Z/2 × Z/6 has order 12
        # Z/2 × Z/8 has order 16
        assert is_valid_torsion_order(4)
        assert is_valid_torsion_order(8)
        assert is_valid_torsion_order(12)
        assert is_valid_torsion_order(16)


class TestReports:
    """Test report generation."""

    def test_curve_report(self):
        """Curve analysis report."""
        report = analyze_curve(-1, 1, 23)
        assert report.passed
        assert not report.is_singular
        assert report.hasse_satisfied
        assert report.point_count > 0
        assert "Valid curve" in str(report)

    def test_singular_curve_report(self):
        """Singular curve detected."""
        report = analyze_curve(-3, 2, 23)
        assert not report.passed
        assert report.is_singular
        assert "SINGULAR" in str(report)

    def test_arithmetic_report(self):
        """Point arithmetic report."""
        report = analyze_point_arithmetic(-1, 1, 23, (0, 1), (1, 1))
        assert report.passed
        assert report.P_on_curve
        assert report.Q_on_curve
        assert report.result_on_curve
        assert "Arithmetic verified" in str(report)

    def test_arithmetic_invalid_point(self):
        """Invalid point detected in arithmetic."""
        report = analyze_point_arithmetic(-1, 1, 23, (1, 2), (0, 1))
        assert not report.passed
        assert not report.P_on_curve
        assert "INPUT ERROR" in str(report)


class TestGroupStructure:
    """Test group structure properties."""

    def test_associativity(self):
        """(P + Q) + R = P + (Q + R)."""
        # Use a curve without 2-torsion edge cases
        E = {"a": 1, "b": 1, "p": 7}
        pts = find_points(E, limit=5)
        P, Q, R = pts[1], pts[2], pts[3]
        left = point_add(E, point_add(E, P, Q), R)
        right = point_add(E, P, point_add(E, Q, R))
        assert left == right

    def test_commutativity(self):
        """P + Q = Q + P."""
        P = (0, 1)
        Q = (1, 1)
        assert point_add(E23, P, Q) == point_add(E23, Q, P)

    def test_closure(self):
        """P + Q is always on the curve."""
        # Use a curve with cleaner structure
        E = {"a": 1, "b": 0, "p": 7}  # y² = x³ + x over F_7, order 8
        pts = find_points(E, limit=10)
        for i, P in enumerate(pts):
            for Q in pts[i:]:
                R = point_add(E, P, Q)
                assert is_on_curve(E, R)

    def test_group_order_divides_curve_order(self):
        """Order of any point divides #E(F_p) on a curve with cyclic group."""
        # y² = x³ + x over F_7 has order 8 = 2³ (cyclic-ish structure)
        E = {"a": 1, "b": 0, "p": 7}
        curve_order = count_points_naive(E)
        pts = find_points(E, limit=10)
        for P in pts[1:5]:  # Skip O, test a few
            order = point_order(E, P)
            assert order is not None
            assert curve_order % order == 0, f"Order {order} doesn't divide {curve_order}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_prime(self):
        """Curve over F_2 and F_3."""
        E2 = {"a": 1, "b": 1, "p": 2}
        # y² + y = x³ + x + 1 over F_2 is different, but y² = x³ + x + 1 mod 2
        count = count_points_naive(E2)
        assert count >= 1  # At least O

    def test_large_scalar(self):
        """Large scalar multiplication."""
        P = (0, 1)
        # Multiply by a large number
        n = 1000
        result = scalar_mult(E23, n, P)
        assert result is None or is_on_curve(E23, result)

    def test_identity_negate(self):
        """Negation of O is O."""
        assert point_negate(E23, None) is None

    def test_double_negate(self):
        """-(-P) = P."""
        P = (0, 1)
        assert point_negate(E23, point_negate(E23, P)) == P
