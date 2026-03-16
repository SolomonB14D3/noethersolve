"""Tests for intersection theory module."""

import pytest
from noethersolve.intersection_theory import (
    bezout_intersection,
    genus_degree_formula,
    self_intersection_line_P2,
    self_intersection_exceptional,
    canonical_P2,
    canonical_cubic_surface,
    del_pezzo_degree,
    noether_formula,
    lines_on_cubic_surface,
    bitangents_to_quartic,
    conics_through_5_points,
    lines_meeting_4_general_lines_P3,
    rational_curves_on_quintic_threefold,
    chow_ring_Pn,
    segre_embedding_degree,
    # New imports
    adjunction_formula,
    adjunction_complete_intersection,
    blowup_K_squared,
    blowup_P2,
    blowup_transform_divisor,
    ruled_surface,
    hirzebruch_surface,
    toric_canonical,
    toric_Pn_canonical,
)


class TestBezout:
    """Test Bezout's theorem calculations."""

    def test_line_conic(self):
        """Line and conic intersect in 2 points."""
        report = bezout_intersection(1, 2)
        assert report.passed
        assert report.intersection_count == 2

    def test_two_conics(self):
        """Two conics intersect in 4 points."""
        report = bezout_intersection(2, 2)
        assert report.passed
        assert report.intersection_count == 4

    def test_cubic_quartic(self):
        """Cubic and quartic intersect in 12 points."""
        report = bezout_intersection(3, 4)
        assert report.passed
        assert report.intersection_count == 12

    def test_two_lines(self):
        """Two lines intersect in 1 point."""
        report = bezout_intersection(1, 1)
        assert report.passed
        assert report.intersection_count == 1

    def test_invalid_degree(self):
        """Invalid degree fails."""
        report = bezout_intersection(0, 2)
        assert not report.passed


class TestGenusDegree:
    """Test genus-degree formula."""

    def test_line_genus_0(self):
        """A line has genus 0."""
        report = genus_degree_formula(1)
        assert report.genus == 0

    def test_conic_genus_0(self):
        """A conic has genus 0."""
        report = genus_degree_formula(2)
        assert report.genus == 0

    def test_cubic_genus_1(self):
        """A smooth cubic has genus 1 (elliptic curve)."""
        report = genus_degree_formula(3)
        assert report.genus == 1
        assert "elliptic" in report.curve_type.lower()

    def test_quartic_genus_3(self):
        """A smooth quartic has genus 3."""
        report = genus_degree_formula(4)
        assert report.genus == 3

    def test_quintic_genus_6(self):
        """A smooth quintic has genus 6."""
        report = genus_degree_formula(5)
        # g = (5-1)(5-2)/2 = 4*3/2 = 6
        assert report.genus == 6


class TestSelfIntersection:
    """Test self-intersection calculations."""

    def test_line_P2(self):
        """A line in P² has self-intersection 1."""
        report = self_intersection_line_P2()
        assert report.self_intersection == 1

    def test_exceptional_divisor(self):
        """Exceptional divisor has self-intersection -1."""
        report = self_intersection_exceptional()
        assert report.self_intersection == -1


class TestCanonical:
    """Test canonical divisor calculations."""

    def test_P2_canonical(self):
        """K_P² has degree -3."""
        report = canonical_P2()
        assert report.K_degree == -3
        assert report.is_fano

    def test_cubic_surface(self):
        """Cubic surface has K² = 3."""
        report = canonical_cubic_surface()
        assert report.K_squared == 3
        assert report.is_fano

    def test_del_pezzo_6_blowups(self):
        """Blow-up of P² at 6 points has K² = 3."""
        report = del_pezzo_degree(6)
        assert report.K_squared == 3

    def test_del_pezzo_8_blowups(self):
        """Blow-up of P² at 8 points has K² = 1."""
        report = del_pezzo_degree(8)
        assert report.K_squared == 1

    def test_del_pezzo_invalid(self):
        """More than 8 blowups is invalid."""
        report = del_pezzo_degree(9)
        assert not report.passed


class TestNoether:
    """Test Noether formula."""

    def test_P2_noether(self):
        """P²: c₁² = 9, c₂ = 3, so χ = (9+3)/12 = 1."""
        report = noether_formula(9, 3)
        assert report.formula_check
        assert report.chi == 1

    def test_cubic_surface_noether(self):
        """Cubic surface: c₁² = 3, c₂ = 9, χ = 1."""
        report = noether_formula(3, 9)
        assert report.formula_check
        assert report.chi == 1


class TestEnumerative:
    """Test classical enumerative results."""

    def test_27_lines(self):
        """Cubic surface has exactly 27 lines."""
        report = lines_on_cubic_surface()
        assert report.count == 27
        assert "E6" in report.symmetry_group

    def test_28_bitangents(self):
        """Quartic has exactly 28 bitangents."""
        report = bitangents_to_quartic()
        assert report.count == 28

    def test_conics_through_5_points(self):
        """1 conic through 5 general points."""
        report = conics_through_5_points()
        assert report.count == 1

    def test_lines_meeting_4_lines(self):
        """2 lines meet 4 general lines in P³."""
        report = lines_meeting_4_general_lines_P3()
        assert report.count == 2

    def test_2875_lines_on_quintic(self):
        """2875 lines on general quintic threefold."""
        report = rational_curves_on_quintic_threefold(1)
        assert report.count == 2875

    def test_609250_conics_on_quintic(self):
        """609250 conics on general quintic threefold."""
        report = rational_curves_on_quintic_threefold(2)
        assert report.count == 609250


class TestChowRing:
    """Test Chow ring descriptions."""

    def test_chow_P2(self):
        """Chow ring of P² is Z[h]/(h³)."""
        desc = chow_ring_Pn(2)
        assert "h^{3}" in desc or "h^3" in desc or "h³" in desc

    def test_segre_degree_P1_P1(self):
        """Segre embedding P¹×P¹ → P³ has degree 2."""
        # C(1+1, 1) = 2
        assert segre_embedding_degree(1, 1) == 2

    def test_segre_degree_P2_P2(self):
        """Segre embedding P²×P² → P⁸ has degree 6."""
        # C(2+2, 2) = 6
        assert segre_embedding_degree(2, 2) == 6


# ============================================================
# New Tests: Adjunction Formula
# ============================================================

class TestAdjunction:
    """Test adjunction formula calculations."""

    def test_plane_cubic_genus_1(self):
        """Smooth cubic in P² has genus 1 via adjunction."""
        report = adjunction_formula(ambient_dim=2, divisor_degree=3)
        assert report.passed
        assert report.genus == 1

    def test_plane_quartic_genus_3(self):
        """Smooth quartic in P² has genus 3 via adjunction."""
        report = adjunction_formula(ambient_dim=2, divisor_degree=4)
        assert report.passed
        assert report.genus == 3

    def test_plane_quintic_genus_6(self):
        """Smooth quintic in P² has genus 6 via adjunction."""
        report = adjunction_formula(ambient_dim=2, divisor_degree=5)
        # g = (5-1)(5-2)/2 = 6
        assert report.genus == 6

    def test_cubic_in_P3_is_fano(self):
        """Smooth cubic surface in P³ is Fano (K_coeff = 3-4 = -1)."""
        report = adjunction_formula(ambient_dim=3, divisor_degree=3)
        assert report.passed
        # K_D = (3-4)H = -H, so d < n+1 means Fano
        assert "Fano" in report.explanation

    def test_quartic_in_P3_is_calabi_yau(self):
        """Quartic surface in P³ is Calabi-Yau (K = 0)."""
        report = adjunction_formula(ambient_dim=3, divisor_degree=4)
        # K_D = (4-4)H = 0
        assert "0" in report.K_divisor

    def test_quintic_threefold_calabi_yau(self):
        """Quintic threefold in P⁴ is Calabi-Yau."""
        report = adjunction_formula(ambient_dim=4, divisor_degree=5)
        # K_D = (5-5)H = 0
        assert "0" in report.K_divisor

    def test_complete_intersection_curve(self):
        """CI(2,2) in P³ is a genus 1 curve."""
        report = adjunction_complete_intersection([2, 2], 3)
        assert report.passed
        # d1*d2 = 4, sum - n - 1 = 4-3-1 = 0 (Calabi-Yau type)
        # Actually: g = 1 + 4*0/2 = 1
        assert report.genus == 1

    def test_complete_intersection_curve_2_3(self):
        """CI(2,3) in P³ is a canonical curve of genus 4."""
        report = adjunction_complete_intersection([2, 3], 3)
        # d1*d2 = 6, K_coeff = 2+3-3-1 = 1
        # g = 1 + 6*1/2 = 4
        assert report.passed
        assert report.genus == 4


# ============================================================
# New Tests: Blow-up Formulas
# ============================================================

class TestBlowup:
    """Test blow-up formula calculations."""

    def test_blowup_decreases_K_squared(self):
        """K² decreases by 1 per point blown up."""
        report = blowup_K_squared(original_K_sq=9, n_points=1)
        assert report.K_blowup_squared == 8
        assert report.n_points == 1

    def test_blowup_multiple_points(self):
        """K² = original - n for n points."""
        report = blowup_K_squared(original_K_sq=9, n_points=6)
        assert report.K_blowup_squared == 3

    def test_blowup_P2_once(self):
        """Bl_1(P²) has K² = 8."""
        report = blowup_P2(1)
        assert report.K_blowup_squared == 8
        assert report.K_original_squared == 9

    def test_blowup_P2_del_pezzo_6(self):
        """Bl_6(P²) is del Pezzo of degree 3."""
        report = blowup_P2(6)
        assert report.K_blowup_squared == 3
        assert "del Pezzo" in report.explanation

    def test_blowup_P2_degree_1_del_pezzo(self):
        """Bl_8(P²) is del Pezzo of degree 1."""
        report = blowup_P2(8)
        assert report.K_blowup_squared == 1

    def test_blowup_P2_rational_elliptic(self):
        """Bl_9(P²) has K² = 0, rational elliptic surface."""
        report = blowup_P2(9)
        assert report.K_blowup_squared == 0
        assert "elliptic" in report.explanation.lower()

    def test_blowup_picard_rank(self):
        """Picard rank increases by n."""
        report = blowup_P2(5)
        assert report.picard_rank == 6  # 1 + 5

    def test_blowup_exceptional_count(self):
        """Number of exceptional curves equals points blown up."""
        report = blowup_P2(7)
        assert report.exceptional_curves == 7

    def test_blowup_transform_divisor_through_point(self):
        """Divisor passing through point once: D̃² = D² - 1."""
        new_int, int_E, _ = blowup_transform_divisor(
            original_mult=1, curve_mult_at_point=1, intersection_original=4
        )
        assert new_int == 3  # 4 - 1² = 3
        assert int_E == 1

    def test_blowup_transform_double_point(self):
        """Divisor with double point: D̃² = D² - 4."""
        new_int, int_E, _ = blowup_transform_divisor(
            original_mult=2, curve_mult_at_point=2, intersection_original=9
        )
        assert new_int == 5  # 9 - 4 = 5
        assert int_E == 2


# ============================================================
# New Tests: Ruled Surfaces
# ============================================================

class TestRuledSurface:
    """Test ruled surface calculations."""

    def test_hirzebruch_F0(self):
        """F_0 = P¹ × P¹ has K² = 8."""
        report = hirzebruch_surface(0)
        assert report.K_squared == 8
        assert report.is_hirzebruch
        assert report.base_genus == 0

    def test_hirzebruch_F1(self):
        """F_1 = Bl_p(P²) has K² = 8."""
        report = hirzebruch_surface(1)
        assert report.K_squared == 8
        assert "not minimal" in report.explanation.lower() or "Bl" in report.explanation

    def test_hirzebruch_F2(self):
        """F_2 has a (-2)-curve."""
        report = hirzebruch_surface(2)
        assert report.K_squared == 8
        assert report.invariant_e == 2

    def test_ruled_over_genus_1(self):
        """Ruled surface over elliptic curve has K² = 0."""
        report = ruled_surface(base_genus=1, invariant_e=0)
        # K² = 8(1-1) = 0
        assert report.K_squared == 0

    def test_ruled_over_genus_2(self):
        """Ruled surface over genus 2 curve has K² = -8."""
        report = ruled_surface(base_genus=2, invariant_e=0)
        # K² = 8(1-2) = -8
        assert report.K_squared == -8

    def test_hirzebruch_invalid_negative(self):
        """F_n requires n ≥ 0."""
        report = hirzebruch_surface(-1)
        assert not report.passed


# ============================================================
# New Tests: Toric Varieties
# ============================================================

class TestToricVarieties:
    """Test toric variety canonical class calculations."""

    def test_toric_P1(self):
        """P¹ has K = -2H."""
        report = toric_canonical("P1")
        assert report.is_fano
        assert "-2" in report.K_description

    def test_toric_P2(self):
        """P² has K = -3H."""
        report = toric_canonical("P2")
        assert report.is_fano
        assert "-3" in report.K_description
        assert report.dimension == 2

    def test_toric_P3(self):
        """P³ has K = -4H."""
        report = toric_canonical("P3")
        assert report.is_fano
        assert "-4" in report.K_description
        assert report.dimension == 3

    def test_toric_Pn_canonical(self):
        """toric_Pn_canonical(n) agrees with toric_canonical('Pn')."""
        for n in [1, 2, 3, 4]:
            r1 = toric_Pn_canonical(n)
            r2 = toric_canonical(f"P{n}")
            assert r1.K_description == r2.K_description

    def test_toric_P1xP1(self):
        """P¹×P¹ has K² = 8."""
        report = toric_canonical("P1xP1")
        assert report.is_fano
        assert report.anti_canonical_degree == 8

    def test_toric_hirzebruch_F0(self):
        """F_0 via toric."""
        report = toric_canonical("F0")
        assert report.is_fano
        assert report.anti_canonical_degree == 8

    def test_toric_hirzebruch_F2(self):
        """F_2 is Fano."""
        report = toric_canonical("F2")
        assert report.is_fano

    def test_toric_hirzebruch_F3(self):
        """F_3 is NOT Fano (n > 2)."""
        report = toric_canonical("F3")
        assert not report.is_fano

    def test_toric_unknown(self):
        """Unknown variety returns error."""
        report = toric_canonical("unknown_variety")
        assert not report.passed


# ============================================================
# Tests for Report String Methods
# ============================================================

class TestReportStrings:
    """Test __str__ methods for new report types."""

    def test_adjunction_report_str(self):
        """Adjunction report is readable."""
        report = adjunction_formula(2, 3)
        s = str(report)
        assert "Adjunction" in s
        assert "genus" in s.lower()

    def test_blowup_report_str(self):
        """Blowup report is readable."""
        report = blowup_P2(3)
        s = str(report)
        assert "Blow-up" in s
        assert "K²" in s

    def test_ruled_surface_report_str(self):
        """Ruled surface report is readable."""
        report = hirzebruch_surface(2)
        s = str(report)
        assert "Ruled" in s or "Hirzebruch" in s

    def test_toric_report_str(self):
        """Toric canonical report is readable."""
        report = toric_canonical("P2")
        s = str(report)
        assert "Toric" in s or "P" in s
