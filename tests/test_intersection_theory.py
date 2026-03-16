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
