"""Tests for noethersolve.knot — Knot invariant verification."""

import pytest

from noethersolve.knot import (
    KnotMonitor,
    KnotReport,
    KnotDiagram,
    Crossing,
    unknot,
    trefoil,
    figure_eight_knot,
    apply_r1,
    apply_r1_remove,
    bracket_polynomial,
    jones_polynomial,
    _poly_equal,
)


# ─── Built-in knot construction ─────────────────────────────────────────────

class TestBuiltInKnots:
    def test_unknot_zero_crossings(self):
        k = unknot()
        assert k.n_crossings == 0
        assert k.writhe == 0

    def test_trefoil_three_crossings(self):
        k = trefoil()
        assert k.n_crossings == 3

    def test_trefoil_writhe_positive_three(self):
        k = trefoil()
        assert k.writhe == 3

    def test_trefoil_all_positive_crossings(self):
        k = trefoil()
        assert all(c.sign == +1 for c in k.crossings)

    def test_figure_eight_four_crossings(self):
        k = figure_eight_knot()
        assert k.n_crossings == 4

    def test_figure_eight_writhe_zero(self):
        """Figure-eight is amphichiral, writhe = 0."""
        k = figure_eight_knot()
        assert k.writhe == 0

    def test_figure_eight_alternating_signs(self):
        k = figure_eight_knot()
        signs = [c.sign for c in k.crossings]
        assert signs == [+1, -1, +1, -1]

    def test_gauss_code_length(self):
        """Gauss code has 2 * n_crossings entries."""
        k = trefoil()
        assert len(k.gauss_code) == 2 * k.n_crossings

    def test_unknot_gauss_code_empty(self):
        k = unknot()
        assert k.gauss_code == []


# ─── Polynomial values for known knots ──────────────────────────────────────

class TestPolynomials:
    def test_unknot_bracket(self):
        k = unknot()
        bp = bracket_polynomial(k)
        assert _poly_equal(bp, {0: 1})

    def test_unknot_jones(self):
        k = unknot()
        jp = jones_polynomial(k)
        assert _poly_equal(jp, {0: 1})

    def test_trefoil_bracket(self):
        """Bracket of trefoil: -A^16 + A^12 + A^4."""
        k = trefoil()
        bp = bracket_polynomial(k)
        assert bp[16] == -1
        assert bp[12] == 1
        assert bp[4] == 1

    def test_trefoil_jones(self):
        """Jones of trefoil (in A variable): A^{-4} + A^{-12} - A^{-16}."""
        k = trefoil()
        jp = jones_polynomial(k)
        assert jp[-4] == 1
        assert jp[-12] == 1
        assert jp[-16] == -1

    def test_figure_eight_jones_palindromic(self):
        """Figure-eight Jones is palindromic (amphichiral)."""
        k = figure_eight_knot()
        jp = jones_polynomial(k)
        # Check symmetry: coeff(A^n) == coeff(A^{-n})
        all_exps = list(jp.keys())
        for exp in all_exps:
            assert jp.get(exp, 0) == jp.get(-exp, 0), (
                f"Jones not palindromic at exponent {exp}"
            )

    def test_trefoil_jones_differs_from_unknot(self):
        """Jones polynomial distinguishes trefoil from unknot."""
        j_unknot = jones_polynomial(unknot())
        j_trefoil = jones_polynomial(trefoil())
        assert not _poly_equal(j_unknot, j_trefoil)

    def test_unknown_knot_raises(self):
        """Polynomial lookup raises for unknown knots with crossings."""
        k = KnotDiagram(
            crossings=[Crossing(0, 1, +1), Crossing(1, 0, -1)],
            gauss_code=[1, -2],
        )
        with pytest.raises(ValueError, match="not available"):
            bracket_polynomial(k)
        with pytest.raises(ValueError, match="not available"):
            jones_polynomial(k)


# ─── Reidemeister move R1 ───────────────────────────────────────────────────

class TestR1:
    def test_r1_adds_one_crossing(self):
        k = trefoil()
        after = apply_r1(k, sign=+1)
        assert after.n_crossings == k.n_crossings + 1

    def test_r1_positive_increases_writhe(self):
        k = trefoil()
        after = apply_r1(k, sign=+1)
        assert after.writhe == k.writhe + 1

    def test_r1_negative_decreases_writhe(self):
        k = trefoil()
        after = apply_r1(k, sign=-1)
        assert after.writhe == k.writhe - 1

    def test_r1_changes_bracket(self):
        """Bracket should change under R1 (multiplied by -A^{-3*sign})."""
        k = unknot()
        after = apply_r1(k, sign=+1)
        bp_before = bracket_polynomial(k)
        bp_after = bracket_polynomial(after)
        assert not _poly_equal(bp_before, bp_after)

    def test_r1_does_not_change_jones(self):
        """Jones polynomial must be invariant under R1."""
        k = trefoil()
        after = apply_r1(k, sign=+1)
        jp_before = jones_polynomial(k)
        jp_after = jones_polynomial(after)
        assert _poly_equal(jp_before, jp_after)

    def test_r1_does_not_change_jones_negative(self):
        """Jones invariant under R1 with negative sign too."""
        k = unknot()
        after = apply_r1(k, sign=-1)
        jp_before = jones_polynomial(k)
        jp_after = jones_polynomial(after)
        assert _poly_equal(jp_before, jp_after)

    def test_r1_invalid_sign(self):
        k = unknot()
        with pytest.raises(ValueError, match="sign must be"):
            apply_r1(k, sign=0)

    def test_r1_on_unknot_writhe(self):
        k = unknot()
        after = apply_r1(k, sign=+1)
        assert after.writhe == 1
        assert after.n_crossings == 1


# ─── R1 remove ──────────────────────────────────────────────────────────────

class TestR1Remove:
    def test_r1_add_then_remove(self):
        k = unknot()
        with_kink = apply_r1(k, sign=+1)
        assert with_kink.n_crossings == 1
        removed = apply_r1_remove(with_kink)
        assert removed.n_crossings == 0

    def test_r1_remove_no_kink_raises(self):
        k = trefoil()
        with pytest.raises(ValueError, match="No removable kink"):
            apply_r1_remove(k)


# ─── KnotMonitor.check_invariance ───────────────────────────────────────────

class TestCheckInvariance:
    def test_r1_writhe_expected_change(self):
        mon = KnotMonitor(trefoil())
        results = mon.check_invariance("R1")
        assert results["writhe"]["verdict"] == "EXPECTED_CHANGE"

    def test_r2_writhe_preserved(self):
        mon = KnotMonitor(trefoil())
        results = mon.check_invariance("R2")
        assert results["writhe"]["verdict"] == "PASS"

    def test_r3_writhe_preserved(self):
        mon = KnotMonitor(trefoil())
        results = mon.check_invariance("R3")
        assert results["writhe"]["verdict"] == "PASS"

    def test_r1_jones_preserved(self):
        mon = KnotMonitor(trefoil())
        results = mon.check_invariance("R1")
        assert results["jones_polynomial"]["verdict"] == "PASS"

    def test_r2_jones_preserved(self):
        mon = KnotMonitor(trefoil())
        results = mon.check_invariance("R2")
        assert results["jones_polynomial"]["verdict"] == "PASS"

    def test_r3_jones_preserved(self):
        mon = KnotMonitor(trefoil())
        results = mon.check_invariance("R3")
        assert results["jones_polynomial"]["verdict"] == "PASS"

    def test_r1_bracket_expected_change(self):
        mon = KnotMonitor(trefoil())
        results = mon.check_invariance("R1")
        assert results["bracket_polynomial"]["verdict"] == "EXPECTED_CHANGE"

    def test_r2_bracket_preserved(self):
        mon = KnotMonitor(trefoil())
        results = mon.check_invariance("R2")
        assert results["bracket_polynomial"]["verdict"] == "PASS"

    def test_r3_bracket_preserved(self):
        mon = KnotMonitor(trefoil())
        results = mon.check_invariance("R3")
        assert results["bracket_polynomial"]["verdict"] == "PASS"

    def test_invalid_move_raises(self):
        mon = KnotMonitor(unknot())
        with pytest.raises(ValueError, match="must be R1, R2, or R3"):
            mon.check_invariance("R4")

    def test_r1_crossing_number_changes(self):
        mon = KnotMonitor(unknot())
        results = mon.check_invariance("R1")
        assert results["crossing_number_upper"]["verdict"] == "EXPECTED_CHANGE"
        assert results["crossing_number_upper"]["after"] == 1

    def test_r3_crossing_number_preserved(self):
        mon = KnotMonitor(trefoil())
        results = mon.check_invariance("R3")
        assert results["crossing_number_upper"]["verdict"] == "PASS"


# ─── KnotMonitor.validate (full report) ────────────────────────────────────

class TestValidate:
    def test_trefoil_validates(self):
        mon = KnotMonitor(trefoil())
        report = mon.validate()
        assert isinstance(report, KnotReport)
        assert report.passed

    def test_unknot_validates(self):
        mon = KnotMonitor(unknot())
        report = mon.validate()
        assert report.passed

    def test_figure_eight_validates(self):
        mon = KnotMonitor(figure_eight_knot())
        report = mon.validate()
        assert report.passed

    def test_report_has_all_moves(self):
        mon = KnotMonitor(trefoil())
        report = mon.validate()
        keys = list(report.quantities.keys())
        for move in ("R1", "R2", "R3"):
            assert any(move in k for k in keys), f"Missing move {move}"

    def test_report_has_all_quantities(self):
        mon = KnotMonitor(trefoil())
        report = mon.validate()
        keys = list(report.quantities.keys())
        for q in ("writhe", "bracket_polynomial", "jones_polynomial",
                  "crossing_number_upper"):
            assert any(q in k for k in keys), f"Missing quantity {q}"

    def test_no_violations_for_valid_knots(self):
        for constructor in (unknot, trefoil, figure_eight_knot):
            mon = KnotMonitor(constructor())
            report = mon.validate()
            assert report.violations == [], (
                f"{constructor.__name__} has violations: {report.violations}"
            )


# ─── Report formatting ─────────────────────────────────────────────────────

class TestReportFormatting:
    def test_str_contains_knot_name(self):
        mon = KnotMonitor(trefoil())
        report = mon.validate()
        s = str(report)
        assert "trefoil" in s

    def test_str_contains_verdict(self):
        mon = KnotMonitor(trefoil())
        report = mon.validate()
        s = str(report)
        assert report.verdict in s

    def test_str_contains_crossing_count(self):
        mon = KnotMonitor(trefoil())
        report = mon.validate()
        s = str(report)
        assert "3 crossings" in s

    def test_str_contains_section_header(self):
        mon = KnotMonitor(trefoil())
        report = mon.validate()
        s = str(report)
        assert "Knot Invariant Validation" in s

    def test_passed_property(self):
        mon = KnotMonitor(trefoil())
        report = mon.validate()
        assert report.passed == (report.verdict == "PASS")


# ─── Error handling ─────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_empty_knot_diagram(self):
        """A bare KnotDiagram (no registration) still works for basic ops."""
        k = KnotDiagram(crossings=[], gauss_code=[])
        assert k.n_crossings == 0
        assert k.writhe == 0

    def test_custom_knot_monitor_skips_polys(self):
        """Monitor on unregistered knot with crossings skips polynomials."""
        k = KnotDiagram(
            crossings=[Crossing(0, 1, +1), Crossing(1, 0, -1)],
            gauss_code=[1, -2],
        )
        mon = KnotMonitor(k)
        results = mon.check_invariance("R2")
        # Should get SKIP for polynomials since knot is unregistered
        assert results["bracket_polynomial"]["verdict"] == "SKIP"
        assert results["jones_polynomial"]["verdict"] == "SKIP"
