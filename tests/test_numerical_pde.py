"""Tests for noethersolve.numerical_pde module.

Tests cover:
- CFL condition analysis
- Von Neumann stability analysis
- Scheme information
- Lax equivalence theorem
- Accuracy analysis
- Common errors
"""

import math
import pytest

from noethersolve.numerical_pde import (
    check_cfl,
    cfl_hyperbolic,
    cfl_parabolic,
    max_timestep,
    von_neumann_analysis,
    get_scheme_info,
    list_schemes,
    check_lax_equivalence,
    analyze_accuracy,
    check_common_error,
)


# ─── CFL Condition Tests ──────────────────────────────────────────────────

class TestCFLHyperbolic:
    """Tests for hyperbolic CFL number computation."""

    def test_cfl_formula(self):
        """CFL = c × Δt / Δx."""
        cfl = cfl_hyperbolic(c=1.0, dt=0.01, dx=0.1)
        assert cfl == pytest.approx(0.1, rel=1e-6)

    def test_cfl_at_limit(self):
        """CFL = 1 at stability limit."""
        cfl = cfl_hyperbolic(c=1.0, dt=0.1, dx=0.1)
        assert cfl == pytest.approx(1.0, rel=1e-6)

    def test_cfl_above_limit(self):
        """CFL > 1 for unstable parameters."""
        cfl = cfl_hyperbolic(c=2.0, dt=0.1, dx=0.1)
        assert cfl == pytest.approx(2.0, rel=1e-6)

    def test_negative_speed(self):
        """Absolute value of speed used."""
        cfl = cfl_hyperbolic(c=-1.0, dt=0.1, dx=0.1)
        assert cfl == pytest.approx(1.0, rel=1e-6)


class TestCFLParabolic:
    """Tests for parabolic CFL number computation."""

    def test_cfl_formula(self):
        """CFL = D × Δt / Δx²."""
        cfl = cfl_parabolic(D=0.1, dt=0.01, dx=0.1)
        assert cfl == pytest.approx(0.1, rel=1e-6)

    def test_cfl_at_ftcs_limit(self):
        """CFL = 0.5 at FTCS stability limit."""
        # D × Δt / Δx² = 0.5  =>  Δt = 0.5 × Δx² / D
        D, dx = 1.0, 0.1
        dt = 0.5 * dx * dx / D
        cfl = cfl_parabolic(D=D, dt=dt, dx=dx)
        assert cfl == pytest.approx(0.5, rel=1e-6)


class TestCheckCFL:
    """Tests for check_cfl function."""

    def test_upwind_stable(self):
        """Upwind stable for CFL ≤ 1."""
        r = check_cfl("upwind", cfl_number=0.8)
        assert r.is_stable is True
        assert r.cfl_limit == pytest.approx(1.0)
        assert r.stability_type == "conditionally stable"

    def test_upwind_unstable(self):
        """Upwind unstable for CFL > 1."""
        r = check_cfl("upwind", cfl_number=1.2)
        assert r.is_stable is False

    def test_ftcs_stable(self):
        """FTCS stable for CFL ≤ 0.5."""
        r = check_cfl("ftcs", cfl_number=0.4)
        assert r.is_stable is True
        assert r.cfl_limit == pytest.approx(0.5)

    def test_ftcs_unstable(self):
        """FTCS unstable for CFL > 0.5."""
        r = check_cfl("ftcs", cfl_number=0.6)
        assert r.is_stable is False

    def test_crank_nicolson_unconditional(self):
        """Crank-Nicolson unconditionally stable."""
        r = check_cfl("crank_nicolson", cfl_number=100.0)
        assert r.is_stable is True
        assert r.stability_type == "unconditionally stable"
        assert r.cfl_limit == float('inf')

    def test_leapfrog_parabolic_never_stable(self):
        """Leapfrog for parabolic is NEVER stable."""
        r = check_cfl("leapfrog_parabolic", cfl_number=0.001)
        assert r.is_stable is False
        assert r.stability_type == "unconditionally unstable"
        assert r.cfl_limit == 0.0

    def test_unknown_scheme_error(self):
        """Unknown scheme raises error."""
        with pytest.raises(ValueError):
            check_cfl("nonexistent_scheme", cfl_number=0.5)


class TestMaxTimestep:
    """Tests for max_timestep function."""

    def test_hyperbolic_max_dt(self):
        """Max Δt for hyperbolic scheme."""
        dt = max_timestep("upwind", c_or_D=1.0, dx=0.1, pde_type="hyperbolic", safety_factor=1.0)
        # CFL = c*dt/dx ≤ 1  =>  dt ≤ dx/c = 0.1
        assert dt == pytest.approx(0.1, rel=1e-6)

    def test_parabolic_max_dt(self):
        """Max Δt for parabolic scheme."""
        dt = max_timestep("ftcs", c_or_D=1.0, dx=0.1, pde_type="parabolic", safety_factor=1.0)
        # CFL = D*dt/dx² ≤ 0.5  =>  dt ≤ 0.5*dx²/D = 0.005
        assert dt == pytest.approx(0.005, rel=1e-6)

    def test_safety_factor(self):
        """Safety factor reduces max Δt."""
        dt_full = max_timestep("upwind", c_or_D=1.0, dx=0.1, pde_type="hyperbolic", safety_factor=1.0)
        dt_safe = max_timestep("upwind", c_or_D=1.0, dx=0.1, pde_type="hyperbolic", safety_factor=0.9)
        assert dt_safe == pytest.approx(0.9 * dt_full, rel=1e-6)

    def test_unconditional_infinite(self):
        """Unconditionally stable scheme returns inf."""
        dt = max_timestep("crank_nicolson", c_or_D=1.0, dx=0.1, pde_type="parabolic")
        assert dt == float('inf')


# ─── Von Neumann Analysis Tests ───────────────────────────────────────────

class TestVonNeumannAnalysis:
    """Tests for Von Neumann stability analysis."""

    def test_upwind_stable(self):
        """Upwind |G| ≤ 1 for CFL ≤ 1."""
        r = von_neumann_analysis("upwind", cfl=0.5)
        assert r.is_stable is True
        assert r.amplitude <= 1.0

    def test_upwind_dissipative(self):
        """Upwind is dissipative (|G| < 1)."""
        r = von_neumann_analysis("upwind", cfl=0.5)
        assert r.is_dissipative is True
        assert r.amplitude < 1.0

    def test_leapfrog_non_dissipative(self):
        """Leapfrog |G| = 1 exactly (non-dissipative)."""
        r = von_neumann_analysis("leapfrog_hyperbolic", cfl=0.5)
        assert r.is_stable is True
        # For stable leapfrog, |G| = 1 exactly
        assert r.amplitude == pytest.approx(1.0, rel=1e-6)
        assert r.is_dissipative is False

    def test_ftcs_stable(self):
        """FTCS |G| ≤ 1 for CFL ≤ 0.5."""
        r = von_neumann_analysis("ftcs", cfl=0.4)
        assert r.is_stable is True

    def test_ftcs_unstable(self):
        """FTCS |G| > 1 for CFL > 0.5."""
        r = von_neumann_analysis("ftcs", cfl=0.6, wavenumber_dx=math.pi)
        # At k*dx = π, G = 1 - 4*CFL*sin²(π/2) = 1 - 4*0.6*1 = -1.4
        assert r.is_stable is False
        assert r.amplitude > 1.0

    def test_crank_nicolson_always_stable(self):
        """Crank-Nicolson |G| ≤ 1 for any CFL."""
        for cfl in [0.1, 1.0, 10.0, 100.0]:
            r = von_neumann_analysis("crank_nicolson", cfl=cfl)
            assert r.is_stable is True
            assert r.amplitude <= 1.0

    def test_btcs_always_stable(self):
        """BTCS |G| < 1 for any CFL."""
        r = von_neumann_analysis("btcs", cfl=10.0)
        assert r.is_stable is True
        assert r.amplitude < 1.0


# ─── Scheme Information Tests ─────────────────────────────────────────────

class TestSchemeInfo:
    """Tests for get_scheme_info function."""

    def test_upwind_properties(self):
        """Upwind scheme properties correct."""
        r = get_scheme_info("upwind")
        assert r.order_space == 1
        assert r.order_time == 1
        assert r.is_explicit is True
        assert r.cfl_limit == 1.0
        assert r.pde_type == "hyperbolic"

    def test_crank_nicolson_properties(self):
        """Crank-Nicolson properties correct."""
        r = get_scheme_info("crank_nicolson")
        assert r.order_space == 2
        assert r.order_time == 2
        assert r.is_explicit is False
        assert r.cfl_limit is None
        assert r.pde_type == "parabolic"

    def test_lax_wendroff_second_order(self):
        """Lax-Wendroff is second-order."""
        r = get_scheme_info("lax_wendroff")
        assert r.order_space == 2
        assert r.order_time == 2

    def test_unknown_scheme_error(self):
        """Unknown scheme raises error."""
        with pytest.raises(ValueError):
            get_scheme_info("invalid_scheme")


class TestListSchemes:
    """Tests for list_schemes function."""

    def test_list_all(self):
        """List all schemes."""
        schemes = list_schemes()
        assert "upwind" in schemes
        assert "ftcs" in schemes
        assert "crank_nicolson" in schemes
        assert len(schemes) >= 10

    def test_filter_hyperbolic(self):
        """Filter hyperbolic schemes."""
        schemes = list_schemes("hyperbolic")
        assert "upwind" in schemes
        assert "lax_wendroff" in schemes
        assert "ftcs" not in schemes

    def test_filter_parabolic(self):
        """Filter parabolic schemes."""
        schemes = list_schemes("parabolic")
        assert "ftcs" in schemes
        assert "crank_nicolson" in schemes
        assert "upwind" not in schemes


# ─── Lax Equivalence Tests ────────────────────────────────────────────────

class TestLaxEquivalence:
    """Tests for Lax equivalence theorem checking."""

    def test_linear_consistent_stable_convergent(self):
        """Linear + consistent + stable ⟹ convergent."""
        r = check_lax_equivalence(
            is_consistent=True,
            consistency_order=2,
            is_stable=True,
            is_linear=True,
        )
        assert r.is_convergent is True
        assert r.theorem_applies is True

    def test_linear_not_consistent_not_convergent(self):
        """Not consistent ⟹ not convergent."""
        r = check_lax_equivalence(
            is_consistent=False,
            consistency_order=0,
            is_stable=True,
            is_linear=True,
        )
        assert r.is_convergent is False

    def test_linear_not_stable_not_convergent(self):
        """Not stable ⟹ not convergent."""
        r = check_lax_equivalence(
            is_consistent=True,
            consistency_order=2,
            is_stable=False,
            is_linear=True,
        )
        assert r.is_convergent is False

    def test_nonlinear_theorem_not_applicable(self):
        """Lax theorem does NOT apply to nonlinear problems."""
        r = check_lax_equivalence(
            is_consistent=True,
            consistency_order=2,
            is_stable=True,
            is_linear=False,
        )
        assert r.theorem_applies is False
        assert r.problem_type == "nonlinear"


# ─── Accuracy Analysis Tests ──────────────────────────────────────────────

class TestAccuracyAnalysis:
    """Tests for accuracy analysis."""

    def test_upwind_first_order(self):
        """Upwind is first-order."""
        r = analyze_accuracy("upwind")
        assert r.order_space == 1
        assert r.order_time == 1
        assert "dissipative" in r.leading_error_type.lower()

    def test_lax_wendroff_second_order(self):
        """Lax-Wendroff is second-order with dispersion."""
        r = analyze_accuracy("lax_wendroff")
        assert r.order_space == 2
        assert r.order_time == 2
        assert "dispersive" in r.leading_error_type.lower()

    def test_crank_nicolson_balanced(self):
        """Crank-Nicolson is balanced."""
        r = analyze_accuracy("crank_nicolson")
        assert r.order_space == 2
        assert r.order_time == 2

    def test_richardson_applicable(self):
        """Richardson extrapolation applicable to regular schemes."""
        r = analyze_accuracy("ftcs")
        assert r.richardson_possible is True


# ─── Common Errors Tests ──────────────────────────────────────────────────

class TestCommonErrors:
    """Tests for common error checking."""

    def test_cfl_sufficient_is_error(self):
        """CFL sufficient claim is an error."""
        r = check_common_error("CFL is sufficient for stability")
        assert r["is_error"] is True
        assert "necessary" in r["correct"].lower()

    def test_leapfrog_diffusion_is_error(self):
        """Leapfrog for diffusion is an error."""
        r = check_common_error("Use leapfrog for diffusion equation")
        assert r["is_error"] is True
        assert "unstable" in r["correct"].lower()

    def test_lax_nonlinear_is_error(self):
        """Lax applies to nonlinear is an error."""
        r = check_common_error("Lax applies to nonlinear PDEs")
        assert r["is_error"] is True
        assert "linear" in r["correct"].lower()

    def test_consistency_implies_convergence_is_error(self):
        """Consistency implies convergence is an error."""
        r = check_common_error("Consistency implies convergence")
        assert r["is_error"] is True
        assert "stability" in r["correct"].lower()


# ─── Report String Tests ──────────────────────────────────────────────────

class TestReportStrings:
    """Tests for report __str__ methods."""

    def test_cfl_report_str(self):
        """CFL report is readable."""
        r = check_cfl("upwind", cfl_number=0.8)
        s = str(r)
        assert "CFL" in s
        assert "STABLE" in s

    def test_von_neumann_report_str(self):
        """Von Neumann report is readable."""
        r = von_neumann_analysis("upwind", cfl=0.5)
        s = str(r)
        assert "amplification" in s.lower() or "|G|" in s

    def test_scheme_report_str(self):
        """Scheme report is readable."""
        r = get_scheme_info("crank_nicolson")
        s = str(r)
        assert "Crank" in s
        assert "parabolic" in s

    def test_lax_report_str(self):
        """Lax equivalence report is readable."""
        r = check_lax_equivalence(True, 2, True, is_linear=True)
        s = str(r)
        assert "Lax" in s
        assert "CONVERGENT" in s

    def test_accuracy_report_str(self):
        """Accuracy report is readable."""
        r = analyze_accuracy("lax_wendroff")
        s = str(r)
        assert "O(" in s
        assert "Lax-Wendroff" in s


# ─── Physical Consistency Tests ───────────────────────────────────────────

class TestPhysicalConsistency:
    """Tests for mathematical/physical consistency."""

    def test_implicit_always_stable(self):
        """Implicit schemes unconditionally stable."""
        implicit_schemes = ["btcs", "crank_nicolson", "beam_warming"]
        for scheme in implicit_schemes:
            info = get_scheme_info(scheme)
            assert info.is_explicit is False
            assert info.cfl_limit is None

    def test_second_order_schemes(self):
        """Second-order schemes identified correctly."""
        second_order = ["lax_wendroff", "leapfrog_hyperbolic", "crank_nicolson"]
        for scheme in second_order:
            info = get_scheme_info(scheme)
            assert info.order_space >= 2
            assert info.order_time >= 2

    def test_stability_implies_bounded_amplification(self):
        """Stable scheme has |G| ≤ 1."""
        stable_cases = [
            ("upwind", 0.5),
            ("lax_wendroff", 0.8),
            ("ftcs", 0.4),
            ("crank_nicolson", 10.0),
        ]
        for scheme, cfl in stable_cases:
            r = von_neumann_analysis(scheme, cfl=cfl)
            if check_cfl(scheme, cfl).is_stable:
                assert r.amplitude <= 1.0 + 1e-10


# ─── Edge Cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_cfl(self):
        """CFL = 0 is stable."""
        r = check_cfl("upwind", cfl_number=0.0)
        assert r.is_stable is True

    def test_exact_cfl_limit(self):
        """CFL exactly at limit is stable."""
        r = check_cfl("upwind", cfl_number=1.0)
        assert r.is_stable is True

    def test_very_large_cfl_implicit(self):
        """Very large CFL for implicit scheme."""
        r = check_cfl("crank_nicolson", cfl_number=1e10)
        assert r.is_stable is True

    def test_negative_dx_error(self):
        """Negative dx raises error."""
        with pytest.raises(ValueError):
            cfl_hyperbolic(c=1.0, dt=0.1, dx=-0.1)

    def test_scheme_name_normalization(self):
        """Scheme names normalized (spaces, hyphens)."""
        r1 = get_scheme_info("crank_nicolson")
        r2 = get_scheme_info("crank-nicolson")
        r3 = get_scheme_info("Crank Nicolson")
        assert r1.name == r2.name == r3.name
