"""Tests for noethersolve.optimization_convergence module.

Tests cover:
- Gradient descent convergence rate
- Nesterov accelerated convergence
- Algorithm comparison
- Condition number analysis
- Oracle lower bounds
- Step size analysis
"""

import pytest

from noethersolve.optimization_convergence import (
    gradient_descent_rate,
    nesterov_rate,
    compare_algorithms,
    analyze_conditioning,
    oracle_lower_bound,
    optimal_step_size,
    non_convex_rate,
    list_algorithms,
    iterations_needed,
)


# ─── Gradient Descent Tests ────────────────────────────────────────────────

class TestGradientDescentRate:
    """Tests for gradient_descent_rate()."""

    def test_exact_rate_formula(self):
        """Rate is exactly 1 - μ/L."""
        L, mu = 10.0, 1.0
        r = gradient_descent_rate(L, mu)
        assert r.rate == pytest.approx(1 - 1/10, rel=1e-6)
        assert r.rate == pytest.approx(0.9, rel=1e-6)

    def test_condition_number(self):
        """Condition number is L/μ."""
        L, mu = 100.0, 2.0
        r = gradient_descent_rate(L, mu)
        assert r.condition_number == pytest.approx(50.0, rel=1e-6)

    def test_not_optimal_for_kappa_gt_1(self):
        """GD is not optimal when κ > 1."""
        r = gradient_descent_rate(L=10.0, mu=1.0)
        assert r.is_optimal is False

    def test_optimal_at_kappa_1(self):
        """GD is optimal when κ = 1 (perfectly conditioned)."""
        r = gradient_descent_rate(L=1.0, mu=1.0)
        assert r.is_optimal is True
        assert r.rate == 0.0

    def test_iterations_calculated(self):
        """Iterations to epsilon calculated correctly."""
        r = gradient_descent_rate(L=10.0, mu=1.0, epsilon=1e-6)
        # rate = 0.9, need 0.9^k < 1e-6
        # k > log(1e-6) / log(0.9) ≈ 131
        assert r.iterations_to_epsilon > 100
        assert r.iterations_to_epsilon < 200

    def test_invalid_parameters(self):
        """Invalid parameters raise errors."""
        with pytest.raises(ValueError):
            gradient_descent_rate(L=-1.0, mu=1.0)
        with pytest.raises(ValueError):
            gradient_descent_rate(L=1.0, mu=2.0)  # μ > L


# ─── Nesterov Rate Tests ───────────────────────────────────────────────────

class TestNesterovRate:
    """Tests for nesterov_rate()."""

    def test_exact_rate_formula(self):
        """Rate is exactly 1 - √(μ/L)."""
        L, mu = 100.0, 1.0  # κ = 100
        r = nesterov_rate(L, mu)
        assert r.rate == pytest.approx(1 - 0.1, rel=1e-6)  # 1 - 1/√100

    def test_always_optimal(self):
        """Nesterov is always optimal (achieves lower bound)."""
        r = nesterov_rate(L=100.0, mu=1.0)
        assert r.is_optimal is True

    def test_faster_than_gd(self):
        """Nesterov has better rate than GD for κ > 1."""
        L, mu = 100.0, 1.0
        gd = gradient_descent_rate(L, mu)
        nest = nesterov_rate(L, mu)
        assert nest.rate < gd.rate
        assert nest.iterations_to_epsilon < gd.iterations_to_epsilon

    def test_same_as_gd_at_kappa_1(self):
        """At κ = 1, same as GD."""
        L, mu = 1.0, 1.0
        gd = gradient_descent_rate(L, mu)
        nest = nesterov_rate(L, mu)
        assert gd.rate == nest.rate == 0.0

    def test_sqrt_speedup(self):
        """Speedup is approximately √κ in iterations."""
        L, mu = 10000.0, 1.0  # κ = 10000, √κ = 100
        gd = gradient_descent_rate(L, mu)
        nest = nesterov_rate(L, mu)
        speedup = gd.iterations_to_epsilon / nest.iterations_to_epsilon
        # Should be close to √κ = 100
        assert 50 < speedup < 200


# ─── Comparison Tests ──────────────────────────────────────────────────────

class TestCompareAlgorithms:
    """Tests for compare_algorithms()."""

    def test_nesterov_wins_for_ill_conditioned(self):
        """Nesterov wins for κ >> 1."""
        r = compare_algorithms(L=1000.0, mu=1.0)
        assert r.winner == "Nesterov"
        assert r.speedup_factor > 1

    def test_tie_at_kappa_1(self):
        """Tie at κ = 1."""
        r = compare_algorithms(L=1.0, mu=1.0)
        assert "Tie" in r.winner

    def test_speedup_factor_sqrt_kappa(self):
        """Speedup factor is approximately √κ."""
        L, mu = 100.0, 1.0  # κ = 100, √κ = 10
        r = compare_algorithms(L, mu)
        # log ratio of rates gives iteration speedup
        assert 5 < r.speedup_factor < 20


# ─── Conditioning Tests ────────────────────────────────────────────────────

class TestAnalyzeConditioning:
    """Tests for analyze_conditioning()."""

    def test_well_conditioned(self):
        """κ ≤ 10 is well-conditioned."""
        r = analyze_conditioning(L=5.0, mu=1.0)
        assert r.classification == "Well-conditioned"

    def test_ill_conditioned(self):
        """κ > 1000 is ill-conditioned."""
        r = analyze_conditioning(L=10000.0, mu=1.0)
        assert r.classification == "Ill-conditioned"

    def test_acceleration_benefit(self):
        """Acceleration benefit calculated correctly."""
        r = analyze_conditioning(L=100.0, mu=1.0)
        # GD needs ~κ iterations, Nesterov needs ~√κ
        assert r.gd_iterations > r.nesterov_iterations
        assert r.acceleration_benefit > 1


# ─── Lower Bound Tests ─────────────────────────────────────────────────────

class TestOracleLowerBound:
    """Tests for oracle_lower_bound()."""

    def test_gd_suboptimal(self):
        """GD does not achieve lower bound."""
        r = oracle_lower_bound(L=100.0, mu=1.0)
        assert r.gd_achieves is False

    def test_nesterov_optimal(self):
        """Nesterov achieves lower bound."""
        r = oracle_lower_bound(L=100.0, mu=1.0)
        assert r.nesterov_achieves is True

    def test_lower_bound_formula(self):
        """Lower bound is (√κ - 1)/(√κ + 1)."""
        L, mu = 100.0, 1.0  # κ = 100, √κ = 10
        r = oracle_lower_bound(L, mu)
        expected = (10 - 1) / (10 + 1)  # 9/11 ≈ 0.818
        assert r.lower_bound_rate == pytest.approx(expected, rel=1e-6)


# ─── Step Size Tests ───────────────────────────────────────────────────────

class TestOptimalStepSize:
    """Tests for optimal_step_size()."""

    def test_gd_step_1_over_L(self):
        """GD optimal step is 1/L for L-smooth."""
        r = optimal_step_size(L=10.0, mu=0.0, algorithm="gd")
        assert r.optimal_step == pytest.approx(0.1, rel=1e-6)

    def test_gd_step_strongly_convex(self):
        """GD step is 2/(L+μ) for strongly convex."""
        L, mu = 10.0, 2.0
        r = optimal_step_size(L, mu, algorithm="gd")
        assert r.optimal_step == pytest.approx(2/(10+2), rel=1e-6)

    def test_divergence_threshold(self):
        """Step > 2/L causes divergence."""
        r = optimal_step_size(L=10.0, algorithm="gd")
        assert r.diverges_above == pytest.approx(0.2, rel=1e-6)

    def test_convergent_range(self):
        """Convergent range is (0, 2/L)."""
        r = optimal_step_size(L=5.0, algorithm="gd")
        assert r.convergent_range[0] == 0.0
        assert r.convergent_range[1] == pytest.approx(0.4, rel=1e-6)


# ─── Non-Convex Tests ──────────────────────────────────────────────────────

class TestNonConvexRate:
    """Tests for non_convex_rate()."""

    def test_iterations_formula(self):
        """Iterations scale as O(LΔ/ε²)."""
        L, _delta = 10.0, 100.0
        epsilon = 0.1
        r = non_convex_rate(L, f_init=100.0, f_star=0.0, epsilon=epsilon)
        # k = 2LΔ/ε² = 2 × 10 × 100 / 0.01 = 200000
        assert r.iterations_to_epsilon == 200000

    def test_stationary_guarantee(self):
        """Guarantee is stationary point only."""
        r = non_convex_rate(L=1.0, f_init=1.0, f_star=0.0)
        assert "stationary" in r.guarantee.lower()
        assert "global" in r.no_guarantee.lower()


# ─── Utility Tests ─────────────────────────────────────────────────────────

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_list_algorithms(self):
        """List returns available algorithms."""
        algs = list_algorithms()
        assert "gradient_descent" in algs
        assert "nesterov" in algs

    def test_iterations_needed(self):
        """iterations_needed matches rate functions."""
        L, mu = 100.0, 1.0
        gd_rate = gradient_descent_rate(L, mu)
        gd_iters = iterations_needed("gd", L, mu)
        assert gd_iters == gd_rate.iterations_to_epsilon


# ─── Report String Tests ───────────────────────────────────────────────────

class TestReportStrings:
    """Tests for report __str__ methods."""

    def test_convergence_report_str(self):
        """Convergence report is readable."""
        r = gradient_descent_rate(L=10.0, mu=1.0)
        s = str(r)
        assert "Gradient Descent" in s
        assert "1 - μ/L" in s or "1 - 1/κ" in s

    def test_comparison_report_str(self):
        """Comparison report is readable."""
        r = compare_algorithms(L=100.0, mu=1.0)
        s = str(r)
        assert "Nesterov" in s
        assert "faster" in s.lower() or "speedup" in s.lower()

    def test_condition_report_str(self):
        """Condition report is readable."""
        r = analyze_conditioning(L=100.0, mu=1.0)
        s = str(r)
        assert "κ" in s or "kappa" in s.lower()
        assert "100" in s  # The condition number

    def test_lower_bound_report_str(self):
        """Lower bound report is readable."""
        r = oracle_lower_bound(L=100.0, mu=1.0)
        s = str(r)
        assert "Lower" in s
        assert "Nesterov" in s


# ─── Physical Consistency Tests ────────────────────────────────────────────

class TestPhysicalConsistency:
    """Tests for mathematical consistency."""

    def test_rate_between_0_and_1(self):
        """Contraction rate is in (0, 1) for κ > 1."""
        for kappa in [2, 10, 100, 1000]:
            L, mu = kappa, 1.0
            gd = gradient_descent_rate(L, mu)
            nest = nesterov_rate(L, mu)
            assert 0 < gd.rate < 1
            assert 0 < nest.rate < 1

    def test_nesterov_always_better_or_equal(self):
        """Nesterov rate ≤ GD rate for all valid κ."""
        for kappa in [1, 2, 10, 100, 1000]:
            L, mu = kappa, 1.0
            gd = gradient_descent_rate(L, mu)
            nest = nesterov_rate(L, mu)
            assert nest.rate <= gd.rate + 1e-10

    def test_iterations_increase_with_kappa(self):
        """Worse conditioning needs more iterations."""
        iters_prev = 0
        for kappa in [2, 10, 100]:
            L, mu = kappa, 1.0
            r = gradient_descent_rate(L, mu)
            assert r.iterations_to_epsilon > iters_prev
            iters_prev = r.iterations_to_epsilon

    def test_lower_bound_below_gd(self):
        """Lower bound rate < GD rate (GD is suboptimal)."""
        r = oracle_lower_bound(L=100.0, mu=1.0)
        gd_rate = 1 - 1/100
        assert r.lower_bound_rate < gd_rate


# ─── Edge Cases ────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases."""

    def test_kappa_exactly_1(self):
        """κ = 1 (perfect conditioning) works."""
        r = gradient_descent_rate(L=1.0, mu=1.0)
        assert r.rate == 0.0
        assert r.is_optimal is True

    def test_large_kappa(self):
        """Very large κ works."""
        r = gradient_descent_rate(L=1e10, mu=1.0)
        assert r.rate == pytest.approx(1 - 1e-10, rel=1e-6)

    def test_small_epsilon(self):
        """Very small epsilon gives large iteration count."""
        r = gradient_descent_rate(L=10.0, mu=1.0, epsilon=1e-20)
        assert r.iterations_to_epsilon > 400

    def test_non_convex_zero_delta(self):
        """f_init = f_star is edge case."""
        with pytest.raises(ValueError):
            non_convex_rate(L=1.0, f_init=0.0, f_star=1.0)  # f_init < f_star
