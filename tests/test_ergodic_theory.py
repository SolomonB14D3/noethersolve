"""Tests for noethersolve.ergodic_theory module.

Tests cover:
- Ergodic hierarchy classification
- Level comparisons and implications
- Lyapunov exponent analysis
- Entropy calculations (Pesin formula)
- Poincaré recurrence
- Mixing rate analysis
"""

import math
import pytest

from noethersolve.ergodic_theory import (
    classify_system,
    compare_levels,
    lyapunov_analysis,
    entropy_analysis,
    poincare_recurrence,
    mixing_rate,
    list_systems,
    list_levels,
    is_stronger,
    implies,
    HIERARCHY_LEVELS,
)


# ─── Hierarchy Classification Tests ──────────────────────────────────────────

class TestClassifySystem:
    """Tests for classify_system()."""

    def test_bernoulli_shift(self):
        """Bernoulli shift is the strongest level."""
        r = classify_system(name="bernoulli_shift")
        assert r.level == "Bernoulli"
        assert r.is_bernoulli is True
        assert r.is_k_mixing is True
        assert r.is_mixing is True
        assert r.is_weak_mixing is True
        assert r.is_ergodic is True
        assert r.entropy == pytest.approx(math.log(2), rel=1e-6)

    def test_k_mixing_system(self):
        """K-mixing implies mixing and below."""
        r = classify_system(level="k_mixing")
        assert r.is_bernoulli is False
        assert r.is_k_mixing is True
        assert r.is_mixing is True
        assert r.is_weak_mixing is True
        assert r.is_ergodic is True

    def test_mixing_not_k(self):
        """Mixing does not imply K-mixing."""
        r = classify_system(name="horocycle_flow")
        assert r.level == "mixing"
        assert r.is_k_mixing is False
        assert r.is_mixing is True
        assert r.entropy == 0.0  # Zero entropy!

    def test_ergodic_only(self):
        """Ergodic does not imply weak mixing."""
        r = classify_system(name="irrational_rotation")
        assert r.level == "ergodic"
        assert r.is_ergodic is True
        assert r.is_weak_mixing is False
        assert r.is_mixing is False
        assert r.entropy == 0.0

    def test_arnolds_cat(self):
        """Arnold's cat map is K-mixing."""
        r = classify_system(name="arnolds_cat")
        assert r.level == "K-mixing"
        assert r.is_k_mixing is True
        assert r.entropy > 0

    def test_logistic_map_chaos(self):
        """Logistic map at r=4 is Bernoulli."""
        r = classify_system(name="logistic_map_chaos")
        assert r.level == "Bernoulli"
        assert r.is_bernoulli is True

    def test_unknown_level_error(self):
        """Unknown level raises error."""
        with pytest.raises(ValueError):
            classify_system(level="superergodic")

    def test_custom_system(self):
        """Custom system with explicit level."""
        r = classify_system(name="My system", level="mixing", entropy=1.5)
        assert r.system_name == "My system"
        assert r.level == "mixing"
        assert r.entropy == 1.5


# ─── Level Comparison Tests ─────────────────────────────────────────────────

class TestCompareLevels:
    """Tests for compare_levels()."""

    def test_bernoulli_implies_ergodic(self):
        """Bernoulli implies ergodic (strictly)."""
        r = compare_levels("bernoulli", "ergodic")
        assert r.relationship == "implies"
        assert r.strict is True

    def test_ergodic_does_not_imply_mixing(self):
        """Ergodic does not imply mixing."""
        r = compare_levels("ergodic", "mixing")
        assert r.relationship == "implied_by"
        assert r.strict is True

    def test_mixing_vs_k_mixing(self):
        """Mixing does not imply K-mixing."""
        r = compare_levels("mixing", "k_mixing")
        assert r.relationship == "implied_by"
        assert r.counterexample is not None
        assert "horocycle" in r.counterexample.lower()

    def test_same_level(self):
        """Same level is equivalent."""
        r = compare_levels("mixing", "mixing")
        assert r.relationship == "equivalent"
        assert r.strict is False

    def test_weak_mixing_vs_ergodic(self):
        """Weak mixing strictly implies ergodic."""
        r = compare_levels("weak_mixing", "ergodic")
        assert r.relationship == "implies"
        assert r.strict is True
        assert "irrational rotation" in r.counterexample.lower()


# ─── Hierarchy Order Tests ──────────────────────────────────────────────────

class TestHierarchyOrder:
    """Tests for is_stronger() and implies()."""

    def test_bernoulli_strongest(self):
        """Bernoulli is stronger than everything."""
        for level in ["k_mixing", "mixing", "weak_mixing", "ergodic"]:
            assert is_stronger("bernoulli", level)
            assert implies("bernoulli", level)

    def test_ergodic_weakest(self):
        """Ergodic is weaker than everything except itself."""
        for level in ["bernoulli", "k_mixing", "mixing", "weak_mixing"]:
            assert not is_stronger("ergodic", level)
            assert not implies("ergodic", level)

    def test_implies_self(self):
        """Every level implies itself."""
        for level in HIERARCHY_LEVELS:
            assert implies(level, level)

    def test_transitivity(self):
        """Implication is transitive."""
        assert implies("bernoulli", "k_mixing")
        assert implies("k_mixing", "mixing")
        assert implies("bernoulli", "mixing")  # Transitivity


# ─── Lyapunov Exponent Tests ────────────────────────────────────────────────

class TestLyapunovAnalysis:
    """Tests for lyapunov_analysis()."""

    def test_chaotic_system(self):
        """Positive largest exponent means chaos."""
        r = lyapunov_analysis([0.5, 0.0, -0.5])
        assert r.is_chaotic is True
        assert r.exponents[0] == 0.5

    def test_regular_system(self):
        """All non-positive exponents means regular."""
        r = lyapunov_analysis([0.0, -0.2, -0.5])
        assert r.is_chaotic is False

    def test_conservative_system(self):
        """Sum = 0 means conservative/Hamiltonian."""
        r = lyapunov_analysis([0.3, -0.3])
        assert r.system_type == "conservative"
        assert abs(sum(r.exponents)) < 1e-10

    def test_dissipative_system(self):
        """Sum < 0 means dissipative."""
        r = lyapunov_analysis([0.5, -0.2, -0.8])
        assert r.system_type == "dissipative"

    def test_sum_positive_exponents(self):
        """Sum of positive exponents calculated correctly."""
        r = lyapunov_analysis([0.5, 0.2, -0.3, -0.5])
        assert r.sum_positive == pytest.approx(0.7, rel=1e-6)

    def test_kaplan_yorke_dimension(self):
        """Kaplan-Yorke dimension calculation."""
        # For Lorenz: λ = (0.91, 0, -14.57)
        r = lyapunov_analysis([0.91, 0.0, -14.57])
        # D_KY = 2 + 0.91/14.57 ≈ 2.06
        assert 2.0 < r.kaplan_yorke_dim < 2.1

    def test_sorted_exponents(self):
        """Exponents are sorted largest to smallest."""
        r = lyapunov_analysis([-0.5, 0.3, 0.0])
        assert r.exponents == [0.3, 0.0, -0.5]

    def test_empty_exponents_error(self):
        """Empty exponents list raises error."""
        with pytest.raises(ValueError):
            lyapunov_analysis([])


# ─── Entropy Tests ─────────────────────────────────────────────────────────

class TestEntropyAnalysis:
    """Tests for entropy_analysis()."""

    def test_positive_entropy_chaotic(self):
        """Positive entropy means deterministic chaos."""
        r = entropy_analysis(ks_entropy=0.5)
        assert r.is_deterministic is True
        assert r.ks_entropy == 0.5

    def test_zero_entropy_regular(self):
        """Zero entropy means predictable."""
        r = entropy_analysis(ks_entropy=0.0)
        assert r.is_deterministic is False

    def test_pesin_formula_satisfied(self):
        """Pesin formula h = Σλ⁺ for SRB measure."""
        r = entropy_analysis(ks_entropy=0.5, lyapunov_positive_sum=0.5)
        assert r.satisfies_pesin is True

    def test_pesin_formula_violated(self):
        """Pesin inequality h < Σλ⁺ for non-SRB."""
        r = entropy_analysis(ks_entropy=0.3, lyapunov_positive_sum=0.5)
        assert r.satisfies_pesin is False

    def test_negative_entropy_error(self):
        """Negative entropy raises error."""
        with pytest.raises(ValueError):
            entropy_analysis(ks_entropy=-0.1)

    def test_topological_entropy_included(self):
        """Topological entropy shown in report."""
        r = entropy_analysis(ks_entropy=0.5, topological_entropy=0.8)
        assert r.topological_entropy == 0.8
        s = str(r)
        assert "h_top" in s


# ─── Poincaré Recurrence Tests ─────────────────────────────────────────────

class TestPoincareRecurrence:
    """Tests for poincare_recurrence()."""

    def test_return_time_formula(self):
        """Return time ~ 1/measure (Kac's lemma)."""
        r = poincare_recurrence(set_measure=0.01)
        assert r.estimated_return_time == pytest.approx(100.0, rel=1e-6)

    def test_small_set_warning(self):
        """Very small sets give warning about return time."""
        r = poincare_recurrence(set_measure=1e-15)
        s = str(r)
        assert "WARNING" in s or "10" in s  # Should mention large time

    def test_finite_recurrence(self):
        """Finite phase space guarantees recurrence."""
        r = poincare_recurrence(set_measure=0.1, phase_space_volume=1.0)
        assert r.is_finite_recurrence is True

    def test_invalid_measure_error(self):
        """Measure must be positive."""
        with pytest.raises(ValueError):
            poincare_recurrence(set_measure=0.0)

    def test_measure_exceeds_volume_error(self):
        """Measure cannot exceed phase space volume."""
        with pytest.raises(ValueError):
            poincare_recurrence(set_measure=2.0, phase_space_volume=1.0)


# ─── Mixing Rate Tests ─────────────────────────────────────────────────────

class TestMixingRate:
    """Tests for mixing_rate()."""

    def test_exponential_mixing(self):
        """Exponential mixing implies K-system."""
        r = mixing_rate(rate_type="exponential", rate_value=0.5)
        assert r.rate_type == "exponential"
        assert r.decay_rate == 0.5
        s = str(r)
        assert "K-system" in s or "K" in s

    def test_polynomial_mixing(self):
        """Polynomial mixing is slower."""
        r = mixing_rate(rate_type="polynomial", rate_value=2.0)
        assert r.rate_type == "polynomial"
        assert r.decay_exponent == 2.0

    def test_rapid_mixing(self):
        """Fast decay is rapid mixing."""
        r = mixing_rate(rate_type="exponential", rate_value=1.0)
        assert r.is_rapid_mixing is True

    def test_slow_mixing(self):
        """Slow decay is not rapid mixing."""
        r = mixing_rate(rate_type="polynomial", rate_value=0.5)
        assert r.is_rapid_mixing is False

    def test_invalid_rate_type_error(self):
        """Invalid rate type raises error."""
        with pytest.raises(ValueError):
            mixing_rate(rate_type="linear", rate_value=1.0)

    def test_nonpositive_rate_error(self):
        """Rate must be positive."""
        with pytest.raises(ValueError):
            mixing_rate(rate_type="exponential", rate_value=-0.1)


# ─── Utility Function Tests ────────────────────────────────────────────────

class TestUtilityFunctions:
    """Tests for list_systems(), list_levels(), etc."""

    def test_list_systems(self):
        """List systems returns non-empty list."""
        systems = list_systems()
        assert len(systems) > 5
        assert "bernoulli_shift" in systems
        assert "arnolds_cat" in systems

    def test_list_levels(self):
        """List levels returns correct order."""
        levels = list_levels()
        assert levels == ["Bernoulli", "K-mixing", "mixing", "weak_mixing", "ergodic"]

    def test_hierarchy_order(self):
        """Hierarchy levels in correct order."""
        levels = list_levels()
        assert levels[0] == "Bernoulli"  # Strongest
        assert levels[-1] == "ergodic"   # Weakest


# ─── Report String Tests ───────────────────────────────────────────────────

class TestReportStrings:
    """Tests for report __str__ methods."""

    def test_hierarchy_report_str(self):
        """Hierarchy report is readable."""
        r = classify_system(name="bernoulli_shift")
        s = str(r)
        assert "Bernoulli" in s
        assert "✓" in s  # Should show checkmarks
        assert "entropy" in s.lower() or "h =" in s

    def test_lyapunov_report_str(self):
        """Lyapunov report is readable."""
        r = lyapunov_analysis([0.5, -0.5])
        s = str(r)
        assert "Lyapunov" in s
        assert "0.5" in s or ".5" in s

    def test_entropy_report_str(self):
        """Entropy report is readable."""
        r = entropy_analysis(ks_entropy=0.42)
        s = str(r)
        assert "Entropy" in s or "entropy" in s
        assert "0.42" in s or ".42" in s

    def test_recurrence_report_str(self):
        """Recurrence report is readable."""
        r = poincare_recurrence(set_measure=0.01)
        s = str(r)
        assert "Poincaré" in s or "Recurrence" in s
        assert "1.00e+02" in s or "100" in s  # Expected return time


# ─── Physical Consistency Tests ────────────────────────────────────────────

class TestPhysicalConsistency:
    """Tests for physical self-consistency."""

    def test_positive_entropy_implies_k_mixing(self):
        """Positive entropy systems should be K-mixing or stronger."""
        # Systems with h > 0 are K-systems
        r = classify_system(name="arnolds_cat")
        assert r.entropy > 0
        assert r.is_k_mixing is True

    def test_zero_entropy_not_k(self):
        """Zero entropy systems are not K-mixing."""
        r = classify_system(name="horocycle_flow")
        assert r.entropy == 0.0
        assert r.is_k_mixing is False

    def test_irrational_rotation_properties(self):
        """Irrational rotation: ergodic, not weak mixing, zero entropy."""
        r = classify_system(name="irrational_rotation")
        assert r.is_ergodic is True
        assert r.is_weak_mixing is False
        assert r.entropy == 0.0

    def test_conservative_lyapunov_sum(self):
        """Conservative systems have zero Lyapunov sum."""
        # Hamiltonian systems: symplectic, so sum = 0
        r = lyapunov_analysis([1.0, -1.0])  # Paired exponents
        assert r.system_type == "conservative"

    def test_pesin_equality_srb(self):
        """SRB measures satisfy Pesin equality."""
        # For SRB (Sinai-Ruelle-Bowen) measures: h = Σλ⁺
        r = entropy_analysis(ks_entropy=0.91, lyapunov_positive_sum=0.91)
        assert r.satisfies_pesin is True


# ─── Edge Cases ────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_lyapunov_exponent(self):
        """Single exponent works."""
        r = lyapunov_analysis([0.5])
        assert r.dimension == 1
        assert r.is_chaotic is True

    def test_all_zero_exponents(self):
        """All zero exponents are conservative and not chaotic."""
        r = lyapunov_analysis([0.0, 0.0, 0.0])
        assert r.is_chaotic is False
        assert r.system_type == "conservative"

    def test_very_small_entropy(self):
        """Very small entropy treated as deterministic."""
        r = entropy_analysis(ks_entropy=1e-15)
        assert r.is_deterministic is False  # Essentially zero

    def test_large_return_time(self):
        """Large return time for small measure."""
        r = poincare_recurrence(set_measure=1e-20)
        assert r.estimated_return_time == pytest.approx(1e20, rel=1e-6)
