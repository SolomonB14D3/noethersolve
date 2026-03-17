"""Tests for noethersolve.gr_constraints module.

Tests cover:
- Hamiltonian constraint analysis
- Momentum constraint analysis
- ADM, Bondi, Komar mass definitions
- Mass comparison
- ADM formalism analysis
"""

import math
import pytest

from noethersolve.gr_constraints import (
    check_hamiltonian_constraint,
    check_momentum_constraint,
    check_adm_mass,
    check_bondi_mass,
    check_komar_mass,
    compare_mass_definitions,
    analyze_adm_formalism,
    list_gr_concepts,
    ConstraintReport,
    MassReport,
    MassComparisonReport,
    ADMReport,
)


# ─── Hamiltonian Constraint Tests ─────────────────────────────────────────

class TestHamiltonianConstraint:
    """Tests for Hamiltonian constraint checking."""

    def test_satisfied(self):
        """Constraint satisfied for small value."""
        r = check_hamiltonian_constraint(value=1e-12, tolerance=1e-10)
        assert r.is_satisfied is True
        assert r.constraint_type == "Hamiltonian"

    def test_violated(self):
        """Constraint violated for large value."""
        r = check_hamiltonian_constraint(value=1e-5, tolerance=1e-10)
        assert r.is_satisfied is False

    def test_physical_meaning(self):
        """Physical meaning is explained."""
        r = check_hamiltonian_constraint(value=0.0)
        assert "energy" in r.physical_meaning.lower()

    def test_exactly_zero(self):
        """Zero value satisfies constraint."""
        r = check_hamiltonian_constraint(value=0.0, tolerance=1e-10)
        assert r.is_satisfied is True


class TestMomentumConstraint:
    """Tests for momentum constraint checking."""

    def test_satisfied(self):
        """Constraint satisfied for small value."""
        r = check_momentum_constraint(value=1e-12, tolerance=1e-10)
        assert r.is_satisfied is True

    def test_violated(self):
        """Constraint violated for large value."""
        r = check_momentum_constraint(value=1e-5, tolerance=1e-10)
        assert r.is_satisfied is False

    def test_component_labeled(self):
        """Component is labeled in report."""
        r = check_momentum_constraint(value=0.0, component="x")
        assert "x" in r.constraint_type

    def test_physical_meaning(self):
        """Physical meaning is explained."""
        r = check_momentum_constraint(value=0.0)
        assert "momentum" in r.physical_meaning.lower()


# ─── ADM Mass Tests ───────────────────────────────────────────────────────

class TestADMMass:
    """Tests for ADM mass analysis."""

    def test_applicable(self):
        """ADM mass applicable for asymptotically flat isolated spacetime."""
        r = check_adm_mass(is_asymptotically_flat=True, is_isolated=True)
        assert r.is_applicable is True
        assert r.mass_type == "ADM"

    def test_not_applicable_not_flat(self):
        """ADM mass not applicable if not asymptotically flat."""
        r = check_adm_mass(is_asymptotically_flat=False, is_isolated=True)
        assert r.is_applicable is False

    def test_conserved(self):
        """ADM mass is conserved (constant)."""
        r = check_adm_mass(is_asymptotically_flat=True, is_isolated=True)
        assert "constant" in r.conservation_property.lower()
        assert r.related_to_radiation is False

    def test_mass_value(self):
        """Mass value is stored."""
        r = check_adm_mass(mass_value=10.0)
        assert r.value == 10.0


# ─── Bondi Mass Tests ─────────────────────────────────────────────────────

class TestBondiMass:
    """Tests for Bondi mass analysis."""

    def test_applicable(self):
        """Bondi mass applicable for asymptotically flat with null infinity."""
        r = check_bondi_mass(is_asymptotically_flat=True, has_null_infinity=True)
        assert r.is_applicable is True
        assert r.mass_type == "Bondi"

    def test_not_applicable(self):
        """Bondi mass not applicable without null infinity."""
        r = check_bondi_mass(is_asymptotically_flat=True, has_null_infinity=False)
        assert r.is_applicable is False

    def test_decreases_with_radiation(self):
        """Bondi mass decreases with gravitational radiation."""
        r = check_bondi_mass(has_radiation=True)
        assert r.related_to_radiation is True
        assert "decrease" in r.conservation_property.lower()

    def test_constant_without_radiation(self):
        """Bondi mass constant without radiation."""
        r = check_bondi_mass(has_radiation=False)
        assert "constant" in r.conservation_property.lower()


# ─── Komar Mass Tests ─────────────────────────────────────────────────────

class TestKomarMass:
    """Tests for Komar mass analysis."""

    def test_applicable_stationary(self):
        """Komar mass applicable for stationary spacetime."""
        r = check_komar_mass(is_stationary=True, has_killing_vector=True, killing_type="timelike")
        assert r.is_applicable is True
        assert r.mass_type == "Komar"

    def test_not_applicable_dynamical(self):
        """Komar mass not applicable for dynamical spacetime."""
        r = check_komar_mass(is_stationary=False)
        assert r.is_applicable is False

    def test_not_applicable_wrong_killing(self):
        """Komar mass not applicable for non-timelike Killing vector."""
        r = check_komar_mass(is_stationary=True, has_killing_vector=True, killing_type="spacelike")
        assert r.is_applicable is False

    def test_requires_killing_vector(self):
        """Komar mass requires Killing vector."""
        r = check_komar_mass(is_stationary=True, has_killing_vector=False)
        assert r.is_applicable is False


# ─── Mass Comparison Tests ────────────────────────────────────────────────

class TestMassComparison:
    """Tests for mass definition comparison."""

    def test_schwarzschild_all_agree(self):
        """All masses agree for Schwarzschild (stationary)."""
        r = compare_mass_definitions(spacetime_type="schwarzschild")
        assert r.adm_applicable is True
        assert r.bondi_applicable is True
        assert r.komar_applicable is True
        assert r.masses_agree is True

    def test_kerr_all_agree(self):
        """All masses agree for Kerr (stationary)."""
        r = compare_mass_definitions(spacetime_type="kerr")
        assert r.masses_agree is True

    def test_dynamical_no_komar(self):
        """Komar not applicable for dynamical spacetime."""
        r = compare_mass_definitions(spacetime_type="binary_merger")
        assert r.komar_applicable is False
        assert r.adm_applicable is True
        assert r.bondi_applicable is True

    def test_dynamical_with_radiation_disagree(self):
        """ADM and Bondi disagree with radiation."""
        r = compare_mass_definitions(spacetime_type="bbh", has_radiation=True)
        assert r.masses_agree is False
        assert "Bondi" in r.disagreement_reason

    def test_cosmological_no_masses(self):
        """Cosmological spacetime: no ADM/Bondi."""
        r = compare_mass_definitions(spacetime_type="flrw")
        assert r.adm_applicable is False
        assert r.bondi_applicable is False


# ─── ADM Formalism Tests ──────────────────────────────────────────────────

class TestADMFormalism:
    """Tests for ADM formalism analysis."""

    def test_valid_setup(self):
        """Valid ADM setup with positive lapse."""
        r = analyze_adm_formalism(lapse_value=1.0, hamiltonian_constraint=0.0, momentum_constraint=0.0)
        assert r.is_valid_foliation is True
        assert r.lapse_positive is True
        assert "satisfied" in r.constraint_status.lower()

    def test_invalid_lapse(self):
        """Invalid setup with non-positive lapse."""
        r = analyze_adm_formalism(lapse_value=0.0)
        assert r.lapse_positive is False
        assert r.is_valid_foliation is False

    def test_constraint_violation_detected(self):
        """Constraint violation detected."""
        r = analyze_adm_formalism(hamiltonian_constraint=1.0, momentum_constraint=0.0)
        assert "Hamiltonian" in r.constraint_status or "violated" in r.constraint_status.lower()

    def test_gauge_choices(self):
        """Different gauge choices recognized."""
        for gauge in ["geodesic", "harmonic", "BSSN", "puncture"]:
            r = analyze_adm_formalism(gauge=gauge)
            assert len(r.gauge_choice) > 0


# ─── Utility Tests ────────────────────────────────────────────────────────

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_list_concepts(self):
        """List returns available concepts."""
        concepts = list_gr_concepts()
        assert "hamiltonian_constraint" in concepts
        assert "adm_mass" in concepts
        assert "bondi_mass" in concepts
        assert "komar_mass" in concepts


# ─── Report String Tests ──────────────────────────────────────────────────

class TestReportStrings:
    """Tests for report __str__ methods."""

    def test_constraint_report_str(self):
        """Constraint report is readable."""
        r = check_hamiltonian_constraint(value=0.0)
        s = str(r)
        assert "Hamiltonian" in s
        assert "SATISFIED" in s

    def test_mass_report_str(self):
        """Mass report is readable."""
        r = check_adm_mass()
        s = str(r)
        assert "ADM" in s

    def test_comparison_report_str(self):
        """Comparison report is readable."""
        r = compare_mass_definitions(spacetime_type="schwarzschild")
        s = str(r)
        assert "Schwarzschild" in s or "schwarzschild" in s.lower()

    def test_adm_report_str(self):
        """ADM formalism report is readable."""
        r = analyze_adm_formalism()
        s = str(r)
        assert "ADM" in s
        assert "Lapse" in s


# ─── Physical Consistency Tests ───────────────────────────────────────────

class TestPhysicalConsistency:
    """Tests for physical consistency."""

    def test_adm_never_changes_with_radiation(self):
        """ADM mass never changes with radiation."""
        r = check_adm_mass()
        assert r.related_to_radiation is False
        assert "constant" in r.conservation_property.lower()

    def test_bondi_always_decreases_with_radiation(self):
        """Bondi mass always decreases with radiation."""
        r = check_bondi_mass(has_radiation=True)
        assert r.related_to_radiation is True
        assert "decrease" in r.conservation_property.lower()

    def test_komar_only_stationary(self):
        """Komar mass only for stationary spacetimes."""
        r_stat = check_komar_mass(is_stationary=True, has_killing_vector=True, killing_type="timelike")
        r_dyn = check_komar_mass(is_stationary=False)
        assert r_stat.is_applicable is True
        assert r_dyn.is_applicable is False

    def test_stationary_all_masses_equal(self):
        """For stationary spacetimes, all masses should agree."""
        for st in ["schwarzschild", "kerr", "reissner_nordstrom"]:
            r = compare_mass_definitions(spacetime_type=st)
            assert r.masses_agree is True


# ─── Edge Cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases."""

    def test_exact_zero_constraint(self):
        """Exactly zero constraint value."""
        r = check_hamiltonian_constraint(value=0.0)
        assert r.is_satisfied is True

    def test_negative_constraint_value(self):
        """Negative constraint value handled."""
        r = check_hamiltonian_constraint(value=-1e-12, tolerance=1e-10)
        assert r.is_satisfied is True  # abs(value) < tolerance

    def test_small_positive_lapse(self):
        """Small positive lapse is valid."""
        r = analyze_adm_formalism(lapse_value=1e-10)
        assert r.lapse_positive is True

    def test_negative_lapse_invalid(self):
        """Negative lapse is invalid."""
        r = analyze_adm_formalism(lapse_value=-1.0)
        assert r.lapse_positive is False
        assert r.is_valid_foliation is False
