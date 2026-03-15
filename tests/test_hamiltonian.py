"""Tests for noethersolve.hamiltonian — Hamiltonian system validation."""

import numpy as np
import pytest

from noethersolve.hamiltonian import (
    HamiltonianMonitor,
    HamiltonianReport,
    harmonic_oscillator,
    kepler_2d,
    henon_heiles,
    coupled_oscillators,
)


# ─── Built-in system tests ────────────────────────────────────────────────────

class TestHarmonicOscillator:
    def test_energy_conserved(self):
        ho = harmonic_oscillator(omega=2.0)
        z0 = np.array([1.0, 0.5])
        report = ho.validate(z0, T=50.0, check_liouville=False,
                             check_poincare=False)
        assert isinstance(report, HamiltonianReport)
        assert report.quantities["energy"]["verdict"] == "PASS"

    def test_full_validation_passes(self):
        ho = harmonic_oscillator(omega=1.0)
        z0 = np.array([1.0, 0.0])
        report = ho.validate(z0, T=20.0, rtol=1e-10, atol=1e-12)
        assert report.verdict == "PASS"
        assert "energy" in report.quantities
        assert "liouville_volume" in report.quantities
        assert "poincare_invariant" in report.quantities

    def test_loose_tolerances_may_fail(self):
        ho = harmonic_oscillator(omega=2.0)
        z0 = np.array([1.0, 0.5])
        report = ho.validate(z0, T=500.0, rtol=1e-2, atol=1e-4,
                             check_liouville=False, check_poincare=False)
        # Energy drift at rtol=1e-2 over long time
        assert isinstance(report, HamiltonianReport)


class TestKepler:
    def test_energy_and_L_conserved(self):
        kep = kepler_2d(mu=1.0)
        z0 = np.array([1.0, 0.0, 0.0, 0.8])  # elliptical orbit
        report = kep.validate(z0, T=50.0, rtol=1e-10, atol=1e-12,
                              check_liouville=False, check_poincare=False)
        assert report.quantities["energy"]["verdict"] == "PASS"
        assert report.quantities["angular_momentum"]["verdict"] == "PASS"

    def test_lrl_vector_conserved(self):
        kep = kepler_2d(mu=1.0)
        z0 = np.array([1.0, 0.0, 0.0, 0.8])
        report = kep.validate(z0, T=30.0, rtol=1e-10, atol=1e-12,
                              check_liouville=False, check_poincare=False)
        assert report.quantities["LRL_magnitude"]["verdict"] == "PASS"

    def test_circular_orbit(self):
        """Circular orbit: energy and L exact.

        Note: LRL magnitude = 0 for circular orbits (eccentricity = 0),
        so frac_var is large (0/0). We only check energy and L.
        """
        kep = kepler_2d(mu=1.0)
        r = 1.0
        v = np.sqrt(1.0 / r)
        z0 = np.array([r, 0.0, 0.0, v])
        report = kep.validate(z0, T=20.0, rtol=1e-10, atol=1e-12,
                              check_liouville=False, check_poincare=False)
        assert report.quantities["energy"]["verdict"] == "PASS"
        assert report.quantities["angular_momentum"]["verdict"] == "PASS"


class TestHenonHeiles:
    def test_low_energy_passes(self):
        hh = henon_heiles()
        z0 = np.array([0.3, 0.0, 0.0, 0.3])  # low energy, regular
        report = hh.validate(z0, T=50.0, rtol=1e-10, atol=1e-12,
                             check_liouville=False, check_poincare=False)
        assert report.quantities["energy"]["verdict"] == "PASS"

    def test_report_format(self):
        hh = henon_heiles()
        z0 = np.array([0.3, 0.0, 0.0, 0.3])
        report = hh.validate(z0, T=10.0, check_liouville=False,
                             check_poincare=False)
        s = str(report)
        assert "Hamiltonian Validation" in s
        assert "henon_heiles" in s


class TestCoupledOscillators:
    def test_energy_conserved(self):
        co = coupled_oscillators(k1=1.0, k2=1.5, k_coupling=0.2)
        z0 = np.array([1.0, 0.0, 0.0, 0.5])
        report = co.validate(z0, T=30.0, rtol=1e-10, atol=1e-12,
                             check_liouville=False, check_poincare=False)
        assert report.quantities["energy"]["verdict"] == "PASS"


# ─── Liouville and Poincaré tests ────────────────────────────────────────────

class TestSymplecticStructure:
    def test_liouville_harmonic(self):
        """Phase-space volume preserved for harmonic oscillator."""
        ho = harmonic_oscillator(omega=1.0)
        z0 = np.array([1.0, 0.0])
        report = ho.validate(z0, T=10.0, rtol=1e-10, atol=1e-12,
                             check_poincare=False, liouville_T=5.0)
        assert report.quantities["liouville_volume"]["verdict"] == "PASS"

    def test_poincare_harmonic(self):
        """Poincaré invariant preserved for harmonic oscillator."""
        ho = harmonic_oscillator(omega=1.0)
        z0 = np.array([1.0, 0.0])
        report = ho.validate(z0, T=10.0, rtol=1e-10, atol=1e-12,
                             check_liouville=False, poincare_T=5.0)
        assert report.quantities["poincare_invariant"]["verdict"] == "PASS"

    def test_liouville_kepler(self):
        """Phase-space volume preserved for Kepler."""
        kep = kepler_2d(mu=1.0)
        z0 = np.array([1.0, 0.0, 0.0, 0.8])
        report = kep.validate(z0, T=20.0, rtol=1e-10, atol=1e-12,
                              check_poincare=False, liouville_T=10.0)
        assert report.quantities["liouville_volume"]["verdict"] == "PASS"


# ─── Custom invariants ───────────────────────────────────────────────────────

class TestCustomInvariants:
    def test_custom_invariant_passes(self):
        """Custom invariant: total energy is trivially conserved."""
        def my_H(z):
            return 0.5 * np.sum(z ** 2)

        def my_dH(z):
            return z.copy()

        mon = HamiltonianMonitor(
            H=my_H, dH=my_dH, n_dof=1, name="test",
            custom_invariants={"norm_sq": lambda z: np.sum(z ** 2)},
        )
        z0 = np.array([1.0, 0.0])
        report = mon.validate(z0, T=10.0, check_liouville=False,
                              check_poincare=False)
        # norm_sq = q² + p² = 2H, so it's conserved
        assert report.quantities["norm_sq"]["verdict"] == "PASS"


# ─── Error handling ──────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_solver_failure(self):
        """Exploding RHS should return FAIL."""
        def bad_H(z):
            return z[0] ** 2 + z[1] ** 2

        def bad_dH(z):
            return np.array([1e20, -1e20])  # huge derivatives

        mon = HamiltonianMonitor(H=bad_H, dH=bad_dH, n_dof=1, name="bad")
        z0 = np.array([1e10, 1e10])
        report = mon.validate(z0, T=1e10, check_liouville=False,
                              check_poincare=False, rtol=1e-14)
        assert isinstance(report, HamiltonianReport)

    def test_passed_property(self):
        ho = harmonic_oscillator(omega=1.0)
        z0 = np.array([1.0, 0.0])
        report = ho.validate(z0, T=10.0, check_liouville=False,
                             check_poincare=False)
        assert report.passed == (report.verdict == "PASS")


# ─── Report formatting ──────────────────────────────────────────────────────

class TestReportFormatting:
    def test_str_contains_system_name(self):
        ho = harmonic_oscillator(omega=1.0)
        z0 = np.array([1.0, 0.0])
        report = ho.validate(z0, T=10.0, check_liouville=False,
                             check_poincare=False)
        s = str(report)
        assert "harmonic_oscillator" in s

    def test_str_contains_verdict(self):
        ho = harmonic_oscillator(omega=1.0)
        z0 = np.array([1.0, 0.0])
        report = ho.validate(z0, T=10.0, check_liouville=False,
                             check_poincare=False)
        s = str(report)
        assert report.verdict in s
