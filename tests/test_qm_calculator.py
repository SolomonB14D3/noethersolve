"""Tests for noethersolve.qm_calculator — quantum mechanics computational engine."""

import math
import pytest

from noethersolve.qm_calculator import (
    particle_in_box,
    hydrogen_energy,
    uncertainty_check,
    tunneling_probability,
    harmonic_oscillator,
    angular_momentum_addition,
    ParticleInBoxReport,
    HydrogenEnergyReport,
    UncertaintyReport,
    TunnelingReport,
    HarmonicOscillatorReport,
    AngularMomentumReport,
    HBAR, ME, A0, EV_TO_J,
)


# ── Particle in a Box ────────────────────────────────────────────────────

class TestParticleInBox:
    def test_ground_state_energy(self):
        """E_1 for electron in 1nm box ≈ 0.376 eV."""
        r = particle_in_box(n=1, L=1e-9)
        assert isinstance(r, ParticleInBoxReport)
        assert abs(r.E_n_eV - 0.376) < 0.01

    def test_energy_scales_as_n_squared(self):
        r1 = particle_in_box(n=1, L=1e-9)
        r2 = particle_in_box(n=2, L=1e-9)
        r3 = particle_in_box(n=3, L=1e-9)
        assert abs(r2.E_n_J / r1.E_n_J - 4) < 1e-10
        assert abs(r3.E_n_J / r1.E_n_J - 9) < 1e-10

    def test_wavelength(self):
        r = particle_in_box(n=3, L=1e-9)
        assert abs(r.wavelength - 2e-9 / 3) < 1e-20

    def test_nodes(self):
        r = particle_in_box(n=5, L=1e-9)
        assert r.nodes == 4

    def test_ground_state_note(self):
        r = particle_in_box(n=1, L=1e-9)
        assert any("Ground state" in n for n in r.notes)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n must be"):
            particle_in_box(n=0, L=1e-9)

    def test_invalid_L_raises(self):
        with pytest.raises(ValueError, match="L must be positive"):
            particle_in_box(n=1, L=-1)

    def test_str_output(self):
        r = particle_in_box(n=2, L=1e-9)
        assert "Particle in a Box" in str(r)


# ── Hydrogen Energy ──────────────────────────────────────────────────────

class TestHydrogenEnergy:
    def test_ground_state(self):
        """E_1 = -13.6 eV for hydrogen."""
        r = hydrogen_energy(n=1)
        assert isinstance(r, HydrogenEnergyReport)
        assert abs(r.E_n_eV - (-13.6)) < 0.01

    def test_n2_energy(self):
        r = hydrogen_energy(n=2)
        assert abs(r.E_n_eV - (-3.4)) < 0.01

    def test_bohr_radius(self):
        """Ground state radius = a₀ ≈ 0.529 Å."""
        r = hydrogen_energy(n=1)
        assert abs(r.radius_A - 0.529) < 0.01

    def test_radius_scales_as_n_squared(self):
        r1 = hydrogen_energy(n=1)
        r2 = hydrogen_energy(n=2)
        assert abs(r2.radius_m / r1.radius_m - 4) < 1e-10

    def test_degeneracy(self):
        r = hydrogen_energy(n=3)
        assert r.degeneracy == 18  # 2 × 3² = 18

    def test_lyman_alpha_wavelength(self):
        """Lyman-alpha (n=2→1) ≈ 121.5 nm."""
        r = hydrogen_energy(n=2)
        assert r.wavelength_nm is not None
        assert abs(r.wavelength_nm - 121.5) < 1.0

    def test_helium_ion(self):
        """He+ (Z=2): E_1 = -54.4 eV."""
        r = hydrogen_energy(n=1, Z=2)
        assert abs(r.E_n_eV - (-54.4)) < 0.1

    def test_ionization_energy(self):
        r = hydrogen_energy(n=1)
        assert abs(r.ionization_eV - 13.6) < 0.01

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            hydrogen_energy(n=0)

    def test_str_output(self):
        r = hydrogen_energy(n=2)
        assert "Hydrogen" in str(r)


# ── Uncertainty Check ────────────────────────────────────────────────────

class TestUncertaintyCheck:
    def test_satisfied(self):
        """Δx=1Å, Δp=1e-24 kg·m/s → product ≈ 1e-34 ≈ 2ℏ."""
        r = uncertainty_check(delta_x=1e-10, delta_p=1e-24)
        assert isinstance(r, UncertaintyReport)
        assert r.satisfied

    def test_violated(self):
        """Tiny Δx and Δp that violate the principle."""
        r = uncertainty_check(delta_x=1e-15, delta_p=1e-25)
        assert not r.satisfied

    def test_minimum_uncertainty(self):
        """Product exactly ℏ/2 should be satisfied."""
        hbar_2 = HBAR / 2
        r = uncertainty_check(delta_x=1.0, delta_p=hbar_2)
        assert r.satisfied
        assert abs(r.ratio - 1.0) < 1e-5

    def test_large_product(self):
        r = uncertainty_check(delta_x=1.0, delta_p=1.0)
        assert r.satisfied
        assert r.ratio > 1e30

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            uncertainty_check(delta_x=-1, delta_p=1)

    def test_str_output(self):
        r = uncertainty_check(delta_x=1e-10, delta_p=1e-24)
        assert "Uncertainty" in str(r)


# ── Tunneling Probability ────────────────────────────────────────────────

class TestTunnelingProbability:
    def test_below_barrier(self):
        """E < V: tunneling regime."""
        r = tunneling_probability(E=5.0, V=10.0, L=1e-10)
        assert isinstance(r, TunnelingReport)
        assert r.regime == "tunneling"
        assert 0 < r.T < 1
        assert abs(r.T + r.R - 1.0) < 1e-10

    def test_above_barrier(self):
        """E > V: above-barrier oscillatory transmission."""
        r = tunneling_probability(E=15.0, V=10.0, L=1e-10)
        assert r.regime == "above_barrier"
        assert 0 < r.T <= 1

    def test_thick_barrier_zero_transmission(self):
        """Very thick barrier: T → 0."""
        r = tunneling_probability(E=5.0, V=10.0, L=1e-7)
        assert r.T < 1e-10

    def test_thin_barrier_high_transmission(self):
        """Very thin barrier: T → 1."""
        r = tunneling_probability(E=5.0, V=10.0, L=1e-12)
        assert r.T > 0.99

    def test_conservation(self):
        """T + R = 1."""
        r = tunneling_probability(E=3.0, V=8.0, L=5e-11)
        assert abs(r.T + r.R - 1.0) < 1e-12

    def test_negative_energy_raises(self):
        with pytest.raises(ValueError):
            tunneling_probability(E=-1, V=10, L=1e-10)

    def test_str_output(self):
        r = tunneling_probability(E=5.0, V=10.0, L=1e-10)
        assert "Tunneling" in str(r)


# ── Harmonic Oscillator ──────────────────────────────────────────────────

class TestHarmonicOscillator:
    def test_ground_state(self):
        """E_0 = ½ℏω."""
        omega = 1e14
        r = harmonic_oscillator(n=0, omega=omega)
        assert isinstance(r, HarmonicOscillatorReport)
        expected = 0.5 * HBAR * omega
        assert abs(r.E_n_J - expected) < 1e-30

    def test_zero_point_energy(self):
        omega = 1e14
        r = harmonic_oscillator(n=0, omega=omega)
        assert abs(r.E_n_J - r.zero_point_J) < 1e-30

    def test_equally_spaced(self):
        """E_n - E_{n-1} = ℏω for all n."""
        omega = 1e14
        r0 = harmonic_oscillator(n=0, omega=omega)
        r1 = harmonic_oscillator(n=1, omega=omega)
        r2 = harmonic_oscillator(n=2, omega=omega)
        spacing01 = r1.E_n_J - r0.E_n_J
        spacing12 = r2.E_n_J - r1.E_n_J
        expected = HBAR * omega
        assert abs(spacing01 - expected) < 1e-30
        assert abs(spacing12 - expected) < 1e-30

    def test_energy_formula(self):
        r = harmonic_oscillator(n=5, omega=1e13)
        expected = 5.5 * HBAR * 1e13
        assert abs(r.E_n_J - expected) < 1e-30

    def test_ground_state_note(self):
        r = harmonic_oscillator(n=0, omega=1e14)
        assert any("Ground state" in n for n in r.notes)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            harmonic_oscillator(n=-1, omega=1e14)

    def test_str_output(self):
        r = harmonic_oscillator(n=3, omega=1e14)
        assert "Harmonic Oscillator" in str(r)


# ── Angular Momentum Addition ────────────────────────────────────────────

class TestAngularMomentumAddition:
    def test_two_spin_half(self):
        """½ ⊗ ½ = 0 ⊕ 1 (singlet + triplet)."""
        r = angular_momentum_addition(j1=0.5, j2=0.5)
        assert isinstance(r, AngularMomentumReport)
        assert r.j_min == 0
        assert r.j_max == 1.0
        assert r.allowed_j == [0, 1.0]
        assert r.total_states == 4

    def test_spin_half_plus_one(self):
        """½ ⊗ 1 = ½ ⊕ 3/2."""
        r = angular_momentum_addition(j1=0.5, j2=1.0)
        assert r.allowed_j == [0.5, 1.5]
        assert r.total_states == 6  # 2×3 = 6

    def test_two_spin_one(self):
        """1 ⊗ 1 = 0 ⊕ 1 ⊕ 2."""
        r = angular_momentum_addition(j1=1, j2=1)
        assert r.allowed_j == [0, 1, 2]
        assert r.total_states == 9  # 3×3 = 9

    def test_state_counting(self):
        """Σ(2J+1) must equal (2j1+1)(2j2+1)."""
        r = angular_momentum_addition(j1=2, j2=1.5)
        check_sum = sum(int(2 * j + 1) for j in r.allowed_j)
        assert check_sum == r.total_states

    def test_j_zero(self):
        """j1=0: J = j2 only."""
        r = angular_momentum_addition(j1=0, j2=2)
        assert r.allowed_j == [2.0]
        assert r.total_states == 5

    def test_negative_j_raises(self):
        with pytest.raises(ValueError):
            angular_momentum_addition(j1=-1, j2=1)

    def test_non_half_integer_raises(self):
        with pytest.raises(ValueError):
            angular_momentum_addition(j1=0.3, j2=1)

    def test_str_output(self):
        r = angular_momentum_addition(j1=1, j2=1)
        s = str(r)
        assert "Angular Momentum" in s
        assert "Decomposition" in s


# ── Integration Tests ────────────────────────────────────────────────────

class TestIntegration:
    def test_hydrogen_vs_box(self):
        """Hydrogen ground state energy should differ from a box of similar size."""
        r_h = hydrogen_energy(n=1)
        r_box = particle_in_box(n=1, L=A0)  # box of Bohr radius
        # Different potential → different energies, but same order of magnitude
        assert abs(r_h.E_n_eV) > 1  # should be ~13.6
        assert abs(r_box.E_n_eV) > 1

    def test_uncertainty_from_box(self):
        """For particle in box, Δx ~ L, Δp ~ nπℏ/L. Check consistency."""
        L = 1e-9
        n = 1
        delta_x = L / (2 * math.sqrt(3))  # exact for ground state: L/(2√3)
        delta_p = math.pi * HBAR / L  # ~ ℏπ/L
        r = uncertainty_check(delta_x=delta_x, delta_p=delta_p)
        assert r.satisfied

    def test_tunneling_energy_dependence(self):
        """Higher energy → higher transmission."""
        r_low = tunneling_probability(E=3.0, V=10.0, L=1e-10)
        r_high = tunneling_probability(E=8.0, V=10.0, L=1e-10)
        assert r_high.T > r_low.T
