"""Tests for noethersolve.monitor_em — EM field conservation monitor."""

import numpy as np
import pytest

from noethersolve.monitor_em import EMMonitor
from noethersolve.monitor import MonitorReport


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_plane_wave(N=32, L=2*np.pi, k_z=2.0):
    """Simple circularly polarized plane wave propagating along z.

    E = (cos(kz), sin(kz), 0), B = (-sin(kz), cos(kz), 0)
    This is an exact solution of Maxwell's equations.
    """
    L / N
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    phase = k_z * Z

    Ex = np.cos(phase)
    Ey = np.sin(phase)
    Ez = np.zeros_like(X)
    Bx = -np.sin(phase)
    By = np.cos(phase)
    Bz = np.zeros_like(X)

    return (Ex, Ey, Ez), (Bx, By, Bz)


def make_gaussian_packet(N=32, L=2*np.pi, sigma=0.6, k_z=4.0):
    """Gaussian wave packet with circular polarization."""
    L / N
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    r2 = (X - L/2)**2 + (Y - L/2)**2 + (Z - L/2)**2
    envelope = np.exp(-r2 / (2 * sigma**2))
    phase = k_z * Z

    Ex = envelope * np.cos(phase)
    Ey = envelope * np.sin(phase)
    Ez = np.zeros_like(X)
    Bx = -envelope * np.sin(phase)
    By = envelope * np.cos(phase)
    Bz = np.zeros_like(X)

    return (Ex, Ey, Ez), (Bx, By, Bz)


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestEMMonitorInit:
    def test_initialization(self):
        mon = EMMonitor(N=16, L=2*np.pi)
        assert mon.N == 16
        assert mon.dV > 0

    def test_set_initial(self):
        mon = EMMonitor(N=16)
        E, B = make_plane_wave(N=16)
        mon.set_initial(E, B)
        assert "energy" in mon._initial
        assert "chirality" in mon._initial
        assert "helicity" in mon._initial
        assert "super_energy" in mon._initial

    def test_initial_values_positive_energy(self):
        mon = EMMonitor(N=16)
        E, B = make_plane_wave(N=16)
        mon.set_initial(E, B)
        assert mon._initial["energy"] > 0


class TestEMMonitorCheck:
    def test_check_returns_report(self):
        mon = EMMonitor(N=16)
        E, B = make_plane_wave(N=16)
        mon.set_initial(E, B)
        report = mon.check(E, B)
        assert isinstance(report, MonitorReport)
        assert report.step == 1
        assert "energy" in report.quantities

    def test_identical_fields_no_drift(self):
        """Same fields should have zero drift."""
        mon = EMMonitor(N=16)
        E, B = make_plane_wave(N=16)
        mon.set_initial(E, B)
        report = mon.check(E, B)
        assert report.worst_drift < 1e-10

    def test_check_requires_set_initial(self):
        mon = EMMonitor(N=16)
        E, B = make_plane_wave(N=16)
        with pytest.raises(RuntimeError, match="set_initial"):
            mon.check(E, B)


class TestEMMonitorConservation:
    """Test that exact solutions of Maxwell's equations preserve invariants."""

    def test_plane_wave_energy_conserved(self):
        """A plane wave is a static solution — fields don't change."""
        N = 16
        mon = EMMonitor(N=N)
        E, B = make_plane_wave(N=N)
        mon.set_initial(E, B)

        # Check same fields multiple times (simulating zero time evolution)
        for _ in range(10):
            mon.check(E, B)

        summary = mon.summary()
        assert summary["energy"]["frac_var"] < 1e-10

    def test_plane_wave_chirality_conserved(self):
        N = 16
        mon = EMMonitor(N=N)
        E, B = make_plane_wave(N=N)
        mon.set_initial(E, B)
        for _ in range(10):
            mon.check(E, B)
        summary = mon.summary()
        assert summary["chirality"]["frac_var"] < 1e-10

    def test_plane_wave_all_quantities_stable(self):
        """All quantities should be stable for identical fields."""
        N = 16
        mon = EMMonitor(N=N)
        E, B = make_plane_wave(N=N)
        mon.set_initial(E, B)
        for _ in range(10):
            mon.check(E, B)
        summary = mon.summary()
        for name, data in summary.items():
            assert data["frac_var"] < 1e-10, f"{name} drifted: frac_var={data['frac_var']}"


class TestEMMonitorCorruption:
    def test_detects_energy_corruption(self):
        """Adding noise to B should change energy."""
        N = 16
        mon = EMMonitor(N=N, threshold=1e-4)
        E, B = make_gaussian_packet(N=N)
        mon.set_initial(E, B)

        # Corrupt B field
        Bx, By, Bz = B
        noise = 0.1 * np.random.randn(*Bx.shape)
        B_bad = (Bx + noise, By + noise, Bz)
        report = mon.check(E, B_bad)
        assert len(report.alerts) > 0
        assert "energy" in report.alerts

    def test_detects_chirality_corruption(self):
        """Flipping handedness should change chirality."""
        N = 16
        mon = EMMonitor(N=N, threshold=1e-4)
        E, B = make_gaussian_packet(N=N)
        mon.set_initial(E, B)

        # Flip Ey sign → changes polarization handedness
        Ex, Ey, Ez = E
        E_flipped = (Ex, -Ey, Ez)
        report = mon.check(E_flipped, B)
        assert report.worst_drift > 1e-3


class TestEMMonitorSummary:
    def test_summary_has_all_keys(self):
        N = 16
        mon = EMMonitor(N=N)
        E, B = make_plane_wave(N=N)
        mon.set_initial(E, B)
        mon.check(E, B)
        summary = mon.summary()
        expected_keys = {"energy", "momentum", "chirality", "helicity",
                         "zilch_vector", "super_energy", "enstrophy"}
        assert set(summary.keys()) == expected_keys
        for name, data in summary.items():
            assert "initial" in data
            assert "final" in data
            assert "frac_var" in data

    def test_summary_n_samples(self):
        N = 16
        mon = EMMonitor(N=N)
        E, B = make_plane_wave(N=N)
        mon.set_initial(E, B)
        for _ in range(5):
            mon.check(E, B)
        summary = mon.summary()
        assert summary["energy"]["n_samples"] == 6  # 1 initial + 5 checks


class TestEMMonitorNonZeroChirality:
    """Verify that circular polarization gives non-zero chirality/helicity."""

    def test_circular_polarization_has_chirality(self):
        N = 32
        mon = EMMonitor(N=N)
        E, B = make_gaussian_packet(N=N, sigma=0.6, k_z=4.0)
        mon.set_initial(E, B)
        # Chirality should be non-zero for circularly polarized light
        assert abs(mon._initial["chirality"]) > 1e-3

    def test_circular_polarization_has_helicity(self):
        N = 32
        mon = EMMonitor(N=N)
        E, B = make_gaussian_packet(N=N, sigma=0.6, k_z=4.0)
        mon.set_initial(E, B)
        assert abs(mon._initial["helicity"]) > 1e-3
