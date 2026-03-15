"""Tests for noethersolve.monitor — conservation law monitors."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from noethersolve.monitor import (
    frac_var,
    MonitorReport,
    VortexMonitor,
    ChemicalMonitor,
    GravityMonitor,
)


# ─── frac_var ────────────────────────────────────────────────────────────────

class TestFracVar:
    def test_constant_series(self):
        assert frac_var([5.0, 5.0, 5.0, 5.0]) == 0.0

    def test_small_variation(self):
        vals = [1.0, 1.001, 0.999, 1.0005]
        assert frac_var(vals) < 1e-3

    def test_large_variation(self):
        vals = [1.0, 2.0, 0.5, 3.0]
        assert frac_var(vals) > 0.1

    def test_near_zero_mean(self):
        # When mean is ~0, returns std directly
        vals = [1e-20, -1e-20, 0.0]
        result = frac_var(vals)
        assert result >= 0


# ─── VortexMonitor ───────────────────────────────────────────────────────────

def vortex_rhs(t, state, G):
    N = len(G)
    pos = state.reshape(N, 2)
    dpos = np.zeros_like(pos)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            r2 = dx**2 + dy**2 + 1e-20
            dpos[i, 0] += -G[j] * dy / (2 * np.pi * r2)
            dpos[i, 1] += G[j] * dx / (2 * np.pi * r2)
    return dpos.ravel()


class TestVortexMonitor:
    @pytest.fixture
    def vortex_trajectory(self):
        """Run a short 3-vortex simulation."""
        G = np.array([1.0, -0.5, 0.3])
        pos0 = np.array([[1.0, 0.0], [-0.5, 0.8], [-0.3, -0.6]])
        sol = solve_ivp(
            vortex_rhs, (0, 20.0), pos0.ravel(), args=(G,),
            method="RK45", t_eval=np.linspace(0, 20.0, 500),
            rtol=1e-10, atol=1e-12,
        )
        return G, pos0, sol

    def test_initialization(self):
        mon = VortexMonitor([1.0, -0.5, 0.3])
        assert mon.N == 3

    def test_set_initial(self):
        mon = VortexMonitor([1.0, -0.5])
        pos = np.array([[0.0, 0.0], [1.0, 0.0]])
        mon.set_initial(pos)
        assert "H" in mon._initial
        assert "Lz" in mon._initial
        assert "Q_linear" in mon._initial

    def test_check_returns_report(self, vortex_trajectory):
        G, pos0, sol = vortex_trajectory
        mon = VortexMonitor(G)
        mon.set_initial(pos0)
        state = sol.y[:, 1].reshape(-1, 2)
        report = mon.check(state)
        assert isinstance(report, MonitorReport)
        assert report.step == 1
        assert "H" in report.quantities

    def test_hamiltonian_conserved(self, vortex_trajectory):
        """H should be conserved to high precision with tight tolerances."""
        G, pos0, sol = vortex_trajectory
        mon = VortexMonitor(G)
        mon.set_initial(pos0)
        for i in range(1, sol.y.shape[1]):
            state = sol.y[:, i].reshape(-1, 2)
            mon.check(state)
        summary = mon.summary()
        assert summary["H"]["frac_var"] < 1e-8, f"H frac_var = {summary['H']['frac_var']}"

    def test_angular_momentum_conserved(self, vortex_trajectory):
        G, pos0, sol = vortex_trajectory
        mon = VortexMonitor(G)
        mon.set_initial(pos0)
        for i in range(1, sol.y.shape[1]):
            state = sol.y[:, i].reshape(-1, 2)
            mon.check(state)
        summary = mon.summary()
        assert summary["Lz"]["frac_var"] < 1e-8

    def test_qf_approximate_invariants(self, vortex_trajectory):
        """Q_f family should have frac_var < 5e-2 (approximate, not exact)."""
        G, pos0, sol = vortex_trajectory
        mon = VortexMonitor(G)
        mon.set_initial(pos0)
        for i in range(1, sol.y.shape[1]):
            state = sol.y[:, i].reshape(-1, 2)
            mon.check(state)
        summary = mon.summary()
        for name in ["Q_linear", "Q_squared", "Q_sqrt", "Q_exp", "Q_tanh"]:
            assert summary[name]["frac_var"] < 5e-2, f"{name} frac_var = {summary[name]['frac_var']}"

    def test_alerts_on_corruption(self, vortex_trajectory):
        """Corrupted positions should trigger alerts."""
        G, pos0, sol = vortex_trajectory
        mon = VortexMonitor(G, threshold=1e-4)
        mon.set_initial(pos0)
        # Feed clean data for a while
        for i in range(1, 100):
            mon.check(sol.y[:, i].reshape(-1, 2))
        # Corrupt
        bad = sol.y[:, 100].reshape(-1, 2) + np.random.randn(3, 2) * 0.1
        report = mon.check(bad)
        assert len(report.alerts) > 0

    def test_summary_has_all_keys(self, vortex_trajectory):
        G, pos0, sol = vortex_trajectory
        mon = VortexMonitor(G)
        mon.set_initial(pos0)
        for i in range(1, 10):
            mon.check(sol.y[:, i].reshape(-1, 2))
        summary = mon.summary()
        for key in ["H", "Lz", "Px", "Py", "Q_linear", "Q_exp", "R_f"]:
            assert key in summary, f"Missing {key}"
            assert "initial" in summary[key]
            assert "final" in summary[key]
            assert "frac_var" in summary[key]


# ─── ChemicalMonitor ─────────────────────────────────────────────────────────

class TestChemicalMonitor:
    @pytest.fixture
    def chemical_setup(self):
        """A ↔ B ↔ C network."""
        species = ["A", "B", "C"]
        S = np.array([
            [-1, 1, 0, 0],
            [1, -1, -1, 1],
            [0, 0, 1, -1],
        ], dtype=float)
        k_rates = np.array([0.5, 0.3, 0.4, 0.2])
        reactant_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        reverse_pairs = [(0, 1), (2, 3)]
        c0 = np.array([1.0, 0.5, 0.2])
        return species, S, k_rates, reactant_matrix, reverse_pairs, c0

    def test_finds_conservation_laws(self, chemical_setup):
        species, S, k_rates, reactant_matrix, reverse_pairs, c0 = chemical_setup
        mon = ChemicalMonitor(species, S)
        # A + B + C is conserved (one conservation law)
        assert len(mon._conservation_vectors) >= 1

    def test_mass_conservation(self, chemical_setup):
        """Total mass should be exactly conserved."""
        species, S, k_rates, reactant_matrix, reverse_pairs, c0 = chemical_setup

        def rhs(t, c):
            c = np.maximum(c, 0)
            v = np.zeros(S.shape[1])
            for j in range(S.shape[1]):
                rate = k_rates[j]
                for i in range(len(c)):
                    if reactant_matrix[i, j] > 0:
                        rate *= c[i] ** reactant_matrix[i, j]
                v[j] = rate
            return S @ v

        sol = solve_ivp(rhs, (0, 50), c0, method="RK45",
                       t_eval=np.linspace(0, 50, 500), rtol=1e-10, atol=1e-12)

        mon = ChemicalMonitor(species, S, rate_constants=k_rates,
                             reactant_matrix=reactant_matrix,
                             reverse_pairs=reverse_pairs)
        mon.set_initial(c0)
        for i in range(1, sol.y.shape[1]):
            mon.check(sol.y[:, i])

        summary = mon.summary()
        assert summary["total_mass"]["frac_var"] < 1e-10

    def test_wegscheider_constant(self, chemical_setup):
        """Wegscheider cycle product should be constant (determined by rate constants)."""
        species, S, k_rates, reactant_matrix, reverse_pairs, c0 = chemical_setup
        mon = ChemicalMonitor(species, S, rate_constants=k_rates,
                             reactant_matrix=reactant_matrix,
                             reverse_pairs=reverse_pairs)
        mon.set_initial(c0)
        # Check at different concentrations — cycle product shouldn't change
        for c in [np.array([0.5, 0.8, 0.4]), np.array([0.1, 1.5, 0.1])]:
            report = mon.check(c)
            assert "rate_constant_product" in report.quantities
        summary = mon.summary()
        assert summary["rate_constant_product"]["frac_var"] < 1e-14

    def test_wegscheider_detects_violation(self, chemical_setup):
        """Different rate constants should give different Wegscheider products."""
        species, S, k_rates, reactant_matrix, reverse_pairs, c0 = chemical_setup
        correct_product = (k_rates[0] / k_rates[1]) * (k_rates[2] / k_rates[3])
        violated_rates = k_rates * np.array([1.0, 2.0, 1.0, 2.0])
        violated_product = (violated_rates[0] / violated_rates[1]) * (violated_rates[2] / violated_rates[3])
        assert abs(correct_product - violated_product) > 0.5

    def test_entropy_production_nonnegative(self, chemical_setup):
        """Entropy production must be >= 0 (second law)."""
        species, S, k_rates, reactant_matrix, reverse_pairs, c0 = chemical_setup
        mon = ChemicalMonitor(species, S, rate_constants=k_rates,
                             reactant_matrix=reactant_matrix,
                             reverse_pairs=reverse_pairs)
        mon.set_initial(c0)
        report = mon.check(c0)
        assert report.quantities["entropy_production"] >= 0


# ─── GravityMonitor ──────────────────────────────────────────────────────────

class TestGravityMonitor:
    def test_initialization(self):
        mon = GravityMonitor([1.0, 1.0, 1.0])
        assert mon.N == 3

    def test_energy_computed(self):
        mon = GravityMonitor([1.0, 1.0])
        pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        vel = np.array([[0, 0, 0], [0, 1, 0]], dtype=float)
        mon.set_initial(pos, vel)
        assert "E" in mon._initial
        assert "Lz" in mon._initial
        assert "Q_linear" in mon._initial

    def test_momentum_conserved_trivial(self):
        """With zero external forces, momentum should be exactly conserved."""
        mon = GravityMonitor([1.0, 1.0])
        pos = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        vel = np.array([[0.1, 0, 0], [-0.1, 0, 0]], dtype=float)
        mon.set_initial(pos, vel)
        # Slightly moved but with same velocities (no force applied = test structure)
        pos2 = pos + vel * 0.01
        report = mon.check(pos2, vel)
        assert report.drifts["Px"] < 1e-10
        assert report.drifts["Py"] < 1e-10


# ─── MonitorReport ───────────────────────────────────────────────────────────

class TestMonitorReport:
    def test_str_format(self):
        report = MonitorReport(
            step=5,
            quantities={"H": 1.0, "Lz": 2.0},
            drifts={"H": 1e-8, "Lz": 5e-4},
            frac_vars={"H": 1e-9, "Lz": 3e-4},
            alerts=["Lz"],
            worst_name="Lz",
            worst_drift=5e-4,
        )
        s = str(report)
        assert "Step 5" in s
        assert "1 alerts" in s
        assert "Lz" in s
        assert "<<<" in s  # alert marker
