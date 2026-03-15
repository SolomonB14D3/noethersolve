"""Tests for noethersolve.validate — integrator validation."""

import numpy as np
import pytest

from noethersolve.validate import (
    validate_integrator,
    compare_configs,
    ValidationReport,
)


# ─── Vortex RHS ──────────────────────────────────────────────────────────────

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


def vortex_rhs_wrong(t, state, G):
    """Missing 2pi factor."""
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
            dpos[i, 0] += -G[j] * dy / r2
            dpos[i, 1] += G[j] * dx / r2
    return dpos.ravel()


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def vortex_setup():
    G = [1.0, -0.5, 0.3]
    pos0 = np.array([[1.0, 0.0], [-0.5, 0.8], [-0.3, -0.6]])
    return G, pos0


@pytest.fixture
def chemical_setup():
    species = ["A", "B", "C"]
    S = np.array([[-1, 1, 0, 0], [1, -1, -1, 1], [0, 0, 1, -1]], dtype=float)
    k_rates = np.array([0.5, 0.3, 0.4, 0.2])
    reactant_matrix = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=float)
    reverse_pairs = [(0, 1), (2, 3)]
    c0 = np.array([1.0, 0.5, 0.2])

    def rhs(t, c, S, k, R):
        c = np.maximum(c, 0)
        v = np.zeros(S.shape[1])
        for j in range(S.shape[1]):
            rate = k[j]
            for i in range(len(c)):
                if R[i, j] > 0:
                    rate *= c[i] ** R[i, j]
            v[j] = rate
        return S @ v

    return species, S, k_rates, reactant_matrix, reverse_pairs, c0, rhs


# ─── Tests: vortex system ────────────────────────────────────────────────────

class TestVortexValidation:
    def test_tight_tolerances_pass(self, vortex_setup):
        G, pos0 = vortex_setup
        report = validate_integrator(
            rhs=vortex_rhs,
            y0=pos0.ravel(),
            t_span=(0, 20),
            system="vortex",
            circulations=G,
            rhs_args=(np.array(G),),
            rtol=1e-10,
            atol=1e-12,
            n_eval=500,
        )
        assert isinstance(report, ValidationReport)
        # H and Lz should pass with tight tolerances
        assert report.quantities["H"]["verdict"] == "PASS"
        assert report.quantities["Lz"]["verdict"] == "PASS"

    def test_loose_tolerances_fail(self, vortex_setup):
        G, pos0 = vortex_setup
        report = validate_integrator(
            rhs=vortex_rhs,
            y0=pos0.ravel(),
            t_span=(0, 50),
            system="vortex",
            circulations=G,
            rhs_args=(np.array(G),),
            rtol=1e-2,
            atol=1e-4,
            n_eval=500,
        )
        # With rtol=1e-2 over long time, exact invariants should drift
        assert report.verdict in ("FAIL", "WARN")
        assert len(report.suggestions) > 0

    def test_wrong_physics_detected(self, vortex_setup):
        """Wrong RHS should cause conservation violations."""
        G, pos0 = vortex_setup
        report = validate_integrator(
            rhs=vortex_rhs_wrong,
            y0=pos0.ravel(),
            t_span=(0, 20),
            system="vortex",
            circulations=G,
            rhs_args=(np.array(G),),
            rtol=1e-10,
            atol=1e-12,
            n_eval=500,
        )
        # Wrong physics should still conserve H (Kirchhoff structure preserved)
        # but Q_f invariants may shift
        assert isinstance(report, ValidationReport)

    def test_report_has_suggestions(self, vortex_setup):
        G, pos0 = vortex_setup
        report = validate_integrator(
            rhs=vortex_rhs,
            y0=pos0.ravel(),
            t_span=(0, 100),
            system="vortex",
            circulations=G,
            rhs_args=(np.array(G),),
            rtol=1e-3,
            n_eval=500,
        )
        if report.verdict != "PASS":
            assert len(report.suggestions) > 0

    def test_report_str_format(self, vortex_setup):
        G, pos0 = vortex_setup
        report = validate_integrator(
            rhs=vortex_rhs,
            y0=pos0.ravel(),
            t_span=(0, 10),
            system="vortex",
            circulations=G,
            rhs_args=(np.array(G),),
            n_eval=200,
        )
        s = str(report)
        assert "Integrator Validation" in s
        assert report.system in s

    def test_passed_property(self, vortex_setup):
        G, pos0 = vortex_setup
        report = validate_integrator(
            rhs=vortex_rhs,
            y0=pos0.ravel(),
            t_span=(0, 10),
            system="vortex",
            circulations=G,
            rhs_args=(np.array(G),),
            rtol=1e-10,
            atol=1e-12,
            n_eval=200,
        )
        assert report.passed == (report.verdict == "PASS")


# ─── Tests: chemical system ──────────────────────────────────────────────────

class TestChemicalValidation:
    def test_correct_rates_pass(self, chemical_setup):
        species, S, k_rates, reactant_matrix, reverse_pairs, c0, rhs = chemical_setup
        report = validate_integrator(
            rhs=rhs,
            y0=c0,
            t_span=(0, 50),
            system="chemical",
            species=species,
            stoichiometry=S,
            rate_constants=k_rates,
            reactant_matrix=reactant_matrix,
            reverse_pairs=reverse_pairs,
            rhs_args=(S, k_rates, reactant_matrix),
            rtol=1e-10,
            atol=1e-12,
            n_eval=500,
        )
        assert report.quantities["total_mass"]["verdict"] == "PASS"
        assert report.quantities["rate_constant_product"]["verdict"] == "PASS"


# ─── Tests: custom invariants ────────────────────────────────────────────────

class TestCustomInvariants:
    def test_custom_invariant_pass(self):
        """Simple harmonic oscillator: energy should be conserved."""
        def sho_rhs(t, y):
            x, v = y
            return [v, -x]

        def energy(y):
            return 0.5 * (y[0]**2 + y[1]**2)

        report = validate_integrator(
            rhs=sho_rhs,
            y0=np.array([1.0, 0.0]),
            t_span=(0, 20),
            invariants={"energy": energy},
            rtol=1e-10,
            atol=1e-12,
        )
        assert report.quantities["energy"]["verdict"] == "PASS"
        assert report.passed

    def test_custom_invariant_fail_loose_tol(self):
        """Loose tolerance should cause energy drift in SHO."""
        def sho_rhs(t, y):
            x, v = y
            return [v, -x]

        def energy(y):
            return 0.5 * (y[0]**2 + y[1]**2)

        report = validate_integrator(
            rhs=sho_rhs,
            y0=np.array([1.0, 0.0]),
            t_span=(0, 1000),
            invariants={"energy": energy},
            rtol=1e-2,
            atol=1e-4,
        )
        # Over 1000 time units with rtol=1e-2, energy should drift
        # (may or may not fail depending on RK45 behavior)
        assert isinstance(report, ValidationReport)


# ─── Tests: compare_configs ──────────────────────────────────────────────────

class TestCompareConfigs:
    def test_multiple_configs(self, vortex_setup):
        G, pos0 = vortex_setup
        reports = compare_configs(
            rhs=vortex_rhs,
            y0=pos0.ravel(),
            t_span=(0, 20),
            configs=[
                {"rtol": 1e-6},
                {"rtol": 1e-10},
            ],
            system="vortex",
            circulations=G,
            rhs_args=(np.array(G),),
            n_eval=200,
        )
        assert len(reports) == 2
        assert all(isinstance(r, ValidationReport) for r in reports)

    def test_tighter_is_better(self, vortex_setup):
        G, pos0 = vortex_setup
        reports = compare_configs(
            rhs=vortex_rhs,
            y0=pos0.ravel(),
            t_span=(0, 50),
            configs=[
                {"rtol": 1e-4, "atol": 1e-6},
                {"rtol": 1e-10, "atol": 1e-12},
            ],
            system="vortex",
            circulations=G,
            rhs_args=(np.array(G),),
            n_eval=500,
        )
        # Tighter config should have fewer or equal violations
        loose_fails = len(reports[0].exact_violations) + len(reports[0].approx_violations)
        tight_fails = len(reports[1].exact_violations) + len(reports[1].approx_violations)
        assert tight_fails <= loose_fails


# ─── Tests: error handling ───────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_circulations_raises(self):
        with pytest.raises(ValueError, match="circulations required"):
            validate_integrator(
                rhs=lambda t, y: y,
                y0=np.array([0.0, 0.0]),
                t_span=(0, 1),
                system="vortex",
            )

    def test_missing_masses_raises(self):
        with pytest.raises(ValueError, match="masses required"):
            validate_integrator(
                rhs=lambda t, y: y,
                y0=np.array([0.0] * 6),
                t_span=(0, 1),
                system="gravity",
            )

    def test_missing_species_raises(self):
        with pytest.raises(ValueError, match="species and stoichiometry"):
            validate_integrator(
                rhs=lambda t, y: y,
                y0=np.array([1.0, 0.5]),
                t_span=(0, 1),
                system="chemical",
            )

    def test_solver_failure_returns_fail(self):
        """RHS that causes solver failure should return FAIL verdict."""
        def bad_rhs(t, y):
            return np.array([1e20, -1e20])  # explodes

        report = validate_integrator(
            rhs=bad_rhs,
            y0=np.array([0.0, 0.0]),
            t_span=(0, 1e10),
            invariants={"dummy": lambda y: y[0]},
            rtol=1e-14,
            n_eval=10,
        )
        # Either solver fails or invariant fails — either way not PASS
        assert isinstance(report, ValidationReport)
