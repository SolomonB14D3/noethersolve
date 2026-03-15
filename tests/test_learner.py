"""Tests for noethersolve.learner — automatic conservation law discovery."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from noethersolve.learner import (
    InvariantLearner,
    LearnerReport,
    evaluate_basis,
    evaluate_f,
    BASIS_NAMES,
    N_BASIS,
)


# ─── Basis function tests ────────────────────────────────────────────────────

class TestBasisFunctions:
    def test_evaluate_basis_length(self):
        r = np.array([0.5, 1.0, 2.0])
        bases = evaluate_basis(r)
        assert len(bases) == N_BASIS

    def test_evaluate_basis_positive(self):
        """All basis functions should return finite values."""
        r = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        bases = evaluate_basis(r, regularize=0.01)
        for i, b in enumerate(bases):
            assert np.all(np.isfinite(b)), f"Basis {BASIS_NAMES[i]} has non-finite values"

    def test_evaluate_f_linear_combination(self):
        r = np.array([1.0])
        params = np.zeros(N_BASIS)
        params[4] = 2.0  # 2 * e^(-r)
        result = evaluate_f(params, r, regularize=0.01)
        expected = 2.0 * np.exp(-(1.0 + 0.01))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_evaluate_f_zero_params(self):
        r = np.array([1.0, 2.0])
        params = np.zeros(N_BASIS)
        result = evaluate_f(params, r)
        np.testing.assert_allclose(result, 0.0)


# ─── Helper: generate vortex trajectory ──────────────────────────────────────

def vortex_trajectory(G, pos0, T=10.0, n_steps=200):
    """Generate a 2D point-vortex trajectory, returning pairwise distances."""
    N = len(G)

    def rhs(t, state):
        pos = state.reshape(N, 2)
        dpos = np.zeros_like(pos)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                r2 = dx ** 2 + dy ** 2 + 1e-20
                dpos[i, 0] += -G[j] * dy / (2 * np.pi * r2)
                dpos[i, 1] += G[j] * dx / (2 * np.pi * r2)
        return dpos.ravel()

    sol = solve_ivp(rhs, (0, T), pos0.ravel(),
                    t_eval=np.linspace(0, T, n_steps),
                    rtol=1e-10, atol=1e-12)

    # Extract pairwise distances at each step
    dist_traj = []
    for k in range(sol.y.shape[1]):
        pos = sol.y[:, k].reshape(N, 2)
        dists = {}
        for i in range(N):
            for j in range(i + 1, N):
                r = np.linalg.norm(pos[i] - pos[j])
                dists[f"r{i+1}{j+1}"] = r
        dist_traj.append(dists)

    # Also return position trajectory
    pos_traj = np.array([sol.y[:, k].reshape(N, 2) for k in range(sol.y.shape[1])])

    return dist_traj, pos_traj


# ─── InvariantLearner tests ──────────────────────────────────────────────────

class TestLearnerFromDistances:
    @pytest.fixture
    def vortex_data(self):
        G = [1.0, -0.5, 0.3]
        pos0 = np.array([[1.0, 0.0], [-0.5, 0.8], [-0.3, -0.6]])
        dist_traj, pos_traj = vortex_trajectory(G, pos0, T=5.0, n_steps=100)
        return G, dist_traj, pos_traj

    def test_returns_report(self, vortex_data):
        G, dist_traj, _ = vortex_data
        learner = InvariantLearner(maxiter=10)
        result = learner.learn_from_distances([dist_traj], G)
        assert isinstance(result, LearnerReport)

    def test_improves_over_initial(self, vortex_data):
        G, dist_traj, _ = vortex_data
        learner = InvariantLearner(maxiter=50)
        result = learner.learn_from_distances([dist_traj], G)
        assert result.improvement_pct >= 0

    def test_has_coefficients(self, vortex_data):
        G, dist_traj, _ = vortex_data
        learner = InvariantLearner(maxiter=10)
        result = learner.learn_from_distances([dist_traj], G)
        assert len(result.coefficients) == N_BASIS
        for name in BASIS_NAMES:
            assert name in result.coefficients

    def test_has_individual_losses(self, vortex_data):
        G, dist_traj, _ = vortex_data
        learner = InvariantLearner(maxiter=10)
        result = learner.learn_from_distances([dist_traj], G)
        assert len(result.individual_losses) == N_BASIS
        assert result.best_single_basis in BASIS_NAMES

    def test_report_str(self, vortex_data):
        G, dist_traj, _ = vortex_data
        learner = InvariantLearner(maxiter=10)
        result = learner.learn_from_distances([dist_traj], G)
        s = str(result)
        assert "Invariant Learner" in s
        assert "Improvement" in s

    def test_formula_generated(self, vortex_data):
        G, dist_traj, _ = vortex_data
        learner = InvariantLearner(maxiter=10)
        result = learner.learn_from_distances([dist_traj], G)
        assert len(result.formula) > 0


class TestLearnerFromPositions:
    def test_from_positions(self):
        G = [1.0, -0.5, 0.3]
        pos0 = np.array([[1.0, 0.0], [-0.5, 0.8], [-0.3, -0.6]])
        _, pos_traj = vortex_trajectory(G, pos0, T=5.0, n_steps=100)
        learner = InvariantLearner(maxiter=10)
        result = learner.learn_from_positions([pos_traj], G)
        assert isinstance(result, LearnerReport)
        assert result.final_loss < 1.0  # should find something decent


class TestLearnerFromField:
    def test_from_field_basic(self):
        """Test with simple vortex-like field snapshots."""
        N = 32
        L = 2 * np.pi
        dx = L / N
        x = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, x)

        # Two Gaussian vortices — just slightly perturbed
        def make_field(offset=0.0):
            sigma = 0.3
            w = (np.exp(-((X - L/3 - offset)**2 + (Y - L/2)**2) / (2 * sigma**2)) +
                 np.exp(-((X - 2*L/3 + offset)**2 + (Y - L/2)**2) / (2 * sigma**2)))
            return w

        snapshots = [make_field(o) for o in np.linspace(0, 0.1, 10)]

        learner = InvariantLearner(maxiter=10)
        result = learner.learn_from_field(snapshots, dx=dx, L=L)
        assert isinstance(result, LearnerReport)
        assert len(result.coefficients) == N_BASIS


class TestLearnerMultipleTrajectories:
    def test_two_trajectories(self):
        """Optimizer should handle multiple trajectories."""
        G = [1.0, -0.5, 0.3]
        pos0_a = np.array([[1.0, 0.0], [-0.5, 0.8], [-0.3, -0.6]])
        pos0_b = np.array([[0.5, 0.5], [-0.3, -0.5], [0.8, -0.2]])
        dist_a, _ = vortex_trajectory(G, pos0_a, T=3.0, n_steps=50)
        dist_b, _ = vortex_trajectory(G, pos0_b, T=3.0, n_steps=50)

        learner = InvariantLearner(maxiter=50)
        result = learner.learn_from_distances([dist_a, dist_b], G)
        assert isinstance(result, LearnerReport)
        # L-BFGS-B may not converge in limited iterations but should improve
        assert result.final_loss <= result.initial_loss


class TestLearnerCustomGuess:
    def test_custom_initial_guess(self):
        G = [1.0, -0.5, 0.3]
        pos0 = np.array([[1.0, 0.0], [-0.5, 0.8], [-0.3, -0.6]])
        dist_traj, _ = vortex_trajectory(G, pos0, T=3.0, n_steps=50)

        # Start from -ln(r) instead of e^(-r)
        guess = np.zeros(N_BASIS)
        guess[7] = 1.0  # -ln(r)

        learner = InvariantLearner(maxiter=20)
        result = learner.learn_from_distances([dist_traj], G,
                                              initial_guess=guess)
        assert isinstance(result, LearnerReport)
