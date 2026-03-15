"""
noethersolve.learner — Automatic conservation law discovery.

Given a trajectory (time series of states), discovers new conserved
quantities by optimizing over a basis of functions. Uses gradient-free
optimization (L-BFGS-B) to find the combination of basis functions
that minimizes fractional variation along the trajectory.

Works for any system — feed it a trajectory and it tells you what's
conserved that you might not have known about.

Usage:
    from noethersolve import InvariantLearner

    # Discover conserved quantities from a vortex trajectory
    learner = InvariantLearner()
    result = learner.learn_from_distances(
        trajectories=[trajectory1, trajectory2],
        weights=[1.0, -0.5, 0.3],  # vortex circulations
    )
    print(result)
    # Shows: optimal f(r) coefficients, conservation improvement, dominant terms

    # Or provide arbitrary observable time series
    result = learner.learn_from_observables(
        observables_list=[
            {"r12": r12_series, "r13": r13_series, "r23": r23_series},
        ],
        weights=[1.0, -0.5, 0.3],
    )

Basis functions (12):
    √r, r, r^1.5, r², e^(-r), e^(-r/2), e^(-2r),
    -ln(r), ln(1+r), tanh(r), sin(r), 1/r
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from scipy.optimize import minimize

from noethersolve.monitor import frac_var


# ─── Basis functions ──────────────────────────────────────────────────────────

BASIS_NAMES = [
    "√r", "r", "r^1.5", "r²",
    "e^(-r)", "e^(-r/2)", "e^(-2r)",
    "-ln(r)", "ln(1+r)",
    "tanh(r)", "sin(r)", "1/r",
]

N_BASIS = len(BASIS_NAMES)


def evaluate_basis(r: np.ndarray, regularize: float = 0.01) -> List[np.ndarray]:
    """Evaluate all basis functions at given r values.

    Args:
        r: Array of distance values (any shape).
        regularize: Small offset to avoid singularities at r=0.

    Returns:
        List of arrays, one per basis function.
    """
    r_reg = r + regularize
    return [
        np.sqrt(r_reg),         # 0: √r
        r_reg,                   # 1: r
        r_reg ** 1.5,            # 2: r^1.5
        r_reg ** 2,              # 3: r²
        np.exp(-r_reg),          # 4: e^(-r)
        np.exp(-r_reg / 2),      # 5: e^(-r/2)
        np.exp(-2 * r_reg),      # 6: e^(-2r)
        -np.log(r_reg),          # 7: -ln(r)
        np.log(1 + r_reg),       # 8: ln(1+r)
        np.tanh(r_reg),          # 9: tanh(r)
        np.sin(r_reg),           # 10: sin(r)
        1.0 / r_reg,             # 11: 1/r
    ]


def evaluate_f(params: np.ndarray, r: np.ndarray,
               regularize: float = 0.01) -> np.ndarray:
    """Evaluate f(r) = Σᵢ aᵢ φᵢ(r) for given parameters."""
    bases = evaluate_basis(r, regularize)
    f = np.zeros_like(r, dtype=np.float64)
    for i, basis in enumerate(bases):
        if i < len(params):
            f += params[i] * basis
    return f


# ─── Report ───────────────────────────────────────────────────────────────────

@dataclass
class LearnerReport:
    """Result of InvariantLearner.learn_*()."""
    success: bool
    initial_loss: float             # Loss with the starting guess
    final_loss: float               # Loss after optimization
    improvement_pct: float          # (1 - final/initial) * 100
    coefficients: Dict[str, float]  # basis name -> coefficient
    dominant_terms: List[str]       # basis names with |coef| > 0.01
    individual_losses: Dict[str, float]  # loss using each basis alone
    best_single_basis: str          # name of the single best basis
    formula: str                    # human-readable f(r) formula

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Invariant Learner: {'SUCCESS' if self.success else 'FAILED'}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Initial loss (starting guess): {self.initial_loss:.6f}")
        lines.append(f"  Final loss (optimized):        {self.final_loss:.6f}")
        lines.append(f"  Improvement:                   {self.improvement_pct:.1f}%")
        lines.append(f"")
        lines.append(f"  Optimal f(r) = {self.formula}")
        lines.append(f"")
        lines.append(f"  Dominant terms:")
        for name in self.dominant_terms:
            coef = self.coefficients[name]
            lines.append(f"    {coef:+.4f} × {name}")
        lines.append(f"")
        lines.append(f"  Individual basis losses (lower = better):")
        sorted_bases = sorted(self.individual_losses.items(), key=lambda x: x[1])
        for name, loss in sorted_bases[:5]:
            marker = " ← best single" if name == self.best_single_basis else ""
            lines.append(f"    {name:<12s}  loss={loss:.6f}{marker}")
        if len(sorted_bases) > 5:
            lines.append(f"    ... ({len(sorted_bases) - 5} more)")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


# ─── InvariantLearner ─────────────────────────────────────────────────────────

class InvariantLearner:
    """Discover conserved quantities by optimizing over basis functions.

    The learner searches for f(r) = Σᵢ aᵢ φᵢ(r) such that

        Q_f = Σᵢ<ⱼ wᵢ wⱼ f(rᵢⱼ)

    has minimal variation along one or more trajectories. Here wᵢ are
    weights (e.g., vortex circulations, masses) and rᵢⱼ are pairwise
    distances.

    Args:
        regularize: Small offset to avoid singularities.
        l2_penalty: L2 regularization on coefficients.
        maxiter: Maximum optimization iterations.
    """

    def __init__(self, regularize: float = 0.01, l2_penalty: float = 0.001,
                 maxiter: int = 200):
        self.regularize = regularize
        self.l2_penalty = l2_penalty
        self.maxiter = maxiter

    def _compute_Qf(self, params: np.ndarray,
                    distances_series: List[Dict[str, np.ndarray]],
                    weights: List[float]) -> List[List[float]]:
        """Compute Q_f time series for each trajectory.

        Args:
            params: Coefficient vector (length N_BASIS).
            distances_series: List of trajectories. Each trajectory is a
                list of dicts mapping pair names to distance values.
            weights: Interaction weights (circulations, masses, etc.).

        Returns:
            List of Q_f time series (one per trajectory).
        """
        N = len(weights)
        w = np.asarray(weights)

        # Build pair weight products: w_ij = w_i * w_j for i < j
        pair_weights = {}
        pair_names = []
        for i in range(N):
            for j in range(i + 1, N):
                pname = f"r{i}{j}" if f"r{i}{j}" in distances_series[0][0] else f"r{i+1}{j+1}"
                # Try both naming conventions
                for candidate in [f"r{i}{j}", f"r{i+1}{j+1}", f"r_{i}_{j}", f"r_{i+1}_{j+1}"]:
                    if candidate in distances_series[0][0]:
                        pname = candidate
                        break
                pair_weights[pname] = w[i] * w[j]
                pair_names.append(pname)

        all_Qf = []
        for traj in distances_series:
            Qf_series = []
            for step_distances in traj:
                Qf = 0.0
                for pname in pair_names:
                    if pname in step_distances:
                        r = step_distances[pname]
                        f_val = evaluate_f(params, np.array([r]),
                                           self.regularize)[0]
                        Qf += pair_weights[pname] * f_val
                Qf_series.append(Qf)
            all_Qf.append(Qf_series)

        return all_Qf

    def _loss(self, params, distances_series, weights):
        """Total loss: sum of frac_var over all trajectories + L2 penalty."""
        all_Qf = self._compute_Qf(params, distances_series, weights)
        total = 0.0
        for Qf_series in all_Qf:
            total += frac_var(Qf_series)
        total += self.l2_penalty * np.sum(params ** 2)
        return total

    def learn_from_distances(
        self,
        trajectories: List[List[Dict[str, float]]],
        weights: List[float],
        initial_guess: Optional[np.ndarray] = None,
    ) -> LearnerReport:
        """Find the optimal f(r) from pairwise distance trajectories.

        Args:
            trajectories: List of trajectories. Each trajectory is a list
                of dicts mapping pair names (e.g., "r12", "r13", "r23")
                to distance values at each timestep.
            weights: Interaction weights (one per particle/vortex/body).
            initial_guess: Starting coefficients. Default: start with
                e^(-r) (known good for vortex systems).

        Returns:
            LearnerReport with optimal coefficients and comparison.
        """
        if initial_guess is None:
            initial_guess = np.zeros(N_BASIS)
            initial_guess[4] = 1.0  # e^(-r)

        initial_loss = self._loss(initial_guess, trajectories, weights)

        # Optimize
        result = minimize(
            self._loss,
            initial_guess,
            args=(trajectories, weights),
            method='L-BFGS-B',
            options={'maxiter': self.maxiter},
        )

        optimal = result.x
        final_loss = result.fun
        improvement = (1 - final_loss / initial_loss) * 100 if initial_loss > 0 else 0.0

        # Coefficients
        coefficients = {name: float(optimal[i]) for i, name in enumerate(BASIS_NAMES)}

        # Dominant terms (|coef| > 0.01)
        dominant = [name for name, c in coefficients.items() if abs(c) > 0.01]
        dominant.sort(key=lambda n: -abs(coefficients[n]))

        # Individual basis losses
        individual = {}
        for i, name in enumerate(BASIS_NAMES):
            single = np.zeros(N_BASIS)
            single[i] = 1.0
            individual[name] = self._loss(single, trajectories, weights)

        best_single = min(individual, key=individual.get)

        # Human-readable formula
        formula_parts = []
        sorted_idx = np.argsort(-np.abs(optimal))
        for i in sorted_idx:
            c = optimal[i]
            if abs(c) > 1e-4:
                sign = "+" if c > 0 else "-"
                if not formula_parts:
                    formula_parts.append(f"{c:.4f}·{BASIS_NAMES[i]}")
                else:
                    formula_parts.append(f"{sign} {abs(c):.4f}·{BASIS_NAMES[i]}")
        formula = " ".join(formula_parts) if formula_parts else "0"

        return LearnerReport(
            success=result.success,
            initial_loss=initial_loss,
            final_loss=final_loss,
            improvement_pct=improvement,
            coefficients=coefficients,
            dominant_terms=dominant,
            individual_losses=individual,
            best_single_basis=best_single,
            formula=formula,
        )

    def learn_from_positions(
        self,
        position_trajectories: List[np.ndarray],
        weights: List[float],
        initial_guess: Optional[np.ndarray] = None,
    ) -> LearnerReport:
        """Find optimal f(r) from position trajectories directly.

        More convenient than learn_from_distances — computes pairwise
        distances automatically.

        Args:
            position_trajectories: List of trajectories. Each trajectory
                is an array of shape (n_steps, N, dim) giving positions
                at each timestep.
            weights: Interaction weights (length N).
            initial_guess: Starting coefficients.

        Returns:
            LearnerReport with optimal coefficients.
        """
        N = len(weights)
        all_distance_trajs = []

        for pos_traj in position_trajectories:
            dist_traj = []
            for step in range(len(pos_traj)):
                pos = pos_traj[step]  # (N, dim)
                dists = {}
                for i in range(N):
                    for j in range(i + 1, N):
                        r = np.linalg.norm(pos[i] - pos[j])
                        dists[f"r{i+1}{j+1}"] = r
                dist_traj.append(dists)
            all_distance_trajs.append(dist_traj)

        return self.learn_from_distances(
            all_distance_trajs, weights, initial_guess)

    def learn_from_field(
        self,
        field_snapshots: List[np.ndarray],
        dx: float,
        L: float,
        initial_guess: Optional[np.ndarray] = None,
    ) -> LearnerReport:
        """Find optimal f(r) for continuous field Q_f conservation.

        For continuous vorticity/density fields:
            Q_f[ω] = ∫∫ ω(x) ω(y) f(|x-y|) dx dy

        Uses convolution via FFT for efficiency.

        Args:
            field_snapshots: List of 2D field arrays (one per timestep).
            dx: Grid spacing.
            L: Domain length (periodic).
            initial_guess: Starting coefficients.

        Returns:
            LearnerReport with optimal coefficients.
        """
        from numpy.fft import fft2, ifft2

        if initial_guess is None:
            initial_guess = np.zeros(N_BASIS)
            initial_guess[4] = 1.0  # e^(-r)

        N_grid = field_snapshots[0].shape[0]
        x = np.linspace(0, L, N_grid, endpoint=False)
        X, Y = np.meshgrid(x, x)
        # Periodic distance
        Rx = np.minimum(X, L - X)
        Ry = np.minimum(Y, L - Y)
        R = np.sqrt(Rx ** 2 + Ry ** 2)

        def field_loss(params):
            f_kernel = evaluate_f(params, R, self.regularize)
            f_hat = fft2(f_kernel)
            Qf_values = []
            for omega in field_snapshots:
                omega_hat = fft2(omega)
                conv = np.real(ifft2(omega_hat * f_hat))
                Q = np.sum(omega * conv) * dx ** 4
                Qf_values.append(Q)
            return frac_var(Qf_values) + self.l2_penalty * np.sum(params ** 2)

        initial_loss = field_loss(initial_guess)

        result = minimize(
            field_loss, initial_guess,
            method='L-BFGS-B',
            options={'maxiter': self.maxiter},
        )

        optimal = result.x
        final_loss = result.fun
        improvement = (1 - final_loss / initial_loss) * 100 if initial_loss > 0 else 0.0

        coefficients = {name: float(optimal[i]) for i, name in enumerate(BASIS_NAMES)}
        dominant = [n for n, c in coefficients.items() if abs(c) > 0.01]
        dominant.sort(key=lambda n: -abs(coefficients[n]))

        # Individual losses
        individual = {}
        for i, name in enumerate(BASIS_NAMES):
            single = np.zeros(N_BASIS)
            single[i] = 1.0
            individual[name] = field_loss(single)
        best_single = min(individual, key=individual.get)

        # Formula
        formula_parts = []
        sorted_idx = np.argsort(-np.abs(optimal))
        for i in sorted_idx:
            c = optimal[i]
            if abs(c) > 1e-4:
                sign = "+" if c > 0 else "-"
                if not formula_parts:
                    formula_parts.append(f"{c:.4f}·{BASIS_NAMES[i]}")
                else:
                    formula_parts.append(f"{sign} {abs(c):.4f}·{BASIS_NAMES[i]}")
        formula = " ".join(formula_parts) if formula_parts else "0"

        return LearnerReport(
            success=result.success,
            initial_loss=initial_loss,
            final_loss=final_loss,
            improvement_pct=improvement,
            coefficients=coefficients,
            dominant_terms=dominant,
            individual_losses=individual,
            best_single_basis=best_single,
            formula=formula,
        )
