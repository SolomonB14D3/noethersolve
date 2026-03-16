#!/usr/bin/env python3
"""
True Higher-Order Invariants: Searching with Correct Mathematics

LESSON LEARNED: The earlier "triplet invariant" was a false alarm.
- T_full = Σ_{all perms} Γi Γj Γk × Area(ijk) = 0 identically
  (symmetric weights × antisymmetric area = 0)
- T_ordered = Σ_{i<j<k} Γi Γj Γk × Area(ijk) is NOT conserved

This script searches for GENUINE higher-order invariants.

The key insight: for a non-trivial invariant, we need:
1. The quantity to not be identically zero
2. The time derivative dT/dt = 0 for vortex dynamics
"""

import numpy as np
from scipy.integrate import odeint
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def vortex_rhs(state, t, gammas):
    """Point vortex dynamics."""
    N = len(gammas)
    positions = state.reshape(N, 2)
    velocities = np.zeros_like(positions)

    for i in range(N):
        for j in range(N):
            if i != j:
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                r2 = dx**2 + dy**2 + 1e-10
                velocities[i, 0] += -gammas[j] * dy / (2 * np.pi * r2)
                velocities[i, 1] += gammas[j] * dx / (2 * np.pi * r2)

    return velocities.flatten()


def compute_Q_f(positions, gammas, f):
    """Compute pairwise Q_f."""
    N = len(gammas)
    pos = positions.reshape(N, 2)
    Q = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                r = np.sqrt((pos[i,0] - pos[j,0])**2 + (pos[i,1] - pos[j,1])**2)
                Q += gammas[i] * gammas[j] * f(r)
    return Q


def frac_var(x):
    return np.std(x) / (np.abs(np.mean(x)) + 1e-10)


# ============================================================================
# APPROACH 1: Products of known invariants
# ============================================================================

def test_products_of_invariants():
    """
    If Q_f is conserved, then products Q_{f1} * Q_{f2} should also be conserved.
    But these are trivially derived from pairwise invariants.

    Are there IRREDUCIBLE higher-order invariants?
    """
    print("="*60)
    print("APPROACH 1: Products of invariants")
    print("="*60)

    np.random.seed(42)
    N = 5

    theta = np.random.uniform(0, 2*np.pi, N)
    r = np.sqrt(np.random.uniform(0, 1, N)) * 0.8
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    gammas = np.random.randn(N)
    gammas -= gammas.mean()
    gammas /= np.abs(gammas).max()

    t = np.arange(0, 20.0, 0.01)
    trajectory = odeint(vortex_rhs, positions.flatten(), t, args=(gammas,))

    # Known conserved
    f_ln = lambda r: -np.log(r + 1e-10)
    Q_ln = np.array([compute_Q_f(trajectory[i], gammas, f_ln) for i in range(len(t))])

    print(f"  Q_ln: frac_var = {frac_var(Q_ln):.2e}")
    print(f"  Q_ln²: frac_var = {frac_var(Q_ln**2):.2e}")
    print(f"  (Products of conserved quantities are trivially conserved)")


# ============================================================================
# APPROACH 2: Antisymmetric combinations that aren't zero
# ============================================================================

def test_antisymmetric_triplets():
    """
    For non-zero triplet invariants, we need the weight to be ANTISYMMETRIC
    to match the antisymmetric area.

    Try: Σ_{i<j<k} ε_{ijk} × Area(ijk)

    where ε_{ijk} is some antisymmetric weight involving Γ.
    """
    print("\n" + "="*60)
    print("APPROACH 2: Antisymmetric triplet weights")
    print("="*60)

    np.random.seed(42)
    N = 5

    theta = np.random.uniform(0, 2*np.pi, N)
    r = np.sqrt(np.random.uniform(0, 1, N)) * 0.8
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    gammas = np.random.randn(N)
    gammas -= gammas.mean()
    gammas /= np.abs(gammas).max()

    t = np.arange(0, 20.0, 0.01)
    trajectory = odeint(vortex_rhs, positions.flatten(), t, args=(gammas,))

    def signed_area(ri, rj, rk):
        return 0.5 * ((rj[0] - ri[0]) * (rk[1] - ri[1]) -
                      (rk[0] - ri[0]) * (rj[1] - ri[1]))

    # Try different antisymmetric weights
    def compute_T_antisym(positions, gammas, weight_func):
        N = len(gammas)
        pos = positions.reshape(N, 2)
        T = 0.0
        for i, j, k in combinations(range(N), 3):
            area = signed_area(pos[i], pos[j], pos[k])
            weight = weight_func(gammas[i], gammas[j], gammas[k])
            T += weight * area
        return T

    # Weight options
    weights = {
        # These are symmetric - will give non-zero but not conserved
        'Γ³': lambda gi, gj, gk: gi * gj * gk,

        # Antisymmetric options
        '(Γi-Γj)(Γj-Γk)(Γk-Γi)': lambda gi, gj, gk: (gi-gj) * (gj-gk) * (gk-gi),

        # Mixed
        'Γi(Γj-Γk)': lambda gi, gj, gk: gi * (gj - gk),
        '(Γi-Γj)': lambda gi, gj, gk: gi - gj,

        # Totally antisymmetric (like epsilon tensor)
        'det(Γ)': lambda gi, gj, gk: gi*(1) + gj*(-1) + gk*(1),  # Not quite right
    }

    for name, w in weights.items():
        T_values = np.array([compute_T_antisym(trajectory[i], gammas, w)
                            for i in range(len(t))])
        mean_T = np.mean(T_values)
        fv = frac_var(T_values)
        print(f"  {name:30s}: frac_var = {fv:.2e}, mean = {mean_T:.4f}")


# ============================================================================
# APPROACH 3: Functions of pairwise distances
# ============================================================================

def test_triplet_distance_functions():
    """
    Instead of areas, use functions of the three pairwise distances.

    T = Σ_{i<j<k} Γi Γj Γk × g(r_ij, r_jk, r_ki)

    This is symmetric in the indices (for symmetric g), so it's non-trivially
    weighted.
    """
    print("\n" + "="*60)
    print("APPROACH 3: Triplet functions of pairwise distances")
    print("="*60)

    np.random.seed(42)
    N = 5

    theta = np.random.uniform(0, 2*np.pi, N)
    r = np.sqrt(np.random.uniform(0, 1, N)) * 0.8
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    gammas = np.random.randn(N)
    gammas -= gammas.mean()
    gammas /= np.abs(gammas).max()

    t = np.arange(0, 20.0, 0.01)
    trajectory = odeint(vortex_rhs, positions.flatten(), t, args=(gammas,))

    def dist(ri, rj):
        return np.sqrt((ri[0]-rj[0])**2 + (ri[1]-rj[1])**2)

    def compute_T_dist(positions, gammas, g):
        N = len(gammas)
        pos = positions.reshape(N, 2)
        T = 0.0
        for i, j, k in combinations(range(N), 3):
            r_ij = dist(pos[i], pos[j])
            r_jk = dist(pos[j], pos[k])
            r_ki = dist(pos[k], pos[i])
            T += gammas[i] * gammas[j] * gammas[k] * g(r_ij, r_jk, r_ki)
        return T

    # Various symmetric functions of distances
    dist_functions = {
        'sum_ln': lambda a, b, c: -np.log(a+1e-10) - np.log(b+1e-10) - np.log(c+1e-10),
        'product': lambda a, b, c: a * b * c,
        'sum': lambda a, b, c: a + b + c,
        'harmonic': lambda a, b, c: 1/(a+0.1) + 1/(b+0.1) + 1/(c+0.1),
        'sum_sq': lambda a, b, c: a**2 + b**2 + c**2,
        'product_ln': lambda a, b, c: np.log(a+1e-10) * np.log(b+1e-10) * np.log(c+1e-10),
    }

    for name, g in dist_functions.items():
        T_values = np.array([compute_T_dist(trajectory[i], gammas, g)
                            for i in range(len(t))])
        fv = frac_var(T_values)
        print(f"  T_{name:15s}: frac_var = {fv:.2e}")

    # Compare to pairwise
    print("\n  Pairwise Q_ln for comparison:")
    f_ln = lambda r: -np.log(r + 1e-10)
    Q_ln = np.array([compute_Q_f(trajectory[i], gammas, f_ln) for i in range(len(t))])
    print(f"  Q_ln: frac_var = {frac_var(Q_ln):.2e}")


# ============================================================================
# APPROACH 4: Mixed-order invariants
# ============================================================================

def test_mixed_order():
    """
    Combinations like Q_f × Γ or Q_f × L (angular momentum)

    These might reveal new conservation laws.
    """
    print("\n" + "="*60)
    print("APPROACH 4: Mixed-order invariants")
    print("="*60)

    np.random.seed(42)
    N = 5

    theta = np.random.uniform(0, 2*np.pi, N)
    r = np.sqrt(np.random.uniform(0, 1, N)) * 0.8
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    gammas = np.random.randn(N)
    gammas -= gammas.mean()
    gammas /= np.abs(gammas).max()

    t = np.arange(0, 20.0, 0.01)
    trajectory = odeint(vortex_rhs, positions.flatten(), t, args=(gammas,))

    def compute_angular_momentum(positions, gammas):
        """L = Σ Γi × (xi² + yi²)"""
        N = len(gammas)
        pos = positions.reshape(N, 2)
        return np.sum(gammas * (pos[:,0]**2 + pos[:,1]**2))

    def compute_center_of_vorticity(positions, gammas):
        """X = Σ Γi × xi, Y = Σ Γi × yi"""
        N = len(gammas)
        pos = positions.reshape(N, 2)
        total_gamma = np.sum(gammas)
        if abs(total_gamma) < 1e-10:
            return 0, 0
        X = np.sum(gammas * pos[:,0]) / total_gamma
        Y = np.sum(gammas * pos[:,1]) / total_gamma
        return X, Y

    # Compute time series
    f_ln = lambda r: -np.log(r + 1e-10)
    Q_ln = np.array([compute_Q_f(trajectory[i], gammas, f_ln) for i in range(len(t))])
    L = np.array([compute_angular_momentum(trajectory[i], gammas) for i in range(len(t))])

    # Total circulation (trivially conserved)
    Gamma_total = np.sum(gammas)

    print(f"  Q_ln: frac_var = {frac_var(Q_ln):.2e}")
    print(f"  L (angular momentum): frac_var = {frac_var(L):.2e}")
    print(f"  Total Γ = {Gamma_total:.4f} (trivially conserved)")

    # Mixed quantities
    print("\n  Mixed quantities:")
    print(f"  Q_ln × L: frac_var = {frac_var(Q_ln * L):.2e}")
    print(f"  Q_ln / L: frac_var = {frac_var(Q_ln / (L + 1e-10)):.2e}")
    print(f"  Q_ln + L: frac_var = {frac_var(Q_ln + L):.2e}")


# ============================================================================
# APPROACH 5: Information-theoretic quantities
# ============================================================================

def test_entropy_like():
    """
    Entropy-like quantities: S = -Σ pi log(pi)

    where pi could be based on:
    - Normalized circulation |Γi| / Σ|Γj|
    - Distance weights
    - Area-based
    """
    print("\n" + "="*60)
    print("APPROACH 5: Entropy-like quantities")
    print("="*60)

    np.random.seed(42)
    N = 5

    theta = np.random.uniform(0, 2*np.pi, N)
    r = np.sqrt(np.random.uniform(0, 1, N)) * 0.8
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    gammas = np.random.randn(N)
    gammas -= gammas.mean()
    gammas /= np.abs(gammas).max()

    t = np.arange(0, 20.0, 0.01)
    trajectory = odeint(vortex_rhs, positions.flatten(), t, args=(gammas,))

    def compute_distance_entropy(positions, gammas):
        """Entropy based on pairwise distances."""
        N = len(gammas)
        pos = positions.reshape(N, 2)

        # Compute all pairwise distances
        distances = []
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                distances.append(d * abs(gammas[i] * gammas[j]))

        distances = np.array(distances)
        distances = distances / (distances.sum() + 1e-10)  # Normalize

        # Entropy
        return -np.sum(distances * np.log(distances + 1e-10))

    def compute_position_entropy(positions, gammas):
        """Entropy based on radial positions."""
        N = len(gammas)
        pos = positions.reshape(N, 2)

        radii = np.sqrt(pos[:,0]**2 + pos[:,1]**2)
        weights = np.abs(gammas) * radii
        weights = weights / (weights.sum() + 1e-10)

        return -np.sum(weights * np.log(weights + 1e-10))

    S_dist = np.array([compute_distance_entropy(trajectory[i], gammas) for i in range(len(t))])
    S_pos = np.array([compute_position_entropy(trajectory[i], gammas) for i in range(len(t))])

    print(f"  S_distance: frac_var = {frac_var(S_dist):.2e}")
    print(f"  S_position: frac_var = {frac_var(S_pos):.2e}")


# ============================================================================
# APPROACH 6: Neural network discovery
# ============================================================================

def neural_search():
    """
    Use gradient descent to find the optimal higher-order invariant.

    Parameterize T = Σ_{i<j<k} w(Γi, Γj, Γk, r_ij, r_jk, r_ki)
    and learn w to minimize frac_var.
    """
    print("\n" + "="*60)
    print("APPROACH 6: Learning optimal triplet function")
    print("="*60)

    np.random.seed(42)
    N = 5

    theta = np.random.uniform(0, 2*np.pi, N)
    r = np.sqrt(np.random.uniform(0, 1, N)) * 0.8
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    gammas = np.random.randn(N)
    gammas -= gammas.mean()
    gammas /= np.abs(gammas).max()

    t = np.arange(0, 20.0, 0.02)  # Coarser for speed
    trajectory = odeint(vortex_rhs, positions.flatten(), t, args=(gammas,))

    def dist(ri, rj):
        return np.sqrt((ri[0]-rj[0])**2 + (ri[1]-rj[1])**2)

    # Basis functions for the triplet
    def compute_basis(trajectory, gammas):
        """Compute basis function values for all triplets."""
        n_times = len(trajectory)
        N = len(gammas)
        pos = trajectory.reshape(n_times, N, 2)

        # Basis: 1, ln(r_ij), ln(r_jk), ln(r_ki), r_ij, r_jk, r_ki
        n_basis = 7
        n_triplets = N * (N-1) * (N-2) // 6

        basis = np.zeros((n_times, n_triplets, n_basis))
        weights = np.zeros(n_triplets)

        idx = 0
        for i, j, k in combinations(range(N), 3):
            weights[idx] = gammas[i] * gammas[j] * gammas[k]

            for ti in range(n_times):
                r_ij = dist(pos[ti, i], pos[ti, j])
                r_jk = dist(pos[ti, j], pos[ti, k])
                r_ki = dist(pos[ti, k], pos[ti, i])

                basis[ti, idx, 0] = 1.0
                basis[ti, idx, 1] = -np.log(r_ij + 1e-10)
                basis[ti, idx, 2] = -np.log(r_jk + 1e-10)
                basis[ti, idx, 3] = -np.log(r_ki + 1e-10)
                basis[ti, idx, 4] = r_ij
                basis[ti, idx, 5] = r_jk
                basis[ti, idx, 6] = r_ki

            idx += 1

        return basis, weights

    basis, weights = compute_basis(trajectory, gammas)

    # Optimize coefficients
    from scipy.optimize import minimize

    def loss(coeffs):
        # T_t = Σ_triplets weight_triplet × (Σ_b coeff_b × basis_b)
        # Weighted basis values
        weighted_basis = basis * weights[None, :, None]
        # Sum over triplets
        T_t = np.sum(weighted_basis * coeffs[None, None, :], axis=(1, 2))
        # Fractional variance
        return np.std(T_t) / (np.abs(np.mean(T_t)) + 1e-10)

    # Initial guess
    x0 = np.random.randn(7) * 0.1
    result = minimize(loss, x0, method='Nelder-Mead')

    print(f"  Optimal coefficients: {result.x.round(4)}")
    print(f"  Achieved frac_var: {result.fun:.2e}")

    # Compare to pairwise Q_ln
    f_ln = lambda r: -np.log(r + 1e-10)
    Q_ln = np.array([compute_Q_f(trajectory[i], gammas, f_ln) for i in range(len(t))])
    print(f"  Q_ln frac_var: {frac_var(Q_ln):.2e} (for comparison)")


# ============================================================================
# CONCLUSION
# ============================================================================

def conclusion():
    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
    NEGATIVE RESULT: The "triplet area invariant" was a false alarm.

    T_full = Σ_{all perms} Γi Γj Γk × Area(ijk) = 0 identically
    (Symmetric weight × antisymmetric area cancels)

    T_ordered = Σ_{i<j<k} Γi Γj Γk × Area(ijk) is NOT conserved

    WHAT WE LEARNED:
    1. Higher-order invariants are HARD to find
    2. Symmetry properties matter critically
    3. The pairwise Q_f family is remarkably special

    POSSIBLE REASONS Q_f is special:
    - The 2D vortex dynamics has infinite-dimensional symmetry
    - Particle relabeling + area preservation → Q_f family
    - Higher-order structures might require higher-dimensional dynamics

    REMAINING DIRECTIONS:
    1. Look for CONDITIONAL conservation (e.g., N=3 only)
    2. Look for APPROXIMATE invariants with slow drift
    3. Look for TOPOLOGICAL invariants (knot types, linking)
    4. Look in 3D or other physical systems
    """)


def main():
    test_products_of_invariants()
    test_antisymmetric_triplets()
    test_triplet_distance_functions()
    test_mixed_order()
    test_entropy_like()
    neural_search()
    conclusion()


if __name__ == "__main__":
    main()
