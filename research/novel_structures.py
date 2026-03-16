#!/usr/bin/env python3
"""
Novel Mathematical Structures in Vortex Dynamics

After the triplet invariant false alarm, we search more carefully for
genuine mathematical structures that might reveal new vocabulary.

APPROACHES:
1. Spectral properties - eigenvalues of configuration matrices
2. Topological/combinatorial - pairing patterns, graph structures
3. Differential-geometric - curvature of trajectory in config space
4. Algebraic - polynomial invariants, resultants
5. Information-theoretic - mutual information between vortex pairs
"""

import numpy as np
from scipy.integrate import odeint
from scipy.linalg import eig, svd
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


def frac_var(x):
    return np.std(x) / (np.abs(np.mean(x)) + 1e-10)


def simulate(N, T_sim=20.0, seed=42):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2*np.pi, N)
    r = np.sqrt(np.random.uniform(0, 1, N)) * 0.8
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    gammas = np.random.randn(N)
    gammas -= gammas.mean()
    gammas /= np.abs(gammas).max()

    t = np.arange(0, T_sim, 0.01)
    trajectory = odeint(vortex_rhs, positions.flatten(), t, args=(gammas,))
    return t, trajectory, gammas


# ============================================================================
# APPROACH 1: Spectral Invariants
# ============================================================================

def spectral_invariants():
    """
    Look at eigenvalues of matrices constructed from vortex positions.

    The "interaction matrix" M_ij = Γi Γj / r_ij might have conserved
    spectral properties (trace, determinant, eigenvalue ratios).
    """
    print("="*60)
    print("APPROACH 1: Spectral Invariants")
    print("="*60)

    t, trajectory, gammas = simulate(N=5, T_sim=20.0)
    N = len(gammas)

    def compute_interaction_matrix(positions, gammas):
        pos = positions.reshape(N, 2)
        M = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    r = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                    M[i, j] = gammas[i] * gammas[j] * (-np.log(r + 1e-10))
        return M

    def compute_distance_matrix(positions):
        pos = positions.reshape(N, 2)
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
        return D

    # Spectral quantities
    traces = []
    dets = []
    eig_ratios = []
    dist_traces = []

    for i in range(len(t)):
        M = compute_interaction_matrix(trajectory[i], gammas)
        D = compute_distance_matrix(trajectory[i])

        traces.append(np.trace(M @ M))  # Frobenius norm squared
        dets.append(np.linalg.det(M + np.eye(N) * 0.01))

        # Eigenvalue ratio
        eigvals = np.sort(np.abs(np.linalg.eigvals(M)))[::-1]
        if eigvals[1] > 1e-10:
            eig_ratios.append(eigvals[0] / eigvals[1])
        else:
            eig_ratios.append(0)

        dist_traces.append(np.trace(D @ D))

    traces = np.array(traces)
    dets = np.array(dets)
    eig_ratios = np.array(eig_ratios)
    dist_traces = np.array(dist_traces)

    print(f"  Tr(M²): frac_var = {frac_var(traces):.2e}")
    print(f"  det(M): frac_var = {frac_var(dets):.2e}")
    print(f"  λ₁/λ₂: frac_var = {frac_var(eig_ratios):.2e}")
    print(f"  Tr(D²): frac_var = {frac_var(dist_traces):.2e}")


# ============================================================================
# APPROACH 2: Graph-Theoretic Invariants
# ============================================================================

def graph_invariants():
    """
    Treat vortex pairs as edges in a weighted graph.
    Graph invariants like spanning tree weight might be conserved.
    """
    print("\n" + "="*60)
    print("APPROACH 2: Graph-Theoretic Invariants")
    print("="*60)

    t, trajectory, gammas = simulate(N=5, T_sim=20.0)
    N = len(gammas)

    def compute_graph_quantities(positions, gammas):
        pos = positions.reshape(N, 2)

        # Weighted adjacency: w_ij = |Γi Γj| / r_ij
        W = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                r = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                w = abs(gammas[i] * gammas[j]) / (r + 0.01)
                W[i, j] = w
                W[j, i] = w

        # Degree sum
        degrees = np.sum(W, axis=1)

        # Laplacian: L = D - W
        L = np.diag(degrees) - W

        # Number of spanning trees (product of non-zero eigenvalues of L / N)
        eigvals = np.sort(np.linalg.eigvals(L).real)
        spanning_tree_count = np.prod(eigvals[1:]) / N if N > 1 else 0

        return {
            'total_weight': np.sum(W),
            'max_degree': np.max(degrees),
            'algebraic_connectivity': eigvals[1] if N > 1 else 0,
            'spanning_trees': spanning_tree_count
        }

    results = {k: [] for k in ['total_weight', 'max_degree', 'algebraic_connectivity', 'spanning_trees']}

    for i in range(len(t)):
        q = compute_graph_quantities(trajectory[i], gammas)
        for k in results:
            results[k].append(q[k])

    for k in results:
        arr = np.array(results[k])
        print(f"  {k:25s}: frac_var = {frac_var(arr):.2e}")


# ============================================================================
# APPROACH 3: Curvature of Configuration Space Trajectory
# ============================================================================

def trajectory_curvature():
    """
    The trajectory in configuration space has curvature.
    Is there a conserved relationship between speed and curvature?
    (Like v²κ = const for centripetal motion)
    """
    print("\n" + "="*60)
    print("APPROACH 3: Configuration Space Curvature")
    print("="*60)

    t, trajectory, gammas = simulate(N=5, T_sim=20.0)

    # Velocity and acceleration in config space
    dt = t[1] - t[0]
    velocity = np.diff(trajectory, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt

    # Speed |v|
    speed = np.linalg.norm(velocity, axis=1)

    # Curvature: κ = |v × a| / |v|³ (generalized)
    # In high dim: κ = |a_perp| / |v|² where a_perp = a - (a·v̂)v̂

    curvature = []
    for i in range(len(acceleration)):
        v = velocity[i]
        a = acceleration[i]
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:
            v_hat = v / v_norm
            a_parallel = np.dot(a, v_hat) * v_hat
            a_perp = a - a_parallel
            kappa = np.linalg.norm(a_perp) / (v_norm**2)
            curvature.append(kappa)
        else:
            curvature.append(0)

    curvature = np.array(curvature)
    speed_sq = speed[:-1]**2

    print(f"  |v|: frac_var = {frac_var(speed):.2e}")
    print(f"  κ: frac_var = {frac_var(curvature):.2e}")
    print(f"  |v|²κ: frac_var = {frac_var(speed_sq * curvature):.2e}")
    print(f"  |v|κ: frac_var = {frac_var(speed[:-1] * curvature):.2e}")


# ============================================================================
# APPROACH 4: Polynomial Invariants
# ============================================================================

def polynomial_invariants():
    """
    Look for polynomial combinations of positions that are conserved.

    The center of vorticity is linear, Q_ln involves log...
    What about pure polynomials?
    """
    print("\n" + "="*60)
    print("APPROACH 4: Polynomial Invariants")
    print("="*60)

    t, trajectory, gammas = simulate(N=5, T_sim=20.0)
    N = len(gammas)

    def compute_poly_invariants(positions, gammas):
        pos = positions.reshape(N, 2)
        x, y = pos[:, 0], pos[:, 1]

        return {
            # Linear (known)
            'Γ·x': np.sum(gammas * x),
            'Γ·y': np.sum(gammas * y),
            'Γ·r²': np.sum(gammas * (x**2 + y**2)),  # Angular momentum

            # Quadratic
            'Γ·x²': np.sum(gammas * x**2),
            'Γ·xy': np.sum(gammas * x * y),
            'Γ²·x·x': sum(gammas[i]*gammas[j]*x[i]*x[j]
                          for i in range(N) for j in range(N) if i != j),
            'Γ²·r²': sum(gammas[i]*gammas[j]*(x[i]**2+y[i]**2)
                         for i in range(N) for j in range(N) if i != j),

            # Cross products
            'Γ²·(x_i·y_j - x_j·y_i)': sum(gammas[i]*gammas[j]*(x[i]*y[j]-x[j]*y[i])
                                          for i in range(N) for j in range(N) if i != j),
        }

    results = {k: [] for k in compute_poly_invariants(trajectory[0], gammas).keys()}

    for i in range(len(t)):
        q = compute_poly_invariants(trajectory[i], gammas)
        for k in results:
            results[k].append(q[k])

    for k in results:
        arr = np.array(results[k])
        print(f"  {k:30s}: frac_var = {frac_var(arr):.2e}")


# ============================================================================
# APPROACH 5: Mutual Information
# ============================================================================

def mutual_information():
    """
    The mutual information between pairs of vortices might be conserved.
    """
    print("\n" + "="*60)
    print("APPROACH 5: Mutual Information / Correlation Structure")
    print("="*60)

    t, trajectory, gammas = simulate(N=5, T_sim=20.0)
    N = len(gammas)

    # Compute correlation matrix of velocities
    velocities = []
    for i in range(len(t)):
        v = vortex_rhs(trajectory[i], 0, gammas).reshape(N, 2)
        velocities.append(v.flatten())
    velocities = np.array(velocities)

    # Time-averaged correlation
    corr = np.corrcoef(velocities.T)

    print(f"  Velocity correlation matrix shape: {corr.shape}")
    print(f"  Mean off-diagonal correlation: {np.mean(np.abs(corr - np.eye(2*N))):.4f}")

    # Track correlation over time windows
    window_size = 100
    correlations = []

    for start in range(0, len(t) - window_size, window_size // 2):
        window = velocities[start:start+window_size]
        c = np.corrcoef(window.T)
        # Extract vortex-pair correlations
        pair_corr = []
        for i in range(N):
            for j in range(i+1, N):
                # Correlation between vortex i and j velocities
                idx_i = [2*i, 2*i+1]
                idx_j = [2*j, 2*j+1]
                pair_corr.append(np.mean(np.abs(c[np.ix_(idx_i, idx_j)])))
        correlations.append(np.mean(pair_corr))

    correlations = np.array(correlations)
    print(f"  Pairwise velocity correlation: frac_var = {frac_var(correlations):.2e}")


# ============================================================================
# APPROACH 6: Action-Angle Variables
# ============================================================================

def action_angle():
    """
    In integrable systems, action variables are conserved and angles evolve linearly.
    Look for linear combinations that evolve linearly in time.
    """
    print("\n" + "="*60)
    print("APPROACH 6: Action-Angle Structure")
    print("="*60)

    t, trajectory, gammas = simulate(N=4, T_sim=50.0)  # N=4 for quasi-integrability
    N = len(gammas)

    # Compute angles between all vortex pairs
    def compute_angles(positions):
        pos = positions.reshape(N, 2)
        angles = []
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                angles.append(np.arctan2(dy, dx))
        return np.array(angles)

    angles_over_time = np.array([compute_angles(trajectory[i]) for i in range(len(t))])

    # Unwrap to remove 2π jumps
    angles_unwrapped = np.unwrap(angles_over_time, axis=0)

    # Check for linear evolution: angle = ω*t + φ
    # Fit lines and measure residuals
    print("  Angle linearity (lower = more linear):")

    n_pairs = N * (N - 1) // 2
    for p in range(min(3, n_pairs)):  # First few pairs
        # Linear fit
        coeffs = np.polyfit(t, angles_unwrapped[:, p], 1)
        fitted = np.polyval(coeffs, t)
        residual = np.std(angles_unwrapped[:, p] - fitted)
        omega = coeffs[0]
        print(f"    Pair {p}: ω = {omega:.4f} rad/s, residual = {residual:.4f}")

    # Look for linear combinations of angles that are conserved
    print("\n  Searching for conserved angle combinations...")

    # Try simple differences: θ_i - θ_j
    for i in range(min(n_pairs-1, 3)):
        for j in range(i+1, min(n_pairs, 4)):
            diff = angles_unwrapped[:, i] - angles_unwrapped[:, j]
            fv = frac_var(diff)
            if fv < 0.1:
                print(f"    θ_{i} - θ_{j}: frac_var = {fv:.2e}")


# ============================================================================
# APPROACH 7: Emergent Dimensionality
# ============================================================================

def emergent_dimension():
    """
    The trajectory might effectively live in a lower-dimensional subspace.
    What is the effective dimension, and is it conserved?
    """
    print("\n" + "="*60)
    print("APPROACH 7: Emergent Dimensionality (PCA)")
    print("="*60)

    t, trajectory, gammas = simulate(N=6, T_sim=30.0)

    # PCA on trajectory
    mean = np.mean(trajectory, axis=0)
    centered = trajectory - mean
    _, s, _ = svd(centered, full_matrices=False)

    # Explained variance ratios
    explained = (s**2) / np.sum(s**2)
    cumulative = np.cumsum(explained)

    print(f"  Configuration space dimension: {trajectory.shape[1]}")
    print(f"  Variance explained by first 3 PCs: {cumulative[2]*100:.1f}%")
    print(f"  Variance explained by first 5 PCs: {cumulative[4]*100:.1f}%")

    # Participation ratio (effective dimension)
    P = 1.0 / np.sum(explained**2)
    print(f"  Participation ratio (effective dim): {P:.2f}")

    # Check if participation ratio is conserved across windows
    window_size = 300
    Ps = []
    for start in range(0, len(t) - window_size, window_size // 2):
        window = trajectory[start:start+window_size]
        m = np.mean(window, axis=0)
        c = window - m
        _, sw, _ = svd(c, full_matrices=False)
        exp = (sw**2) / np.sum(sw**2)
        Ps.append(1.0 / np.sum(exp**2))

    Ps = np.array(Ps)
    print(f"  Participation ratio: frac_var = {frac_var(Ps):.2e}")


# ============================================================================
# Summary
# ============================================================================

def summary():
    print("\n" + "="*70)
    print("SUMMARY: What Novel Structures Were Found?")
    print("="*70)
    print("""
    Structures tested:
    1. Spectral properties - NOT well conserved
    2. Graph invariants - NOT well conserved
    3. Curvature relations - NOT conserved
    4. Polynomial invariants - Known ones conserved (Γ·r²), new ones NOT
    5. Mutual information - Varies over time
    6. Action-angle - Some pairs show linear angle evolution
    7. Effective dimension - Participation ratio varies

    CONCLUSION:
    The pairwise Q_f family with f(r) = -ln(r) remains the special structure.

    The "beyond known math" we're looking for might not be a new conservation
    law, but rather:
    - A unifying framework for WHY Q_{-ln(r)} is special
    - A connection to infinite-dimensional symmetries
    - A categorical/algebraic structure underlying the Q_f family

    The mathematics IS known (symplectic geometry, Poisson manifolds),
    just not widely applied to this specific problem.
    """)


def main():
    spectral_invariants()
    graph_invariants()
    trajectory_curvature()
    polynomial_invariants()
    mutual_information()
    action_angle()
    emergent_dimension()
    summary()


if __name__ == "__main__":
    main()
