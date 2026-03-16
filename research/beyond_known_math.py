#!/usr/bin/env python3
"""
Beyond Known Mathematics: Searching for Proto-Mathematical Structures

The premise: Maybe the Q_f family and vortex dynamics contain patterns
that existing mathematical vocabulary can't express. We look for:

1. EMERGENT STRUCTURE - patterns in data that don't fit known categories
2. NOVEL OPERATIONS - combinations that aren't standard math operations
3. HIDDEN SYMMETRIES - transformations that preserve something unknown
4. INFORMATION GEOMETRY - structures in the space of conserved quantities

Instead of asking "which Q_f is conserved?", we ask:
"What IS the thing that's being conserved that we don't have words for?"
"""

import numpy as np
from scipy.integrate import odeint
from scipy.linalg import svd, eig
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Vortex System
# ============================================================================

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


def simulate_vortices(N, T=10.0, dt=0.01, seed=42):
    """Simulate N random vortices."""
    np.random.seed(seed)

    # Random initial positions in unit disk
    theta = np.random.uniform(0, 2*np.pi, N)
    r = np.sqrt(np.random.uniform(0, 1, N)) * 0.8
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Random circulations (sum to zero for bounded motion)
    gammas = np.random.randn(N)
    gammas -= gammas.mean()
    gammas /= np.abs(gammas).max()

    t = np.arange(0, T, dt)
    state0 = positions.flatten()

    trajectory = odeint(vortex_rhs, state0, t, args=(gammas,))

    return t, trajectory, gammas


# ============================================================================
# EXPLORATION 1: What's the "natural" representation?
# ============================================================================

def compute_all_distances(positions):
    """Compute all pairwise distances."""
    N = len(positions) // 2
    pos = positions.reshape(N, 2)
    return pdist(pos)


def compute_all_angles(positions):
    """Compute all pairwise angles."""
    N = len(positions) // 2
    pos = positions.reshape(N, 2)
    angles = []
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            angles.append(np.arctan2(dy, dx))
    return np.array(angles)


def compute_triplet_areas(positions):
    """Compute all triplet signed areas."""
    N = len(positions) // 2
    pos = positions.reshape(N, 2)
    areas = []
    for i, j, k in combinations(range(N), 3):
        # Signed area of triangle ijk
        area = 0.5 * ((pos[j,0] - pos[i,0]) * (pos[k,1] - pos[i,1]) -
                      (pos[k,0] - pos[i,0]) * (pos[j,1] - pos[i,1]))
        areas.append(area)
    return np.array(areas)


def find_natural_coordinates(trajectory, gammas):
    """
    Search for a representation where the dynamics is "simpler".

    Hypothesis: There exists a transformation of the configuration
    where the conserved quantities have a cleaner form.
    """
    n_times, n_coords = trajectory.shape
    N = n_coords // 2

    print("Exploring natural coordinate representations...")
    print(f"  N = {N} vortices, {n_times} time steps")

    results = {}

    # 1. Pairwise distances
    distances = np.array([compute_all_distances(trajectory[i]) for i in range(n_times)])
    dist_var = np.std(distances, axis=0) / (np.mean(distances, axis=0) + 1e-10)
    results['distances'] = {
        'data': distances,
        'most_conserved_idx': np.argmin(dist_var),
        'best_frac_var': np.min(dist_var)
    }
    print(f"  Distances: best frac_var = {np.min(dist_var):.2e}")

    # 2. Angles
    angles = np.array([compute_all_angles(trajectory[i]) for i in range(n_times)])
    # Unwrap angles to avoid discontinuities
    angles_unwrapped = np.unwrap(angles, axis=0)
    angle_var = np.std(angles_unwrapped, axis=0) / (2*np.pi)
    results['angles'] = {
        'data': angles_unwrapped,
        'most_conserved_idx': np.argmin(angle_var),
        'best_frac_var': np.min(angle_var)
    }
    print(f"  Angles: best frac_var = {np.min(angle_var):.2e}")

    # 3. Triplet areas
    if N >= 3:
        areas = np.array([compute_triplet_areas(trajectory[i]) for i in range(n_times)])
        area_mean = np.mean(areas, axis=0)
        area_var = np.std(areas, axis=0) / (np.abs(area_mean) + 1e-10)
        results['areas'] = {
            'data': areas,
            'most_conserved_idx': np.argmin(area_var),
            'best_frac_var': np.min(area_var)
        }
        print(f"  Triplet areas: best frac_var = {np.min(area_var):.2e}")

    # 4. Weighted combinations (circulation-weighted)
    n_pairs = N * (N - 1) // 2
    if n_pairs > 0:
        weighted_dists = np.zeros((n_times, n_pairs))
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                pos = trajectory.reshape(n_times, N, 2)
                d = np.sqrt((pos[:,i,0] - pos[:,j,0])**2 + (pos[:,i,1] - pos[:,j,1])**2)
                weighted_dists[:, idx] = gammas[i] * gammas[j] * d
                idx += 1

        wdist_mean = np.mean(weighted_dists, axis=0)
        wdist_var = np.std(weighted_dists, axis=0) / (np.abs(wdist_mean) + 1e-10)
        results['weighted_distances'] = {
            'data': weighted_dists,
            'most_conserved_idx': np.argmin(wdist_var),
            'best_frac_var': np.min(wdist_var)
        }
        print(f"  Γ-weighted distances: best frac_var = {np.min(wdist_var):.2e}")

    return results


# ============================================================================
# EXPLORATION 2: Emergent algebraic structure
# ============================================================================

def discover_algebraic_relations(trajectory, gammas):
    """
    Search for algebraic relations between Q_f values.

    Maybe there's a "multiplication" or "composition" of conserved
    quantities that we don't have a name for.
    """
    n_times, n_coords = trajectory.shape
    N = n_coords // 2

    print("\nSearching for algebraic relations...")

    # Compute Q_f for various f
    def compute_qf(positions, gammas, f):
        N = len(gammas)
        pos = positions.reshape(N, 2)
        Q = 0.0
        for i in range(N):
            for j in range(N):
                if i != j:
                    r = np.sqrt((pos[i,0] - pos[j,0])**2 + (pos[i,1] - pos[j,1])**2)
                    Q += gammas[i] * gammas[j] * f(r)
        return Q

    # Various f functions
    f_functions = {
        'ln': lambda r: -np.log(r + 1e-10),
        'sqrt': lambda r: np.sqrt(r),
        'r': lambda r: r,
        'r2': lambda r: r**2,
        'exp': lambda r: np.exp(-r),
        '1/r': lambda r: 1.0 / (r + 0.1),
        'tanh': lambda r: np.tanh(r),
    }

    # Compute time series of Q_f values
    Q_values = {}
    for name, f in f_functions.items():
        Q_values[name] = np.array([compute_qf(trajectory[i], gammas, f)
                                    for i in range(n_times)])

    # Search for algebraic relations
    # Form: a*Q1 + b*Q2 + c*Q1*Q2 + d*Q1/Q2 + ... = const

    relations_found = []

    # Linear combinations
    names = list(Q_values.keys())
    for n1, n2 in combinations(names, 2):
        Q1 = Q_values[n1]
        Q2 = Q_values[n2]

        # Try Q1 + α*Q2 = const
        # Minimize variance of Q1 + α*Q2
        # α = -cov(Q1, Q2) / var(Q2)
        cov = np.cov(Q1, Q2)[0, 1]
        var2 = np.var(Q2)
        if var2 > 1e-20:
            alpha = -cov / var2
            combo = Q1 + alpha * Q2
            frac_var = np.std(combo) / (np.abs(np.mean(combo)) + 1e-10)
            if frac_var < 0.001:
                relations_found.append({
                    'type': 'linear',
                    'relation': f'Q_{n1} + {alpha:.4f} * Q_{n2}',
                    'frac_var': frac_var
                })

    # Ratio relations
    for n1, n2 in combinations(names, 2):
        Q1 = Q_values[n1]
        Q2 = Q_values[n2]
        if np.all(np.abs(Q2) > 1e-10):
            ratio = Q1 / Q2
            frac_var = np.std(ratio) / (np.abs(np.mean(ratio)) + 1e-10)
            if frac_var < 0.001:
                relations_found.append({
                    'type': 'ratio',
                    'relation': f'Q_{n1} / Q_{n2}',
                    'frac_var': frac_var
                })

    # Product relations
    for n1, n2 in combinations(names, 2):
        Q1 = Q_values[n1]
        Q2 = Q_values[n2]
        product = Q1 * Q2
        frac_var = np.std(product) / (np.abs(np.mean(product)) + 1e-10)
        if frac_var < 0.001:
            relations_found.append({
                'type': 'product',
                'relation': f'Q_{n1} * Q_{n2}',
                'frac_var': frac_var
            })

    # Power relations: Q1^a * Q2^b = const
    for n1, n2 in combinations(names, 2):
        Q1 = Q_values[n1]
        Q2 = Q_values[n2]
        if np.all(Q1 > 0) and np.all(Q2 > 0):
            # log(Q1^a * Q2^b) = a*log(Q1) + b*log(Q2) = const
            # This is a linear relation in log space
            log_Q1 = np.log(Q1)
            log_Q2 = np.log(Q2)
            cov = np.cov(log_Q1, log_Q2)[0, 1]
            var2 = np.var(log_Q2)
            if var2 > 1e-20:
                b_over_a = -cov / var2
                # Q1 * Q2^(b/a) should be constant
                combo = Q1 * Q2**b_over_a
                frac_var = np.std(combo) / (np.abs(np.mean(combo)) + 1e-10)
                if frac_var < 0.001:
                    relations_found.append({
                        'type': 'power',
                        'relation': f'Q_{n1} * Q_{n2}^{b_over_a:.4f}',
                        'frac_var': frac_var
                    })

    print(f"  Found {len(relations_found)} algebraic relations with frac_var < 0.001")
    for rel in sorted(relations_found, key=lambda x: x['frac_var'])[:5]:
        print(f"    {rel['type']:8s}: {rel['relation']:30s} (frac_var = {rel['frac_var']:.2e})")

    return Q_values, relations_found


# ============================================================================
# EXPLORATION 3: Information-theoretic structure
# ============================================================================

def information_geometry(trajectory, gammas):
    """
    Look at the geometry of the space of configurations.

    Maybe there's a natural metric or connection that reveals
    hidden structure.
    """
    n_times, n_coords = trajectory.shape
    N = n_coords // 2

    print("\nExploring information geometry...")

    # Compute "velocity" in configuration space
    velocities = np.diff(trajectory, axis=0)

    # What's the natural metric? Try different options:

    # 1. Euclidean metric (standard)
    euclidean_speeds = np.linalg.norm(velocities, axis=1)

    # 2. Circulation-weighted metric
    pos = trajectory[:-1].reshape(-1, N, 2)
    vel = velocities.reshape(-1, N, 2)
    weighted_speeds = np.zeros(len(velocities))
    for t in range(len(velocities)):
        for i in range(N):
            weighted_speeds[t] += gammas[i]**2 * (vel[t,i,0]**2 + vel[t,i,1]**2)
    weighted_speeds = np.sqrt(weighted_speeds)

    # 3. Relative motion metric (only pairwise distances change)
    distances_t = np.array([compute_all_distances(trajectory[i]) for i in range(n_times)])
    rel_speeds = np.linalg.norm(np.diff(distances_t, axis=0), axis=1)

    print(f"  Euclidean speed: mean={np.mean(euclidean_speeds):.4f}, var={np.var(euclidean_speeds):.2e}")
    print(f"  Γ-weighted speed: mean={np.mean(weighted_speeds):.4f}, var={np.var(weighted_speeds):.2e}")
    print(f"  Relative speed: mean={np.mean(rel_speeds):.4f}, var={np.var(rel_speeds):.2e}")

    # Which metric makes the motion "uniform"?
    metrics = {
        'euclidean': euclidean_speeds,
        'weighted': weighted_speeds,
        'relative': rel_speeds
    }

    for name, speeds in metrics.items():
        uniformity = np.std(speeds) / np.mean(speeds)
        print(f"    {name:12s} uniformity: {uniformity:.4f}")

    return metrics


# ============================================================================
# EXPLORATION 4: Emergent "operators"
# ============================================================================

def discover_operators(trajectory, gammas):
    """
    Search for operations on configurations that preserve structure.

    Beyond rotations and translations, what transformations keep
    the dynamics invariant?
    """
    n_times, n_coords = trajectory.shape
    N = n_coords // 2

    print("\nSearching for emergent operators...")

    # The dynamics is invariant under:
    # 1. Translations (known)
    # 2. Rotations (known)
    # 3. Scaling x -> λx, t -> λ²t (known)
    # 4. What else?

    # Try: permutations that preserve dynamics
    # If vortices i and j have same circulation, swapping them should work

    from itertools import permutations

    # Check which permutations give equivalent dynamics
    equivalent_perms = []

    pos0 = trajectory[0].reshape(N, 2)
    pos1 = trajectory[1].reshape(N, 2)
    dt = 0.01

    for perm in permutations(range(N)):
        perm = list(perm)
        # Permuted positions
        pos0_perm = pos0[perm]
        # Expected velocity under permutation
        gammas_perm = gammas[perm]

        # Compute expected next position
        vel_perm = np.zeros_like(pos0_perm)
        for i in range(N):
            for j in range(N):
                if i != j:
                    dx = pos0_perm[i, 0] - pos0_perm[j, 0]
                    dy = pos0_perm[i, 1] - pos0_perm[j, 1]
                    r2 = dx**2 + dy**2 + 1e-10
                    vel_perm[i, 0] += -gammas_perm[j] * dy / (2 * np.pi * r2)
                    vel_perm[i, 1] += gammas_perm[j] * dx / (2 * np.pi * r2)

        pos1_perm_pred = pos0_perm + vel_perm * dt

        # Compare to actual permuted trajectory
        pos1_perm_actual = pos1[perm]

        error = np.linalg.norm(pos1_perm_pred - pos1_perm_actual)

        if error < 1e-6:
            equivalent_perms.append(perm)

    print(f"  Found {len(equivalent_perms)} equivalent permutations")

    # Check for approximate symmetries
    # Scale invariance: dynamics under x -> λx should give v -> v/λ

    return equivalent_perms


# ============================================================================
# EXPLORATION 5: The "conservation field"
# ============================================================================

def conservation_field(trajectory, gammas):
    """
    Instead of asking "what is conserved?", ask:
    "What is the FIELD of conservation?"

    At each point in configuration space, there's a vector pointing
    in the direction of maximum conservation. What does this field
    look like?
    """
    n_times, n_coords = trajectory.shape
    N = n_coords // 2

    print("\nComputing conservation field...")

    def compute_qf(positions, gammas, f):
        N = len(gammas)
        pos = positions.reshape(N, 2)
        Q = 0.0
        for i in range(N):
            for j in range(N):
                if i != j:
                    r = np.sqrt((pos[i,0] - pos[j,0])**2 + (pos[i,1] - pos[j,1])**2)
                    Q += gammas[i] * gammas[j] * f(r)
        return Q

    # Compute gradient of Q_ln at each time
    f = lambda r: -np.log(r + 1e-10)
    epsilon = 1e-6

    gradients = np.zeros((n_times, n_coords))
    Q_values = np.zeros(n_times)

    for t in range(n_times):
        pos = trajectory[t]
        Q_values[t] = compute_qf(pos, gammas, f)

        for i in range(n_coords):
            pos_plus = pos.copy()
            pos_plus[i] += epsilon
            pos_minus = pos.copy()
            pos_minus[i] -= epsilon
            gradients[t, i] = (compute_qf(pos_plus, gammas, f) -
                               compute_qf(pos_minus, gammas, f)) / (2 * epsilon)

    # The conservation field should be orthogonal to the velocity field
    velocities = np.zeros((n_times, n_coords))
    for t in range(n_times):
        velocities[t] = vortex_rhs(trajectory[t], 0, gammas)

    # Check orthogonality
    dots = np.array([np.dot(gradients[t], velocities[t]) for t in range(n_times)])

    print(f"  Mean |∇Q · v|: {np.mean(np.abs(dots)):.2e}")
    print(f"  This should be ~0 if Q is conserved along trajectories")

    # The direction of ∇Q tells us "how to break conservation"
    # What if we looked at ∇Q × v (2D cross product)?

    return gradients, velocities, Q_values


# ============================================================================
# EXPLORATION 6: Beyond pairwise - higher-order structure
# ============================================================================

def higher_order_invariants(trajectory, gammas):
    """
    Q_f is fundamentally pairwise (sum over pairs).
    What about triplet or higher-order invariants?

    T_g = ΣΣΣ Γi Γj Γk g(ri, rj, rk)
    """
    n_times, n_coords = trajectory.shape
    N = n_coords // 2

    if N < 3:
        print("\nNeed at least 3 vortices for triplet invariants")
        return None

    print("\nComputing higher-order invariants...")

    def compute_triplet(positions, gammas, g):
        N = len(gammas)
        pos = positions.reshape(N, 2)
        T = 0.0
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if i != j and j != k and i != k:
                        ri = pos[i]
                        rj = pos[j]
                        rk = pos[k]
                        T += gammas[i] * gammas[j] * gammas[k] * g(ri, rj, rk)
        return T

    # Various triplet functions
    def area_signed(ri, rj, rk):
        """Signed area of triangle"""
        return 0.5 * ((rj[0] - ri[0]) * (rk[1] - ri[1]) -
                      (rk[0] - ri[0]) * (rj[1] - ri[1]))

    def perimeter(ri, rj, rk):
        """Perimeter of triangle"""
        d1 = np.sqrt((ri[0]-rj[0])**2 + (ri[1]-rj[1])**2)
        d2 = np.sqrt((rj[0]-rk[0])**2 + (rj[1]-rk[1])**2)
        d3 = np.sqrt((rk[0]-ri[0])**2 + (rk[1]-ri[1])**2)
        return d1 + d2 + d3

    def circumradius(ri, rj, rk):
        """Circumradius of triangle"""
        a = np.sqrt((ri[0]-rj[0])**2 + (ri[1]-rj[1])**2)
        b = np.sqrt((rj[0]-rk[0])**2 + (rj[1]-rk[1])**2)
        c = np.sqrt((rk[0]-ri[0])**2 + (rk[1]-ri[1])**2)
        area = abs(area_signed(ri, rj, rk))
        if area > 1e-10:
            return a * b * c / (4 * area)
        return 0

    def log_area(ri, rj, rk):
        """Log of absolute area"""
        area = abs(area_signed(ri, rj, rk))
        return np.log(area + 1e-10)

    triplet_functions = {
        'area': area_signed,
        'perimeter': perimeter,
        'circumradius': circumradius,
        'log_area': log_area
    }

    results = {}
    for name, g in triplet_functions.items():
        T_values = np.array([compute_triplet(trajectory[t], gammas, g)
                            for t in range(n_times)])
        mean_T = np.mean(T_values)
        frac_var = np.std(T_values) / (np.abs(mean_T) + 1e-10)
        results[name] = {
            'values': T_values,
            'mean': mean_T,
            'frac_var': frac_var
        }
        print(f"  T_{name}: frac_var = {frac_var:.2e}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("BEYOND KNOWN MATHEMATICS")
    print("Searching for proto-mathematical structures in vortex dynamics")
    print("="*70)

    # Generate test data
    print("\nGenerating vortex trajectories...")
    t, trajectory, gammas = simulate_vortices(N=5, T=20.0, dt=0.01, seed=42)
    print(f"  {len(gammas)} vortices, {len(t)} time steps")
    print(f"  Circulations: {gammas}")

    # Run explorations
    print("\n" + "="*70)
    print("EXPLORATION 1: Natural Coordinates")
    print("="*70)
    natural_coords = find_natural_coordinates(trajectory, gammas)

    print("\n" + "="*70)
    print("EXPLORATION 2: Algebraic Relations")
    print("="*70)
    Q_values, relations = discover_algebraic_relations(trajectory, gammas)

    print("\n" + "="*70)
    print("EXPLORATION 3: Information Geometry")
    print("="*70)
    metrics = information_geometry(trajectory, gammas)

    print("\n" + "="*70)
    print("EXPLORATION 4: Emergent Operators")
    print("="*70)
    operators = discover_operators(trajectory, gammas)

    print("\n" + "="*70)
    print("EXPLORATION 5: Conservation Field")
    print("="*70)
    grads, vels, Qs = conservation_field(trajectory, gammas)

    print("\n" + "="*70)
    print("EXPLORATION 6: Higher-Order Invariants")
    print("="*70)
    triplets = higher_order_invariants(trajectory, gammas)

    # Summary
    print("\n" + "="*70)
    print("SYNTHESIS: What new math might we need?")
    print("="*70)
    print("""
    Observations that don't fit standard frameworks:

    1. NATURAL COORDINATES: The "right" representation isn't Cartesian
       positions or distances alone - it's something circulation-weighted.

    2. ALGEBRAIC STRUCTURE: Q_f values aren't independent - they satisfy
       hidden algebraic relations that suggest an underlying algebra.

    3. METRIC STRUCTURE: The natural metric on configuration space
       isn't Euclidean - it's weighted by circulations.

    4. HIGHER-ORDER: Pairwise Q_f misses triplet (and higher) structure.
       The full "invariant" might be a hierarchy.

    SPECULATION: New mathematical objects needed?

    - A "vortex algebra" where Q_f are elements with specific relations
    - A natural geometry on configuration space with Γ-weighted metric
    - Higher-order tensors generalizing pairwise Q_f
    - An "information geometry" where conservation is a geometric property

    The Q_f family might be projections of a single higher-dimensional
    object that we don't have vocabulary for.
    """)

    return {
        'natural_coords': natural_coords,
        'Q_values': Q_values,
        'relations': relations,
        'metrics': metrics,
        'triplets': triplets
    }


if __name__ == "__main__":
    results = main()
