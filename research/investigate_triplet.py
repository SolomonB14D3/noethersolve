#!/usr/bin/env python3
"""
Investigate when T_area is conserved vs not conserved.

The earlier beyond_known_math.py showed T_area with frac_var = 6.54e-07
but triplet_invariant.py showed frac_var ~ 1-100.

What's the difference?
"""

import numpy as np
from scipy.integrate import odeint
from itertools import combinations, permutations
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


def signed_area(ri, rj, rk):
    """Signed area of triangle ijk."""
    return 0.5 * ((rj[0] - ri[0]) * (rk[1] - ri[1]) -
                  (rk[0] - ri[0]) * (rj[1] - ri[1]))


def compute_T_ordered(positions, gammas):
    """
    T_area using i<j<k ordering (combinations).
    """
    N = len(gammas)
    pos = positions.reshape(N, 2)
    T = 0.0

    for i, j, k in combinations(range(N), 3):
        area = signed_area(pos[i], pos[j], pos[k])
        T += gammas[i] * gammas[j] * gammas[k] * area

    return T


def compute_T_full(positions, gammas):
    """
    T_area using full sum over all i,j,k (like beyond_known_math.py).
    """
    N = len(gammas)
    pos = positions.reshape(N, 2)
    T = 0.0

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if i != j and j != k and i != k:
                    area = signed_area(pos[i], pos[j], pos[k])
                    T += gammas[i] * gammas[j] * gammas[k] * area

    return T


def compute_Q_ln(positions, gammas):
    """Standard Q_ln for comparison."""
    N = len(gammas)
    pos = positions.reshape(N, 2)
    Q = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                r = np.sqrt((pos[i,0] - pos[j,0])**2 + (pos[i,1] - pos[j,1])**2)
                Q += gammas[i] * gammas[j] * (-np.log(r + 1e-10))
    return Q


def test_configuration(seed, N, T_sim, label):
    """Test a specific configuration."""
    np.random.seed(seed)

    # Random vortices
    theta = np.random.uniform(0, 2*np.pi, N)
    r = np.sqrt(np.random.uniform(0, 1, N)) * 0.8
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    gammas = np.random.randn(N)
    gammas -= gammas.mean()  # Sum to zero
    gammas /= np.abs(gammas).max()  # Normalize

    # Simulate
    t = np.arange(0, T_sim, 0.01)
    state0 = positions.flatten()
    trajectory = odeint(vortex_rhs, state0, t, args=(gammas,))

    # Compute invariants
    T_ordered = np.array([compute_T_ordered(trajectory[i], gammas) for i in range(len(t))])
    T_full = np.array([compute_T_full(trajectory[i], gammas) for i in range(len(t))])
    Q_ln = np.array([compute_Q_ln(trajectory[i], gammas) for i in range(len(t))])

    # Statistics
    def frac_var(x):
        return np.std(x) / (np.abs(np.mean(x)) + 1e-10)

    print(f"\n{label} (seed={seed}, N={N}, T={T_sim}):")
    print(f"  Circulations: {gammas.round(3)}")
    print(f"  T_ordered frac_var: {frac_var(T_ordered):.2e} (mean={np.mean(T_ordered):.4f})")
    print(f"  T_full frac_var:    {frac_var(T_full):.2e} (mean={np.mean(T_full):.4f})")
    print(f"  Q_ln frac_var:      {frac_var(Q_ln):.2e} (mean={np.mean(Q_ln):.4f})")

    return {
        'T_ordered': T_ordered,
        'T_full': T_full,
        'Q_ln': Q_ln,
        'gammas': gammas
    }


def test_symmetric_configs():
    """Test symmetric vortex configurations where conservation might be exact."""
    print("\n" + "="*60)
    print("TESTING SYMMETRIC CONFIGURATIONS")
    print("="*60)

    # Configuration 1: Equal circulations
    print("\n--- Equal circulations ---")
    N = 4
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = 0.5
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    gammas = np.ones(N)
    gammas -= gammas.mean()  # Now all zero!

    print(f"  Equal gammas with zero mean: gammas = {gammas}")
    print("  This makes T_area = 0 trivially (all gammas are zero)")

    # Configuration 2: Symmetric arrangement with non-trivial gammas
    print("\n--- Square with alternating signs ---")
    N = 4
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = 0.5
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    gammas = np.array([1.0, -1.0, 1.0, -1.0])

    t = np.arange(0, 10.0, 0.01)
    state0 = positions.flatten()
    trajectory = odeint(vortex_rhs, state0, t, args=(gammas,))

    T_ordered = np.array([compute_T_ordered(trajectory[i], gammas) for i in range(len(t))])
    T_full = np.array([compute_T_full(trajectory[i], gammas) for i in range(len(t))])

    def frac_var(x):
        return np.std(x) / (np.abs(np.mean(x)) + 1e-10)

    print(f"  Gammas: {gammas}")
    print(f"  T_ordered: frac_var = {frac_var(T_ordered):.2e}, mean = {np.mean(T_ordered):.6f}")
    print(f"  T_full:    frac_var = {frac_var(T_full):.2e}, mean = {np.mean(T_full):.6f}")


def test_special_cases():
    """Test cases where T_area might be exactly conserved."""
    print("\n" + "="*60)
    print("TESTING SPECIAL CASES")
    print("="*60)

    # Case 1: Equilateral triangle with specific circulations
    print("\n--- Equilateral triangle ---")
    N = 3
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = 0.5
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Try equal circulations (sum to zero impossible with 3)
    gammas = np.array([1.0, 1.0, -2.0])

    t = np.arange(0, 10.0, 0.01)
    state0 = positions.flatten()
    trajectory = odeint(vortex_rhs, state0, t, args=(gammas,))

    T_ordered = np.array([compute_T_ordered(trajectory[i], gammas) for i in range(len(t))])
    Q_ln = np.array([compute_Q_ln(trajectory[i], gammas) for i in range(len(t))])

    def frac_var(x):
        return np.std(x) / (np.abs(np.mean(x)) + 1e-10)

    print(f"  Gammas: {gammas} (sum = {gammas.sum():.3f})")
    print(f"  T_ordered: frac_var = {frac_var(T_ordered):.2e}")
    print(f"  Q_ln:      frac_var = {frac_var(Q_ln):.2e}")

    # Case 2: Co-rotating pair + observer
    print("\n--- Co-rotating pair with observer ---")
    positions = np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]])
    gammas = np.array([1.0, 1.0, -2.0])  # Zero sum

    t = np.arange(0, 10.0, 0.01)
    state0 = positions.flatten()
    trajectory = odeint(vortex_rhs, state0, t, args=(gammas,))

    T_ordered = np.array([compute_T_ordered(trajectory[i], gammas) for i in range(len(t))])

    print(f"  Gammas: {gammas}")
    print(f"  T_ordered: frac_var = {frac_var(T_ordered):.2e}")


def analytical_check():
    """
    Check if T_area can be analytically shown to be non-conserved.
    """
    print("\n" + "="*60)
    print("ANALYTICAL CHECK: Is T_area dT/dt = 0?")
    print("="*60)

    print("""
    T_area = Σ_{triplets} Γi Γj Γk × Area(ijk)

    For signed area:
    Area(i,j,k) = (1/2)[(xj-xi)(yk-yi) - (xk-xi)(yj-yi)]

    Time derivative:
    d/dt Area(ijk) = (1/2)[(ẋj-ẋi)(yk-yi) + (xj-xi)(ẏk-ẏi)
                         - (ẋk-ẋi)(yj-yi) - (xk-xi)(ẏj-ẏi)]

    With vortex velocities:
    ẋi = -Σ_m Γm (yi-ym) / (2π r²im)
    ẏi = +Σ_m Γm (xi-xm) / (2π r²im)

    This gives d/dt Area(ijk) that depends on ALL vortex positions,
    not just i, j, k.

    For T_area to be conserved, we need:
    Σ_{i<j<k} Γi Γj Γk × d/dt[Area(ijk)] = 0

    This is NOT generally true because:
    1. Each d/dt Area(ijk) involves vortices outside {i,j,k}
    2. The Γ³ weighting doesn't cancel with the 1/r² velocity weights

    CONCLUSION: T_area is NOT exactly conserved in general.
    """)

    # Verify numerically for a small case
    print("Numerical verification (N=3 triangle):")
    np.random.seed(123)
    N = 3
    theta = np.random.uniform(0, 2*np.pi, N)
    r = np.sqrt(np.random.uniform(0, 1, N)) * 0.5
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    gammas = np.array([1.0, 0.5, -1.5])

    # Compute dT/dt numerically
    dt = 1e-6
    pos0 = positions.flatten()
    vel = vortex_rhs(pos0, 0, gammas)
    pos1 = pos0 + vel * dt

    T0 = compute_T_ordered(pos0, gammas)
    T1 = compute_T_ordered(pos1, gammas)
    dT_dt = (T1 - T0) / dt

    print(f"  T(0) = {T0:.6f}")
    print(f"  T(dt) = {T1:.6f}")
    print(f"  dT/dt = {dT_dt:.6f}")
    print(f"  |dT/dt| / |T| = {abs(dT_dt) / abs(T0):.2e}")

    if abs(dT_dt) > 1e-6:
        print("\n  *** dT/dt ≠ 0: T_area is NOT exactly conserved ***")


def what_IS_conserved():
    """
    If T_area isn't conserved, what IS the correct triplet invariant?
    """
    print("\n" + "="*60)
    print("SEARCHING FOR CORRECT TRIPLET INVARIANT")
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
    state0 = positions.flatten()
    trajectory = odeint(vortex_rhs, state0, t, args=(gammas,))

    def frac_var(x):
        return np.std(x) / (np.abs(np.mean(x)) + 1e-10)

    # Try different triplet functions
    def compute_triplet_general(positions, gammas, g):
        N = len(gammas)
        pos = positions.reshape(N, 2)
        T = 0.0
        for i, j, k in combinations(range(N), 3):
            T += gammas[i] * gammas[j] * gammas[k] * g(pos[i], pos[j], pos[k])
        return T

    # Different geometric functions
    def area_signed(ri, rj, rk):
        return 0.5 * ((rj[0] - ri[0]) * (rk[1] - ri[1]) -
                      (rk[0] - ri[0]) * (rj[1] - ri[1]))

    def log_perimeter(ri, rj, rk):
        d1 = np.sqrt((ri[0]-rj[0])**2 + (ri[1]-rj[1])**2)
        d2 = np.sqrt((rj[0]-rk[0])**2 + (rj[1]-rk[1])**2)
        d3 = np.sqrt((rk[0]-ri[0])**2 + (rk[1]-ri[1])**2)
        return np.log(d1 * d2 * d3 + 1e-10)

    def sum_log_dist(ri, rj, rk):
        d1 = np.sqrt((ri[0]-rj[0])**2 + (ri[1]-rj[1])**2)
        d2 = np.sqrt((rj[0]-rk[0])**2 + (rj[1]-rk[1])**2)
        d3 = np.sqrt((rk[0]-ri[0])**2 + (rk[1]-ri[1])**2)
        return np.log(d1 + 1e-10) + np.log(d2 + 1e-10) + np.log(d3 + 1e-10)

    def circumradius(ri, rj, rk):
        a = np.sqrt((ri[0]-rj[0])**2 + (ri[1]-rj[1])**2)
        b = np.sqrt((rj[0]-rk[0])**2 + (rj[1]-rk[1])**2)
        c = np.sqrt((rk[0]-ri[0])**2 + (rk[1]-ri[1])**2)
        area = abs(area_signed(ri, rj, rk))
        if area > 1e-10:
            return np.log(a * b * c / (4 * area) + 1e-10)
        return 0

    def product_distances(ri, rj, rk):
        d1 = np.sqrt((ri[0]-rj[0])**2 + (ri[1]-rj[1])**2)
        d2 = np.sqrt((rj[0]-rk[0])**2 + (rj[1]-rk[1])**2)
        d3 = np.sqrt((rk[0]-ri[0])**2 + (rk[1]-ri[1])**2)
        return d1 * d2 * d3

    candidates = {
        'area': area_signed,
        'log_perimeter': log_perimeter,
        'sum_log_dist': sum_log_dist,
        'circumradius': circumradius,
        'product_dist': product_distances,
    }

    print(f"Testing triplet invariants (N={N}, seed=42):\n")
    for name, g in candidates.items():
        T_values = np.array([compute_triplet_general(trajectory[i], gammas, g)
                            for i in range(len(t))])
        fv = frac_var(T_values)
        print(f"  T_{name:15s}: frac_var = {fv:.2e}")

    # For comparison, also compute pairwise Q_f
    print("\nPairwise Q_f for comparison:")
    for fname, f in [('ln', lambda r: -np.log(r + 1e-10)),
                     ('sqrt', lambda r: np.sqrt(r)),
                     ('exp', lambda r: np.exp(-r))]:
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

        Q_values = np.array([compute_qf(trajectory[i], gammas, f) for i in range(len(t))])
        print(f"  Q_{fname:5s}: frac_var = {frac_var(Q_values):.2e}")


def main():
    print("="*70)
    print("INVESTIGATING T_AREA CONSERVATION")
    print("="*70)

    # Test same config as beyond_known_math.py
    print("\n" + "="*60)
    print("TEST 1: Same config as beyond_known_math.py (seed=42, N=5, T=20)")
    print("="*60)
    test_configuration(42, 5, 20.0, "beyond_known_math config")

    # Test the random configs from triplet_invariant.py
    print("\n" + "="*60)
    print("TEST 2: Random configs (like triplet_invariant.py)")
    print("="*60)
    for seed in [300, 400, 500]:
        test_configuration(seed, 5, 30.0, f"Random config")

    # Symmetric configs
    test_symmetric_configs()

    # Special cases
    test_special_cases()

    # Analytical check
    analytical_check()

    # What IS conserved?
    what_IS_conserved()


if __name__ == "__main__":
    main()
