#!/usr/bin/env python3
"""
The Green's Function Principle: Why Q_{-ln(r)} is Special

HYPOTHESIS: The optimal f(r) for Q_f conservation is always the Green's
function of the relevant differential operator.

In 2D: -ln(r) is the Green's function of the Laplacian (Δ G = δ)
In 3D: 1/r is the Green's function of the Laplacian

This suggests a UNIFYING PRINCIPLE: Conservation laws are tied to
fundamental solutions of the governing PDEs.

This script explores this connection and whether it suggests new math.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.special import kn  # Modified Bessel function
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


def test_greens_functions():
    """
    Test conservation with different Green's functions.
    """
    print("="*70)
    print("TESTING GREEN'S FUNCTION PRINCIPLE")
    print("="*70)

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

    # Different Green's functions
    greens_functions = {
        # 2D Laplacian: Δ G = δ
        '-ln(r) [2D Laplacian]': lambda r: -np.log(r + 1e-10),

        # 3D Laplacian: Δ G = δ
        '1/r [3D Laplacian]': lambda r: 1.0 / (r + 0.1),

        # Helmholtz: (Δ + k²) G = δ → K₀(kr)
        'K₀(r) [Helmholtz k=1]': lambda r: kn(0, r + 0.1),
        'K₀(2r) [Helmholtz k=2]': lambda r: kn(0, 2*r + 0.1),

        # 2D Biharmonic: Δ² G = δ → r² ln(r)
        'r²ln(r) [Biharmonic]': lambda r: (r**2) * np.log(r + 1e-10),

        # Screened Poisson: (Δ - k²) G = δ → e^(-kr)
        'e^(-r) [Screened k=1]': lambda r: np.exp(-r),
        'e^(-2r) [Screened k=2]': lambda r: np.exp(-2*r),

        # For comparison: non-Green's functions
        'r [not Green]': lambda r: r,
        'r² [not Green]': lambda r: r**2,
        'tanh(r) [not Green]': lambda r: np.tanh(r),
    }

    print("\nGreen's functions vs regular functions:")
    print("-" * 50)

    results = []
    for name, f in greens_functions.items():
        Q_values = np.array([compute_Q_f(trajectory[i], gammas, f)
                            for i in range(len(t))])
        fv = frac_var(Q_values)
        is_green = 'Green' in name or 'Laplacian' in name or 'Helmholtz' in name or 'Biharmonic' in name or 'Screened' in name
        results.append((name, fv, is_green))
        print(f"  {name:30s}: frac_var = {fv:.2e}")

    # Analysis
    print("\n" + "-" * 50)
    print("ANALYSIS:")

    green_fvs = [fv for _, fv, is_green in results if is_green]
    other_fvs = [fv for _, fv, is_green in results if not is_green]

    print(f"  Green's functions mean frac_var: {np.mean(green_fvs):.2e}")
    print(f"  Non-Green's functions mean frac_var: {np.mean(other_fvs):.2e}")
    print(f"  Ratio: {np.mean(other_fvs) / np.mean(green_fvs):.1f}x")


def explore_operator_connection():
    """
    The connection between Green's function and conservation might be
    related to the HAMILTONIAN structure.

    H = (1/4π) ΣΣ Γi Γj (-ln|ri - rj|)

    This is EXACTLY Q_{-ln(r)} / 4π!

    The Hamiltonian generates time evolution via Poisson brackets.
    Conservation of H (energy) is EXACTLY conservation of Q_{-ln(r)}.
    """
    print("\n" + "="*70)
    print("OPERATOR-HAMILTONIAN CONNECTION")
    print("="*70)

    print("""
    The point vortex Hamiltonian is:

        H = (1/4π) Σ_{i≠j} Γi Γj (-ln|ri - rj|)

    This equals Q_{-ln(r)} / 4π!

    The equations of motion are:

        Γi dxi/dt = ∂H/∂yi
        Γi dyi/dt = -∂H/∂xi

    So Q_{-ln(r)} = 4π H, and its conservation is just energy conservation.

    WHY is -ln(r) special?

    Because it's the STREAM FUNCTION of a point vortex:
        ψ(r) = (Γ/2π) ln|r|
        u = -∂ψ/∂y,  v = ∂ψ/∂x

    The stream function satisfies:
        Δψ = ω  (vorticity equation)

    For a point vortex at origin:
        Δψ = Γ δ(r)

    So ψ = (Γ/2π) ln|r| is the GREEN'S FUNCTION of the Laplacian!

    The Green's function principle:
    ---------------------------------
    The optimal Q_f is the one where f is the Green's function of the
    operator that appears in the governing equation.

    For 2D Euler:  Δψ = ω  →  G = -ln(r)  →  Q_{-ln(r)} = H
    For 3D:        Δψ = ω  →  G = 1/r     →  Q_{1/r} should be optimal
    """)


def test_modified_dynamics():
    """
    If we modify the dynamics (different operator), does the optimal
    Q_f change accordingly?

    Test: Screened Poisson dynamics (like Yukawa interaction)
        (Δ - κ²) ψ = ω

    The Green's function is K₀(κr), a modified Bessel function.
    """
    print("\n" + "="*70)
    print("TESTING MODIFIED DYNAMICS: SCREENED POISSON")
    print("="*70)

    def screened_vortex_rhs(state, t, gammas, kappa):
        """
        Modified dynamics where velocity is derived from screened potential.
        ψ = Σ Γi K₀(κ|r - ri|)
        """
        N = len(gammas)
        positions = state.reshape(N, 2)
        velocities = np.zeros_like(positions)

        for i in range(N):
            for j in range(N):
                if i != j:
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    r = np.sqrt(dx**2 + dy**2)
                    if r > 0.01:
                        # K₀'(x) = -K₁(x)
                        # Velocity magnitude = (Γ/2π) × κ × K₁(κr)
                        k1_kr = kn(1, kappa * r)
                        v_mag = gammas[j] * kappa * k1_kr / (2 * np.pi)
                        # Perpendicular direction
                        velocities[i, 0] += -v_mag * dy / r
                        velocities[i, 1] += v_mag * dx / r

        return velocities.flatten()

    kappa = 1.0  # Screening parameter

    np.random.seed(42)
    N = 5

    theta = np.random.uniform(0, 2*np.pi, N)
    r = np.sqrt(np.random.uniform(0, 1, N)) * 0.5  # Smaller region
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    gammas = np.random.randn(N)
    gammas -= gammas.mean()
    gammas /= np.abs(gammas).max()

    t = np.arange(0, 20.0, 0.02)

    # Simulate both dynamics
    print(f"  Screening parameter κ = {kappa}")
    print("\n  Standard dynamics (Q_{-ln(r)} should be best):")
    traj_std = odeint(vortex_rhs, positions.flatten(), t, args=(gammas,))

    Q_ln_std = np.array([compute_Q_f(traj_std[i], gammas, lambda r: -np.log(r + 1e-10))
                         for i in range(len(t))])
    Q_k0_std = np.array([compute_Q_f(traj_std[i], gammas, lambda r: kn(0, r + 0.1))
                         for i in range(len(t))])

    print(f"    Q_{{-ln(r)}}: frac_var = {frac_var(Q_ln_std):.2e}")
    print(f"    Q_{{K₀(r)}}: frac_var = {frac_var(Q_k0_std):.2e}")

    print("\n  Screened dynamics (Q_{K₀(κr)} should be best):")
    traj_scr = odeint(screened_vortex_rhs, positions.flatten(), t, args=(gammas, kappa))

    Q_ln_scr = np.array([compute_Q_f(traj_scr[i], gammas, lambda r: -np.log(r + 1e-10))
                         for i in range(len(t))])
    Q_k0_scr = np.array([compute_Q_f(traj_scr[i], gammas, lambda r: kn(0, kappa * r + 0.1))
                         for i in range(len(t))])

    print(f"    Q_{{-ln(r)}}: frac_var = {frac_var(Q_ln_scr):.2e}")
    print(f"    Q_{{K₀(κr)}}: frac_var = {frac_var(Q_k0_scr):.2e}")


def unifying_framework():
    """
    Propose a unifying framework.
    """
    print("\n" + "="*70)
    print("UNIFYING FRAMEWORK: Operator-Conservation Correspondence")
    print("="*70)

    print("""
    PRINCIPLE: For any linear PDE with Green's function G(r),
    the pairwise sum Q_G = Σ Γi Γj G(|ri - rj|) is conserved
    under dynamics derived from that PDE.

    This is NOT new mathematics, but it IS an organizing principle
    that connects:

    1. PDEs and their Green's functions
    2. Conservation laws in particle systems
    3. Hamiltonian structure

    EXAMPLES:

    | PDE                | Green's function | Conserved Q_f   | System        |
    |--------------------|------------------|-----------------|---------------|
    | Δψ = ω (2D)       | -ln(r)/(2π)      | Q_{-ln(r)} = H | Point vortices|
    | Δψ = ω (3D)       | 1/(4πr)          | Q_{1/r}        | Vortex filaments|
    | (Δ-κ²)ψ = ω       | K₀(κr)/(2π)      | Q_{K₀}         | Screened vortices|
    | ∂ψ/∂t = Δψ        | G(r,t)           | ?              | Diffusing vortices|

    WHAT'S "BEYOND KNOWN MATH" HERE?

    The question isn't "what new invariants exist?" but rather
    "what's the unified theory that explains the Q_f family?"

    The answer involves:
    - Symplectic geometry (Poisson manifolds)
    - Infinite-dimensional Lie groups (area-preserving diffeomorphisms)
    - Moment maps (Q_f as moment of symmetry action)

    This mathematics IS known, just not commonly applied here.

    The INSIGHT is that the Green's function principle provides
    a SELECTION RULE: among all possible f(r), the physics
    (via the PDE) picks out the Green's function as special.
    """)


def main():
    test_greens_functions()
    explore_operator_connection()
    test_modified_dynamics()
    unifying_framework()

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
    The Green's function principle explains WHY Q_{-ln(r)} is special:
    it's the Hamiltonian, and -ln(r) is the Green's function of the
    Laplacian that governs the dynamics.

    This isn't "beyond known math" in the sense of needing new axioms,
    but it IS a unifying perspective that connects:
    - PDEs (Laplacian, Helmholtz, etc.)
    - Green's functions (fundamental solutions)
    - Conserved quantities (Hamiltonians)
    - Particle systems (vortices, charges, etc.)

    The "new vocabulary" might be:
    - "Operator-induced conservation laws"
    - "Green's function invariants"
    - "PDE-Hamiltonian correspondence"
    """)


if __name__ == "__main__":
    main()
