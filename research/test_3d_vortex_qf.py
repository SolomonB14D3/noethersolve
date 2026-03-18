#!/usr/bin/env python3
"""
Test Q_f conservation for 3D vortex filaments.

We use the Biot-Savart law to evolve vortex rings and test whether
Q_f = Σᵢⱼ ΓᵢΓⱼ ∫∫ f(|γᵢ(s) - γⱼ(t)|) ds dt
is approximately conserved.

Simplest case: Two coaxial vortex rings (same circulation direction).
"""

import numpy as np

# ============================================================================
# Vortex Ring Class
# ============================================================================

class VortexRing:
    """A circular vortex ring in 3D."""

    def __init__(self, center, radius, circulation, n_points=64, axis='z'):
        """
        center: (x, y, z) center of ring
        radius: ring radius R
        circulation: Γ
        axis: normal direction ('x', 'y', or 'z')
        """
        self.circulation = circulation
        self.n_points = n_points

        # Parameterize ring
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)

        if axis == 'z':
            self.points = np.array([
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
                np.full_like(theta, center[2])
            ]).T
        elif axis == 'x':
            self.points = np.array([
                np.full_like(theta, center[0]),
                center[1] + radius * np.cos(theta),
                center[2] + radius * np.sin(theta)
            ]).T
        elif axis == 'y':
            self.points = np.array([
                center[0] + radius * np.cos(theta),
                np.full_like(theta, center[1]),
                center[2] + radius * np.sin(theta)
            ]).T

    def get_tangents(self):
        """Compute unit tangent vectors at each point."""
        # Forward difference for tangent
        tangents = np.roll(self.points, -1, axis=0) - self.points
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        return tangents / (norms + 1e-10)

    def arc_lengths(self):
        """Compute arc length segments."""
        dl = np.linalg.norm(np.roll(self.points, -1, axis=0) - self.points, axis=1)
        return dl


# ============================================================================
# Biot-Savart Velocity
# ============================================================================

def biot_savart_velocity(x, rings, core_radius=0.1):
    """
    Compute velocity at point x due to all vortex rings using Biot-Savart.

    Uses regularized kernel to avoid singularity:
    u(x) = (Γ/4π) ∫ (dl × r) / (|r|² + a²)^(3/2)
    """
    u = np.zeros(3)

    for ring in rings:
        Gamma = ring.circulation
        points = ring.points
        n = len(points)

        for i in range(n):
            # Segment from point i to i+1
            p1 = points[i]
            p2 = points[(i+1) % n]
            dl = p2 - p1

            # Midpoint approximation
            p_mid = 0.5 * (p1 + p2)
            r = x - p_mid
            r_mag = np.linalg.norm(r)

            # Regularized Biot-Savart kernel
            denom = (r_mag**2 + core_radius**2)**(1.5)
            if denom > 1e-10:
                u += Gamma / (4 * np.pi) * np.cross(dl, r) / denom

    return u


# ============================================================================
# Q_f Computation
# ============================================================================

def compute_Qf_rings(rings, f):
    """
    Compute Q_f = Σᵢⱼ ΓᵢΓⱼ ∫∫ f(|γᵢ(s) - γⱼ(t)|) ds dt

    For discrete rings, this becomes:
    Q_f = Σᵢⱼ ΓᵢΓⱼ Σₛₜ f(|pᵢₛ - pⱼₜ|) dlᵢₛ dlⱼₜ
    """
    Q = 0.0

    for i, ring_i in enumerate(rings):
        Gamma_i = ring_i.circulation
        points_i = ring_i.points
        dl_i = ring_i.arc_lengths()

        for j, ring_j in enumerate(rings):
            Gamma_j = ring_j.circulation
            points_j = ring_j.points
            dl_j = ring_j.arc_lengths()

            # Compute double sum
            for si, pi in enumerate(points_i):
                for tj, pj in enumerate(points_j):
                    if i == j and si == tj:
                        continue  # Skip self-interaction at same point
                    r = np.linalg.norm(pi - pj)
                    if r > 1e-10:
                        Q += Gamma_i * Gamma_j * f(r) * dl_i[si] * dl_j[tj]

    return Q


def compute_kinetic_energy(rings, core_radius=0.1, grid_size=20):
    """Approximate kinetic energy using Biot-Savart."""
    # Create a grid around the rings
    all_points = np.vstack([r.points for r in rings])
    x_min, x_max = all_points[:,0].min() - 1, all_points[:,0].max() + 1
    y_min, y_max = all_points[:,1].min() - 1, all_points[:,1].max() + 1
    z_min, z_max = all_points[:,2].min() - 1, all_points[:,2].max() + 1

    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    z = np.linspace(z_min, z_max, grid_size)

    dx = (x_max - x_min) / grid_size
    dy = (y_max - y_min) / grid_size
    dz = (z_max - z_min) / grid_size
    dV = dx * dy * dz

    E = 0.0
    for xi in x:
        for yi in y:
            for zi in z:
                u = biot_savart_velocity(np.array([xi, yi, zi]), rings, core_radius)
                E += 0.5 * np.dot(u, u) * dV

    return E


# ============================================================================
# Time Evolution (Simplified)
# ============================================================================

def evolve_rings_step(rings, dt, core_radius=0.1):
    """Evolve rings by one time step using Biot-Savart velocity."""
    for ring in rings:
        new_points = []
        for p in ring.points:
            u = biot_savart_velocity(p, rings, core_radius)
            new_points.append(p + u * dt)
        ring.points = np.array(new_points)


# ============================================================================
# Test Functions
# ============================================================================

def f_linear(r): return r
def f_sqrt(r): return np.sqrt(r)
def f_inv(r): return 1.0 / (r + 0.01)  # Regularized 1/r
def f_exp(r): return np.exp(-r)
def f_gaussian(r): return np.exp(-r**2 / 2)


# ============================================================================
# Main Test
# ============================================================================

def main():
    print("="*60)
    print("3D Vortex Ring Q_f Conservation Test")
    print("="*60)
    print()

    # Create two coaxial vortex rings
    ring1 = VortexRing(center=[0, 0, 0], radius=1.0, circulation=1.0, n_points=32, axis='z')
    ring2 = VortexRing(center=[0, 0, 1.5], radius=1.0, circulation=1.0, n_points=32, axis='z')
    rings = [ring1, ring2]

    print("Initial configuration:")
    print("  Ring 1: center=(0,0,0), R=1.0, Γ=1.0")
    print("  Ring 2: center=(0,0,1.5), R=1.0, Γ=1.0")
    print("  Points per ring: 32")
    print()

    # Test functions
    test_functions = [
        ("r", f_linear),
        ("√r", f_sqrt),
        ("1/r", f_inv),
        ("e^(-r)", f_exp),
        ("e^(-r²/2)", f_gaussian),
    ]

    # Time evolution
    T = 2.0
    dt = 0.05
    n_steps = int(T / dt)

    print(f"Evolving for T={T} with dt={dt} ({n_steps} steps)...")
    print()

    # Track Q_f over time
    Qf_history = {name: [] for name, _ in test_functions}
    times = []

    for step in range(n_steps + 1):
        t = step * dt
        times.append(t)

        # Compute Q_f for each test function
        for name, f in test_functions:
            Qf = compute_Qf_rings(rings, f)
            Qf_history[name].append(Qf)

        # Evolve
        if step < n_steps:
            evolve_rings_step(rings, dt, core_radius=0.1)

    # Compute conservation metrics
    print("="*60)
    print("RESULTS: Q_f Conservation for 3D Vortex Rings")
    print("="*60)
    print()
    print(f"{'f(r)':<12} {'Mean Q_f':>12} {'Std':>10} {'frac_var':>12} {'Status':>8}")
    print("-"*56)

    results = {}
    for name, _ in test_functions:
        Qf_values = np.array(Qf_history[name])
        mean_Qf = np.mean(Qf_values)
        std_Qf = np.std(Qf_values)
        frac_var = std_Qf / abs(mean_Qf) if abs(mean_Qf) > 1e-10 else np.inf

        status = "✓" if frac_var < 0.1 else "✗"
        results[name] = frac_var

        print(f"{name:<12} {mean_Qf:>12.4f} {std_Qf:>10.4f} {frac_var:>12.2e} {status:>8}")

    print()
    print("Pass threshold: frac_var < 0.1 (relaxed for 3D)")

    # Summary
    n_pass = sum(1 for fv in results.values() if fv < 0.1)
    print()
    print(f"Summary: {n_pass}/{len(results)} test functions show approximate conservation")

    # Identify best performers
    print()
    print("Best performers:")
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for name, fv in sorted_results[:3]:
        print(f"  {name}: frac_var = {fv:.2e}")

    return results, Qf_history, times


if __name__ == "__main__":
    results, history, times = main()
