#!/usr/bin/env python3
"""
Test aligned Q_f in 3D: Q_f^{aligned} = ∫∫ (ω(x)·ω(y))^p f(|x-y|) dx dy

The hypothesis is that weighting by alignment of vorticity vectors
might give better conservation under stretching.
"""

import numpy as np

class VortexRing:
    """A circular vortex ring in 3D."""

    def __init__(self, center, radius, circulation, n_points=64, axis='z'):
        self.circulation = circulation
        self.n_points = n_points
        self.center = np.array(center)
        self.radius = radius

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

    def get_tangents(self):
        """Compute unit tangent vectors at each point."""
        tangents = np.roll(self.points, -1, axis=0) - self.points
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        return tangents / (norms + 1e-10)

    def arc_lengths(self):
        """Compute arc length segments."""
        return np.linalg.norm(np.roll(self.points, -1, axis=0) - self.points, axis=1)


def compute_Qf_aligned(rings, f, power=1):
    """
    Compute Q_f^{aligned} = Σᵢⱼ ΓᵢΓⱼ ∫∫ (Tᵢ·Tⱼ)^p f(|γᵢ(s) - γⱼ(t)|) ds dt

    For vortex filaments:
    - Γᵢ is circulation
    - Tᵢ is unit tangent
    - The (T·T)^p term weights by alignment
    """
    Q = 0.0

    for i, ring_i in enumerate(rings):
        Gamma_i = ring_i.circulation
        points_i = ring_i.points
        tangents_i = ring_i.get_tangents()
        dl_i = ring_i.arc_lengths()

        for j, ring_j in enumerate(rings):
            Gamma_j = ring_j.circulation
            points_j = ring_j.points
            tangents_j = ring_j.get_tangents()
            dl_j = ring_j.arc_lengths()

            for si in range(len(points_i)):
                pi = points_i[si]
                Ti = tangents_i[si]

                for tj in range(len(points_j)):
                    if i == j and si == tj:
                        continue

                    pj = points_j[tj]
                    Tj = tangents_j[tj]

                    r = np.linalg.norm(pi - pj)
                    if r > 1e-10:
                        # Alignment factor
                        dot = np.dot(Ti, Tj)
                        alignment = abs(dot)**power

                        Q += Gamma_i * Gamma_j * alignment * f(r) * dl_i[si] * dl_j[tj]

    return Q


def compute_Qf_standard(rings, f):
    """Standard Q_f without alignment weighting."""
    return compute_Qf_aligned(rings, f, power=0)  # power=0 means alignment=1 always


def biot_savart_velocity(x, rings, core_radius=0.1):
    """Compute velocity at point x due to all vortex rings."""
    u = np.zeros(3)
    for ring in rings:
        Gamma = ring.circulation
        points = ring.points
        n = len(points)
        for i in range(n):
            p1 = points[i]
            p2 = points[(i+1) % n]
            dl = p2 - p1
            p_mid = 0.5 * (p1 + p2)
            r = x - p_mid
            r_mag = np.linalg.norm(r)
            denom = (r_mag**2 + core_radius**2)**(1.5)
            if denom > 1e-10:
                u += Gamma / (4 * np.pi) * np.cross(dl, r) / denom
    return u


def evolve_rings_step(rings, dt, core_radius=0.1):
    """Evolve rings by one time step."""
    for ring in rings:
        new_points = []
        for p in ring.points:
            u = biot_savart_velocity(p, rings, core_radius)
            new_points.append(p + u * dt)
        ring.points = np.array(new_points)


# Test functions
def f_linear(r): return r
def f_sqrt(r): return np.sqrt(r)
def f_inv(r): return 1.0 / (r + 0.01)
def f_exp(r): return np.exp(-r)


def main():
    print("="*70)
    print("3D Aligned Q_f Conservation Test")
    print("="*70)
    print()
    print("Testing whether alignment-weighted Q_f:")
    print("  Q_f^p = ∫∫ |T_i·T_j|^p × ΓᵢΓⱼ f(|xᵢ-xⱼ|) ds dt")
    print("is better conserved than standard Q_f.")
    print()

    # Create two coaxial vortex rings
    ring1 = VortexRing(center=[0, 0, 0], radius=1.0, circulation=1.0, n_points=32, axis='z')
    ring2 = VortexRing(center=[0, 0, 1.5], radius=1.0, circulation=1.0, n_points=32, axis='z')
    rings = [ring1, ring2]

    print("Configuration: Two coaxial vortex rings")
    print("  Ring 1: center=(0,0,0), R=1.0, Γ=1.0")
    print("  Ring 2: center=(0,0,1.5), R=1.0, Γ=1.0")
    print()

    # Test functions and powers
    test_functions = [
        ("1/r", f_inv),
        ("e^(-r)", f_exp),
        ("√r", f_sqrt),
        ("r", f_linear),
    ]

    powers = [0, 1, 2]  # 0 = standard, 1 = linear alignment, 2 = quadratic alignment

    # Time evolution
    T = 2.0
    dt = 0.05
    n_steps = int(T / dt)

    print(f"Evolving for T={T} with dt={dt}...")
    print()

    # Track Q_f over time
    history = {}
    for name, _ in test_functions:
        for p in powers:
            history[(name, p)] = []

    times = []

    for step in range(n_steps + 1):
        t = step * dt
        times.append(t)

        for name, f in test_functions:
            for p in powers:
                Qf = compute_Qf_aligned(rings, f, power=p)
                history[(name, p)].append(Qf)

        if step < n_steps:
            evolve_rings_step(rings, dt, core_radius=0.1)

    # Compute conservation metrics
    print("="*70)
    print("RESULTS: Q_f Conservation vs Alignment Power")
    print("="*70)
    print()

    print(f"{'f(r)':<12} {'p=0 (std)':>12} {'p=1':>12} {'p=2':>12}")
    print("-"*50)

    results = {}
    for name, _ in test_functions:
        row = []
        for p in powers:
            Qf_values = np.array(history[(name, p)])
            mean_Qf = np.mean(Qf_values)
            std_Qf = np.std(Qf_values)
            frac_var = std_Qf / abs(mean_Qf) if abs(mean_Qf) > 1e-10 else np.inf
            row.append(frac_var)
            results[(name, p)] = frac_var

        # Find best
        best_p = powers[np.argmin(row)]
        markers = ["*" if p == best_p else "" for p in powers]

        print(f"{name:<12}", end="")
        for i, p in enumerate(powers):
            print(f" {row[i]:>10.2e}{markers[i]}", end="")
        print()

    print()
    print("* = best alignment power for this f(r)")

    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    # Count wins for each power
    wins = {p: 0 for p in powers}
    for name, _ in test_functions:
        row = [results[(name, p)] for p in powers]
        best_p = powers[np.argmin(row)]
        wins[best_p] += 1

    print("Number of wins per alignment power:")
    for p in powers:
        label = "standard" if p == 0 else f"p={p}"
        print(f"  {label}: {wins[p]}/{len(test_functions)}")

    print()

    # Check if alignment helps
    alignment_helps = wins[1] + wins[2] > wins[0]
    if alignment_helps:
        print("FINDING: Alignment weighting IMPROVES conservation for some f(r)!")
        print()
        print("This suggests that in 3D, weighting by vorticity alignment")
        print("could provide stronger conservation laws.")
    else:
        print("FINDING: Standard Q_f (no alignment) performs best.")
        print()
        print("Alignment weighting does not improve conservation for this configuration.")

    # Best overall
    print()
    print("Best overall performers:")
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for (name, p), fv in sorted_results[:5]:
        label = "standard" if p == 0 else f"p={p}"
        print(f"  {name} ({label}): frac_var = {fv:.2e}")

    return history, results


if __name__ == "__main__":
    history, results = main()
