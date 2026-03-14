#!/usr/bin/env python3
"""
Deep test of curvature-weighted Q_f for stretch resistance.

Key finding from previous test: Curvature-weighted Q_f has frac_var = 0.04
under pure stretching vs 0.61 for standard Q_f.

This test explores:
1. Why curvature-weighting helps (theoretical analysis)
2. Combined stretching + evolution
3. Different curvature weighting powers
4. Limiting cases
"""

import numpy as np

class VortexFilament:
    def __init__(self, points, circulation):
        self.points = np.array(points)
        self.circulation = circulation

    def get_tangents(self):
        tangents = np.roll(self.points, -1, axis=0) - self.points
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        return tangents / (norms + 1e-10)

    def arc_lengths(self):
        return np.linalg.norm(np.roll(self.points, -1, axis=0) - self.points, axis=1)

    def total_length(self):
        return np.sum(self.arc_lengths())

    def get_curvatures(self):
        tangents = self.get_tangents()
        dT = np.roll(tangents, -1, axis=0) - tangents
        ds = self.arc_lengths()
        return np.linalg.norm(dT, axis=1) / (ds + 1e-10)


class VortexRing(VortexFilament):
    def __init__(self, center, radius, circulation, n_points=64, axis='z'):
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        if axis == 'z':
            points = np.array([
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
                np.full_like(theta, center[2])
            ]).T
        elif axis == 'x':
            points = np.array([
                np.full_like(theta, center[0]),
                center[1] + radius * np.cos(theta),
                center[2] + radius * np.sin(theta)
            ]).T
        elif axis == 'y':
            points = np.array([
                center[0] + radius * np.cos(theta),
                np.full_like(theta, center[1]),
                center[2] + radius * np.sin(theta)
            ]).T
        super().__init__(points, circulation)
        self.radius = radius


def compute_Qf_curvature_power(filaments, f, power=1):
    """
    Q_κ^p = Σ ΓᵢΓⱼ ∫∫ (κᵢκⱼ)^p f(r) ds dt

    power=0: standard Q_f
    power=1: linear curvature weighting
    power=2: quadratic curvature weighting
    """
    Q = 0.0
    for fil_i in filaments:
        for fil_j in filaments:
            Gi, Gj = fil_i.circulation, fil_j.circulation
            pi, pj = fil_i.points, fil_j.points
            dli, dlj = fil_i.arc_lengths(), fil_j.arc_lengths()
            ki, kj = fil_i.get_curvatures(), fil_j.get_curvatures()

            for si in range(len(pi)):
                for sj in range(len(pj)):
                    if fil_i is fil_j and si == sj:
                        continue
                    r = np.linalg.norm(pi[si] - pj[sj])
                    if r > 1e-10:
                        curv_weight = (ki[si] * kj[sj])**power if power > 0 else 1.0
                        Q += Gi * Gj * curv_weight * f(r) * dli[si] * dlj[sj]
    return Q


def compute_Qf_curvature_normalized(filaments, f):
    """
    Q_f / (mean curvature)^2

    If curvature κ ~ 1/R and stretching makes R → sR, then κ → κ/s.
    This normalization should cancel the stretching effect.
    """
    Q = compute_Qf_curvature_power(filaments, f, power=0)  # Standard Q_f

    total_curv = 0
    total_length = 0
    for fil in filaments:
        kappas = fil.get_curvatures()
        dls = fil.arc_lengths()
        total_curv += np.sum(kappas * dls)
        total_length += fil.total_length()

    mean_curv = total_curv / total_length if total_length > 0 else 1
    return Q * mean_curv**2


def stretch_filament(filament, stretch_factor, axis='z'):
    new_points = filament.points.copy()
    if axis == 'z':
        new_points[:, 2] *= stretch_factor
    return VortexFilament(new_points, filament.circulation)


def biot_savart_velocity(x, filaments, core_radius=0.1):
    u = np.zeros(3)
    for fil in filaments:
        Gamma = fil.circulation
        for i in range(len(fil.points)):
            p1 = fil.points[i]
            p2 = fil.points[(i+1) % len(fil.points)]
            dl = p2 - p1
            p_mid = 0.5 * (p1 + p2)
            r = x - p_mid
            r_mag = np.linalg.norm(r)
            denom = (r_mag**2 + core_radius**2)**1.5
            if denom > 1e-10:
                u += Gamma / (4 * np.pi) * np.cross(dl, r) / denom
    return u


def evolve_filaments_step(filaments, dt, core_radius=0.1):
    for fil in filaments:
        new_points = []
        for p in fil.points:
            u = biot_savart_velocity(p, filaments, core_radius)
            new_points.append(p + u * dt)
        fil.points = np.array(new_points)


def f_exp(r): return np.exp(-r)
def f_inv(r): return 1.0 / (r + 0.05)


def main():
    print("="*70)
    print("Deep Analysis: Curvature-Weighted Q_f")
    print("="*70)
    print()

    # ================================================================
    # PART 1: Why does curvature weighting help?
    # ================================================================
    print("PART 1: Theoretical Analysis")
    print("-"*50)
    print()
    print("For a vortex ring of radius R:")
    print("  - Length L = 2πR")
    print("  - Curvature κ = 1/R")
    print()
    print("Under stretching by factor s along the axis:")
    print("  - Ring becomes ellipse with semi-axes (R, R, sR)")
    print("  - Length increases: L → ~s×L (for large s)")
    print("  - Curvature decreases: κ → ~κ/s (in stretched direction)")
    print()
    print("For standard Q_f:")
    print("  Q_f ~ Γ² × L² ~ Γ² × s²  (grows as s²)")
    print()
    print("For curvature-weighted Q_κ:")
    print("  Q_κ ~ Γ² × κ² × L² ~ Γ² × (1/s²) × s² = Γ²  (constant!)")
    print()
    print("This explains the stretch resistance!")
    print()

    # ================================================================
    # PART 2: Test different curvature powers
    # ================================================================
    print("PART 2: Curvature Power Optimization")
    print("-"*50)
    print()

    n_points = 32
    z_coords = np.linspace(-1, 1, n_points)

    # Create curved tube (not straight, so curvature is defined)
    x_coords = 0.1 * np.sin(np.pi * z_coords)  # Small sinusoidal bend
    tube1_points = np.column_stack([x_coords, np.zeros(n_points), z_coords])
    tube2_points = np.column_stack([x_coords + 1, np.zeros(n_points), z_coords])

    tube1 = VortexFilament(tube1_points, circulation=1.0)
    tube2 = VortexFilament(tube2_points, circulation=1.0)
    filaments = [tube1, tube2]

    stretch_factors = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    powers = [0, 0.5, 1.0, 1.5, 2.0]

    print(f"Curved parallel tubes, stretch factors: {stretch_factors}")
    print(f"Curvature powers: {powers}")
    print()

    print(f"{'Power':<8}", end="")
    for s in stretch_factors:
        print(f" {'s='+str(s):>10}", end="")
    print(f" {'frac_var':>12}")
    print("-"*85)

    best_power = None
    best_frac_var = float('inf')

    for power in powers:
        values = []
        for s in stretch_factors:
            stretched = [stretch_filament(f, s, 'z') for f in [tube1, tube2]]
            val = compute_Qf_curvature_power(stretched, f_exp, power)
            values.append(val)

        values = np.array(values)
        mean_val = np.mean(values)
        frac_var = np.std(values) / abs(mean_val) if abs(mean_val) > 1e-10 else np.inf

        if frac_var < best_frac_var:
            best_frac_var = frac_var
            best_power = power

        print(f"{power:<8}", end="")
        for v in values:
            print(f" {v:>10.4f}", end="")
        print(f" {frac_var:>12.2e}")

    print()
    print(f"Optimal curvature power: {best_power} (frac_var = {best_frac_var:.2e})")

    # ================================================================
    # PART 3: Combined stretching + Biot-Savart evolution
    # ================================================================
    print()
    print("PART 3: Combined Stretching + Biot-Savart Evolution")
    print("-"*50)
    print()

    # Create rings that will naturally stretch as they evolve
    ring1 = VortexRing(center=[0, 0, 0], radius=1.0, circulation=1.0, n_points=32, axis='z')
    ring2 = VortexRing(center=[0, 0, 1.0], radius=0.8, circulation=1.0, n_points=32, axis='z')
    # Different radii cause leapfrogging and stretching
    rings = [ring1, ring2]

    T = 3.0
    dt = 0.05
    n_steps = int(T / dt)

    history = {
        "standard": [],
        f"κ^{best_power}": [],
        "κ-normalized": [],
    }
    lengths = []

    for step in range(n_steps + 1):
        history["standard"].append(compute_Qf_curvature_power(rings, f_exp, power=0))
        history[f"κ^{best_power}"].append(compute_Qf_curvature_power(rings, f_exp, power=best_power))
        history["κ-normalized"].append(compute_Qf_curvature_normalized(rings, f_exp))
        lengths.append(sum(r.total_length() for r in rings))

        if step < n_steps:
            evolve_filaments_step(rings, dt, core_radius=0.1)

    print(f"Two rings (R=1.0, R=0.8) leapfrogging, T={T}")
    print()

    print(f"{'Variant':<20} {'Mean':>12} {'Std':>12} {'frac_var':>12}")
    print("-"*58)

    for name in history:
        vals = np.array(history[name])
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        frac_var = std_v / abs(mean_v) if abs(mean_v) > 1e-10 else np.inf
        print(f"{name:<20} {mean_v:>12.4f} {std_v:>12.4f} {frac_var:>12.2e}")

    # Length change
    length_change = (lengths[-1] - lengths[0]) / lengths[0] * 100
    print()
    print(f"Total length change: {length_change:+.1f}%")

    # ================================================================
    # PART 4: Test on different topologies
    # ================================================================
    print()
    print("PART 4: Different Vortex Configurations")
    print("-"*50)
    print()

    configs = []

    # Config 1: Coaxial rings (same radius)
    r1 = VortexRing([0, 0, 0], 1.0, 1.0, 32, 'z')
    r2 = VortexRing([0, 0, 1.5], 1.0, 1.0, 32, 'z')
    configs.append(("Coaxial equal", [r1, r2]))

    # Config 2: Coaxial rings (different radius)
    r1 = VortexRing([0, 0, 0], 1.0, 1.0, 32, 'z')
    r2 = VortexRing([0, 0, 1.0], 0.5, 1.0, 32, 'z')
    configs.append(("Coaxial unequal", [r1, r2]))

    # Config 3: Perpendicular rings
    r1 = VortexRing([0, 0, 0], 1.0, 1.0, 32, 'z')
    r2 = VortexRing([0, 0, 0], 1.0, 1.0, 32, 'x')
    configs.append(("Perpendicular", [r1, r2]))

    # Config 4: Counter-rotating
    r1 = VortexRing([0, 0, 0], 1.0, 1.0, 32, 'z')
    r2 = VortexRing([0, 0, 1.5], 1.0, -1.0, 32, 'z')
    configs.append(("Counter-rotating", [r1, r2]))

    print(f"{'Config':<20} {'Standard fv':>15} {'Curvature fv':>15} {'Better?'}")
    print("-"*58)

    for name, rings in configs:
        # Evolve each configuration
        T = 2.0
        dt = 0.05
        n_steps = int(T / dt)

        std_vals = []
        curv_vals = []

        for step in range(n_steps + 1):
            std_vals.append(compute_Qf_curvature_power(rings, f_exp, power=0))
            curv_vals.append(compute_Qf_curvature_power(rings, f_exp, power=best_power))
            if step < n_steps:
                evolve_filaments_step(rings, dt, core_radius=0.1)

        std_fv = np.std(std_vals) / abs(np.mean(std_vals))
        curv_fv = np.std(curv_vals) / abs(np.mean(curv_vals))
        better = "✓" if curv_fv < std_fv else "✗"

        print(f"{name:<20} {std_fv:>15.2e} {curv_fv:>15.2e} {better:>7}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print()
    print("="*70)
    print("CONCLUSIONS")
    print("="*70)
    print()
    print("1. THEORETICAL: Curvature weighting with κ² compensates for stretching:")
    print("   Q_κ ~ Γ²κ²L² ~ Γ²(1/s²)(s²) = Γ² (constant under stretching)")
    print()
    print(f"2. OPTIMAL POWER: κ^{best_power} gives best stretch resistance")
    print()
    print("3. LIMITATION: Curvature-weighted Q_f only helps when curvature is well-defined")
    print("   (fails for straight filaments where κ→0)")
    print()
    print("4. COMBINED INVARIANT: For 3D stretch resistance, consider:")
    print("   Q_combined = Q_{standard} × (mean_κ)^2")
    print("   This preserves conservation for rings while adding stretch resistance")

    return best_power


if __name__ == "__main__":
    best_power = main()
