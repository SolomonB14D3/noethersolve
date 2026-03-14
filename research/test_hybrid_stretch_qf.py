#!/usr/bin/env python3
"""
Test hybrid Q_f that adapts to stretching.

Key insight from previous tests:
- Standard Q_f: Best during normal evolution (frac_var 0.0014)
- Curvature Q_f: Best under pure stretching (frac_var 0.20)

Hybrid approaches to test:
1. Enstrophy-normalized: Q_f / Ω  (Ω tracks stretching)
2. Length-curvature balanced: Q_f × κ_mean / L
3. Helicity-regularized: Q_f + λ×H (helicity is topological)
4. Energy-residual: Q_f - α×E (subtract energy contribution)
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

    def mean_curvature(self):
        kappas = self.get_curvatures()
        dls = self.arc_lengths()
        return np.sum(kappas * dls) / self.total_length()


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
        else:
            points = np.array([
                center[0] + radius * np.cos(theta),
                np.full_like(theta, center[1]),
                center[2] + radius * np.sin(theta)
            ]).T
        super().__init__(points, circulation)
        self.radius = radius


def compute_Qf_standard(filaments, f):
    Q = 0.0
    for fil_i in filaments:
        for fil_j in filaments:
            Gi, Gj = fil_i.circulation, fil_j.circulation
            pi, pj = fil_i.points, fil_j.points
            dli, dlj = fil_i.arc_lengths(), fil_j.arc_lengths()
            for si in range(len(pi)):
                for sj in range(len(pj)):
                    if fil_i is fil_j and si == sj:
                        continue
                    r = np.linalg.norm(pi[si] - pj[sj])
                    if r > 1e-10:
                        Q += Gi * Gj * f(r) * dli[si] * dlj[sj]
    return Q


def compute_energy(filaments, reg=0.1):
    E = 0.0
    for fil_i in filaments:
        for fil_j in filaments:
            Gi, Gj = fil_i.circulation, fil_j.circulation
            pi, pj = fil_i.points, fil_j.points
            dli, dlj = fil_i.arc_lengths(), fil_j.arc_lengths()
            Ti, Tj = fil_i.get_tangents(), fil_j.get_tangents()
            for si in range(len(pi)):
                for sj in range(len(pj)):
                    if fil_i is fil_j and si == sj:
                        continue
                    r = np.linalg.norm(pi[si] - pj[sj])
                    if r > reg:
                        dot = np.dot(Ti[si], Tj[sj])
                        E += Gi * Gj * dot / r * dli[si] * dlj[sj]
    return E / (8 * np.pi)


def compute_helicity(filaments, reg=0.1):
    H = 0.0
    for fil_i in filaments:
        for fil_j in filaments:
            Gi, Gj = fil_i.circulation, fil_j.circulation
            pi, pj = fil_i.points, fil_j.points
            dli, dlj = fil_i.arc_lengths(), fil_j.arc_lengths()
            Ti, Tj = fil_i.get_tangents(), fil_j.get_tangents()
            for si in range(len(pi)):
                for sj in range(len(pj)):
                    if fil_i is fil_j and si == sj:
                        continue
                    rvec = pi[si] - pj[sj]
                    r = np.linalg.norm(rvec)
                    if r > reg:
                        cross = np.cross(Tj[sj], rvec)
                        H += Gi * Gj * np.dot(Ti[si], cross) / r**3 * dli[si] * dlj[sj]
    return H / (4 * np.pi)


def compute_enstrophy_proxy(filaments):
    return sum(f.circulation**2 * f.total_length() for f in filaments)


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
    print("Hybrid Stretch-Resistant Q_f")
    print("="*70)
    print()

    # Define hybrid variants
    def Q_standard(fils): return compute_Qf_standard(fils, f_exp)
    def Q_enstrophy_norm(fils): return compute_Qf_standard(fils, f_exp) / compute_enstrophy_proxy(fils)

    def Q_length_curv_balanced(fils):
        Q = compute_Qf_standard(fils, f_exp)
        L = sum(f.total_length() for f in fils)
        kappa = sum(f.mean_curvature() * f.total_length() for f in fils) / L
        return Q * kappa / L

    def Q_energy_residual(fils):
        Q = compute_Qf_standard(fils, f_exp)
        E = compute_energy(fils)
        return Q - 10 * E  # Subtract energy contribution

    def Q_helicity_combo(fils):
        Q = compute_Qf_standard(fils, f_exp)
        H = compute_helicity(fils)
        return Q + 0.1 * abs(H)  # Add helicity (always positive)

    def Q_ratio(fils):
        # Q_exp / Q_inv - ratio of two Q_f
        Q_e = compute_Qf_standard(fils, f_exp)
        Q_i = compute_Qf_standard(fils, f_inv)
        return Q_e / Q_i if abs(Q_i) > 1e-10 else 0

    variants = [
        ("Standard Q_f", Q_standard),
        ("Q_f / Ω", Q_enstrophy_norm),
        ("Q_f × κ / L", Q_length_curv_balanced),
        ("Q_f - 10E", Q_energy_residual),
        ("Q_f + 0.1|H|", Q_helicity_combo),
        ("Q_exp / Q_inv", Q_ratio),
    ]

    # ================================================================
    # TEST 1: Pure Stretching
    # ================================================================
    print("TEST 1: Pure Stretching")
    print("-"*50)
    print()

    n_points = 32
    z_coords = np.linspace(-1, 1, n_points)
    x_coords = 0.1 * np.sin(np.pi * z_coords)

    tube1 = VortexFilament(np.column_stack([x_coords, np.zeros(n_points), z_coords]), 1.0)
    tube2 = VortexFilament(np.column_stack([x_coords + 1, np.zeros(n_points), z_coords]), 1.0)

    stretch_factors = [1.0, 1.5, 2.0, 3.0, 4.0]

    print(f"{'Variant':<20}", end="")
    for s in stretch_factors:
        print(f" {'s='+str(s):>10}", end="")
    print(f" {'frac_var':>12}")
    print("-"*85)

    stretch_results = {}
    for name, func in variants:
        values = []
        for s in stretch_factors:
            stretched = [stretch_filament(f, s, 'z') for f in [tube1, tube2]]
            values.append(func(stretched))

        values = np.array(values)
        mean_val = np.mean(values)
        frac_var = np.std(values) / abs(mean_val) if abs(mean_val) > 1e-10 else np.inf
        stretch_results[name] = frac_var

        print(f"{name:<20}", end="")
        for v in values:
            print(f" {v:>10.4f}", end="")
        print(f" {frac_var:>12.2e}")

    # ================================================================
    # TEST 2: Biot-Savart Evolution
    # ================================================================
    print()
    print("TEST 2: Biot-Savart Evolution (Leapfrogging Rings)")
    print("-"*50)
    print()

    ring1 = VortexRing([0, 0, 0], 1.0, 1.0, 32, 'z')
    ring2 = VortexRing([0, 0, 1.0], 0.8, 1.0, 32, 'z')
    rings = [ring1, ring2]

    T = 3.0
    dt = 0.05
    n_steps = int(T / dt)

    history = {name: [] for name, _ in variants}

    for step in range(n_steps + 1):
        for name, func in variants:
            history[name].append(func(rings))
        if step < n_steps:
            evolve_filaments_step(rings, dt, core_radius=0.1)

    print(f"{'Variant':<20} {'Mean':>12} {'Std':>12} {'frac_var':>12}")
    print("-"*58)

    evolution_results = {}
    for name, _ in variants:
        vals = np.array(history[name])
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        frac_var = std_v / abs(mean_v) if abs(mean_v) > 1e-10 else np.inf
        evolution_results[name] = frac_var
        print(f"{name:<20} {mean_v:>12.4f} {std_v:>12.4f} {frac_var:>12.2e}")

    # ================================================================
    # TEST 3: Combined Score
    # ================================================================
    print()
    print("TEST 3: Combined Score (Stretch + Evolution)")
    print("-"*50)
    print()

    print(f"{'Variant':<20} {'Stretch fv':>12} {'Evol fv':>12} {'Combined':>12}")
    print("-"*58)

    combined = {}
    for name, _ in variants:
        s_fv = stretch_results[name]
        e_fv = evolution_results[name]
        # Geometric mean of the two fractional variances
        c = np.sqrt(s_fv * e_fv)
        combined[name] = c
        print(f"{name:<20} {s_fv:>12.2e} {e_fv:>12.2e} {c:>12.2e}")

    # Best overall
    best = min(combined, key=combined.get)
    print()
    print(f"Best overall: {best} (combined frac_var = {combined[best]:.2e})")

    # ================================================================
    # SUMMARY
    # ================================================================
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    sorted_stretch = sorted(stretch_results.items(), key=lambda x: x[1])
    sorted_evolution = sorted(evolution_results.items(), key=lambda x: x[1])
    sorted_combined = sorted(combined.items(), key=lambda x: x[1])

    print("Best for stretching resistance:")
    for name, fv in sorted_stretch[:2]:
        print(f"  {name}: {fv:.2e}")

    print()
    print("Best for evolution conservation:")
    for name, fv in sorted_evolution[:2]:
        print(f"  {name}: {fv:.2e}")

    print()
    print("Best overall (balanced):")
    for name, fv in sorted_combined[:2]:
        print(f"  {name}: {fv:.2e}")

    return stretch_results, evolution_results, combined


if __name__ == "__main__":
    s, e, c = main()
