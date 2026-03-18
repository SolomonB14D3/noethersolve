#!/usr/bin/env python3
"""
Test modified Q_f variants that might survive vortex stretching in 3D.

The challenge: Standard Q_f ∝ s² under stretching (s = stretch factor).

Candidates to test:
1. Helicity-hybrid: Incorporate cross-product structure from helicity
2. Enstrophy-normalized: Q_f / Ω (divide by enstrophy)
3. Energy-normalized: Q_f / E (divide by energy)
4. Stretch-compensated: Q_f / L² where L is total filament length
5. Circulation-density: Weight by Γ/L instead of Γ
6. Alignment-filtered: Only count anti-aligned contributions
7. Gradient-weighted: Weight by |dT/ds| (curvature)
"""

import numpy as np

# ============================================================================
# Vortex Filament Classes
# ============================================================================

class VortexFilament:
    """A general vortex filament in 3D."""

    def __init__(self, points, circulation):
        """
        points: Nx3 array of filament positions
        circulation: scalar circulation Γ
        """
        self.points = np.array(points)
        self.circulation = circulation
        self.n = len(points)

    def get_tangents(self):
        """Unit tangent vectors."""
        tangents = np.roll(self.points, -1, axis=0) - self.points
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        return tangents / (norms + 1e-10)

    def arc_lengths(self):
        """Arc length of each segment."""
        return np.linalg.norm(np.roll(self.points, -1, axis=0) - self.points, axis=1)

    def total_length(self):
        """Total filament length."""
        return np.sum(self.arc_lengths())

    def get_curvatures(self):
        """Curvature κ = |dT/ds| at each point."""
        tangents = self.get_tangents()
        dT = np.roll(tangents, -1, axis=0) - tangents
        ds = self.arc_lengths()
        return np.linalg.norm(dT, axis=1) / (ds + 1e-10)


class VortexRing(VortexFilament):
    """Circular vortex ring."""

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

        super().__init__(points, circulation)
        self.center = np.array(center)
        self.radius = radius


# ============================================================================
# Standard Quantities
# ============================================================================

def compute_energy(filaments, reg=0.1):
    """Approximate kinetic energy from filament interactions."""
    E = 0.0
    for i, fil_i in enumerate(filaments):
        for j, fil_j in enumerate(filaments):
            Gi, Gj = fil_i.circulation, fil_j.circulation
            pi, pj = fil_i.points, fil_j.points
            dli, dlj = fil_i.arc_lengths(), fil_j.arc_lengths()
            Ti, Tj = fil_i.get_tangents(), fil_j.get_tangents()

            for si in range(len(pi)):
                for sj in range(len(pj)):
                    if i == j and si == sj:
                        continue
                    r = np.linalg.norm(pi[si] - pj[sj])
                    if r > 1e-10:
                        # Energy kernel: T·T / r
                        dot = np.dot(Ti[si], Tj[sj])
                        E += Gi * Gj * dot / (r + reg) * dli[si] * dlj[sj]
    return E / (8 * np.pi)


def compute_enstrophy_proxy(filaments):
    """
    Enstrophy proxy for filaments.
    For thin tubes: Ω ~ Σ Γ² / A where A is cross-section.
    We use Γ² × L as a proxy (assuming fixed core radius).
    """
    return sum(f.circulation**2 * f.total_length() for f in filaments)


def compute_helicity(filaments, reg=0.1):
    """
    Helicity H = ∫ u·ω d³x
    For filaments: H ~ Σᵢⱼ ΓᵢΓⱼ ∫∫ Tᵢ·[Tⱼ × (xᵢ-xⱼ)] / |xᵢ-xⱼ|³ ds dt
    """
    H = 0.0
    for i, fil_i in enumerate(filaments):
        for j, fil_j in enumerate(filaments):
            Gi, Gj = fil_i.circulation, fil_j.circulation
            pi, pj = fil_i.points, fil_j.points
            dli, dlj = fil_i.arc_lengths(), fil_j.arc_lengths()
            Ti, Tj = fil_i.get_tangents(), fil_j.get_tangents()

            for si in range(len(pi)):
                for sj in range(len(pj)):
                    if i == j and si == sj:
                        continue
                    rvec = pi[si] - pj[sj]
                    r = np.linalg.norm(rvec)
                    if r > reg:
                        # H kernel: T_i · (T_j × r) / r³
                        cross = np.cross(Tj[sj], rvec)
                        H += Gi * Gj * np.dot(Ti[si], cross) / r**3 * dli[si] * dlj[sj]
    return H / (4 * np.pi)


# ============================================================================
# Modified Q_f Variants
# ============================================================================

def compute_Qf_standard(filaments, f):
    """Standard Q_f = Σ ΓᵢΓⱼ ∫∫ f(r) ds dt"""
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


def compute_Qf_helicity_hybrid(filaments, f, alpha=0.5):
    """
    Helicity-Q_f hybrid:
    Q_H = Σ ΓᵢΓⱼ ∫∫ [α(Tᵢ·Tⱼ) + (1-α)|Tᵢ×Tⱼ|] f(r) ds dt

    This combines alignment (Q_f-like) with orthogonality (helicity-like).
    """
    Q = 0.0
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
                    if r > 1e-10:
                        dot = np.dot(Ti[si], Tj[sj])
                        cross_mag = np.linalg.norm(np.cross(Ti[si], Tj[sj]))
                        weight = alpha * abs(dot) + (1 - alpha) * cross_mag
                        Q += Gi * Gj * weight * f(r) * dli[si] * dlj[sj]
    return Q


def compute_Qf_enstrophy_normalized(filaments, f):
    """Q_f / Ω - normalized by enstrophy proxy."""
    Qf = compute_Qf_standard(filaments, f)
    Omega = compute_enstrophy_proxy(filaments)
    return Qf / Omega if Omega > 1e-10 else 0


def compute_Qf_energy_normalized(filaments, f):
    """Q_f / E - normalized by energy."""
    Qf = compute_Qf_standard(filaments, f)
    E = compute_energy(filaments)
    return Qf / E if abs(E) > 1e-10 else 0


def compute_Qf_length_normalized(filaments, f):
    """Q_f / L² - normalized by total length squared."""
    Qf = compute_Qf_standard(filaments, f)
    L = sum(fil.total_length() for fil in filaments)
    return Qf / L**2 if L > 1e-10 else 0


def compute_Qf_circulation_density(filaments, f):
    """
    Use circulation density Γ/L instead of Γ.
    Q_ρ = Σ (Γᵢ/Lᵢ)(Γⱼ/Lⱼ) ∫∫ f(r) ds dt
    """
    Q = 0.0
    for fil_i in filaments:
        for fil_j in filaments:
            Li = fil_i.total_length()
            Lj = fil_j.total_length()
            rho_i = fil_i.circulation / Li if Li > 0 else 0
            rho_j = fil_j.circulation / Lj if Lj > 0 else 0

            pi, pj = fil_i.points, fil_j.points
            dli, dlj = fil_i.arc_lengths(), fil_j.arc_lengths()

            for si in range(len(pi)):
                for sj in range(len(pj)):
                    if fil_i is fil_j and si == sj:
                        continue
                    r = np.linalg.norm(pi[si] - pj[sj])
                    if r > 1e-10:
                        Q += rho_i * rho_j * f(r) * dli[si] * dlj[sj]
    return Q


def compute_Qf_curvature_weighted(filaments, f):
    """
    Weight by curvature: emphasizes bent regions.
    Q_κ = Σ ΓᵢΓⱼ ∫∫ κᵢκⱼ f(r) ds dt
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
                        Q += Gi * Gj * ki[si] * kj[sj] * f(r) * dli[si] * dlj[sj]
    return Q


def compute_Qf_anti_aligned(filaments, f):
    """
    Only count anti-aligned contributions (T_i · T_j < 0).
    This might cancel under stretching.
    """
    Q = 0.0
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
                    dot = np.dot(Ti[si], Tj[sj])
                    if r > 1e-10 and dot < 0:  # Only anti-aligned
                        Q += Gi * Gj * abs(dot) * f(r) * dli[si] * dlj[sj]
    return Q


# ============================================================================
# Stretching Simulation
# ============================================================================

def stretch_filament(filament, stretch_factor, axis='z'):
    """
    Stretch a filament along an axis.
    - Length increases by stretch_factor
    - Cross-section area decreases (volume conserved)
    - Circulation stays constant
    """
    new_points = filament.points.copy()
    if axis == 'z':
        new_points[:, 2] = filament.points[:, 2] * stretch_factor
    elif axis == 'x':
        new_points[:, 0] = filament.points[:, 0] * stretch_factor

    return VortexFilament(new_points, filament.circulation)


def biot_savart_velocity(x, filaments, core_radius=0.1):
    """Biot-Savart velocity."""
    u = np.zeros(3)
    for fil in filaments:
        Gamma = fil.circulation
        points = fil.points
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i+1) % len(points)]
            dl = p2 - p1
            p_mid = 0.5 * (p1 + p2)
            r = x - p_mid
            r_mag = np.linalg.norm(r)
            denom = (r_mag**2 + core_radius**2)**1.5
            if denom > 1e-10:
                u += Gamma / (4 * np.pi) * np.cross(dl, r) / denom
    return u


def evolve_filaments_step(filaments, dt, core_radius=0.1):
    """Evolve filaments by one timestep."""
    for fil in filaments:
        new_points = []
        for p in fil.points:
            u = biot_savart_velocity(p, filaments, core_radius)
            new_points.append(p + u * dt)
        fil.points = np.array(new_points)


# ============================================================================
# Test Functions
# ============================================================================

def f_inv(r): return 1.0 / (r + 0.05)
def f_exp(r): return np.exp(-r)
def f_sqrt(r): return np.sqrt(r + 0.01)
def f_linear(r): return r


# ============================================================================
# Main Test
# ============================================================================

def main():
    print("="*70)
    print("Testing Stretch-Resistant Q_f Modifications")
    print("="*70)
    print()

    # Test 1: Pure stretching (artificial, to isolate stretching effect)
    print("TEST 1: Artificial Pure Stretching")
    print("-"*50)
    print()

    # Create two parallel vortex tubes
    n_points = 32
    z_coords = np.linspace(-1, 1, n_points)

    tube1_points = np.column_stack([
        np.zeros(n_points),
        np.zeros(n_points),
        z_coords
    ])
    tube2_points = np.column_stack([
        np.ones(n_points),
        np.zeros(n_points),
        z_coords
    ])

    tube1 = VortexFilament(tube1_points, circulation=1.0)
    tube2 = VortexFilament(tube2_points, circulation=1.0)

    # Compute variants for different stretch factors
    stretch_factors = [1.0, 1.5, 2.0, 3.0, 4.0]

    variants = [
        ("Standard Q_f", lambda fils, f: compute_Qf_standard(fils, f)),
        ("Length-norm Q_f/L²", lambda fils, f: compute_Qf_length_normalized(fils, f)),
        ("Enstrophy-norm Q_f/Ω", lambda fils, f: compute_Qf_enstrophy_normalized(fils, f)),
        ("Circ-density Q_ρ", lambda fils, f: compute_Qf_circulation_density(fils, f)),
        ("Helicity-hybrid", lambda fils, f: compute_Qf_helicity_hybrid(fils, f, alpha=0.5)),
        ("Curvature-weighted", lambda fils, f: compute_Qf_curvature_weighted(fils, f)),
    ]

    f = f_exp  # Use exponential kernel

    print(f"f(r) = e^(-r), Two parallel tubes with stretch factors: {stretch_factors}")
    print()

    print(f"{'Variant':<25}", end="")
    for s in stretch_factors:
        print(f" {'s='+str(s):>10}", end="")
    print(f" {'frac_var':>12}")
    print("-"*85)

    stretch_results = {}
    for name, compute_func in variants:
        values = []
        for s in stretch_factors:
            # Create stretched filaments
            stretched_fils = [stretch_filament(f, s, 'z') for f in [tube1, tube2]]
            val = compute_func(stretched_fils, f_exp)
            values.append(val)

        # Compute fractional variance
        values = np.array(values)
        mean_val = np.mean(values)
        frac_var = np.std(values) / abs(mean_val) if abs(mean_val) > 1e-10 else np.inf
        stretch_results[name] = frac_var

        print(f"{name:<25}", end="")
        for v in values:
            print(f" {v:>10.4f}", end="")
        print(f" {frac_var:>12.2e}")

    # Test 2: Biot-Savart evolution of coaxial rings
    print()
    print("TEST 2: Biot-Savart Evolution (Coaxial Rings)")
    print("-"*50)
    print()

    ring1 = VortexRing(center=[0, 0, 0], radius=1.0, circulation=1.0, n_points=32, axis='z')
    ring2 = VortexRing(center=[0, 0, 1.5], radius=1.0, circulation=1.0, n_points=32, axis='z')
    rings = [ring1, ring2]

    # Time evolution
    T = 2.0
    dt = 0.05
    n_steps = int(T / dt)

    history = {name: [] for name, _ in variants}

    for step in range(n_steps + 1):
        for name, compute_func in variants:
            val = compute_func(rings, f_exp)
            history[name].append(val)

        if step < n_steps:
            evolve_filaments_step(rings, dt, core_radius=0.1)

    print(f"f(r) = e^(-r), T={T}, dt={dt}")
    print()

    print(f"{'Variant':<25} {'Mean':>12} {'Std':>12} {'frac_var':>12} {'Status'}")
    print("-"*70)

    evolution_results = {}
    for name, _ in variants:
        vals = np.array(history[name])
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        frac_var = std_v / abs(mean_v) if abs(mean_v) > 1e-10 else np.inf
        evolution_results[name] = frac_var
        status = "✓" if frac_var < 0.05 else "✗"
        print(f"{name:<25} {mean_v:>12.4f} {std_v:>12.4f} {frac_var:>12.2e} {status:>6}")

    # Test 3: Different f(r) functions with best variant
    print()
    print("TEST 3: Best Variant Across Different f(r)")
    print("-"*50)
    print()

    # Reset rings
    ring1 = VortexRing(center=[0, 0, 0], radius=1.0, circulation=1.0, n_points=32, axis='z')
    ring2 = VortexRing(center=[0, 0, 1.5], radius=1.0, circulation=1.0, n_points=32, axis='z')
    rings = [ring1, ring2]

    test_fs = [
        ("1/r", f_inv),
        ("e^(-r)", f_exp),
        ("√r", f_sqrt),
        ("r", f_linear),
    ]

    # Find best variant from Test 2
    best_variant_name = min(evolution_results, key=evolution_results.get)
    best_variant = next(func for name, func in variants if name == best_variant_name)

    print(f"Best variant from Test 2: {best_variant_name}")
    print()

    history_f = {fname: {"standard": [], "best": []} for fname, _ in test_fs}

    for step in range(n_steps + 1):
        for fname, f in test_fs:
            history_f[fname]["standard"].append(compute_Qf_standard(rings, f))
            history_f[fname]["best"].append(best_variant(rings, f))

        if step < n_steps:
            evolve_filaments_step(rings, dt, core_radius=0.1)

    print(f"{'f(r)':<12} {'Standard frac_var':>18} {'Best variant frac_var':>22} {'Improvement'}")
    print("-"*60)

    for fname, _ in test_fs:
        std_vals = np.array(history_f[fname]["standard"])
        best_vals = np.array(history_f[fname]["best"])

        std_fv = np.std(std_vals) / abs(np.mean(std_vals)) if abs(np.mean(std_vals)) > 1e-10 else np.inf
        best_fv = np.std(best_vals) / abs(np.mean(best_vals)) if abs(np.mean(best_vals)) > 1e-10 else np.inf

        improvement = (std_fv - best_fv) / std_fv * 100 if std_fv > 0 else 0

        print(f"{fname:<12} {std_fv:>18.2e} {best_fv:>22.2e} {improvement:>10.1f}%")

    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    # Sort by stretch resistance
    sorted_stretch = sorted(stretch_results.items(), key=lambda x: x[1])
    print("Most stretch-resistant (Test 1 - artificial stretching):")
    for name, fv in sorted_stretch[:3]:
        print(f"  {name}: frac_var = {fv:.2e}")

    print()

    # Sort by evolution conservation
    sorted_evolution = sorted(evolution_results.items(), key=lambda x: x[1])
    print("Best conserved (Test 2 - Biot-Savart evolution):")
    for name, fv in sorted_evolution[:3]:
        print(f"  {name}: frac_var = {fv:.2e}")

    print()

    # Key finding
    best_stretch = sorted_stretch[0][0]
    best_evolution = sorted_evolution[0][0]

    if best_stretch == best_evolution:
        print(f"FINDING: '{best_stretch}' is best for both stretching AND evolution!")
    else:
        print("FINDING: Different variants optimal for different tests:")
        print(f"  Stretching resistance: {best_stretch}")
        print(f"  Evolution conservation: {best_evolution}")

    return stretch_results, evolution_results


if __name__ == "__main__":
    stretch_results, evolution_results = main()
