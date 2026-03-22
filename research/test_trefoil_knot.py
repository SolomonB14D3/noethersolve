#!/usr/bin/env python3
"""
Test Q_f conservation for a TREFOIL KNOT vortex filament.

The trefoil is self-linked with self-linking number (writhe + twist).
Key question: Does self-linking affect Q_f conservation similarly to linked pairs?
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Callable


@dataclass
class VortexFilament:
    """A vortex filament in 3D (arbitrary curve)."""
    points: np.ndarray  # (N, 3) array
    circulation: float

    def arc_lengths(self):
        """Compute arc length segments."""
        dl = np.linalg.norm(np.roll(self.points, -1, axis=0) - self.points, axis=1)
        return dl

    def get_tangents(self):
        """Unit tangent vectors at each point."""
        tangents = np.roll(self.points, -1, axis=0) - self.points
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        return tangents / (norms + 1e-10)

    def get_curvatures(self):
        """Curvature at each point."""
        t = self.get_tangents()
        dt = np.roll(t, -1, axis=0) - t
        dl = self.arc_lengths()
        kappa = np.linalg.norm(dt, axis=1) / (dl + 1e-10)
        return kappa

    def total_length(self):
        """Total arc length."""
        return np.sum(self.arc_lengths())


def create_trefoil(scale: float = 1.0, circulation: float = 1.0, n_points: int = 128) -> VortexFilament:
    """
    Create a trefoil knot vortex filament.

    Parametric equations:
    x(t) = sin(t) + 2*sin(2t)
    y(t) = cos(t) - 2*cos(2t)
    z(t) = -sin(3t)

    This is a (2,3)-torus knot with self-linking.
    """
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    x = scale * (np.sin(t) + 2*np.sin(2*t))
    y = scale * (np.cos(t) - 2*np.cos(2*t))
    z = scale * (-np.sin(3*t))

    points = np.column_stack([x, y, z])
    return VortexFilament(points=points, circulation=circulation)


def create_figure_eight_knot(scale: float = 1.0, circulation: float = 1.0, n_points: int = 128) -> VortexFilament:
    """
    Create a figure-eight knot (4_1 knot).

    Parametric equations (Lissajous-like):
    x(t) = (2 + cos(2t)) * cos(3t)
    y(t) = (2 + cos(2t)) * sin(3t)
    z(t) = sin(4t)
    """
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    x = scale * (2 + np.cos(2*t)) * np.cos(3*t)
    y = scale * (2 + np.cos(2*t)) * np.sin(3*t)
    z = scale * np.sin(4*t)

    points = np.column_stack([x, y, z])
    return VortexFilament(points=points, circulation=circulation)


def create_unknot_circle(radius: float = 1.0, circulation: float = 1.0, n_points: int = 128) -> VortexFilament:
    """Create an unknot (simple circle) for comparison."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros_like(t)

    points = np.column_stack([x, y, z])
    return VortexFilament(points=points, circulation=circulation)


def compute_writhe(filament: VortexFilament) -> float:
    """
    Compute the writhe of a closed curve using the Gauss integral.

    Wr = (1/4π) ∫∫ (r(s) - r(t)) · (dr/ds × dr/dt) / |r(s) - r(t)|³ ds dt

    Writhe measures the "coiling" of the curve in 3D.
    """
    points = filament.points
    n = len(points)
    Wr = 0.0

    for i in range(n):
        p1 = points[i]
        p1_next = points[(i + 1) % n]
        dr1 = p1_next - p1

        for j in range(n):
            if abs(i - j) < 2 or abs(i - j) > n - 2:
                continue  # Skip nearby segments (regularization)

            p2 = points[j]
            p2_next = points[(j + 1) % n]
            dr2 = p2_next - p2

            r1_mid = 0.5 * (p1 + p1_next)
            r2_mid = 0.5 * (p2 + p2_next)

            r = r1_mid - r2_mid
            r_mag = np.linalg.norm(r)

            if r_mag > 1e-10:
                cross = np.cross(dr1, dr2)
                Wr += np.dot(r, cross) / (r_mag ** 3)

    return Wr / (4 * np.pi)


def biot_savart_velocity(x: np.ndarray, filaments: List[VortexFilament], core_radius: float = 0.1) -> np.ndarray:
    """Compute velocity at point x due to all vortex filaments."""
    u = np.zeros(3)

    for fil in filaments:
        Gamma = fil.circulation
        points = fil.points
        n = len(points)

        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            dl = p2 - p1

            p_mid = 0.5 * (p1 + p2)
            r = x - p_mid
            r_mag = np.linalg.norm(r)

            denom = (r_mag**2 + core_radius**2) ** 1.5
            if denom > 1e-10:
                u += Gamma / (4 * np.pi) * np.cross(dl, r) / denom

    return u


def compute_Qf_self(filament: VortexFilament, f: Callable[[float], float]) -> float:
    """
    Compute self-interaction Q_f for a single filament.

    Q_f = Γ² ∫∫ f(|γ(s) - γ(t)|) ds dt  (s ≠ t)
    """
    Q = 0.0
    Gamma = filament.circulation
    points = filament.points
    dl = filament.arc_lengths()
    n = len(points)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Skip adjacent points to avoid singularity
            if abs(i - j) < 2 or abs(i - j) > n - 2:
                continue

            r = np.linalg.norm(points[i] - points[j])
            if r > 1e-10:
                Q += Gamma**2 * f(r) * dl[i] * dl[j]

    return Q


def compute_Qf_curvature_self(filament: VortexFilament, f: Callable[[float], float]) -> float:
    """Curvature-weighted self-interaction Q_κ."""
    Q = 0.0
    Gamma = filament.circulation
    points = filament.points
    dl = filament.arc_lengths()
    kappa = filament.get_curvatures()
    n = len(points)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(i - j) < 2 or abs(i - j) > n - 2:
                continue

            r = np.linalg.norm(points[i] - points[j])
            if r > 1e-10:
                Q += Gamma**2 * kappa[i] * kappa[j] * f(r) * dl[i] * dl[j]

    return Q


def compute_helicity_self(filament: VortexFilament) -> float:
    """
    Self-helicity for a knotted filament.

    For a thin vortex tube: H = Γ² × (Wr + Tw)
    where Wr = writhe, Tw = twist.

    For a filament with no internal twist: H ≈ Γ² × Wr
    """
    Wr = compute_writhe(filament)
    return filament.circulation**2 * Wr


def evolve_filament(filament: VortexFilament, dt: float, core_radius: float = 0.15):
    """Evolve filament by one time step using Biot-Savart self-induction."""
    filaments = [filament]
    new_points = []

    for p in filament.points:
        u = biot_savart_velocity(p, filaments, core_radius)
        new_points.append(p + u * dt)

    filament.points = np.array(new_points)


def frac_var(values):
    """Fractional variation: std / |mean|."""
    arr = np.asarray(values)
    mean = np.mean(arr)
    if abs(mean) < 1e-15:
        return float(np.std(arr))
    return float(np.std(arr) / abs(mean))


# Test functions
def f_inv(r): return 1.0 / (r + 0.1)
def f_exp(r): return np.exp(-r)
def f_sqrt(r): return np.sqrt(r)
def f_linear(r): return r


def run_knot_test(filament: VortexFilament, name: str, T: float = 0.5, dt: float = 0.01):
    """Run evolution test on a knotted filament."""
    n_steps = int(T / dt)

    # Initial properties
    Wr_initial = compute_writhe(filament)
    H_initial = compute_helicity_self(filament)
    L_initial = filament.total_length()

    test_functions = [
        ("Q_{1/r}", f_inv),
        ("Q_{e^-r}", f_exp),
        ("Q_√r", f_sqrt),
        ("Q_r", f_linear),
    ]

    history = {fn[0]: [] for fn in test_functions}
    history["Q_κ(1/r)"] = []
    history["Writhe"] = []
    history["Length"] = []

    for step in range(n_steps + 1):
        for fn_name, f in test_functions:
            Q = compute_Qf_self(filament, f)
            history[fn_name].append(Q)

        Qk = compute_Qf_curvature_self(filament, f_inv)
        history["Q_κ(1/r)"].append(Qk)

        Wr = compute_writhe(filament)
        history["Writhe"].append(Wr)

        L = filament.total_length()
        history["Length"].append(L)

        if step < n_steps:
            evolve_filament(filament, dt, core_radius=0.15)

    # Results
    print(f"\n{'='*65}")
    print(f"KNOT: {name}")
    print(f"Initial writhe: {Wr_initial:.3f}")
    print(f"Initial self-helicity proxy: {H_initial:.3f}")
    print(f"Initial length: {L_initial:.3f}")
    print(f"Evolution: T={T}, dt={dt}, steps={n_steps}")
    print(f"{'='*65}")
    print(f"{'Quantity':<15} {'Initial':>12} {'Final':>12} {'frac_var':>12} {'Status':>8}")
    print("-" * 65)

    results = {}
    for qname in list(history.keys()):
        values = history[qname]
        fv = frac_var(values)
        results[qname] = fv

        status = "PASS" if fv < 0.01 else ("~" if fv < 0.05 else "FAIL")
        print(f"{qname:<15} {values[0]:>12.4f} {values[-1]:>12.4f} {fv:>12.2e} {status:>8}")

    return results, history, Wr_initial


def main():
    print("=" * 70)
    print("TREFOIL KNOT Q_f CONSERVATION TEST")
    print("=" * 70)
    print("\nKey question: Does self-linking (writhe) affect Q_f conservation?")
    print("Hypothesis: Positive writhe → good conservation (like positive helicity)")

    # Test 1: Unknot (circle) - control
    print("\n" + "=" * 70)
    print("TEST 1: UNKNOT (CIRCLE) - Control")
    print("=" * 70)
    unknot = create_unknot_circle(radius=2.0, circulation=1.0, n_points=96)
    results_unknot, _, Wr_unknot = run_knot_test(unknot, "Unknot (circle)")

    # Test 2: Trefoil knot (positive writhe)
    print("\n" + "=" * 70)
    print("TEST 2: TREFOIL KNOT (right-handed, positive writhe)")
    print("=" * 70)
    trefoil = create_trefoil(scale=1.0, circulation=1.0, n_points=128)
    results_trefoil, _, Wr_trefoil = run_knot_test(trefoil, "Trefoil (2,3)")

    # Test 3: Mirror trefoil (negative writhe)
    print("\n" + "=" * 70)
    print("TEST 3: MIRROR TREFOIL (left-handed, negative writhe)")
    print("=" * 70)
    trefoil_mirror = create_trefoil(scale=1.0, circulation=1.0, n_points=128)
    # Mirror by flipping z coordinates
    trefoil_mirror.points[:, 2] *= -1
    results_mirror, _, Wr_mirror = run_knot_test(trefoil_mirror, "Mirror Trefoil")

    # Test 4: Figure-eight knot (zero writhe, amphichiral)
    print("\n" + "=" * 70)
    print("TEST 4: FIGURE-EIGHT KNOT (amphichiral, zero writhe)")
    print("=" * 70)
    fig8 = create_figure_eight_knot(scale=0.8, circulation=1.0, n_points=128)
    results_fig8, _, Wr_fig8 = run_knot_test(fig8, "Figure-8 (4_1)")

    # Test 5: Opposite circulation trefoil
    print("\n" + "=" * 70)
    print("TEST 5: TREFOIL WITH Γ = -1")
    print("=" * 70)
    trefoil_neg = create_trefoil(scale=1.0, circulation=-1.0, n_points=128)
    results_neg, _, Wr_neg = run_knot_test(trefoil_neg, "Trefoil Γ=-1")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Comparing Q_f conservation across knot types")
    print("=" * 70)
    print(f"{'Knot':<20} {'Writhe':>8} {'Γ':>5} {'Γ²×Wr':>8} {'Q_{{1/r}}':>12}")
    print("-" * 60)

    all_results = [
        ("Unknot", Wr_unknot, 1.0, results_unknot),
        ("Trefoil", Wr_trefoil, 1.0, results_trefoil),
        ("Mirror Trefoil", Wr_mirror, 1.0, results_mirror),
        ("Figure-8", Wr_fig8, 1.0, results_fig8),
        ("Trefoil Γ=-1", Wr_neg, -1.0, results_neg),
    ]

    for name, Wr, Gamma, res in all_results:
        H_proxy = Gamma**2 * Wr
        fv = res["Q_{1/r}"]
        print(f"{name:<20} {Wr:>8.3f} {Gamma:>5.1f} {H_proxy:>8.3f} {fv:>12.2e}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Check if writhe correlates with conservation
    writhes = [Wr_trefoil, Wr_mirror, Wr_fig8]
    fvs = [results_trefoil["Q_{1/r}"], results_mirror["Q_{1/r}"], results_fig8["Q_{1/r}"]]

    print(f"\nWrithe values: trefoil={Wr_trefoil:.3f}, mirror={Wr_mirror:.3f}, fig8={Wr_fig8:.3f}")
    print(f"Q_{{1/r}} frac_var: trefoil={results_trefoil['Q_{1/r}']:.2e}, "
          f"mirror={results_mirror['Q_{1/r}']:.2e}, fig8={results_fig8['Q_{1/r}']:.2e}")

    # Compare to linked rings
    print("\nComparison to linked ring results:")
    print("  Linked pairs: H>0 → 3.45e-04, H<0 → 4.36e-03")
    print(f"  Trefoil (Wr>0): {results_trefoil['Q_{1/r}']:.2e}")
    print(f"  Mirror (Wr<0): {results_mirror['Q_{1/r}']:.2e}")

    return {
        "unknot": results_unknot,
        "trefoil": results_trefoil,
        "mirror": results_mirror,
        "fig8": results_fig8,
        "neg": results_neg
    }


if __name__ == "__main__":
    results = main()
