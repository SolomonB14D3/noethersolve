#!/usr/bin/env python3
"""
Test Q_f conservation for LINKED vortex rings (Hopf link).

Compares:
1. Unlinked coaxial rings (control)
2. Linked rings (Hopf link configuration)

Key question: Does topological linking affect Q_f conservation?
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Callable


@dataclass
class VortexRing:
    """A circular vortex ring in 3D."""
    points: np.ndarray  # (N, 3) array of ring points
    circulation: float

    @classmethod
    def create_xy_ring(cls, center, radius, circulation, n_points=48):
        """Ring in XY plane (normal along Z)."""
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        points = np.array([
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            np.full_like(theta, center[2])
        ]).T
        return cls(points=points, circulation=circulation)

    @classmethod
    def create_xz_ring(cls, center, radius, circulation, n_points=48):
        """Ring in XZ plane (normal along Y)."""
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        points = np.array([
            center[0] + radius * np.cos(theta),
            np.full_like(theta, center[1]),
            center[2] + radius * np.sin(theta)
        ]).T
        return cls(points=points, circulation=circulation)

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


def compute_linking_number(ring1: VortexRing, ring2: VortexRing) -> float:
    """
    Compute linking number of two vortex rings using Gauss linking integral.

    Lk = (1/4π) ∫∫ (r₁ - r₂) · (dr₁ × dr₂) / |r₁ - r₂|³

    For discrete curves, this becomes a double sum.
    """
    Lk = 0.0
    n1 = len(ring1.points)
    n2 = len(ring2.points)

    for i in range(n1):
        p1 = ring1.points[i]
        p1_next = ring1.points[(i + 1) % n1]
        dr1 = p1_next - p1

        for j in range(n2):
            p2 = ring2.points[j]
            p2_next = ring2.points[(j + 1) % n2]
            dr2 = p2_next - p2

            # Midpoint approximation
            r1_mid = 0.5 * (p1 + p1_next)
            r2_mid = 0.5 * (p2 + p2_next)

            r = r1_mid - r2_mid
            r_mag = np.linalg.norm(r)

            if r_mag > 1e-10:
                cross = np.cross(dr1, dr2)
                Lk += np.dot(r, cross) / (r_mag ** 3)

    return Lk / (4 * np.pi)


def biot_savart_velocity(x: np.ndarray, rings: List[VortexRing], core_radius: float = 0.1) -> np.ndarray:
    """Compute velocity at point x due to all vortex rings."""
    u = np.zeros(3)

    for ring in rings:
        Gamma = ring.circulation
        points = ring.points
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


def compute_Qf(rings: List[VortexRing], f: Callable[[float], float]) -> float:
    """Compute Q_f = Σᵢⱼ ΓᵢΓⱼ ∫∫ f(|γᵢ(s) - γⱼ(t)|) ds dt"""
    Q = 0.0

    for i, ring_i in enumerate(rings):
        Gamma_i = ring_i.circulation
        points_i = ring_i.points
        dl_i = ring_i.arc_lengths()

        for j, ring_j in enumerate(rings):
            Gamma_j = ring_j.circulation
            points_j = ring_j.points
            dl_j = ring_j.arc_lengths()

            for si, pi in enumerate(points_i):
                for tj, pj in enumerate(points_j):
                    if i == j and si == tj:
                        continue
                    r = np.linalg.norm(pi - pj)
                    if r > 1e-10:
                        Q += Gamma_i * Gamma_j * f(r) * dl_i[si] * dl_j[tj]

    return Q


def compute_Qf_curvature_weighted(rings: List[VortexRing], f: Callable[[float], float]) -> float:
    """Compute curvature-weighted Q_κ = Σᵢⱼ ΓᵢΓⱼ ∫∫ κᵢκⱼ f(r) ds dt"""
    Q = 0.0

    for i, ring_i in enumerate(rings):
        Gamma_i = ring_i.circulation
        points_i = ring_i.points
        dl_i = ring_i.arc_lengths()
        kappa_i = ring_i.get_curvatures()

        for j, ring_j in enumerate(rings):
            Gamma_j = ring_j.circulation
            points_j = ring_j.points
            dl_j = ring_j.arc_lengths()
            kappa_j = ring_j.get_curvatures()

            for si, pi in enumerate(points_i):
                for tj, pj in enumerate(points_j):
                    if i == j and si == tj:
                        continue
                    r = np.linalg.norm(pi - pj)
                    if r > 1e-10:
                        Q += (Gamma_i * Gamma_j * kappa_i[si] * kappa_j[tj]
                              * f(r) * dl_i[si] * dl_j[tj])

    return Q


def evolve_rings(rings: List[VortexRing], dt: float, core_radius: float = 0.1):
    """Evolve all rings by one time step."""
    for ring in rings:
        new_points = []
        for p in ring.points:
            u = biot_savart_velocity(p, rings, core_radius)
            new_points.append(p + u * dt)
        ring.points = np.array(new_points)


def frac_var(values):
    """Fractional variation: std / |mean|."""
    arr = np.asarray(values)
    mean = np.mean(arr)
    if abs(mean) < 1e-15:
        return float(np.std(arr))
    return float(np.std(arr) / abs(mean))


# Test functions
def f_inv(r): return 1.0 / (r + 0.05)  # Q_{1/r} ~ energy
def f_exp(r): return np.exp(-r)
def f_sqrt(r): return np.sqrt(r)
def f_linear(r): return r


def create_hopf_link(radius: float = 1.0, separation: float = 0.0,
                     circ1: float = 1.0, circ2: float = 1.0, n_points: int = 48) -> Tuple[VortexRing, VortexRing]:
    """
    Create a Hopf link: two linked circles.

    Ring 1: XY plane, centered at origin
    Ring 2: XZ plane, centered at (radius + separation, 0, 0)

    When separation=0, the rings are maximally linked (Lk = ±1).
    """
    ring1 = VortexRing.create_xy_ring([0, 0, 0], radius, circ1, n_points)
    ring2 = VortexRing.create_xz_ring([radius + separation, 0, 0], radius, circ2, n_points)
    return ring1, ring2


def create_unlinked_coaxial(radius: float = 1.0, separation: float = 1.5,
                            circ1: float = 1.0, circ2: float = 1.0, n_points: int = 48) -> Tuple[VortexRing, VortexRing]:
    """Create unlinked coaxial rings (Lk = 0)."""
    ring1 = VortexRing.create_xy_ring([0, 0, 0], radius, circ1, n_points)
    ring2 = VortexRing.create_xy_ring([0, 0, separation], radius, circ2, n_points)
    return ring1, ring2


def run_evolution_test(rings: List[VortexRing], config_name: str, T: float = 1.5, dt: float = 0.02):
    """Run evolution and track Q_f quantities."""
    n_steps = int(T / dt)

    # Initial linking number
    Lk_initial = compute_linking_number(rings[0], rings[1])

    test_functions = [
        ("Q_{1/r}", f_inv),
        ("Q_{e^-r}", f_exp),
        ("Q_√r", f_sqrt),
        ("Q_r", f_linear),
    ]

    # Track quantities
    history = {name: [] for name, _ in test_functions}
    history["Q_κ(1/r)"] = []
    history["Lk"] = []

    for step in range(n_steps + 1):
        # Compute Q_f for each test function
        for name, f in test_functions:
            Q = compute_Qf(rings, f)
            history[name].append(Q)

        # Curvature-weighted
        Qk = compute_Qf_curvature_weighted(rings, f_inv)
        history["Q_κ(1/r)"].append(Qk)

        # Linking number (should be conserved for smooth evolution)
        Lk = compute_linking_number(rings[0], rings[1])
        history["Lk"].append(Lk)

        if step < n_steps:
            evolve_rings(rings, dt, core_radius=0.08)

    # Results
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"Initial linking number: {Lk_initial:.3f}")
    print(f"Evolution: T={T}, dt={dt}, steps={n_steps}")
    print(f"{'='*60}")
    print(f"{'Quantity':<15} {'Initial':>12} {'Final':>12} {'frac_var':>12} {'Status':>8}")
    print("-" * 60)

    results = {}
    for name in list(history.keys()):
        values = history[name]
        fv = frac_var(values)
        results[name] = fv

        status = "PASS" if fv < 0.01 else ("~" if fv < 0.05 else "FAIL")
        print(f"{name:<15} {values[0]:>12.4f} {values[-1]:>12.4f} {fv:>12.2e} {status:>8}")

    return results, history


def main():
    print("=" * 70)
    print("LINKED vs UNLINKED VORTEX RING Q_f CONSERVATION TEST")
    print("=" * 70)
    print("\nKey question: Does topological linking affect Q_f conservation?")
    print("Hypothesis: Linking number constrains Q_f dynamics")

    # Configuration 1: Unlinked coaxial rings (control)
    print("\n" + "=" * 70)
    print("TEST 1: UNLINKED COAXIAL RINGS (Lk = 0)")
    print("=" * 70)
    ring1, ring2 = create_unlinked_coaxial(radius=1.0, separation=1.5, circ1=1.0, circ2=1.0)
    results_unlinked, history_unlinked = run_evolution_test([ring1, ring2], "Unlinked coaxial")

    # Configuration 2: Hopf link (Lk = ±1)
    print("\n" + "=" * 70)
    print("TEST 2: HOPF LINK (Lk = ±1)")
    print("=" * 70)
    ring1, ring2 = create_hopf_link(radius=1.0, separation=0.0, circ1=1.0, circ2=1.0)
    results_linked, history_linked = run_evolution_test([ring1, ring2], "Hopf link")

    # Configuration 3: Counter-rotating Hopf link
    print("\n" + "=" * 70)
    print("TEST 3: COUNTER-ROTATING HOPF LINK (Γ₁Γ₂ < 0)")
    print("=" * 70)
    ring1, ring2 = create_hopf_link(radius=1.0, separation=0.0, circ1=1.0, circ2=-1.0)
    results_counter, history_counter = run_evolution_test([ring1, ring2], "Counter-rotating Hopf")

    # Configuration 4: Nearly unlinked (large separation)
    print("\n" + "=" * 70)
    print("TEST 4: NEARLY UNLINKED PERPENDICULAR RINGS")
    print("=" * 70)
    ring1, ring2 = create_hopf_link(radius=1.0, separation=1.5, circ1=1.0, circ2=1.0)
    results_perp, history_perp = run_evolution_test([ring1, ring2], "Perpendicular unlinked")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Comparing Q_f conservation across configurations")
    print("=" * 70)
    print(f"{'Quantity':<15} {'Unlinked':>12} {'Hopf':>12} {'Counter':>12} {'Perp':>12}")
    print("-" * 60)

    for name in ["Q_{1/r}", "Q_{e^-r}", "Q_√r", "Q_r", "Q_κ(1/r)", "Lk"]:
        print(f"{name:<15} {results_unlinked[name]:>12.2e} {results_linked[name]:>12.2e} "
              f"{results_counter[name]:>12.2e} {results_perp[name]:>12.2e}")

    # Novel findings check
    print("\n" + "=" * 70)
    print("NOVEL FINDING CHECK")
    print("=" * 70)

    # Compare linked vs unlinked for Q_{1/r}
    ratio = results_linked["Q_{1/r}"] / (results_unlinked["Q_{1/r}"] + 1e-10)
    if ratio < 0.5:
        print(f"  FINDING: Hopf link conserves Q_{{1/r}} {1/ratio:.1f}x BETTER than unlinked!")
    elif ratio > 2.0:
        print(f"  FINDING: Hopf link conserves Q_{{1/r}} {ratio:.1f}x WORSE than unlinked!")
    else:
        print(f"  Q_{{1/r}} conservation similar (ratio: {ratio:.2f})")

    # Check linking number conservation
    Lk_var = results_linked["Lk"]
    if Lk_var < 1e-6:
        print(f"  CONFIRMED: Linking number is EXACTLY conserved (frac_var: {Lk_var:.2e})")
    else:
        print(f"  WARNING: Linking number drifted (frac_var: {Lk_var:.2e})")

    # Check counter-rotating behavior
    counter_better = results_counter["Q_{1/r}"] < results_linked["Q_{1/r}"]
    print(f"  Counter-rotating {'improves' if counter_better else 'worsens'} Q_{{1/r}} conservation")

    return {
        "unlinked": results_unlinked,
        "hopf": results_linked,
        "counter": results_counter,
        "perp": results_perp
    }


if __name__ == "__main__":
    results = main()
