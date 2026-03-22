#!/usr/bin/env python3
"""
Test Q_f conservation for HIGHER LINKING NUMBERS (Lk = 1, 2, 3, ...).

Creates torus links with varying linking numbers to test:
Does Q_f conservation degrade progressively with |Lk|?

A (2, 2n) torus link consists of two curves on a torus with linking number n.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Tuple


@dataclass
class VortexCurve:
    """A closed vortex curve in 3D."""
    points: np.ndarray  # (N, 3)
    circulation: float

    def arc_lengths(self):
        dl = np.linalg.norm(np.roll(self.points, -1, axis=0) - self.points, axis=1)
        return dl


def create_torus_link_component(
    R: float, r: float,
    p: int, q: int,
    phase: float = 0.0,
    circulation: float = 1.0,
    n_points: int = 128
) -> VortexCurve:
    """
    Create one component of a (p,q) torus knot/link.

    R = major radius (torus center to tube center)
    r = minor radius (tube radius)
    p = number of times around the torus hole
    q = number of times around the torus tube
    phase = angular offset for second component

    For a two-component link, use:
    - Component 1: phase = 0
    - Component 2: phase = π/q
    """
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    # Torus knot parameterization
    x = (R + r * np.cos(q * t + phase)) * np.cos(p * t)
    y = (R + r * np.cos(q * t + phase)) * np.sin(p * t)
    z = r * np.sin(q * t + phase)

    points = np.column_stack([x, y, z])
    return VortexCurve(points=points, circulation=circulation)


def create_torus_link(
    linking_number: int,
    R: float = 2.0,
    r: float = 0.8,
    circ1: float = 1.0,
    circ2: float = 1.0,
    n_points: int = 128
) -> Tuple[VortexCurve, VortexCurve]:
    """
    Create a two-component torus link with specified linking number.

    Linking number 1: Hopf link (2,2 torus link)
    Linking number 2: Solomon's seal (2,4 torus link)
    Linking number 3: (2,6 torus link)
    etc.

    For a (2, 2n) torus link, the linking number is n.
    """
    q = 2 * linking_number  # q determines linking number
    p = 2  # Two components

    # Phase offset to create two separate components
    phase_offset = np.pi / q

    curve1 = create_torus_link_component(R, r, p, q, phase=0.0,
                                         circulation=circ1, n_points=n_points)
    curve2 = create_torus_link_component(R, r, p, q, phase=phase_offset,
                                         circulation=circ2, n_points=n_points)

    return curve1, curve2


def create_wound_link(
    n_winds: int,
    R: float = 2.0,
    r: float = 0.5,
    circ1: float = 1.0,
    circ2: float = 1.0,
    n_points: int = 128
) -> Tuple[VortexCurve, VortexCurve]:
    """
    Alternative: Create a link by winding one circle through another n times.

    Ring 1: Simple circle in XY plane
    Ring 2: Helix that winds through Ring 1 n times
    """
    # Ring 1: Simple circle
    t1 = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x1 = R * np.cos(t1)
    y1 = R * np.sin(t1)
    z1 = np.zeros_like(t1)
    curve1 = VortexCurve(points=np.column_stack([x1, y1, z1]), circulation=circ1)

    # Ring 2: Winds through ring 1 n times
    t2 = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    # Center traces a circle, but displaced and wound
    x2 = (R + r * np.cos(n_winds * t2)) * np.cos(t2)
    y2 = (R + r * np.cos(n_winds * t2)) * np.sin(t2)
    z2 = r * np.sin(n_winds * t2)
    curve2 = VortexCurve(points=np.column_stack([x2, y2, z2]), circulation=circ2)

    return curve1, curve2


def compute_linking_number(curve1: VortexCurve, curve2: VortexCurve) -> float:
    """Compute linking number using Gauss integral."""
    Lk = 0.0
    n1, n2 = len(curve1.points), len(curve2.points)

    for i in range(n1):
        p1 = curve1.points[i]
        p1_next = curve1.points[(i + 1) % n1]
        dr1 = p1_next - p1

        for j in range(n2):
            p2 = curve2.points[j]
            p2_next = curve2.points[(j + 1) % n2]
            dr2 = p2_next - p2

            r1_mid = 0.5 * (p1 + p1_next)
            r2_mid = 0.5 * (p2 + p2_next)

            r = r1_mid - r2_mid
            r_mag = np.linalg.norm(r)

            if r_mag > 1e-10:
                cross = np.cross(dr1, dr2)
                Lk += np.dot(r, cross) / (r_mag ** 3)

    return Lk / (4 * np.pi)


def biot_savart_velocity(x: np.ndarray, curves: List[VortexCurve],
                         core_radius: float = 0.1) -> np.ndarray:
    """Compute velocity at point x."""
    u = np.zeros(3)

    for curve in curves:
        Gamma = curve.circulation
        points = curve.points
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


def compute_Qf(curves: List[VortexCurve], f: Callable[[float], float]) -> float:
    """Compute Q_f for multiple curves."""
    Q = 0.0

    for i, c1 in enumerate(curves):
        Gamma1 = c1.circulation
        dl1 = c1.arc_lengths()

        for j, c2 in enumerate(curves):
            Gamma2 = c2.circulation
            dl2 = c2.arc_lengths()

            for si, pi in enumerate(c1.points):
                for tj, pj in enumerate(c2.points):
                    if i == j and si == tj:
                        continue
                    r = np.linalg.norm(pi - pj)
                    if r > 1e-10:
                        Q += Gamma1 * Gamma2 * f(r) * dl1[si] * dl2[tj]

    return Q


def evolve_curves(curves: List[VortexCurve], dt: float, core_radius: float = 0.1):
    """Evolve all curves by one time step."""
    for curve in curves:
        new_points = []
        for p in curve.points:
            u = biot_savart_velocity(p, curves, core_radius)
            new_points.append(p + u * dt)
        curve.points = np.array(new_points)


def frac_var(values):
    arr = np.asarray(values)
    mean = np.mean(arr)
    if abs(mean) < 1e-15:
        return float(np.std(arr))
    return float(np.std(arr) / abs(mean))


def f_inv(r): return 1.0 / (r + 0.1)


def test_linking_number(Lk_target: int, same_sign: bool = True,
                        T: float = 0.8, dt: float = 0.015) -> dict:
    """Test Q_f conservation for a specific linking number."""
    circ2 = 1.0 if same_sign else -1.0

    # Create torus link
    curve1, curve2 = create_torus_link(
        linking_number=Lk_target,
        R=2.0, r=0.6,
        circ1=1.0, circ2=circ2,
        n_points=96
    )
    curves = [curve1, curve2]

    # Verify linking number
    Lk_actual = compute_linking_number(curve1, curve2)

    # Helicity proxy
    H = curve1.circulation * curve2.circulation * Lk_actual

    n_steps = int(T / dt)
    Qf_history = []
    Lk_history = []

    for step in range(n_steps + 1):
        Qf = compute_Qf(curves, f_inv)
        Qf_history.append(Qf)

        Lk = compute_linking_number(curve1, curve2)
        Lk_history.append(Lk)

        if step < n_steps:
            evolve_curves(curves, dt, core_radius=0.12)

    fv = frac_var(Qf_history)
    Lk_fv = frac_var(Lk_history)

    return {
        "Lk_target": Lk_target,
        "Lk_actual": Lk_actual,
        "H": H,
        "same_sign": same_sign,
        "Q_frac_var": fv,
        "Lk_frac_var": Lk_fv,
        "Q_initial": Qf_history[0],
        "Q_final": Qf_history[-1]
    }


def main():
    print("=" * 75)
    print("HIGHER LINKING NUMBER Q_f CONSERVATION TEST")
    print("=" * 75)
    print("\nHypothesis: Q_f conservation degrades with increasing |Lk|")
    print("Testing torus links with Lk = 1, 2, 3")

    results = []

    # Test same-sign (negative helicity)
    print("\n" + "=" * 75)
    print("SAME-SIGN CIRCULATION (Γ₁Γ₂ > 0, H < 0)")
    print("=" * 75)
    print(f"{'Lk':>4} {'Lk_actual':>10} {'H':>8} {'Q_frac_var':>12} {'Lk_conserved':>12}")
    print("-" * 50)

    for Lk in [1, 2, 3]:
        r = test_linking_number(Lk, same_sign=True)
        results.append(r)
        print(f"{r['Lk_target']:>4} {r['Lk_actual']:>10.2f} {r['H']:>8.2f} "
              f"{r['Q_frac_var']:>12.2e} {r['Lk_frac_var']:>12.2e}")

    # Test counter-rotating (positive helicity)
    print("\n" + "=" * 75)
    print("COUNTER-ROTATING (Γ₁Γ₂ < 0, H > 0)")
    print("=" * 75)
    print(f"{'Lk':>4} {'Lk_actual':>10} {'H':>8} {'Q_frac_var':>12} {'Lk_conserved':>12}")
    print("-" * 50)

    for Lk in [1, 2, 3]:
        r = test_linking_number(Lk, same_sign=False)
        results.append(r)
        print(f"{r['Lk_target']:>4} {r['Lk_actual']:>10.2f} {r['H']:>8.2f} "
              f"{r['Q_frac_var']:>12.2e} {r['Lk_frac_var']:>12.2e}")

    # Summary
    print("\n" + "=" * 75)
    print("SUMMARY: Q_f conservation vs |Lk|")
    print("=" * 75)

    same_sign_results = [r for r in results if r['same_sign']]
    counter_results = [r for r in results if not r['same_sign']]

    print("\nSame-sign (H < 0) — expecting degradation with |Lk|:")
    for r in same_sign_results:
        print(f"  Lk = {r['Lk_target']}: Q_frac_var = {r['Q_frac_var']:.2e}")

    print("\nCounter-rotating (H > 0) — expecting better conservation:")
    for r in counter_results:
        print(f"  Lk = {r['Lk_target']}: Q_frac_var = {r['Q_frac_var']:.2e}")

    # Check monotonicity
    print("\n" + "=" * 75)
    print("ANALYSIS")
    print("=" * 75)

    same_fvs = [r['Q_frac_var'] for r in same_sign_results]
    counter_fvs = [r['Q_frac_var'] for r in counter_results]

    if same_fvs[0] < same_fvs[1] < same_fvs[2]:
        print("✓ CONFIRMED: Same-sign Q_f degrades monotonically with |Lk|")
        ratio_1_to_2 = same_fvs[1] / same_fvs[0]
        ratio_2_to_3 = same_fvs[2] / same_fvs[1]
        print(f"  Lk=1→2: {ratio_1_to_2:.1f}× worse")
        print(f"  Lk=2→3: {ratio_2_to_3:.1f}× worse")
    else:
        print("? Non-monotonic pattern for same-sign")

    if all(c < s for c, s in zip(counter_fvs, same_fvs)):
        print("✓ CONFIRMED: Counter-rotating always better than same-sign at each Lk")
        for i, Lk in enumerate([1, 2, 3]):
            ratio = same_fvs[i] / counter_fvs[i]
            print(f"  Lk={Lk}: counter {ratio:.1f}× better")

    # Check if |H| predicts frac_var
    print("\nCorrelation with |H|:")
    all_H = [abs(r['H']) for r in results]
    all_fv = [r['Q_frac_var'] for r in results]

    # Simple rank correlation
    H_ranks = np.argsort(np.argsort(all_H))
    fv_ranks = np.argsort(np.argsort(all_fv))

    print(f"  |H| values: {[f'{h:.1f}' for h in all_H]}")
    print(f"  frac_var:   {[f'{fv:.2e}' for fv in all_fv]}")

    return results


if __name__ == "__main__":
    results = main()
