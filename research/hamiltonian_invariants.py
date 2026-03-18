#!/usr/bin/env python3
"""
Hamiltonian Mechanics Invariants

Explore conservation laws in Hamiltonian systems:
- Energy conservation (trivial)
- Liouville's theorem (phase space volume)
- Symplectic invariants (Poincaré invariants)
- Action variables in integrable systems
- KAM tori in nearly integrable systems

The oracle may know energy conservation but not the deeper structure.
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict


@dataclass
class HamiltonianSystem:
    """A Hamiltonian system with H(q, p)."""
    name: str
    n_dof: int  # degrees of freedom
    H: Callable[[np.ndarray], float]  # Hamiltonian H(z) where z = (q, p)
    dH: Callable[[np.ndarray], np.ndarray]  # gradient of H

    def equations_of_motion(self, t: float, z: np.ndarray) -> np.ndarray:
        """Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq."""
        n = self.n_dof
        _q, _p = z[:n], z[n:]
        grad_H = self.dH(z)
        dH_dq = grad_H[:n]
        dH_dp = grad_H[n:]
        dq_dt = dH_dp
        dp_dt = -dH_dq
        return np.concatenate([dq_dt, dp_dt])

    def integrate(self, z0: np.ndarray, T: float, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate the Hamiltonian system."""
        sol = solve_ivp(
            self.equations_of_motion,
            (0, T),
            z0,
            t_eval=np.arange(0, T, dt),
            method='RK45',
            rtol=1e-10,
            atol=1e-12
        )
        return sol.t, sol.y.T


def harmonic_oscillator(omega: float = 1.0) -> HamiltonianSystem:
    """1D harmonic oscillator: H = p²/2 + ω²q²/2."""
    def H(z):
        q, p = z[0], z[1]
        return 0.5 * p**2 + 0.5 * omega**2 * q**2

    def dH(z):
        q, p = z[0], z[1]
        return np.array([omega**2 * q, p])

    return HamiltonianSystem(
        name="Harmonic_Oscillator",
        n_dof=1,
        H=H,
        dH=dH
    )


def kepler_problem(mu: float = 1.0) -> HamiltonianSystem:
    """2D Kepler problem: H = |p|²/2 - μ/|q|."""
    def H(z):
        qx, qy, px, py = z
        r = np.sqrt(qx**2 + qy**2)
        return 0.5 * (px**2 + py**2) - mu / r

    def dH(z):
        qx, qy, px, py = z
        r = np.sqrt(qx**2 + qy**2)
        r3 = r**3
        return np.array([
            mu * qx / r3,  # dH/dqx
            mu * qy / r3,  # dH/dqy
            px,            # dH/dpx
            py             # dH/dpy
        ])

    return HamiltonianSystem(
        name="Kepler_2D",
        n_dof=2,
        H=H,
        dH=dH
    )


def henon_heiles() -> HamiltonianSystem:
    """Hénon-Heiles system: H = (px² + py²)/2 + (x² + y²)/2 + x²y - y³/3."""
    def H(z):
        x, y, px, py = z
        return 0.5*(px**2 + py**2) + 0.5*(x**2 + y**2) + x**2*y - y**3/3

    def dH(z):
        x, y, px, py = z
        return np.array([
            x + 2*x*y,       # dH/dx
            y + x**2 - y**2,  # dH/dy
            px,              # dH/dpx
            py               # dH/dpy
        ])

    return HamiltonianSystem(
        name="Henon_Heiles",
        n_dof=2,
        H=H,
        dH=dH
    )


def double_pendulum(m1: float = 1.0, m2: float = 1.0,
                    l1: float = 1.0, l2: float = 1.0, g: float = 9.8) -> HamiltonianSystem:
    """Double pendulum (chaotic system)."""
    def H(z):
        theta1, theta2, p1, p2 = z
        c = np.cos(theta1 - theta2)
        denom = m1 + m2 * (1 - c**2)

        # Kinetic energy in generalized momentum form
        T = (p1**2 * (m1 + m2) + p2**2 * m2 - 2*p1*p2*m2*c) / (2 * l1**2 * l2**2 * m2 * denom)
        # This is approximate; full expression is more complex
        # Use simpler version for testing
        T = 0.5 * (p1**2 + p2**2 + p1*p2*c)
        V = -(m1 + m2)*g*l1*np.cos(theta1) - m2*g*l2*np.cos(theta2)
        return T + V

    def dH(z):
        theta1, theta2, p1, p2 = z
        s = np.sin(theta1 - theta2)
        c = np.cos(theta1 - theta2)

        dT_dtheta1 = 0.5 * p1 * p2 * s
        dT_dtheta2 = -0.5 * p1 * p2 * s
        dV_dtheta1 = (m1 + m2)*g*l1*np.sin(theta1)
        dV_dtheta2 = m2*g*l2*np.sin(theta2)

        return np.array([
            dT_dtheta1 + dV_dtheta1,
            dT_dtheta2 + dV_dtheta2,
            p1 + 0.5*p2*c,
            p2 + 0.5*p1*c
        ])

    return HamiltonianSystem(
        name="Double_Pendulum",
        n_dof=2,
        H=H,
        dH=dH
    )


def coupled_oscillators(k1: float = 1.0, k2: float = 1.0,
                        k_coupling: float = 0.1) -> HamiltonianSystem:
    """Two coupled harmonic oscillators."""
    def H(z):
        q1, q2, p1, p2 = z
        return (0.5*(p1**2 + p2**2) +
                0.5*k1*q1**2 + 0.5*k2*q2**2 +
                0.5*k_coupling*(q1 - q2)**2)

    def dH(z):
        q1, q2, p1, p2 = z
        return np.array([
            k1*q1 + k_coupling*(q1 - q2),
            k2*q2 - k_coupling*(q1 - q2),
            p1,
            p2
        ])

    return HamiltonianSystem(
        name="Coupled_Oscillators",
        n_dof=2,
        H=H,
        dH=dH
    )


def check_energy_conservation(system: HamiltonianSystem, z0: np.ndarray,
                              T: float = 100.0) -> Dict:
    """Check energy conservation."""
    t, z_history = system.integrate(z0, T)
    energies = [system.H(z) for z in z_history]
    E0 = energies[0]
    frac_var = np.std(energies) / abs(E0) if abs(E0) > 1e-10 else np.std(energies)

    return {
        "name": "Energy",
        "formula": "H(q, p)",
        "initial": E0,
        "final": energies[-1],
        "frac_var": frac_var,
        "conserved": frac_var < 1e-6
    }


def check_liouville_volume(system: HamiltonianSystem, z0: np.ndarray,
                           T: float = 50.0, n_samples: int = 100,
                           epsilon: float = 0.01) -> Dict:
    """
    Check Liouville's theorem: phase space volume is conserved.

    Create a small cloud of initial conditions and track volume evolution.
    """
    n = 2 * system.n_dof

    # Create initial cloud around z0
    np.random.seed(42)
    perturbations = np.random.randn(n_samples, n) * epsilon
    cloud0 = z0 + perturbations

    # Compute initial "volume" (covariance determinant)
    cov0 = np.cov(cloud0.T)
    vol0 = np.linalg.det(cov0)

    # Evolve each point
    final_cloud = []
    for z in cloud0:
        t, z_history = system.integrate(z, T)
        final_cloud.append(z_history[-1])
    final_cloud = np.array(final_cloud)

    # Compute final volume
    cov_final = np.cov(final_cloud.T)
    vol_final = np.linalg.det(cov_final)

    # Volume ratio should be 1
    vol_ratio = vol_final / vol0 if abs(vol0) > 1e-15 else 1.0
    frac_var = abs(vol_ratio - 1.0)

    return {
        "name": "Liouville_Volume",
        "formula": "det(phase_space_covariance)",
        "initial": vol0,
        "final": vol_final,
        "ratio": vol_ratio,
        "frac_var": frac_var,
        "conserved": frac_var < 0.1  # Looser tolerance for volume
    }


def check_angular_momentum(z: np.ndarray) -> float:
    """L = q × p for 2D systems."""
    if len(z) == 4:  # 2 DOF
        qx, qy, px, py = z
        return qx * py - qy * px
    return 0.0


def check_kepler_invariants(system: HamiltonianSystem, z0: np.ndarray,
                            T: float = 100.0, mu: float = 1.0) -> List[Dict]:
    """Check Kepler-specific invariants: L (angular momentum), LRL vector."""
    t, z_history = system.integrate(z0, T)

    results = []

    # Angular momentum L = r × p
    L_values = [check_angular_momentum(z) for z in z_history]
    L0 = L_values[0]
    L_frac_var = np.std(L_values) / abs(L0) if abs(L0) > 1e-10 else np.std(L_values)
    results.append({
        "name": "Angular_Momentum",
        "formula": "L = q × p",
        "initial": L0,
        "final": L_values[-1],
        "frac_var": L_frac_var,
        "conserved": L_frac_var < 1e-6
    })

    # Laplace-Runge-Lenz vector magnitude
    # A = p × L - μ r̂
    def lrl_magnitude(z):
        qx, qy, px, py = z
        r = np.sqrt(qx**2 + qy**2)
        L = qx * py - qy * px
        # A = p × L - μ r̂
        Ax = py * L - mu * qx / r
        Ay = -px * L - mu * qy / r
        return np.sqrt(Ax**2 + Ay**2)

    LRL_values = [lrl_magnitude(z) for z in z_history]
    LRL0 = LRL_values[0]
    LRL_frac_var = np.std(LRL_values) / abs(LRL0) if abs(LRL0) > 1e-10 else np.std(LRL_values)
    results.append({
        "name": "LRL_Vector_Magnitude",
        "formula": "|p × L - μ r̂|",
        "initial": LRL0,
        "final": LRL_values[-1],
        "frac_var": LRL_frac_var,
        "conserved": LRL_frac_var < 1e-5
    })

    return results


def check_poincare_invariant(system: HamiltonianSystem, z0: np.ndarray,
                             T: float = 50.0, n_points: int = 20) -> Dict:
    """
    First Poincaré integral invariant: ∮ p dq around a loop.

    Create a loop in phase space and check that ∮ p dq is preserved.
    """
    n = system.n_dof

    # Create a loop in (q1, p1) plane
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    radius = 0.1

    loop0 = []
    for th in theta:
        z = z0.copy()
        z[0] += radius * np.cos(th)  # q1 perturbation
        z[n] += radius * np.sin(th)  # p1 perturbation
        loop0.append(z)
    loop0 = np.array(loop0)

    # Compute initial ∮ p dq (approximate with sum)
    def loop_integral(loop):
        integral = 0.0
        for i in range(len(loop)):
            j = (i + 1) % len(loop)
            # p_avg * dq
            p_avg = 0.5 * (loop[i, n] + loop[j, n])
            dq = loop[j, 0] - loop[i, 0]
            integral += p_avg * dq
        return integral

    I0 = loop_integral(loop0)

    # Evolve each point
    final_loop = []
    for z in loop0:
        t, z_history = system.integrate(z, T)
        final_loop.append(z_history[-1])
    final_loop = np.array(final_loop)

    I_final = loop_integral(final_loop)

    frac_var = abs(I_final - I0) / abs(I0) if abs(I0) > 1e-10 else abs(I_final - I0)

    return {
        "name": "Poincare_Invariant",
        "formula": "∮ p dq",
        "initial": I0,
        "final": I_final,
        "frac_var": frac_var,
        "conserved": frac_var < 0.1
    }


def check_action_variable(system: HamiltonianSystem, z0: np.ndarray,
                          T: float = 100.0) -> Dict:
    """
    Action variable J = (1/2π) ∮ p dq for periodic orbits.

    For integrable systems, J is an adiabatic invariant.
    """
    t, z_history = system.integrate(z0, T)
    n = system.n_dof

    # Compute ∮ p dq over trajectory segments
    # Use running sum of p * dq
    action = 0.0
    for i in range(len(t) - 1):
        t[i+1] - t[i]
        p_avg = 0.5 * (z_history[i, n] + z_history[i+1, n])
        dq = z_history[i+1, 0] - z_history[i, 0]
        action += p_avg * dq

    # Normalize by period estimate
    # For simple oscillators, expect 2πJ
    return {
        "name": "Action_Variable",
        "formula": "J = (1/2π) ∮ p dq",
        "value": action / (2 * np.pi),
        "raw_integral": action
    }


def main():
    print("=" * 70)
    print("Hamiltonian Mechanics Invariants")
    print("=" * 70)

    results = {}

    # Test 1: Harmonic Oscillator
    print("\n[1] Harmonic Oscillator")
    ho = harmonic_oscillator(omega=2.0)
    z0 = np.array([1.0, 0.5])  # (q, p)

    energy = check_energy_conservation(ho, z0)
    print(f"    Energy: frac_var = {energy['frac_var']:.2e} [{'CONSERVED' if energy['conserved'] else 'BROKEN'}]")

    liouville = check_liouville_volume(ho, z0, n_samples=50)
    print(f"    Liouville: ratio = {liouville['ratio']:.4f} [{'CONSERVED' if liouville['conserved'] else 'BROKEN'}]")

    poincare = check_poincare_invariant(ho, z0)
    print(f"    Poincaré: frac_var = {poincare['frac_var']:.2e} [{'CONSERVED' if poincare['conserved'] else 'BROKEN'}]")

    results["Harmonic_Oscillator"] = {"energy": energy, "liouville": liouville, "poincare": poincare}

    # Test 2: Kepler Problem
    print("\n[2] Kepler Problem (2D)")
    kepler = kepler_problem(mu=1.0)
    # Elliptical orbit initial condition
    z0 = np.array([1.0, 0.0, 0.0, 0.8])  # r=1, v tangential

    energy = check_energy_conservation(kepler, z0, T=50.0)
    print(f"    Energy: frac_var = {energy['frac_var']:.2e} [{'CONSERVED' if energy['conserved'] else 'BROKEN'}]")

    kepler_inv = check_kepler_invariants(kepler, z0, T=50.0)
    for inv in kepler_inv:
        print(f"    {inv['name']}: frac_var = {inv['frac_var']:.2e} [{'CONSERVED' if inv['conserved'] else 'BROKEN'}]")

    liouville = check_liouville_volume(kepler, z0, T=30.0, n_samples=30)
    print(f"    Liouville: ratio = {liouville['ratio']:.4f} [{'CONSERVED' if liouville['conserved'] else 'BROKEN'}]")

    results["Kepler"] = {"energy": energy, "kepler_invariants": kepler_inv, "liouville": liouville}

    # Test 3: Hénon-Heiles (chaotic)
    print("\n[3] Hénon-Heiles (chaotic at high energy)")
    hh = henon_heiles()
    z0 = np.array([0.3, 0.0, 0.0, 0.3])  # Low energy, regular

    energy = check_energy_conservation(hh, z0)
    print(f"    Energy (low E): frac_var = {energy['frac_var']:.2e} [{'CONSERVED' if energy['conserved'] else 'BROKEN'}]")

    liouville = check_liouville_volume(hh, z0, T=30.0, n_samples=30)
    print(f"    Liouville (low E): ratio = {liouville['ratio']:.4f} [{'CONSERVED' if liouville['conserved'] else 'BROKEN'}]")

    # Higher energy (chaotic)
    z0_high = np.array([0.5, 0.0, 0.0, 0.5])
    energy_high = check_energy_conservation(hh, z0_high, T=50.0)
    print(f"    Energy (high E): frac_var = {energy_high['frac_var']:.2e} [{'CONSERVED' if energy_high['conserved'] else 'BROKEN'}]")

    results["Henon_Heiles"] = {"energy_low": energy, "energy_high": energy_high, "liouville": liouville}

    # Test 4: Coupled Oscillators
    print("\n[4] Coupled Oscillators")
    co = coupled_oscillators(k1=1.0, k2=1.5, k_coupling=0.2)
    z0 = np.array([1.0, 0.0, 0.0, 0.5])

    energy = check_energy_conservation(co, z0)
    print(f"    Energy: frac_var = {energy['frac_var']:.2e} [{'CONSERVED' if energy['conserved'] else 'BROKEN'}]")

    liouville = check_liouville_volume(co, z0, T=50.0, n_samples=30)
    print(f"    Liouville: ratio = {liouville['ratio']:.4f} [{'CONSERVED' if liouville['conserved'] else 'BROKEN'}]")

    poincare = check_poincare_invariant(co, z0)
    print(f"    Poincaré: frac_var = {poincare['frac_var']:.2e} [{'CONSERVED' if poincare['conserved'] else 'BROKEN'}]")

    results["Coupled_Oscillators"] = {"energy": energy, "liouville": liouville, "poincare": poincare}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Hamiltonian invariants tested:
1. Energy H(q,p) - conserved in all autonomous systems
2. Liouville volume - phase space volume preserved
3. Poincaré invariant ∮p dq - symplectic invariant
4. Angular momentum L - conserved with rotational symmetry
5. Laplace-Runge-Lenz vector - Kepler-specific (SO(4) symmetry)

Knowledge gap candidates:
- Does the oracle know Liouville's theorem applies to ALL Hamiltonian systems?
- Does it know the LRL vector is conserved for Kepler?
- Does it understand why volume is preserved (symplectic structure)?
- Does it know about KAM theory for nearly integrable systems?
""")

    return results


if __name__ == "__main__":
    results = main()
