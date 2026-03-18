"""
noethersolve.hamiltonian — Hamiltonian system validation toolkit.

Validates that an ODE integrator preserves the symplectic structure of
Hamiltonian systems. Goes beyond energy conservation to check Liouville's
theorem (phase-space volume), the first Poincaré integral invariant
(∮ p dq), and system-specific quantities (angular momentum, Laplace–
Runge–Lenz vector for Kepler).

Usage:
    from noethersolve import HamiltonianMonitor

    monitor = HamiltonianMonitor(H=my_hamiltonian, dH=my_gradient, n_dof=2)
    report = monitor.validate(
        z0=np.array([1.0, 0.0, 0.0, 0.8]),
        T=100.0,
        rtol=1e-10,
    )
    print(report)
    # Shows: energy, Liouville volume, Poincaré invariant — PASS/WARN/FAIL

Built-in systems:
    harmonic_oscillator(omega)
    kepler_2d(mu)
    henon_heiles()
    coupled_oscillators(k1, k2, k_coupling)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from scipy.integrate import solve_ivp

from noethersolve.monitor import frac_var


# ─── Report ───────────────────────────────────────────────────────────────────

@dataclass
class HamiltonianReport:
    """Result of HamiltonianMonitor.validate()."""
    verdict: str                    # PASS, WARN, FAIL
    system_name: str
    n_dof: int
    solver_method: str
    rtol: float
    atol: float
    T: float
    quantities: Dict[str, dict]    # name -> {value, frac_var, verdict, ...}
    violations: List[str]
    suggestions: List[str]

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Hamiltonian Validation: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  System: {self.system_name} ({self.n_dof} DOF)")
        lines.append(f"  Solver: {self.solver_method}, rtol={self.rtol:.0e}, atol={self.atol:.0e}")
        lines.append(f"  T = {self.T}")
        lines.append("")

        failed = [(k, v) for k, v in self.quantities.items() if v["verdict"] == "FAIL"]
        warned = [(k, v) for k, v in self.quantities.items() if v["verdict"] == "WARN"]
        passed = [(k, v) for k, v in self.quantities.items() if v["verdict"] == "PASS"]

        if failed:
            lines.append(f"  FAILED ({len(failed)}):")
            for name, data in failed:
                lines.append(f"    {name:<30s}  frac_var={data['frac_var']:.2e}")
        if warned:
            lines.append(f"  WARNINGS ({len(warned)}):")
            for name, data in warned:
                lines.append(f"    {name:<30s}  frac_var={data['frac_var']:.2e}")
        if passed:
            lines.append(f"  PASSED ({len(passed)}):")
            for name, data in passed:
                lines.append(f"    {name:<30s}  frac_var={data['frac_var']:.2e}")

        if self.suggestions:
            lines.append("")
            lines.append("  Suggestions:")
            for s in self.suggestions:
                lines.append(f"    - {s}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


# ─── Thresholds ───────────────────────────────────────────────────────────────

ENERGY_THRESHOLD = 1e-8       # Energy is exact
LIOUVILLE_THRESHOLD = 0.05    # Volume ratio — looser due to cloud sampling noise
POINCARE_THRESHOLD = 0.05     # Integral invariant — also uses cloud, so looser
WARN_THRESHOLD = 0.1


# ─── HamiltonianMonitor ──────────────────────────────────────────────────────

class HamiltonianMonitor:
    """Validate symplectic structure preservation in Hamiltonian systems.

    Checks three levels of Hamiltonian structure:
    1. Energy conservation — H(q, p) = const.
    2. Liouville's theorem — phase-space volume preservation.
    3. Poincaré integral invariant — ∮ p dq preserved along flow.

    Plus system-specific invariants when applicable (angular momentum,
    Laplace–Runge–Lenz vector for Kepler).

    Args:
        H: Hamiltonian function H(z) where z = (q1, ..., qn, p1, ..., pn).
        dH: Gradient of H — returns array of length 2n.
        n_dof: Number of degrees of freedom.
        name: Optional system name for reports.
    """

    def __init__(self, H: Callable, dH: Callable, n_dof: int,
                 name: str = "custom",
                 custom_invariants: Optional[Dict[str, Callable]] = None):
        self.H = H
        self.dH = dH
        self.n_dof = n_dof
        self.name = name
        self.custom_invariants = custom_invariants or {}

    def _eom(self, t: float, z: np.ndarray) -> np.ndarray:
        """Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q."""
        n = self.n_dof
        grad_H = self.dH(z)
        dH_dq = grad_H[:n]
        dH_dp = grad_H[n:]
        return np.concatenate([dH_dp, -dH_dq])

    def _integrate(self, z0, T, method="RK45", rtol=1e-10, atol=1e-12,
                   n_eval=1000):
        """Integrate Hamilton's equations."""
        t_eval = np.linspace(0, T, n_eval)
        sol = solve_ivp(self._eom, (0, T), z0, method=method,
                        t_eval=t_eval, rtol=rtol, atol=atol)
        return sol

    def _check_energy(self, sol) -> dict:
        """Check energy conservation along the trajectory."""
        energies = [float(self.H(sol.y[:, i])) for i in range(sol.y.shape[1])]
        fv = frac_var(energies)
        if fv < ENERGY_THRESHOLD:
            verdict = "PASS"
        elif fv < WARN_THRESHOLD:
            verdict = "WARN"
        else:
            verdict = "FAIL"
        return {
            "frac_var": fv,
            "initial": energies[0],
            "final": energies[-1],
            "verdict": verdict,
            "type": "exact",
        }

    def _check_liouville(self, z0, T, method, rtol, atol,
                         n_cloud=50, epsilon=0.01) -> dict:
        """Check Liouville's theorem: phase-space volume preservation.

        Creates a small cloud of initial conditions around z0, evolves all
        of them, and compares the covariance determinant before/after.

        Physics: For any Hamiltonian flow, the phase-space volume element
        is preserved (Liouville's theorem). This is equivalent to the
        Jacobian determinant of the flow map being 1. A non-symplectic
        integrator (e.g., forward Euler) will violate this.
        """
        n = 2 * self.n_dof
        rng = np.random.RandomState(42)
        perturbations = rng.randn(n_cloud, n) * epsilon
        cloud0 = z0 + perturbations

        # Initial volume (covariance determinant)
        cov0 = np.cov(cloud0.T)
        vol0 = np.linalg.det(cov0)

        # Evolve each point
        final_cloud = []
        for z in cloud0:
            sol = solve_ivp(self._eom, (0, T), z, method=method,
                            rtol=rtol, atol=atol)
            if sol.status == 0:
                final_cloud.append(sol.y[:, -1])
            else:
                final_cloud.append(z)  # solver failed, use original
        final_cloud = np.array(final_cloud)

        cov_final = np.cov(final_cloud.T)
        vol_final = np.linalg.det(cov_final)

        vol_ratio = vol_final / vol0 if abs(vol0) > 1e-15 else 1.0
        fv = abs(vol_ratio - 1.0)

        if fv < LIOUVILLE_THRESHOLD:
            verdict = "PASS"
        elif fv < WARN_THRESHOLD:
            verdict = "WARN"
        else:
            verdict = "FAIL"

        return {
            "frac_var": fv,
            "initial": vol0,
            "final": vol_final,
            "ratio": vol_ratio,
            "verdict": verdict,
            "type": "exact",
        }

    def _check_poincare(self, z0, T, method, rtol, atol,
                        n_loop=20, radius=0.1) -> dict:
        """Check first Poincaré integral invariant: ∮ p dq preserved.

        Creates a loop in (q1, p1) phase space, evolves it, and compares
        the enclosed area (∮ p dq) before and after.

        Physics: For Hamiltonian flow, the symplectic 2-form ω = Σ dpᵢ∧dqᵢ
        is preserved. The integral ∮ p dq over any closed loop is invariant
        under the flow. This is stronger than volume preservation (Liouville
        follows from Poincaré, but not vice versa).
        """
        n = self.n_dof
        theta = np.linspace(0, 2 * np.pi, n_loop, endpoint=False)

        # Create loop in (q1, p1) plane
        loop0 = []
        for th in theta:
            z = z0.copy()
            z[0] += radius * np.cos(th)    # q1 perturbation
            z[n] += radius * np.sin(th)     # p1 perturbation
            loop0.append(z)
        loop0 = np.array(loop0)

        def loop_integral(loop):
            """Compute ∮ p₁ dq₁ by trapezoidal rule."""
            integral = 0.0
            for i in range(len(loop)):
                j = (i + 1) % len(loop)
                p_avg = 0.5 * (loop[i, n] + loop[j, n])
                dq = loop[j, 0] - loop[i, 0]
                integral += p_avg * dq
            return integral

        I0 = loop_integral(loop0)

        # Evolve each loop point
        final_loop = []
        for z in loop0:
            sol = solve_ivp(self._eom, (0, T), z, method=method,
                            rtol=rtol, atol=atol)
            if sol.status == 0:
                final_loop.append(sol.y[:, -1])
            else:
                final_loop.append(z)
        final_loop = np.array(final_loop)

        I_final = loop_integral(final_loop)

        fv = abs(I_final - I0) / abs(I0) if abs(I0) > 1e-10 else abs(I_final - I0)

        if fv < POINCARE_THRESHOLD:
            verdict = "PASS"
        elif fv < WARN_THRESHOLD:
            verdict = "WARN"
        else:
            verdict = "FAIL"

        return {
            "frac_var": fv,
            "initial": I0,
            "final": I_final,
            "verdict": verdict,
            "type": "exact",
        }

    def _check_custom_invariants(self, sol) -> Dict[str, dict]:
        """Check any user-supplied invariants along the trajectory."""
        results = {}
        for name, func in self.custom_invariants.items():
            values = [float(func(sol.y[:, i])) for i in range(sol.y.shape[1])]
            fv = frac_var(values)
            if fv < ENERGY_THRESHOLD:
                verdict = "PASS"
            elif fv < WARN_THRESHOLD:
                verdict = "WARN"
            else:
                verdict = "FAIL"
            results[name] = {
                "frac_var": fv,
                "initial": values[0],
                "final": values[-1],
                "verdict": verdict,
                "type": "custom",
            }
        return results

    # ── Public API ────────────────────────────────────────────────────

    def validate(self, z0: np.ndarray, T: float = 100.0,
                 method: str = "RK45", rtol: float = 1e-10,
                 atol: float = 1e-12, n_eval: int = 1000,
                 check_liouville: bool = True,
                 check_poincare: bool = True,
                 liouville_T: Optional[float] = None,
                 poincare_T: Optional[float] = None,
                 ) -> HamiltonianReport:
        """Validate integrator on a Hamiltonian system.

        Runs the integration and checks energy conservation, Liouville's
        theorem, and the Poincaré invariant.

        Args:
            z0: Initial state vector [q1, ..., qn, p1, ..., pn].
            T: Integration time for energy check.
            method: scipy solver method.
            rtol, atol: Solver tolerances.
            n_eval: Number of evaluation points.
            check_liouville: Whether to run Liouville volume test.
            check_poincare: Whether to run Poincaré invariant test.
            liouville_T: Override T for Liouville test (default: T/2).
            poincare_T: Override T for Poincaré test (default: T/2).

        Returns:
            HamiltonianReport with per-quantity breakdown.
        """
        z0 = np.asarray(z0, dtype=np.float64)

        quantities = {}
        violations = []
        suggestions = []

        # 1. Energy conservation (main integration)
        sol = self._integrate(z0, T, method=method, rtol=rtol, atol=atol,
                              n_eval=n_eval)
        if sol.status != 0:
            return HamiltonianReport(
                verdict="FAIL",
                system_name=self.name,
                n_dof=self.n_dof,
                solver_method=method,
                rtol=rtol, atol=atol, T=T,
                quantities={},
                violations=["solver_failed"],
                suggestions=[f"Solver failed: {sol.message}"],
            )

        energy_result = self._check_energy(sol)
        quantities["energy"] = energy_result
        if energy_result["verdict"] == "FAIL":
            violations.append("energy")

        # 2. Custom invariants (on the same trajectory)
        custom_results = self._check_custom_invariants(sol)
        for name, result in custom_results.items():
            quantities[name] = result
            if result["verdict"] == "FAIL":
                violations.append(name)

        # 3. Liouville volume test (shorter T, cloud of ICs)
        if check_liouville:
            L_T = liouville_T or min(T / 2, 50.0)
            liouville_result = self._check_liouville(
                z0, L_T, method, rtol, atol)
            quantities["liouville_volume"] = liouville_result
            if liouville_result["verdict"] == "FAIL":
                violations.append("liouville_volume")

        # 4. Poincaré invariant test (shorter T, loop of ICs)
        if check_poincare:
            P_T = poincare_T or min(T / 2, 50.0)
            poincare_result = self._check_poincare(
                z0, P_T, method, rtol, atol)
            quantities["poincare_invariant"] = poincare_result
            if poincare_result["verdict"] == "FAIL":
                violations.append("poincare_invariant")

        # ── Verdict ───────────────────────────────────────────────────
        if violations:
            verdict = "FAIL"
        elif any(v["verdict"] == "WARN" for v in quantities.values()):
            verdict = "WARN"
        else:
            verdict = "PASS"

        # ── Suggestions ───────────────────────────────────────────────
        if "energy" in violations:
            suggestions.append(
                f"Energy not conserved (frac_var={energy_result['frac_var']:.2e}). "
                f"Try tighter rtol (current: {rtol:.0e})."
            )
        if "liouville_volume" in violations:
            suggestions.append(
                "Phase-space volume not preserved. Your integrator may not be "
                "symplectic. Consider using a symplectic method (e.g., Störmer–Verlet)."
            )
        if "poincare_invariant" in violations:
            suggestions.append(
                "Poincaré integral invariant violated. The symplectic 2-form "
                "is not being preserved. Check your equations of motion."
            )

        return HamiltonianReport(
            verdict=verdict,
            system_name=self.name,
            n_dof=self.n_dof,
            solver_method=method,
            rtol=rtol, atol=atol, T=T,
            quantities=quantities,
            violations=violations,
            suggestions=suggestions,
        )


# ─── Built-in systems ────────────────────────────────────────────────────────

def harmonic_oscillator(omega: float = 1.0) -> HamiltonianMonitor:
    """1D harmonic oscillator: H = p²/2 + ω²q²/2."""
    def H(z):
        q, p = z[0], z[1]
        return 0.5 * p ** 2 + 0.5 * omega ** 2 * q ** 2

    def dH(z):
        q, p = z[0], z[1]
        return np.array([omega ** 2 * q, p])

    return HamiltonianMonitor(H=H, dH=dH, n_dof=1, name="harmonic_oscillator")


def kepler_2d(mu: float = 1.0) -> HamiltonianMonitor:
    """2D Kepler problem: H = |p|²/2 − μ/|q|.

    Includes angular momentum L = q×p and Laplace–Runge–Lenz vector |A|
    as custom invariants.
    """
    def H(z):
        qx, qy, px, py = z
        r = np.sqrt(qx ** 2 + qy ** 2)
        return 0.5 * (px ** 2 + py ** 2) - mu / r

    def dH(z):
        qx, qy, px, py = z
        r = np.sqrt(qx ** 2 + qy ** 2)
        r3 = r ** 3
        return np.array([mu * qx / r3, mu * qy / r3, px, py])

    def angular_momentum(z):
        qx, qy, px, py = z
        return qx * py - qy * px

    def lrl_magnitude(z):
        """Laplace–Runge–Lenz vector magnitude (Kepler-specific)."""
        qx, qy, px, py = z
        r = np.sqrt(qx ** 2 + qy ** 2)
        L = qx * py - qy * px
        Ax = py * L - mu * qx / r
        Ay = -px * L - mu * qy / r
        return np.sqrt(Ax ** 2 + Ay ** 2)

    return HamiltonianMonitor(
        H=H, dH=dH, n_dof=2, name="kepler_2d",
        custom_invariants={
            "angular_momentum": angular_momentum,
            "LRL_magnitude": lrl_magnitude,
        },
    )


def henon_heiles() -> HamiltonianMonitor:
    """Hénon-Heiles system: H = (px²+py²)/2 + (x²+y²)/2 + x²y − y³/3.

    Integrable at low energy, chaotic at high energy. Energy is the only
    conserved quantity (no additional symmetries).
    """
    def H(z):
        x, y, px, py = z
        return 0.5 * (px ** 2 + py ** 2) + 0.5 * (x ** 2 + y ** 2) + x ** 2 * y - y ** 3 / 3

    def dH(z):
        x, y, px, py = z
        return np.array([x + 2 * x * y, y + x ** 2 - y ** 2, px, py])

    return HamiltonianMonitor(H=H, dH=dH, n_dof=2, name="henon_heiles")


def coupled_oscillators(k1: float = 1.0, k2: float = 1.0,
                        k_coupling: float = 0.1) -> HamiltonianMonitor:
    """Two coupled harmonic oscillators.

    H = (p₁²+p₂²)/2 + k₁q₁²/2 + k₂q₂²/2 + k_c(q₁−q₂)²/2
    """
    def H(z):
        q1, q2, p1, p2 = z
        return (0.5 * (p1 ** 2 + p2 ** 2) +
                0.5 * k1 * q1 ** 2 + 0.5 * k2 * q2 ** 2 +
                0.5 * k_coupling * (q1 - q2) ** 2)

    def dH(z):
        q1, q2, p1, p2 = z
        return np.array([
            k1 * q1 + k_coupling * (q1 - q2),
            k2 * q2 - k_coupling * (q1 - q2),
            p1, p2,
        ])

    return HamiltonianMonitor(
        H=H, dH=dH, n_dof=2, name="coupled_oscillators")
