"""
noethersolve.validate — Integrator validation via conservation law monitoring.

Runs a short integration of your ODE system and checks whether the solver
configuration respects conservation laws. Catches loose tolerances, wrong
physics, and implementation bugs before they corrupt a long simulation.

Usage:
    from noethersolve.validate import validate_integrator, ValidationReport

    # Validate a vortex simulation
    report = validate_integrator(
        rhs=my_vortex_rhs,
        y0=initial_positions.ravel(),
        t_span=(0, 50),
        system="vortex",
        circulations=[1.0, -0.5, 0.3],
        rtol=1e-8,
    )
    print(report)
    # PASS / WARN / FAIL with per-quantity breakdown

    # Validate with custom conservation laws
    report = validate_integrator(
        rhs=my_ode,
        y0=y0,
        t_span=(0, 100),
        invariants={"energy": lambda y: compute_energy(y)},
    )
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple, Any
from scipy.integrate import solve_ivp

from noethersolve.monitor import frac_var, VortexMonitor, ChemicalMonitor, GravityMonitor


# ─── Thresholds ──────────────────────────────────────────────────────────────

THRESHOLDS = {
    "exact": 1e-8,      # H, Lz, momentum — should be near machine precision
    "approximate": 5e-3, # Q_f family — approximate invariants
    "warn": 1e-2,        # borderline
}


# ─── ValidationReport ────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Result of validate_integrator()."""
    verdict: str                            # PASS, WARN, or FAIL
    system: str                             # vortex, chemical, gravity, or custom
    solver_method: str
    rtol: float
    atol: float
    n_steps: int
    t_span: Tuple[float, float]
    quantities: Dict[str, dict]             # name -> {frac_var, initial, final, verdict}
    exact_violations: List[str]             # exact invariants that drifted
    approx_violations: List[str]            # approximate invariants that drifted
    suggestions: List[str]                  # human-readable fix suggestions

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Integrator Validation: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  System: {self.system}")
        lines.append(f"  Solver: {self.solver_method}, rtol={self.rtol:.0e}, atol={self.atol:.0e}")
        lines.append(f"  Steps: {self.n_steps}, t=[{self.t_span[0]}, {self.t_span[1]}]")
        lines.append("")

        # Group by verdict
        passed = [(k, v) for k, v in self.quantities.items() if v["verdict"] == "PASS"]
        warned = [(k, v) for k, v in self.quantities.items() if v["verdict"] == "WARN"]
        failed = [(k, v) for k, v in self.quantities.items() if v["verdict"] == "FAIL"]

        if failed:
            lines.append(f"  FAILED ({len(failed)}):")
            for name, data in sorted(failed, key=lambda x: -x[1]["frac_var"]):
                lines.append(f"    {name:<25s}  frac_var={data['frac_var']:.2e}")
        if warned:
            lines.append(f"  WARNINGS ({len(warned)}):")
            for name, data in sorted(warned, key=lambda x: -x[1]["frac_var"]):
                lines.append(f"    {name:<25s}  frac_var={data['frac_var']:.2e}")
        if passed:
            lines.append(f"  PASSED ({len(passed)}):")
            for name, data in sorted(passed, key=lambda x: -x[1]["frac_var"]):
                lines.append(f"    {name:<25s}  frac_var={data['frac_var']:.2e}")

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


# ─── Classify quantities ─────────────────────────────────────────────────────

# Quantities known to be exactly conserved (frac_var should be < 1e-8)
EXACT_QUANTITIES = {
    "H", "Lz", "Px", "Py", "Pz", "Lx", "Ly",
    "E",  # total energy (KE + PE); KE and PE individually are NOT conserved
    "Q_squared",  # = ΣΓ*Lz - Px² - Py², derived from exact integrals
    "total_mass", "rate_constant_product",
}

# Quantities that are expected to change and should NOT be flagged
DYNAMIC_QUANTITIES = {
    "KE", "PE",                    # only E = KE + PE is conserved
    "lyapunov_G",                  # Lyapunov function, dG/dt ≤ 0 (monotone decrease)
    "entropy_production",          # decreases toward 0 at equilibrium
    "K",                           # approximate, often noisy
}

# Quantities that include total_mass patterns from ChemicalMonitor
def _is_exact(name):
    if name in EXACT_QUANTITIES:
        return True
    if name.startswith("conservation_law_"):
        return True
    # Chemical monitor conservation laws like "-1.0*A + -1.0*B + -1.0*C"
    if "*" in name and ("+" in name or "-" in name):
        return True
    return False


# Quantities that are approximate invariants (frac_var should be < 5e-3)
APPROX_QUANTITIES = {
    "Q_linear", "Q_sqrt", "Q_exp", "Q_inv",
    "Q_tanh", "Q_sin", "R_f",
}


def _classify_quantity(name):
    """Return 'exact', 'approximate', or 'dynamic' for a quantity name."""
    if name in DYNAMIC_QUANTITIES:
        return "dynamic"
    if name.startswith("detailed_balance_"):
        return "dynamic"  # ratios approach 1.0 at equilibrium; expected to change
    if _is_exact(name):
        return "exact"
    if name in APPROX_QUANTITIES:
        return "approximate"
    # Remaining dynamic quantities (entropy_production, lyapunov_G, etc.)
    # are expected to change — don't flag them as violations
    return "dynamic"


# ─── Main validator ──────────────────────────────────────────────────────────

def validate_integrator(
    rhs: Callable,
    y0: np.ndarray,
    t_span: Tuple[float, float],
    system: str = "custom",
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    n_eval: int = 1000,
    rhs_args: tuple = (),
    # System-specific params
    circulations: Optional[List[float]] = None,
    masses: Optional[List[float]] = None,
    species: Optional[List[str]] = None,
    stoichiometry: Optional[np.ndarray] = None,
    rate_constants: Optional[np.ndarray] = None,
    reactant_matrix: Optional[np.ndarray] = None,
    reverse_pairs: Optional[List[tuple]] = None,
    # Custom invariants
    invariants: Optional[Dict[str, Callable]] = None,
    # Thresholds
    exact_threshold: float = THRESHOLDS["exact"],
    approx_threshold: float = THRESHOLDS["approximate"],
    warn_threshold: float = THRESHOLDS["warn"],
) -> ValidationReport:
    """Validate an ODE integrator by checking conservation law preservation.

    Runs solve_ivp with the given configuration, monitors all known conservation
    laws for the system type, and reports which ones are violated.

    Args:
        rhs: ODE right-hand side function f(t, y, *args)
        y0: initial state vector
        t_span: (t_start, t_end)
        system: one of "vortex", "chemical", "gravity", or "custom"
        method: scipy solver method (default "RK45")
        rtol, atol: solver tolerances
        n_eval: number of evaluation points
        rhs_args: extra args passed to rhs
        circulations: vortex strengths (required if system="vortex")
        masses: body masses (required if system="gravity")
        species, stoichiometry, etc.: chemical network params
        invariants: dict of {name: func(y) -> float} for custom invariants
        exact_threshold: frac_var above this = FAIL for exact invariants
        approx_threshold: frac_var above this = FAIL for approximate invariants
        warn_threshold: frac_var above this = WARN for approximate invariants

    Returns:
        ValidationReport with verdict, per-quantity breakdown, and suggestions.
    """
    y0 = np.asarray(y0, dtype=np.float64)

    # ── Integrate ────────────────────────────────────────────────────────
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    sol = solve_ivp(
        rhs, t_span, y0, args=rhs_args,
        method=method, t_eval=t_eval, rtol=rtol, atol=atol,
    )
    if sol.status != 0:
        return ValidationReport(
            verdict="FAIL",
            system=system,
            solver_method=method,
            rtol=rtol,
            atol=atol,
            n_steps=0,
            t_span=t_span,
            quantities={},
            exact_violations=[],
            approx_violations=[],
            suggestions=[f"Solver failed: {sol.message}. Try shorter t_span or tighter tolerances."],
        )

    # ── Build monitor ────────────────────────────────────────────────────
    monitor = None
    dt = (t_span[1] - t_span[0]) / n_eval

    if system == "vortex":
        if circulations is None:
            raise ValueError("circulations required for system='vortex'")
        G = np.asarray(circulations)
        N = len(G)
        monitor = VortexMonitor(G, threshold=exact_threshold)
        pos0 = y0.reshape(N, 2)
        monitor.set_initial(pos0)
        for i in range(1, sol.y.shape[1]):
            state = sol.y[:, i].reshape(N, 2)
            monitor.check(state, dt=dt)

    elif system == "gravity":
        if masses is None:
            raise ValueError("masses required for system='gravity'")
        m = np.asarray(masses)
        N = len(m)
        monitor = GravityMonitor(m, threshold=exact_threshold)
        # Expect y0 = [positions (N*3), velocities (N*3)]
        pos0 = y0[:N * 3].reshape(N, 3)
        vel0 = y0[N * 3:].reshape(N, 3)
        monitor.set_initial(pos0, vel0)
        for i in range(1, sol.y.shape[1]):
            pos = sol.y[:N * 3, i].reshape(N, 3)
            vel = sol.y[N * 3:, i].reshape(N, 3)
            monitor.check(pos, vel)

    elif system == "chemical":
        if species is None or stoichiometry is None:
            raise ValueError("species and stoichiometry required for system='chemical'")
        monitor = ChemicalMonitor(
            species, stoichiometry, threshold=exact_threshold,
            rate_constants=rate_constants,
            reactant_matrix=reactant_matrix,
            reverse_pairs=reverse_pairs,
        )
        monitor.set_initial(y0)
        for i in range(1, sol.y.shape[1]):
            monitor.check(sol.y[:, i])

    # ── Collect results ──────────────────────────────────────────────────
    quantities = {}
    exact_violations = []
    approx_violations = []

    if monitor is not None:
        summary = monitor.summary()
        for name, data in summary.items():
            fv = data["frac_var"]
            qtype = _classify_quantity(name)

            if qtype == "exact":
                if fv > exact_threshold:
                    verdict_q = "FAIL"
                    exact_violations.append(name)
                else:
                    verdict_q = "PASS"
            elif qtype == "approximate":
                if fv > approx_threshold:
                    verdict_q = "FAIL"
                    approx_violations.append(name)
                elif fv > warn_threshold:
                    verdict_q = "WARN"
                else:
                    verdict_q = "PASS"
            else:
                # Dynamic quantities — always PASS (they're supposed to change)
                verdict_q = "PASS"

            quantities[name] = {
                "frac_var": fv,
                "initial": data["initial"],
                "final": data["final"],
                "type": qtype,
                "verdict": verdict_q,
            }

    # ── Custom invariants ────────────────────────────────────────────────
    if invariants:
        for name, func in invariants.items():
            values = [float(func(sol.y[:, i])) for i in range(sol.y.shape[1])]
            fv = frac_var(values)
            if fv > exact_threshold:
                verdict_q = "FAIL"
                exact_violations.append(name)
            else:
                verdict_q = "PASS"
            quantities[name] = {
                "frac_var": fv,
                "initial": values[0],
                "final": values[-1],
                "type": "custom",
                "verdict": verdict_q,
            }

    # ── Overall verdict ──────────────────────────────────────────────────
    if exact_violations:
        verdict = "FAIL"
    elif approx_violations:
        verdict = "WARN"
    else:
        verdict = "PASS"

    # ── Suggestions ──────────────────────────────────────────────────────
    suggestions = []
    if exact_violations:
        suggestions.append(
            f"Exact invariants violated: {', '.join(exact_violations)}. "
            f"Try tighter rtol (current: {rtol:.0e})."
        )
        if rtol > 1e-10:
            suggestions.append("Recommended: rtol=1e-10, atol=1e-12")
        else:
            suggestions.append(
                "Tolerances are already tight. This may indicate wrong equations "
                "of motion or a missing term. Compare your RHS against the analytical form."
            )
    if approx_violations:
        suggestions.append(
            f"Approximate invariants violated: {', '.join(approx_violations)}. "
            f"This may indicate incorrect physics rather than loose tolerances."
        )
    if "K" in quantities and quantities["K"]["frac_var"] > 1.0:
        # K (kinetic invariant) requires velocity estimates — high variation
        # is expected with finite-difference velocity estimation
        pass  # Don't alarm on K — it's noisy by construction

    return ValidationReport(
        verdict=verdict,
        system=system,
        solver_method=method,
        rtol=rtol,
        atol=atol,
        n_steps=sol.y.shape[1],
        t_span=t_span,
        quantities=quantities,
        exact_violations=exact_violations,
        approx_violations=approx_violations,
        suggestions=suggestions,
    )


# ─── Convenience: compare two configurations ─────────────────────────────────

def compare_configs(
    rhs: Callable,
    y0: np.ndarray,
    t_span: Tuple[float, float],
    configs: List[Dict[str, Any]],
    **shared_kwargs,
) -> List[ValidationReport]:
    """Run validate_integrator with multiple solver configs and compare.

    Args:
        rhs, y0, t_span: same as validate_integrator
        configs: list of dicts with keys like {"method": "RK45", "rtol": 1e-8}
        **shared_kwargs: passed to all validate_integrator calls

    Returns:
        List of ValidationReport, one per config.

    Example:
        reports = compare_configs(
            rhs=vortex_rhs, y0=pos0.ravel(), t_span=(0, 100),
            configs=[
                {"rtol": 1e-6},
                {"rtol": 1e-8},
                {"rtol": 1e-10},
                {"method": "DOP853", "rtol": 1e-10},
            ],
            system="vortex",
            circulations=[1.0, -0.5, 0.3],
        )
        for r in reports:
            print(f"{r.solver_method} rtol={r.rtol:.0e}: {r.verdict}")
    """
    reports = []
    for cfg in configs:
        kwargs = {**shared_kwargs, **cfg}
        report = validate_integrator(rhs, y0, t_span, **kwargs)
        reports.append(report)
    return reports
