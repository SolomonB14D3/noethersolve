"""
noether_monitor.py — Conservation law monitoring library.

Sits on top of existing simulations and monitors invariants discovered by
NoetherSolve. Detects numerical drift, corruption, and wrong physics by
tracking quantities that should stay constant.

Usage:
    from noether_monitor import VortexMonitor, ChemicalMonitor

    monitor = VortexMonitor(circulations=[1.0, -0.5, 0.3])
    monitor.set_initial(positions)

    for step in simulation:
        state = integrator.step()
        report = monitor.check(state)
        if report.worst_drift > 1e-3:
            print(f"WARNING: {report.worst_name} drifted {report.worst_drift:.2e}")
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List


# ─── frac_var (shared) ───────────────────────────────────────────────────────

def frac_var(values):
    """Fractional variation: std / |mean|. The universal conservation metric."""
    arr = np.asarray(values, dtype=np.float64)
    mean = np.mean(arr)
    if abs(mean) < 1e-15:
        return float(np.std(arr))
    return float(np.std(arr) / abs(mean))


# ─── Report dataclass ────────────────────────────────────────────────────────

@dataclass
class MonitorReport:
    """Result of a single check() call."""
    step: int
    quantities: Dict[str, float]       # current values
    drifts: Dict[str, float]           # |current - initial| / |initial|
    frac_vars: Dict[str, float]        # running frac_var over all history
    alerts: List[str]                  # any quantities exceeding threshold
    worst_name: str = ""
    worst_drift: float = 0.0

    def __str__(self):
        lines = [f"Step {self.step}: {len(self.alerts)} alerts"]
        for name in sorted(self.drifts, key=lambda k: -self.drifts[k]):
            d = self.drifts[name]
            fv = self.frac_vars[name]
            flag = " <<<" if name in self.alerts else ""
            lines.append(f"  {name:30s}  drift={d:.2e}  frac_var={fv:.2e}{flag}")
        return "\n".join(lines)


# ─── Vortex Monitor ──────────────────────────────────────────────────────────

class VortexMonitor:
    """Monitor conservation laws for 2D point-vortex systems.

    Tracks:
      - H (Hamiltonian)
      - Lz (angular momentum)
      - Px, Py (linear impulse)
      - Q_f family: Q_linear, Q_squared, Q_sqrt, Q_exp, Q_inv
      - R_f = Q_exp / Q_inv (stretch-resistant ratio)
      - K = Σ Γᵢ vᵢ² (kinetic invariant)

    Args:
        circulations: list of vortex strengths [Γ₁, Γ₂, ...]
        threshold: drift above this triggers an alert (default 1e-3)
    """

    def __init__(self, circulations, threshold=1e-3):
        self.G = np.asarray(circulations, dtype=np.float64)
        self.N = len(self.G)
        self.threshold = threshold
        self._initial = {}
        self._history = {}
        self._step = 0
        self._prev_pos = None

    def set_initial(self, positions):
        """Set initial state. positions shape: (N, 2)."""
        pos = np.asarray(positions, dtype=np.float64).reshape(self.N, 2)
        self._prev_pos = pos.copy()
        vals = self._compute_all(pos, velocities=None)
        self._initial = vals.copy()
        self._history = {k: [v] for k, v in vals.items()}
        self._step = 0

    def check(self, positions, velocities=None, dt=None):
        """Check conservation at current state.

        Args:
            positions: (N, 2) array of vortex positions
            velocities: (N, 2) optional — if None, estimated from positions
            dt: timestep for velocity estimation (required if velocities is None
                and K invariant is desired)
        """
        self._step += 1
        pos = np.asarray(positions, dtype=np.float64).reshape(self.N, 2)

        # Estimate velocities if not provided
        if velocities is None and self._prev_pos is not None and dt is not None:
            velocities = (pos - self._prev_pos) / dt
        self._prev_pos = pos.copy()

        vals = self._compute_all(pos, velocities)

        # Update history
        for k, v in vals.items():
            if k in self._history:
                self._history[k].append(v)
            else:
                self._history[k] = [v]

        # Compute drifts and frac_vars
        drifts = {}
        fvars = {}
        alerts = []
        for k, v in vals.items():
            init = self._initial.get(k, v)
            if abs(init) > 1e-15:
                drifts[k] = abs(v - init) / abs(init)
            else:
                drifts[k] = abs(v - init)
            fvars[k] = frac_var(self._history[k])
            if drifts[k] > self.threshold:
                alerts.append(k)

        worst_name = max(drifts, key=drifts.get) if drifts else ""
        worst_drift = drifts.get(worst_name, 0.0)

        return MonitorReport(
            step=self._step,
            quantities=vals,
            drifts=drifts,
            frac_vars=fvars,
            alerts=alerts,
            worst_name=worst_name,
            worst_drift=worst_drift,
        )

    def _compute_all(self, pos, velocities=None):
        """Compute all monitored quantities from positions."""
        vals = {}
        G = self.G
        N = self.N

        # Pairwise distances
        dists = {}
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                r = np.sqrt(dx**2 + dy**2 + 1e-20)
                dists[(i, j)] = r

        # Hamiltonian: H = -1/(4π) Σ ΓᵢΓⱼ ln(rᵢⱼ²)
        H = 0.0
        for (i, j), r in dists.items():
            H -= G[i] * G[j] * np.log(r**2) / (4 * np.pi)
        vals["H"] = H

        # Angular momentum: Lz = Σ Γᵢ (xᵢ² + yᵢ²)
        Lz = np.sum(G * (pos[:, 0]**2 + pos[:, 1]**2))
        vals["Lz"] = Lz

        # Linear impulse
        vals["Px"] = np.sum(G * pos[:, 1])
        vals["Py"] = -np.sum(G * pos[:, 0])

        # Q_f family: Q = Σᵢ<ⱼ ΓᵢΓⱼ f(rᵢⱼ)
        r_vals = list(dists.values())
        ij_pairs = list(dists.keys())
        gg = np.array([G[i] * G[j] for i, j in ij_pairs])
        r_arr = np.array(r_vals)

        vals["Q_linear"] = np.sum(gg * r_arr)
        vals["Q_squared"] = np.sum(gg * r_arr**2)
        vals["Q_sqrt"] = np.sum(gg * np.sqrt(r_arr + 1e-10))
        vals["Q_exp"] = np.sum(gg * np.exp(-r_arr))
        vals["Q_inv"] = np.sum(gg / (r_arr + 0.05))
        vals["Q_tanh"] = np.sum(gg * np.tanh(r_arr))
        vals["Q_sin"] = np.sum(gg * np.sin(r_arr))

        # R_f = Q_exp / Q_inv (stretch-resistant ratio)
        if abs(vals["Q_inv"]) > 1e-15:
            vals["R_f"] = vals["Q_exp"] / vals["Q_inv"]
        else:
            vals["R_f"] = 0.0

        # K = Σ Γᵢ vᵢ² (kinetic invariant)
        # Compute analytically from positions — finite-difference velocity
        # estimation is too noisy. For 2D point vortices, velocity of vortex i
        # is fully determined by positions:
        #   vx_i = -Σ_{j≠i} Γⱼ (yᵢ-yⱼ) / (2π rᵢⱼ²)
        #   vy_i =  Σ_{j≠i} Γⱼ (xᵢ-xⱼ) / (2π rᵢⱼ²)
        vel_a = np.zeros_like(pos)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                r2 = dx**2 + dy**2 + 1e-20
                vel_a[i, 0] += -G[j] * dy / (2 * np.pi * r2)
                vel_a[i, 1] += G[j] * dx / (2 * np.pi * r2)
        v2 = np.sum(vel_a**2, axis=1)
        vals["K"] = np.sum(G * v2)

        return vals

    def summary(self):
        """Return summary of all monitored quantities over full history."""
        results = {}
        for k, hist in self._history.items():
            results[k] = {
                "initial": hist[0],
                "final": hist[-1],
                "frac_var": frac_var(hist),
                "n_steps": len(hist),
            }
        return results


# ─── Chemical Monitor ─────────────────────────────────────────────────────────

class ChemicalMonitor:
    """Monitor conservation laws for chemical reaction networks.

    Automatically discovers conservation laws from the stoichiometry matrix
    (left null space of S) and monitors them during simulation.

    Args:
        species: list of species names
        stoichiometry: (n_species, n_reactions) array
        threshold: drift above this triggers alert
    """

    def __init__(self, species, stoichiometry, threshold=1e-6,
                 rate_constants=None, reactant_matrix=None, reverse_pairs=None):
        """
        Args:
            species: list of species names
            stoichiometry: (n_species, n_reactions) array
            threshold: drift above this triggers alert
            rate_constants: array of rate constants (for Wegscheider/Lyapunov)
            reactant_matrix: (n_species, n_reactions) reactant stoichiometry
            reverse_pairs: list of (forward_idx, reverse_idx) tuples for
                          reversible reaction pairs (for detailed balance check)
        """
        self.species = species
        self.S = np.asarray(stoichiometry, dtype=np.float64)
        self.threshold = threshold
        self.k_rates = np.asarray(rate_constants) if rate_constants is not None else None
        self.reactant_matrix = np.asarray(reactant_matrix) if reactant_matrix is not None else None
        self.reverse_pairs = reverse_pairs or []
        self._conservation_vectors = self._find_conservation_laws()
        self._initial = {}
        self._history = {}
        self._step = 0

    def _find_conservation_laws(self):
        """Find conservation laws from left null space of S."""
        from scipy.linalg import null_space
        ns = null_space(self.S.T)
        if ns.size == 0:
            return []
        # Clean up near-zero entries
        laws = []
        for i in range(ns.shape[1]):
            w = ns[:, i]
            w[np.abs(w) < 1e-10] = 0
            if np.any(w != 0):
                # Normalize so largest coefficient is 1
                w = w / np.max(np.abs(w))
                laws.append(w)
        return laws

    def set_initial(self, concentrations):
        """Set initial concentrations."""
        c = np.asarray(concentrations, dtype=np.float64)
        vals = self._compute_all(c)
        self._initial = vals.copy()
        self._history = {k: [v] for k, v in vals.items()}
        self._step = 0

    def check(self, concentrations):
        """Check conservation at current state."""
        self._step += 1
        c = np.asarray(concentrations, dtype=np.float64)
        vals = self._compute_all(c)

        for k, v in vals.items():
            if k in self._history:
                self._history[k].append(v)
            else:
                self._history[k] = [v]

        drifts = {}
        fvars = {}
        alerts = []
        for k, v in vals.items():
            init = self._initial.get(k, v)
            if abs(init) > 1e-15:
                drifts[k] = abs(v - init) / abs(init)
            else:
                drifts[k] = abs(v - init)
            fvars[k] = frac_var(self._history[k])
            if drifts[k] > self.threshold:
                alerts.append(k)

        worst_name = max(drifts, key=drifts.get) if drifts else ""
        worst_drift = drifts.get(worst_name, 0.0)

        return MonitorReport(
            step=self._step,
            quantities=vals,
            drifts=drifts,
            frac_vars=fvars,
            alerts=alerts,
            worst_name=worst_name,
            worst_drift=worst_drift,
        )

    def _compute_all(self, c):
        """Compute all conserved quantities plus thermodynamic monitors."""
        vals = {}
        c = np.maximum(c, 1e-30)  # avoid log(0)

        # Total mass
        vals["total_mass"] = np.sum(c)

        # Conservation laws from null space
        for i, w in enumerate(self._conservation_vectors):
            name = f"conservation_law_{i}"
            nonzero = [(self.species[j], w[j]) for j in range(len(w)) if abs(w[j]) > 1e-10]
            if len(nonzero) <= 3:
                name = " + ".join(f"{coef:.1f}*{sp}" for sp, coef in nonzero)
            vals[name] = float(np.dot(w, c))

        # Detailed balance ratio for each reversible pair
        # At equilibrium: k_fwd * prod(c^reactants_fwd) = k_rev * prod(c^reactants_rev)
        # The ratio should approach 1.0 as system equilibrates
        if self.k_rates is not None and self.reactant_matrix is not None:
            for idx, (fwd, rev) in enumerate(self.reverse_pairs):
                # Forward rate
                rate_fwd = self.k_rates[fwd]
                for i in range(len(c)):
                    if self.reactant_matrix[i, fwd] > 0:
                        rate_fwd *= c[i] ** self.reactant_matrix[i, fwd]
                # Reverse rate
                rate_rev = self.k_rates[rev]
                for i in range(len(c)):
                    if self.reactant_matrix[i, rev] > 0:
                        rate_rev *= c[i] ** self.reactant_matrix[i, rev]
                # Ratio (should → 1.0 at equilibrium)
                if rate_rev > 1e-30:
                    vals[f"detailed_balance_{idx}"] = rate_fwd / rate_rev
                else:
                    vals[f"detailed_balance_{idx}"] = float("inf")

            # Rate constant consistency: product of (k_fwd/k_rev) across all
            # reversible pairs. This is strictly a Wegscheider cyclicity check
            # only when the pairs form a closed cycle (A↔B↔C↔A). For linear
            # chains (A↔B↔C), the product has no thermodynamic constraint on
            # its value, but it MUST be constant over time — if it changes,
            # the rate constants are being modified, which is non-physical.
            if len(self.reverse_pairs) >= 2:
                rate_product = 1.0
                for fwd, rev in self.reverse_pairs:
                    if self.k_rates[rev] > 1e-30:
                        rate_product *= self.k_rates[fwd] / self.k_rates[rev]
                vals["rate_constant_product"] = rate_product

            # Lyapunov function: G = Σ cᵢ(ln(cᵢ) - 1)
            # For closed systems at detailed balance, dG/dt ≤ 0
            vals["lyapunov_G"] = float(np.sum(c * (np.log(c) - 1)))

            # Entropy production: σ = Σ (v_fwd - v_rev) * ln(v_fwd/v_rev) ≥ 0
            entropy_prod = 0.0
            for fwd, rev in self.reverse_pairs:
                rate_fwd = self.k_rates[fwd]
                for i in range(len(c)):
                    if self.reactant_matrix[i, fwd] > 0:
                        rate_fwd *= c[i] ** self.reactant_matrix[i, fwd]
                rate_rev = self.k_rates[rev]
                for i in range(len(c)):
                    if self.reactant_matrix[i, rev] > 0:
                        rate_rev *= c[i] ** self.reactant_matrix[i, rev]
                if rate_fwd > 1e-30 and rate_rev > 1e-30:
                    entropy_prod += (rate_fwd - rate_rev) * np.log(rate_fwd / rate_rev)
            vals["entropy_production"] = entropy_prod
            # Must be ≥ 0 (second law). Negative = violation.

        return vals

    def summary(self):
        results = {}
        for k, hist in self._history.items():
            results[k] = {
                "initial": hist[0],
                "final": hist[-1],
                "frac_var": frac_var(hist),
                "n_steps": len(hist),
            }
        return results


# ─── 3-Body Gravitational Monitor ────────────────────────────────────────────

class GravityMonitor:
    """Monitor conservation laws for N-body gravitational systems.

    Tracks: E (total energy), Px/Py/Pz (momentum), Lz (angular momentum),
    plus Q_f family on pairwise distances.
    """

    def __init__(self, masses, threshold=1e-3, G_const=1.0):
        self.masses = np.asarray(masses, dtype=np.float64)
        self.N = len(self.masses)
        self.threshold = threshold
        self.G_const = G_const
        self._initial = {}
        self._history = {}
        self._step = 0

    def set_initial(self, positions, velocities):
        """positions: (N, 3), velocities: (N, 3)"""
        pos = np.asarray(positions, dtype=np.float64).reshape(self.N, 3)
        vel = np.asarray(velocities, dtype=np.float64).reshape(self.N, 3)
        vals = self._compute_all(pos, vel)
        self._initial = vals.copy()
        self._history = {k: [v] for k, v in vals.items()}
        self._step = 0

    def check(self, positions, velocities):
        self._step += 1
        pos = np.asarray(positions, dtype=np.float64).reshape(self.N, 3)
        vel = np.asarray(velocities, dtype=np.float64).reshape(self.N, 3)
        vals = self._compute_all(pos, vel)

        for k, v in vals.items():
            if k in self._history:
                self._history[k].append(v)
            else:
                self._history[k] = [v]

        drifts = {}
        fvars = {}
        alerts = []
        for k, v in vals.items():
            init = self._initial.get(k, v)
            if abs(init) > 1e-15:
                drifts[k] = abs(v - init) / abs(init)
            else:
                drifts[k] = abs(v - init)
            fvars[k] = frac_var(self._history[k])
            if drifts[k] > self.threshold:
                alerts.append(k)

        worst_name = max(drifts, key=drifts.get) if drifts else ""
        worst_drift = drifts.get(worst_name, 0.0)

        return MonitorReport(
            step=self._step, quantities=vals, drifts=drifts,
            frac_vars=fvars, alerts=alerts,
            worst_name=worst_name, worst_drift=worst_drift,
        )

    def _compute_all(self, pos, vel):
        vals = {}
        m = self.masses
        G = self.G_const

        # Kinetic energy
        KE = 0.5 * np.sum(m[:, None] * vel**2)
        # Potential energy
        PE = 0.0
        dists = {}
        for i in range(self.N):
            for j in range(i + 1, self.N):
                r = np.linalg.norm(pos[i] - pos[j]) + 1e-20
                PE -= G * m[i] * m[j] / r
                dists[(i, j)] = r
        vals["E"] = KE + PE
        vals["KE"] = KE
        vals["PE"] = PE

        # Momentum
        p = m[:, None] * vel
        vals["Px"] = np.sum(p[:, 0])
        vals["Py"] = np.sum(p[:, 1])
        vals["Pz"] = np.sum(p[:, 2])

        # Angular momentum
        L = np.cross(pos, p)
        vals["Lx"] = np.sum(L[:, 0])
        vals["Ly"] = np.sum(L[:, 1])
        vals["Lz"] = np.sum(L[:, 2])

        # Q_f on pairwise distances (mass-weighted)
        r_vals = list(dists.values())
        ij_pairs = list(dists.keys())
        mm = np.array([m[i] * m[j] for i, j in ij_pairs])
        r_arr = np.array(r_vals)

        vals["Q_linear"] = np.sum(mm * r_arr)
        vals["Q_exp"] = np.sum(mm * np.exp(-r_arr))
        vals["Q_inv"] = np.sum(mm / (r_arr + 0.05))

        if abs(vals["Q_inv"]) > 1e-15:
            vals["R_f"] = vals["Q_exp"] / vals["Q_inv"]

        return vals

    def summary(self):
        results = {}
        for k, hist in self._history.items():
            results[k] = {
                "initial": hist[0],
                "final": hist[-1],
                "frac_var": frac_var(hist),
                "n_steps": len(hist),
            }
        return results
