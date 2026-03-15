"""
noethersolve.monitor_em — Electromagnetic field conservation monitor.

Monitors conservation of standard and obscure EM invariants during
Maxwell field evolution. Catches numerical dissipation, wrong boundary
conditions, and missing terms in EM solvers.

Tracked quantities:
  Exact:   Energy (Poynting), Momentum, Helicity, Optical Chirality (Zilch Z⁰),
           Super-energy (Chevreton tensor trace)
  Derived: Zilch 3-vector magnitude, Enstrophy analog

Usage:
    from noethersolve import EMMonitor

    monitor = EMMonitor(N=64, L=2*np.pi)
    monitor.set_initial(E_fields, B_fields)

    for step in simulation:
        E, B = solver.step()
        report = monitor.check(E, B)
        if report.worst_drift > 1e-6:
            print(f"WARNING: {report.worst_name} drifted {report.worst_drift:.2e}")

References:
    - Lipkin, D.M. (1964) "Existence of a new conservation law in EM theory"
    - Tang & Cohen (2010) - Optical chirality rediscovery
    - Bliokh & Nori (2011) - Helicity and chirality in electromagnetism
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from noethersolve.monitor import frac_var, MonitorReport


class EMMonitor:
    """Monitor conservation laws in electromagnetic field simulations.

    Tracks 7 quantities: energy, momentum, optical chirality (Zilch Z⁰),
    helicity, zilch vector magnitude, super-energy, and EM enstrophy.

    The monitor accepts E and B fields as 3-tuples of 3D arrays
    (Ex, Ey, Ez) and (Bx, By, Bz). It computes spectral curls internally
    using FFT, so the grid must be periodic with uniform spacing.

    Args:
        N: Grid points per dimension.
        L: Domain length (assumed cubic, periodic).
        threshold: Fractional drift above which to alert.
    """

    def __init__(self, N: int = 64, L: float = 2 * np.pi,
                 threshold: float = 1e-6):
        self.N = N
        self.L = L
        self.dx = L / N
        self.dV = self.dx ** 3
        self.threshold = threshold

        # Wavenumbers for spectral curl
        k = fftfreq(N, d=self.dx) * 2 * np.pi
        self.KX, self.KY, self.KZ = np.meshgrid(k, k, k, indexing='ij')
        self.K2 = self.KX ** 2 + self.KY ** 2 + self.KZ ** 2
        # Avoid division by zero for helicity (A from B via k-space)
        self._K2_safe = self.K2.copy()
        self._K2_safe[0, 0, 0] = 1.0

        self._initial = {}
        self._history = {}  # name -> list of values
        self._step = 0

    # ── Spectral curl ─────────────────────────────────────────────────

    def _curl_spectral(self, Fx_hat, Fy_hat, Fz_hat):
        """Compute curl in spectral space: ∇×F = ik×F."""
        curl_x = 1j * (self.KY * Fz_hat - self.KZ * Fy_hat)
        curl_y = 1j * (self.KZ * Fx_hat - self.KX * Fz_hat)
        curl_z = 1j * (self.KX * Fy_hat - self.KY * Fx_hat)
        return curl_x, curl_y, curl_z

    def _curl_physical(self, F_hat):
        """Compute curl and return in physical space."""
        Fx_hat, Fy_hat, Fz_hat = F_hat
        cx, cy, cz = self._curl_spectral(Fx_hat, Fy_hat, Fz_hat)
        return (np.real(ifftn(cx)), np.real(ifftn(cy)), np.real(ifftn(cz)))

    # ── Invariant computations ────────────────────────────────────────

    def _compute_all(self, E, B, E_hat, B_hat):
        """Compute all monitored quantities.

        Args:
            E: (Ex, Ey, Ez) physical-space fields.
            B: (Bx, By, Bz) physical-space fields.
            E_hat: (Ex_hat, Ey_hat, Ez_hat) spectral-space fields.
            B_hat: (Bx_hat, By_hat, Bz_hat) spectral-space fields.
        """
        Ex, Ey, Ez = E
        Bx, By, Bz = B
        vals = {}

        # 1. Energy: U = (1/2) ∫ (E² + B²) d³x  (natural units)
        E_sq = Ex ** 2 + Ey ** 2 + Ez ** 2
        B_sq = Bx ** 2 + By ** 2 + Bz ** 2
        vals["energy"] = 0.5 * np.sum(E_sq + B_sq) * self.dV

        # 2. Momentum: |P| = |∫ E × B d³x|
        Px = np.sum(Ey * Bz - Ez * By) * self.dV
        Py = np.sum(Ez * Bx - Ex * Bz) * self.dV
        Pz = np.sum(Ex * By - Ey * Bx) * self.dV
        vals["momentum"] = np.sqrt(Px ** 2 + Py ** 2 + Pz ** 2)

        # Precompute curls (reused by multiple quantities)
        curl_E = self._curl_physical(E_hat)
        curl_B = self._curl_physical(B_hat)

        # 3. Optical chirality (Zilch Z⁰):
        #    C = (1/2) [E·(∇×E) + B·(∇×B)]
        E_dot_curlE = Ex * curl_E[0] + Ey * curl_E[1] + Ez * curl_E[2]
        B_dot_curlB = Bx * curl_B[0] + By * curl_B[1] + Bz * curl_B[2]
        vals["chirality"] = 0.5 * np.sum(E_dot_curlE + B_dot_curlB) * self.dV

        # 4. Helicity: H = ∫ A·B d³x where A computed from B in Coulomb gauge
        #    A_hat = -i (k × B_hat) / k²
        Bx_hat, By_hat, Bz_hat = B_hat
        kxB_x = self.KY * Bz_hat - self.KZ * By_hat
        kxB_y = self.KZ * Bx_hat - self.KX * Bz_hat
        kxB_z = self.KX * By_hat - self.KY * Bx_hat
        Ax = np.real(ifftn(-1j * kxB_x / self._K2_safe))
        Ay = np.real(ifftn(-1j * kxB_y / self._K2_safe))
        Az = np.real(ifftn(-1j * kxB_z / self._K2_safe))
        vals["helicity"] = np.sum(Ax * Bx + Ay * By + Az * Bz) * self.dV

        # 5. Zilch 3-vector magnitude:
        #    Z = c(E × ∇×B - B × ∇×E), we compute |∫ Z d³x|
        ExcB_x = Ey * curl_B[2] - Ez * curl_B[1]
        ExcB_y = Ez * curl_B[0] - Ex * curl_B[2]
        ExcB_z = Ex * curl_B[1] - Ey * curl_B[0]
        BxcE_x = By * curl_E[2] - Bz * curl_E[1]
        BxcE_y = Bz * curl_E[0] - Bx * curl_E[2]
        BxcE_z = Bx * curl_E[1] - By * curl_E[0]
        Zx = np.sum(ExcB_x - BxcE_x) * self.dV
        Zy = np.sum(ExcB_y - BxcE_y) * self.dV
        Zz = np.sum(ExcB_z - BxcE_z) * self.dV
        vals["zilch_vector"] = np.sqrt(Zx ** 2 + Zy ** 2 + Zz ** 2)

        # 6. Super-energy (Chevreton tensor trace):
        #    S = ∫ [(∇×E)² + (∇×B)²] d³x
        curlE_sq = curl_E[0] ** 2 + curl_E[1] ** 2 + curl_E[2] ** 2
        curlB_sq = curl_B[0] ** 2 + curl_B[1] ** 2 + curl_B[2] ** 2
        vals["super_energy"] = np.sum(curlE_sq + curlB_sq) * self.dV

        # 7. EM enstrophy analog: Ω = ∫ |∇×B|² d³x
        vals["enstrophy"] = np.sum(curlB_sq) * self.dV

        return vals

    # ── Public API ────────────────────────────────────────────────────

    def set_initial(self, E: Tuple, B: Tuple):
        """Set the initial E and B fields.

        Args:
            E: (Ex, Ey, Ez) — 3-tuple of 3D numpy arrays.
            B: (Bx, By, Bz) — 3-tuple of 3D numpy arrays.
        """
        E = tuple(np.asarray(f, dtype=np.float64) for f in E)
        B = tuple(np.asarray(f, dtype=np.float64) for f in B)
        E_hat = tuple(fftn(f) for f in E)
        B_hat = tuple(fftn(f) for f in B)

        self._initial = self._compute_all(E, B, E_hat, B_hat)
        self._history = {name: [val] for name, val in self._initial.items()}
        self._step = 0

    def check(self, E: Tuple, B: Tuple) -> MonitorReport:
        """Check conservation at the current step.

        Args:
            E: (Ex, Ey, Ez) — current E fields.
            B: (Bx, By, Bz) — current B fields.

        Returns:
            MonitorReport with quantities, drifts, and alerts.
        """
        if not self._initial:
            raise RuntimeError("Call set_initial() before check()")

        self._step += 1

        E = tuple(np.asarray(f, dtype=np.float64) for f in E)
        B = tuple(np.asarray(f, dtype=np.float64) for f in B)
        E_hat = tuple(fftn(f) for f in E)
        B_hat = tuple(fftn(f) for f in B)

        vals = self._compute_all(E, B, E_hat, B_hat)

        # Update history and compute drifts / frac_vars
        drifts = {}
        frac_vars = {}
        alerts = []

        for name, val in vals.items():
            self._history[name].append(val)
            init = self._initial[name]
            if abs(init) > 1e-15:
                drifts[name] = abs(val - init) / abs(init)
            else:
                drifts[name] = abs(val - init)
            frac_vars[name] = frac_var(self._history[name])
            if drifts[name] > self.threshold:
                alerts.append(name)

        worst_name = max(drifts, key=drifts.get) if drifts else ""
        worst_drift = drifts.get(worst_name, 0.0)

        return MonitorReport(
            step=self._step,
            quantities=vals,
            drifts=drifts,
            frac_vars=frac_vars,
            alerts=alerts,
            worst_name=worst_name,
            worst_drift=worst_drift,
        )

    def summary(self) -> Dict[str, dict]:
        """Return a summary of all tracked quantities."""
        result = {}
        for name, history in self._history.items():
            result[name] = {
                "initial": history[0],
                "final": history[-1],
                "frac_var": frac_var(history),
                "n_samples": len(history),
            }
        return result
