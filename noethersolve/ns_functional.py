"""
ns_functional.py — Scaling-critical functionals for 3D Navier-Stokes regularity.

Computes and monitors modified enstrophy functionals that may have better
conservation properties than raw enstrophy. Based on the pairwise vorticity
interaction framework from the Q_f family (Paper D1).

Key quantities:
- Z = ∫|ω|² dx  (enstrophy, scaling-subcritical, NOT conserved in 3D)
- E = ½∫|u|² dx  (energy, scaling-subcritical, conserved in Euler)
- H = ∫u·ω dx  (helicity, scaling-critical, conserved but NOT sign-definite)
- Q_f[ω] = ∫∫|ω(x)||ω(y)| f(|x-y|) dx dy  (pairwise functional, sign-definite)

The goal: find f such that Q_f is scaling-critical AND approximately conserved.
If such a Q_f exists, it bounds the critical Sobolev norm and regularity follows.

Physical-space kernels and their scaling:
- f(r) = r^{-2}: scaling-critical in 3D (our vortex dynamics finding)
- f(r) = r^{-1}: 3D Green's function (gives energy, subcritical)
- f(r) = 1: gives total enstrophy (subcritical)
- f(r) = -ln(r): 2D Green's function (Hamiltonian in 2D)

Usage:
    from noethersolve.ns_functional import NSFunctionalAnalyzer

    analyzer = NSFunctionalAnalyzer(grid_size=64, viscosity=1e-3)
    analyzer.set_vorticity(omega)  # 3D vorticity field
    report = analyzer.analyze()
    print(report)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable


@dataclass
class FunctionalReport:
    """Report on scaling-critical functional analysis."""
    grid_size: int
    viscosity: float
    enstrophy: float
    energy: float
    helicity: float
    stretching_rate: float          # dZ/dt from vortex stretching
    dissipation_rate: float         # dZ/dt from viscous dissipation
    enstrophy_growth_ratio: float   # (dZ/dt) / Z
    functionals: Dict[str, float]   # Q_f values for different kernels
    functional_rates: Dict[str, float]  # dQ_f/dt for each kernel
    functional_ratios: Dict[str, float]  # (dQ_f/dt) / Q_f for each kernel
    best_kernel: str                # kernel with lowest |ratio|
    best_ratio: float
    scaling_analysis: Dict[str, str]  # subcritical/critical/supercritical

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  3D Navier-Stokes Functional Analysis",
            "=" * 60,
            f"  Grid: {self.grid_size}³, ν = {self.viscosity:.1e}",
            f"  Enstrophy Z = {self.enstrophy:.4e}",
            f"  Energy E = {self.energy:.4e}",
            f"  Helicity H = {self.helicity:+.4e}",
            "",
            "  Enstrophy dynamics:",
            f"    Stretching rate:  {self.stretching_rate:+.4e}",
            f"    Dissipation rate: {self.dissipation_rate:+.4e}",
            f"    Growth ratio dZ/Z/dt: {self.enstrophy_growth_ratio:+.4e}",
            "",
            "  Pairwise functionals Q_f[ω]:",
        ]
        for name in sorted(self.functionals.keys()):
            val = self.functionals[name]
            rate = self.functional_rates.get(name, 0)
            ratio = self.functional_ratios.get(name, 0)
            scaling = self.scaling_analysis.get(name, "?")
            lines.append(
                f"    {name:20s}: Q={val:.4e}  dQ/Q/dt={ratio:+.4e}  [{scaling}]"
            )
        lines.extend([
            "",
            f"  Best kernel: {self.best_kernel} (|ratio| = {abs(self.best_ratio):.4e})",
            "=" * 60,
        ])
        return "\n".join(lines)


class NSFunctionalAnalyzer:
    """Analyze scaling-critical functionals on 3D vorticity fields.

    Uses spectral methods for accurate derivative computation.
    """

    def __init__(self, grid_size: int = 64, viscosity: float = 1e-3,
                 domain_size: float = 2 * np.pi):
        self.N = grid_size
        self.nu = viscosity
        self.L = domain_size
        self.dx = domain_size / grid_size
        self.omega = None  # (3, N, N, N) vorticity field
        self._setup_spectral()

    def _setup_spectral(self):
        """Set up spectral differentiation operators."""
        N = self.N
        L = self.L
        # Wavenumbers
        k1d = np.fft.fftfreq(N, d=L / (2 * np.pi * N))
        self.kx, self.ky, self.kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2[0, 0, 0] = 1.0  # Avoid division by zero
        self.k_mag = np.sqrt(self.k2)

    def set_vorticity(self, omega: np.ndarray):
        """Set the 3D vorticity field.

        Args:
            omega: Array of shape (3, N, N, N) representing the vorticity vector field.
        """
        assert omega.shape == (3, self.N, self.N, self.N), \
            f"Expected shape (3, {self.N}, {self.N}, {self.N}), got {omega.shape}"
        self.omega = omega.copy()

    def set_taylor_green(self, amplitude: float = 1.0):
        """Set vorticity to the Taylor-Green vortex (standard test case).

        This is an exact solution of the Euler equations with known
        enstrophy growth behavior.
        """
        N = self.N
        L = self.L
        x = np.linspace(0, L, N, endpoint=False)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

        # Taylor-Green velocity field: u = (sin x cos y cos z, -cos x sin y cos z, 0)
        # Vorticity: ω = curl u
        A = amplitude
        omega = np.zeros((3, N, N, N))
        omega[0] = A * np.cos(X) * np.sin(Y) * np.sin(Z)     # ω_x
        omega[1] = A * np.sin(X) * np.cos(Y) * np.sin(Z)      # ω_y (sign from curl)
        omega[2] = -2 * A * np.sin(X) * np.sin(Y) * np.cos(Z) # ω_z
        self.omega = omega

    def set_vortex_tube(self, radius: float = 0.3, strength: float = 1.0):
        """Set vorticity to a single vortex tube along z-axis.

        Gaussian core profile. Tests the stretching response.
        """
        N = self.N
        L = self.L
        x = np.linspace(0, L, N, endpoint=False) - L / 2
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        r2 = X**2 + Y**2

        omega = np.zeros((3, N, N, N))
        omega[2] = strength * np.exp(-r2 / (2 * radius**2))  # ω_z only
        self.omega = omega

    def _velocity_from_vorticity(self) -> np.ndarray:
        """Recover velocity from vorticity via Biot-Savart (spectral)."""
        # u = curl(ψ) where Δψ = -ω
        # In Fourier: û = ik × ω̂ / |k|²
        omega_hat = np.array([np.fft.fftn(self.omega[i]) for i in range(3)])

        u_hat = np.zeros_like(omega_hat)
        # û = ik × ω̂ / |k|²
        u_hat[0] = 1j * (self.ky * omega_hat[2] - self.kz * omega_hat[1]) / self.k2
        u_hat[1] = 1j * (self.kz * omega_hat[0] - self.kx * omega_hat[2]) / self.k2
        u_hat[2] = 1j * (self.kx * omega_hat[1] - self.ky * omega_hat[0]) / self.k2

        # Zero mean flow
        u_hat[:, 0, 0, 0] = 0

        u = np.array([np.real(np.fft.ifftn(u_hat[i])) for i in range(3)])
        return u

    def _strain_tensor(self, u: np.ndarray) -> np.ndarray:
        """Compute the strain rate tensor S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i) / 2."""
        S = np.zeros((3, 3, self.N, self.N, self.N))

        # Spectral derivatives
        u_hat = np.array([np.fft.fftn(u[i]) for i in range(3)])
        k_vec = [self.kx, self.ky, self.kz]

        for i in range(3):
            for j in range(i, 3):
                # ∂u_i/∂x_j
                du_i_dx_j = np.real(np.fft.ifftn(1j * k_vec[j] * u_hat[i]))
                # ∂u_j/∂x_i
                du_j_dx_i = np.real(np.fft.ifftn(1j * k_vec[i] * u_hat[j]))
                S[i, j] = 0.5 * (du_i_dx_j + du_j_dx_i)
                S[j, i] = S[i, j]

        return S

    def _compute_enstrophy(self) -> float:
        """Z = ∫|ω|² dx"""
        return float(np.sum(self.omega**2) * self.dx**3)

    def _compute_energy(self, u: np.ndarray) -> float:
        """E = ½∫|u|² dx"""
        return float(0.5 * np.sum(u**2) * self.dx**3)

    def _compute_helicity(self, u: np.ndarray) -> float:
        """H = ∫u·ω dx"""
        return float(np.sum(u * self.omega) * self.dx**3)

    def _compute_stretching_rate(self, S: np.ndarray) -> float:
        """Stretching contribution to dZ/dt: 2∫ω_i S_ij ω_j dx"""
        result = 0.0
        for i in range(3):
            for j in range(3):
                result += np.sum(self.omega[i] * S[i, j] * self.omega[j])
        return float(2 * result * self.dx**3)

    def _compute_dissipation_rate(self) -> float:
        """Viscous contribution to dZ/dt: -2ν∫|∇ω|² dx"""
        grad_omega_sq = 0.0
        omega_hat = np.array([np.fft.fftn(self.omega[i]) for i in range(3)])

        for i in range(3):
            # |∇ω_i|² in spectral space = |k|² |ω̂_i|²
            grad_omega_sq += np.sum(self.k2 * np.abs(omega_hat[i])**2)

        # Parseval: sum in Fourier = N^3 × sum in physical
        grad_omega_sq /= self.N**3
        return float(-2 * self.nu * grad_omega_sq * self.dx**3)

    def _compute_pairwise_functional(self, kernel_name: str) -> float:
        """Compute Q_f[ω] = ∫∫|ω(x)||ω(y)| f(|x-y|) dx dy.

        Uses spectral convolution: Q_f = ∫ |ω| * (f * |ω|) dx
        where * denotes convolution.
        """
        omega_mag = np.sqrt(np.sum(self.omega**2, axis=0))  # |ω(x)|
        omega_mag_hat = np.fft.fftn(omega_mag)

        # Kernel in Fourier space
        f_hat = self._kernel_fourier(kernel_name)

        # Convolution: (f * |ω|)(x) = IFFT(f̂ · |ω̂|)
        convolved = np.real(np.fft.ifftn(f_hat * omega_mag_hat))

        # Q_f = ∫ |ω(x)| × (f * |ω|)(x) dx
        return float(np.sum(omega_mag * convolved) * self.dx**3)

    def _kernel_fourier(self, name: str) -> np.ndarray:
        """Get the Fourier transform of a kernel f(r).

        Kernels and their Fourier transforms in 3D:
        - r^{-2}: |k|^{-1}  (scaling-critical)
        - r^{-1}: |k|^{-2}  (3D Green's function, subcritical)
        - r^0 = 1: δ(k)     (trivial, gives total enstrophy)
        - r^{-3/2}: |k|^{-3/2} (intermediate)
        """
        k_mag = self.k_mag.copy()
        k_mag[0, 0, 0] = 1.0  # Avoid division by zero

        if name == "r_inv2" or name == "r^{-2}":
            # f(r) = r^{-2}, f̂(k) ∝ |k|^{-1} (scaling-critical in 3D)
            f_hat = 1.0 / k_mag
        elif name == "r_inv1" or name == "r^{-1}":
            # f(r) = r^{-1}, f̂(k) ∝ |k|^{-2} (Green's function)
            f_hat = 1.0 / self.k2
        elif name == "r_inv32" or name == "r^{-3/2}":
            # f(r) = r^{-3/2}, f̂(k) ∝ |k|^{-3/2}
            f_hat = 1.0 / k_mag**1.5
        elif name == "const" or name == "r^0":
            # f(r) = 1, Q = (∫|ω|dx)²
            f_hat = np.zeros_like(self.k2)
            f_hat[0, 0, 0] = 1.0
        elif name == "gaussian":
            # f(r) = exp(-r²/σ²), regularized version
            sigma = self.L / 4
            f_hat = np.exp(-self.k2 * sigma**2 / 2)
        else:
            raise ValueError(f"Unknown kernel: {name}")

        f_hat[0, 0, 0] = 0  # Remove zero mode
        return f_hat

    def _scaling_classification(self, name: str) -> str:
        """Classify kernel as subcritical/critical/supercritical.

        In 3D, the scaling-critical Sobolev exponent is s=1/2 (H^{1/2}).
        A pairwise functional Q_f with kernel f(r) = r^{-α} is:
        - subcritical if α < 2 (weaker than needed)
        - critical if α = 2 (exactly right)
        - supercritical if α > 2 (too strong, may not be finite)
        """
        classification = {
            "r^{-2}": "SCALING-CRITICAL",
            "r_inv2": "SCALING-CRITICAL",
            "r^{-1}": "subcritical (gives energy)",
            "r_inv1": "subcritical (gives energy)",
            "r^{-3/2}": "subcritical (intermediate)",
            "r_inv32": "subcritical (intermediate)",
            "r^0": "subcritical (total enstrophy)",
            "const": "subcritical (total enstrophy)",
            "gaussian": "subcritical (regularized)",
        }
        return classification.get(name, "unknown")

    def _estimate_functional_rate(self, kernel_name: str,
                                  dt: float = 1e-4) -> float:
        """Estimate dQ_f/dt by finite difference with one Euler step.

        Evolves ω by one Euler step using the vorticity equation:
        ∂ω/∂t = (ω·∇)u - (u·∇)ω + ν∇²ω
        and computes Q_f at both times.
        """
        omega_save = self.omega.copy()
        Q0 = self._compute_pairwise_functional(kernel_name)

        # Compute RHS of vorticity equation
        u = self._velocity_from_vorticity()
        u_hat = np.array([np.fft.fftn(u[i]) for i in range(3)])
        omega_hat = np.array([np.fft.fftn(self.omega[i]) for i in range(3)])
        k_vec = [self.kx, self.ky, self.kz]

        # (ω·∇)u - (u·∇)ω + ν∇²ω in physical space
        rhs = np.zeros_like(self.omega)

        for i in range(3):
            # Stretching: (ω·∇)u_i = Σ_j ω_j ∂u_i/∂x_j
            for j in range(3):
                du_i_dxj = np.real(np.fft.ifftn(1j * k_vec[j] * u_hat[i]))
                rhs[i] += self.omega[j] * du_i_dxj

            # Advection: -(u·∇)ω_i = -Σ_j u_j ∂ω_i/∂x_j
            for j in range(3):
                domega_i_dxj = np.real(np.fft.ifftn(1j * k_vec[j] * omega_hat[i]))
                rhs[i] -= u[j] * domega_i_dxj

            # Viscous: ν∇²ω_i
            rhs[i] += self.nu * np.real(np.fft.ifftn(-self.k2 * omega_hat[i]))

        # Euler step
        self.omega = omega_save + dt * rhs
        Q1 = self._compute_pairwise_functional(kernel_name)

        # Restore
        self.omega = omega_save
        return (Q1 - Q0) / dt

    def analyze(self, kernels: Optional[List[str]] = None) -> FunctionalReport:
        """Full analysis of all functionals on the current vorticity field.

        Args:
            kernels: List of kernel names to test. Defaults to standard set.

        Returns:
            FunctionalReport with all quantities computed.
        """
        if self.omega is None:
            raise ValueError("Set vorticity field first (set_vorticity, set_taylor_green, etc.)")

        if kernels is None:
            kernels = ["r^{-2}", "r^{-1}", "r^{-3/2}", "gaussian"]

        u = self._velocity_from_vorticity()
        S = self._strain_tensor(u)

        enstrophy = self._compute_enstrophy()
        energy = self._compute_energy(u)
        helicity = self._compute_helicity(u)
        stretching = self._compute_stretching_rate(S)
        dissipation = self._compute_dissipation_rate()
        growth_ratio = (stretching + dissipation) / max(enstrophy, 1e-30)

        # Compute pairwise functionals
        functionals = {}
        functional_rates = {}
        functional_ratios = {}
        scaling = {}

        for name in kernels:
            Q = self._compute_pairwise_functional(name)
            dQdt = self._estimate_functional_rate(name)
            ratio = dQdt / max(abs(Q), 1e-30)
            functionals[name] = Q
            functional_rates[name] = dQdt
            functional_ratios[name] = ratio
            scaling[name] = self._scaling_classification(name)

        # Find best (lowest |ratio|) kernel
        best_name = min(functional_ratios, key=lambda k: abs(functional_ratios[k]))
        best_ratio = functional_ratios[best_name]

        return FunctionalReport(
            grid_size=self.N,
            viscosity=self.nu,
            enstrophy=enstrophy,
            energy=energy,
            helicity=helicity,
            stretching_rate=stretching,
            dissipation_rate=dissipation,
            enstrophy_growth_ratio=growth_ratio,
            functionals=functionals,
            functional_rates=functional_rates,
            functional_ratios=functional_ratios,
            best_kernel=best_name,
            best_ratio=best_ratio,
            scaling_analysis=scaling,
        )


def analyze_ns_functional(
    config: str = "taylor_green",
    grid_size: int = 32,
    viscosity: float = 1e-3,
    amplitude: float = 1.0,
) -> str:
    """Analyze scaling-critical functionals on a 3D NS vorticity field.

    This tool computes pairwise vorticity interaction functionals Q_f
    and their time derivatives for different kernels f(r). The goal is
    to find a kernel that makes Q_f approximately conserved while being
    scaling-critical (r^{-2} in 3D).

    Args:
        config: Initial vorticity configuration.
            "taylor_green" — standard TG vortex
            "vortex_tube" — single Gaussian vortex tube
        grid_size: Grid points per dimension (32, 64, 128)
        viscosity: Kinematic viscosity ν
        amplitude: Vorticity amplitude

    Returns:
        Full functional analysis report.
    """
    analyzer = NSFunctionalAnalyzer(grid_size=grid_size, viscosity=viscosity)

    if config == "taylor_green":
        analyzer.set_taylor_green(amplitude=amplitude)
    elif config == "vortex_tube":
        analyzer.set_vortex_tube(strength=amplitude)
    else:
        raise ValueError(f"Unknown config: {config}. Use 'taylor_green' or 'vortex_tube'.")

    report = analyzer.analyze()
    return str(report)
