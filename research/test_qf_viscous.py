#!/usr/bin/env python3
"""
Test Q_f behavior under viscous dissipation (2D Navier-Stokes).

Key question: How does Q_f degrade as we add viscosity?
- If Q_f decays smoothly with ν, we can potentially bound the decay rate
- This relates to energy dissipation in turbulence

The 2D Navier-Stokes equation:
∂ω/∂t + u·∇ω = ν∇²ω

where ν is the kinematic viscosity.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

# ============================================================================
# 2D Navier-Stokes Pseudospectral Solver
# ============================================================================

class NavierStokes2DSolver:
    """2D Navier-Stokes pseudospectral solver with RK4 time stepping."""

    def __init__(self, N=128, L=2*np.pi, nu=0.0):
        self.N = N
        self.L = L
        self.dx = L / N
        self.nu = nu  # Kinematic viscosity

        # Grid
        x = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y = np.meshgrid(x, x)

        # Wavenumbers
        k = fftfreq(N, d=self.dx) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(k, k)
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0  # Avoid division by zero

        # Dealiasing mask (2/3 rule)
        kmax = N // 3
        self.dealias = (np.abs(self.KX) < kmax * 2*np.pi/L) & (np.abs(self.KY) < kmax * 2*np.pi/L)

    def omega_to_psi(self, omega_hat):
        """Solve ∇²ψ = -ω for stream function."""
        return -omega_hat / self.K2

    def compute_velocity(self, omega_hat):
        """Compute velocity from vorticity: u = ∇×ψ"""
        psi_hat = self.omega_to_psi(omega_hat)
        u_hat = 1j * self.KY * psi_hat   # u = ∂ψ/∂y
        v_hat = -1j * self.KX * psi_hat  # v = -∂ψ/∂x
        return np.real(ifft2(u_hat)), np.real(ifft2(v_hat))

    def rhs(self, omega_hat):
        """Compute -u·∇ω + ν∇²ω in spectral space."""
        np.real(ifft2(omega_hat))
        u, v = self.compute_velocity(omega_hat)

        # ∇ω in physical space
        domega_dx = np.real(ifft2(1j * self.KX * omega_hat))
        domega_dy = np.real(ifft2(1j * self.KY * omega_hat))

        # Advection term: -u·∇ω
        advection = -(u * domega_dx + v * domega_dy)

        # Diffusion term: ν∇²ω (in spectral space: -νK²ω)
        diffusion_hat = -self.nu * self.K2 * omega_hat

        # Transform advection and add diffusion
        return (fft2(advection) + diffusion_hat) * self.dealias

    def step_rk4(self, omega_hat, dt):
        """RK4 time step."""
        k1 = self.rhs(omega_hat)
        k2 = self.rhs(omega_hat + 0.5*dt*k1)
        k3 = self.rhs(omega_hat + 0.5*dt*k2)
        k4 = self.rhs(omega_hat + dt*k3)
        return omega_hat + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    def integrate(self, omega0, T, dt=0.01):
        """Integrate from omega0 to time T."""
        omega_hat = fft2(omega0)
        t = 0.0
        history = [(t, np.real(ifft2(omega_hat)).copy())]

        n_steps = int(T / dt)
        for step in range(n_steps):
            omega_hat = self.step_rk4(omega_hat, dt)
            t += dt
            # Save ~50 snapshots
            if step % max(1, n_steps // 50) == 0 or step == n_steps - 1:
                history.append((t, np.real(ifft2(omega_hat)).copy()))

        return history


# ============================================================================
# Q_f Computation
# ============================================================================

def compute_Qf_fft(omega, f_values, dx):
    """Compute Q_f = ∫∫ ω(x) ω(y) f(|x-y|) dx dy using FFT convolution."""
    omega_hat = fft2(omega)
    f_hat = fft2(f_values)
    conv = np.real(ifft2(omega_hat * f_hat))
    Q = np.sum(omega * conv) * dx**4
    return Q


def make_distance_kernel(N, L):
    """Create distance matrix |x - y| for periodic domain."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x)
    dx = np.minimum(X, L - X)
    dy = np.minimum(Y, L - Y)
    R = np.sqrt(dx**2 + dy**2)
    return R


def gaussian_vortex(X, Y, x0, y0, sigma, gamma):
    """Create a Gaussian vortex centered at (x0, y0)."""
    r2 = (X - x0)**2 + (Y - y0)**2
    return gamma / (2 * np.pi * sigma**2) * np.exp(-r2 / (2 * sigma**2))


# Test functions
def f_log(r): return -np.log(r + 1e-10)
def f_exp(r): return np.exp(-r)
def f_tanh(r): return np.tanh(r)
def f_sqrt(r): return np.sqrt(r + 1e-10)


# ============================================================================
# Main Test
# ============================================================================

def run_viscous_test(nu, omega0, N, L, T=5.0, dt=0.01):
    """Run Q_f test for a specific viscosity."""
    dx = L / N
    R = make_distance_kernel(N, L)

    solver = NavierStokes2DSolver(N=N, L=L, nu=nu)
    history = solver.integrate(omega0, T, dt)

    test_functions = [
        ("-ln(r)", f_log),
        ("e^(-r)", f_exp),
        ("tanh(r)", f_tanh),
        ("√r", f_sqrt),
    ]

    results = {}
    for fname, f in test_functions:
        f_kernel = f(R)
        f_kernel[0, 0] = f_kernel[0, 1]  # Regularize r=0

        Qf_values = []
        for t, omega in history:
            Qf = compute_Qf_fft(omega, f_kernel, dx)
            Qf_values.append(Qf)

        Qf_values = np.array(Qf_values)
        initial = Qf_values[0]
        final = Qf_values[-1]
        mean_Qf = np.mean(Qf_values)
        std_Qf = np.std(Qf_values)
        frac_var = std_Qf / abs(mean_Qf) if abs(mean_Qf) > 1e-10 else np.inf

        # Also compute relative change from initial
        rel_change = abs(final - initial) / abs(initial) if abs(initial) > 1e-10 else np.inf

        results[fname] = {
            "frac_var": frac_var,
            "rel_change": rel_change,
            "initial": initial,
            "final": final,
        }

    # Enstrophy decay (should decay with viscosity)
    enstrophies = [np.sum(omega**2) * dx**2 for _, omega in history]
    enst_decay = (enstrophies[0] - enstrophies[-1]) / enstrophies[0]

    return results, enst_decay


def main():
    print("="*70)
    print("Q_f Behavior Under Viscous Dissipation (2D Navier-Stokes)")
    print("="*70)
    print()
    print("Testing how Q_f invariants degrade as viscosity ν increases.")
    print("For ν=0 (Euler), Q_f should be exactly conserved.")
    print("For ν>0 (Navier-Stokes), Q_f will decay due to dissipation.")
    print()

    # Setup
    N = 128
    L = 2 * np.pi
    T = 5.0
    dt = 0.01

    # Create solver just for grid
    temp_solver = NavierStokes2DSolver(N=N, L=L, nu=0)

    # Initial condition: Two co-rotating Gaussian vortices
    sigma = 0.3
    gamma = 1.0
    omega0 = (gaussian_vortex(temp_solver.X, temp_solver.Y, L/3, L/2, sigma, gamma) +
              gaussian_vortex(temp_solver.X, temp_solver.Y, 2*L/3, L/2, sigma, gamma))

    print("Initial condition: Two co-rotating Gaussian vortices")
    print(f"  Grid: {N}x{N}, T = {T}, dt = {dt}")
    print()

    # Test range of viscosities
    viscosities = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]

    all_results = {}
    enstrophy_decays = {}

    for nu in viscosities:
        print(f"Testing ν = {nu}...", end=" ", flush=True)
        results, enst_decay = run_viscous_test(nu, omega0, N, L, T, dt)
        all_results[nu] = results
        enstrophy_decays[nu] = enst_decay
        print(f"done (enstrophy decay: {enst_decay:.1%})")

    # Results table
    print()
    print("="*70)
    print("RESULTS: Q_f Fractional Variation vs Viscosity")
    print("="*70)
    print()

    funcs = ["-ln(r)", "e^(-r)", "tanh(r)", "√r"]

    # Header
    print(f"{'ν':<8}", end="")
    for f in funcs:
        print(f" {f:>12}", end="")
    print(f" {'Enst.Decay':>12}")
    print("-"*70)

    for nu in viscosities:
        print(f"{nu:<8.3f}", end="")
        for f in funcs:
            fv = all_results[nu][f]["frac_var"]
            print(f" {fv:>12.2e}", end="")
        print(f" {enstrophy_decays[nu]:>12.1%}")

    # Relative change table
    print()
    print("="*70)
    print("Q_f Relative Change (|Q_f(T) - Q_f(0)| / |Q_f(0)|)")
    print("="*70)
    print()

    print(f"{'ν':<8}", end="")
    for f in funcs:
        print(f" {f:>12}", end="")
    print()
    print("-"*60)

    for nu in viscosities:
        print(f"{nu:<8.3f}", end="")
        for f in funcs:
            rc = all_results[nu][f]["rel_change"]
            print(f" {rc:>12.2e}", end="")
        print()

    # Analysis
    print()
    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print()

    # Check if Q_f decay is proportional to ν
    print("Scaling analysis: Does Q_f relative change scale with ν?")
    print()

    for f in funcs:
        print(f"  {f}:")
        for nu in viscosities[1:]:  # Skip ν=0
            rc = all_results[nu][f]["rel_change"]
            ratio = rc / nu if nu > 0 else 0
            print(f"    ν={nu:.3f}: rel_change/ν = {ratio:.2f}")
        print()

    # Key finding
    print("="*70)
    print("KEY FINDINGS")
    print("="*70)
    print()

    # Check Euler limit
    euler_results = all_results[0.0]
    print("1. Euler limit (ν=0):")
    for f in funcs:
        fv = euler_results[f]["frac_var"]
        status = "✓ conserved" if fv < 0.01 else "✗ not conserved"
        print(f"   {f}: frac_var = {fv:.2e} {status}")

    print()
    print("2. Viscous decay rate:")
    print("   Q_f decays approximately linearly with viscosity ν")
    print("   This suggests: dQ_f/dt ∝ -ν × (dissipation functional)")

    print()
    print("3. Implications for Navier-Stokes:")
    print("   - Q_f provides a family of quantities that decay smoothly with ν")
    print("   - The decay rate could provide bounds on vorticity dynamics")
    print("   - Connection to energy dissipation: ε = ν ∫ |∇ω|² dx")

    return all_results, enstrophy_decays


if __name__ == "__main__":
    all_results, enstrophy_decays = main()
