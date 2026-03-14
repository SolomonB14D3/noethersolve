#!/usr/bin/env python3
"""
Test the Continuous Q_f Hypothesis

Does Q_f = ∫∫ ω(x) ω(y) f(|x-y|) dx dy remain approximately constant
for solutions of the 2D Euler equations?

We test with a simple vortex merger scenario using a pseudospectral solver.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ============================================================================
# 2D Euler Pseudospectral Solver
# ============================================================================

class Euler2DSolver:
    """Simple 2D Euler pseudospectral solver with RK4 time stepping."""

    def __init__(self, N=128, L=2*np.pi):
        self.N = N
        self.L = L
        self.dx = L / N

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
        """Compute -u·∇ω in spectral space."""
        omega = np.real(ifft2(omega_hat))
        u, v = self.compute_velocity(omega_hat)

        # ∇ω in physical space
        domega_dx = np.real(ifft2(1j * self.KX * omega_hat))
        domega_dy = np.real(ifft2(1j * self.KY * omega_hat))

        # Nonlinear term
        advection = -(u * domega_dx + v * domega_dy)

        # Transform back and dealias
        return fft2(advection) * self.dealias

    def step_rk4(self, omega_hat, dt):
        """RK4 time step."""
        k1 = self.rhs(omega_hat)
        k2 = self.rhs(omega_hat + 0.5*dt*k1)
        k3 = self.rhs(omega_hat + 0.5*dt*k2)
        k4 = self.rhs(omega_hat + dt*k3)
        return omega_hat + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    def integrate(self, omega0, T, dt=0.01, callback=None):
        """Integrate from omega0 to time T."""
        omega_hat = fft2(omega0)
        t = 0.0
        history = [(t, np.real(ifft2(omega_hat)).copy())]

        while t < T:
            if callback:
                callback(t, np.real(ifft2(omega_hat)))
            omega_hat = self.step_rk4(omega_hat, dt)
            t += dt
            if len(history) < 100 or t >= T - dt:
                history.append((t, np.real(ifft2(omega_hat)).copy()))

        return history


# ============================================================================
# Q_f Computation
# ============================================================================

def compute_Qf_fft(omega, f_values, dx):
    """
    Compute Q_f = ∫∫ ω(x) ω(y) f(|x-y|) dx dy using FFT convolution.

    f_values: precomputed f(r) on the distance grid
    """
    N = omega.shape[0]

    # Q_f = ∫∫ ω(x) ω(y) f(|x-y|) dx dy
    #     = ∫ ω(x) [∫ ω(y) f(|x-y|) dy] dx
    #     = ∫ ω(x) (ω * f)(x) dx
    # where * is convolution

    omega_hat = fft2(omega)
    f_hat = fft2(f_values)
    conv = np.real(ifft2(omega_hat * f_hat))

    Q = np.sum(omega * conv) * dx**4
    return Q


def make_distance_kernel(N, L):
    """Create distance matrix |x - y| for periodic domain."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x)

    # Distance from origin, accounting for periodicity
    dx = np.minimum(X, L - X)
    dy = np.minimum(Y, L - Y)
    R = np.sqrt(dx**2 + dy**2)

    return R


def gaussian_vortex(X, Y, x0, y0, sigma, gamma):
    """Create a Gaussian vortex centered at (x0, y0)."""
    r2 = (X - x0)**2 + (Y - y0)**2
    return gamma / (2 * np.pi * sigma**2) * np.exp(-r2 / (2 * sigma**2))


# ============================================================================
# Test Functions f(r)
# ============================================================================

def f_linear(r):
    return r

def f_sqrt(r):
    return np.sqrt(r + 1e-10)

def f_squared(r):
    return r**2

def f_log(r):
    return -np.log(r + 1e-10)  # Energy kernel (with regularization)

def f_exp(r):
    return np.exp(-r)

def f_sin(r):
    return np.sin(r)


# ============================================================================
# Main Test
# ============================================================================

def main():
    print("="*60)
    print("Continuous Q_f Hypothesis Test")
    print("="*60)
    print()
    print("Testing whether Q_f = ∫∫ ω(x) ω(y) f(|x-y|) dx dy")
    print("is approximately conserved for 2D Euler evolution.")
    print()

    # Setup
    N = 128
    L = 2 * np.pi
    dx = L / N

    solver = Euler2DSolver(N=N, L=L)
    R = make_distance_kernel(N, L)

    # Initial condition: Two co-rotating Gaussian vortices
    sigma = 0.3
    gamma = 1.0
    omega0 = (gaussian_vortex(solver.X, solver.Y, L/3, L/2, sigma, gamma) +
              gaussian_vortex(solver.X, solver.Y, 2*L/3, L/2, sigma, gamma))

    print(f"Initial condition: Two co-rotating Gaussian vortices")
    print(f"  Grid: {N}x{N}, Domain: [0, {L:.2f}]²")
    print(f"  Vortex width σ = {sigma:.2f}, circulation Γ = {gamma:.2f}")
    print()

    # Test functions
    test_functions = [
        ("f(r) = r", f_linear),
        ("f(r) = √r", f_sqrt),
        ("f(r) = r²", f_squared),
        ("f(r) = -ln(r)", f_log),
        ("f(r) = e^(-r)", f_exp),
        ("f(r) = sin(r)", f_sin),
    ]

    # Integrate
    T = 5.0
    dt = 0.01
    print(f"Integrating 2D Euler to T = {T} with dt = {dt}...")
    history = solver.integrate(omega0, T, dt)
    print(f"  Collected {len(history)} snapshots")
    print()

    # Compute Q_f time series for each test function
    print("Computing Q_f time series...")
    results = {}

    for name, f in test_functions:
        f_kernel = f(R)
        # Handle infinities at r=0
        f_kernel[0, 0] = f_kernel[0, 1]

        Qf_series = []
        for t, omega in history:
            Qf = compute_Qf_fft(omega, f_kernel, dx)
            Qf_series.append((t, Qf))

        Qf_values = np.array([q for _, q in Qf_series])
        mean_Qf = np.mean(Qf_values)
        std_Qf = np.std(Qf_values)
        frac_var = std_Qf / abs(mean_Qf) if abs(mean_Qf) > 1e-10 else np.inf

        results[name] = {
            "series": Qf_series,
            "mean": mean_Qf,
            "std": std_Qf,
            "frac_var": frac_var,
        }

    # Report
    print()
    print("="*60)
    print("RESULTS: Q_f Conservation Test")
    print("="*60)
    print()
    print(f"{'Function':<20} {'Mean Q_f':>15} {'Std':>12} {'frac_var':>12} {'Status':>10}")
    print("-"*70)

    for name, r in results.items():
        status = "✓ PASS" if r["frac_var"] < 0.05 else "✗ FAIL"
        print(f"{name:<20} {r['mean']:>15.4f} {r['std']:>12.4f} {r['frac_var']:>12.2e} {status:>10}")

    print()
    print("Pass threshold: frac_var < 0.05")

    # Summary
    n_pass = sum(1 for r in results.values() if r["frac_var"] < 0.05)
    print()
    print(f"Summary: {n_pass}/{len(results)} test functions show approximate conservation")

    # Known invariants check
    print()
    print("Verification of known invariants:")

    # Total circulation (should be exactly conserved)
    circulations = [np.sum(omega) * dx**2 for _, omega in history]
    circ_frac_var = np.std(circulations) / abs(np.mean(circulations))
    print(f"  Total circulation Γ: frac_var = {circ_frac_var:.2e}")

    # Enstrophy (should be exactly conserved for Euler)
    enstrophies = [np.sum(omega**2) * dx**2 for _, omega in history]
    enst_frac_var = np.std(enstrophies) / abs(np.mean(enstrophies))
    print(f"  Enstrophy Ω: frac_var = {enst_frac_var:.2e}")

    print()
    print("="*60)

    # Identify which Q_f are potentially new invariants
    new_invariants = []
    for name, r in results.items():
        if r["frac_var"] < 0.01 and "ln" not in name and "r²" not in name:
            new_invariants.append((name, r["frac_var"]))

    if new_invariants:
        print()
        print("POTENTIALLY NEW INVARIANTS (not ln(r) or r²):")
        for name, fv in new_invariants:
            print(f"  {name}: frac_var = {fv:.2e}")

    return results


if __name__ == "__main__":
    results = main()
