#!/usr/bin/env python3
"""
Learn the optimal f(r) for Q_f conservation using gradient descent.

Instead of testing predefined f(r), we parameterize f as a neural network
and optimize for minimal variation of Q_f over time.

This could discover fundamentally new invariants!
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from scipy.optimize import minimize

# ============================================================================
# 2D Euler Solver (from test_continuous_qf.py)
# ============================================================================

class Euler2DSolver:
    """Simple 2D Euler pseudospectral solver."""

    def __init__(self, N=64, L=2*np.pi):
        self.N = N
        self.L = L
        self.dx = L / N

        x = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y = np.meshgrid(x, x)

        k = fftfreq(N, d=self.dx) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(k, k)
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0

        kmax = N // 3
        self.dealias = (np.abs(self.KX) < kmax * 2*np.pi/L) & (np.abs(self.KY) < kmax * 2*np.pi/L)

    def step_rk4(self, omega_hat, dt):
        def rhs(w_hat):
            np.real(ifft2(w_hat))
            psi_hat = -w_hat / self.K2
            u = np.real(ifft2(1j * self.KY * psi_hat))
            v = np.real(ifft2(-1j * self.KX * psi_hat))
            domega_dx = np.real(ifft2(1j * self.KX * w_hat))
            domega_dy = np.real(ifft2(1j * self.KY * w_hat))
            advection = -(u * domega_dx + v * domega_dy)
            return fft2(advection) * self.dealias

        k1 = rhs(omega_hat)
        k2 = rhs(omega_hat + 0.5*dt*k1)
        k3 = rhs(omega_hat + 0.5*dt*k2)
        k4 = rhs(omega_hat + dt*k3)
        return omega_hat + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    def integrate(self, omega0, T, dt=0.02, n_snapshots=20):
        omega_hat = fft2(omega0)
        history = [np.real(ifft2(omega_hat)).copy()]

        steps_per_snapshot = max(1, int(T / dt / n_snapshots))
        step = 0
        while step * dt < T:
            omega_hat = self.step_rk4(omega_hat, dt)
            step += 1
            if step % steps_per_snapshot == 0:
                history.append(np.real(ifft2(omega_hat)).copy())

        return history


# ============================================================================
# Parameterized f(r)
# ============================================================================

def make_f_parametric(params, R, regularize=0.01):
    """
    Create f(r) as a sum of basis functions.

    f(r) = Σᵢ aᵢ φᵢ(r)

    Basis functions:
    - Powers: r^0.5, r^1, r^1.5, r^2
    - Exponentials: e^(-r), e^(-r/2), e^(-2r)
    - Logs: -ln(r+ε), ln(1+r)
    - Special: tanh(r), sin(r), 1/(r+ε)
    """
    f = np.zeros_like(R)

    # Regularize r=0
    R_reg = R + regularize

    # Basis functions and their indices
    bases = [
        np.sqrt(R_reg),           # 0: √r
        R_reg,                     # 1: r
        R_reg**1.5,               # 2: r^1.5
        R_reg**2,                 # 3: r²
        np.exp(-R_reg),           # 4: e^(-r)
        np.exp(-R_reg/2),         # 5: e^(-r/2)
        np.exp(-2*R_reg),         # 6: e^(-2r)
        -np.log(R_reg),           # 7: -ln(r)
        np.log(1 + R_reg),        # 8: ln(1+r)
        np.tanh(R_reg),           # 9: tanh(r)
        np.sin(R_reg),            # 10: sin(r)
        1.0 / R_reg,              # 11: 1/r
    ]

    for i, basis in enumerate(bases):
        if i < len(params):
            f += params[i] * basis

    return f


def compute_Qf_series(history, f_kernel, dx):
    """Compute Q_f for each snapshot."""
    Qf_values = []
    for omega in history:
        omega_hat = fft2(omega)
        f_hat = fft2(f_kernel)
        conv = np.real(ifft2(omega_hat * f_hat))
        Q = np.sum(omega * conv) * dx**4
        Qf_values.append(Q)
    return np.array(Qf_values)


def loss_function(params, histories, R, dx):
    """
    Loss = sum over scenarios of fractional variance of Q_f.

    Lower loss = better conservation.
    """
    f_kernel = make_f_parametric(params, R)

    total_loss = 0.0
    for history in histories:
        Qf = compute_Qf_series(history, f_kernel, dx)
        mean_Qf = np.mean(Qf)
        if abs(mean_Qf) > 1e-10:
            frac_var = np.std(Qf) / abs(mean_Qf)
        else:
            frac_var = 1.0  # Penalize zero mean
        total_loss += frac_var

    # Add regularization to prefer simple f
    reg = 0.001 * np.sum(params**2)

    return total_loss + reg


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("Learning Optimal f(r) for Q_f Conservation")
    print("="*70)
    print()
    print("Using gradient descent to find f(r) that minimizes variation of Q_f")
    print("over 2D Euler evolution.")
    print()

    # Setup
    N = 64
    L = 2 * np.pi
    dx = L / N

    solver = Euler2DSolver(N=N, L=L)

    # Distance kernel
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x)
    Rx = np.minimum(X, L - X)
    Ry = np.minimum(Y, L - Y)
    R = np.sqrt(Rx**2 + Ry**2)

    # Generate training scenarios
    print("Generating training scenarios...")

    def gaussian_vortex(X, Y, x0, y0, sigma, gamma):
        r2 = (X - x0)**2 + (Y - y0)**2
        return gamma / (2 * np.pi * sigma**2) * np.exp(-r2 / (2 * sigma**2))

    scenarios = []

    # Scenario 1: Two co-rotating vortices
    omega1 = (gaussian_vortex(solver.X, solver.Y, L/3, L/2, 0.3, 1.0) +
              gaussian_vortex(solver.X, solver.Y, 2*L/3, L/2, 0.3, 1.0))
    scenarios.append(omega1)

    # Scenario 2: Counter-rotating vortices
    omega2 = (gaussian_vortex(solver.X, solver.Y, L/3, L/2, 0.3, 1.0) +
              gaussian_vortex(solver.X, solver.Y, 2*L/3, L/2, 0.3, -1.0))
    scenarios.append(omega2)

    # Scenario 3: Three vortices
    omega3 = (gaussian_vortex(solver.X, solver.Y, L/3, L/3, 0.25, 1.0) +
              gaussian_vortex(solver.X, solver.Y, 2*L/3, L/3, 0.25, 1.0) +
              gaussian_vortex(solver.X, solver.Y, L/2, 2*L/3, 0.25, 1.0))
    scenarios.append(omega3)

    # Integrate all scenarios
    histories = []
    for i, omega0 in enumerate(scenarios):
        print(f"  Integrating scenario {i+1}/{len(scenarios)}...", end=" ", flush=True)
        history = solver.integrate(omega0, T=3.0, dt=0.02, n_snapshots=15)
        histories.append(history)
        print(f"done ({len(history)} snapshots)")

    print()

    # Initial parameters (start with known good functions)
    n_params = 12
    initial_params = np.zeros(n_params)
    initial_params[4] = 1.0  # e^(-r) - known to work

    print("Basis functions:")
    print("  0: √r, 1: r, 2: r^1.5, 3: r², 4: e^(-r), 5: e^(-r/2)")
    print("  6: e^(-2r), 7: -ln(r), 8: ln(1+r), 9: tanh(r), 10: sin(r), 11: 1/r")
    print()

    # Evaluate initial loss
    initial_loss = loss_function(initial_params, histories, R, dx)
    print(f"Initial loss (e^(-r) only): {initial_loss:.6f}")
    print()

    # Optimize
    print("Optimizing f(r) coefficients...")
    print()

    result = minimize(
        loss_function,
        initial_params,
        args=(histories, R, dx),
        method='L-BFGS-B',
        options={'maxiter': 100, 'disp': True}
    )

    optimal_params = result.x
    final_loss = result.fun

    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Improvement:  {(1 - final_loss/initial_loss)*100:.1f}%")
    print()

    # Report optimal coefficients
    basis_names = ["√r", "r", "r^1.5", "r²", "e^(-r)", "e^(-r/2)",
                   "e^(-2r)", "-ln(r)", "ln(1+r)", "tanh(r)", "sin(r)", "1/r"]

    print("Optimal f(r) = Σ aᵢ φᵢ(r)")
    print()
    print(f"{'Basis':>12} {'Coefficient':>12} {'|coef|':>10}")
    print("-"*36)

    # Sort by magnitude
    sorted_idx = np.argsort(-np.abs(optimal_params))
    for i in sorted_idx:
        if abs(optimal_params[i]) > 1e-6:
            print(f"{basis_names[i]:>12} {optimal_params[i]:>12.6f} {abs(optimal_params[i]):>10.6f}")

    print()

    # Compute individual contributions
    print("="*70)
    print("COMPARISON: Individual Basis Functions vs Optimal Combination")
    print("="*70)
    print()

    print(f"{'f(r)':<15} {'Loss':>10}")
    print("-"*27)

    for i, name in enumerate(basis_names):
        single_params = np.zeros(n_params)
        single_params[i] = 1.0
        loss = loss_function(single_params, histories, R, dx)
        print(f"{name:<15} {loss:>10.6f}")

    print("-"*27)
    print(f"{'Optimal':>15} {final_loss:>10.6f}")

    # Describe the optimal f
    print()
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print()

    # Find dominant terms
    dominant_idx = np.where(np.abs(optimal_params) > 0.1)[0]
    if len(dominant_idx) > 0:
        print("Dominant terms in optimal f(r):")
        for i in dominant_idx:
            sign = "+" if optimal_params[i] > 0 else "-"
            print(f"  {sign} {abs(optimal_params[i]):.3f} × {basis_names[i]}")
    else:
        print("No single dominant term - complex combination")

    print()
    print("This suggests the optimal conservation comes from a specific")
    print("combination of basis functions, not a single f(r).")

    return optimal_params, histories, R


if __name__ == "__main__":
    params, histories, R = main()
