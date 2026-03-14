#!/usr/bin/env python3
"""
Verify that the learned optimal f(r) generalizes to new scenarios.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

# Optimal coefficients from learning
OPTIMAL_PARAMS = np.array([
    0.018201,   # √r
    0.002180,   # r
    -0.010122,  # r^1.5
    0.001597,   # r²
    -0.010657,  # e^(-r)
    0.022819,   # e^(-r/2)
    -0.004005,  # e^(-2r)
    0.002845,   # -ln(r)
    0.008696,   # ln(1+r)
    0.021369,   # tanh(r)
    -0.019035,  # sin(r)
    0.012282,   # 1/r
])


class Euler2DSolver:
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
            omega = np.real(ifft2(w_hat))
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


def make_f_optimal(R, regularize=0.01):
    """Create optimal f(r) using learned coefficients."""
    R_reg = R + regularize
    bases = [
        np.sqrt(R_reg),
        R_reg,
        R_reg**1.5,
        R_reg**2,
        np.exp(-R_reg),
        np.exp(-R_reg/2),
        np.exp(-2*R_reg),
        -np.log(R_reg),
        np.log(1 + R_reg),
        np.tanh(R_reg),
        np.sin(R_reg),
        1.0 / R_reg,
    ]
    f = np.zeros_like(R)
    for i, basis in enumerate(bases):
        f += OPTIMAL_PARAMS[i] * basis
    return f


def compute_Qf(history, f_kernel, dx):
    """Compute Q_f for each snapshot."""
    Qf_values = []
    for omega in history:
        omega_hat = fft2(omega)
        f_hat = fft2(f_kernel)
        conv = np.real(ifft2(omega_hat * f_hat))
        Q = np.sum(omega * conv) * dx**4
        Qf_values.append(Q)
    return np.array(Qf_values)


def gaussian_vortex(X, Y, x0, y0, sigma, gamma):
    r2 = (X - x0)**2 + (Y - y0)**2
    return gamma / (2 * np.pi * sigma**2) * np.exp(-r2 / (2 * sigma**2))


def random_vortices(X, Y, L, n=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    omega = np.zeros_like(X)
    for _ in range(n):
        x0 = np.random.uniform(0.5, L-0.5)
        y0 = np.random.uniform(0.5, L-0.5)
        sigma = np.random.uniform(0.15, 0.4)
        gamma = np.random.choice([-1, 1]) * np.random.uniform(0.5, 2.0)
        omega += gaussian_vortex(X, Y, x0, y0, sigma, gamma)
    return omega


def main():
    print("="*70)
    print("Verification: Learned Optimal f(r) on New Scenarios")
    print("="*70)
    print()

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

    # Create kernels
    f_optimal = make_f_optimal(R)
    f_exp = np.exp(-R - 0.01)
    f_sqrt = np.sqrt(R + 0.01)
    f_tanh = np.tanh(R + 0.01)

    # Test on NEW scenarios (not used in training)
    test_scenarios = [
        ("4 random vortices (seed=111)", random_vortices(solver.X, solver.Y, L, n=4, seed=111)),
        ("6 random vortices (seed=222)", random_vortices(solver.X, solver.Y, L, n=6, seed=222)),
        ("8 random vortices (seed=333)", random_vortices(solver.X, solver.Y, L, n=8, seed=333)),
        ("Single strong vortex", gaussian_vortex(solver.X, solver.Y, L/2, L/2, 0.5, 3.0)),
        ("Dipole", gaussian_vortex(solver.X, solver.Y, L/3, L/2, 0.25, 2.0) +
                   gaussian_vortex(solver.X, solver.Y, 2*L/3, L/2, 0.25, -2.0)),
    ]

    print("Test scenarios (NEW, not used in training):")
    for name, _ in test_scenarios:
        print(f"  - {name}")
    print()

    print("="*70)
    print("RESULTS")
    print("="*70)
    print()

    results = {"optimal": [], "e^(-r)": [], "√r": [], "tanh(r)": []}

    print(f"{'Scenario':<35} {'Optimal':>10} {'e^(-r)':>10} {'√r':>10} {'tanh(r)':>10}")
    print("-"*77)

    for name, omega0 in test_scenarios:
        history = solver.integrate(omega0, T=4.0, dt=0.02, n_snapshots=20)

        # Compute Q_f for each kernel
        Qf_opt = compute_Qf(history, f_optimal, dx)
        Qf_exp = compute_Qf(history, f_exp, dx)
        Qf_sqrt = compute_Qf(history, f_sqrt, dx)
        Qf_tanh = compute_Qf(history, f_tanh, dx)

        # Fractional variance
        fv_opt = np.std(Qf_opt) / abs(np.mean(Qf_opt)) if abs(np.mean(Qf_opt)) > 1e-10 else np.inf
        fv_exp = np.std(Qf_exp) / abs(np.mean(Qf_exp)) if abs(np.mean(Qf_exp)) > 1e-10 else np.inf
        fv_sqrt = np.std(Qf_sqrt) / abs(np.mean(Qf_sqrt)) if abs(np.mean(Qf_sqrt)) > 1e-10 else np.inf
        fv_tanh = np.std(Qf_tanh) / abs(np.mean(Qf_tanh)) if abs(np.mean(Qf_tanh)) > 1e-10 else np.inf

        results["optimal"].append(fv_opt)
        results["e^(-r)"].append(fv_exp)
        results["√r"].append(fv_sqrt)
        results["tanh(r)"].append(fv_tanh)

        # Determine best
        fvs = [fv_opt, fv_exp, fv_sqrt, fv_tanh]
        best_idx = np.argmin(fvs)
        markers = ["", "", "", ""]
        markers[best_idx] = "*"

        print(f"{name:<35} {fv_opt:>9.2e}{markers[0]} {fv_exp:>9.2e}{markers[1]} {fv_sqrt:>9.2e}{markers[2]} {fv_tanh:>9.2e}{markers[3]}")

    print()
    print("* = best for this scenario")
    print()

    # Summary statistics
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    print(f"{'Kernel':<15} {'Mean frac_var':>15} {'Wins':>8}")
    print("-"*40)

    for name in ["optimal", "e^(-r)", "√r", "tanh(r)"]:
        mean_fv = np.mean(results[name])
        n_wins = sum(1 for i in range(len(test_scenarios))
                     if results[name][i] == min(results[k][i] for k in results))
        print(f"{name:<15} {mean_fv:>15.2e} {n_wins:>8}")

    print()

    # Check if optimal generalizes
    opt_wins = sum(1 for i in range(len(test_scenarios))
                   if results["optimal"][i] == min(results[k][i] for k in results))

    if opt_wins >= len(test_scenarios) // 2:
        print("CONCLUSION: Learned optimal f(r) GENERALIZES to new scenarios!")
    else:
        print("CONCLUSION: Learned optimal f(r) may be overfitting to training scenarios.")

    print()
    print("The optimal f(r) combines:")
    print("  f(r) ≈ 0.023 e^(-r/2) + 0.021 tanh(r) - 0.019 sin(r)")
    print("       + 0.018 √r + 0.012/r - 0.011 e^(-r) + ...")

    return results


if __name__ == "__main__":
    results = main()
