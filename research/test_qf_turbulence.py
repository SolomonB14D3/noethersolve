#!/usr/bin/env python3
"""
Test Q_f conservation in 2D turbulence (more challenging test case).

This uses random initial conditions to create turbulent-like dynamics
with vortex mergers, filament stretching, and cascade behavior.
"""

import numpy as np
from test_continuous_qf import Euler2DSolver, compute_Qf_fft, make_distance_kernel

def random_vorticity_field(N, L, n_modes=20, amplitude=1.0, seed=42):
    """Generate random vorticity field with specified energy spectrum."""
    np.random.seed(seed)

    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x)

    omega = np.zeros((N, N))

    for _ in range(n_modes):
        # Random wavenumber
        kx = np.random.randint(1, N//4)
        ky = np.random.randint(1, N//4)

        # Random amplitude and phase
        a = amplitude * np.random.randn() / np.sqrt(kx**2 + ky**2)
        phi = np.random.uniform(0, 2*np.pi)

        omega += a * np.sin(2*np.pi*kx*X/L + 2*np.pi*ky*Y/L + phi)

    return omega


def vortex_patch_field(N, L, n_vortices=8, seed=42):
    """Generate field with multiple random vortex patches."""
    np.random.seed(seed)

    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x)

    omega = np.zeros((N, N))

    for _ in range(n_vortices):
        x0 = np.random.uniform(0.5, L-0.5)
        y0 = np.random.uniform(0.5, L-0.5)
        sigma = np.random.uniform(0.2, 0.5)
        gamma = np.random.choice([-1, 1]) * np.random.uniform(0.5, 2.0)

        r2 = (X - x0)**2 + (Y - y0)**2
        omega += gamma / (2*np.pi*sigma**2) * np.exp(-r2 / (2*sigma**2))

    return omega


# Test functions
def f_linear(r): return r
def f_sqrt(r): return np.sqrt(r + 1e-10)
def f_squared(r): return r**2
def f_log(r): return -np.log(r + 1e-10)
def f_exp(r): return np.exp(-r)
def f_sin(r): return np.sin(r)
def f_tanh(r): return np.tanh(r)
def f_gaussian(r): return np.exp(-r**2 / 2)


def run_turbulence_test(omega0, name, T=10.0, dt=0.01):
    """Run Q_f conservation test on given initial condition."""
    N = omega0.shape[0]
    L = 2 * np.pi
    dx = L / N

    solver = Euler2DSolver(N=N, L=L)
    R = make_distance_kernel(N, L)

    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    print(f"  Grid: {N}x{N}, T = {T}, dt = {dt}")
    print(f"  Initial enstrophy: {np.sum(omega0**2) * dx**2:.4f}")

    # Integrate
    print("  Integrating...")
    history = solver.integrate(omega0, T, dt)
    print(f"  Collected {len(history)} snapshots")

    # Test functions
    test_functions = [
        ("r", f_linear),
        ("√r", f_sqrt),
        ("r²", f_squared),
        ("-ln(r)", f_log),
        ("e^(-r)", f_exp),
        ("sin(r)", f_sin),
        ("tanh(r)", f_tanh),
        ("e^(-r²/2)", f_gaussian),
    ]

    results = {}
    print(f"\n  {'f(r)':<12} {'Mean':>12} {'Std':>10} {'frac_var':>12} {'Status':>8}")
    print(f"  {'-'*56}")

    for fname, f in test_functions:
        f_kernel = f(R)
        f_kernel[0, 0] = f_kernel[0, 1]  # Regularize r=0

        Qf_values = []
        for t, omega in history:
            Qf = compute_Qf_fft(omega, f_kernel, dx)
            Qf_values.append(Qf)

        Qf_values = np.array(Qf_values)
        mean_Qf = np.mean(Qf_values)
        std_Qf = np.std(Qf_values)
        frac_var = std_Qf / abs(mean_Qf) if abs(mean_Qf) > 1e-10 else np.inf

        status = "✓" if frac_var < 0.05 else "✗"
        results[fname] = frac_var

        print(f"  {fname:<12} {mean_Qf:>12.4f} {std_Qf:>10.4f} {frac_var:>12.2e} {status:>8}")

    # Verify standard invariants
    circs = [np.sum(omega) * dx**2 for _, omega in history]
    ensts = [np.sum(omega**2) * dx**2 for _, omega in history]
    print("\n  Control invariants:")
    print(f"    Circulation: frac_var = {np.std(circs)/abs(np.mean(circs)):.2e}")
    print(f"    Enstrophy:   frac_var = {np.std(ensts)/abs(np.mean(ensts)):.2e}")

    return results


def main():
    print("="*60)
    print("Q_f Conservation in 2D Turbulence")
    print("="*60)

    N = 128
    L = 2 * np.pi

    # Test 1: Multiple vortex patches (violent merging)
    omega1 = vortex_patch_field(N, L, n_vortices=8, seed=42)
    r1 = run_turbulence_test(omega1, "8 random vortex patches", T=8.0)

    # Test 2: Random Fourier modes (turbulent-like)
    omega2 = random_vorticity_field(N, L, n_modes=30, amplitude=2.0, seed=123)
    r2 = run_turbulence_test(omega2, "Random Fourier modes (30)", T=8.0)

    # Test 3: High enstrophy initial condition
    omega3 = vortex_patch_field(N, L, n_vortices=16, seed=999)
    r3 = run_turbulence_test(omega3, "16 vortex patches (chaotic)", T=5.0)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Q_f Conservation Across All Tests")
    print("="*60)

    all_results = {"8 patches": r1, "Fourier": r2, "16 patches": r3}

    print(f"\n{'f(r)':<12}", end="")
    for test_name in all_results:
        print(f" {test_name:>12}", end="")
    print()
    print("-" * 50)

    funcs = list(r1.keys())
    for f in funcs:
        print(f"{f:<12}", end="")
        for test_name, results in all_results.items():
            fv = results[f]
            status = "✓" if fv < 0.05 else "✗"
            print(f" {fv:>10.2e}{status}", end="")
        print()

    # Count passes
    print("\n" + "-"*50)
    print("Pass counts (frac_var < 0.05):")
    for f in funcs:
        n_pass = sum(1 for r in all_results.values() if r[f] < 0.05)
        print(f"  {f:<12}: {n_pass}/3")

    # Best performers
    print("\nBest performers (all tests pass):")
    for f in funcs:
        if all(r[f] < 0.05 for r in all_results.values()):
            avg_fv = np.mean([r[f] for r in all_results.values()])
            print(f"  {f}: avg frac_var = {avg_fv:.2e}")


if __name__ == "__main__":
    main()
