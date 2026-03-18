#!/usr/bin/env python3
"""
Test whether Q_f can detect vorticity concentration.

Key question: As vorticity concentrates (a precursor to potential blowup),
how do different Q_f respond?

We simulate artificial concentration by shrinking a vortex while preserving
circulation, then measure Q_f.
"""

import numpy as np
from numpy.fft import fft2, ifft2

def make_distance_kernel(N, L):
    """Create distance matrix |x - y| for periodic domain."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x)
    dx = np.minimum(X, L - X)
    dy = np.minimum(Y, L - Y)
    return np.sqrt(dx**2 + dy**2)

def compute_Qf_fft(omega, f_values, dx):
    """Compute Q_f using FFT convolution."""
    omega_hat = fft2(omega)
    f_hat = fft2(f_values)
    conv = np.real(ifft2(omega_hat * f_hat))
    return np.sum(omega * conv) * dx**4

def gaussian_vortex(X, Y, x0, y0, sigma, gamma):
    """Gaussian vortex with circulation gamma."""
    r2 = (X - x0)**2 + (Y - y0)**2
    return gamma / (2 * np.pi * sigma**2) * np.exp(-r2 / (2 * sigma**2))

# Test functions
def f_log(r): return -np.log(r + 1e-10)
def f_exp(r): return np.exp(-r)
def f_sqrt(r): return np.sqrt(r + 1e-10)
def f_inv_sqrt(r): return 1.0 / np.sqrt(r + 0.01)
def f_linear(r): return r
def f_tanh(r): return np.tanh(r)

def main():
    print("="*70)
    print("Q_f Response to Vorticity Concentration")
    print("="*70)
    print()
    print("As vorticity concentrates (σ → 0 with fixed circulation Γ),")
    print("we track how different Q_f respond.")
    print()
    print("Key insight: f(r) that DECREASE near r=0 will detect concentration,")
    print("while f(r) that INCREASE near r=0 will be blind to it.")
    print()

    N = 256
    L = 2 * np.pi
    dx = L / N

    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x)
    R = make_distance_kernel(N, L)

    gamma = 1.0  # Fixed circulation
    x0, y0 = L/2, L/2

    test_functions = [
        ("-ln(r)", f_log, "diverges as r→0, DETECTS concentration"),
        ("e^(-r)", f_exp, "bounded, approaches 1 as r→0"),
        ("√r", f_sqrt, "goes to 0 as r→0"),
        ("1/√r", f_inv_sqrt, "diverges as r→0, STRONGLY DETECTS"),
        ("r", f_linear, "goes to 0 as r→0"),
        ("tanh(r)", f_tanh, "goes to 0 as r→0"),
    ]

    # Range of vortex widths (concentrating)
    sigmas = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05]

    print(f"Vortex widths σ: {sigmas}")
    print()

    # Compute Q_f for each sigma and each f
    results = {name: [] for name, _, _ in test_functions}
    enstrophies = []
    l_inf_norms = []

    for sigma in sigmas:
        omega = gaussian_vortex(X, Y, x0, y0, sigma, gamma)

        enstrophy = np.sum(omega**2) * dx**2
        l_inf = np.max(np.abs(omega))
        enstrophies.append(enstrophy)
        l_inf_norms.append(l_inf)

        for name, f, _ in test_functions:
            f_kernel = f(R)
            f_kernel[0, 0] = f_kernel[0, 1]  # Regularize
            Qf = compute_Qf_fft(omega, f_kernel, dx)
            results[name].append(Qf)

    # Results table
    print("="*70)
    print("Q_f vs Vortex Width σ")
    print("="*70)
    print()

    print(f"{'σ':<8} {'||ω||_∞':>10} {'Enstrophy':>12}", end="")
    for name, _, _ in test_functions[:4]:
        print(f" {name:>12}", end="")
    print()
    print("-"*70)

    for i, sigma in enumerate(sigmas):
        print(f"{sigma:<8.3f} {l_inf_norms[i]:>10.2f} {enstrophies[i]:>12.4f}", end="")
        for name, _, _ in test_functions[:4]:
            Qf = results[name][i]
            print(f" {Qf:>12.4f}", end="")
        print()

    # Second table for remaining functions
    print()
    print(f"{'σ':<8} {'1/√r':>12} {'r':>12} {'tanh(r)':>12}")
    print("-"*50)
    for i, sigma in enumerate(sigmas):
        print(f"{sigma:<8.3f}", end="")
        for name in ["1/√r", "r", "tanh(r)"]:
            Qf = results[name][i]
            print(f" {Qf:>12.4f}", end="")
        print()

    # Scaling analysis
    print()
    print("="*70)
    print("SCALING ANALYSIS")
    print("="*70)
    print()
    print("For Gaussian vortex with width σ and fixed circulation Γ:")
    print("  ||ω||_∞ ∝ 1/σ²")
    print("  Enstrophy ∝ 1/σ²")
    print()

    print("Q_f scaling with σ:")
    print()

    # Compute scaling exponents
    np.log(sigmas)

    for name, _, description in test_functions:
        Qf_values = np.array(results[name])
        # Use ratio of first and last to estimate scaling
        if Qf_values[-1] != 0 and Qf_values[0] != 0:
            log_ratio = np.log(abs(Qf_values[-1] / Qf_values[0]))
            sigma_ratio = np.log(sigmas[-1] / sigmas[0])
            exponent = log_ratio / sigma_ratio
            print(f"  {name:<10}: Q_f ∝ σ^{exponent:.2f}  ({description})")
        else:
            print(f"  {name:<10}: Could not determine scaling")

    # Key findings
    print()
    print("="*70)
    print("KEY FINDINGS")
    print("="*70)
    print()
    print("1. CONCENTRATION-DETECTING Q_f (sensitive to blowup):")
    print("   - Q_{-ln(r)}: Energy - diverges as σ → 0")
    print("   - Q_{1/√r}: Strongly diverges as σ → 0")
    print()
    print("2. CONCENTRATION-BLIND Q_f (insensitive to blowup):")
    print("   - Q_{√r}: Decreases as σ → 0")
    print("   - Q_r: Decreases as σ → 0")
    print("   - Q_{tanh}: Decreases as σ → 0")
    print()
    print("3. BOUNDED Q_f (intermediate behavior):")
    print("   - Q_{e^(-r)}: Approximately constant for moderate concentration")
    print()
    print("IMPLICATION FOR NAVIER-STOKES:")
    print("   If Q_{√r} is conserved but √r → 0 as vorticity concentrates,")
    print("   then circulation must spread out to compensate, preventing blowup!")
    print()
    print("   Conversely, if Q_{-ln(r)} grows during evolution, it indicates")
    print("   dangerous concentration that could lead to singularity.")

    return results, sigmas


if __name__ == "__main__":
    results, sigmas = main()
