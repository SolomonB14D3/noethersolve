#!/usr/bin/env python3
"""
High-Re verification of the alignment-weighted enstrophy invariant.
Tests whether the safe fraction stays bounded as enstrophy grows.

Key test: At high Re, enstrophy GROWS (stretching > dissipation).
Does the safe fraction Q_align/Ω stay bounded away from 0?
If yes → conditional regularity criterion is physically grounded.
If no → the functional doesn't help.
"""

import numpy as np
import json
import os


def spectral_derivative(field, axis, N, dx):
    """Compute derivative using spectral method (more accurate than FD)."""
    k = np.fft.fftfreq(N, d=dx/(2*np.pi))
    field_hat = np.fft.fftn(field)
    # Build wavenumber array for the right axis
    shape = [1, 1, 1]
    shape[axis] = N
    k_arr = k.reshape(shape)
    deriv_hat = 1j * k_arr * field_hat
    return np.real(np.fft.ifftn(deriv_hat))


def velocity_from_vorticity(omega, N, dx):
    """Spectral Biot-Savart: u = curl(ψ), ∇²ψ = -ω."""
    kx = np.fft.fftfreq(N, d=dx/(2*np.pi))
    ky = np.fft.fftfreq(N, d=dx/(2*np.pi))
    kz = np.fft.fftfreq(N, d=dx/(2*np.pi))
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1

    omega_hat = np.array([np.fft.fftn(omega[i]) for i in range(3)])
    psi_hat = omega_hat / K2
    psi_hat[:, 0, 0, 0] = 0

    vel_hat = np.zeros_like(omega_hat)
    vel_hat[0] = 1j * (KY * psi_hat[2] - KZ * psi_hat[1])
    vel_hat[1] = 1j * (KZ * psi_hat[0] - KX * psi_hat[2])
    vel_hat[2] = 1j * (KX * psi_hat[1] - KY * psi_hat[0])

    return np.array([np.real(np.fft.ifftn(vel_hat[i])) for i in range(3)])


def compute_rhs_spectral(omega, N, dx, nu):
    """Compute dω/dt = (ω·∇)u + ν∇²ω using spectral methods."""
    vel = velocity_from_vorticity(omega, N, dx)

    # Stretching: (ω·∇)u_i = Σ_j ω_j ∂u_i/∂x_j
    stretch = np.zeros_like(omega)
    for i in range(3):
        for j in range(3):
            du_dx = spectral_derivative(vel[i], j, N, dx)
            stretch[i] += omega[j] * du_dx

    # Advection: -(u·∇)ω_i (for full vorticity equation)
    advection = np.zeros_like(omega)
    for i in range(3):
        for j in range(3):
            domega_dx = spectral_derivative(omega[i], j, N, dx)
            advection[i] += vel[j] * domega_dx

    # Diffusion: ν∇²ω
    laplacian = np.zeros_like(omega)
    kx = np.fft.fftfreq(N, d=dx/(2*np.pi))
    ky = np.fft.fftfreq(N, d=dx/(2*np.pi))
    kz = np.fft.fftfreq(N, d=dx/(2*np.pi))
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    for i in range(3):
        omega_hat = np.fft.fftn(omega[i])
        laplacian[i] = np.real(np.fft.ifftn(-K2 * omega_hat))

    return stretch - advection + nu * laplacian, vel


def compute_strain_eigen_fast(vel, N, dx):
    """Compute strain eigensystem using spectral derivatives."""
    grad_u = np.zeros((3, 3, N, N, N))
    for i in range(3):
        for j in range(3):
            grad_u[i, j] = spectral_derivative(vel[i], j, N, dx)

    S = 0.5 * (grad_u + grad_u.transpose(1, 0, 2, 3, 4))

    eigenvalues = np.zeros((3, N, N, N))
    eigenvectors = np.zeros((3, 3, N, N, N))

    for i in range(N):
        for j in range(N):
            for k in range(N):
                evals, evecs = np.linalg.eigh(S[:, :, i, j, k])
                idx = np.argsort(evals)[::-1]
                eigenvalues[:, i, j, k] = evals[idx]
                eigenvectors[:, :, i, j, k] = evecs[:, idx].T

    return eigenvalues, eigenvectors


def compute_metrics(omega, eigenvalues, eigenvectors, dx):
    """Compute all alignment metrics."""
    omega_mag2 = np.sum(omega**2, axis=0)
    dV = dx**3

    # Enstrophy
    enstrophy = 0.5 * np.sum(omega_mag2) * dV

    # Alignment angles
    cos2_theta = np.zeros((3, *omega.shape[1:]))
    safe_omega_mag2 = np.maximum(omega_mag2, 1e-30)
    for k in range(3):
        dot = np.sum(omega * eigenvectors[k], axis=0)
        cos2_theta[k] = dot**2 / safe_omega_mag2

    # Enstrophy-weighted alignment
    weight = omega_mag2 / (np.sum(omega_mag2) + 1e-30)
    mean_cos2 = [np.sum(weight * cos2_theta[k]) for k in range(3)]

    # Production terms
    production = np.sum(omega_mag2 * np.sum(eigenvalues * cos2_theta, axis=0)) * dV
    dangerous = np.sum(omega_mag2 * eigenvalues[0] * cos2_theta[0]) * dV

    # Safe fraction
    Q_align = np.sum(omega_mag2 * (1 - cos2_theta[0])) * dV
    safe_frac = Q_align / (2 * enstrophy) if enstrophy > 0 else 0

    # Max vorticity (BKM quantity)
    omega_mag = np.sqrt(omega_mag2)
    max_omega = np.max(omega_mag)

    return {
        'enstrophy': enstrophy,
        'Q_align': Q_align,
        'safe_fraction': safe_frac,
        'production': production,
        'dangerous_production': dangerous,
        'mean_cos2': mean_cos2,
        'max_omega': max_omega,
    }


def make_abc_flow(N, A=1.0, B=1.0, C=1.0):
    """ABC flow: Beltrami flow where ω = u (maximum stretching alignment)."""
    dx = 2 * np.pi / N
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.linspace(0, 2*np.pi, N, endpoint=False)
    z = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # ABC velocity field
    u = A * np.sin(Z) + C * np.cos(Y)
    v = B * np.sin(X) + A * np.cos(Z)
    w = C * np.sin(Y) + B * np.cos(X)

    # For Beltrami flow, ω = u (curl(u) = u when wavenumber = 1)
    omega = np.array([u, v, w])
    return omega, dx


def make_perturbed_tgv(N, amplitude=2.0):
    """Taylor-Green vortex with strong perturbations to trigger transition."""
    dx = 2 * np.pi / N
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.linspace(0, 2*np.pi, N, endpoint=False)
    z = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Base TGV
    u = amplitude * np.cos(X) * np.sin(Y) * np.cos(Z)
    v = -amplitude * np.sin(X) * np.cos(Y) * np.cos(Z)
    w = np.zeros_like(X)

    # Add multi-mode perturbations to trigger transition
    np.random.seed(42)
    for mode in range(2, 5):
        phase = np.random.uniform(0, 2*np.pi, 6)
        amp = amplitude * 0.3 / mode
        u += amp * np.sin(mode*Y + phase[0]) * np.cos(mode*Z + phase[1])
        v += amp * np.sin(mode*Z + phase[2]) * np.cos(mode*X + phase[3])
        w += amp * np.sin(mode*X + phase[4]) * np.cos(mode*Y + phase[5])

    # Compute vorticity from velocity (spectral curl)
    vel = np.array([u, v, w])
    omega = np.zeros_like(vel)
    omega[0] = spectral_derivative(w, 1, N, dx) - spectral_derivative(v, 2, N, dx)
    omega[1] = spectral_derivative(u, 2, N, dx) - spectral_derivative(w, 0, N, dx)
    omega[2] = spectral_derivative(v, 0, N, dx) - spectral_derivative(u, 1, N, dx)

    return omega, dx


def run_test(name, omega_init, dx, nu, dt, n_steps, print_every=20):
    """Run a test case and return history."""
    N = omega_init.shape[1]
    omega = omega_init.copy()

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  N={N}, nu={nu:.1e}, dt={dt}, steps={n_steps}")
    E0 = 0.5 * np.sum(velocity_from_vorticity(omega, N, dx)**2) * dx**3
    Omega0 = 0.5 * np.sum(omega**2) * dx**3
    Re_est = np.sqrt(2*E0/(2*np.pi)**3) * (2*np.pi) / nu
    print(f"  Initial E={E0:.2f}, Ω={Omega0:.2f}, Re_est~{Re_est:.0f}")
    print(f"{'='*70}")

    history = []

    for step in range(n_steps + 1):
        t = step * dt

        if step % print_every == 0 or step == n_steps:
            vel = velocity_from_vorticity(omega, N, dx)
            eigenvalues, eigenvectors = compute_strain_eigen_fast(vel, N, dx)
            m = compute_metrics(omega, eigenvalues, eigenvectors, dx)
            history.append({'t': t, **m})

            print(f"  t={t:.3f}: Ω={m['enstrophy']:.2f}  safe={m['safe_fraction']:.3f}  "
                  f"<cos²θ₁>={m['mean_cos2'][0]:.3f}  <cos²θ₂>={m['mean_cos2'][1]:.3f}  "
                  f"|ω|_max={m['max_omega']:.2f}  prod={m['production']:.2f}")

        if step < n_steps:
            # RK2 (Heun's method) for better stability
            rhs1, _ = compute_rhs_spectral(omega, N, dx, nu)
            omega_star = omega + dt * rhs1
            rhs2, _ = compute_rhs_spectral(omega_star, N, dx, nu)
            omega = omega + 0.5 * dt * (rhs1 + rhs2)

            # Dealiasing (2/3 rule)
            kx = np.fft.fftfreq(N, d=dx/(2*np.pi))
            k_max = N // 3
            for i in range(3):
                omega_hat = np.fft.fftn(omega[i])
                KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
                mask = (np.abs(KX) < k_max) & (np.abs(KY) < k_max) & (np.abs(KZ) < k_max)
                omega_hat *= mask
                omega[i] = np.real(np.fft.ifftn(omega_hat))

    return history


def analyze_history(history, name):
    """Analyze and print summary of a test run."""
    print(f"\n--- Analysis: {name} ---")

    enstrophy = np.array([h['enstrophy'] for h in history])
    safe_frac = np.array([h['safe_fraction'] for h in history])
    cos2_1 = np.array([h['mean_cos2'][0] for h in history])
    cos2_2 = np.array([h['mean_cos2'][1] for h in history])
    max_omega = np.array([h['max_omega'] for h in history])
    production = np.array([h['production'] for h in history])

    # Enstrophy growth
    enstrophy_growth = enstrophy[-1] / enstrophy[0]
    print(f"  Enstrophy growth factor: {enstrophy_growth:.3f}x")
    print(f"  Max vorticity growth:    {max_omega[-1]/max_omega[0]:.3f}x")

    # Safe fraction stability
    print(f"  Safe fraction: {safe_frac[0]:.4f} → {safe_frac[-1]:.4f}")
    print(f"    min={np.min(safe_frac):.4f}, max={np.max(safe_frac):.4f}, std={np.std(safe_frac):.4f}")

    # Depletion
    print(f"  Alignment <cos²θ₁>: {np.mean(cos2_1):.4f} (stretching)")
    print(f"  Alignment <cos²θ₂>: {np.mean(cos2_2):.4f} (intermediate)")
    depletion = np.mean(cos2_2) > np.mean(cos2_1)
    print(f"  Depletion: {'YES' if depletion else 'NO'}")

    # Critical test: does safe fraction stay bounded away from 0?
    min_safe = np.min(safe_frac)
    print(f"\n  *** CRITICAL: min safe fraction = {min_safe:.4f} ***")
    if min_safe > 0.5:
        print(f"  STRONG: >50% of enstrophy is always in safe (non-stretching) directions")
    elif min_safe > 0.33:
        print(f"  MODERATE: >33% safe — stretching doesn't dominate")
    elif min_safe > 0.1:
        print(f"  WEAK: safe fraction drops low — stretching gains ground")
    else:
        print(f"  FAILED: safe fraction collapses — alignment weighting doesn't help")

    # Correlation between enstrophy growth rate and dangerous alignment
    if len(enstrophy) > 2:
        d_enstrophy = np.diff(enstrophy) / np.diff([h['t'] for h in history])
        mid_cos2_1 = 0.5 * (cos2_1[:-1] + cos2_1[1:])
        if np.std(d_enstrophy) > 0 and np.std(mid_cos2_1) > 0:
            corr = np.corrcoef(d_enstrophy, mid_cos2_1)[0, 1]
            print(f"\n  Correlation(dΩ/dt, <cos²θ₁>): {corr:.3f}")
            if corr > 0.5:
                print(f"  CONFIRMED: enstrophy growth tracks stretching alignment")

    return {
        'name': name,
        'enstrophy_growth': float(enstrophy_growth),
        'safe_fraction_min': float(min_safe),
        'safe_fraction_mean': float(np.mean(safe_frac)),
        'safe_fraction_std': float(np.std(safe_frac)),
        'depletion': bool(depletion),
        'mean_cos2_stretching': float(np.mean(cos2_1)),
        'mean_cos2_intermediate': float(np.mean(cos2_2)),
    }


def main():
    print("="*70)
    print("  HIGH-Re ALIGNMENT INVARIANT VERIFICATION")
    print("="*70)

    results = []

    # Test 1: Perturbed TGV at moderate Re (~500)
    N = 32
    omega, dx = make_perturbed_tgv(N, amplitude=2.0)
    h1 = run_test("Perturbed TGV (Re~500)", omega, dx, nu=0.005, dt=0.003, n_steps=200)
    results.append(analyze_history(h1, "Perturbed TGV Re~500"))

    # Test 2: ABC flow at moderate Re
    omega, dx = make_abc_flow(N, A=1.0, B=np.sqrt(2/3), C=np.sqrt(1/3))
    h2 = run_test("ABC flow (Re~500)", omega, dx, nu=0.005, dt=0.003, n_steps=200)
    results.append(analyze_history(h2, "ABC flow Re~500"))

    # Test 3: Perturbed TGV at higher Re (~2000)
    omega, dx = make_perturbed_tgv(N, amplitude=4.0)
    h3 = run_test("Perturbed TGV (Re~2000)", omega, dx, nu=0.002, dt=0.002, n_steps=200)
    results.append(analyze_history(h3, "Perturbed TGV Re~2000"))

    # Summary
    print("\n\n" + "="*70)
    print("  SUMMARY ACROSS ALL TESTS")
    print("="*70)
    print(f"  {'Test':<35s} {'Ω growth':>10s} {'Safe min':>10s} {'Safe mean':>10s} {'Depletion':>10s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        print(f"  {r['name']:<35s} {r['enstrophy_growth']:>10.3f} {r['safe_fraction_min']:>10.4f} "
              f"{r['safe_fraction_mean']:>10.4f} {'YES' if r['depletion'] else 'NO':>10s}")

    # Verdict
    all_safe = all(r['safe_fraction_min'] > 0.3 for r in results)
    all_depleted = all(r['depletion'] for r in results)

    print(f"\n  Safe fraction bounded (>0.3 everywhere): {'YES' if all_safe else 'NO'}")
    print(f"  Depletion universal: {'YES' if all_depleted else 'NO'}")

    if all_safe and all_depleted:
        print(f"\n  *** EVIDENCE SUPPORTS T2 PROPOSAL ***")
        print(f"  The alignment-weighted enstrophy functional Q_align maintains")
        print(f"  a bounded safe fraction across all test cases, consistent with")
        print(f"  depletion of nonlinearity providing a physical basis for")
        print(f"  conditional regularity.")
    elif all_depleted:
        print(f"\n  Depletion holds but safe fraction not always bounded —")
        print(f"  the functional needs refinement (perhaps different weighting g(θ)).")
    else:
        print(f"\n  Mixed results — further investigation needed at higher Re.")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'alignment_invariant_high_re.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == '__main__':
    main()
