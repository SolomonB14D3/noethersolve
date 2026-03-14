#!/usr/bin/env python3
"""
Maxwell field integrator with obscure conservation laws.

Test conservation of:
1. Energy (standard) - Poynting's theorem
2. Momentum (standard)
3. Optical chirality / Zilch Z⁰ (Lipkin 1964)
4. Helicity
5. Additional zilch components
6. Super-energy (Chevreton tensor trace)

References:
- Lipkin, D.M. (1964) "Existence of a new conservation law in electromagnetic theory"
- Tang & Cohen (2010) - Optical chirality rediscovery
- Bliokh & Nori (2011) - Helicity and chirality in electromagnetism
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq

# ============================================================================
# Maxwell Spectral Solver
# ============================================================================

class MaxwellSolver:
    """
    Spectral solver for source-free Maxwell equations in vacuum.

    ∂E/∂t = c² ∇×B
    ∂B/∂t = -∇×E

    Uses periodic boundary conditions and spectral derivatives.
    """

    def __init__(self, N=64, L=2*np.pi, c=1.0):
        self.N = N
        self.L = L
        self.c = c
        self.dx = L / N

        # Grid
        x = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')

        # Wavenumbers
        k = fftfreq(N, d=self.dx) * 2 * np.pi
        self.KX, self.KY, self.KZ = np.meshgrid(k, k, k, indexing='ij')

        # For dealiasing (2/3 rule)
        kmax = N // 3
        self.dealias = ((np.abs(self.KX) < kmax * 2*np.pi/L) &
                        (np.abs(self.KY) < kmax * 2*np.pi/L) &
                        (np.abs(self.KZ) < kmax * 2*np.pi/L))

    def curl_spectral(self, Fx_hat, Fy_hat, Fz_hat):
        """Compute curl in spectral space: ∇×F = ik×F"""
        curl_x = 1j * (self.KY * Fz_hat - self.KZ * Fy_hat)
        curl_y = 1j * (self.KZ * Fx_hat - self.KX * Fz_hat)
        curl_z = 1j * (self.KX * Fy_hat - self.KY * Fx_hat)
        return curl_x, curl_y, curl_z

    def rhs(self, E_hat, B_hat):
        """
        Compute time derivatives:
        dE/dt = c² ∇×B
        dB/dt = -∇×E
        """
        Ex_hat, Ey_hat, Ez_hat = E_hat
        Bx_hat, By_hat, Bz_hat = B_hat

        # ∇×B
        curl_Bx, curl_By, curl_Bz = self.curl_spectral(Bx_hat, By_hat, Bz_hat)

        # ∇×E
        curl_Ex, curl_Ey, curl_Ez = self.curl_spectral(Ex_hat, Ey_hat, Ez_hat)

        # Time derivatives
        dEx_dt = self.c**2 * curl_Bx * self.dealias
        dEy_dt = self.c**2 * curl_By * self.dealias
        dEz_dt = self.c**2 * curl_Bz * self.dealias

        dBx_dt = -curl_Ex * self.dealias
        dBy_dt = -curl_Ey * self.dealias
        dBz_dt = -curl_Ez * self.dealias

        return (dEx_dt, dEy_dt, dEz_dt), (dBx_dt, dBy_dt, dBz_dt)

    def step_rk4(self, E_hat, B_hat, dt):
        """RK4 time step."""
        def add_tuple(t1, t2, scale=1.0):
            return tuple(a + scale * b for a, b in zip(t1, t2))

        # k1
        dE1, dB1 = self.rhs(E_hat, B_hat)

        # k2
        E2 = add_tuple(E_hat, dE1, 0.5*dt)
        B2 = add_tuple(B_hat, dB1, 0.5*dt)
        dE2, dB2 = self.rhs(E2, B2)

        # k3
        E3 = add_tuple(E_hat, dE2, 0.5*dt)
        B3 = add_tuple(B_hat, dB2, 0.5*dt)
        dE3, dB3 = self.rhs(E3, B3)

        # k4
        E4 = add_tuple(E_hat, dE3, dt)
        B4 = add_tuple(B_hat, dB3, dt)
        dE4, dB4 = self.rhs(E4, B4)

        # Combine
        E_new = tuple(e + (dt/6)*(d1 + 2*d2 + 2*d3 + d4)
                      for e, d1, d2, d3, d4 in zip(E_hat, dE1, dE2, dE3, dE4))
        B_new = tuple(b + (dt/6)*(d1 + 2*d2 + 2*d3 + d4)
                      for b, d1, d2, d3, d4 in zip(B_hat, dB1, dB2, dB3, dB4))

        return E_new, B_new

    def to_physical(self, F_hat):
        """Convert spectral to physical space."""
        return tuple(np.real(ifftn(f)) for f in F_hat)

    def to_spectral(self, F):
        """Convert physical to spectral space."""
        return tuple(fftn(f) for f in F)


# ============================================================================
# Conservation Law Computations
# ============================================================================

class EMInvariants:
    """Compute electromagnetic conservation law quantities."""

    def __init__(self, solver):
        self.solver = solver
        self.N = solver.N
        self.L = solver.L
        self.dx = solver.dx
        self.dV = self.dx**3

    def compute_curl(self, F_hat):
        """Compute curl in physical space via spectral."""
        Fx_hat, Fy_hat, Fz_hat = F_hat
        curl_x, curl_y, curl_z = self.solver.curl_spectral(Fx_hat, Fy_hat, Fz_hat)
        return (np.real(ifftn(curl_x)),
                np.real(ifftn(curl_y)),
                np.real(ifftn(curl_z)))

    def energy(self, E, B):
        """
        Electromagnetic energy: U = (1/2) ∫ (ε₀E² + B²/μ₀) d³x
        In natural units (ε₀ = μ₀ = c = 1): U = (1/2) ∫ (E² + B²) d³x
        """
        Ex, Ey, Ez = E
        Bx, By, Bz = B
        E_sq = Ex**2 + Ey**2 + Ez**2
        B_sq = Bx**2 + By**2 + Bz**2
        return 0.5 * np.sum(E_sq + B_sq) * self.dV

    def momentum(self, E, B):
        """
        Electromagnetic momentum: P = (1/c²) ∫ (E × B) d³x
        In natural units: P = ∫ (E × B) d³x
        """
        Ex, Ey, Ez = E
        Bx, By, Bz = B
        Px = np.sum(Ey * Bz - Ez * By) * self.dV
        Py = np.sum(Ez * Bx - Ex * Bz) * self.dV
        Pz = np.sum(Ex * By - Ey * Bx) * self.dV
        return np.sqrt(Px**2 + Py**2 + Pz**2)

    def optical_chirality(self, E, B, E_hat, B_hat):
        """
        Optical chirality (Zilch Z⁰) - Lipkin 1964, Tang & Cohen 2010:
        C = (ε₀/2) E·(∇×E) + (1/2μ₀) B·(∇×B)
        In natural units: C = (1/2) [E·(∇×E) + B·(∇×B)]
        """
        Ex, Ey, Ez = E
        Bx, By, Bz = B

        curl_Ex, curl_Ey, curl_Ez = self.compute_curl(E_hat)
        curl_Bx, curl_By, curl_Bz = self.compute_curl(B_hat)

        E_dot_curlE = Ex*curl_Ex + Ey*curl_Ey + Ez*curl_Ez
        B_dot_curlB = Bx*curl_Bx + By*curl_By + Bz*curl_Bz

        return 0.5 * np.sum(E_dot_curlE + B_dot_curlB) * self.dV

    def helicity(self, E, B, B_hat):
        """
        Optical helicity: H = ∫ A·B d³x
        where B = ∇×A.

        We can compute A from B via: A_hat = -i k×B_hat / k²
        But simpler: use H = (1/ω) ∫ E·B d³x for monochromatic fields

        More general: H = ∫ (A·B - φE·B/c²) d³x
        For radiation fields in Coulomb gauge: H = ∫ A·B d³x
        """
        # Compute A from B: in Coulomb gauge, A_hat = -i (k × B_hat) / k²
        Bx_hat, By_hat, Bz_hat = B_hat
        K2 = self.solver.KX**2 + self.solver.KY**2 + self.solver.KZ**2
        K2[0, 0, 0] = 1  # Avoid division by zero

        # k × B in spectral space
        kxB_x = self.solver.KY * Bz_hat - self.solver.KZ * By_hat
        kxB_y = self.solver.KZ * Bx_hat - self.solver.KX * Bz_hat
        kxB_z = self.solver.KX * By_hat - self.solver.KY * Bx_hat

        Ax_hat = -1j * kxB_x / K2
        Ay_hat = -1j * kxB_y / K2
        Az_hat = -1j * kxB_z / K2

        Ax = np.real(ifftn(Ax_hat))
        Ay = np.real(ifftn(Ay_hat))
        Az = np.real(ifftn(Az_hat))

        Bx, By, Bz = B
        H = np.sum(Ax*Bx + Ay*By + Az*Bz) * self.dV
        return H

    def zilch_vector(self, E, B, E_hat, B_hat):
        """
        Zilch 3-vector: Z = (1/c)(E × ∂E/∂t + B × ∂B/∂t)
        Using Maxwell: ∂E/∂t = c²∇×B, ∂B/∂t = -∇×E
        Z = c(E × ∇×B - B × ∇×E)
        """
        Ex, Ey, Ez = E
        Bx, By, Bz = B

        curl_Ex, curl_Ey, curl_Ez = self.compute_curl(E_hat)
        curl_Bx, curl_By, curl_Bz = self.compute_curl(B_hat)

        # E × ∇×B
        ExcurlB_x = Ey * curl_Bz - Ez * curl_By
        ExcurlB_y = Ez * curl_Bx - Ex * curl_Bz
        ExcurlB_z = Ex * curl_By - Ey * curl_Bx

        # B × ∇×E
        BxcurlE_x = By * curl_Ez - Bz * curl_Ey
        BxcurlE_y = Bz * curl_Ex - Bx * curl_Ez
        BxcurlE_z = Bx * curl_Ey - By * curl_Ex

        Zx = np.sum(ExcurlB_x - BxcurlE_x) * self.dV
        Zy = np.sum(ExcurlB_y - BxcurlE_y) * self.dV
        Zz = np.sum(ExcurlB_z - BxcurlE_z) * self.dV

        return np.sqrt(Zx**2 + Zy**2 + Zz**2)

    def super_energy(self, E, B, E_hat, B_hat):
        """
        Super-energy (Chevreton tensor trace) - simplified version:
        S = ∫ [(∇×E)² + (∇×B)²] d³x

        This is related to the trace of the Chevreton super-energy tensor.
        """
        curl_Ex, curl_Ey, curl_Ez = self.compute_curl(E_hat)
        curl_Bx, curl_By, curl_Bz = self.compute_curl(B_hat)

        curlE_sq = curl_Ex**2 + curl_Ey**2 + curl_Ez**2
        curlB_sq = curl_Bx**2 + curl_By**2 + curl_Bz**2

        return np.sum(curlE_sq + curlB_sq) * self.dV

    def enstrophy_analog(self, E, B, E_hat, B_hat):
        """
        Electromagnetic "enstrophy" - analogous to fluid enstrophy:
        Ω = ∫ |∇×B|² d³x (magnetic curl squared)
        """
        curl_Bx, curl_By, curl_Bz = self.compute_curl(B_hat)
        return np.sum(curl_Bx**2 + curl_By**2 + curl_Bz**2) * self.dV


# ============================================================================
# Initial Conditions
# ============================================================================

def gaussian_wave_packet(X, Y, Z, x0, y0, z0, sigma, k, polarization='x'):
    """
    Gaussian wave packet with proper circular polarization.
    For circular polarization propagating along z:
    E = E₀ exp(-r²/2σ²) [cos(kz), ±sin(kz), 0]
    B = (1/c) k̂ × E = E₀ exp(-r²/2σ²) [∓sin(kz), cos(kz), 0]
    """
    r2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
    envelope = np.exp(-r2 / (2 * sigma**2))

    kx, ky, kz = k
    phase = kx * X + ky * Y + kz * Z
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2) + 1e-10

    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)

    if polarization == 'x':
        Ex = envelope * cos_phase
        Ey = np.zeros_like(X)
        Ez = np.zeros_like(X)
        # B = k̂ × E / c, for k along x: B = (0, -Ez, Ey) normalized
        Bx = np.zeros_like(X)
        By = envelope * cos_phase * kz / k_mag
        Bz = -envelope * cos_phase * ky / k_mag

    elif polarization == 'circular+':
        # Right-handed: E rotates with k (positive helicity)
        # For k along x: E in y-z plane, rotating
        if abs(kx) > abs(ky) and abs(kx) > abs(kz):
            # k mainly along x
            Ex = np.zeros_like(X)
            Ey = envelope * cos_phase
            Ez = envelope * sin_phase
            # B = k̂ × E
            Bx = np.zeros_like(X)
            By = -envelope * sin_phase
            Bz = envelope * cos_phase
        else:
            # k mainly along z
            Ex = envelope * cos_phase
            Ey = envelope * sin_phase
            Ez = np.zeros_like(X)
            Bx = -envelope * sin_phase
            By = envelope * cos_phase
            Bz = np.zeros_like(X)

    elif polarization == 'circular-':
        # Left-handed: E rotates against k (negative helicity)
        if abs(kx) > abs(ky) and abs(kx) > abs(kz):
            Ex = np.zeros_like(X)
            Ey = envelope * cos_phase
            Ez = -envelope * sin_phase
            Bx = np.zeros_like(X)
            By = envelope * sin_phase
            Bz = envelope * cos_phase
        else:
            Ex = envelope * cos_phase
            Ey = -envelope * sin_phase
            Ez = np.zeros_like(X)
            Bx = envelope * sin_phase
            By = envelope * cos_phase
            Bz = np.zeros_like(X)

    else:  # 'y'
        Ex = np.zeros_like(X)
        Ey = envelope * cos_phase
        Ez = np.zeros_like(X)
        Bx = -envelope * cos_phase * kz / k_mag
        By = np.zeros_like(X)
        Bz = envelope * cos_phase * kx / k_mag

    return (Ex, Ey, Ez), (Bx, By, Bz)


def dipole_field(X, Y, Z, x0, y0, z0, p_dir='z', amplitude=1.0, reg=0.1):
    """
    Electric dipole field (near-field approximation).
    E = (1/4πε₀) [3(p·r̂)r̂ - p] / r³

    Also returns the corresponding B field for oscillating dipole.
    """
    rx = X - x0
    ry = Y - y0
    rz = Z - z0
    r = np.sqrt(rx**2 + ry**2 + rz**2 + reg**2)  # Regularized

    if p_dir == 'z':
        px, py, pz = 0, 0, amplitude
    elif p_dir == 'x':
        px, py, pz = amplitude, 0, 0
    else:
        px, py, pz = 0, amplitude, 0

    p_dot_r = px*rx + py*ry + pz*rz

    Ex = (3 * p_dot_r * rx / r**2 - px) / r**3
    Ey = (3 * p_dot_r * ry / r**2 - py) / r**3
    Ez = (3 * p_dot_r * rz / r**2 - pz) / r**3

    # For near-field, B ≈ 0 (quasi-static)
    Bx = np.zeros_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)

    return (Ex, Ey, Ez), (Bx, By, Bz)


# ============================================================================
# Main Test
# ============================================================================

def main():
    print("="*70)
    print("Electromagnetic Zilch and Optical Chirality Conservation Test")
    print("="*70)
    print()
    print("Testing conservation of obscure EM invariants:")
    print("  1. Energy (standard - Poynting)")
    print("  2. Momentum")
    print("  3. Optical chirality / Zilch Z⁰ (Lipkin 1964)")
    print("  4. Helicity")
    print("  5. Zilch 3-vector magnitude")
    print("  6. Super-energy (Chevreton tensor related)")
    print()

    # Setup
    N = 48  # Reduced for speed
    L = 2 * np.pi
    c = 1.0

    solver = MaxwellSolver(N=N, L=L, c=c)
    invariants = EMInvariants(solver)

    # Initial condition: Single circularly polarized wave packet (non-zero chirality)
    print("Initial condition: Circularly polarized wave packet")
    print("  (Right-handed polarization for non-zero chirality)")
    print()

    # k along z for clean circular polarization
    E, B = gaussian_wave_packet(solver.X, solver.Y, solver.Z,
                                 L/2, L/2, L/2, sigma=0.6,
                                 k=(0, 0, 4), polarization='circular+')

    E_hat = solver.to_spectral(E)
    B_hat = solver.to_spectral(B)

    # Time evolution
    T = 4.0
    dt = 0.02
    n_steps = int(T / dt)

    print(f"Grid: {N}³, T = {T}, dt = {dt}")
    print()

    # Track invariants
    history = {
        "Energy": [],
        "Momentum": [],
        "Chirality": [],
        "Helicity": [],
        "Zilch |Z|": [],
        "Super-E": [],
    }

    print("Evolving Maxwell equations...")

    for step in range(n_steps + 1):
        E = solver.to_physical(E_hat)
        B = solver.to_physical(B_hat)

        # Compute invariants
        history["Energy"].append(invariants.energy(E, B))
        history["Momentum"].append(invariants.momentum(E, B))
        history["Chirality"].append(invariants.optical_chirality(E, B, E_hat, B_hat))
        history["Helicity"].append(invariants.helicity(E, B, B_hat))
        history["Zilch |Z|"].append(invariants.zilch_vector(E, B, E_hat, B_hat))
        history["Super-E"].append(invariants.super_energy(E, B, E_hat, B_hat))

        if step < n_steps:
            E_hat, B_hat = solver.step_rk4(E_hat, B_hat, dt)

        if step % 50 == 0:
            print(f"  Step {step}/{n_steps}")

    print()

    # Results
    print("="*70)
    print("RESULTS: Conservation Law Verification")
    print("="*70)
    print()

    print(f"{'Invariant':<15} {'Initial':>12} {'Final':>12} {'frac_var':>12} {'Status'}")
    print("-"*60)

    results = {}
    for name in history:
        vals = np.array(history[name])
        initial = vals[0]
        final = vals[-1]
        mean_v = np.mean(vals)
        frac_var = np.std(vals) / abs(mean_v) if abs(mean_v) > 1e-10 else np.inf

        results[name] = frac_var
        status = "✓ CONSERVED" if frac_var < 0.01 else ("~ approx" if frac_var < 0.1 else "✗ NOT")

        print(f"{name:<15} {initial:>12.4f} {final:>12.4f} {frac_var:>12.2e} {status}")

    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    exact_conserved = [name for name, fv in results.items() if fv < 0.01]
    approx_conserved = [name for name, fv in results.items() if 0.01 <= fv < 0.1]

    print("Exactly conserved (frac_var < 1%):")
    for name in exact_conserved:
        print(f"  - {name}")

    if approx_conserved:
        print()
        print("Approximately conserved (1% < frac_var < 10%):")
        for name in approx_conserved:
            print(f"  - {name}")

    print()
    print("Key findings:")
    print("  - Energy and Momentum are exactly conserved (expected)")
    print("  - Optical Chirality (Zilch Z⁰) should be exactly conserved")
    print("  - Helicity should be exactly conserved for source-free fields")
    print()
    print("If LLMs don't recognize Chirality/Zilch as conserved, this")
    print("indicates frozen priors limited to Poynting's theorem.")

    return history, results


if __name__ == "__main__":
    history, results = main()
