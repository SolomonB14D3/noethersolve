#!/usr/bin/env python3
"""
Numerical conservation checker for electromagnetic fields (Maxwell equations).

Tests conservation of standard and obscure EM invariants:
- Energy (Poynting) - standard, well-known
- Momentum - standard, well-known
- Optical chirality / Zilch Z⁰ (Lipkin 1964) - obscure
- Helicity - moderately known
- Super-energy (Chevreton tensor) - very obscure

The oracle is expected to pass on Energy/Momentum but may fail on Chirality/Helicity
due to frozen priors limited to Poynting's theorem.

Usage:
    # Check all built-in invariants:
    python em_checker.py --all

    # Check a specific invariant:
    python em_checker.py --check Energy --ic circular+

    # Check a custom expression:
    python em_checker.py --expr "s['Energy'] + s['Super_energy']"

References:
- Lipkin, D.M. (1964) "Existence of a new conservation law" J. Math. Phys. 5, 696
- Tang & Cohen (2010) "Optical chirality" Phys. Rev. Lett. 104, 163901
- Chevreton (1964) "Tenseur de superénergie" Nuovo Cimento 34, 901
"""

import argparse
import numpy as np
from numpy.fft import fftn, ifftn, fftfreq


# --------------------------------------------------------------------------
# Maxwell Spectral Solver
# --------------------------------------------------------------------------

class MaxwellSolver:
    """Spectral solver for source-free Maxwell equations."""

    def __init__(self, N=48, L=2*np.pi, c=1.0):
        self.N = N
        self.L = L
        self.c = c
        self.dx = L / N
        self.dV = self.dx**3

        # Grid
        x = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')

        # Wavenumbers
        k = fftfreq(N, d=self.dx) * 2 * np.pi
        self.KX, self.KY, self.KZ = np.meshgrid(k, k, k, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2 + self.KZ**2
        self.K2[0, 0, 0] = 1  # Avoid div by zero

        # Dealiasing (2/3 rule)
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

    def curl_physical(self, F_hat):
        """Compute curl and return in physical space."""
        curl_x, curl_y, curl_z = self.curl_spectral(*F_hat)
        return (np.real(ifftn(curl_x)),
                np.real(ifftn(curl_y)),
                np.real(ifftn(curl_z)))

    def rhs(self, E_hat, B_hat):
        """Time derivatives: dE/dt = c²∇×B, dB/dt = -∇×E"""
        curl_Bx, curl_By, curl_Bz = self.curl_spectral(*B_hat)
        curl_Ex, curl_Ey, curl_Ez = self.curl_spectral(*E_hat)

        dE_dt = (self.c**2 * curl_Bx * self.dealias,
                 self.c**2 * curl_By * self.dealias,
                 self.c**2 * curl_Bz * self.dealias)
        dB_dt = (-curl_Ex * self.dealias,
                 -curl_Ey * self.dealias,
                 -curl_Ez * self.dealias)

        return dE_dt, dB_dt

    def step_rk4(self, E_hat, B_hat, dt):
        """RK4 time step."""
        def add(t1, t2, scale=1.0):
            return tuple(a + scale * b for a, b in zip(t1, t2))

        dE1, dB1 = self.rhs(E_hat, B_hat)
        E2 = add(E_hat, dE1, 0.5*dt)
        B2 = add(B_hat, dB1, 0.5*dt)
        dE2, dB2 = self.rhs(E2, B2)
        E3 = add(E_hat, dE2, 0.5*dt)
        B3 = add(B_hat, dB2, 0.5*dt)
        dE3, dB3 = self.rhs(E3, B3)
        E4 = add(E_hat, dE3, dt)
        B4 = add(B_hat, dB3, dt)
        dE4, dB4 = self.rhs(E4, B4)

        E_new = tuple(e + (dt/6)*(d1 + 2*d2 + 2*d3 + d4)
                      for e, d1, d2, d3, d4 in zip(E_hat, dE1, dE2, dE3, dE4))
        B_new = tuple(b + (dt/6)*(d1 + 2*d2 + 2*d3 + d4)
                      for b, d1, d2, d3, d4 in zip(B_hat, dB1, dB2, dB3, dB4))
        return E_new, B_new

    def to_physical(self, F_hat):
        return tuple(np.real(ifftn(f)) for f in F_hat)

    def to_spectral(self, F):
        return tuple(fftn(f) for f in F)


# --------------------------------------------------------------------------
# Initial Conditions
# --------------------------------------------------------------------------

def ic_circular_plus(solver, sigma=0.6, k_mag=4):
    """Right-handed circularly polarized wave packet (positive chirality)."""
    X, Y, Z = solver.X, solver.Y, solver.Z
    L = solver.L

    r2 = (X - L/2)**2 + (Y - L/2)**2 + (Z - L/2)**2
    envelope = np.exp(-r2 / (2 * sigma**2))
    phase = k_mag * Z

    Ex = envelope * np.cos(phase)
    Ey = envelope * np.sin(phase)
    Ez = np.zeros_like(X)
    Bx = -envelope * np.sin(phase)
    By = envelope * np.cos(phase)
    Bz = np.zeros_like(X)

    return (Ex, Ey, Ez), (Bx, By, Bz)


def ic_circular_minus(solver, sigma=0.6, k_mag=4):
    """Left-handed circularly polarized wave packet (negative chirality)."""
    X, Y, Z = solver.X, solver.Y, solver.Z
    L = solver.L

    r2 = (X - L/2)**2 + (Y - L/2)**2 + (Z - L/2)**2
    envelope = np.exp(-r2 / (2 * sigma**2))
    phase = k_mag * Z

    Ex = envelope * np.cos(phase)
    Ey = -envelope * np.sin(phase)
    Ez = np.zeros_like(X)
    Bx = envelope * np.sin(phase)
    By = envelope * np.cos(phase)
    Bz = np.zeros_like(X)

    return (Ex, Ey, Ez), (Bx, By, Bz)


def ic_linear_x(solver, sigma=0.6, k_mag=4):
    """Linearly polarized wave packet (x-polarization)."""
    X, Y, Z = solver.X, solver.Y, solver.Z
    L = solver.L

    r2 = (X - L/2)**2 + (Y - L/2)**2 + (Z - L/2)**2
    envelope = np.exp(-r2 / (2 * sigma**2))
    phase = k_mag * Z

    Ex = envelope * np.cos(phase)
    Ey = np.zeros_like(X)
    Ez = np.zeros_like(X)
    Bx = np.zeros_like(X)
    By = envelope * np.cos(phase)
    Bz = np.zeros_like(X)

    return (Ex, Ey, Ez), (Bx, By, Bz)


def ic_standing_wave(solver, sigma=0.8, k_mag=3):
    """Standing wave (superposition of counter-propagating waves)."""
    X, Y, Z = solver.X, solver.Y, solver.Z
    L = solver.L

    r2 = (X - L/2)**2 + (Y - L/2)**2 + (Z - L/2)**2
    envelope = np.exp(-r2 / (2 * sigma**2))

    # Standing wave: cos(kz) in space
    Ex = envelope * np.cos(k_mag * Z)
    Ey = np.zeros_like(X)
    Ez = np.zeros_like(X)
    Bx = np.zeros_like(X)
    By = envelope * np.sin(k_mag * Z)  # B is out of phase
    Bz = np.zeros_like(X)

    return (Ex, Ey, Ez), (Bx, By, Bz)


IC_MAP = {
    "circular+":    ic_circular_plus,
    "circular-":    ic_circular_minus,
    "linear_x":     ic_linear_x,
    "standing":     ic_standing_wave,
}


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------

def integrate_em(solver, E0, B0, t_end=4.0, dt=0.02, n_points=100):
    """Integrate Maxwell equations. Returns (t, E_history, B_history)."""
    E_hat = solver.to_spectral(E0)
    B_hat = solver.to_spectral(B0)

    n_steps = int(t_end / dt)
    save_every = max(1, n_steps // n_points)

    t_history = []
    E_history = []
    B_history = []

    for step in range(n_steps + 1):
        if step % save_every == 0:
            solver.to_physical(E_hat)
            solver.to_physical(B_hat)
            t_history.append(step * dt)
            E_history.append(E_hat)  # Keep spectral for curl computations
            B_history.append(B_hat)

        if step < n_steps:
            E_hat, B_hat = solver.step_rk4(E_hat, B_hat, dt)

    return np.array(t_history), E_history, B_history


# --------------------------------------------------------------------------
# Parse State - Extract named invariants
# --------------------------------------------------------------------------

def parse_state(solver, t, E_history, B_history):
    """Extract named quantities from EM field trajectory."""
    n_t = len(t)
    dV = solver.dV

    # Initialize arrays
    Energy = np.zeros(n_t)
    Momentum_x = np.zeros(n_t)
    Momentum_y = np.zeros(n_t)
    Momentum_z = np.zeros(n_t)
    Momentum = np.zeros(n_t)
    Chirality = np.zeros(n_t)
    Helicity = np.zeros(n_t)
    Super_energy = np.zeros(n_t)
    E_squared = np.zeros(n_t)
    B_squared = np.zeros(n_t)

    for i, (E_hat, B_hat) in enumerate(zip(E_history, B_history)):
        E = solver.to_physical(E_hat)
        B = solver.to_physical(B_hat)
        Ex, Ey, Ez = E
        Bx, By, Bz = B

        # Energy: U = (1/2) ∫ (E² + B²) d³x
        E_sq = np.sum(Ex**2 + Ey**2 + Ez**2) * dV
        B_sq = np.sum(Bx**2 + By**2 + Bz**2) * dV
        Energy[i] = 0.5 * (E_sq + B_sq)
        E_squared[i] = E_sq
        B_squared[i] = B_sq

        # Momentum: P = ∫ (E × B) d³x
        Px = np.sum(Ey * Bz - Ez * By) * dV
        Py = np.sum(Ez * Bx - Ex * Bz) * dV
        Pz = np.sum(Ex * By - Ey * Bx) * dV
        Momentum_x[i] = Px
        Momentum_y[i] = Py
        Momentum_z[i] = Pz
        Momentum[i] = np.sqrt(Px**2 + Py**2 + Pz**2)

        # Optical Chirality: C = (1/2) ∫ [E·(∇×E) + B·(∇×B)] d³x
        curl_Ex, curl_Ey, curl_Ez = solver.curl_physical(E_hat)
        curl_Bx, curl_By, curl_Bz = solver.curl_physical(B_hat)
        E_dot_curlE = Ex*curl_Ex + Ey*curl_Ey + Ez*curl_Ez
        B_dot_curlB = Bx*curl_Bx + By*curl_By + Bz*curl_Bz
        Chirality[i] = 0.5 * np.sum(E_dot_curlE + B_dot_curlB) * dV

        # Helicity: H = ∫ A·B d³x  where B = ∇×A
        # Compute A from B: A_hat = -i (k × B_hat) / k²
        Bx_hat, By_hat, Bz_hat = B_hat
        kxB_x = solver.KY * Bz_hat - solver.KZ * By_hat
        kxB_y = solver.KZ * Bx_hat - solver.KX * Bz_hat
        kxB_z = solver.KX * By_hat - solver.KY * Bx_hat
        Ax = np.real(ifftn(-1j * kxB_x / solver.K2))
        Ay = np.real(ifftn(-1j * kxB_y / solver.K2))
        Az = np.real(ifftn(-1j * kxB_z / solver.K2))
        Helicity[i] = np.sum(Ax*Bx + Ay*By + Az*Bz) * dV

        # Super-energy: S = ∫ [(∇×E)² + (∇×B)²] d³x
        curlE_sq = curl_Ex**2 + curl_Ey**2 + curl_Ez**2
        curlB_sq = curl_Bx**2 + curl_By**2 + curl_Bz**2
        Super_energy[i] = np.sum(curlE_sq + curlB_sq) * dV

    return dict(
        t=t,
        Energy=Energy,
        Momentum=Momentum,
        Momentum_x=Momentum_x,
        Momentum_y=Momentum_y,
        Momentum_z=Momentum_z,
        Chirality=Chirality,
        Helicity=Helicity,
        Super_energy=Super_energy,
        E_squared=E_squared,
        B_squared=B_squared,
    )


def frac_var(arr):
    """Fractional variation σ/|mean|."""
    mean = np.mean(arr)
    if abs(mean) < 1e-15:
        return float(np.std(arr))
    return float(np.std(arr) / abs(mean))


# --------------------------------------------------------------------------
# Built-in Candidates
# --------------------------------------------------------------------------

BUILTIN_CANDIDATES = {
    "Energy": {
        "fn": lambda s: s["Energy"],
        "desc": "Energy U = (1/2) ∫ (E² + B²) d³x (Poynting)",
        "expected": "conserved (exact)",
    },
    "Momentum": {
        "fn": lambda s: s["Momentum"],
        "desc": "Momentum |P| = |∫ (E × B) d³x|",
        "expected": "conserved (exact)",
    },
    "Chirality": {
        "fn": lambda s: s["Chirality"],
        "desc": "Optical chirality C = ½ ∫ [E·(∇×E) + B·(∇×B)] (Lipkin 1964)",
        "expected": "conserved (exact, obscure)",
    },
    "Helicity": {
        "fn": lambda s: s["Helicity"],
        "desc": "Helicity H = ∫ A·B d³x",
        "expected": "conserved (exact)",
    },
    "Super_energy": {
        "fn": lambda s: s["Super_energy"],
        "desc": "Super-energy S = ∫ [(∇×E)² + (∇×B)²] (Chevreton)",
        "expected": "conserved (exact, very obscure)",
    },
    "E_squared": {
        "fn": lambda s: s["E_squared"],
        "desc": "∫ E² d³x alone (without B²)",
        "expected": "NOT conserved (only sum is)",
    },
    "B_squared": {
        "fn": lambda s: s["B_squared"],
        "desc": "∫ B² d³x alone (without E²)",
        "expected": "NOT conserved (only sum is)",
    },
    "Pz": {
        "fn": lambda s: s["Momentum_z"],
        "desc": "z-momentum component Pz",
        "expected": "conserved (exact)",
    },
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def run_all(solver, ic_name, E0, B0, t_end=4.0, dt=0.02, threshold=5e-3):
    """Run all built-in candidates and print results."""
    t, E_hist, B_hist = integrate_em(solver, E0, B0, t_end=t_end, dt=dt)
    s = parse_state(solver, t, E_hist, B_hist)

    print(f"\nIC: {ic_name}  N={solver.N}  L={solver.L:.2f}  t_end={t_end}")
    print(f"{'Candidate':20s}  {'frac_var':>12s}  {'verdict':>8s}")
    print("-" * 50)

    for name, cand in BUILTIN_CANDIDATES.items():
        try:
            vals = cand["fn"](s)
            fv = frac_var(vals)
            v = "PASS" if fv < threshold else "fail"
            print(f"{name:20s}  {fv:12.3e}  {v:>8s}  # {cand['expected']}")
        except KeyError as e:
            print(f"{name:20s}  {'N/A':>12s}  {'skip':>8s}  # missing: {e}")

    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Electromagnetic conservation checker")
    parser.add_argument("--all", action="store_true", help="Run all ICs and candidates")
    parser.add_argument("--ic", default="circular+", choices=list(IC_MAP.keys()))
    parser.add_argument("--check", default=None, help="Check a built-in candidate by name")
    parser.add_argument("--expr", default=None, help="Python expression in s['...'] namespace")
    parser.add_argument("--t-end", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--N", type=int, default=48)
    parser.add_argument("--threshold", type=float, default=5e-3)
    args = parser.parse_args()

    solver = MaxwellSolver(N=args.N)

    if args.all:
        for ic_name, ic_fn in IC_MAP.items():
            E0, B0 = ic_fn(solver)
            run_all(solver, ic_name, E0, B0, t_end=args.t_end, dt=args.dt, threshold=args.threshold)
    else:
        ic_fn = IC_MAP[args.ic]
        E0, B0 = ic_fn(solver)

        if args.check:
            t, E_hist, B_hist = integrate_em(solver, E0, B0, t_end=args.t_end, dt=args.dt)
            s = parse_state(solver, t, E_hist, B_hist)
            cand = BUILTIN_CANDIDATES[args.check]
            vals = cand["fn"](s)
            fv = frac_var(vals)
            v = "PASS" if fv < args.threshold else "FAIL"
            print(f"{args.ic} / {args.check}: frac_var={fv:.3e}  {v}")

        elif args.expr:
            t, E_hist, B_hist = integrate_em(solver, E0, B0, t_end=args.t_end, dt=args.dt)
            s = parse_state(solver, t, E_hist, B_hist)
            vals = eval(args.expr, {"s": s, "np": np})
            fv = frac_var(np.asarray(vals))
            v = "PASS" if fv < args.threshold else "FAIL"
            print(f"{args.ic} / expr: frac_var={fv:.3e}  {v}")
            print(f"  expr: {args.expr}")

        else:
            run_all(solver, args.ic, E0, B0, t_end=args.t_end, dt=args.dt, threshold=args.threshold)
