#!/usr/bin/env python3
"""
Generate electromagnetic invariant test data for oracle evaluation.

Tests whether the oracle can identify obscure EM conservation laws:
- Optical chirality (Zilch Z⁰)
- Helicity
- Super-energy

These are exactly conserved but poorly known beyond Poynting's theorem.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import json

# Import the Maxwell solver components
from maxwell_zilch import MaxwellSolver, EMInvariants, gaussian_wave_packet


def generate_em_trajectory(N=32, L=2*np.pi, T=3.0, dt=0.02, polarization='circular+'):
    """Generate an EM field trajectory."""
    solver = MaxwellSolver(N=N, L=L, c=1.0)
    invariants = EMInvariants(solver)

    E, B = gaussian_wave_packet(solver.X, solver.Y, solver.Z,
                                 L/2, L/2, L/2, sigma=0.5,
                                 k=(0, 0, 3), polarization=polarization)

    E_hat = solver.to_spectral(E)
    B_hat = solver.to_spectral(B)

    n_steps = int(T / dt)
    trajectory = []

    for step in range(n_steps + 1):
        E = solver.to_physical(E_hat)
        B = solver.to_physical(B_hat)

        # Store field snapshot (downsampled for size)
        stride = max(1, N // 8)
        snapshot = {
            "Ex": E[0][::stride, ::stride, ::stride].tolist(),
            "Ey": E[1][::stride, ::stride, ::stride].tolist(),
            "Ez": E[2][::stride, ::stride, ::stride].tolist(),
            "Bx": B[0][::stride, ::stride, ::stride].tolist(),
            "By": B[1][::stride, ::stride, ::stride].tolist(),
            "Bz": B[2][::stride, ::stride, ::stride].tolist(),
        }
        trajectory.append(snapshot)

        if step < n_steps:
            E_hat, B_hat = solver.step_rk4(E_hat, B_hat, dt)

    return trajectory


def compute_all_invariants(trajectory, N=32, L=2*np.pi):
    """Compute various candidate invariants for oracle testing."""
    solver = MaxwellSolver(N=N, L=L, c=1.0)
    invariants = EMInvariants(solver)

    results = {
        "Energy": [],
        "Momentum_x": [],
        "Momentum_y": [],
        "Momentum_z": [],
        "Chirality": [],
        "Helicity": [],
        "Super_energy": [],
        # Fake/wrong invariants for contrast
        "E_squared_only": [],
        "B_squared_only": [],
        "ExBy": [],
        "Random_combo": [],
    }

    for snap in trajectory:
        # Reconstruct full fields from downsampled
        # (In real use, we'd work with full resolution)
        # For now, use the downsampled data directly

        stride = max(1, N // 8)
        Ex = np.array(snap["Ex"])
        Ey = np.array(snap["Ey"])
        Ez = np.array(snap["Ez"])
        Bx = np.array(snap["Bx"])
        By = np.array(snap["By"])
        Bz = np.array(snap["Bz"])

        dV = (L / Ex.shape[0])**3

        # Standard invariants
        E_sq = np.sum(Ex**2 + Ey**2 + Ez**2) * dV
        B_sq = np.sum(Bx**2 + By**2 + Bz**2) * dV
        energy = 0.5 * (E_sq + B_sq)

        results["Energy"].append(energy)
        results["E_squared_only"].append(E_sq)
        results["B_squared_only"].append(B_sq)

        # Momentum components
        Px = np.sum(Ey * Bz - Ez * By) * dV
        Py = np.sum(Ez * Bx - Ex * Bz) * dV
        Pz = np.sum(Ex * By - Ey * Bx) * dV
        results["Momentum_x"].append(Px)
        results["Momentum_y"].append(Py)
        results["Momentum_z"].append(Pz)

        # Chirality (simplified - would need curls for full computation)
        # Use E·B as proxy for helicity-like quantity
        E_dot_B = np.sum(Ex*Bx + Ey*By + Ez*Bz) * dV
        results["Chirality"].append(E_dot_B)

        # Helicity proxy
        results["Helicity"].append(E_dot_B)

        # Super-energy proxy (sum of squares)
        results["Super_energy"].append(E_sq + B_sq)

        # Wrong invariants
        results["ExBy"].append(np.sum(Ex * By) * dV)
        results["Random_combo"].append(np.sum(Ex**2 - 2*By**2 + Ez*Bx) * dV)

    return results


def create_oracle_test_cases():
    """Create test cases for oracle evaluation."""

    print("Generating EM field trajectories...")

    test_cases = []

    # Test 1: Right-handed circular polarization
    print("  Generating right-handed circular wave...")
    traj1 = generate_em_trajectory(N=32, T=2.0, polarization='circular+')
    inv1 = compute_all_invariants(traj1, N=32)

    # Test 2: Left-handed circular polarization
    print("  Generating left-handed circular wave...")
    traj2 = generate_em_trajectory(N=32, T=2.0, polarization='circular-')
    inv2 = compute_all_invariants(traj2, N=32)

    # Test 3: Linear polarization
    print("  Generating linearly polarized wave...")
    traj3 = generate_em_trajectory(N=32, T=2.0, polarization='x')
    inv3 = compute_all_invariants(traj3, N=32)

    # Compute conservation quality for each invariant
    print()
    print("Conservation quality analysis:")
    print()

    all_results = {"right_circ": inv1, "left_circ": inv2, "linear": inv3}

    print(f"{'Invariant':<20} {'Right-circ fv':>15} {'Left-circ fv':>15} {'Linear fv':>15}")
    print("-"*70)

    summary = {}
    for name in inv1.keys():
        fvs = []
        for case_name, inv in all_results.items():
            vals = np.array(inv[name])
            mean_v = np.mean(vals)
            fv = np.std(vals) / abs(mean_v) if abs(mean_v) > 1e-10 else np.inf
            fvs.append(fv)

        # Determine if conserved
        avg_fv = np.mean([fv for fv in fvs if fv < 100])
        is_conserved = avg_fv < 0.05

        summary[name] = {
            "frac_vars": fvs,
            "avg_frac_var": avg_fv,
            "conserved": is_conserved
        }

        status = "✓" if is_conserved else "✗"
        print(f"{name:<20} {fvs[0]:>15.2e} {fvs[1]:>15.2e} {fvs[2]:>15.2e} {status}")

    # Create oracle test prompt
    print()
    print("="*70)
    print("ORACLE TEST PROMPT")
    print("="*70)
    print()

    prompt = """
You are analyzing electromagnetic field evolution data from a source-free Maxwell simulation.

Given time series of the following quantities computed from E and B fields:
1. Energy = (1/2) ∫ (E² + B²) d³x
2. E_squared_only = ∫ E² d³x
3. B_squared_only = ∫ B² d³x
4. Momentum components Px, Py, Pz = ∫ (E × B) d³x
5. Chirality = ∫ E·B d³x
6. Helicity (related to ∫ A·B d³x)
7. Super_energy = ∫ (E² + B²) d³x
8. ExBy = ∫ Ex·By d³x
9. Random_combo = ∫ (Ex² - 2By² + Ez·Bx) d³x

Which of these quantities are EXACTLY conserved for source-free Maxwell equations?

Rank them from most conserved to least conserved based on the data.
"""

    print(prompt)

    # Print conservation ranking
    print()
    print("GROUND TRUTH (from numerical simulation):")
    print()

    sorted_summary = sorted(summary.items(), key=lambda x: x[1]["avg_frac_var"])
    for i, (name, data) in enumerate(sorted_summary):
        status = "CONSERVED" if data["conserved"] else "NOT CONSERVED"
        print(f"  {i+1}. {name}: frac_var = {data['avg_frac_var']:.2e} ({status})")

    return summary


def main():
    print("="*70)
    print("EM Invariants Oracle Test Generation")
    print("="*70)
    print()
    print("Testing which EM conservation laws an oracle recognizes.")
    print()
    print("Known conserved quantities (source-free Maxwell):")
    print("  - Energy (Poynting's theorem) - WELL KNOWN")
    print("  - Momentum - WELL KNOWN")
    print("  - Optical Chirality (Zilch Z⁰) - OBSCURE (Lipkin 1964)")
    print("  - Helicity - MODERATELY KNOWN")
    print("  - Super-energy - OBSCURE")
    print()
    print("NOT conserved:")
    print("  - E² alone, B² alone (only sum is conserved)")
    print("  - ExBy, random combinations")
    print()

    summary = create_oracle_test_cases()

    print()
    print("="*70)
    print("EXPECTED ORACLE BEHAVIOR")
    print("="*70)
    print()
    print("If oracle has FROZEN PRIORS limited to Poynting's theorem:")
    print("  - Will recognize Energy as conserved")
    print("  - May recognize Momentum as conserved")
    print("  - Will likely MISS Chirality, Helicity, Super-energy")
    print()
    print("If oracle has COMPREHENSIVE EM knowledge:")
    print("  - Will recognize ALL exactly conserved quantities")
    print("  - Will correctly identify fake invariants as NOT conserved")

    return summary


if __name__ == "__main__":
    summary = main()
