#!/usr/bin/env python3
"""
Test Q_f conservation under vortex stretching in 3D.

Vortex stretching is the key mechanism that could lead to blowup in 3D.
When a vortex tube is stretched, its cross-section shrinks while vorticity
increases (due to conservation of circulation).

We model this with a simple stretching flow and track Q_f.
"""

import numpy as np

def compute_Qf_tubes(tubes, f):
    """
    Compute Q_f for a collection of parallel vortex tubes.

    For parallel tubes aligned along z:
    Q_f = Σᵢⱼ ΓᵢΓⱼ ∫∫ f(|xᵢ - xⱼ|) dzᵢ dzⱼ × L²

    where xᵢ is the 2D position of tube i.

    For tubes of length L, this simplifies to:
    Q_f = L² × Σᵢⱼ ΓᵢΓⱼ f(|xᵢ - xⱼ|)
    """
    Q = 0.0
    n = len(tubes)

    for i in range(n):
        xi, yi, Gamma_i, L_i = tubes[i]
        for j in range(n):
            xj, yj, Gamma_j, L_j = tubes[j]

            if i == j:
                continue  # Skip self-interaction

            r = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if r > 1e-10:
                # Assuming both tubes have same length L
                L = min(L_i, L_j)
                Q += Gamma_i * Gamma_j * f(r) * L**2

    return Q


def stretching_evolution(tube, stretch_factor, dt):
    """
    Model stretching of a vortex tube.

    When stretched by factor s:
    - Length: L → s*L
    - Cross-section area: A → A/s (volume conservation)
    - Vorticity: ω → s*ω (from ∇·ω = 0)
    - Circulation: Γ = ω*A → Γ (conserved!)

    So stretching increases vorticity but preserves circulation.
    """
    x, y, Gamma, L = tube
    # Stretch factor applied over dt
    s = 1 + stretch_factor * dt
    new_L = L * s
    # Gamma is conserved
    return (x, y, Gamma, new_L)


def compute_enstrophy_density(tubes):
    """
    Approximate enstrophy per unit length for tubes.

    For a vortex tube with circulation Γ and core radius a:
    ω ~ Γ/(πa²) in the core
    Enstrophy ~ ∫ω² dA ~ Γ²/(πa²)

    When stretched by factor s, a → a/√s, so:
    Enstrophy ~ Γ² s / (πa₀²)

    We track this as a function of stretch.
    """
    total = 0
    for x, y, Gamma, L in tubes:
        # Stretch factor encoded in length change from initial L=1
        s = L  # If initial L=1
        total += Gamma**2 * s
    return total


# Test functions
def f_linear(r): return r
def f_sqrt(r): return np.sqrt(r + 1e-10)
def f_log(r): return -np.log(r + 0.01)
def f_exp(r): return np.exp(-r)
def f_inv(r): return 1.0 / (r + 0.1)


def main():
    print("="*70)
    print("3D Vortex Stretching and Q_f Conservation")
    print("="*70)
    print()
    print("When vortex tubes stretch:")
    print("  - Length increases: L → sL")
    print("  - Circulation conserved: Γ = const")
    print("  - Vorticity increases: ω → sω")
    print()
    print("Question: How do different Q_f respond to stretching?")
    print()

    # Initial configuration: two parallel vortex tubes
    # (x, y, Gamma, L)
    initial_tubes = [
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
    ]

    # Test functions
    test_functions = [
        ("r", f_linear),
        ("√r", f_sqrt),
        ("-ln(r)", f_log),
        ("e^(-r)", f_exp),
        ("1/r", f_inv),
    ]

    # Stretching simulation
    stretch_rate = 0.5  # stretch factor per unit time
    T = 3.0
    dt = 0.1
    n_steps = int(T / dt)

    tubes = list(initial_tubes)
    history = {name: [] for name, _ in test_functions}
    stretch_factors = []
    enstrophy_densities = []

    for step in range(n_steps + 1):
        t = step * dt
        s = tubes[0][3]  # Stretch factor = current length (initial L=1)
        stretch_factors.append(s)

        enst = compute_enstrophy_density(tubes)
        enstrophy_densities.append(enst)

        for name, f in test_functions:
            Qf = compute_Qf_tubes(tubes, f)
            history[name].append(Qf)

        # Evolve (stretch both tubes)
        if step < n_steps:
            tubes = [stretching_evolution(tube, stretch_rate, dt) for tube in tubes]

    # Results
    print("="*70)
    print(f"RESULTS: Q_f Under Stretching (stretch rate = {stretch_rate}/time)")
    print("="*70)
    print()

    print(f"{'t':<6} {'Stretch':>8} {'Enstrophy':>10}", end="")
    for name, _ in test_functions:
        print(f" {name:>10}", end="")
    print()
    print("-"*70)

    for step in range(0, n_steps + 1, 3):
        t = step * dt
        s = stretch_factors[step]
        enst = enstrophy_densities[step]
        print(f"{t:<6.1f} {s:>8.2f} {enst:>10.4f}", end="")
        for name, _ in test_functions:
            Qf = history[name][step]
            print(f" {Qf:>10.4f}", end="")
        print()

    # Analysis
    print()
    print("="*70)
    print("SCALING ANALYSIS")
    print("="*70)
    print()

    print("Expected scaling with stretch factor s:")
    print("  - Length L → sL")
    print("  - Q_f = L² × Σ ΓᵢΓⱼ f(rᵢⱼ)")
    print("  - Since rᵢⱼ is unchanged (tubes stay at same 2D positions),")
    print("    Q_f should scale as L² ~ s²")
    print()

    print("Observed scaling:")
    for name, _ in test_functions:
        initial = history[name][0]
        final = history[name][-1]
        s_final = stretch_factors[-1]
        observed_ratio = final / initial if initial != 0 else 0
        expected_ratio = s_final**2
        print(f"  {name:<10}: Q_f ratio = {observed_ratio:.4f}, expected s² = {expected_ratio:.4f}")

    print()
    print("="*70)
    print("KEY FINDINGS FOR 3D NAVIER-STOKES")
    print("="*70)
    print()
    print("1. Under pure stretching, Q_f grows as s² where s is stretch factor")
    print()
    print("2. But in real 3D flow, stretching is constrained by:")
    print("   - Incompressibility (volume conservation)")
    print("   - Energy conservation")
    print("   - Helicity conservation (for inviscid)")
    print()
    print("3. If Q_f were conserved in 3D (like in 2D), it would PREVENT")
    print("   unbounded stretching!")
    print()
    print("   Specifically: Q_f = const implies s ≤ C (bounded stretch)")
    print()
    print("4. HYPOTHESIS: In 3D Euler, Q_f may not be conserved, but")
    print("   dQ_f/dt may be bounded by a function of enstrophy.")
    print()
    print("   If dQ_f/dt ≤ C × ∫|ω|² dx, then Gronwall's lemma gives")
    print("   Q_f(t) ≤ Q_f(0) e^{Ct}, preventing finite-time blowup.")

    # Test the hypothesis with a more complex scenario
    print()
    print("="*70)
    print("TESTING: Stretching + Lateral Motion")
    print("="*70)
    print()

    # Now simulate tubes that both stretch AND move laterally
    # This is more realistic for 3D vortex dynamics

    tubes = list(initial_tubes)
    lateral_velocity = 0.2  # tubes move apart

    history2 = {name: [] for name, _ in test_functions}

    for step in range(n_steps + 1):
        for name, f in test_functions:
            Qf = compute_Qf_tubes(tubes, f)
            history2[name].append(Qf)

        if step < n_steps:
            new_tubes = []
            for i, (x, y, Gamma, L) in enumerate(tubes):
                # Stretch
                L_new = L * (1 + stretch_rate * dt)
                # Move laterally (tubes 0 and 1 move apart)
                direction = 1 if i == 1 else -1
                x_new = x + direction * lateral_velocity * dt
                new_tubes.append((x_new, y, Gamma, L_new))
            tubes = new_tubes

    print("With lateral motion (tubes moving apart):")
    print()
    print(f"{'f(r)':<10} {'Initial':>12} {'Final':>12} {'Ratio':>10}")
    print("-"*46)

    for name, _ in test_functions:
        initial = history2[name][0]
        final = history2[name][-1]
        ratio = final / initial if initial != 0 else 0
        print(f"{name:<10} {initial:>12.4f} {final:>12.4f} {ratio:>10.4f}")

    print()
    print("Observation: When tubes move apart (r increases), the effect")
    print("depends on f(r):")
    print("  - f(r) = r: Q_f increases (larger r × larger L²)")
    print("  - f(r) = 1/r: Q_f may stay bounded (smaller 1/r × larger L²)")
    print()
    print("This suggests Q_{1/r} (related to 3D energy) could provide")
    print("bounds even under stretching, if lateral motion accompanies")
    print("the stretching.")

    return history, stretch_factors


if __name__ == "__main__":
    history, stretch_factors = main()
