"""
corruption_benchmark.py — Prove conservation monitors catch errors faster.

Four experiments:
1. Tolerance degradation: reduce integrator rtol, measure when each monitor detects drift
2. Single-step corruption: inject noise at one timestep, measure detection latency
3. Wrong physics: drop a term from equations of motion, measure which monitors catch it
4. Chemical: violate detailed balance with non-physical rate constants

For each experiment, compare NoetherSolve-discovered invariants (Q_f, R_f, K)
against standard monitors (E, Lz, enstrophy) to show detection sensitivity gain.
"""

import numpy as np
import sys
import os
import json
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from noether_monitor import VortexMonitor, ChemicalMonitor, frac_var


# ─── Vortex ODE (correct) ────────────────────────────────────────────────────

def vortex_rhs(t, state, G):
    N = len(G)
    pos = state.reshape(N, 2)
    dpos = np.zeros_like(pos)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            r2 = dx**2 + dy**2 + 1e-20
            dpos[i, 0] += -G[j] * dy / (2 * np.pi * r2)
            dpos[i, 1] += G[j] * dx / (2 * np.pi * r2)
    return dpos.ravel()


def vortex_rhs_wrong(t, state, G):
    """Wrong physics: drop the 1/(2π) factor (common unit error)."""
    N = len(G)
    pos = state.reshape(N, 2)
    dpos = np.zeros_like(pos)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            r2 = dx**2 + dy**2 + 1e-20
            # Missing 2π factor — dynamics too fast by 2π
            dpos[i, 0] += -G[j] * dy / r2
            dpos[i, 1] += G[j] * dx / r2
    return dpos.ravel()


def vortex_rhs_missing_term(t, state, G):
    """Wrong physics: drop the weakest vortex's contribution."""
    N = len(G)
    pos = state.reshape(N, 2)
    dpos = np.zeros_like(pos)
    skip_j = int(np.argmin(np.abs(G)))  # skip weakest
    for i in range(N):
        for j in range(N):
            if i == j or j == skip_j:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            r2 = dx**2 + dy**2 + 1e-20
            dpos[i, 0] += -G[j] * dy / (2 * np.pi * r2)
            dpos[i, 1] += G[j] * dx / (2 * np.pi * r2)
    return dpos.ravel()


# ─── Chemical ODE ─────────────────────────────────────────────────────────────

def chemical_rhs_correct(t, c, S, k_rates, reactant_matrix):
    """Mass-action kinetics: dc/dt = S @ v(c)."""
    c = np.maximum(c, 0)  # enforce non-negativity
    n_rxn = S.shape[1]
    v = np.zeros(n_rxn)
    for j in range(n_rxn):
        rate = k_rates[j]
        for i in range(len(c)):
            if reactant_matrix[i, j] > 0:
                rate *= c[i] ** reactant_matrix[i, j]
        v[j] = rate
    return S @ v


def chemical_rhs_wrong(t, c, S, k_rates, reactant_matrix):
    """Violate detailed balance: forward rates don't match reverse."""
    c = np.maximum(c, 0)
    n_rxn = S.shape[1]
    v = np.zeros(n_rxn)
    for j in range(n_rxn):
        # Perturb odd-indexed reaction rates (breaks Wegscheider)
        rate = k_rates[j] * (1.5 if j % 2 == 1 else 1.0)
        for i in range(len(c)):
            if reactant_matrix[i, j] > 0:
                rate *= c[i] ** reactant_matrix[i, j]
        v[j] = rate
    return S @ v


# ─── Initial conditions ──────────────────────────────────────────────────────

def get_vortex_ic():
    """3-vortex system with unequal circulations."""
    G = np.array([1.0, -0.5, 0.3])
    pos = np.array([
        [1.0, 0.0],
        [-0.5, 0.8],
        [-0.3, -0.6],
    ])
    return G, pos


def get_chemical_ic():
    """A ↔ B ↔ C reaction network."""
    species = ["A", "B", "C"]
    # Reactions: A→B (k1), B→A (k2), B→C (k3), C→B (k4)
    S = np.array([
        [-1, 1, 0, 0],   # A
        [1, -1, -1, 1],   # B
        [0, 0, 1, -1],    # C
    ], dtype=float)
    k_rates = np.array([0.5, 0.3, 0.4, 0.2])
    reactant_matrix = np.array([
        [1, 0, 0, 0],  # A consumed in rxn 0
        [0, 1, 1, 0],  # B consumed in rxn 1, 2
        [0, 0, 0, 1],  # C consumed in rxn 3
    ], dtype=float)
    # Reversible pairs: (A→B, B→A) and (B→C, C→B)
    reverse_pairs = [(0, 1), (2, 3)]
    c0 = np.array([1.0, 0.5, 0.2])
    return species, S, k_rates, reactant_matrix, reverse_pairs, c0


# ─── Experiment 1: Tolerance degradation ──────────────────────────────────────

def exp1_tolerance_sweep():
    """Reduce integrator tolerance, measure when each monitor first alerts."""
    print("=" * 72)
    print("EXPERIMENT 1: Tolerance degradation sweep")
    print("=" * 72)

    G, pos0 = get_vortex_ic()
    t_end = 100.0
    n_points = 2000

    rtol_values = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]

    results = []
    for rtol in rtol_values:
        sol = solve_ivp(
            vortex_rhs, (0, t_end), pos0.ravel(), args=(G,),
            method="RK45", t_eval=np.linspace(0, t_end, n_points),
            rtol=rtol, atol=rtol * 1e-2,
        )
        if sol.status != 0:
            print(f"  rtol={rtol:.0e}: solver failed")
            continue

        monitor = VortexMonitor(G, threshold=1e-6)
        monitor.set_initial(pos0)

        dt = t_end / n_points
        first_alert = {}
        for i in range(1, sol.y.shape[1]):
            state = sol.y[:, i].reshape(-1, 2)
            report = monitor.check(state, dt=dt)
            for name in report.alerts:
                if name not in first_alert:
                    first_alert[name] = i

        # Get final frac_vars
        summary = monitor.summary()
        row = {"rtol": rtol, "first_alert": first_alert, "final_frac_vars": {}}
        for k, v in summary.items():
            row["final_frac_vars"][k] = v["frac_var"]

        results.append(row)

        # Print compact summary
        detected = sorted(first_alert.items(), key=lambda x: x[1])
        first_str = detected[0] if detected else ("none", "-")
        n_alerted = len(first_alert)
        print(f"  rtol={rtol:.0e}: {n_alerted} alerts. "
              f"First: {first_str[0]} at step {first_str[1] if detected else '-'}")

    # Summary table
    print(f"\n{'Monitor':<20} ", end="")
    for r in results:
        print(f" {r['rtol']:.0e}", end="")
    print()
    print("-" * (20 + 8 * len(results)))

    # Collect all monitor names
    all_names = set()
    for r in results:
        all_names.update(r["final_frac_vars"].keys())

    for name in sorted(all_names):
        print(f"{name:<20} ", end="")
        for r in results:
            fv = r["final_frac_vars"].get(name, float("nan"))
            if fv < 1e-10:
                print(f" {'ok':>7}", end="")
            else:
                print(f" {fv:.1e}", end="")
        print()

    return results


# ─── Experiment 2: Single-step corruption ─────────────────────────────────────

def exp2_single_step_corruption():
    """Inject noise at step 500, measure detection latency for each monitor."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Single-step corruption detection latency")
    print("=" * 72)

    G, pos0 = get_vortex_ic()
    t_end = 100.0
    n_points = 2000
    corrupt_step = 500

    noise_levels = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]

    results = []
    for noise in noise_levels:
        # Run clean simulation
        sol = solve_ivp(
            vortex_rhs, (0, t_end), pos0.ravel(), args=(G,),
            method="RK45", t_eval=np.linspace(0, t_end, n_points),
            rtol=1e-10, atol=1e-12,
        )

        # Inject noise at corrupt_step
        trajectory = sol.y.T.copy()  # (n_points, 6)
        rng = np.random.RandomState(42)
        trajectory[corrupt_step] += rng.randn(trajectory.shape[1]) * noise

        # Re-integrate from corrupted state
        t_corrupt = sol.t[corrupt_step]
        sol2 = solve_ivp(
            vortex_rhs, (t_corrupt, t_end), trajectory[corrupt_step], args=(G,),
            method="RK45",
            t_eval=np.linspace(t_corrupt, t_end, n_points - corrupt_step),
            rtol=1e-10, atol=1e-12,
        )
        if sol2.status != 0:
            print(f"  noise={noise:.0e}: post-corruption solver failed")
            continue

        # Stitch together: clean[:corrupt_step] + corrupted[corrupt_step:]
        full_traj = np.vstack([trajectory[:corrupt_step], sol2.y.T])

        monitor = VortexMonitor(G, threshold=1e-6)
        monitor.set_initial(pos0)

        dt = t_end / n_points
        first_alert = {}
        for i in range(1, len(full_traj)):
            state = full_traj[i].reshape(-1, 2)
            report = monitor.check(state, dt=dt)
            for name in report.alerts:
                if name not in first_alert:
                    first_alert[name] = i

        # Detection latency = steps after corruption
        latencies = {}
        for name, step in first_alert.items():
            if step >= corrupt_step:
                latencies[name] = step - corrupt_step
            else:
                latencies[name] = 0  # was already alerting before corruption

        results.append({
            "noise": noise,
            "latencies": latencies,
            "first_alert": first_alert,
        })

        # First detector
        post_corrupt = {k: v for k, v in latencies.items() if v >= 0}
        if post_corrupt:
            fastest = min(post_corrupt, key=post_corrupt.get)
            print(f"  noise={noise:.0e}: fastest={fastest} at +{post_corrupt[fastest]} steps. "
                  f"Total detectors: {len(post_corrupt)}")
        else:
            print(f"  noise={noise:.0e}: no detection")

    # Summary: detection latency by monitor
    print(f"\n{'Monitor':<20} ", end="")
    for r in results:
        print(f" {r['noise']:.0e}", end="")
    print("  (steps after corruption)")
    print("-" * (20 + 8 * len(results)))

    all_names = set()
    for r in results:
        all_names.update(r["latencies"].keys())
    for name in sorted(all_names):
        print(f"{name:<20} ", end="")
        for r in results:
            lat = r["latencies"].get(name, None)
            if lat is not None:
                print(f" {lat:>7d}", end="")
            else:
                print(f" {'miss':>7}", end="")
        print()

    return results


# ─── Experiment 3: Wrong physics ──────────────────────────────────────────────

def exp3_wrong_physics():
    """Run with wrong equations of motion, measure which monitors detect it."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Wrong physics detection")
    print("=" * 72)

    G, pos0 = get_vortex_ic()
    t_end = 50.0
    n_points = 2000
    dt = t_end / n_points

    scenarios = [
        ("Correct physics", vortex_rhs),
        ("Missing 2π factor", vortex_rhs_wrong),
        ("Dropped weakest vortex", vortex_rhs_missing_term),
    ]

    all_results = {}
    for label, rhs_fn in scenarios:
        sol = solve_ivp(
            rhs_fn, (0, t_end), pos0.ravel(), args=(G,),
            method="RK45", t_eval=np.linspace(0, t_end, n_points),
            rtol=1e-10, atol=1e-12,
        )
        if sol.status != 0:
            print(f"  {label}: solver failed")
            continue

        monitor = VortexMonitor(G, threshold=1e-4)
        monitor.set_initial(pos0)

        for i in range(1, sol.y.shape[1]):
            state = sol.y[:, i].reshape(-1, 2)
            monitor.check(state, dt=dt)

        summary = monitor.summary()
        all_results[label] = summary

        print(f"\n  {label}:")
        # Sort by frac_var descending
        sorted_q = sorted(summary.items(), key=lambda x: -x[1]["frac_var"])
        for name, data in sorted_q[:10]:
            fv = data["frac_var"]
            flag = " <<<" if fv > 1e-4 else ""
            print(f"    {name:<20s}  frac_var={fv:.2e}{flag}")

    # Comparison table: which monitors distinguish correct from wrong?
    print(f"\n{'Monitor':<20} {'Correct':>12} {'No 2π':>12} {'Drop vortex':>12}  {'Sensitivity':>12}")
    print("-" * 72)

    correct = all_results.get("Correct physics", {})
    wrong1 = all_results.get("Missing 2π factor", {})
    wrong2 = all_results.get("Dropped weakest vortex", {})

    all_names = set(correct.keys()) | set(wrong1.keys()) | set(wrong2.keys())
    rows = []
    for name in sorted(all_names):
        fv_c = correct.get(name, {}).get("frac_var", float("nan"))
        fv_w1 = wrong1.get(name, {}).get("frac_var", float("nan"))
        fv_w2 = wrong2.get(name, {}).get("frac_var", float("nan"))
        # Sensitivity = ratio of wrong to correct frac_var
        sens1 = fv_w1 / (fv_c + 1e-20) if not np.isnan(fv_w1) else 0
        sens2 = fv_w2 / (fv_c + 1e-20) if not np.isnan(fv_w2) else 0
        max_sens = max(sens1, sens2)
        rows.append((name, fv_c, fv_w1, fv_w2, max_sens))

    rows.sort(key=lambda x: -x[4])
    for name, fv_c, fv_w1, fv_w2, sens in rows:
        print(f"{name:<20} {fv_c:>12.2e} {fv_w1:>12.2e} {fv_w2:>12.2e}  {sens:>12.1f}x")

    return all_results


# ─── Experiment 4: Chemical detailed balance violation ────────────────────────

def exp4_chemical_violation():
    """Run chemical network with violated detailed balance."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: Chemical detailed balance violation")
    print("=" * 72)

    species, S, k_rates, reactant_matrix, reverse_pairs, c0 = get_chemical_ic()
    t_end = 50.0
    n_points = 2000

    scenarios = [
        ("Correct rates", k_rates),
        ("Violated balance (1.5x odd)", k_rates * np.array([1.0, 1.5, 1.0, 1.5])),
        ("Violated balance (2x odd)", k_rates * np.array([1.0, 2.0, 1.0, 2.0])),
        ("Violated balance (5x odd)", k_rates * np.array([1.0, 5.0, 1.0, 5.0])),
    ]

    for label, rates in scenarios:
        sol = solve_ivp(
            chemical_rhs_correct, (0, t_end), c0, args=(S, rates, reactant_matrix),
            method="RK45", t_eval=np.linspace(0, t_end, n_points),
            rtol=1e-10, atol=1e-12,
        )
        if sol.status != 0:
            print(f"  {label}: solver failed")
            continue

        monitor = ChemicalMonitor(
            species, S, threshold=1e-6,
            rate_constants=rates, reactant_matrix=reactant_matrix,
            reverse_pairs=reverse_pairs,
        )
        monitor.set_initial(c0)

        for i in range(1, sol.y.shape[1]):
            monitor.check(sol.y[:, i])

        summary = monitor.summary()
        print(f"\n  {label}:")
        for name, data in sorted(summary.items(), key=lambda x: -x[1]["frac_var"]):
            fv = data["frac_var"]
            flag = " <<<" if fv > 1e-6 else ""
            print(f"    {name:<35s}  frac_var={fv:.2e}  "
                  f"init={data['initial']:.4f}  final={data['final']:.4f}{flag}")

    # Wegscheider comparison
    print(f"\n  Wegscheider cycle product (should be constant if thermodynamically consistent):")
    for label, rates in scenarios:
        # Product of k_fwd/k_rev around the cycle
        product = 1.0
        for fwd, rev in reverse_pairs:
            product *= rates[fwd] / rates[rev]
        print(f"    {label:<35s}  cycle_product = {product:.4f}")


# ─── Experiment 5: R_f vs standard monitors sensitivity comparison ────────────

def exp5_rf_sensitivity():
    """Direct comparison: R_f detection sensitivity vs H, Lz, E for vortex corruption."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 5: R_f vs standard monitors — sensitivity comparison")
    print("=" * 72)

    G, pos0 = get_vortex_ic()
    t_end = 100.0
    n_points = 2000

    # Standard monitors (textbook)
    standard = ["H", "Lz", "Px", "Py"]
    # NoetherSolve-discovered
    discovered = ["Q_linear", "Q_sqrt", "Q_exp", "R_f", "Q_tanh"]

    noise_levels = np.logspace(-10, -1, 20)
    results = {name: [] for name in standard + discovered}

    for noise in noise_levels:
        sol = solve_ivp(
            vortex_rhs, (0, t_end), pos0.ravel(), args=(G,),
            method="RK45", t_eval=np.linspace(0, t_end, n_points),
            rtol=1e-10, atol=1e-12,
        )
        # Corrupt every 100th step
        trajectory = sol.y.T.copy()
        rng = np.random.RandomState(42)
        for step in range(100, len(trajectory), 100):
            trajectory[step] += rng.randn(trajectory.shape[1]) * noise

        monitor = VortexMonitor(G)
        monitor.set_initial(pos0)
        dt = t_end / n_points
        for i in range(1, len(trajectory)):
            monitor.check(trajectory[i].reshape(-1, 2), dt=dt)

        summary = monitor.summary()
        for name in standard + discovered:
            if name in summary:
                results[name].append(summary[name]["frac_var"])
            else:
                results[name].append(0.0)

    # Print table
    print(f"\n{'Noise':>10} ", end="")
    print(f"  {'|':>3} ", end="")
    for name in standard:
        print(f" {name:>10}", end="")
    print(f"  {'|':>3} ", end="")
    for name in discovered:
        print(f" {name:>10}", end="")
    print()
    print("-" * (14 + 12 * len(standard) + 5 + 12 * len(discovered)))

    for idx, noise in enumerate(noise_levels):
        print(f"{noise:>10.1e} ", end="")
        print(f"  {'|':>3} ", end="")
        for name in standard:
            fv = results[name][idx]
            print(f" {fv:>10.2e}", end="")
        print(f"  {'|':>3} ", end="")
        for name in discovered:
            fv = results[name][idx]
            print(f" {fv:>10.2e}", end="")
        print()

    # Find detection threshold for each monitor (first noise where frac_var > 1e-6)
    print(f"\nDetection threshold (frac_var > 1e-6):")
    print(f"  {'Standard monitors':}")
    for name in standard:
        for idx, fv in enumerate(results[name]):
            if fv > 1e-6:
                print(f"    {name:<12} detects at noise = {noise_levels[idx]:.1e}")
                break
        else:
            print(f"    {name:<12} never detects")

    print(f"  {'NoetherSolve monitors':}")
    for name in discovered:
        for idx, fv in enumerate(results[name]):
            if fv > 1e-6:
                print(f"    {name:<12} detects at noise = {noise_levels[idx]:.1e}")
                break
        else:
            print(f"    {name:<12} never detects")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("NoetherSolve Conservation Monitor — Corruption Benchmark")
    print("All CPU, no models needed.\n")

    r1 = exp1_tolerance_sweep()
    r2 = exp2_single_step_corruption()
    r3 = exp3_wrong_physics()
    r4 = exp4_chemical_violation()
    r5 = exp5_rf_sensitivity()

    print("\n" + "=" * 72)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 72)

    # Save results
    output = {
        "exp1_tolerance": [{
            "rtol": r["rtol"],
            "final_frac_vars": r["final_frac_vars"],
        } for r in r1],
        "exp3_wrong_physics": {
            k: {name: data["frac_var"] for name, data in v.items()}
            for k, v in r3.items()
        },
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "corruption_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
