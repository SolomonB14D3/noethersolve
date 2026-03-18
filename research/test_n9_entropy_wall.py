#!/usr/bin/env python3
"""
N=9 Entropy Wall Test

A rigorous test of the physics-supervised adapter on a chaotic 9-vortex system.

Test Parameters:
- 9 point vortices with random circulations Γᵢ ∈ [-1, 1]
- Random positions in unit disk
- High chaos (no symmetric/stable configurations)

Probes:
- Q_f family: n ∈ [0.1, 0.5, 1.0, 1.5, 2.0]
- Kinetic invariant K = Σ Γᵢ vᵢ²

Success Criteria:
1. n=0.5 dominance: Does √r get highest margin?
2. K vs Q orthogonality: Both recognized simultaneously?
3. Computational consistency: Stable inference across complex tokens
"""

import os
import time
import numpy as np
from scipy.integrate import solve_ivp
import mlx.core as mx
import mlx_lm

from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve import train_utils as t3

HERE = os.path.dirname(os.path.abspath(__file__))

def vortex_rhs(t, y, gammas):
    """Point-vortex equations."""
    n = len(gammas)
    x, yc = y[:n], y[n:]
    dxdt, dydt = np.zeros(n), np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                dx, dy = x[i] - x[j], yc[i] - yc[j]
                r2 = dx**2 + dy**2
                if r2 > 1e-12:
                    f = gammas[j] / (2 * np.pi * r2)
                    dxdt[i] += -f * dy
                    dydt[i] += f * dx
    return np.concatenate([dxdt, dydt])

def compute_Qn(state, gammas, n_power):
    """Q_n = Σ ΓᵢΓⱼ rᵢⱼ^n"""
    n_vort = len(gammas)
    x, y = state[:n_vort], state[n_vort:]
    Q = 0.0
    for i in range(n_vort):
        for j in range(i+1, n_vort):
            r = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if r > 1e-10:
                Q += gammas[i] * gammas[j] * (r ** n_power)
    return Q

def compute_K(state, gammas):
    """K = Σ Γᵢ vᵢ²"""
    n = len(gammas)
    x, y = state[:n], state[n:]
    vx, vy = np.zeros(n), np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                dx, dy = x[i] - x[j], y[i] - y[j]
                r2 = dx**2 + dy**2
                if r2 > 1e-12:
                    f = gammas[j] / (2 * np.pi * r2)
                    vx[i] += -f * dy
                    vy[i] += f * dx
    return sum(gammas[i] * (vx[i]**2 + vy[i]**2) for i in range(n))

def frac_var(v):
    return np.var(v) / (np.mean(v)**2 + 1e-20)

def estimate_lyapunov(gammas, state0, T=10, dt=0.01):
    """Estimate maximum Lyapunov exponent."""
    n = len(gammas)
    eps = 1e-8

    # Reference trajectory
    sol_ref = solve_ivp(lambda t, y: vortex_rhs(t, y, gammas), (0, T), state0,
                        t_eval=[T], rtol=1e-10, atol=1e-12)

    # Perturbed trajectory
    perturbation = np.random.randn(2*n) * eps
    state_pert = state0 + perturbation
    sol_pert = solve_ivp(lambda t, y: vortex_rhs(t, y, gammas), (0, T), state_pert,
                         t_eval=[T], rtol=1e-10, atol=1e-12)

    # Compute divergence
    delta = np.linalg.norm(sol_pert.y[:, 0] - sol_ref.y[:, 0])
    lyap = np.log(delta / (eps * np.sqrt(2*n))) / T
    return lyap

print("=" * 70)
print("N=9 ENTROPY WALL TEST")
print("=" * 70)

# Generate chaotic N=9 system
np.random.seed(42)  # For reproducibility
N = 9

# Random circulations in [-1, 1]
gammas = 2 * np.random.rand(N) - 1
# Ensure not all same sign (more chaotic)
if np.all(gammas > 0) or np.all(gammas < 0):
    gammas[0] *= -1

# Random positions in unit disk
r_pos = np.sqrt(np.random.rand(N)) * 0.8  # Slightly inside to avoid boundary
theta_pos = 2 * np.pi * np.random.rand(N)
x0 = r_pos * np.cos(theta_pos)
y0 = r_pos * np.sin(theta_pos)
state0 = np.concatenate([x0, y0])

print("\nSystem Configuration:")
print(f"  N = {N} vortices")
print(f"  Circulations: {[f'{g:+.3f}' for g in gammas]}")
print("  Positions: random in unit disk")

# Estimate Lyapunov exponent
lyap = estimate_lyapunov(gammas, state0)
print(f"  Est. Lyapunov exponent: λ ≈ {lyap:.3f}")
if lyap > 0.1:
    print("  ✓ CHAOTIC: High Lyapunov exponent")
else:
    print("  ⚠ May not be sufficiently chaotic")

# Run numerical simulation
print("\nRunning numerical simulation (T=100)...")
t0 = time.time()
sol = solve_ivp(lambda t, y: vortex_rhs(t, y, gammas), (0, 100), state0,
                t_eval=np.linspace(0, 100, 1001), rtol=1e-10, atol=1e-12)
sim_time = time.time() - t0
print(f"  Simulation time: {sim_time:.2f}s")

# Compute conservation for Q_n family
print("\nQ_n Family Conservation Test:")
print("-" * 50)
n_powers = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
q_results = {}

for n_power in n_powers:
    Q_vals = [compute_Qn(sol.y[:, k], gammas, n_power) for k in range(len(sol.t))]
    fv = frac_var(Q_vals)
    status = "PASS" if fv < 5e-3 else "fail"
    q_results[n_power] = fv
    print(f"  n={n_power:4.2f}: frac_var = {fv:.2e} [{status}]")

# Kinetic invariant K
K_vals = [compute_K(sol.y[:, k], gammas) for k in range(len(sol.t))]
K_fv = frac_var(K_vals)
K_status = "PASS" if K_fv < 5e-3 else "fail"
print(f"\n  K (kinetic): frac_var = {K_fv:.2e} [{K_status}]")

# Format for Oracle prompts
print("\n" + "=" * 70)
print("ORACLE TEST WITH PHYSICS-SUPERVISED ADAPTER")
print("=" * 70)

# Load model and adapter
print("\nLoading model and adapter...")
model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
model.freeze()
lm_head = t3.get_lm_head_fn(model)

# Load physics-supervised adapter
vocab_size = model.model.embed_tokens.weight.shape[0]
d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
cfg = SnapOnConfig(d_model=d_model, d_inner=64, n_layers=0,
                   n_heads=8, mode="logit", vocab_size=vocab_size)
adapter = create_adapter(cfg)

adapter_path = os.path.join(HERE, "adapters", "physics_supervised.npz")
if os.path.exists(adapter_path):
    weights = dict(mx.load(adapter_path))
    adapter.load_weights(list(weights.items()))
    print(f"  Loaded: {adapter_path}")
else:
    print("  WARNING: No physics_supervised.npz found!")

def get_margin_timed(adapter, prompt, truth, distractors):
    """Get margin with timing."""
    t0 = time.time()

    def lp(text):
        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(prompt + text)
        n_prompt = len(prompt_ids)
        if len(full_ids) <= n_prompt:
            return mx.array(-1e9)
        tokens = mx.array(full_ids)[None, :]
        h = model.model(tokens)
        base_logits = lm_head(h)
        if adapter is not None:
            shifts = adapter(base_logits)
            shifts = shifts - shifts.mean(axis=-1, keepdims=True)
            logits = base_logits + shifts
            logits = t3.LOGIT_SOFTCAP * mx.tanh(logits / t3.LOGIT_SOFTCAP)
        else:
            logits = base_logits
        total = mx.array(0.0)
        for i, tok_id in enumerate(full_ids[n_prompt:]):
            pos = n_prompt - 1 + i
            lv = logits[0, pos]
            lse = mx.log(mx.sum(mx.exp(lv - mx.max(lv))) + 1e-8) + mx.max(lv)
            total = total + lv[tok_id] - lse
        return total

    truth_lp = lp(f" {truth}")
    dist_lps = [lp(f" {d}") for d in distractors]
    best_dist = mx.max(mx.stack(dist_lps))
    margin = float(truth_lp - best_dist)

    elapsed = time.time() - t0
    return margin, elapsed

# Build high-precision prompts
gammas_str = ", ".join([f"{g:+.6f}" for g in gammas])

print("\nOracle Probes (high-precision coordinates):")
print("-" * 70)

oracle_results = []

# Q_n family probes
for n_power in [0.1, 0.5, 1.0, 1.5, 2.0]:
    fv = q_results[n_power]
    conserved = fv < 5e-3

    prompt = (f"N=9 chaotic vortex system. Circulations: [{gammas_str}]. "
              f"Is Q_{n_power} = Σ ΓᵢΓⱼ rᵢⱼ^{n_power} conserved?:")

    if conserved:
        truth = f" Yes, approximately conserved with frac_var ≈ {fv:.1e}."
    else:
        truth = f" No, not well conserved (frac_var = {fv:.1e} > threshold)."

    distractors = [
        " Cannot determine from given data",
        " Only exact for N=2",
        " Requires symmetric circulations"
    ]

    margin, elapsed = get_margin_timed(adapter, prompt, truth, distractors)
    oracle_results.append({
        "type": f"Q_{n_power}",
        "n": n_power,
        "frac_var": fv,
        "conserved": conserved,
        "margin": margin,
        "time_ms": elapsed * 1000
    })

    status = "✓" if margin > 0 else "✗"
    print(f"  Q_{n_power:4.1f}: margin={margin:+8.1f}  frac_var={fv:.1e}  time={elapsed*1000:.0f}ms {status}")

# Kinetic invariant K probe
prompt = (f"N=9 chaotic vortex system. Circulations: [{gammas_str}]. "
          f"Is K = Σ Γᵢ vᵢ² (kinetic invariant) conserved?:")
truth = f" Yes, approximately conserved with frac_var ≈ {K_fv:.1e}." if K_fv < 5e-3 else " No, not conserved."
distractors = [" Only Q_f is conserved", " K requires equal circulations", " Cannot determine"]

margin_K, elapsed_K = get_margin_timed(adapter, prompt, truth, distractors)
oracle_results.append({
    "type": "K",
    "n": None,
    "frac_var": K_fv,
    "conserved": K_fv < 5e-3,
    "margin": margin_K,
    "time_ms": elapsed_K * 1000
})
print(f"  K:      margin={margin_K:+8.1f}  frac_var={K_fv:.1e}  time={elapsed_K*1000:.0f}ms")

# Analysis
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

# 1. n=0.5 dominance test
q_margins = {r["n"]: r["margin"] for r in oracle_results if r["type"].startswith("Q_")}
best_n = max(q_margins, key=q_margins.get)
print("\n1. n=0.5 (√r) DOMINANCE TEST:")
print(f"   Best margin: n={best_n} with margin={q_margins[best_n]:+.1f}")
if best_n == 0.5:
    print("   ✓ Model learned sub-linear filtering property!")
elif best_n < 1.0:
    print(f"   ~ Model prefers sub-linear (n={best_n}) but not exactly 0.5")
else:
    print(f"   ⚠ Model prefers n={best_n}, not the expected sub-linear regime")

# 2. K vs Q orthogonality
print("\n2. K vs Q ORTHOGONALITY TEST:")
print(f"   K margin: {margin_K:+.1f}")
print(f"   Q_0.5 margin: {q_margins[0.5]:+.1f}")
if margin_K > 0 and q_margins[0.5] > 0:
    print("   ✓ Both K and Q_0.5 recognized simultaneously!")
elif margin_K > 0 or q_margins[0.5] > 0:
    print("   ~ Partial: one recognized, one not")
else:
    print("   ⚠ Neither recognized on chaotic N=9")

# 3. Computational consistency
times = [r["time_ms"] for r in oracle_results]
time_mean = np.mean(times)
time_std = np.std(times)
time_cv = time_std / time_mean

print("\n3. COMPUTATIONAL CONSISTENCY:")
print(f"   Mean inference time: {time_mean:.0f}ms")
print(f"   Std: {time_std:.0f}ms (CV={time_cv:.2%})")
if time_cv < 0.2:
    print("   ✓ Consistent computational effort across probes")
else:
    print("   ⚠ Variable effort - may indicate shortcuts on some probes")

# 4. Physics correlation on N=9
frac_vars = [r["frac_var"] for r in oracle_results if r["type"].startswith("Q_")]
margins = [r["margin"] for r in oracle_results if r["type"].startswith("Q_")]
inv_fv = [1.0/fv for fv in frac_vars]
corr = np.corrcoef(margins, inv_fv)[0, 1]

print("\n4. PHYSICS CORRELATION (N=9):")
print(f"   Pearson(margin, 1/frac_var) = {corr:+.3f}")
if corr > 0.5:
    print("   ✓ Physics learned generalizes to N=9 chaotic!")
elif corr > 0:
    print("   ~ Weak positive correlation")
else:
    print("   ⚠ No correlation - may need more training")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
passed = sum([
    best_n <= 0.5,
    margin_K > 0 and q_margins.get(0.5, -999) > 0,
    time_cv < 0.2,
    corr > 0.3
])
print(f"Tests passed: {passed}/4")

if passed == 4:
    print("\n✓ ENTROPY WALL BREACHED: Adapter generalizes to chaotic N=9!")
elif passed >= 2:
    print("\n~ PARTIAL SUCCESS: Some generalization achieved")
else:
    print("\n⚠ ENTROPY WALL HOLDS: Adapter may be overfitting to training distribution")

