#!/usr/bin/env python3
"""
Ranking Adapter Training

Goal: Learn to rank conservation quality by 1/frac_var, not just binary classification.

Strategy:
1. Use diverse configurations (N=3,4,5,9) with actual frac_var values
2. Pairwise ranking loss: if fv_i < fv_j, then margin_i > margin_j
3. Correlation loss: Pearson(margins, 1/frac_var) → 1
4. Maintain anchors (Q₂ exact, K rejection for chaos)
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import mlx.core as mx
import mlx.optimizers as optim
import mlx_lm
from mlx.utils import tree_flatten, tree_unflatten

from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve import train_utils as t3

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "adapters")


def vortex_rhs(t, y, gammas):
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


def compute_Qn(sol, gammas, n_power):
    n_vort = len(gammas)
    Q_vals = []
    for k in range(sol.y.shape[1]):
        x, y = sol.y[:n_vort, k], sol.y[n_vort:, k]
        Q = sum(gammas[i]*gammas[j]*(np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2))**n_power
                for i in range(n_vort) for j in range(i+1, n_vort))
        Q_vals.append(Q)
    return np.array(Q_vals)


def frac_var(v):
    return np.var(v) / (np.mean(v)**2 + 1e-20)


def generate_training_data():
    """Generate diverse training examples with actual frac_var values."""
    np.random.seed(123)
    examples = []

    configs = [
        # (name, N, gammas, x0, y0)
        ("restricted_3", 3, [1.0, 1.0, 0.01], [0.5, -0.5, 0.3], [0.0, 0.0, 0.8]),
        ("equal_3", 3, [1.0, 1.0, 1.0], [0.5, -0.5, 0.3], [0.0, 0.0, 0.8]),
        ("hierarchical_4", 4, [1.0, 0.5, 0.2, 0.1], [0.5, -0.5, -0.3, 0.4], [0.3, 0.4, -0.5, -0.3]),
        ("equal_4", 4, [1.0, 1.0, 1.0, 1.0], [0.5, -0.5, -0.5, 0.5], [0.5, 0.5, -0.5, -0.5]),
    ]

    # Add random N=5 and N=9 configs
    for seed in [42, 43, 44]:
        np.random.seed(seed)
        for N in [5, 9]:
            gammas = (2 * np.random.rand(N) - 1).tolist()
            r_pos = np.sqrt(np.random.rand(N)) * 0.8
            theta = 2 * np.pi * np.random.rand(N)
            x0 = (r_pos * np.cos(theta)).tolist()
            y0 = (r_pos * np.sin(theta)).tolist()
            configs.append((f"random_{N}_s{seed}", N, gammas, x0, y0))

    n_powers = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    print("Generating training data from simulations...")
    for name, N, gammas, x0, y0 in configs:
        state0 = np.array(x0 + y0)

        try:
            sol = solve_ivp(lambda t, y: vortex_rhs(t, y, gammas), (0, 50), state0,
                           t_eval=np.linspace(0, 50, 501), rtol=1e-9, atol=1e-11)

            if not sol.success:
                continue

            for n_power in n_powers:
                Q_vals = compute_Qn(sol, gammas, n_power)
                fv = frac_var(Q_vals)

                if fv < 1e-15:  # Exact (n=2 case)
                    fv = 1e-15

                examples.append({
                    "config": name,
                    "N": N,
                    "n_power": n_power,
                    "frac_var": fv,
                    "gammas": gammas,
                    "conserved": fv < 5e-3
                })
        except Exception:
            continue

    print(f"  Generated {len(examples)} examples")
    return examples


def get_margin(adapter, lm_head, model, tokenizer, prompt, truth, distractors):
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
    return truth_lp - best_dist


def ranking_loss(adapter, lm_head, model, tokenizer, batch_examples,
                 hinge_weight=1.0, rank_weight=3.0, corr_weight=2.0):
    """
    Ranking-focused loss:
    1. Hinge: push conserved examples to positive margin
    2. Pairwise ranking: if fv_i < fv_j, ensure margin_i > margin_j
    3. Correlation: maximize Pearson(margins, 1/frac_var)
    """
    margins = []
    targets = []  # 1/frac_var (higher = better conservation)

    for ex in batch_examples:
        fv = ex["frac_var"]
        n_power = ex["n_power"]
        N = ex["N"]

        prompt = f"N={N} vortex system. Is Q_{n_power} = Σ ΓᵢΓⱼ rᵢⱼ^{n_power} conserved?:"

        if ex["conserved"]:
            truth = f" Yes, conserved with frac_var ≈ {fv:.1e}."
        else:
            truth = f" No, poorly conserved (frac_var = {fv:.1e})."

        distractors = [" Cannot determine", " Only for N=2", " Depends on initial conditions"]

        m = get_margin(adapter, lm_head, model, tokenizer, prompt, truth, distractors)
        margins.append(m)
        targets.append(1.0 / (fv + 1e-15))

    margins_stack = mx.stack(margins)
    targets_stack = mx.array(targets)

    # Normalize targets to [0, 1]
    targets_norm = targets_stack / (mx.max(targets_stack) + 1e-8)

    # 1. Hinge loss for conserved examples
    conserved_mask = mx.array([1.0 if ex["conserved"] else 0.0 for ex in batch_examples])
    hinge_loss = mx.mean(conserved_mask * mx.maximum(mx.array(0.0), mx.array(1.0) - margins_stack))

    # 2. Pairwise ranking loss
    # For each pair (i,j), if target_i > target_j, we want margin_i > margin_j
    rank_loss = mx.array(0.0)
    n_pairs = 0
    for i in range(len(batch_examples)):
        for j in range(i+1, len(batch_examples)):
            if float(targets_stack[i]) > float(targets_stack[j]) * 1.5:  # Significant difference
                # We want margin_i > margin_j
                # Loss if margin_j >= margin_i
                pair_loss = mx.maximum(mx.array(0.0), margins_stack[j] - margins_stack[i] + mx.array(0.5))
                rank_loss = rank_loss + pair_loss
                n_pairs += 1
            elif float(targets_stack[j]) > float(targets_stack[i]) * 1.5:
                pair_loss = mx.maximum(mx.array(0.0), margins_stack[i] - margins_stack[j] + mx.array(0.5))
                rank_loss = rank_loss + pair_loss
                n_pairs += 1

    if n_pairs > 0:
        rank_loss = rank_loss / n_pairs

    # 3. Correlation loss
    margins_mean = mx.mean(margins_stack)
    targets_mean = mx.mean(targets_norm)
    margins_centered = margins_stack - margins_mean
    targets_centered = targets_norm - targets_mean

    numerator = mx.sum(margins_centered * targets_centered)
    denominator = mx.sqrt(mx.sum(margins_centered**2) * mx.sum(targets_centered**2) + 1e-8)
    corr = numerator / denominator
    corr_loss = 1.0 - corr

    total_loss = hinge_weight * hinge_loss + rank_weight * rank_loss + corr_weight * corr_loss

    return total_loss, float(hinge_loss), float(rank_loss), float(corr), margins


def clip_grads(grads, max_norm=1.0):
    leaves = tree_flatten(grads)
    total_sq = sum(float(mx.sum(g ** 2)) for _, g in leaves)
    norm = total_sq ** 0.5
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        leaves = [(k, g * scale) for k, g in leaves]
    return tree_unflatten(leaves)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--load-adapter", default=os.path.join(OUT_DIR, "physics_supervised.npz"))
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "ranking_adapter.npz"))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("RANKING ADAPTER TRAINING")
    print("=" * 70)

    # Generate training data
    examples = generate_training_data()

    # Sort by frac_var for analysis
    examples_sorted = sorted(examples, key=lambda x: x["frac_var"])
    print(f"\nFrac_var range: {examples_sorted[0]['frac_var']:.1e} to {examples_sorted[-1]['frac_var']:.1e}")
    print(f"Conserved: {sum(1 for e in examples if e['conserved'])}/{len(examples)}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)

    # Create adapter
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=64, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)

    # Load previous adapter
    if os.path.exists(args.load_adapter):
        print(f"Loading adapter: {args.load_adapter}")
        weights = dict(mx.load(args.load_adapter))
        adapter.load_weights(list(weights.items()))

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
    loss_and_grad = mx.value_and_grad(ranking_loss, argnums=0)

    # Training
    print(f"\nTraining: {args.steps} steps, batch={args.batch}, lr={args.lr}")
    print("-" * 70)

    best_corr = -1.0
    best_state = None

    for step in range(args.steps):
        # Random batch
        batch_idx = np.random.choice(len(examples), min(args.batch, len(examples)), replace=False)
        batch = [examples[i] for i in batch_idx]

        (loss, hinge, rank, corr, margins), grads = loss_and_grad(
            adapter, lm_head, model, tokenizer, batch
        )

        grads = clip_grads(grads)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if corr > best_corr:
            best_corr = corr
            best_state = dict(tree_flatten(adapter.parameters()))

        if (step + 1) % 50 == 0:
            margins_f = [float(m) for m in margins]
            print(f"  step {step+1:4d}/{args.steps}  loss={float(loss):.2f}  "
                  f"hinge={hinge:.2f}  rank={rank:.2f}  corr={corr:+.3f}  "
                  f"margins=[{min(margins_f):+.0f},{max(margins_f):+.0f}]")

    # Evaluation
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    # Test on all examples
    all_margins = []
    all_fv = []
    for ex in examples:
        fv = ex["frac_var"]
        n_power = ex["n_power"]
        N = ex["N"]

        prompt = f"N={N} vortex system. Is Q_{n_power} = Σ ΓᵢΓⱼ rᵢⱼ^{n_power} conserved?:"
        truth = " Yes, conserved." if ex["conserved"] else " No, not conserved."
        distractors = [" Cannot determine", " Only for N=2", " Depends"]

        m = float(get_margin(adapter, lm_head, model, tokenizer, prompt, truth, distractors))
        all_margins.append(m)
        all_fv.append(fv)

    # Final correlation
    inv_fv = [1.0/fv for fv in all_fv]
    final_corr = np.corrcoef(all_margins, inv_fv)[0, 1]

    print(f"\nFinal Pearson(margin, 1/frac_var) = {final_corr:+.3f}")

    # Show ranking
    print("\nTop 10 by margin:")
    sorted_idx = np.argsort(all_margins)[::-1]
    for i in sorted_idx[:10]:
        ex = examples[i]
        print(f"  {ex['config']:20s} n={ex['n_power']:.1f}  margin={all_margins[i]:+.1f}  frac_var={ex['frac_var']:.1e}")

    print("\nBottom 10 by margin:")
    for i in sorted_idx[-10:]:
        ex = examples[i]
        print(f"  {ex['config']:20s} n={ex['n_power']:.1f}  margin={all_margins[i]:+.1f}  frac_var={ex['frac_var']:.1e}")

    # Save
    if best_state is not None:
        mx.savez(args.out, **best_state)
        print(f"\nBest adapter (corr={best_corr:.3f}) saved: {args.out}")

    # Verdict
    print("\n" + "=" * 70)
    if final_corr > 0.7:
        print("✓ RANKING LEARNED: Strong correlation with 1/frac_var")
    elif final_corr > 0.4:
        print("~ PARTIAL: Moderate correlation")
    else:
        print("⚠ RANKING NOT LEARNED: Weak correlation")


if __name__ == "__main__":
    main()
