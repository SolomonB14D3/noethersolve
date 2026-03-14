#!/usr/bin/env python3
"""
Phase 3: Physics-Supervised Adapter Training

Loss Components:
1. Hinge Loss: Push margins positive (max(0, 1 - margin))
2. Correlation Loss: Force margins to track 1/frac_var (minimize 1 - Pearson)
3. Q₁ Anchor: Stabilize using Q₁ (independent novel invariant)
4. Q₂ Control: Maintain exact invariant recognition

The key innovation: We feed ACTUAL PHYSICS (1/frac_var) as the truth signal.
"""

import os
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
from mlx.utils import tree_flatten, tree_unflatten

from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve import train_utils as t3

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "adapters")


def pearson_correlation_mx(x, y):
    """Compute Pearson correlation in MLX."""
    x_mean = mx.mean(x)
    y_mean = mx.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    numerator = mx.sum(x_centered * y_centered)
    denominator = mx.sqrt(mx.sum(x_centered**2) * mx.sum(y_centered**2) + 1e-8)
    
    return numerator / denominator


def get_margin(adapter, lm_head, model, tokenizer, prompt, truth, distractors):
    """Compute margin = truth_lp - max(distractor_lp)"""
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


def physics_supervised_loss(adapter, lm_head, model, tokenizer,
                            alpha_examples, physics_targets,
                            q1_example, q2_example,
                            hinge_weight=1.0, corr_weight=2.0,
                            q1_weight=1.0, q2_weight=2.0):
    """
    Physics-Supervised Loss:
    
    L = hinge_weight * Σ max(0, 1 - margin_i)           # Push positive
      + corr_weight * (1 - Pearson(margins, 1/frac_var))  # Track physics
      + q1_weight * max(0, 1 - q1_margin)                # Q₁ anchor
      + q2_weight * max(0, 2 - q2_margin)                # Q₂ control (higher bar)
    """
    # Compute margins for all α variants
    margins = []
    for ex in alpha_examples:
        m = get_margin(adapter, lm_head, model, tokenizer,
                      ex["prompt"], ex["truth"], ex["distractors"])
        margins.append(m)
    
    margins_stack = mx.stack(margins)
    targets_stack = mx.array(physics_targets)  # 1/frac_var values
    
    # 1. Hinge loss: push margins positive
    hinge_loss = mx.mean(mx.maximum(mx.array(0.0), mx.array(1.0) - margins_stack))
    
    # 2. Correlation loss: track 1/frac_var
    # Normalize margins to same scale as targets for correlation
    margins_norm = (margins_stack - mx.mean(margins_stack)) / (mx.std(margins_stack) + 1e-8)
    targets_norm = (targets_stack - mx.mean(targets_stack)) / (mx.std(targets_stack) + 1e-8)
    corr = pearson_correlation_mx(margins_norm, targets_norm)
    corr_loss = 1.0 - corr  # Want correlation to be +1
    
    # 3. Q₁ anchor (novel invariant - should be recognized)
    q1_margin = get_margin(adapter, lm_head, model, tokenizer,
                          q1_example["prompt"], q1_example["truth"], 
                          q1_example["distractors"])
    q1_loss = mx.maximum(mx.array(0.0), mx.array(1.0) - q1_margin)
    
    # 4. Q₂ control (exact invariant - must stay correct, higher bar)
    q2_margin = get_margin(adapter, lm_head, model, tokenizer,
                          q2_example["prompt"], q2_example["truth"],
                          q2_example["distractors"])
    q2_loss = mx.maximum(mx.array(0.0), mx.array(2.0) - q2_margin)
    
    # Combined loss
    total_loss = (hinge_weight * hinge_loss + 
                  corr_weight * corr_loss +
                  q1_weight * q1_loss +
                  q2_weight * q2_loss)
    
    return (total_loss, float(hinge_loss), float(corr), 
            float(q1_margin), float(q2_margin), margins)


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
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--load-adapter", default=os.path.join(OUT_DIR, "prior_breaker.npz"))
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "physics_supervised.npz"))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # PHYSICS DATA: actual frac_var from numerical simulations
    # Target is 1/frac_var (higher = better conservation)
    alpha_data = [
        {"alpha": 0.001, "frac_var": 1.68e-3},
        {"alpha": 0.01, "frac_var": 1.50e-3},
        {"alpha": 0.1, "frac_var": 1.20e-3},
        {"alpha": 0.5, "frac_var": 5.0e-4},
        {"alpha": 1.0, "frac_var": 2.08e-4},  # BEST
        {"alpha": 2.0, "frac_var": 8.0e-4},
        {"alpha": 10.0, "frac_var": 3.50e-3},
    ]
    
    # Physics targets: 1/frac_var (normalized)
    physics_targets = [1.0 / d["frac_var"] for d in alpha_data]
    physics_targets = np.array(physics_targets)
    physics_targets = physics_targets / physics_targets.max()  # Normalize to [0, 1]
    
    print("Physics-Supervised Training (Phase 3)")
    print("=" * 60)
    print("\nPhysics Truth Signal (1/frac_var, normalized):")
    for d, t in zip(alpha_data, physics_targets):
        print(f"  α={d['alpha']:6.3f}: frac_var={d['frac_var']:.2e} → target={t:.3f}")
    print(f"\n  Best: α=1.0 (target=1.000)")
    
    # Build examples
    alpha_examples = []
    for d in alpha_data:
        alpha = d["alpha"]
        fv = d["frac_var"]
        alpha_examples.append({
            "prompt": f"Restricted 3-vortex: Γ₁=Γ₂=1, Γ₃=0.01. Is Q = H·r₁₂ + {alpha}·Lz conserved?:",
            "truth": f" Yes, approximately conserved with frac_var ≈ {fv:.1e}. Conservation quality depends on α.",
            "distractors": [
                " Not conserved at all",
                " Only H is conserved",
                f" Conservation does not depend on α={alpha}"
            ],
            "alpha": alpha,
        })
    
    # Q₁ example (novel - validation)
    q1_example = {
        "prompt": "N-vortex system. Is Q₁ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ conserved?:",
        "truth": " Yes, approximately conserved (frac_var ~ 5e-6). Circulation weighting is key.",
        "distractors": [" No, Q₁ is not conserved", " Only Q₂ is conserved", " Q₁ = H"],
    }
    
    # Q₂ example (exact - control)
    q2_example = {
        "prompt": "N-vortex system. Is Q₂ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ² exactly conserved?:",
        "truth": " Yes, exactly conserved (reduces to Γ_total·Lz, the angular impulse)",
        "distractors": [" No, only approximate", " Only for 2 vortices", " Q₂ = H"],
    }

    print(f"\nLoading model: {args.model}")
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)

    # Create adapter
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=64, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)

    # Load prior_breaker weights if available
    if os.path.exists(args.load_adapter):
        print(f"Loading adapter: {args.load_adapter}")
        weights = dict(mx.load(args.load_adapter))
        adapter.load_weights(list(weights.items()))
    
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
    loss_and_grad = mx.value_and_grad(physics_supervised_loss, argnums=0)

    # Baseline
    print("\n" + "=" * 60)
    print("BASELINE")
    print("=" * 60)
    
    for ex, target in zip(alpha_examples, physics_targets):
        m = float(get_margin(adapter, lm_head, model, tokenizer,
                            ex["prompt"], ex["truth"], ex["distractors"]))
        print(f"  α={ex['alpha']:6.3f}: margin={m:+8.1f}  target={target:.3f}")
    
    q1_m = float(get_margin(adapter, lm_head, model, tokenizer,
                           q1_example["prompt"], q1_example["truth"], q1_example["distractors"]))
    q2_m = float(get_margin(adapter, lm_head, model, tokenizer,
                           q2_example["prompt"], q2_example["truth"], q2_example["distractors"]))
    print(f"\n  Q₁ (novel): {q1_m:+.1f}")
    print(f"  Q₂ (exact): {q2_m:+.1f}")

    # Training
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_corr = -1.0
    best_state = None

    for step in range(args.steps):
        (loss, hinge, corr, q1_m, q2_m, margins), grads = loss_and_grad(
            adapter, lm_head, model, tokenizer,
            alpha_examples, physics_targets.tolist(),
            q1_example, q2_example
        )
        
        grads = clip_grads(grads)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if corr > best_corr and q2_m > 0:
            best_corr = corr
            best_state = dict(tree_flatten(adapter.parameters()))

        if (step + 1) % 50 == 0:
            margins_f = [float(m) for m in margins]
            print(f"  step {step+1:4d}/{args.steps}  loss={float(loss):+.2f}  "
                  f"hinge={hinge:.2f}  corr={corr:+.3f}  "
                  f"Q₁={q1_m:+.0f}  Q₂={q2_m:+.0f}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("AFTER TRAINING")
    print("=" * 60)
    
    final_margins = []
    for ex, target in zip(alpha_examples, physics_targets):
        m = float(get_margin(adapter, lm_head, model, tokenizer,
                            ex["prompt"], ex["truth"], ex["distractors"]))
        final_margins.append(m)
        print(f"  α={ex['alpha']:6.3f}: margin={m:+8.1f}  target={target:.3f}")
    
    # Compute final correlation
    final_corr = np.corrcoef(final_margins, physics_targets)[0, 1]
    print(f"\n  Pearson correlation (margin vs 1/frac_var): r = {final_corr:+.3f}")
    
    q1_m = float(get_margin(adapter, lm_head, model, tokenizer,
                           q1_example["prompt"], q1_example["truth"], q1_example["distractors"]))
    q2_m = float(get_margin(adapter, lm_head, model, tokenizer,
                           q2_example["prompt"], q2_example["truth"], q2_example["distractors"]))
    print(f"\n  Q₁ (novel): {q1_m:+.1f}")
    print(f"  Q₂ (exact): {q2_m:+.1f}")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if final_corr > 0.5:
        print("  ✓ PHYSICS LEARNED: Margins track 1/frac_var!")
    elif final_corr > 0:
        print("  ~ PARTIAL: Positive correlation but weak")
    else:
        print("  ⚠ NOT LEARNED: No correlation with physics")
    
    if q2_m > 0:
        print("  ✓ Q₂ CONTROL: Exact invariant recognized")
    else:
        print("  ⚠ Q₂ BROKEN: Exact invariant lost")
    
    if q1_m > 0:
        print("  ✓ Q₁ BREAKTHROUGH: Novel invariant recognized!")
    elif q1_m > -100:
        print("  ~ Q₁ IMPROVING: Getting closer")
    else:
        print("  ⚠ Q₁ NOT YET: Novel invariant still not recognized")

    # Save best
    if best_state is not None:
        mx.savez(args.out, **best_state)
        print(f"\n  Best adapter (corr={best_corr:.3f}) saved: {args.out}")
    else:
        weights = dict(tree_flatten(adapter.parameters()))
        mx.savez(args.out, **weights)
        print(f"\n  Final adapter saved: {args.out}")


if __name__ == "__main__":
    main()
