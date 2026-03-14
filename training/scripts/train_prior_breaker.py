#!/usr/bin/env python3
"""
Prior Breaker Adapter Training

Strategy:
1. ANCHOR: Q₂ (exact invariant) - maintain physics capability
2. TARGET: H·r₁₂ + α·Lz family - break frozen -77.5 prior
3. LOSS: Margin Divergence - force margins to VARY with α

The key insight: if margins don't vary with α, the model is using a shortcut.
We penalize constant margins across the α family.
"""

import argparse
import json
import os
import sys
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


def margin_divergence_loss(adapter, lm_head, model, tokenizer, 
                           alpha_examples, anchor_example,
                           divergence_weight=1.0, anchor_weight=2.0):
    """
    Combined loss:
    1. Anchor loss: Hinge loss on Q₂ (must stay correct)
    2. Divergence loss: Variance of margins across α values (must increase)
    """
    # Compute margins for all α variants
    margins = []
    for ex in alpha_examples:
        m = get_margin(adapter, lm_head, model, tokenizer,
                      ex["prompt"], ex["truth"], ex["distractors"])
        margins.append(m)
    
    margins_stack = mx.stack(margins)
    
    # Divergence loss: MAXIMIZE variance of margins
    # We want margins to diverge, so we MINIMIZE negative variance
    margin_mean = mx.mean(margins_stack)
    margin_var = mx.mean((margins_stack - margin_mean) ** 2)
    divergence_loss = -margin_var  # Negative because we want to maximize variance
    
    # Anchor loss: Hinge loss on Q₂ (must have positive margin)
    anchor_margin = get_margin(adapter, lm_head, model, tokenizer,
                               anchor_example["prompt"], 
                               anchor_example["truth"],
                               anchor_example["distractors"])
    anchor_loss = mx.maximum(mx.array(0.0), mx.array(1.5) - anchor_margin)
    
    # Combined loss
    total_loss = divergence_weight * divergence_loss + anchor_weight * anchor_loss
    
    return total_loss, float(margin_var), float(anchor_margin), margins


def clip_grads(grads, max_norm=1.0):
    leaves = tree_flatten(grads)
    total_sq = sum(float(mx.sum(g ** 2)) for _, g in leaves)
    norm = total_sq ** 0.5
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        leaves = [(k, g * scale) for k, g in leaves]
    return tree_unflatten(leaves)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--div-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=2.0)
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "prior_breaker.npz"))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Define the α variants (TARGET)
    alpha_variants = [
        {"alpha": 0.001, "frac_var": "1.68e-3"},
        {"alpha": 0.01, "frac_var": "1.50e-3"},
        {"alpha": 0.1, "frac_var": "1.20e-3"},
        {"alpha": 0.5, "frac_var": "5.0e-4"},
        {"alpha": 1.0, "frac_var": "2.08e-4"},
        {"alpha": 2.0, "frac_var": "8.0e-4"},
        {"alpha": 10.0, "frac_var": "3.50e-3"},
    ]

    alpha_examples = []
    for v in alpha_variants:
        alpha = v["alpha"]
        alpha_examples.append({
            "prompt": f"Restricted 3-vortex: Γ₁=Γ₂=1, Γ₃=0.01. Is Q = H·r₁₂ + {alpha}·Lz conserved?:",
            "truth": f" Near-conserved with frac_var ≈ {v['frac_var']}. The coefficient α={alpha} affects conservation quality.",
            "distractors": [
                " Not conserved at all",
                " Only H is conserved, the Lz term makes it worse",
                f" Same conservation regardless of α={alpha}"
            ],
            "alpha": alpha,
        })

    # Define the ANCHOR (Q₂ exact)
    anchor_example = {
        "prompt": "N-vortex system. Is Q₂ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ² exactly conserved?:",
        "truth": " Yes, Q₂ is exactly conserved because it reduces to Γ_total·Lz (angular impulse)",
        "distractors": [
            " No, Q₂ is only approximately conserved",
            " Only for 2 vortices",
            " Q₂ equals the Hamiltonian H"
        ],
    }

    print(f"\nPrior Breaker Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Steps: {args.steps}, LR: {args.lr}")
    print(f"Divergence weight: {args.div_weight}, Anchor weight: {args.anchor_weight}")
    print(f"α variants: {len(alpha_examples)}")
    print()

    print("Loading model...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Create adapter
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=args.d_inner, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    # Baseline evaluation
    print("\n" + "=" * 60)
    print("BASELINE (no adapter)")
    print("=" * 60)
    
    baseline_margins = []
    for ex in alpha_examples:
        m = get_margin(None, lm_head, model, tokenizer,
                      ex["prompt"], ex["truth"], ex["distractors"])
        baseline_margins.append(float(m))
        print(f"  α={ex['alpha']:6.3f}: margin = {float(m):+.3f}")
    
    baseline_var = np.var(baseline_margins)
    baseline_mean = np.mean(baseline_margins)
    print(f"\n  Margin mean: {baseline_mean:+.3f}")
    print(f"  Margin std:  {np.std(baseline_margins):.3f}")
    print(f"  Margin var:  {baseline_var:.3f}")
    
    anchor_m = get_margin(None, lm_head, model, tokenizer,
                         anchor_example["prompt"], anchor_example["truth"],
                         anchor_example["distractors"])
    print(f"\n  ANCHOR (Q₂): margin = {float(anchor_m):+.3f}")

    # Training
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    loss_and_grad = mx.value_and_grad(margin_divergence_loss, argnums=0)

    best_var = 0.0
    best_adapter_state = None

    for step in range(args.steps):
        (loss_val, margin_var, anchor_margin, margins), grads = loss_and_grad(
            adapter, lm_head, model, tokenizer,
            alpha_examples, anchor_example,
            args.div_weight, args.anchor_weight
        )
        
        grads = clip_grads(grads)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        # Track best
        if margin_var > best_var and anchor_margin > 0:
            best_var = margin_var
            best_adapter_state = dict(tree_flatten(adapter.parameters()))

        if (step + 1) % 50 == 0:
            margins_list = [float(m) for m in margins]
            print(f"  step {step+1:4d}/{args.steps}  loss={float(loss_val):+.3f}  "
                  f"var={margin_var:.3f}  anchor={anchor_margin:+.3f}  "
                  f"margins=[{min(margins_list):+.1f}, {max(margins_list):+.1f}]")

    # Final evaluation
    print("\n" + "=" * 60)
    print("AFTER TRAINING")
    print("=" * 60)

    final_margins = []
    for ex in alpha_examples:
        m = get_margin(adapter, lm_head, model, tokenizer,
                      ex["prompt"], ex["truth"], ex["distractors"])
        final_margins.append(float(m))
        print(f"  α={ex['alpha']:6.3f}: margin = {float(m):+.3f}")

    final_var = np.var(final_margins)
    final_mean = np.mean(final_margins)
    print(f"\n  Margin mean: {final_mean:+.3f}")
    print(f"  Margin std:  {np.std(final_margins):.3f}")
    print(f"  Margin var:  {final_var:.3f}")
    
    anchor_m = get_margin(adapter, lm_head, model, tokenizer,
                         anchor_example["prompt"], anchor_example["truth"],
                         anchor_example["distractors"])
    print(f"\n  ANCHOR (Q₂): margin = {float(anchor_m):+.3f}")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    var_improvement = final_var / (baseline_var + 1e-10)
    print(f"  Variance improvement: {var_improvement:.1f}x")
    
    if final_var > baseline_var * 2:
        print("  ✓ PRIOR BROKEN: Margins now diverge with α!")
    else:
        print("  ⚠ Prior still active: Margins not diverging enough")
    
    if anchor_margin > 0:
        print("  ✓ ANCHOR INTACT: Q₂ still recognized as conserved")
    else:
        print("  ⚠ Anchor damaged: Q₂ margin went negative")

    # Save
    if best_adapter_state is not None:
        mx.savez(args.out, **best_adapter_state)
        print(f"\n  Best adapter saved: {args.out}")
    else:
        weights = dict(tree_flatten(adapter.parameters()))
        mx.savez(args.out, **weights)
        print(f"\n  Final adapter saved: {args.out}")


if __name__ == "__main__":
    main()
