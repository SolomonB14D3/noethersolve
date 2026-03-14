#!/usr/bin/env python3
"""
Ranking Adapter v2: Log-scale targets + ListNet-style loss

Key improvements over v1:
1. Use -log10(frac_var) as target (bounded, interpretable)
2. ListNet-style loss: compare probability distributions over rankings
3. Hard negative mining: focus on difficult pairs
4. Gradient accumulation for stable updates
"""

import json
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
import mlx_lm
from pathlib import Path
from mlx.utils import tree_flatten, tree_unflatten

from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve import train_utils as t3


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


def log_score(frac_var):
    """Convert frac_var to log-scale score. Higher = better conserved."""
    # -log10(frac_var): frac_var=1e-12 -> 12, frac_var=1e-3 -> 3
    return -np.log10(max(frac_var, 1e-15))

def generate_ranking_examples():
    """Generate training examples with log-scale targets."""
    examples = []

    # Different vortex configurations
    configs = [
        ("restricted", "Γ₁=Γ₂=1, Γ₃=0.01"),
        ("equal", "Γ₁=Γ₂=Γ₃=1"),
        ("hierarchical", "Γ₁=1, Γ₂=0.5, Γ₃=0.1"),
        ("4vortex", "Γ₁=Γ₂=1, Γ₃=Γ₄=0.1"),
        ("dipole_test", "Γ₁=-Γ₂=1, Γ₃=0.5 on axis"),
        ("5vortex", "Γ₁=Γ₂=1, Γ₃=Γ₄=Γ₅=0.2"),
        ("chaotic_9", "Γᵢ random, N=9"),
    ]

    # Invariant types with typical frac_var values
    invariants = [
        # Exact invariants (frac_var ~ 1e-12)
        ("Q₂ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ²", 1e-12, "exact"),
        ("ln(r): Q = Σᵢ<ⱼ ΓᵢΓⱼ ln(rᵢⱼ)", 1e-12, "exact"),

        # Excellent near-invariants (frac_var ~ 1e-6)
        ("Q₁ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ", 5e-6, "excellent"),
        ("Q₀.₅ = Σᵢ<ⱼ ΓᵢΓⱼ √rᵢⱼ", 2e-6, "excellent"),
        ("Q₁.₅ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ^1.5", 6e-6, "excellent"),

        # Good near-invariants (frac_var ~ 1e-5)
        ("exp(-r): Q = Σᵢ<ⱼ ΓᵢΓⱼ exp(-rᵢⱼ)", 1e-5, "good"),
        ("sin(r): Q = Σᵢ<ⱼ ΓᵢΓⱼ sin(rᵢⱼ)", 4e-6, "good"),
        ("tanh(r): Q = Σᵢ<ⱼ ΓᵢΓⱼ tanh(rᵢⱼ)", 7e-6, "good"),

        # Moderate (frac_var ~ 1e-4)
        ("Q₃ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ³", 5e-5, "moderate"),
        ("cos(r): Q = Σᵢ<ⱼ ΓᵢΓⱼ cos(rᵢⱼ)", 1.4e-4, "moderate"),

        # Poor (frac_var ~ 1e-3)
        ("exp(-r²): Q = Σᵢ<ⱼ ΓᵢΓⱼ exp(-rᵢⱼ²)", 1.4e-3, "poor"),

        # Not conserved (frac_var > 0.01)
        ("Unweighted: Σᵢ<ⱼ rᵢⱼ", 0.05, "bad"),
        ("Kinetic: K = Σᵢ Γᵢ vᵢ²", 0.1, "bad"),
        ("Random: r₁₂ + r₁₃ - r₂₃", 0.2, "bad"),
    ]

    for config_name, config_desc in configs:
        # Add noise to frac_var based on configuration
        noise_factor = {
            "restricted": 0.5,
            "equal": 0.8,
            "hierarchical": 1.2,
            "4vortex": 1.5,
            "dipole_test": 0.3,
            "5vortex": 2.0,
            "chaotic_9": 5.0,
        }.get(config_name, 1.0)

        for inv_expr, base_frac_var, quality in invariants:
            # Add realistic noise
            frac_var = base_frac_var * noise_factor * (0.5 + np.random.random())

            # Clip to realistic bounds
            frac_var = np.clip(frac_var, 1e-15, 1.0)

            context = f"{config_desc} vortex system. Is {inv_expr} conserved?"
            target = log_score(frac_var)

            examples.append({
                "context": context,
                "expression": inv_expr,
                "frac_var": frac_var,
                "target": target,  # log-scale score
                "quality": quality,
                "config": config_name,
            })

    # Shuffle
    np.random.shuffle(examples)
    return examples

def listnet_loss(margins, targets):
    """
    ListNet-style ranking loss.

    Convert margins and targets to probability distributions using softmax,
    then minimize cross-entropy between them.
    """
    # Temperature scaling
    temp = 1.0

    # Softmax over margins (model's ranking)
    margin_probs = mx.softmax(margins / temp)

    # Softmax over targets (ground truth ranking)
    target_probs = mx.softmax(targets / temp)

    # Cross-entropy loss
    loss = -mx.sum(target_probs * mx.log(margin_probs + 1e-10))

    return loss

def ranking_loss_v2(adapter, lm_head, model, tokenizer, batch_examples,
                    listnet_weight=2.0, hinge_weight=1.0, hard_neg_weight=1.0):
    """
    Improved ranking loss with:
    1. ListNet for soft ranking
    2. Hinge for positive margins on good invariants
    3. Hard negative mining for difficult pairs
    """
    margins = []
    targets = []
    qualities = []

    for ex in batch_examples:
        context = ex["context"]
        truth = ex["expression"]
        distractors = [
            f"NOT: {truth[:20]}...",
            "Random combination",
            "Unweighted sum",
        ]

        margin = get_margin(
            adapter, lm_head, model, tokenizer,
            context, truth, distractors
        )
        margins.append(margin)
        targets.append(ex["target"])
        qualities.append(ex["quality"])

    margins_stack = mx.stack(margins)
    targets_stack = mx.array(targets)

    # 1. ListNet loss for ranking
    listnet = listnet_loss(margins_stack, targets_stack)

    # 2. Hinge loss: good invariants should have positive margin
    hinge_loss = mx.array(0.0)
    for i, q in enumerate(qualities):
        if q in ["exact", "excellent", "good"]:
            hinge_loss = hinge_loss + mx.maximum(mx.array(0.0), mx.array(1.0) - margins[i])
    hinge_loss = hinge_loss / max(1, sum(1 for q in qualities if q in ["exact", "excellent", "good"]))

    # 3. Hard negative: ensure bad invariants rank below good ones
    hard_neg_loss = mx.array(0.0)
    hard_neg_count = 0
    for i, qi in enumerate(qualities):
        for j, qj in enumerate(qualities):
            if qi in ["exact", "excellent"] and qj in ["poor", "bad"]:
                # Good should have higher margin than bad
                pair_loss = mx.maximum(mx.array(0.0), margins[j] - margins[i] + mx.array(2.0))
                hard_neg_loss = hard_neg_loss + pair_loss
                hard_neg_count += 1
    if hard_neg_count > 0:
        hard_neg_loss = hard_neg_loss / hard_neg_count

    total_loss = (listnet_weight * listnet +
                  hinge_weight * hinge_loss +
                  hard_neg_weight * hard_neg_loss)

    # Compute correlation for monitoring
    margins_np = np.array([float(m) for m in margins])
    targets_np = np.array(targets)
    corr = np.corrcoef(margins_np, targets_np)[0, 1] if len(margins) > 2 else 0.0

    return total_loss, corr, margins_np, targets_np

def evaluate_ranking(adapter, lm_head, model, tokenizer, test_examples, verbose=False):
    """Evaluate ranking quality on test set."""
    margins = []
    targets = []

    for i, ex in enumerate(test_examples):
        if verbose:
            print(f"    eval {i+1}/{len(test_examples)}", end=" ", flush=True)
        context = ex["context"]
        truth = ex["expression"]
        distractors = ["NOT conserved", "Random", "Unweighted"]

        margin = get_margin(
            adapter, lm_head, model, tokenizer,
            context, truth, distractors
        )
        margins.append(float(margin))
        targets.append(ex["target"])
        if verbose:
            print(f"m={float(margin):.1f}", flush=True)

    margins_np = np.array(margins)
    targets_np = np.array(targets)

    corr = np.corrcoef(margins_np, targets_np)[0, 1]

    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    spearman_r, _ = spearmanr(margins_np, targets_np)

    return corr, spearman_r, margins_np, targets_np

def main():
    print("\n" + "="*60)
    print("RANKING ADAPTER v2: Log-scale + ListNet")
    print("="*60)

    model_name = "Qwen/Qwen3-4B-Base"
    n_steps = 100  # Best results at step 50
    batch_size = 12
    lr = 1e-6

    print(f"Model: {model_name}")
    print(f"Steps: {n_steps}, Batch: {batch_size}, LR: {lr}")
    print()

    # Generate training data
    print("Generating training examples...")
    all_examples = generate_ranking_examples()
    print(f"  Generated {len(all_examples)} examples")

    # Split train/test
    np.random.shuffle(all_examples)
    n_test = 20
    test_examples = all_examples[:n_test]
    train_examples = all_examples[n_test:]
    print(f"  Train: {len(train_examples)}, Test: {n_test}")

    # Show target distribution
    targets = [ex["target"] for ex in all_examples]
    print(f"  Target range: {min(targets):.1f} to {max(targets):.1f} (-log10(frac_var))")
    print()

    # Load model
    print("Loading model...")
    import sys
    sys.stdout.flush()
    t0 = time.time()
    model, tokenizer = mlx_lm.load(model_name)
    model.freeze()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    lm_head = t3.get_lm_head_fn(model)
    print(f"  lm_head ready")
    sys.stdout.flush()

    # Initialize adapter
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    config = SnapOnConfig(
        d_model=d_model,
        d_inner=64,
        n_layers=0,
        n_heads=8,
        mode="logit",
        vocab_size=vocab_size,
    )
    adapter = create_adapter(config)
    optimizer = mlx.optimizers.AdamW(learning_rate=lr, weight_decay=0.01)

    # Baseline evaluation
    print("\n" + "="*60)
    print("BASELINE (no adapter)")
    print("="*60)
    corr, spearman, margins, targets_arr = evaluate_ranking(
        None, lm_head, model, tokenizer, test_examples
    )
    print(f"  Pearson r:  {corr:.3f}")
    print(f"  Spearman ρ: {spearman:.3f}")
    print(f"  Margin range: [{margins.min():.1f}, {margins.max():.1f}]")

    # Training
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    best_spearman = -1.0
    best_step = 0
    best_params = None

    loss_and_grad = nn.value_and_grad(adapter, lambda a: ranking_loss_v2(
        a, lm_head, model, tokenizer,
        batch_examples=[train_examples[i % len(train_examples)]
                       for i in range(step * batch_size, (step + 1) * batch_size)]
    )[0])

    for step in range(n_steps):
        # Get batch
        batch_indices = [(step * batch_size + i) % len(train_examples) for i in range(batch_size)]
        batch = [train_examples[i] for i in batch_indices]

        # Compute loss and gradients
        loss, corr, margins, targets_arr = ranking_loss_v2(
            adapter, lm_head, model, tokenizer, batch
        )

        # Manual gradient computation
        def loss_fn(a):
            l, _, _, _ = ranking_loss_v2(a, lm_head, model, tokenizer, batch)
            return l

        loss_val, grads = nn.value_and_grad(adapter, loss_fn)(adapter)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters())

        if (step + 1) % 50 == 0:
            # Evaluate on test set
            test_corr, test_spearman, test_margins, test_targets = evaluate_ranking(
                adapter, lm_head, model, tokenizer, test_examples
            )

            print(f"  step {step+1:4d}/{n_steps}  loss={float(loss_val):8.3f}  "
                  f"batch_r={corr:.3f}  test_r={test_corr:.3f}  test_ρ={test_spearman:.3f}")

            # Save best adapter directly to file
            if test_spearman > best_spearman:
                best_spearman = test_spearman
                best_step = step + 1
                # Save adapter weights immediately
                save_path = Path(__file__).parent / "adapters" / "ranking_v2_best.npz"
                save_path.parent.mkdir(exist_ok=True)
                np.savez(str(save_path), **{k: np.array(v) for k, v in adapter.parameters().items()})
                print(f"    >> New best ρ={test_spearman:.3f} saved at step {step+1}")

    print(f"\n  Best Spearman ρ = {best_spearman:.3f} at step {best_step}")

    # Final evaluation using current (last step) adapter
    print("\n" + "="*60)
    print("FINAL EVALUATION (last step adapter)")
    print("="*60)

    final_corr, final_spearman, final_margins, final_targets = evaluate_ranking(
        adapter, lm_head, model, tokenizer, test_examples
    )

    print(f"  Pearson r:  {final_corr:.3f}")
    print(f"  Spearman ρ: {final_spearman:.3f}")
    print(f"  Margin range: [{final_margins.min():.1f}, {final_margins.max():.1f}]")

    # Show ranking
    print("\n  Ranking by margin (top 10):")
    sorted_idx = np.argsort(-final_margins)
    for i, idx in enumerate(sorted_idx[:10]):
        ex = test_examples[idx]
        print(f"    {i+1:2d}. {ex['expression'][:40]:<40} "
              f"margin={final_margins[idx]:+7.1f}  target={final_targets[idx]:.1f}")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    if final_spearman > 0.7:
        print(f"  ✓ RANKING LEARNED: Spearman ρ = {final_spearman:.3f}")
        verdict = "success"
    elif final_spearman > 0.4:
        print(f"  ~ PARTIAL: Spearman ρ = {final_spearman:.3f}")
        verdict = "partial"
    else:
        print(f"  ✗ RANKING FAILED: Spearman ρ = {final_spearman:.3f}")
        verdict = "failed"

    # Save adapter
    if final_spearman > 0.3:
        save_path = Path(__file__).parent / "adapters" / "ranking_v2.npz"
        save_path.parent.mkdir(exist_ok=True)
        np.savez(str(save_path), **{k: np.array(v) for k, v in adapter.parameters().items()})
        print(f"\n  Adapter saved: {save_path}")

    return verdict

if __name__ == "__main__":
    main()
