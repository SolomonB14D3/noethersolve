#!/usr/bin/env python3
"""Quick ranking adapter training - 60 steps, minimal evaluation."""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
import mlx_lm
from pathlib import Path
from scipy.stats import spearmanr

from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve import train_utils as t3

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

def log_score(fv):
    return -np.log10(max(fv, 1e-15))

def listnet_loss(margins, targets):
    margin_probs = mx.softmax(margins)
    target_probs = mx.softmax(targets)
    return -mx.sum(target_probs * mx.log(margin_probs + 1e-10))

# Fixed test set for consistent evaluation
TEST_SET = [
    ("Q₂ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ²", 1e-12, "exact"),
    ("Q₁ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ", 5e-6, "excellent"),
    ("Q₀.₅ = Σᵢ<ⱼ ΓᵢΓⱼ √rᵢⱼ", 2e-6, "excellent"),
    ("exp(-r) weighted", 1e-5, "good"),
    ("Q₃ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ³", 5e-5, "moderate"),
    ("Unweighted Σrᵢⱼ", 0.05, "bad"),
    ("Kinetic K", 0.1, "bad"),
]

def evaluate(adapter, lm_head, model, tokenizer):
    margins, targets = [], []
    for expr, fv, _ in TEST_SET:
        ctx = f"Vortex system. Is {expr} conserved?"
        m = get_margin(adapter, lm_head, model, tokenizer, ctx, expr, ["Not conserved", "Random"])
        margins.append(float(m))
        targets.append(log_score(fv))
    return spearmanr(margins, targets)[0], margins, targets

def main():
    print("Ranking Adapter - Quick Training")
    print("="*50)

    # Load model
    print("Loading model...")
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)

    # Create adapter
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    config = SnapOnConfig(d_model=d_model, d_inner=64, n_layers=0, n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(config)
    optimizer = mlx.optimizers.AdamW(learning_rate=1e-6, weight_decay=0.01)

    # Baseline
    print("\nBaseline evaluation...")
    baseline_rho, baseline_margins, targets = evaluate(None, lm_head, model, tokenizer)
    print(f"  Baseline Spearman ρ: {baseline_rho:.3f}")

    # Generate training examples
    train_examples = []
    configs = ["restricted Γ₁=Γ₂=1,Γ₃=0.01", "equal Γ₁=Γ₂=Γ₃=1", "4-vortex", "chaotic N=9"]
    invariants = [
        ("Q₂ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ²", 1e-12), ("Q₁ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ", 5e-6),
        ("Q₀.₅ = Σᵢ<ⱼ ΓᵢΓⱼ √rᵢⱼ", 2e-6), ("exp(-r) weighted", 1e-5),
        ("Q₃ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ³", 5e-5), ("cos(r) weighted", 1.4e-4),
        ("exp(-r²) weighted", 1.4e-3), ("Unweighted Σrᵢⱼ", 0.05), ("Kinetic K", 0.1),
    ]
    for cfg in configs:
        for expr, fv in invariants:
            fv_noisy = fv * (0.5 + np.random.random())
            train_examples.append({
                "context": f"{cfg} vortex. Is {expr} conserved?",
                "expr": expr,
                "target": log_score(fv_noisy),
                "quality": "good" if fv < 1e-3 else "bad"
            })
    np.random.shuffle(train_examples)

    # Training
    print("\nTraining (60 steps)...")
    best_rho = baseline_rho
    batch_size = 8

    for step in range(60):
        batch = [train_examples[(step * batch_size + i) % len(train_examples)] for i in range(batch_size)]

        def loss_fn(a):
            margins = []
            targets = []
            for ex in batch:
                m = get_margin(a, lm_head, model, tokenizer, ex["context"], ex["expr"], ["Not conserved", "Random"])
                margins.append(m)
                targets.append(ex["target"])
            margins_stack = mx.stack(margins)
            targets_stack = mx.array(targets)
            # ListNet + hinge
            listnet = listnet_loss(margins_stack, targets_stack)
            hinge = mx.mean(mx.maximum(mx.array(0.0), mx.array(1.0) - margins_stack))
            return listnet + hinge

        loss, grads = nn.value_and_grad(adapter, loss_fn)(adapter)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters())

        if (step + 1) % 20 == 0:
            rho, margins, _ = evaluate(adapter, lm_head, model, tokenizer)
            print(f"  Step {step+1:3d}: loss={float(loss):.3f}, ρ={rho:.3f}")
            if rho > best_rho:
                best_rho = rho
                # Save best adapter
                save_path = Path(__file__).parent / "adapters" / "ranking_v2_best.npz"
                save_path.parent.mkdir(exist_ok=True)
                np.savez(str(save_path), **{k: np.array(v) for k, v in adapter.parameters().items()})
                print(f"    >> Saved best adapter (ρ={rho:.3f})")

    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    final_rho, final_margins, final_targets = evaluate(adapter, lm_head, model, tokenizer)

    print(f"\n  Baseline ρ: {baseline_rho:.3f}")
    print(f"  Final ρ:    {final_rho:.3f}")
    print(f"  Best ρ:     {best_rho:.3f}")

    print("\n  Ranking:")
    sorted_idx = np.argsort(-np.array(final_margins))
    for i, idx in enumerate(sorted_idx):
        expr, fv, q = TEST_SET[idx]
        print(f"    {i+1}. {expr:<30} margin={final_margins[idx]:+6.1f} target={final_targets[idx]:.1f} ({q})")

    if best_rho > 0.7:
        print(f"\n  ✓ RANKING LEARNED (ρ={best_rho:.3f})")
    elif best_rho > 0.4:
        print(f"\n  ~ PARTIAL SUCCESS (ρ={best_rho:.3f})")
    else:
        print(f"\n  ✗ RANKING FAILED (ρ={best_rho:.3f})")

if __name__ == "__main__":
    main()
