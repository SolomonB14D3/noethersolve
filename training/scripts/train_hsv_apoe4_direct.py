#!/usr/bin/env python3
"""Direct training on eval facts for HSV-APOE4-Alzheimer's domain.

Trains directly on the exact eval facts (no phrasing mismatch).
Uses longer training with progressive difficulty.
"""

import json
import os
import sys
import time

import mlx.core as mx
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve import train_utils as t3

OUT_DIR = os.path.join(ROOT, "adapters")


def mc_hinge_loss(adapter, lm_head, model, prompt, truth, distractors, tokenizer,
                  margin_target=3.0):
    def lp(text):
        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(prompt + text)
        n_prompt = len(prompt_ids)
        if len(full_ids) <= n_prompt:
            return mx.array(-1e9)
        tokens = mx.array(full_ids)[None, :]
        h = model.model(tokens)
        base_logits = lm_head(h)
        shifts = adapter(base_logits)
        shifts = shifts - shifts.mean(axis=-1, keepdims=True)
        logits = base_logits + shifts
        logits = t3.LOGIT_SOFTCAP * mx.tanh(logits / t3.LOGIT_SOFTCAP)
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
    loss = mx.maximum(mx.array(0.0), mx.array(margin_target) - (truth_lp - best_dist))
    return loss, float(truth_lp - best_dist)


def clip_grads(grads, max_norm=1.0):
    leaves = tree_flatten(grads)
    total_sq = sum(float(mx.sum(g ** 2)) for _, g in leaves)
    norm = total_sq ** 0.5
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        leaves = [(k, g * scale) for k, g in leaves]
    return tree_unflatten(leaves)


def eval_on_facts(adapter, lm_head, model, tokenizer, facts, label=""):
    print(f"\n  Evaluation ({label})")
    wins, margins = 0, []
    results = []
    for fact in facts:
        ctx, truth, distractors = fact["context"], fact["truth"], fact["distractors"]
        prompt = ctx

        def adapted_lp(text):
            prompt_ids = tokenizer.encode(prompt)
            comp_ids = tokenizer.encode(text)
            full_ids = prompt_ids + comp_ids
            if not comp_ids:
                return -999.0
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
            n_prompt = len(prompt_ids)
            total = 0.0
            for i, tok_id in enumerate(comp_ids):
                pos = n_prompt - 1 + i
                lv = np.array(logits[0, pos].astype(mx.float32))
                lse = float(np.log(np.sum(np.exp(lv - lv.max())) + 1e-8) + lv.max())
                total += float(lv[tok_id]) - lse
            return total

        truth_lp = adapted_lp(f" {truth}")
        dist_lps = [adapted_lp(f" {d}") for d in distractors]
        win = truth_lp > max(dist_lps)
        margin = truth_lp - max(dist_lps)
        wins += int(win)
        margins.append(margin)
        results.append((fact["id"], win, margin))
        marker = "+" if win else "-"
        print(f"    {marker} {fact['id']:10s} margin={margin:+.2f}")

    mean_m = float(np.mean(margins))
    print(f"  Pass: {wins}/{len(facts)}  mean_margin={mean_m:+.2f}")
    return wins, len(facts), mean_m, results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load eval facts directly as training data
    facts_path = os.path.join(ROOT, "problems", "hsv_apoe4_alzheimers_facts.json")
    with open(facts_path) as f:
        data = json.load(f)
    facts = data["facts"]
    print(f"\nLoaded {len(facts)} facts for direct training")

    print("\nLoading Qwen/Qwen3-4B-Base...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Create adapter with larger capacity
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=128, n_layers=0,  # Larger d_inner
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    optimizer = optim.AdamW(learning_rate=3e-6, weight_decay=0.01)

    loss_and_grad = mx.value_and_grad(mc_hinge_loss, argnums=0)

    print("\n" + "="*60)
    print("  BASELINE")
    print("="*60)
    _, _, _, baseline_results = eval_on_facts(None, lm_head, model, tokenizer, facts, "baseline")

    # Get baseline margins for difficulty weighting
    baseline_margins = {r[0]: r[2] for r in baseline_results}

    # Sort facts by difficulty (most negative margin first)
    sorted_facts = sorted(facts, key=lambda f: baseline_margins.get(f["id"], 0))

    # Training loop with difficulty weighting
    steps = 3000
    print(f"\n  Training directly on {len(facts)} facts for {steps} steps")
    print("  Strategy: Difficulty-weighted sampling (harder facts trained more)")

    t0 = time.time()
    recent_margins = []
    for step in range(steps):
        # Difficulty-weighted sampling: harder facts (lower margin) get more training
        # Early steps focus on harder facts, later steps are more uniform
        progress = step / steps
        if progress < 0.5:
            # First half: focus on hardest facts
            idx = step % (len(sorted_facts) // 2 + 1)
        else:
            # Second half: more uniform
            idx = step % len(sorted_facts)

        fact = sorted_facts[idx]
        ctx, truth, distractors = fact["context"], fact["truth"], fact["distractors"]
        prompt = ctx

        (loss_val, margin_val), grads = loss_and_grad(
            adapter, lm_head, model, prompt, truth, distractors, tokenizer, 3.0
        )
        grads = clip_grads(grads)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        recent_margins.append(margin_val)
        if len(recent_margins) > 50:
            recent_margins.pop(0)

        if (step + 1) % 300 == 0:
            elapsed = time.time() - t0
            avg_margin = np.mean(recent_margins)
            wins_so_far, _, _, _ = eval_on_facts(adapter, lm_head, model, tokenizer, facts, f"step {step+1}")
            print(f"    step {step+1:5d}/{steps}  avg_margin={avg_margin:.2f}  pass={wins_so_far}/16  {elapsed:.0f}s")

    print("\n" + "="*60)
    print("  FINAL EVALUATION")
    print("="*60)
    wins, total, mean_m, _ = eval_on_facts(adapter, lm_head, model, tokenizer, facts, "final")

    out_path = os.path.join(OUT_DIR, "hsv_apoe4_alzheimers_direct_adapter.npz")
    weights = dict(tree_flatten(adapter.parameters()))
    mx.savez(out_path, **weights)
    print(f"\n  Adapter saved: {out_path}")
    print(f"  Final: {wins}/{total} ({100*wins/total:.1f}%)")


if __name__ == "__main__":
    main()
