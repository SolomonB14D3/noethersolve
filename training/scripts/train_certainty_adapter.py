#!/usr/bin/env python3
"""Train certainty decontamination adapters with correct SnapOnLogitMLP architecture.

Uses the same training data format as certainty_decontamination.json:
  [{"context": ..., "truth": ..., "distractor": ...}, ...]

Trains SnapOnLogitMLP (gate/up/down_proj) to prefer hedged truths over
definitive distractors, fixing the certainty contamination bias.
"""

import argparse
import json
import os
import sys
import time

import mlx.core as mx
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from noethersolve.adapter import SnapOnConfig, SnapOnLogitMLP
from noethersolve import train_utils as t3


def load_training_data(path):
    """Load certainty training data.

    Supports both formats:
    - {context, truth, distractor}
    - {context, positive, negative}
    """
    with open(path) as f:
        data = json.load(f)
    examples = []
    for ex in data:
        context = ex["context"]
        # Handle both naming conventions
        truth = ex.get("truth") or ex.get("positive")
        distractor = ex.get("distractor") or ex.get("negative")
        distractors = [distractor] if distractor else ex.get("distractors", [])
        if truth and distractors:
            examples.append((context, truth, distractors))
    return examples


def mc_hinge_loss(adapter, lm_head, model, prompt, truth, distractors, tokenizer, margin_target=1.5):
    """Hinge loss: max(0, margin_target - (truth_lp - best_distractor_lp))."""
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
    best_dist = mx.max(mx.stack(dist_lps)) if dist_lps else truth_lp - margin_target - 1
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


def eval_adapter(adapter, lm_head, model, examples, tokenizer, label=""):
    """Evaluate adapter on examples."""
    def lp(prompt, text, use_adapter=True):
        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(prompt + text)
        n_prompt = len(prompt_ids)
        if len(full_ids) <= n_prompt:
            return -1e9
        tokens = mx.array(full_ids)[None, :]
        h = model.model(tokens)
        base_logits = lm_head(h)
        if use_adapter and adapter is not None:
            shifts = adapter(base_logits)
            shifts = shifts - shifts.mean(axis=-1, keepdims=True)
            logits = base_logits + shifts
        else:
            logits = base_logits
        logits = t3.LOGIT_SOFTCAP * mx.tanh(logits / t3.LOGIT_SOFTCAP)
        total = 0.0
        for i, tok_id in enumerate(full_ids[n_prompt:]):
            pos = n_prompt - 1 + i
            lv = logits[0, pos]
            lse = float(mx.log(mx.sum(mx.exp(lv - mx.max(lv))) + 1e-8) + mx.max(lv))
            total += float(lv[tok_id]) - lse
        return total

    correct = 0
    margins = []
    for ctx, truth, distractors in examples:
        truth_lp = lp(ctx, f" {truth}")
        best_dist = max([lp(ctx, f" {d}") for d in distractors], default=-999)
        margin = truth_lp - best_dist
        margins.append(margin)
        if margin > 0:
            correct += 1

    print(f"  {label}: {correct}/{len(examples)} ({100*correct/len(examples):.1f}%)  mean_margin={np.mean(margins):.2f}")
    return correct, margins


def save_adapter(adapter, path):
    """Save adapter in the correct format for SnapOnLogitMLP."""
    flat = tree_flatten(adapter.parameters())
    np.savez(path, **{k: np.array(v) for k, v in flat})
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Training data JSON")
    parser.add_argument("--out", default="adapters/certainty_decontamination_adapter.npz")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--margin", type=float, default=1.5)
    args = parser.parse_args()

    # Load data
    examples = load_training_data(args.data)
    print(f"Loaded {len(examples)} training examples")

    # Load model
    print("\nLoading Qwen/Qwen3-4B-Base...")
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]
    print(f"  vocab={vocab_size}")

    # Create adapter (SnapOnLogitMLP with gate/up/down_proj)
    cfg = SnapOnConfig(d_inner=args.d_inner, mode="logit", vocab_size=vocab_size)
    adapter = SnapOnLogitMLP(cfg)
    mx.eval(adapter.parameters())

    # Baseline
    print("\n=== BASELINE ===")
    eval_adapter(None, lm_head, model, examples[:30], tokenizer, "baseline")

    # Training
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
    loss_and_grad = mx.value_and_grad(mc_hinge_loss, argnums=0)

    print(f"\n=== TRAINING ({args.steps} steps) ===")
    t0 = time.time()
    recent_margins = []

    for step in range(args.steps):
        ctx, truth, distractors = examples[step % len(examples)]
        prompt = ctx if ctx.endswith(":") else ctx + ":"

        (loss_val, margin_val), grads = loss_and_grad(
            adapter, lm_head, model, prompt, truth, distractors, tokenizer, args.margin
        )
        grads = clip_grads(grads)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)
        recent_margins.append(margin_val)

        if (step + 1) % 100 == 0:
            avg = np.mean(recent_margins[-100:])
            print(f"  step {step+1:4d}  avg_margin={avg:.2f}  loss={float(loss_val):.3f}")

    print(f"  Training time: {time.time()-t0:.1f}s")

    # Final eval
    print("\n=== FINAL ===")
    eval_adapter(adapter, lm_head, model, examples[:30], tokenizer, "adapter")

    # Save
    save_adapter(adapter, args.out)


if __name__ == "__main__":
    main()
