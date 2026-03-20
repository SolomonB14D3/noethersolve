#!/usr/bin/env python3
"""
Simple adapter training from facts file.
Used by adapter_trainer.py to train 4B adapters on failing domains.

Usage:
    python scripts/train_from_facts.py --facts problems/chemical_conservation_facts.json \
                                       --model Qwen/Qwen3-4B-Base \
                                       --output adapters/chemical_adapter.npz
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve import train_utils as t3


def load_facts(facts_file: str) -> list:
    """Load facts from JSON and convert to training examples."""
    with open(facts_file) as f:
        data = json.load(f)

    facts = data.get("facts", data.get("verifications", []))
    examples = []

    for fact in facts:
        context = fact.get("context", "")
        truth = fact.get("truth", fact.get("fact", ""))
        distractors = fact.get("distractors", [])

        if context and truth and distractors:
            examples.append((context, truth, distractors))

    return examples


def mc_hinge_loss(adapter, lm_head, model, prompt, truth, distractors, tokenizer, margin_target=1.5):
    """Differentiable hinge loss: max(0, margin_target - (truth_lp - best_dist_lp))."""
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


def eval_facts(adapter, lm_head, model, examples, tokenizer, label=""):
    """Quick eval: count how many facts the adapter gets right."""
    correct = 0
    margins = []
    for context, truth, distractors in examples:
        prompt = context if context.endswith(":") else context + ":"

        def lp(text, use_adapter=True):
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

        truth_lp = lp(f" {truth}")
        best_dist = max([lp(f" {d}") for d in distractors], default=-999)
        margin = truth_lp - best_dist
        margins.append(margin)
        if margin > 0:
            correct += 1

    avg_margin = np.mean(margins) if margins else 0
    print(f"  {label}: {correct}/{len(examples)} ({100*correct/len(examples):.1f}%)  avg_margin={avg_margin:.2f}")
    return correct, margins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", required=True, help="Facts JSON file")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--output", required=True, help="Output adapter path")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--margin", type=float, default=1.5)
    args = parser.parse_args()

    # Load facts
    examples = load_facts(args.facts)
    print(f"Loaded {len(examples)} training examples from {args.facts}")

    if len(examples) < 3:
        print("ERROR: Need at least 3 examples to train")
        sys.exit(1)

    # Load model
    print(f"Loading {args.model}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"  Loaded in {time.time()-t0:.1f}s  d_model={d_model} vocab={vocab_size}")

    # Create adapter
    config = SnapOnConfig(d_model=d_model, d_inner=args.d_inner, n_layers=0,
                          n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(config)

    # Baseline
    print("\n=== BASELINE ===")
    eval_facts(None, lm_head, model, examples, tokenizer, "baseline")

    # Train with proper differentiable loss
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
        if len(recent_margins) > 50:
            recent_margins.pop(0)

        if (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            avg = np.mean(recent_margins)
            print(f"  step {step+1:4d}  loss={float(loss_val):.3f}  margin={margin_val:.2f}  avg={avg:.2f}  {elapsed:.0f}s")

    # Final eval
    print("\n=== FINAL ===")
    eval_facts(adapter, lm_head, model, examples, tokenizer, "adapted")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    flat = tree_flatten(adapter.parameters())
    np.savez(args.output, **{k: np.array(v) for k, v in flat})
    print(f"Saved adapter to {args.output}")


if __name__ == "__main__":
    main()
