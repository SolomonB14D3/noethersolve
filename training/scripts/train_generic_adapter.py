#!/usr/bin/env python3
"""Generic adapter training on any fact domain."""

import argparse, json, os, sys, time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve import train_utils as t3

def load_training_data(path):
    with open(path) as f:
        data = json.load(f)
    examples = []
    for ex in data.get("examples", []):
        examples.append((ex["context"], ex["truth"], ex.get("distractors", [])))
    return examples

def load_eval_facts(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("facts", [])

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

def eval_adapter(adapter, lm_head, model, facts, tokenizer, label=""):
    """Evaluate adapter on facts."""
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
    for fact in facts:
        prompt = fact["context"]
        truth = fact["truth"]
        distractors = fact.get("distractors", [])
        truth_lp = lp(prompt, f" {truth}")
        best_dist = max([lp(prompt, f" {d}") for d in distractors], default=-999)
        margin = truth_lp - best_dist
        margins.append(margin)
        if margin > 0:
            correct += 1
    
    print(f"  {label}: {correct}/{len(facts)} ({100*correct/len(facts):.1f}%)  mean_margin={np.mean(margins):.2f}")
    return correct, margins

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Training data JSON")
    parser.add_argument("--facts", required=True, help="Evaluation facts JSON")
    parser.add_argument("--out", default="adapters/generic.npz")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--margin", type=float, default=1.5)
    args = parser.parse_args()
    
    # Load data
    examples = load_training_data(args.data)
    facts = load_eval_facts(args.facts)
    print(f"Loaded {len(examples)} training examples, {len(facts)} eval facts")
    
    # Load model
    print("\nLoading Qwen/Qwen3-4B-Base...")
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"  d_model={d_model}  vocab={vocab_size}")
    
    # Create adapter
    cfg = SnapOnConfig(d_model=d_model, d_inner=args.d_inner, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    
    # Baseline
    print("\n=== BASELINE ===")
    eval_adapter(None, lm_head, model, facts, tokenizer, "baseline")
    
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
        if len(recent_margins) > 50:
            recent_margins.pop(0)
        
        if (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            avg = np.mean(recent_margins)
            print(f"  step {step+1:4d}  loss={float(loss_val):.3f}  margin={margin_val:.2f}  avg={avg:.2f}  {elapsed:.0f}s")
    
    # Final eval
    print("\n=== FINAL ===")
    eval_adapter(adapter, lm_head, model, facts, tokenizer, "adapted")
    
    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    flat = tree_flatten(adapter.parameters())
    np.savez(args.out, **{k: np.array(v) for k, v in flat})
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
