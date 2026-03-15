#!/usr/bin/env python3
"""
Prior-breaking adapter training.

Phase 0: Break the model's confident wrong priors by penalizing wrong completions.
This clears the ground before anchored training can plant correct facts.

The insight: If the base model strongly prefers wrong answers, the adapter
must first overcome that headwind before learning correct answers.

Usage:
    python train_prior_breaker.py --facts ../../problems/ns_regularity_facts.json \
        --out ../../adapters/ns_prior_broken.npz
"""

import argparse
import json
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
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


def load_facts(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("facts", data.get("examples", []))


def compute_margin_no_adapter(model, lm_head, tokenizer, prompt, truth, distractors):
    """Compute margin using base model only."""
    def lp(text):
        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(prompt + text)
        n_prompt = len(prompt_ids)
        if len(full_ids) <= n_prompt:
            return -1e9
        tokens = mx.array(full_ids)[None, :]
        h = model.model(tokens)
        logits = lm_head(h)
        total = 0.0
        for i, tok_id in enumerate(full_ids[n_prompt:]):
            pos = n_prompt - 1 + i
            lv = logits[0, pos]
            lse = float(mx.log(mx.sum(mx.exp(lv - mx.max(lv))) + 1e-8) + mx.max(lv))
            total += float(lv[tok_id]) - lse
        return total

    truth_lp = lp(f" {truth}")
    dist_lps = [(lp(f" {d}"), d) for d in distractors]
    best_dist_lp, best_dist = max(dist_lps, key=lambda x: x[0])
    return truth_lp - best_dist_lp, best_dist


def prior_break_loss(adapter, lm_head, model, prompt, truth, wrong_preferred, tokenizer,
                     break_margin=5.0):
    """
    Prior-breaking loss: penalize when wrong answer scores too high.
    Loss = max(0, wrong_lp + break_margin - truth_lp)
    """
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
    wrong_lp = lp(f" {wrong_preferred}")
    margin = truth_lp - wrong_lp
    loss = mx.maximum(mx.array(0.0), mx.array(break_margin) - margin)
    return loss, float(margin)


def clip_grads(grads, max_norm=0.5):
    leaves = tree_flatten(grads)
    total_sq = sum(float(mx.sum(g ** 2)) for _, g in leaves)
    norm = total_sq ** 0.5
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        leaves = [(k, g * scale) for k, g in leaves]
    return tree_unflatten(leaves)


def train_prior_breaker(model, tokenizer, lm_head, facts,
                        steps=800, lr=2e-6, break_margin=5.0, d_inner=64):
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]

    cfg = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)

    print("\n  Identifying confident wrong priors...")
    wrong_priors = []
    for fact in facts:
        ctx = fact["context"]
        truth = fact.get("truth", fact.get("completion", ""))
        distractors = fact.get("distractors", [])
        margin, preferred_wrong = compute_margin_no_adapter(
            model, lm_head, tokenizer, ctx + ":", truth, distractors
        )
        if margin < -10:
            wrong_priors.append({
                "context": ctx, "truth": truth,
                "wrong_preferred": preferred_wrong, "base_margin": margin
            })
            print(f"    {fact.get('id', '?'):25s} m={margin:+.1f} prefers: {preferred_wrong[:35]}...")

    if not wrong_priors:
        print("  No confident wrong priors found!")
        return adapter

    print(f"\n  Breaking {len(wrong_priors)} priors, {steps} steps, lr={lr}")

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)
    loss_and_grad = mx.value_and_grad(prior_break_loss, argnums=0)

    t0 = time.time()
    recent_margins = []

    for step in range(steps):
        prior = wrong_priors[step % len(wrong_priors)]
        (loss_val, margin_val), grads = loss_and_grad(
            adapter, lm_head, model, prior["context"] + ":",
            prior["truth"], prior["wrong_preferred"], tokenizer, break_margin
        )
        grads = clip_grads(grads, max_norm=0.5)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        recent_margins.append(margin_val)
        if len(recent_margins) > 50: recent_margins.pop(0)

        if (step + 1) % 100 == 0:
            print(f"    step {step+1:4d}/{steps}  loss={float(loss_val):.3f}  "
                  f"margin={margin_val:+.2f}  avg={np.mean(recent_margins):+.2f}  {time.time()-t0:.0f}s")

    return adapter


def eval_on_facts(adapter, lm_head, model, tokenizer, facts, label=""):
    print(f"\n  Evaluation ({label})")
    wins, margins = 0, []
    for fact in facts:
        ctx, truth = fact["context"], fact.get("truth", "")
        distractors = fact.get("distractors", [])
        prompt = ctx + ":"
        def lp(text):
            p_ids, f_ids = tokenizer.encode(prompt), tokenizer.encode(prompt + text)
            n = len(p_ids)
            if len(f_ids) <= n: return -999.0
            h = model.model(mx.array(f_ids)[None, :])
            bl = lm_head(h)
            if adapter:
                sh = adapter(bl) - adapter(bl).mean(axis=-1, keepdims=True)
                lo = t3.LOGIT_SOFTCAP * mx.tanh((bl + sh) / t3.LOGIT_SOFTCAP)
            else:
                lo = bl
            return sum(float(np.array(lo[0,n-1+i].astype(mx.float32))[tid]) -
                       (np.log(np.sum(np.exp(np.array(lo[0,n-1+i].astype(mx.float32)) -
                       np.array(lo[0,n-1+i].astype(mx.float32)).max()))+1e-8) +
                       np.array(lo[0,n-1+i].astype(mx.float32)).max())
                       for i, tid in enumerate(f_ids[n:]))
        tl, dl = lp(f" {truth}"), [lp(f" {d}") for d in distractors]
        w, m = tl > max(dl), tl - max(dl)
        wins += int(w); margins.append(m)
        print(f"    {'+' if w else '-'} {fact.get('id','?'):25s} m={m:+.1f}")
    print(f"\n  Total: {wins}/{len(facts)}  mean={np.mean(margins):+.1f}")
    return wins, len(facts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--break-margin", type=float, default=5.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    facts = load_facts(args.facts)
    print(f"\nLoaded {len(facts)} facts")

    if args.out is None:
        bn = os.path.basename(args.facts).replace("_facts.json", "").replace(".json", "")
        args.out = os.path.join(OUT_DIR, f"{bn}_prior_broken.npz")

    print(f"\nLoading {args.model}...")
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)

    print("\n" + "="*60 + "\n  BASE MODEL\n" + "="*60)
    eval_on_facts(None, lm_head, model, tokenizer, facts, "base")

    adapter = train_prior_breaker(model, tokenizer, lm_head, facts,
                                  args.steps, args.lr, args.break_margin)

    print("\n" + "="*60 + "\n  AFTER PRIOR BREAKING\n" + "="*60)
    eval_on_facts(adapter, lm_head, model, tokenizer, facts, "prior_broken")

    weights = dict(tree_flatten(adapter.parameters()))
    mx.savez(args.out, **weights)
    print(f"\n  Saved: {args.out}")


if __name__ == "__main__":
    main()
