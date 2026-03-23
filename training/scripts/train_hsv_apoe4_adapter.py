#!/usr/bin/env python3
"""Train an HSV-APOE4-Alzheimer's adapter.

Teaches the oracle about:
- APOE4-HSV-1 gene-virus interaction (3-4x risk)
- CNS penetration differences (adibelivir vs valacyclovir)
- NLRP3 inflammasome pathway activation
- Tau's antimicrobial function
- TREM2 antiviral role

Usage:
    python train_hsv_apoe4_adapter.py
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


def train_adapter(model, tokenizer, lm_head, examples,
                  steps=2000, lr=4e-6, d_inner=64, margin_target=3.0):
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    loss_and_grad = mx.value_and_grad(mc_hinge_loss, argnums=0)

    print(f"\n  Training HSV-APOE4-Alzheimer's adapter: {len(examples)} examples, "
          f"{steps} steps, lr={lr}")

    t0 = time.time()
    recent_margins = []
    for step in range(steps):
        ex = examples[step % len(examples)]
        ctx, truth, distractors = ex["context"], ex["truth"], ex["distractors"]
        prompt = ctx + ":"

        (loss_val, margin_val), grads = loss_and_grad(
            adapter, lm_head, model, prompt, truth, distractors, tokenizer, margin_target
        )
        grads = clip_grads(grads)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        recent_margins.append(margin_val)
        if len(recent_margins) > 50:
            recent_margins.pop(0)

        if (step + 1) % 200 == 0:
            elapsed = time.time() - t0
            avg_margin = np.mean(recent_margins)
            print(f"    step {step+1:5d}/{steps}  loss={float(loss_val):.3f}  "
                  f"avg_margin={avg_margin:.3f}  {elapsed:.0f}s")

    return adapter


def eval_on_facts(adapter, lm_head, model, tokenizer, label=""):
    facts_path = os.path.join(ROOT, "problems", "hsv_apoe4_alzheimers_facts.json")
    with open(facts_path) as f:
        data = json.load(f)
    facts = data["facts"]

    print(f"\n  hsv_apoe4_alzheimers_facts.json evaluation ({label})")
    wins, margins = 0, []
    for fact in facts:
        ctx, truth, distractors = fact["context"], fact["truth"], fact["distractors"]
        prompt = ctx + ":"

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
        marker = "+" if win else "-"
        print(f"    {marker} {fact['id']:25s} margin={margin:+.3f}")

    mean_m = float(np.mean(margins))
    print(f"  Pass: {wins}/{len(facts)}  mean_margin={mean_m:+.3f}")
    return wins, len(facts), mean_m


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load training data
    data_path = os.path.join(ROOT, "training", "hsv_apoe4_alzheimers_training.json")
    with open(data_path) as f:
        data = json.load(f)
    examples = data["examples"]
    print(f"\nLoaded {len(examples)} training examples")

    print("\nLoading Qwen/Qwen3-4B-Base...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("\n" + "="*60)
    print("  BASELINE (no adapter)")
    print("="*60)
    eval_on_facts(None, lm_head, model, tokenizer, label="baseline")

    adapter = train_adapter(model, tokenizer, lm_head, examples,
                           steps=2000, lr=4e-6, margin_target=3.0)

    print("\n" + "="*60)
    print("  AFTER TRAINING")
    print("="*60)
    eval_on_facts(adapter, lm_head, model, tokenizer, label="hsv_apoe4_adapter")

    out_path = os.path.join(OUT_DIR, "hsv_apoe4_alzheimers_adapter.npz")
    weights = dict(tree_flatten(adapter.parameters()))
    mx.savez(out_path, **weights)
    print(f"\n  Adapter saved: {out_path}")


if __name__ == "__main__":
    main()
