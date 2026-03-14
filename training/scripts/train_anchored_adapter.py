#!/usr/bin/env python3
"""
Anchored adapter training - protects passing facts while learning new ones.

Key features:
- Anchor loss: penalizes regression on already-passing facts
- Example ordering: protection examples first, new facts last
- Progressive LR with anchor weighting

Usage:
    python train_anchored_adapter.py --data ../stage3_protected.json \
        --base ../../adapters/hamiltonian_stage2.npz --anchor-weight 3.0
"""

import argparse
import json
import os
import sys
import time
import random

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


def load_training_data(path):
    """Load examples, separating anchor (protection) from new facts."""
    with open(path) as f:
        data = json.load(f)

    anchor_examples = []
    new_examples = []

    for ex in data.get("examples", []):
        ctx = ex["context"]
        truth = ex.get("truth", "")
        distractors = ex.get("distractors", ["Unknown", "N/A", "Undefined"])
        is_anchor = ex.get("_anchor", False)

        example = (ctx, truth, distractors)
        if is_anchor:
            anchor_examples.append(example)
        else:
            new_examples.append(example)

    return anchor_examples, new_examples, data.get("stage", 3)


def load_base_adapter(path, d_model, vocab_size):
    cfg = SnapOnConfig(d_model=d_model, d_inner=64, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    weights = dict(np.load(path))
    mlx_weights = {k: mx.array(v) for k, v in weights.items()}
    adapter.load_weights(list(mlx_weights.items()))
    return adapter


def compute_margin(adapter, lm_head, model, prompt, truth, distractors, tokenizer):
    """Compute margin without gradients (for monitoring)."""
    def lp(text):
        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(prompt + text)
        n_prompt = len(prompt_ids)
        if len(full_ids) <= n_prompt:
            return -1e9
        tokens = mx.array(full_ids)[None, :]
        h = model.model(tokens)
        base_logits = lm_head(h)
        shifts = adapter(base_logits)
        shifts = shifts - shifts.mean(axis=-1, keepdims=True)
        logits = base_logits + shifts
        logits = t3.LOGIT_SOFTCAP * mx.tanh(logits / t3.LOGIT_SOFTCAP)
        total = 0.0
        for i, tok_id in enumerate(full_ids[n_prompt:]):
            pos = n_prompt - 1 + i
            lv = logits[0, pos]
            lse = float(mx.log(mx.sum(mx.exp(lv - mx.max(lv))) + 1e-8) + mx.max(lv))
            total += float(lv[tok_id]) - lse
        return total

    truth_lp = lp(f" {truth}")
    dist_lps = [lp(f" {d}") for d in distractors]
    return truth_lp - max(dist_lps)


def mc_hinge_loss(adapter, lm_head, model, prompt, truth, distractors, tokenizer,
                  margin_target=2.0):
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
    margin = truth_lp - best_dist
    loss = mx.maximum(mx.array(0.0), mx.array(margin_target) - margin)
    return loss, float(margin)


def anchor_loss(adapter, lm_head, model, prompt, truth, distractors, tokenizer,
                anchor_weight=3.0, margin_floor=0.5):
    """
    Anchor loss: heavily penalize regression below margin_floor.

    If margin drops below margin_floor, apply anchor_weight penalty.
    This protects already-passing facts from regressing.
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
    dist_lps = [lp(f" {d}") for d in distractors]
    best_dist = mx.max(mx.stack(dist_lps))
    margin = truth_lp - best_dist

    # Anchor loss: penalize if margin drops below floor
    # Loss = anchor_weight * max(0, margin_floor - margin)
    regression_penalty = mx.maximum(mx.array(0.0), mx.array(margin_floor) - margin)
    loss = mx.array(anchor_weight) * regression_penalty

    return loss, float(margin)


def clip_grads(grads, max_norm=0.5):
    leaves = tree_flatten(grads)
    total_sq = sum(float(mx.sum(g ** 2)) for _, g in leaves)
    norm = total_sq ** 0.5
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        leaves = [(k, g * scale) for k, g in leaves]
    return tree_unflatten(leaves)


def train_anchored(model, tokenizer, lm_head, anchor_examples, new_examples,
                   adapter=None, steps=1200, lr=1e-6, anchor_weight=3.0,
                   d_inner=64, margin_target=1.5, anchor_ratio=0.8):
    """
    Training with anchor protection.

    First anchor_ratio of steps: mostly anchor examples (protect)
    Last (1-anchor_ratio) of steps: mix anchor + new examples
    """
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]

    if adapter is None:
        cfg = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                           n_heads=8, mode="logit", vocab_size=vocab_size)
        adapter = create_adapter(cfg)

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    hinge_loss_grad = mx.value_and_grad(mc_hinge_loss, argnums=0)
    anchor_loss_grad = mx.value_and_grad(anchor_loss, argnums=0)

    protection_steps = int(steps * anchor_ratio)

    print(f"\n  Anchored training: {len(anchor_examples)} anchor + {len(new_examples)} new")
    print(f"  Steps: {steps} (first {protection_steps} protection-heavy)")
    print(f"  LR: {lr}, Anchor weight: {anchor_weight}")

    t0 = time.time()
    recent_margins = []
    anchor_margins = []

    for step in range(steps):
        # Decide example type based on phase
        if step < protection_steps:
            # Protection phase: 90% anchor, 10% new
            use_anchor = random.random() < 0.9
        else:
            # Integration phase: 60% anchor, 40% new
            use_anchor = random.random() < 0.6

        if use_anchor and anchor_examples:
            ex = anchor_examples[step % len(anchor_examples)]
            # Use anchor loss (penalize regression)
            (loss_val, margin_val), grads = anchor_loss_grad(
                adapter, lm_head, model, ex[0] + ":", ex[1], ex[2],
                tokenizer, anchor_weight=anchor_weight
            )
            anchor_margins.append(margin_val)
        else:
            ex = new_examples[step % len(new_examples)] if new_examples else anchor_examples[0]
            # Use hinge loss (learn new facts)
            (loss_val, margin_val), grads = hinge_loss_grad(
                adapter, lm_head, model, ex[0] + ":", ex[1], ex[2],
                tokenizer, margin_target=margin_target
            )

        grads = clip_grads(grads, max_norm=0.3)  # Tighter clipping
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        recent_margins.append(margin_val)
        if len(recent_margins) > 50:
            recent_margins.pop(0)

        if (step + 1) % 150 == 0:
            elapsed = time.time() - t0
            avg_margin = np.mean(recent_margins)
            avg_anchor = np.mean(anchor_margins[-50:]) if anchor_margins else 0
            phase = "PROTECT" if step < protection_steps else "INTEGRATE"
            print(f"    step {step+1:4d}/{steps} [{phase}]  "
                  f"loss={float(loss_val):.3f}  margin={margin_val:.2f}  "
                  f"avg={avg_margin:.2f}  anchor_avg={avg_anchor:.2f}  {elapsed:.0f}s")

    return adapter


def eval_on_hamiltonian_facts(adapter, lm_head, model, tokenizer, label=""):
    facts_path = os.path.join(ROOT, "problems", "hamiltonian_facts.json")
    with open(facts_path) as f:
        data = json.load(f)
    facts = data["facts"]

    print(f"\n  hamiltonian_facts.json evaluation ({label})")
    wins, margins = 0, []

    symplectic_ids = ["ham02_liouville", "ham03_symplectic", "ham04_poincare",
                      "ham12_ergodic", "ham13_canonical"]
    stage2_ids = ["ham05_noether", "ham16_poisson"]
    new_ids = ["ham01_energy", "ham09_action", "ham15_integrable"]

    cluster_wins = {"symplectic": 0, "stage2": 0, "new": 0, "other": 0}

    for fact in facts:
        ctx, truth, distractors = fact["context"], fact["truth"], fact["distractors"]
        margin = compute_margin(adapter, lm_head, model, ctx + ":", truth, distractors, tokenizer)
        win = margin > 0
        wins += int(win)
        margins.append(margin)

        fid = fact["id"]
        if fid in symplectic_ids:
            cluster_wins["symplectic"] += int(win)
            tag = "S"
        elif fid in stage2_ids:
            cluster_wins["stage2"] += int(win)
            tag = "2"
        elif fid in new_ids:
            cluster_wins["new"] += int(win)
            tag = "N"
        else:
            cluster_wins["other"] += int(win)
            tag = " "

        marker = "+" if win else "-"
        print(f"    {marker} [{tag}] {fid:25s} margin={margin:+.2f}")

    mean_m = float(np.mean(margins))
    print(f"\n  Total: {wins}/{len(facts)}  mean_margin={mean_m:+.2f}")
    print(f"  Symplectic (protect): {cluster_wins['symplectic']}/5")
    print(f"  Stage 2 (protect): {cluster_wins['stage2']}/2")
    print(f"  New targets: {cluster_wins['new']}/3")
    return wins, len(facts), mean_m, cluster_wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--base", required=True, help="Base adapter (Stage 2)")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--anchor-weight", type=float, default=3.0)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    anchor_examples, new_examples, stage = load_training_data(args.data)
    print(f"\nStage {stage} (Anchored): {len(anchor_examples)} anchor, {len(new_examples)} new")

    if args.out is None:
        args.out = os.path.join(OUT_DIR, f"hamiltonian_stage{stage}_anchored.npz")

    print(f"\nLoading {args.model}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print(f"\nLoading base adapter: {args.base}")
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    base_adapter = load_base_adapter(args.base, d_model, vocab_size)

    print("\n" + "="*60)
    print("  STAGE 2 BASELINE (before anchored training)")
    print("="*60)
    eval_on_hamiltonian_facts(base_adapter, lm_head, model, tokenizer, label="stage2")

    adapter = train_anchored(
        model, tokenizer, lm_head, anchor_examples, new_examples,
        adapter=base_adapter, steps=args.steps, lr=args.lr,
        anchor_weight=args.anchor_weight, margin_target=args.margin,
    )

    print("\n" + "="*60)
    print(f"  AFTER ANCHORED STAGE {stage}")
    print("="*60)
    eval_on_hamiltonian_facts(adapter, lm_head, model, tokenizer, label=f"stage{stage}_anchored")

    weights = dict(tree_flatten(adapter.parameters()))
    mx.savez(args.out, **weights)
    print(f"\n  Anchored adapter saved: {args.out}")


if __name__ == "__main__":
    main()
