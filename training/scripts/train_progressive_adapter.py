#!/usr/bin/env python3
"""
Progressive lens adapter training - LR decays over training.

The idea: Start with higher LR to learn new facts, decay to consolidate
and protect existing knowledge.

Progressive schedule:
- First 1/3: lr_start (focus on new facts)
- Middle 1/3: lr_start * 0.3 (balance)
- Final 1/3: lr_start * 0.1 (consolidate/protect)

Usage:
    python train_progressive_adapter.py --data ../stage4.json --base ../../adapters/stage3.npz
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve import train_utils as t3

OUT_DIR = os.path.join(ROOT, "adapters")


def load_training_data(path):
    with open(path) as f:
        data = json.load(f)
    examples = []
    for ex in data.get("examples", []):
        ctx = ex["context"]
        truth = ex.get("truth", ex.get("completion", ""))
        distractors = ex.get("distractors", ["Unknown", "Not applicable", "Undefined"])
        examples.append((ctx, truth, distractors))
    return examples, data.get("stage", 4)


def load_base_adapter(path, d_model, vocab_size):
    """Load existing adapter weights as starting point."""
    cfg = SnapOnConfig(d_model=d_model, d_inner=64, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    weights = dict(np.load(path))
    mlx_weights = {k: mx.array(v) for k, v in weights.items()}
    adapter.load_weights(list(mlx_weights.items()))
    return adapter


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
    loss = mx.maximum(mx.array(0.0), mx.array(margin_target) - (truth_lp - best_dist))
    return loss, float(truth_lp - best_dist)


def clip_grads(grads, max_norm=0.5):
    leaves = tree_flatten(grads)
    total_sq = sum(float(mx.sum(g ** 2)) for _, g in leaves)
    norm = total_sq ** 0.5
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        leaves = [(k, g * scale) for k, g in leaves]
    return tree_unflatten(leaves)


def get_progressive_lr(step, total_steps, lr_start):
    """Progressive lens: decay LR over training phases."""
    phase1_end = total_steps // 3
    phase2_end = 2 * total_steps // 3

    if step < phase1_end:
        # Phase 1: Full learning rate (focus on new facts)
        return lr_start
    elif step < phase2_end:
        # Phase 2: 30% of initial (balance)
        return lr_start * 0.3
    else:
        # Phase 3: 10% of initial (consolidate/protect)
        return lr_start * 0.1


def train_progressive(model, tokenizer, lm_head, examples, adapter=None,
                      steps=600, lr_start=1e-7, d_inner=64, margin_target=1.5):
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]

    if adapter is None:
        cfg = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                           n_heads=8, mode="logit", vocab_size=vocab_size)
        adapter = create_adapter(cfg)

    loss_and_grad = mx.value_and_grad(mc_hinge_loss, argnums=0)

    print(f"\n  Progressive training: {len(examples)} examples, "
          f"{steps} steps, lr_start={lr_start}")
    print(f"  Schedule: Phase 1 (0-{steps//3}): lr={lr_start:.1e}")
    print(f"            Phase 2 ({steps//3}-{2*steps//3}): lr={lr_start*0.3:.1e}")
    print(f"            Phase 3 ({2*steps//3}-{steps}): lr={lr_start*0.1:.1e}")

    t0 = time.time()
    recent_margins = []

    for step in range(steps):
        # Get progressive learning rate
        current_lr = get_progressive_lr(step, steps, lr_start)
        optimizer = optim.AdamW(learning_rate=current_lr, weight_decay=0.01)

        ex = examples[step % len(examples)]
        ctx, truth, distractors = ex[0], ex[1], ex[2]
        prompt = ctx + ":"

        (loss_val, margin_val), grads = loss_and_grad(
            adapter, lm_head, model, prompt, truth, distractors, tokenizer, margin_target
        )
        grads = clip_grads(grads, max_norm=0.5)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        recent_margins.append(margin_val)
        if len(recent_margins) > 50:
            recent_margins.pop(0)

        if (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            avg_margin = np.mean(recent_margins)
            phase = 1 if step < steps//3 else (2 if step < 2*steps//3 else 3)
            print(f"    step {step+1:4d}/{steps} [P{phase}] lr={current_lr:.1e}  "
                  f"loss={float(loss_val):.3f}  margin={margin_val:.3f}  "
                  f"avg={avg_margin:.3f}  {elapsed:.0f}s")

    return adapter


def eval_on_hamiltonian_facts(adapter, lm_head, model, tokenizer, label=""):
    facts_path = os.path.join(ROOT, "problems", "hamiltonian_facts.json")
    with open(facts_path) as f:
        data = json.load(f)
    facts = data["facts"]

    print(f"\n  hamiltonian_facts.json evaluation ({label})")
    wins, margins = 0, []

    # Track clusters
    symplectic_ids = ["ham02_liouville", "ham03_symplectic", "ham04_poincare",
                      "ham12_ergodic", "ham13_canonical"]
    stage3_ids = ["ham01_energy", "ham05_noether", "ham09_action",
                  "ham15_integrable", "ham16_poisson"]
    recovery_ids = ["ham04_poincare", "ham12_ergodic"]

    cluster_wins = {"symplectic": 0, "stage3": 0, "recovery": 0, "other": 0}

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

        # Track clusters
        fid = fact["id"]
        if fid in recovery_ids:
            cluster_wins["recovery"] += int(win)
            tag = "R"
        elif fid in symplectic_ids:
            cluster_wins["symplectic"] += int(win)
            tag = "S"
        elif fid in stage3_ids:
            cluster_wins["stage3"] += int(win)
            tag = "3"
        else:
            cluster_wins["other"] += int(win)
            tag = " "

        marker = "+" if win else "-"
        print(f"    {marker} [{tag}] {fid:25s} margin={margin:+.3f}")

    mean_m = float(np.mean(margins))
    print(f"\n  Total: {wins}/{len(facts)}  mean_margin={mean_m:+.3f}")
    print(f"  Recovery targets (Poincare, ergodic): {cluster_wins['recovery']}/2")
    print(f"  Symplectic cluster: {cluster_wins['symplectic']}/5")
    print(f"  Stage 3 gains: {cluster_wins['stage3']}/5")
    return wins, len(facts), mean_m, cluster_wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--base", help="Base adapter to continue from")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-7, help="Starting LR (decays progressively)")
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    examples, stage = load_training_data(args.data)
    print(f"\nStage {stage} (Progressive): Loaded {len(examples)} examples from {args.data}")

    if args.out is None:
        args.out = os.path.join(OUT_DIR, f"hamiltonian_stage{stage}_progressive.npz")

    print(f"\nLoading {args.model}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Load base adapter
    base_adapter = None
    if args.base:
        print(f"\nLoading base adapter: {args.base}")
        vocab_size = model.model.embed_tokens.weight.shape[0]
        d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
        base_adapter = load_base_adapter(args.base, d_model, vocab_size)

    print("\n" + "="*60)
    print("  BEFORE PROGRESSIVE TRAINING")
    print("="*60)
    eval_on_hamiltonian_facts(base_adapter, lm_head, model, tokenizer, label="base")

    adapter = train_progressive(
        model, tokenizer, lm_head, examples, adapter=base_adapter,
        steps=args.steps, lr_start=args.lr, margin_target=args.margin,
    )

    print("\n" + "="*60)
    print(f"  AFTER PROGRESSIVE STAGE {stage}")
    print("="*60)
    eval_on_hamiltonian_facts(adapter, lm_head, model, tokenizer, label=f"stage{stage}_prog")

    weights = dict(tree_flatten(adapter.parameters()))
    mx.savez(args.out, **weights)
    print(f"\n  Progressive adapter saved: {args.out}")


if __name__ == "__main__":
    main()
