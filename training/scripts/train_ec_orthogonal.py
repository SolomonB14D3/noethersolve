#!/usr/bin/env python3
"""
Train orthogonal adapters for elliptic curve holdout facts.

Four clusters that showed interference in joint training:
- invariants: j-invariant special values (ec04)
- group_structure: E(F_p) group structure (ec08)
- torsion: 2-torsion characterization (ec09)
- supersingular: supersingular conditions (ec11)

Usage:
    python train_ec_orthogonal.py
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


def train_cluster_adapter(model, tokenizer, lm_head, examples, cluster_name,
                          steps=1500, lr=5e-6, d_inner=64, margin_target=3.0):
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    loss_and_grad = mx.value_and_grad(mc_hinge_loss, argnums=0)

    print(f"\n  Training {cluster_name} adapter: {len(examples)} examples, "
          f"{steps} steps")

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

        if (step + 1) % 500 == 0:
            elapsed = time.time() - t0
            avg_margin = np.mean(recent_margins)
            print(f"    step {step+1:4d}/{steps}  loss={float(loss_val):.3f}  "
                  f"avg_margin={avg_margin:.3f}  {elapsed:.0f}s")

    return adapter


def eval_fact(adapter, lm_head, model, tokenizer, fact):
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
    margin = truth_lp - max(dist_lps)
    return margin > 0, margin


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load cluster data
    clusters_path = os.path.join(ROOT, "training", "elliptic_curves_clusters.json")
    with open(clusters_path) as f:
        data = json.load(f)
    clusters = data["clusters"]

    # Load oracle facts
    facts_path = os.path.join(ROOT, "problems", "elliptic_curves_facts.json")
    with open(facts_path) as f:
        facts_data = json.load(f)
    facts_by_id = {f["id"]: f for f in facts_data["facts"]}

    print("Loading Qwen/Qwen3-4B-Base...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Train each cluster adapter
    adapters = {}
    for cluster_name, cluster_data in clusters.items():
        print(f"\n{'='*60}")
        print(f"  Cluster: {cluster_name}")
        print(f"  Target facts: {cluster_data['facts']}")
        print("="*60)

        examples = cluster_data["examples"]

        # Baseline for target facts
        print("\n  Baseline:")
        for fact_id in cluster_data["facts"]:
            fact = facts_by_id[fact_id]
            win, margin = eval_fact(None, lm_head, model, tokenizer, fact)
            marker = "+" if win else "-"
            print(f"    {marker} {fact_id}: margin={margin:+.3f}")

        # Train adapter
        adapter = train_cluster_adapter(
            model, tokenizer, lm_head, examples, cluster_name,
            steps=1500, lr=5e-6, margin_target=3.0
        )

        # Eval after training
        print("\n  After training:")
        for fact_id in cluster_data["facts"]:
            fact = facts_by_id[fact_id]
            win, margin = eval_fact(adapter, lm_head, model, tokenizer, fact)
            marker = "+" if win else "-"
            print(f"    {marker} {fact_id}: margin={margin:+.3f}")
            if not win:
                pass

        # Save adapter
        out_path = os.path.join(OUT_DIR, f"ec_{cluster_name}_adapter.npz")
        weights = dict(tree_flatten(adapter.parameters()))
        mx.savez(out_path, **weights)
        print(f"  Saved: {out_path}")

        adapters[cluster_name] = adapter

    # Final summary with all adapters routed
    print("\n" + "="*60)
    print("  FINAL SUMMARY — All adapters routed to their facts")
    print("="*60)

    cluster_for_fact = {}
    for cluster_name, cluster_data in clusters.items():
        for fact_id in cluster_data["facts"]:
            cluster_for_fact[fact_id] = cluster_name

    wins = 0
    for fact in facts_data["facts"]:
        fact_id = fact["id"]
        if fact_id in cluster_for_fact:
            adapter = adapters[cluster_for_fact[fact_id]]
        else:
            adapter = None  # Use base model for facts not in clusters
        win, margin = eval_fact(adapter, lm_head, model, tokenizer, fact)
        marker = "+" if win else "-"
        cluster_label = cluster_for_fact.get(fact_id, "base")
        print(f"  {marker} {fact_id} [{cluster_label}]: margin={margin:+.3f}")
        wins += int(win)

    print(f"\n  Orthogonal adapter routing: {wins}/12 facts pass")


if __name__ == "__main__":
    main()
