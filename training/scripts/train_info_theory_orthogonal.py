#!/usr/bin/env python3
"""Train orthogonal adapters for information theory fact clusters.

Single-pass training failed (1/12) due to fact interference. This script
trains separate specialist adapters per conceptual cluster:

Cluster 1 (capacity): BSC, BEC, AWGN, Z-channel capacity formulas
Cluster 2 (rd): Rate-distortion theory (binary, Gaussian)
Cluster 3 (multiuser): Source coding, MAC capacity region
Cluster 4 (inequalities): Data processing, Fano, typical sets

Each cluster gets its own adapter, routed at inference time.
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

# Cluster definitions - which training examples belong to each cluster
CLUSTERS = {
    "capacity": {
        "description": "Channel capacity formulas (BSC, BEC, AWGN, Z-channel)",
        "keywords": ["capacity", "BSC", "BEC", "AWGN", "Z-channel", "crossover", "erasure"],
    },
    "rd": {
        "description": "Rate-distortion theory",
        "keywords": ["rate-distortion", "R(D)", "distortion", "Hamming", "squared error"],
    },
    "multiuser": {
        "description": "Source coding and MAC",
        "keywords": ["source coding", "MAC", "multiple access", "pentagon", "sum rate"],
    },
    "inequalities": {
        "description": "Information inequalities (DPI, Fano, typical sets)",
        "keywords": ["data processing", "Fano", "typical set", "AEP", "I(X;Z)", "H(X|Y)"],
    },
}

# Map fact IDs to clusters
FACT_CLUSTERS = {
    "info01": "capacity",   # BSC capacity
    "info02": "capacity",   # BEC capacity
    "info03": "capacity",   # AWGN capacity
    "info04": "capacity",   # Z-channel
    "info05": "rd",         # Binary rate-distortion
    "info06": "rd",         # Gaussian rate-distortion
    "info07": "multiuser",  # Source coding
    "info08": "multiuser",  # MAC pentagon
    "info09": "multiuser",  # MAC sum rate
    "info10": "inequalities",  # Data processing inequality
    "info11": "inequalities",  # Fano's inequality
    "info12": "inequalities",  # Typical set size
}


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


def assign_training_examples_to_clusters(examples):
    """Assign training examples to clusters based on keywords."""
    clustered = {name: [] for name in CLUSTERS}

    for ex in examples:
        context = ex["context"].lower()
        truth = ex["truth"].lower()
        text = context + " " + truth

        best_cluster = None
        best_score = 0

        for cluster_name, cluster_info in CLUSTERS.items():
            score = sum(1 for kw in cluster_info["keywords"] if kw.lower() in text)
            if score > best_score:
                best_score = score
                best_cluster = cluster_name

        if best_cluster and best_score > 0:
            clustered[best_cluster].append(ex)
        else:
            # Default to capacity if no match
            clustered["capacity"].append(ex)

    return clustered


def train_cluster_adapter(model, tokenizer, lm_head, examples, cluster_name,
                         steps=3000, lr=5e-6, d_inner=64, margin_target=3.0):
    """Train a single cluster adapter."""
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    loss_and_grad = mx.value_and_grad(mc_hinge_loss, argnums=0)

    print(f"\n  Training {cluster_name} adapter: {len(examples)} examples, {steps} steps")

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
            print(f"    step {step+1:5d}/{steps}  loss={float(loss_val):.3f}  "
                  f"avg_margin={avg_margin:.3f}  {elapsed:.0f}s")

    return adapter


def eval_fact_with_adapter(adapter, lm_head, model, tokenizer, fact):
    """Evaluate a single fact with an adapter."""
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


def eval_all_facts_routed(adapters, lm_head, model, tokenizer, facts):
    """Evaluate all facts, routing each to its cluster adapter."""
    print("\n  Routed evaluation (each fact → its cluster adapter)")
    wins, margins = 0, []
    for fact in facts:
        fact_id = fact["id"]
        cluster = FACT_CLUSTERS.get(fact_id, "capacity")
        adapter = adapters.get(cluster)

        passed, margin = eval_fact_with_adapter(adapter, lm_head, model, tokenizer, fact)
        wins += int(passed)
        margins.append(margin)
        marker = "+" if passed else "-"
        print(f"    {marker} {fact_id:10s} ({cluster:12s}) margin={margin:+.3f}")

    mean_m = float(np.mean(margins))
    print(f"  Pass: {wins}/{len(facts)}  mean_margin={mean_m:+.3f}")
    return wins, len(facts), mean_m


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load training data
    data_path = os.path.join(ROOT, "training", "information_theory_synthetic_60.json")
    with open(data_path) as f:
        data = json.load(f)
    examples = data["examples"]
    print(f"\nLoaded {len(examples)} training examples")

    # Assign to clusters
    clustered = assign_training_examples_to_clusters(examples)
    for name, exs in clustered.items():
        print(f"  {name}: {len(exs)} examples")

    # Load facts
    facts_path = os.path.join(ROOT, "problems", "information_theory_facts.json")
    with open(facts_path) as f:
        facts_data = json.load(f)
    facts = facts_data["facts"]

    print("\nLoading Qwen/Qwen3-4B-Base...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Train adapters for each cluster
    adapters = {}
    for cluster_name, cluster_examples in clustered.items():
        if len(cluster_examples) < 3:
            print(f"\n  Skipping {cluster_name} - only {len(cluster_examples)} examples")
            continue

        adapter = train_cluster_adapter(
            model, tokenizer, lm_head, cluster_examples, cluster_name,
            steps=3000, lr=5e-6, margin_target=3.0
        )
        adapters[cluster_name] = adapter

        # Save adapter
        out_path = os.path.join(OUT_DIR, f"info_theory_{cluster_name}_adapter.npz")
        weights = dict(tree_flatten(adapter.parameters()))
        mx.savez(out_path, **weights)
        print(f"  Saved: {out_path}")

    # Evaluate with routing
    print("\n" + "="*60)
    print("  ORTHOGONAL ADAPTERS — ROUTED EVALUATION")
    print("="*60)
    eval_all_facts_routed(adapters, lm_head, model, tokenizer, facts)


if __name__ == "__main__":
    main()
