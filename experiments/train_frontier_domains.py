#!/usr/bin/env python3
"""Train orthogonal adapters for frontier problem domains.

8 domains, 96 facts total:
- battery_technology (12 facts)
- origin_of_life (12 facts)
- consciousness (12 facts)
- antibiotic_resistance (12 facts)
- protein_folding (12 facts)
- aging_biology (12 facts)
- quantum_gravity (12 facts)
- dark_matter_energy (12 facts)

Each cluster gets its own adapter for orthogonal routing.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noethersolve.adapter import SnapOnConfig, SnapOnLogitMLP
from noethersolve.train_utils import (
    LOGIT_SOFTCAP,
    apply_adapter,
    get_lm_head_fn,
)
from noethersolve.oracle import score_fact_mc

MODEL_ID = "Qwen/Qwen3-4B-Base"
PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "problems"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapters"

DOMAINS = [
    "battery_technology",
    "origin_of_life",
    "consciousness",
    "antibiotic_resistance",
    "protein_folding",
    "aging_biology",
    "quantum_gravity",
    "dark_matter_energy",
]


def load_all_facts():
    """Load facts from all domain files, grouped by (domain, cluster)."""
    all_facts = []
    for domain in DOMAINS:
        path = PROBLEMS_DIR / f"{domain}_facts.json"
        with open(path) as f:
            data = json.load(f)
        for fact in data["facts"]:
            fact["_domain"] = domain
            all_facts.append(fact)
    return all_facts


def group_by_cluster(facts):
    """Group facts by (domain, cluster)."""
    groups = defaultdict(list)
    for fact in facts:
        key = (fact["_domain"], fact["cluster"])
        groups[key].append(fact)
    return dict(groups)


def compute_margins(model, tokenizer, lm_head, facts, adapter=None):
    """Compute margins for facts. Returns list of (fact_id, margin)."""
    results = []
    for fact in facts:
        try:
            result = score_fact_mc(
                model, tokenizer,
                fact["context"], fact["truth"], fact["distractors"],
                adapter=adapter,
                lm_head=lm_head,
            )
            results.append((fact["id"], result[1]))  # margin
        except Exception as e:
            results.append((fact["id"], -999))
    return results


def train_step(model, tokenizer, lm_head, adapter, optimizer, fact, rng):
    """One training step with margin hinge loss."""
    d_idx = int(rng.integers(len(fact["distractors"])))
    prompt = fact["context"]
    truth = fact["truth"]
    distractor = fact["distractors"][d_idx]

    prompt_ids = tokenizer.encode(prompt)
    truth_ids = tokenizer.encode(prompt + " " + truth)
    dist_ids = tokenizer.encode(prompt + " " + distractor)
    n_prompt = len(prompt_ids)

    def loss_fn(adapter):
        # Truth log-prob
        tokens_t = mx.array(truth_ids)[None, :]
        h_t = model.model(tokens_t)
        base_t = lm_head(h_t)
        logits_t = apply_adapter(adapter, base_t)

        truth_lp = mx.array(0.0)
        for i, tok_id in enumerate(truth_ids[n_prompt:]):
            pos = n_prompt - 1 + i
            row = logits_t[0, pos]
            truth_lp = truth_lp + row[tok_id] - mx.logsumexp(row)

        # Distractor log-prob
        tokens_d = mx.array(dist_ids)[None, :]
        h_d = model.model(tokens_d)
        base_d = lm_head(h_d)
        logits_d = apply_adapter(adapter, base_d)

        dist_lp = mx.array(0.0)
        for i, tok_id in enumerate(dist_ids[n_prompt:]):
            pos = n_prompt - 1 + i
            row = logits_d[0, pos]
            dist_lp = dist_lp + row[tok_id] - mx.logsumexp(row)

        # Hinge loss: want truth_lp > dist_lp by margin of 2.0
        margin = truth_lp - dist_lp
        loss = mx.maximum(mx.array(0.0), mx.array(2.0) - margin)
        return loss

    loss, grads = nn.value_and_grad(adapter, loss_fn)(adapter)
    optimizer.update(adapter, grads)
    mx.eval(adapter.parameters(), optimizer.state)
    return float(loss)


def train_cluster(model, tokenizer, lm_head, vocab_size,
                  domain, cluster, facts, args, rng):
    """Train adapter for a single cluster."""
    print(f"\n{'='*60}")
    print(f"Training {domain}/{cluster} ({len(facts)} facts)")
    print(f"{'='*60}")

    # Create adapter
    config = SnapOnConfig(d_inner=args.d_inner, vocab_size=vocab_size, mode="logit")
    adapter = SnapOnLogitMLP(config)
    mx.eval(adapter.parameters())

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    # Baseline
    base_margins = compute_margins(model, tokenizer, lm_head, facts)
    n_pass = sum(1 for _, m in base_margins if m > 0)
    print(f"  Baseline: {n_pass}/{len(facts)} passing")

    # Training
    t0 = time.time()
    for step in range(args.steps):
        idx = int(rng.integers(len(facts)))
        fact = facts[idx]
        loss = train_step(model, tokenizer, lm_head, adapter, optimizer, fact, rng)

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{args.steps}: loss={loss:.4f}")

    # Final eval
    final_margins = compute_margins(model, tokenizer, lm_head, facts, adapter)
    n_pass = sum(1 for _, m in final_margins if m > 0)
    print(f"  Final: {n_pass}/{len(facts)} passing")

    # Show any failures
    for fid, margin in final_margins:
        if margin <= 0:
            print(f"    FAIL: {fid} margin={margin:.2f}")

    # Save adapter
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ADAPTER_DIR / f"{domain}_{cluster}_adapter.npz"

    params = {}
    for k, v in adapter.parameters().items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                params[f"{k}.{k2}"] = v2
        else:
            params[k] = v

    mx.savez(str(out_path), **params)
    print(f"  Saved: {out_path.name}")
    print(f"  Time: {time.time()-t0:.1f}s")

    return n_pass, len(facts)


def main():
    parser = argparse.ArgumentParser(description="Train Frontier Domain Adapters")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load all facts
    print("Loading frontier domain facts...")
    all_facts = load_all_facts()
    print(f"  Total: {len(all_facts)} facts across {len(DOMAINS)} domains")

    # Group by cluster
    groups = group_by_cluster(all_facts)
    print(f"  Clusters: {len(groups)}")
    for (domain, cluster), facts in sorted(groups.items()):
        print(f"    {domain}/{cluster}: {len(facts)} facts")

    # Load model
    print(f"\nLoading {MODEL_ID}...")
    model, tokenizer = mlx_lm.load(MODEL_ID)
    model.eval()
    lm_head = get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]
    print(f"  vocab_size={vocab_size}")

    # Train each cluster
    total_pass = 0
    total_facts = 0
    t_start = time.time()

    for (domain, cluster), facts in sorted(groups.items()):
        n_pass, n_total = train_cluster(
            model, tokenizer, lm_head, vocab_size,
            domain, cluster, facts, args, rng
        )
        total_pass += n_pass
        total_facts += n_total

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {total_pass}/{total_facts} facts passing ({100*total_pass/total_facts:.1f}%)")
    print(f"Clusters trained: {len(groups)}")
    print(f"Total time: {(time.time()-t_start)/60:.1f}m")


if __name__ == "__main__":
    main()
