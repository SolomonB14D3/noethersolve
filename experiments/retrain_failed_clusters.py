#!/usr/bin/env python3
"""Retrain only the clusters with failing facts."""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noethersolve.adapter import SnapOnConfig, SnapOnLogitMLP
from noethersolve.train_utils import apply_adapter, get_lm_head_fn
from noethersolve.oracle import score_fact_mc

MODEL_ID = "Qwen/Qwen3-4B-Base"
PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "problems"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapters"

# Clusters to retrain (domain, cluster, fact_file)
FAILED_CLUSTERS = [
    ("aging_biology", "biomarkers", "aging_biology_facts.json"),
    ("antibiotic_resistance", "alternatives", "antibiotic_resistance_facts.json"),
]


def load_cluster_facts(domain, cluster, fact_file):
    """Load facts for a specific cluster."""
    path = PROBLEMS_DIR / fact_file
    with open(path) as f:
        data = json.load(f)
    return [f for f in data["facts"] if f["cluster"] == cluster]


def compute_margins(model, tokenizer, lm_head, facts, adapter=None):
    """Compute margins for facts."""
    results = []
    for fact in facts:
        try:
            result = score_fact_mc(
                model, tokenizer,
                fact["context"], fact["truth"], fact["distractors"],
                adapter=adapter,
                lm_head=lm_head,
            )
            results.append((fact["id"], result[1]))
        except Exception as e:
            results.append((fact["id"], -999))
    return results


def train_step(model, tokenizer, lm_head, adapter, optimizer, fact, rng):
    """One training step."""
    d_idx = int(rng.integers(len(fact["distractors"])))
    prompt = fact["context"]
    truth = fact["truth"]
    distractor = fact["distractors"][d_idx]

    prompt_ids = tokenizer.encode(prompt)
    truth_ids = tokenizer.encode(prompt + " " + truth)
    dist_ids = tokenizer.encode(prompt + " " + distractor)
    n_prompt = len(prompt_ids)

    def loss_fn(adapter):
        tokens_t = mx.array(truth_ids)[None, :]
        h_t = model.model(tokens_t)
        base_t = lm_head(h_t)
        logits_t = apply_adapter(adapter, base_t)

        truth_lp = mx.array(0.0)
        for i, tok_id in enumerate(truth_ids[n_prompt:]):
            pos = n_prompt - 1 + i
            row = logits_t[0, pos]
            truth_lp = truth_lp + row[tok_id] - mx.logsumexp(row)

        tokens_d = mx.array(dist_ids)[None, :]
        h_d = model.model(tokens_d)
        base_d = lm_head(h_d)
        logits_d = apply_adapter(adapter, base_d)

        dist_lp = mx.array(0.0)
        for i, tok_id in enumerate(dist_ids[n_prompt:]):
            pos = n_prompt - 1 + i
            row = logits_d[0, pos]
            dist_lp = dist_lp + row[tok_id] - mx.logsumexp(row)

        margin = truth_lp - dist_lp
        loss = mx.maximum(mx.array(0.0), mx.array(2.0) - margin)
        return loss

    loss, grads = nn.value_and_grad(adapter, loss_fn)(adapter)
    optimizer.update(adapter, grads)
    mx.eval(adapter.parameters(), optimizer.state)
    return float(loss)


def main():
    rng = np.random.default_rng(42)

    print(f"Loading {MODEL_ID}...")
    model, tokenizer = mlx_lm.load(MODEL_ID)
    model.eval()
    lm_head = get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]

    total_pass = 0
    total_facts = 0

    for domain, cluster, fact_file in FAILED_CLUSTERS:
        facts = load_cluster_facts(domain, cluster, fact_file)
        print(f"\n{'='*60}")
        print(f"Retraining {domain}/{cluster} ({len(facts)} facts)")
        print(f"{'='*60}")

        # Create fresh adapter
        config = SnapOnConfig(d_inner=64, vocab_size=vocab_size, mode="logit")
        adapter = SnapOnLogitMLP(config)
        mx.eval(adapter.parameters())

        optimizer = optim.AdamW(learning_rate=1e-5, weight_decay=0.01)

        # Baseline
        base_margins = compute_margins(model, tokenizer, lm_head, facts)
        n_pass = sum(1 for _, m in base_margins if m > 0)
        print(f"  Baseline: {n_pass}/{len(facts)} passing")
        for fid, m in base_margins:
            print(f"    {fid}: margin={m:.2f}")

        # Train
        t0 = time.time()
        for step in range(500):
            idx = int(rng.integers(len(facts)))
            fact = facts[idx]
            loss = train_step(model, tokenizer, lm_head, adapter, optimizer, fact, rng)
            if (step + 1) % 100 == 0:
                print(f"  Step {step+1}/500: loss={loss:.4f}")

        # Final eval
        final_margins = compute_margins(model, tokenizer, lm_head, facts, adapter)
        n_pass = sum(1 for _, m in final_margins if m > 0)
        print(f"  Final: {n_pass}/{len(facts)} passing")
        for fid, m in final_margins:
            status = "PASS" if m > 0 else "FAIL"
            print(f"    {fid}: margin={m:.2f} [{status}]")

        # Save
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

        total_pass += n_pass
        total_facts += len(facts)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_pass}/{total_facts} passing")


if __name__ == "__main__":
    main()
