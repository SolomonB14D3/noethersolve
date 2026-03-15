#!/usr/bin/env python3
"""Train a single unified toolkit adapter from all toolkit module knowledge.

Uses contrastive margin loss: maximize log P(truth) - log P(distractor).

Usage:
    python experiments/train_toolkit_adapter.py
    python experiments/train_toolkit_adapter.py --steps 2000 --lr 3e-4
"""

import argparse
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
from noethersolve.train_utils import (
    LOGIT_SOFTCAP,
    apply_adapter,
    get_lm_head_fn,
)
from noethersolve.oracle import score_fact_mc


MODEL_ID = "Qwen/Qwen3-4B-Base"
FACTS_PATH = Path(__file__).resolve().parent.parent / "problems" / "toolkit_facts.json"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapters"


def compute_margins(model, tokenizer, lm_head, facts, adapter=None):
    """Compute oracle margins for all facts. Returns list of margins."""
    margins = []
    for fact in facts:
        try:
            result = score_fact_mc(
                model, tokenizer,
                fact["context"], fact["truth"], fact["distractors"],
                adapter=adapter,
                lm_head=lm_head,
            )
            # result is tuple: (win, margin, truth_lp, best_dist_lp)
            margins.append(result[1])  # margin = truth_lp - best_dist_lp
        except Exception as e:
            margins.append(-999)
    return margins


def _get_completion_lp(model, tokenizer, lm_head, adapter, prompt, completion):
    """Get sum log-prob of completion given prompt, with adapter."""
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + " " + completion)
    n_prompt = len(prompt_ids)

    if len(full_ids) <= n_prompt:
        return mx.array(-100.0)

    tokens = mx.array(full_ids)[None, :]
    h = model.model(tokens)
    base_logits = lm_head(h)
    logits = apply_adapter(adapter, base_logits)

    lp = mx.array(0.0)
    for i, tok_id in enumerate(full_ids[n_prompt:]):
        pos = n_prompt - 1 + i
        row = logits[0, pos]
        log_probs = row - mx.logsumexp(row)
        lp = lp + log_probs[tok_id]

    return lp


def train_step(model, tokenizer, lm_head, adapter, optimizer, fact, rng):
    """One training step: margin loss between truth and random distractor."""

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


def main():
    parser = argparse.ArgumentParser(description="Train Toolkit Adapter")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=500)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load facts
    with open(FACTS_PATH) as f:
        data = json.load(f)
    facts = data["facts"]
    print(f"Loaded {len(facts)} toolkit facts")

    # Load model
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = mlx_lm.load(MODEL_ID)
    model.eval()
    lm_head = get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]
    print(f"  vocab_size={vocab_size}")

    # Create adapter
    config = SnapOnConfig(d_inner=args.d_inner, vocab_size=vocab_size, mode="logit")
    adapter = SnapOnLogitMLP(config)
    mx.eval(adapter.parameters())

    optimizer = optim.Adam(learning_rate=args.lr)

    # Baseline margins
    print("\nComputing baseline margins...")
    base_margins = compute_margins(model, tokenizer, lm_head, facts)
    valid_margins = [m for m in base_margins if m > -900]
    n_pass = sum(1 for m in base_margins if m > 0)
    if valid_margins:
        print(f"  Baseline: {n_pass}/{len(facts)} passing")
        print(f"  Mean margin: {np.mean(valid_margins):.2f} (over {len(valid_margins)} valid)")
    else:
        print(f"  Baseline: no valid margins computed")

    # Training loop
    print(f"\nTraining for {args.steps} steps (lr={args.lr})...")
    t0 = time.time()
    losses = []

    for step in range(args.steps):
        # Difficulty-weighted sampling
        if step < 200 or not valid_margins:
            idx = int(rng.integers(len(facts)))
        else:
            weights = np.array([max(0.1, 5.0 - m) if m > -900 else 5.0 for m in base_margins])
            weights = weights / weights.sum()
            idx = int(rng.choice(len(facts), p=weights))

        fact = facts[idx]
        loss = train_step(model, tokenizer, lm_head, adapter, optimizer, fact, rng)
        losses.append(loss)

        if (step + 1) % 50 == 0:
            avg_loss = np.mean(losses[-50:])
            elapsed = time.time() - t0
            print(f"  Step {step+1:5d}/{args.steps}: loss={avg_loss:.4f}, "
                  f"elapsed={elapsed:.0f}s")

        if (step + 1) % args.eval_every == 0:
            print(f"\n  === Eval at step {step+1} ===")
            margins = compute_margins(model, tokenizer, lm_head, facts, adapter)
            valid = [m for m in margins if m > -900]
            n_pass = sum(1 for m in margins if m > 0)
            if valid:
                print(f"  {n_pass}/{len(facts)} passing, mean={np.mean(valid):.2f}")
            base_margins = margins  # update for sampling

    # Final eval
    print(f"\n{'='*50}")
    print("Final evaluation...")
    final_margins = compute_margins(model, tokenizer, lm_head, facts, adapter)
    valid = [m for m in final_margins if m > -900]
    n_pass = sum(1 for m in final_margins if m > 0)
    if valid:
        print(f"  Final: {n_pass}/{len(facts)} passing, mean={np.mean(valid):.2f}")
    print(f"  Total time: {(time.time()-t0)/60:.1f}m")

    # Save using the standard pattern
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ADAPTER_DIR / "toolkit_unified_adapter.npz"

    # Flatten parameters for saving
    params = {}
    for k, v in adapter.parameters().items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                params[f"{k}.{k2}"] = v2
        else:
            params[k] = v

    mx.savez(str(out_path), **params)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
