#!/usr/bin/env python3
"""
Simple adapter training from facts file.
Used by research_agent.py to train adapters for domains with severe gaps.

Usage:
    python scripts/train_from_facts.py --facts problems/chemical_conservation_facts.json \
                                       --model meta-llama/Llama-3.1-8B \
                                       --output adapters/chemical_llama8b.npz
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import mlx_lm
import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve import train_utils as t3
from noethersolve.oracle import get_completion_logprob

def load_facts(facts_file: str) -> list:
    """Load facts from JSON and convert to training examples."""
    with open(facts_file) as f:
        data = json.load(f)

    facts = data.get("facts", data.get("verifications", []))
    examples = []

    for fact in facts:
        context = fact.get("context", "")
        truth = fact.get("truth", fact.get("fact", ""))
        distractors = fact.get("distractors", [])

        if context and truth and distractors:
            examples.append((context, truth, distractors))

    return examples

def train_adapter(model, tokenizer, lm_head, examples, steps=500, lr=1e-5, d_inner=64):
    """Train a simple adapter on the examples."""
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]

    config = SnapOnConfig(d_model=d_model, d_inner=d_inner, vocab_size=vocab_size)
    adapter = create_adapter(config)

    optimizer = optim.Adam(learning_rate=lr)

    print(f"Training adapter: {steps} steps, {len(examples)} examples")

    for step in range(steps):
        total_loss = 0.0

        for context, truth, distractors in examples:
            prompt = f"{context}:"

            # Get truth score
            truth_full = prompt + " " + truth
            truth_lp = t3.get_completion_logprob(model, tokenizer, prompt, truth_full, adapter, lm_head)

            # Get distractor scores
            dist_lps = []
            for dist in distractors:
                dist_full = prompt + " " + dist
                dist_lp = t3.get_completion_logprob(model, tokenizer, prompt, dist_full, adapter, lm_head)
                dist_lps.append(dist_lp)

            # Margin loss: want truth_lp > max(dist_lps) + margin
            margin = 1.0
            max_dist_lp = max(dist_lps)
            loss = mx.maximum(0, margin - (truth_lp - max_dist_lp))
            total_loss += loss.item()

        # Backward and update
        if total_loss > 0:
            # Simplified training step
            grads = mx.grad(lambda a: total_loss)(adapter)
            optimizer.update(adapter, grads)
            mx.eval(adapter.parameters())

        if step % 100 == 0:
            print(f"  Step {step}: loss={total_loss:.4f}")

    return adapter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", required=True, help="Facts JSON file")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--output", required=True, help="Output adapter path")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    # Load facts
    examples = load_facts(args.facts)
    print(f"Loaded {len(examples)} training examples from {args.facts}")

    if len(examples) < 3:
        print("ERROR: Need at least 3 examples to train")
        sys.exit(1)

    # Load model
    print(f"Loading {args.model}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Train
    adapter = train_adapter(model, tokenizer, lm_head, examples, steps=args.steps)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    weights = {k: np.array(v) for k, v in adapter.parameters().items()}
    np.savez(args.output, **weights)
    print(f"Saved adapter to {args.output}")

if __name__ == "__main__":
    main()
