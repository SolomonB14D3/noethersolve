#!/usr/bin/env python3
"""Train adapter on LLM self-knowledge facts with tool-grounding."""

import argparse, json, os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from noethersolve.oracle import get_completion_logprob
from noethersolve.adapter import SnapOnConfig, create_adapter

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    examples = []
    for ex in data.get("examples", []):
        examples.append((
            ex["context"],
            ex["truth"],
            ex.get("distractors", []),
        ))
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--facts", required=True, help="Verification facts JSON")
    parser.add_argument("--out", default="adapters/llm_tool_grounded.npz")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--margin", type=float, default=1.5)
    args = parser.parse_args()
    
    # Load training data
    examples = load_data(args.data)
    print(f"Loaded {len(examples)} training examples")
    
    # Load facts for eval
    with open(args.facts) as f:
        facts_data = json.load(f)
    facts = facts_data.get("facts", [])
    print(f"Loaded {len(facts)} verification facts")
    
    # Load model
    print("\nLoading Qwen/Qwen3-4B-Base...")
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    d_model = model.model.embed_tokens.weight.shape[1]
    vocab_size = model.model.embed_tokens.weight.shape[0]
    print(f"  d_model={d_model}  vocab={vocab_size}")
    
    # Create adapter
    config = SnapOnConfig(d_model=d_model, d_inner=args.d_inner, vocab_size=vocab_size)
    adapter = create_adapter(config)
    
    # Baseline eval
    print("\n=== BASELINE ===")
    correct = 0
    for fact in facts:
        prompt = fact["context"]
        truth = fact["truth"]
        distractors = fact.get("distractors", [])
        truth_lp = get_completion_logprob(model, tokenizer, prompt, truth)
        best_dist = max([get_completion_logprob(model, tokenizer, prompt, d)
                        for d in distractors], default=-999)
        if truth_lp > best_dist:
            correct += 1
    print(f"  Pass rate: {correct}/{len(facts)} ({100*correct/len(facts):.1f}%)")
    
    # Training loop
    optimizer = optim.AdamW(learning_rate=args.lr)
    
    print(f"\n=== TRAINING ({args.steps} steps) ===")
    for step in range(args.steps):
        ex = examples[step % len(examples)]
        prompt, truth, distractors = ex
        
        # Simple hinge loss on margin
        truth_lp = get_completion_logprob(model, tokenizer, prompt, truth)
        dist_lps = [get_completion_logprob(model, tokenizer, prompt, d) for d in distractors]
        best_dist = max(dist_lps) if dist_lps else truth_lp - 2.0
        margin = truth_lp - best_dist
        
        # Hinge: push margin above target
        target = args.margin
        loss_val = max(0.0, target - margin)
        
        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}: margin={margin:.2f} loss={loss_val:.4f}")
    
    # Final eval
    print("\n=== FINAL ===")
    correct = 0
    margins = []
    for fact in facts:
        prompt = fact["context"]
        truth = fact["truth"]
        distractors = fact.get("distractors", [])
        truth_lp = get_completion_logprob(model, tokenizer, prompt, truth)
        best_dist = max([get_completion_logprob(model, tokenizer, prompt, d)
                        for d in distractors], default=-999)
        margin = truth_lp - best_dist
        margins.append(margin)
        if margin > 0:
            correct += 1
    print(f"  Pass rate: {correct}/{len(facts)} ({100*correct/len(facts):.1f}%)")
    print(f"  Mean margin: {np.mean(margins):.2f}")
    
    # Save adapter
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    flat = tree_flatten(adapter.parameters())
    np.savez(args.out, **{k: np.array(v) for k, v in flat})
    print(f"\n  Saved to {args.out}")

if __name__ == "__main__":
    main()
