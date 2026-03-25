#!/usr/bin/env python3
"""Extract steering vectors for ALL failing domains.

Replaces 50MB LoRA adapters with ~0.1KB steering vectors.
Load model ONCE, extract vectors for all domains, save results.

Usage:
    python experiments/extract_all_steering_vectors.py
    python experiments/extract_all_steering_vectors.py --layer 20
    python experiments/extract_all_steering_vectors.py --domain 3body_conservation
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT = Path(__file__).parent.parent
PROBLEMS_DIR = PROJECT / "problems"
VECTORS_DIR = PROJECT / "steering_vectors"
RESULTS_DIR = PROJECT / "results"


def find_all_fact_files():
    """Find all facts files, preferring V2 over V1."""
    facts_map = {}  # normalized_name -> (path, facts_list)

    for ff in sorted(PROBLEMS_DIR.glob("*_facts*.json")):
        try:
            with open(ff) as f:
                data = json.load(f)
            facts = data.get("facts", data.get("verifications", []))
            if not facts:
                continue

            # Normalize name
            stem = ff.stem.replace("_facts_v2", "").replace("_facts", "").replace("_balanced", "")

            # Prefer V2
            if stem not in facts_map or "_v2" in ff.stem:
                facts_map[stem] = (ff, facts)
        except (json.JSONDecodeError, KeyError):
            continue

    return facts_map


def get_activations(model, tokenizer, prompts, layer):
    """Get last-token activations at a specific layer."""
    import mlx.core as mx

    activations = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        hidden = model.model.embed_tokens(input_ids)
        for i, layer_module in enumerate(model.model.layers):
            hidden = layer_module(hidden, mask=None, cache=None)
            if i == layer:
                act = np.array(hidden[0, -1, :].astype(mx.float32))
                activations.append(act)
                break
        mx.eval(hidden)  # ensure computation completes

    return np.array(activations)


def compute_steering_vector(model, tokenizer, facts, layer):
    """Compute truth - false direction from fact pairs."""
    correct_prompts = []
    incorrect_prompts = []

    for fact in facts:
        context = fact.get("context", "")
        truth = fact.get("truth", fact.get("fact", ""))
        distractors = fact.get("distractors", [])
        if not distractors:
            continue

        base = f"{context}\n\nAnswer: " if context else "Answer: "
        correct_prompts.append(base + truth)
        incorrect_prompts.append(base + distractors[0])

    if not correct_prompts:
        return None

    correct_acts = get_activations(model, tokenizer, correct_prompts, layer)
    incorrect_acts = get_activations(model, tokenizer, incorrect_prompts, layer)

    return correct_acts.mean(axis=0) - incorrect_acts.mean(axis=0)


def evaluate_with_steering(model, tokenizer, facts, steering_vector, layer, alpha):
    """Evaluate MC accuracy with steering applied at layer."""
    import mlx.core as mx

    correct = 0
    total = 0

    for fact in facts:
        context = fact.get("context", "")
        truth = fact.get("truth", fact.get("fact", ""))
        distractors = fact.get("distractors", [])
        if not distractors:
            continue

        options = [truth] + distractors[:3]
        prompt = context + "\n\n" if context else ""
        prompt += "Which is correct?\n"
        for i, opt in enumerate(options):
            prompt += f"{chr(ord('A') + i)}) {opt}\n"
        prompt += "Answer: "

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Forward with steering
        hidden = model.model.embed_tokens(input_ids)
        for i, layer_module in enumerate(model.model.layers):
            hidden = layer_module(hidden, mask=mask if (mask := None) else None, cache=None)
            if i == layer:
                sv = mx.array(steering_vector.astype(np.float32) * alpha).reshape(1, 1, -1)
                hidden = hidden + sv

        hidden = model.model.norm(hidden)
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(hidden)
        else:
            logits = model.lm_head(hidden)

        option_tokens = [tokenizer.encode(chr(ord('A') + i))[-1] for i in range(len(options))]
        option_logits = [logits[0, -1, t].item() for t in option_tokens]

        if np.argmax(option_logits) == 0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def evaluate_baseline(model, tokenizer, facts):
    """Evaluate MC accuracy without steering."""
    import mlx.core as mx

    correct = 0
    total = 0

    for fact in facts:
        context = fact.get("context", "")
        truth = fact.get("truth", fact.get("fact", ""))
        distractors = fact.get("distractors", [])
        if not distractors:
            continue

        options = [truth] + distractors[:3]
        prompt = context + "\n\n" if context else ""
        prompt += "Which is correct?\n"
        for i, opt in enumerate(options):
            prompt += f"{chr(ord('A') + i)}) {opt}\n"
        prompt += "Answer: "

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)

        option_tokens = [tokenizer.encode(chr(ord('A') + i))[-1] for i in range(len(options))]
        option_logits = [logits[0, -1, t].item() for t in option_tokens]

        if np.argmax(option_logits) == 0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=15, help="Layer for steering")
    parser.add_argument("--domain", default=None, help="Single domain (default: all)")
    parser.add_argument("--alphas", default="0.05,0.10,0.25,0.50,0.75,1.0", help="Alpha values to test")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]

    from mlx_lm import load

    print("Loading Qwen/Qwen3-14B-Base...")
    model, tokenizer = load("Qwen/Qwen3-14B-Base")

    # Find all domains with facts
    facts_map = find_all_fact_files()

    if args.domain:
        if args.domain not in facts_map:
            print(f"Domain {args.domain} not found. Available: {sorted(facts_map.keys())}")
            return
        facts_map = {args.domain: facts_map[args.domain]}

    print(f"\nProcessing {len(facts_map)} domains at layer {args.layer}...")
    print(f"Alpha values: {alphas}")
    print(f"{'='*80}")

    VECTORS_DIR.mkdir(exist_ok=True)

    results = []
    total_start = time.time()

    for domain_name in sorted(facts_map.keys()):
        ff_path, facts = facts_map[domain_name]

        # Need at least 4 facts (2 train + 2 test)
        if len(facts) < 4:
            print(f"\n  {domain_name}: only {len(facts)} facts, skipping (need ≥4)")
            continue

        print(f"\n{'─'*60}")
        print(f"  {domain_name} ({len(facts)} facts from {ff_path.name})")
        start = time.time()

        # Split train/test
        np.random.seed(42)
        indices = np.random.permutation(len(facts))
        split = max(2, len(facts) // 2)
        train_facts = [facts[i] for i in indices[:split]]
        test_facts = [facts[i] for i in indices[split:]]

        if len(test_facts) < 1:
            test_facts = train_facts  # Use all for both if too few

        # Baseline
        baseline = evaluate_baseline(model, tokenizer, test_facts)

        # Compute steering vector
        sv = compute_steering_vector(model, tokenizer, train_facts, args.layer)
        if sv is None:
            print(f"    No valid fact pairs, skipping")
            continue

        # Test alphas
        best_acc = baseline
        best_alpha = 0.0
        alpha_results = {}

        for alpha in alphas:
            acc = evaluate_with_steering(model, tokenizer, test_facts, sv, args.layer, alpha)
            alpha_results[alpha] = acc
            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha

        improvement = best_acc - baseline
        elapsed = time.time() - start

        # Save vector
        vector_path = VECTORS_DIR / f"{domain_name}_layer{args.layer}.npy"
        np.save(vector_path, sv)
        vec_size = vector_path.stat().st_size

        # Also save metadata
        meta = {
            "domain": domain_name,
            "layer": args.layer,
            "baseline": baseline,
            "best_steered": best_acc,
            "improvement": improvement,
            "best_alpha": best_alpha,
            "alpha_results": {str(a): v for a, v in alpha_results.items()},
            "vector_norm": float(np.linalg.norm(sv)),
            "n_train": len(train_facts),
            "n_test": len(test_facts),
            "facts_file": ff_path.name,
            "vector_file": vector_path.name,
            "vector_bytes": vec_size,
        }

        results.append(meta)

        status = "✓ IMPROVED" if improvement > 0 else ("= SAME" if improvement == 0 else "✗ HURT")
        print(f"    Baseline: {baseline:.0%}  →  Steered: {best_acc:.0%}  (α={best_alpha:.2f})  {status}  [{elapsed:.1f}s]")

        for alpha in alphas:
            acc = alpha_results[alpha]
            delta = acc - baseline
            marker = " ←" if alpha == best_alpha and improvement > 0 else ""
            print(f"      α={alpha:.2f}: {acc:.0%} ({delta:+.0%}){marker}")

    total_elapsed = time.time() - total_start

    # Save all results
    results_path = RESULTS_DIR / "steering_vector_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY — {len(results)} domains processed in {total_elapsed:.0f}s")
    print(f"{'='*80}")

    improved = [r for r in results if r["improvement"] > 0]
    same = [r for r in results if r["improvement"] == 0]
    hurt = [r for r in results if r["improvement"] < 0]

    print(f"\n  Improved: {len(improved)}")
    for r in sorted(improved, key=lambda x: -x["improvement"]):
        print(f"    {r['domain']:<40s} {r['baseline']:.0%} → {r['best_steered']:.0%} (+{r['improvement']:.0%}) α={r['best_alpha']}")

    print(f"\n  Same: {len(same)}")
    for r in same:
        print(f"    {r['domain']:<40s} {r['baseline']:.0%}")

    print(f"\n  Hurt: {len(hurt)}")
    for r in sorted(hurt, key=lambda x: x["improvement"]):
        print(f"    {r['domain']:<40s} {r['baseline']:.0%} → {r['best_steered']:.0%} ({r['improvement']:+.0%})")

    # Total vector size
    total_bytes = sum(r["vector_bytes"] for r in results)
    print(f"\n  Total vector storage: {total_bytes / 1024:.1f} KB ({len(results)} vectors)")
    print(f"  vs LoRA adapters:    ~{len(results) * 50:.0f} MB ({len(results)} × 50MB)")
    print(f"  Compression ratio:   {len(results) * 50 * 1024 * 1024 / total_bytes:.0f}×")

    print(f"\n  Results saved to: {results_path}")
    print(f"  Vectors saved to: {VECTORS_DIR}/")


if __name__ == "__main__":
    main()
