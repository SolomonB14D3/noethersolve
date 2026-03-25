#!/usr/bin/env python3
"""Test if steering vectors can replace LoRA adapters.

Hypothesis: The adapters are encoding a "truth direction" that could be
captured by a single steering vector (~4KB) instead of full LoRA weights (~50MB).

Usage:
    python experiments/test_steering_vectors.py --domain biochemistry
    python experiments/test_steering_vectors.py --domain chemistry --layer 15
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class SteeringResult:
    domain: str
    layer: int
    baseline_acc: float
    steered_acc: float
    improvement: float
    vector_norm: float
    optimal_alpha: float

    def __str__(self):
        return (
            f"Domain: {self.domain} (layer {self.layer})\n"
            f"Baseline accuracy: {self.baseline_acc:.1%}\n"
            f"Steered accuracy:  {self.steered_acc:.1%}\n"
            f"Improvement:       {self.improvement:+.1%}\n"
            f"Optimal α:         {self.optimal_alpha:.2f}\n"
            f"Vector norm:       {self.vector_norm:.4f}\n"
            f"Vector size:       ~{self.vector_norm * 4 / 1024:.1f} KB"
        )


def load_facts(domain: str) -> list[dict]:
    """Load facts for a domain."""
    project = Path(__file__).parent.parent

    # Search in problems/ and steering_bank/
    search_dirs = [project / "problems", project / "steering_bank"]
    patterns = [
        f"{domain}_facts.json",
        f"{domain}_facts_v2.json",
        f"{domain}.json",
    ]

    for d in search_dirs:
        for pattern in patterns:
            facts_file = d / pattern
            if facts_file.exists():
                with open(facts_file) as f:
                    data = json.load(f)
                return data.get("facts", data.get("verifications", []))

    raise FileNotFoundError(f"No facts file found for domain: {domain}")


def get_activations(model, tokenizer, prompts: list[str], layer: int):
    """Get activations at a specific layer for a batch of prompts."""
    import mlx.core as mx

    activations = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Get hidden states at layer
        hidden = model.model.embed_tokens(input_ids)

        for i, layer_module in enumerate(model.model.layers):
            hidden = layer_module(hidden, mask=None, cache=None)

            if i == layer:
                # Get the last token's activation (convert bfloat16 to float32)
                act = np.array(hidden[0, -1, :].astype(mx.float32))
                activations.append(act)
                break

    return np.array(activations)


def compute_steering_vector(
    model,
    tokenizer,
    facts: list[dict],
    layer: int
) -> np.ndarray:
    """Compute steering vector from correct vs incorrect activations."""

    correct_prompts = []
    incorrect_prompts = []

    for fact in facts:
        context = fact.get("context", "")
        truth = fact.get("truth", fact.get("fact", ""))
        distractors = fact.get("distractors", [])

        if not distractors:
            continue

        # Build prompts
        base = f"{context}\n\nAnswer: " if context else "Answer: "
        correct_prompts.append(base + truth)
        incorrect_prompts.append(base + distractors[0])

    if not correct_prompts:
        raise ValueError("No valid facts with distractors found")

    print(f"  Computing activations for {len(correct_prompts)} fact pairs...")

    # Get activations
    correct_acts = get_activations(model, tokenizer, correct_prompts, layer)
    incorrect_acts = get_activations(model, tokenizer, incorrect_prompts, layer)

    # Steering vector = mean difference
    steering_vector = correct_acts.mean(axis=0) - incorrect_acts.mean(axis=0)

    return steering_vector


def evaluate_with_steering(
    model,
    tokenizer,
    facts: list[dict],
    steering_vector: np.ndarray,
    layer: int,
    alpha: float
) -> float:
    """Evaluate accuracy with steering vector applied."""
    import mlx.core as mx
    import mlx.nn as nn

    correct = 0
    total = 0

    for fact in facts:
        context = fact.get("context", "")
        truth = fact.get("truth", fact.get("fact", ""))
        distractors = fact.get("distractors", [])

        if not distractors:
            continue

        # Build MC prompt
        options = [truth] + distractors[:3]
        prompt = context + "\n\n" if context else ""
        prompt += "Which is correct?\n"
        for i, opt in enumerate(options):
            prompt += f"{chr(ord('A') + i)}) {opt}\n"
        prompt += "Answer: "

        # Get logits for each option
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Forward pass with steering
        hidden = model.model.embed_tokens(input_ids)
        mask = None

        for i, layer_module in enumerate(model.model.layers):
            hidden = layer_module(hidden, mask=mask, cache=None)

            # Apply steering at target layer
            if i == layer:
                steering = mx.array(steering_vector.astype(np.float32) * alpha).reshape(1, 1, -1)
                hidden = hidden + steering

        # Get final logits
        hidden = model.model.norm(hidden)
        # Handle tied embeddings vs separate lm_head
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(hidden)
        else:
            logits = model.lm_head(hidden)

        # Compare logits for A, B, C, D
        option_tokens = [tokenizer.encode(chr(ord('A') + i))[-1] for i in range(len(options))]
        option_logits = [logits[0, -1, t].item() for t in option_tokens]

        predicted = np.argmax(option_logits)
        if predicted == 0:  # Truth is always option A
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def evaluate_baseline(model, tokenizer, facts: list[dict]) -> float:
    """Evaluate baseline accuracy without steering."""
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

        predicted = np.argmax(option_logits)
        if predicted == 0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def test_steering(
    domain: str,
    layer: int = 15,
    alphas: list[float] = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
) -> SteeringResult:
    """Test steering vector for a domain."""
    from mlx_lm import load

    print(f"Loading model...")
    model, tokenizer = load("Qwen/Qwen3-14B-Base")

    print(f"Loading facts for {domain}...")
    facts = load_facts(domain)
    print(f"  Found {len(facts)} facts")

    # Split into train (for steering vector) and test (for evaluation)
    np.random.seed(42)
    indices = np.random.permutation(len(facts))
    split = len(facts) // 2
    train_facts = [facts[i] for i in indices[:split]]
    test_facts = [facts[i] for i in indices[split:]]

    print(f"  Train: {len(train_facts)}, Test: {len(test_facts)}")

    # Baseline
    print(f"\nEvaluating baseline...")
    baseline_acc = evaluate_baseline(model, tokenizer, test_facts)
    print(f"  Baseline accuracy: {baseline_acc:.1%}")

    # Compute steering vector from train set
    print(f"\nComputing steering vector at layer {layer}...")
    steering_vector = compute_steering_vector(model, tokenizer, train_facts, layer)
    vector_norm = np.linalg.norm(steering_vector)
    print(f"  Vector norm: {vector_norm:.4f}")

    # Test different alpha values
    print(f"\nTesting steering strengths...")
    best_acc = baseline_acc
    best_alpha = 0.0

    for alpha in alphas:
        acc = evaluate_with_steering(model, tokenizer, test_facts, steering_vector, layer, alpha)
        improvement = acc - baseline_acc
        print(f"  α={alpha:.2f}: {acc:.1%} ({improvement:+.1%})")

        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    return SteeringResult(
        domain=domain,
        layer=layer,
        baseline_acc=baseline_acc,
        steered_acc=best_acc,
        improvement=best_acc - baseline_acc,
        vector_norm=vector_norm,
        optimal_alpha=best_alpha,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="biochemistry", help="Domain to test")
    parser.add_argument("--layer", type=int, default=15, help="Layer for steering")
    parser.add_argument("--all-layers", action="store_true", help="Test all layers")
    args = parser.parse_args()

    if args.all_layers:
        # Test multiple layers to find best one
        print(f"Testing steering across layers for {args.domain}...")
        results = []
        for layer in [5, 10, 15, 20, 25, 30]:
            try:
                result = test_steering(args.domain, layer=layer)
                results.append(result)
                print(f"\n{'='*60}")
                print(result)
                print('='*60)
            except Exception as e:
                print(f"Layer {layer} failed: {e}")

        # Find best
        if results:
            best = max(results, key=lambda r: r.improvement)
            print(f"\n{'='*60}")
            print("BEST RESULT:")
            print(best)
    else:
        result = test_steering(args.domain, layer=args.layer)
        print(f"\n{'='*60}")
        print(result)
        print('='*60)

        # Save steering vector
        output_dir = Path(__file__).parent.parent / "steering_vectors"
        output_dir.mkdir(exist_ok=True)

        # We'd need to recompute to save, but show the path
        print(f"\nSteering vector would be saved to:")
        print(f"  {output_dir}/{args.domain}_layer{args.layer}.npy")


if __name__ == "__main__":
    main()
