#!/usr/bin/env python3
"""Test routed steering for MC evaluation (the original use case)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_lm import load
import mlx.core as mx
from noethersolve.steering_router import SteeringRouter


def evaluate_with_steering(model, tokenizer, fact, steering_vector, layer, alpha):
    """Evaluate a single MC fact with steering applied."""

    context = fact.get("context", "")
    truth = fact.get("truth", fact.get("fact", ""))
    distractors = fact.get("distractors", [])

    if not distractors:
        return None

    # Build MC prompt
    options = [truth] + distractors[:3]
    prompt = context + "\n\n" if context else ""
    prompt += "Which is correct?\n"
    for i, opt in enumerate(options):
        prompt += f"{chr(ord('A') + i)}) {opt}\n"
    prompt += "Answer: "

    # Get tokens
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Forward pass with steering
    hidden = model.model.embed_tokens(input_ids)

    steering_tensor = None
    if steering_vector is not None:
        steering_tensor = mx.array(steering_vector.astype(np.float32) * alpha).reshape(1, 1, -1)

    for i, layer_module in enumerate(model.model.layers):
        hidden = layer_module(hidden, mask=None, cache=None)

        # Apply steering at target layer
        if steering_tensor is not None and i == layer:
            hidden = hidden + steering_tensor

    # Get final logits
    hidden = model.model.norm(hidden)
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(hidden)
    else:
        logits = model.lm_head(hidden)

    # Compare logits for A, B, C, D
    option_tokens = [tokenizer.encode(chr(ord('A') + i))[-1] for i in range(len(options))]
    option_logits = [logits[0, -1, t].item() for t in option_tokens]

    predicted = np.argmax(option_logits)
    return predicted == 0  # Truth is always option A


def evaluate_baseline(model, tokenizer, fact):
    """Evaluate without steering."""
    return evaluate_with_steering(model, tokenizer, fact, None, 0, 0)


def main():
    print("Loading model...")
    model, tokenizer = load("Qwen/Qwen3-4B-Base")

    print("Loading steering bank...")
    router = SteeringRouter.load("steering_bank.npz")
    print(f"  {len(router)} domains")

    # Load a few facts from different domains
    facts_dir = Path("problems")
    test_domains = [
        "3body_conservation_facts.json",
        "biochemistry_facts.json",
        "number_theory_conjectures_facts.json",
        "physics_fundamentals_2d_turbulence_facts.json",
    ]

    for facts_file in test_domains:
        path = facts_dir / facts_file
        if not path.exists():
            continue

        with open(path) as f:
            data = json.load(f)
        facts = data.get("facts", data.get("verifications", []))[:5]  # First 5 facts

        if not facts:
            continue

        domain = facts_file.replace("_facts.json", "")
        print(f"\n{'='*60}")
        print(f"Domain: {domain} ({len(facts)} facts)")

        # Baseline accuracy
        baseline_correct = sum(1 for f in facts if evaluate_baseline(model, tokenizer, f))
        baseline_acc = baseline_correct / len(facts)
        print(f"Baseline: {baseline_correct}/{len(facts)} = {baseline_acc:.0%}")

        # Route first fact to get steering vector
        sample_context = facts[0].get("context", facts[0].get("truth", ""))
        result = router.route_and_steer(model, tokenizer, sample_context, threshold=0.10)

        if result:
            print(f"Routed to: {', '.join(result.domains_applied)}")

            # Test with routed steering
            for alpha in [0.25, 0.5, 1.0]:
                steered_correct = sum(
                    1 for f in facts
                    if evaluate_with_steering(model, tokenizer, f, result.steering_vector, router.layer, alpha)
                )
                steered_acc = steered_correct / len(facts)
                improvement = steered_acc - baseline_acc
                status = "+" if improvement > 0 else ("=" if improvement == 0 else "-")
                print(f"α={alpha:.2f}: {steered_correct}/{len(facts)} = {steered_acc:.0%} ({status}{improvement:+.0%})")
        else:
            print("No routing match")


if __name__ == "__main__":
    main()
