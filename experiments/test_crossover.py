#!/usr/bin/env python3
"""Test cross-domain transfer with steering vectors.

Does a steering vector from domain A help with domain B?
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_lm import load
import mlx.core as mx
from noethersolve.steering_router import SteeringRouter


def evaluate_with_steering(model, tokenizer, facts, steering_vector, layer, alpha):
    """Evaluate facts with a specific steering vector."""
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

        hidden = model.model.embed_tokens(input_ids)

        steering_tensor = None
        if steering_vector is not None:
            steering_tensor = mx.array(steering_vector.astype(np.float32) * alpha).reshape(1, 1, -1)

        for i, layer_module in enumerate(model.model.layers):
            hidden = layer_module(hidden, mask=None, cache=None)
            if steering_tensor is not None and i == layer:
                hidden = hidden + steering_tensor

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

    return correct / total if total > 0 else 0


def main():
    print("Loading model...")
    model, tokenizer = load("Qwen/Qwen3-4B-Base")

    print("Loading steering bank...")
    router = SteeringRouter.load("steering_bank.npz")
    print(f"  {len(router)} domains\n")

    # Define domain pairs to test for crossover
    # Format: (source_domain, target_domain, expected_relation)
    crossover_pairs = [
        # Physics family
        ("hamiltonian", "3body_conservation", "both mechanics"),
        ("3body_conservation", "vortex_pair", "both conservation"),
        ("ns_regularity", "physics_fundamentals_2d_turbulence", "both fluid"),

        # Biology family
        ("biochemistry", "chemistry", "related"),
        ("genetics_therapeutics", "protein_structure", "both bio"),

        # Math family
        ("number_theory_conjectures", "millennium_problems", "both math"),
        ("elliptic_curves", "intersection_theory", "both algebraic"),

        # Unrelated (negative control)
        ("networking", "biochemistry", "unrelated"),
        ("operating_systems", "quantum_gravity", "unrelated"),
    ]

    facts_dir = Path("problems")
    results = []

    for source, target, relation in crossover_pairs:
        # Load target facts
        target_file = facts_dir / f"{target}_facts.json"
        if not target_file.exists():
            target_file = facts_dir / f"{target}_facts_v2.json"
        if not target_file.exists():
            continue

        with open(target_file) as f:
            data = json.load(f)
        facts = data.get("facts", data.get("verifications", []))[:10]

        if not facts or source not in router.steering_bank:
            continue

        # Baseline (no steering)
        baseline = evaluate_with_steering(model, tokenizer, facts, None, router.layer, 0)

        # Same-domain steering
        if target in router.steering_bank:
            same_domain = evaluate_with_steering(
                model, tokenizer, facts,
                router.steering_bank[target], router.layer, 0.5
            )
        else:
            same_domain = baseline

        # Cross-domain steering
        cross_domain = evaluate_with_steering(
            model, tokenizer, facts,
            router.steering_bank[source], router.layer, 0.5
        )

        # Calculate transfer
        cross_delta = cross_domain - baseline
        same_delta = same_domain - baseline

        print(f"{source} → {target} ({relation})")
        print(f"  Baseline:     {baseline:.0%}")
        print(f"  Same-domain:  {same_domain:.0%} ({same_delta:+.0%})")
        print(f"  Cross-domain: {cross_domain:.0%} ({cross_delta:+.0%})")

        if cross_delta > 0:
            print(f"  ✓ POSITIVE TRANSFER")
        elif cross_delta < 0:
            print(f"  ✗ Negative transfer")
        print()

        results.append({
            "source": source,
            "target": target,
            "relation": relation,
            "baseline": baseline,
            "same_domain": same_domain,
            "cross_domain": cross_domain,
            "cross_delta": cross_delta,
        })

    # Summary
    print("="*60)
    print("CROSSOVER SUMMARY")
    print("="*60)

    positive = [r for r in results if r["cross_delta"] > 0]
    negative = [r for r in results if r["cross_delta"] < 0]
    neutral = [r for r in results if r["cross_delta"] == 0]

    print(f"Positive transfer: {len(positive)}/{len(results)}")
    print(f"Negative transfer: {len(negative)}/{len(results)}")
    print(f"Neutral: {len(neutral)}/{len(results)}")

    if positive:
        print("\nPositive transfers:")
        for r in sorted(positive, key=lambda x: -x["cross_delta"]):
            print(f"  {r['source']} → {r['target']}: {r['cross_delta']:+.0%}")

    if negative:
        print("\nNegative transfers:")
        for r in sorted(negative, key=lambda x: x["cross_delta"]):
            print(f"  {r['source']} → {r['target']}: {r['cross_delta']:+.0%}")


if __name__ == "__main__":
    main()
