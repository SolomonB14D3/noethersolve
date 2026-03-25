#!/usr/bin/env python3
"""Fast flip pipeline using steering vectors.

1. Load facts
2. Oracle check (find failures)
3. Compute steering vector (instant)
4. Verify flips
5. Save results

Much faster than LoRA training - entire pipeline takes seconds per domain.

Usage:
    python training/fast_flip.py --facts training/generated/enzyme_kinetics_facts.json
    python training/fast_flip.py --facts-dir training/generated/
    python training/fast_flip.py --facts problems/biochemistry_facts.json --save-steering
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class FlipResult:
    domain: str
    n_facts: int
    baseline_acc: float
    steered_acc: float
    n_flipped: int
    flip_rate: float
    best_alpha: float
    steering_norm: float

    def __str__(self):
        return (
            f"{self.domain}: {self.baseline_acc:.0%} → {self.steered_acc:.0%} "
            f"(flipped {self.n_flipped}/{self.n_facts}, α={self.best_alpha})"
        )


def get_activation(model, tokenizer, text: str, layer: int):
    """Get activation at specified layer."""
    import mlx.core as mx

    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    hidden = model.model.embed_tokens(input_ids)

    for i, layer_module in enumerate(model.model.layers):
        hidden = layer_module(hidden, mask=None, cache=None)
        if i == layer:
            act = hidden[0].mean(axis=0)
            return np.array(act.astype(mx.float32))

    raise ValueError(f"Layer {layer} not found")


def compute_steering_vector(model, tokenizer, facts: list, layer: int = 15):
    """Compute steering vector from facts."""
    correct_acts = []
    incorrect_acts = []

    for fact in facts:
        context = fact.get("context", "")
        truth = fact.get("truth", fact.get("fact", ""))
        distractors = fact.get("distractors", [])

        if not distractors:
            continue

        base = f"{context}\n\nAnswer: " if context else "Answer: "

        correct_act = get_activation(model, tokenizer, base + truth, layer)
        incorrect_act = get_activation(model, tokenizer, base + distractors[0], layer)

        correct_acts.append(correct_act)
        incorrect_acts.append(incorrect_act)

    if not correct_acts:
        return None

    return np.mean(correct_acts, axis=0) - np.mean(incorrect_acts, axis=0)


def evaluate_fact(model, tokenizer, fact, steering_vector, layer: int, alpha: float):
    """Evaluate single fact with optional steering."""
    import mlx.core as mx

    context = fact.get("context", "")
    truth = fact.get("truth", fact.get("fact", ""))
    distractors = fact.get("distractors", [])

    if not distractors:
        return None

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

    return np.argmax(option_logits) == 0  # Truth is option A


def run_flip_pipeline(
    model,
    tokenizer,
    facts: list,
    domain: str,
    layer: int = 15,
    alphas: list = [0.1, 0.25, 0.5, 0.75, 1.0],
) -> FlipResult:
    """Run full flip pipeline on facts."""

    # 1. Baseline evaluation
    baseline_results = [evaluate_fact(model, tokenizer, f, None, layer, 0) for f in facts]
    baseline_results = [r for r in baseline_results if r is not None]
    baseline_acc = sum(baseline_results) / len(baseline_results) if baseline_results else 0

    # 2. Find failures (oracle)
    failures = [f for f, passed in zip(facts, baseline_results) if not passed]
    print(f"  Baseline: {sum(baseline_results)}/{len(baseline_results)} ({baseline_acc:.0%})")
    print(f"  Failures: {len(failures)}")

    if not failures:
        return FlipResult(
            domain=domain,
            n_facts=len(facts),
            baseline_acc=baseline_acc,
            steered_acc=baseline_acc,
            n_flipped=0,
            flip_rate=0,
            best_alpha=0,
            steering_norm=0,
        )

    # 3. Compute steering vector from ALL facts (not just failures)
    steering = compute_steering_vector(model, tokenizer, facts, layer)
    if steering is None:
        return FlipResult(
            domain=domain, n_facts=len(facts), baseline_acc=baseline_acc,
            steered_acc=baseline_acc, n_flipped=0, flip_rate=0,
            best_alpha=0, steering_norm=0,
        )

    steering_norm = np.linalg.norm(steering)
    print(f"  Steering norm: {steering_norm:.2f}")

    # 4. Find best alpha
    best_acc = baseline_acc
    best_alpha = 0
    best_flipped = 0

    for alpha in alphas:
        steered_results = [evaluate_fact(model, tokenizer, f, steering, layer, alpha) for f in facts]
        steered_results = [r for r in steered_results if r is not None]
        steered_acc = sum(steered_results) / len(steered_results) if steered_results else 0

        # Count flips (was wrong, now right)
        flipped = sum(1 for base, steer in zip(baseline_results, steered_results) if not base and steer)

        if steered_acc > best_acc:
            best_acc = steered_acc
            best_alpha = alpha
            best_flipped = flipped

        print(f"  α={alpha}: {steered_acc:.0%} (flipped {flipped})")

    return FlipResult(
        domain=domain,
        n_facts=len(facts),
        baseline_acc=baseline_acc,
        steered_acc=best_acc,
        n_flipped=best_flipped,
        flip_rate=best_flipped / len(failures) if failures else 0,
        best_alpha=best_alpha,
        steering_norm=steering_norm,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", help="Facts JSON file")
    parser.add_argument("--facts-dir", help="Directory of facts files")
    parser.add_argument("--layer", type=int, default=15, help="Steering layer")
    parser.add_argument("--save-steering", action="store_true", help="Save steering vectors")
    parser.add_argument("--output", default="steering_vectors", help="Output directory for vectors")
    args = parser.parse_args()

    from mlx_lm import load

    print("Loading model...")
    model, tokenizer = load("Qwen/Qwen3-14B-Base")

    # Collect facts files
    facts_files = []
    if args.facts:
        facts_files.append(Path(args.facts))
    if args.facts_dir:
        facts_files.extend(Path(args.facts_dir).glob("*_facts*.json"))

    if not facts_files:
        parser.print_help()
        return

    results = []
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    for facts_file in sorted(facts_files):
        print(f"\n{'='*60}")
        print(f"Processing: {facts_file.name}")

        with open(facts_file) as f:
            data = json.load(f)

        facts = data.get("facts", data.get("verifications", []))
        domain = data.get("domain", facts_file.stem.replace("_facts", ""))

        if not facts:
            print("  No facts found")
            continue

        result = run_flip_pipeline(model, tokenizer, facts, domain, args.layer)
        results.append(result)
        print(f"  Result: {result}")

        # Save steering vector if requested
        if args.save_steering and result.steering_norm > 0:
            steering = compute_steering_vector(model, tokenizer, facts, args.layer)
            if steering is not None:
                vec_path = output_dir / f"{domain}_layer{args.layer}.npy"
                np.save(vec_path, steering)
                print(f"  Saved: {vec_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    improved = [r for r in results if r.steered_acc > r.baseline_acc]
    print(f"Improved: {len(improved)}/{len(results)} domains")

    for r in sorted(improved, key=lambda x: -(x.steered_acc - x.baseline_acc)):
        delta = r.steered_acc - r.baseline_acc
        print(f"  {r.domain}: {r.baseline_acc:.0%} → {r.steered_acc:.0%} (+{delta:.0%})")


if __name__ == "__main__":
    main()
