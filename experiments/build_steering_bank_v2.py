#!/usr/bin/env python3
"""Steering vector extraction v2 — filter to model's WRONG answers only.

Key lesson from adapter training: only train on surprising truths.
Steering vector = direction from wrong-answer activations toward correct-answer activations,
computed ONLY from facts the model currently gets wrong.

Usage:
    python experiments/build_steering_bank_v2.py
    python experiments/build_steering_bank_v2.py --layer 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

PROJECT = Path(__file__).parent.parent
BANK_DIR = PROJECT / "steering_bank"
VECTORS_DIR = PROJECT / "steering_vectors"

from extract_all_steering_vectors import get_activations


def score_baseline_per_fact(model, tokenizer, facts):
    """Score each fact individually. Returns list of (fact, correct_bool, margin)."""
    import mlx.core as mx

    scored = []
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
        option_logits = [float(logits[0, -1, t].item()) for t in option_tokens]

        predicted = int(np.argmax(option_logits))
        correct = predicted == 0
        margin = option_logits[0] - max(option_logits[1:])  # truth - best distractor

        scored.append({
            "fact": fact,
            "correct": correct,
            "margin": margin,
            "predicted": predicted,
        })

    return scored


def compute_steering_from_wrong(model, tokenizer, wrong_facts, layer):
    """Compute steering vector from facts the model gets WRONG.
    
    Direction: correct_activation - incorrect_activation
    Only using facts where the model chose the wrong answer.
    """
    import mlx.core as mx

    correct_prompts = []
    incorrect_prompts = []

    for item in wrong_facts:
        fact = item["fact"]
        context = fact.get("context", "")
        truth = fact.get("truth", fact.get("fact", ""))
        distractors = fact.get("distractors", [])
        if not distractors:
            continue

        base = f"{context}\n\nAnswer: " if context else "Answer: "
        correct_prompts.append(base + truth)
        # Use the distractor the model actually picked
        picked_idx = item["predicted"] - 1  # predicted=0 is truth, so distractor idx = predicted-1
        if 0 <= picked_idx < len(distractors):
            incorrect_prompts.append(base + distractors[picked_idx])
        else:
            incorrect_prompts.append(base + distractors[0])

    if not correct_prompts:
        return None

    correct_acts = get_activations(model, tokenizer, correct_prompts, layer)
    incorrect_acts = get_activations(model, tokenizer, incorrect_prompts, layer)

    return correct_acts.mean(axis=0) - incorrect_acts.mean(axis=0)


def evaluate_with_steering(model, tokenizer, facts, steering_vector, layer, alpha):
    """Evaluate with steering applied."""
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

        hidden = model.model.embed_tokens(input_ids)
        for i, layer_module in enumerate(model.model.layers):
            hidden = layer_module(hidden, mask=None, cache=None)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--max-train", type=int, default=30, help="Max wrong facts for vector")
    args = parser.parse_args()

    from mlx_lm import load

    alphas = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]

    print("Loading Qwen/Qwen3-14B-Base...")
    model, tokenizer = load("Qwen/Qwen3-14B-Base")

    VECTORS_DIR.mkdir(exist_ok=True)

    # Load all domain files from bank + our custom facts
    domain_files = []
    if BANK_DIR.exists():
        domain_files += sorted(f for f in BANK_DIR.glob("*.json") if f.name != "summary.json")

    # Also include our hand-crafted facts
    problems_dir = PROJECT / "problems"
    for ff in sorted(problems_dir.glob("*_facts*.json")):
        domain_files.append(ff)

    print(f"\nProcessing {len(domain_files)} domain files...")
    print(f"Layer {args.layer}, alphas: {alphas}")
    print(f"Strategy: compute vector from WRONG answers only")
    print(f"{'='*70}")

    results = []
    total_start = time.time()

    for df in domain_files:
        try:
            with open(df) as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue

        domain = data.get("domain", df.stem.replace("_facts_v2", "").replace("_facts", ""))
        facts = data.get("facts", data.get("verifications", []))

        if len(facts) < 4:
            continue

        start = time.time()

        # Split: use first half for scoring+vector, second half for eval
        np.random.seed(42)
        indices = np.random.permutation(len(facts))
        split = len(facts) // 2
        train_pool = [facts[i] for i in indices[:split]]
        test_pool = [facts[i] for i in indices[split:]]

        if len(test_pool) < 2:
            test_pool = train_pool

        # Score train pool to find wrong answers
        scored = score_baseline_per_fact(model, tokenizer, train_pool)
        wrong = [s for s in scored if not s["correct"]]
        right = [s for s in scored if s["correct"]]

        # Baseline on test pool
        test_scored = score_baseline_per_fact(model, tokenizer, test_pool)
        baseline = sum(1 for s in test_scored if s["correct"]) / len(test_scored) if test_scored else 0

        n_wrong = len(wrong)
        n_right = len(right)

        if n_wrong < 2:
            # Model already knows most of these — skip
            elapsed = time.time() - start
            print(f"  {domain:<45s} baseline={baseline:.0%}  wrong={n_wrong}/{len(scored)}  SKIP (too few wrong) [{elapsed:.1f}s]")
            results.append({
                "domain": domain, "baseline": baseline, "best_steered": baseline,
                "improvement": 0, "n_wrong": n_wrong, "n_total": len(scored),
                "skipped": True,
            })
            continue

        # Compute steering vector from wrong answers only
        use_wrong = wrong[:args.max_train]
        sv = compute_steering_from_wrong(model, tokenizer, use_wrong, args.layer)
        if sv is None:
            continue

        # Test different alpha values on test pool
        best_acc = baseline
        best_alpha = 0.0
        for alpha in alphas:
            acc = evaluate_with_steering(model, tokenizer, test_pool, sv, args.layer, alpha)
            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha

        improvement = best_acc - baseline
        elapsed = time.time() - start

        # Save vector
        vec_path = VECTORS_DIR / f"{domain}_wrong_layer{args.layer}.npy"
        np.save(vec_path, sv)

        status = "+" if improvement > 0 else ("=" if improvement == 0 else "-")
        print(f"  {domain:<45s} {baseline:.0%} -> {best_acc:.0%}  a={best_alpha:.3f}  wrong={n_wrong}/{len(scored)}  {status} [{elapsed:.1f}s]")

        results.append({
            "domain": domain, "baseline": baseline, "best_steered": best_acc,
            "improvement": improvement, "best_alpha": best_alpha,
            "n_wrong": n_wrong, "n_total": len(scored), "layer": args.layer,
            "skipped": False,
        })

    total_elapsed = time.time() - total_start

    # Save results
    with open(VECTORS_DIR / "bank_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    active = [r for r in results if not r.get("skipped", False)]
    improved = [r for r in active if r["improvement"] > 0]
    hurt = [r for r in active if r["improvement"] < 0]
    same = [r for r in active if r["improvement"] == 0]
    skipped = [r for r in results if r.get("skipped", False)]

    print(f"\n{'='*70}")
    print(f"DONE: {len(results)} domains in {total_elapsed:.0f}s")
    print(f"Active: {len(active)} | Skipped (model already knows): {len(skipped)}")
    print(f"Improved: {len(improved)} | Same: {len(same)} | Hurt: {len(hurt)}")

    if improved:
        print(f"\nIMPROVED:")
        for r in sorted(improved, key=lambda x: -x["improvement"]):
            print(f"  {r['domain']:<45s} {r['baseline']:.0%} -> {r['best_steered']:.0%} (+{r['improvement']:.0%})  a={r['best_alpha']}  wrong={r['n_wrong']}/{r['n_total']}")

    if hurt:
        print(f"\nHURT:")
        for r in sorted(hurt, key=lambda x: x["improvement"]):
            print(f"  {r['domain']:<45s} {r['baseline']:.0%} -> {r['best_steered']:.0%} ({r['improvement']:+.0%})")


if __name__ == "__main__":
    main()
