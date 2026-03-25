#!/usr/bin/env python3
"""Test if steering vectors improve code generation (HumanEval).

Key hypothesis: The "correctness" direction in activation space is universal.
Vectors computed from MC facts (TruthfulQA, MMLU, etc.) should also nudge
code generation toward correct outputs — even without code-specific training.

Two experiments:
1. Apply existing cross-domain vectors during code generation
2. Compute code-specific vector from correct vs incorrect solutions

Usage:
    python experiments/test_humaneval_steering.py
    python experiments/test_humaneval_steering.py --model Qwen/Qwen3-14B-Base
    python experiments/test_humaneval_steering.py --vector truthfulqa  # use specific vector
"""
import argparse
import json
import sys
import time
import tempfile
import subprocess
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT = Path(__file__).parent.parent
VECTORS_DIR = PROJECT / "steering_vectors"
RESULTS_DIR = PROJECT / "results"


def load_humaneval():
    """Download and load HumanEval problems."""
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = []
    for row in ds:
        problems.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        })
    return problems


def generate_solution(model, tokenizer, prompt, max_tokens=512,
                      steering_vector=None, layer=None, alpha=0.0):
    """Generate code with optional steering."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    generated = list(tokens)

    for _ in range(max_tokens):
        input_ids = mx.array([generated])

        if steering_vector is not None and layer is not None and alpha > 0:
            # Forward with steering
            hidden = model.model.embed_tokens(input_ids)
            for i, lyr in enumerate(model.model.layers):
                hidden = lyr(hidden, mask=None, cache=None)
                if i == layer:
                    sv = mx.array(steering_vector.astype(np.float32) * alpha).reshape(1, 1, -1)
                    hidden = hidden + sv
            hidden = model.model.norm(hidden)
            if model.args.tie_word_embeddings:
                logits = model.model.embed_tokens.as_linear(hidden)
            else:
                logits = model.lm_head(hidden)
            mx.eval(logits)
        else:
            logits = model(input_ids)
            mx.eval(logits)

        next_token = int(logits[0, -1].argmax().item())

        # Stop conditions
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)

        # Stop if we see a new function definition or class (end of solution)
        decoded = tokenizer.decode(generated[len(tokens):])
        if "\ndef " in decoded or "\nclass " in decoded:
            # Keep up to the new definition
            idx = decoded.index("\ndef ") if "\ndef " in decoded else decoded.index("\nclass ")
            decoded = decoded[:idx]
            break

    solution = tokenizer.decode(generated[len(tokens):])
    return solution


def check_solution(prompt, solution, test_code, entry_point):
    """Run the test code to check if solution is correct."""
    full_code = prompt + solution + "\n" + test_code + f"\ncheck({entry_point})"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
        finally:
            Path(f.name).unlink(missing_ok=True)


def find_best_vectors(model_short):
    """Find available steering vectors, sorted by improvement."""
    results_file = RESULTS_DIR / f"steering_vectors_{model_short}.json"
    if not results_file.exists():
        results_file = RESULTS_DIR / "steering_vectors_v2.json"
    if not results_file.exists():
        return []

    with open(results_file) as f:
        results = json.load(f)

    # Sort by improvement, return top vectors
    improved = [r for r in results if r.get("improvement", 0) > 0]
    return sorted(improved, key=lambda x: -x["improvement"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B-Base")
    parser.add_argument("--vector", default=None, help="Specific vector domain to use")
    parser.add_argument("--n-problems", type=int, default=20, help="Number of problems to test")
    parser.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0", help="Alpha values")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    import mlx.core as mx
    from mlx_lm import load

    print("Loading HumanEval...")
    problems = load_humaneval()[:args.n_problems]
    print(f"  {len(problems)} problems loaded")

    print(f"\nLoading {args.model}...")
    model, tokenizer = load(args.model)
    n_layers = len(model.model.layers)
    print(f"  {n_layers} layers")

    # Find vectors to test
    vectors_to_test = []

    if args.vector:
        # Specific vector requested
        vec_dir = VECTORS_DIR / model_short
        if not vec_dir.exists():
            vec_dir = VECTORS_DIR
        vec_path = vec_dir / f"{args.vector}_best.npy"
        if not vec_path.exists():
            # Try layer-specific
            for layer in [15, 20, 10]:
                vec_path = vec_dir / f"{args.vector}_layer{layer}.npy"
                if vec_path.exists():
                    break
        if vec_path.exists():
            vectors_to_test.append(("specified", args.vector, vec_path, 20))
        else:
            print(f"Vector not found for {args.vector}")
            return
    else:
        # Use top cross-domain vectors
        top_vectors = find_best_vectors(model_short)
        vec_dir = VECTORS_DIR / model_short
        if not vec_dir.exists():
            vec_dir = VECTORS_DIR

        for r in top_vectors[:5]:
            domain = r["domain"]
            layer = r.get("best_layer", 15)
            vec_path = vec_dir / f"{domain}_best.npy"
            if not vec_path.exists():
                vec_path = vec_dir / f"{domain}_layer{layer}.npy"
            if vec_path.exists():
                vectors_to_test.append(("cross-domain", domain, vec_path, layer))

    if not vectors_to_test:
        print("No vectors found! Run extract_vectors_fast.py first.")
        return

    print(f"\nTesting {len(vectors_to_test)} vectors × {len(alphas)} alphas × {len(problems)} problems")
    print(f"{'='*70}")

    all_results = []

    # Baseline (no steering)
    print(f"\n--- Baseline (no steering) ---")
    baseline_pass = 0
    for i, prob in enumerate(problems):
        sol = generate_solution(model, tokenizer, prob["prompt"])
        passed = check_solution(prob["prompt"], sol, prob["test"], prob["entry_point"])
        baseline_pass += int(passed)
        status = "✓" if passed else "✗"
        print(f"  {prob['task_id']}: {status}")

    baseline_rate = baseline_pass / len(problems)
    print(f"  Baseline pass@1: {baseline_pass}/{len(problems)} = {baseline_rate:.1%}")

    # Test each vector at each alpha
    for vec_type, domain, vec_path, layer in vectors_to_test:
        sv = np.load(vec_path)

        # Check dimension compatibility
        d_model = model.args.hidden_size
        if sv.shape[0] != d_model:
            print(f"\n  {domain}: dim mismatch ({sv.shape[0]} vs {d_model}), skipping")
            continue

        for alpha in alphas:
            if alpha == 0.0:
                continue  # Already have baseline

            print(f"\n--- {domain} (L{layer}, α={alpha}) ---")
            steered_pass = 0
            for i, prob in enumerate(problems):
                sol = generate_solution(
                    model, tokenizer, prob["prompt"],
                    steering_vector=sv, layer=layer, alpha=alpha
                )
                passed = check_solution(prob["prompt"], sol, prob["test"], prob["entry_point"])
                steered_pass += int(passed)
                status = "✓" if passed else "✗"
                print(f"  {prob['task_id']}: {status}")

            steered_rate = steered_pass / len(problems)
            delta = steered_rate - baseline_rate
            print(f"  Steered pass@1: {steered_pass}/{len(problems)} = {steered_rate:.1%} (Δ{delta:+.1%})")

            all_results.append({
                "vector_type": vec_type,
                "domain": domain,
                "layer": layer,
                "alpha": alpha,
                "baseline_pass": baseline_pass,
                "steered_pass": steered_pass,
                "n_problems": len(problems),
                "baseline_rate": baseline_rate,
                "steered_rate": steered_rate,
                "delta": delta,
            })

    # Save results
    results_path = RESULTS_DIR / f"humaneval_steering_{model_short}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — HumanEval Steering ({args.model})")
    print(f"{'='*70}")
    print(f"Baseline pass@1: {baseline_rate:.1%}")

    if all_results:
        best = max(all_results, key=lambda x: x["steered_rate"])
        print(f"Best steered:     {best['steered_rate']:.1%} ({best['domain']}, L{best['layer']}, α={best['alpha']})")
        print(f"Delta:            {best['delta']:+.1%}")

        if best["delta"] > 0:
            print(f"\n✓ Cross-domain steering IMPROVES code generation!")
        elif best["delta"] == 0:
            print(f"\n= No effect on code generation")
        else:
            print(f"\n✗ Steering hurts code generation")
    else:
        print("No results to compare")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
