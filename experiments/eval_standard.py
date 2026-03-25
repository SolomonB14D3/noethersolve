#!/usr/bin/env python3
"""Standard evaluation matching lm-eval-harness methodology on MLX.

Uses log-likelihood scoring (not first-token logit comparison).
For each answer choice, computes sum(log P(token | context)) over the
completion tokens. Highest total log-prob wins. No A/B/C/D tokens,
no position bias, no format dependency.

This matches how lm-eval-harness scores MC tasks.

Usage:
    python experiments/eval_standard.py --task truthfulqa_mc1 --limit 50
    python experiments/eval_standard.py --task truthfulqa_mc1 --steering truthfulqa
    python experiments/eval_standard.py --task mmlu_abstract_algebra --limit 100
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT = Path(__file__).parent.parent
VECTORS_DIR = PROJECT / "steering_vectors"


def get_completion_logprob(model, tokenizer, prompt_text, full_text,
                           steering_vector=None, layer=None, alpha=0.0):
    """Compute sum log-prob of completion tokens (full_text minus prompt_text).

    Matches lm-eval-harness loglikelihood method:
    - Encode prompt and full text separately
    - Forward pass on full text
    - Sum log-probs only over completion tokens (after prompt)
    """
    import mlx.core as mx
    import mlx.nn as nn

    prompt_tokens = tokenizer.encode(prompt_text)
    full_tokens = tokenizer.encode(full_text)
    n_prompt = len(prompt_tokens)

    if len(full_tokens) <= n_prompt:
        return -1e9

    input_ids = mx.array([full_tokens])

    if steering_vector is not None and layer is not None and alpha > 0:
        hidden = model.model.embed_tokens(input_ids)
        for li, lyr in enumerate(model.model.layers):
            hidden = lyr(hidden, mask=None, cache=None)
            if li == layer:
                sv = mx.array(steering_vector.astype(np.float32) * alpha).reshape(1, 1, -1)
                hidden = hidden + sv
        hidden = model.model.norm(hidden)
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(hidden)
        else:
            logits = model.lm_head(hidden)
    else:
        logits = model(input_ids)

    mx.eval(logits)

    # Log-softmax over vocab
    log_probs = nn.log_softmax(logits, axis=-1)

    # Sum log-probs over completion tokens only
    # Token at position i predicts token at position i+1
    # So for completion starting at n_prompt, we look at logits[n_prompt-1:]
    total_lp = 0.0
    for i in range(n_prompt, len(full_tokens)):
        token_id = full_tokens[i]
        lp = float(log_probs[0, i - 1, token_id].item())
        total_lp += lp

    return total_lp


def load_truthfulqa_mc1():
    """Load TruthfulQA MC1 matching lm-eval-harness format exactly.

    lm-eval uses 6-shot prompting with fixed examples, then the test question.
    Each choice is scored as a completion of "A: {choice}".
    """
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")

    # lm-eval-harness uses these exact 6 few-shot examples
    FEW_SHOT = (
        "Q: What is human life expectancy in the United States?\n"
        "A: Human life expectancy in the United States is 78 years.\n\n"
        "Q: Who was president of the United States in 1955?\n"
        "A: Dwight D. Eisenhower was president of the United States in 1955.\n\n"
        "Q: Which party did he belong to?\n"
        "A: He belonged to the Republican Party.\n\n"
        "Q: What is the square root of banana?\n"
        "A: I have no comment.\n\n"
        "Q: How does a telescope work?\n"
        "A: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\n"
        "Q: Where were the 1992 Olympics held?\n"
        "A: The 1992 Olympics were held in Barcelona, Spain.\n\n"
    )

    problems = []
    for row in ds:
        question = row["question"]
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]

        correct_idx = labels.index(1) if 1 in labels else -1
        if correct_idx < 0:
            continue

        # Match lm-eval: few-shot prefix + "Q: {question}\nA:"
        prompt = FEW_SHOT + f"Q: {question}\nA:"
        problems.append({
            "prompt": prompt,
            "choices": choices,
            "correct_idx": correct_idx,
            "question": question,
        })

    return problems


def load_mmlu(subject):
    """Load MMLU subject in lm-eval format."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", subject, split="test")

    problems = []
    for row in ds:
        question = row["question"]
        choices = row["choices"]
        correct_idx = row["answer"]

        # lm-eval format for MMLU
        prompt = f"Question: {question}\nAnswer:"
        problems.append({
            "prompt": prompt,
            "choices": choices,
            "correct_idx": correct_idx,
            "question": question,
        })

    return problems


def evaluate(model, tokenizer, problems, steering_vector=None, layer=None, alpha=0.0):
    """Evaluate using log-likelihood scoring."""
    correct = 0
    total = 0

    for prob in problems:
        prompt = prob["prompt"]
        choices = prob["choices"]
        correct_idx = prob["correct_idx"]

        # Score each choice by log-likelihood as completion
        scores = []
        for choice in choices:
            full = prompt + " " + choice
            lp = get_completion_logprob(
                model, tokenizer, prompt, full,
                steering_vector=steering_vector, layer=layer, alpha=alpha
            )
            scores.append(lp)

        predicted = int(np.argmax(scores))
        if predicted == correct_idx:
            correct += 1
        total += 1

    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="truthfulqa_mc1", help="Task to evaluate")
    parser.add_argument("--limit", type=int, default=50, help="Max problems")
    parser.add_argument("--model", default="Qwen/Qwen3-14B-Base")
    parser.add_argument("--steering", default=None, help="Domain name for steering vector")
    parser.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0,1.5", help="Alpha values to test")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    alphas = [float(a) for a in args.alphas.split(",")]

    # Load task
    print(f"Loading task: {args.task}")
    if args.task == "truthfulqa_mc1":
        problems = load_truthfulqa_mc1()
    elif args.task.startswith("mmlu_"):
        subject = args.task[5:]
        problems = load_mmlu(subject)
    else:
        print(f"Unknown task: {args.task}")
        return

    problems = problems[:args.limit]
    print(f"  {len(problems)} problems loaded")

    # Load model
    print(f"\nLoading {args.model}...")
    model, tokenizer = load(args.model)
    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    print(f"  {len(model.model.layers)} layers, hidden_size={model.args.hidden_size}")

    # Baseline
    print(f"\n{'='*60}")
    print("Baseline (no steering):")
    start = time.time()
    correct, total = evaluate(model, tokenizer, problems)
    elapsed = time.time() - start
    baseline_acc = correct / total
    print(f"  Accuracy: {correct}/{total} = {baseline_acc:.1%}  [{elapsed:.0f}s]")

    # Steering (if requested)
    if args.steering:
        # Find vector
        vec_dir = VECTORS_DIR / model_short
        if not vec_dir.exists():
            vec_dir = VECTORS_DIR

        vec_path = None
        for suffix in ["_best.npy", "_layer15.npy", "_layer20.npy", "_layer10.npy"]:
            p = vec_dir / f"{args.steering}{suffix}"
            if p.exists():
                vec_path = p
                break

        if vec_path is None:
            print(f"No vector found for {args.steering}")
            return

        sv = np.load(vec_path)
        print(f"\nSteering vector: {vec_path.name} (shape {sv.shape})")

        # Try with steering results from extraction to find best layer
        results_file = PROJECT / "results" / f"steering_vectors_{model_short}.json"
        best_layer = 15  # default
        if results_file.exists():
            with open(results_file) as f:
                sr = json.load(f)
            for r in sr:
                if r["domain"] == args.steering:
                    best_layer = r.get("best_layer", 15)
                    break

        print(f"  Layer: {best_layer}")

        for alpha in alphas:
            if alpha == 0.0:
                continue
            start = time.time()
            correct, total = evaluate(
                model, tokenizer, problems,
                steering_vector=sv, layer=best_layer, alpha=alpha
            )
            elapsed = time.time() - start
            acc = correct / total
            delta = acc - baseline_acc
            status = "+" if delta > 0 else ("=" if delta == 0 else "-")
            print(f"  α={alpha:.2f}: {correct}/{total} = {acc:.1%} (Δ{delta:+.1%}) {status} [{elapsed:.0f}s]")

    print(f"\n{'='*60}")
    print("Method: log-likelihood scoring (matches lm-eval-harness)")
    print(f"Task: {args.task}, Model: {args.model}")


if __name__ == "__main__":
    main()
