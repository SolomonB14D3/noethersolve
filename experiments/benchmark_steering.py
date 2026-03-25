#!/usr/bin/env python3
"""Benchmark steering vectors on standard evals.

Runs canonical test splits with standard methodology so results are
reproducible by anyone. Reports baseline vs steered accuracy.

Usage:
    python experiments/benchmark_steering.py --bench mmlu
    python experiments/benchmark_steering.py --bench truthfulqa
    python experiments/benchmark_steering.py --bench all
"""
import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT = Path(__file__).parent.parent
VECTORS_DIR = PROJECT / "steering_vectors" / "qwen3_4b_base"
RESULTS_DIR = PROJECT / "results"


def load_steering_info():
    """Load per-domain optimal layer and alpha from extraction results."""
    results_file = RESULTS_DIR / "steering_vectors_qwen3_4b_base.json"
    if not results_file.exists():
        results_file = RESULTS_DIR / "steering_vectors_v2.json"
    if not results_file.exists():
        return {}

    with open(results_file) as f:
        results = json.load(f)

    info = {}
    for r in results:
        if r.get("improvement", 0) > 0:
            info[r["domain"]] = {
                "layer": r.get("best_layer", 15),
                "alpha": r.get("best_alpha", 1.0),
            }
    return info


def score_mc(model, tokenizer, prompt, options, steering_vector=None, layer=None, alpha=0.0):
    """Score MC options. Returns index of highest-scoring option.

    IMPORTANT: Uses exact same format as extract_vectors_fast.py to avoid
    token mismatch. Format: "Which is correct?\nA) opt\nB) opt\n...\nAnswer: "
    Token checked: bare letter "A" (not " A" or "A)" or "A.").
    """
    import mlx.core as mx

    # Build prompt in EXACT same format as extraction script
    full = (prompt + "\n\n" if prompt else "") + "Which is correct?\n"
    for i, opt in enumerate(options):
        full += f"{chr(65+i)}) {opt}\n"
    full += "Answer: "

    tokens = tokenizer.encode(full)
    input_ids = mx.array([tokens])

    if steering_vector is not None and alpha > 0:
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

    # Use bare letter tokens — same as extraction script
    opt_toks = [tokenizer.encode(chr(65+i))[-1] for i in range(len(options))]
    opt_logits = [float(logits[0, -1, t].item()) for t in opt_toks]

    return int(np.argmax(opt_logits))


def bench_mmlu(model, tokenizer, steering_info, max_per_subject=30):
    """MMLU benchmark — 57 subjects, MC accuracy."""
    from datasets import load_dataset

    print("Loading MMLU test split...")
    ds = load_dataset("cais/mmlu", "all", split="test")

    # Group by subject
    by_subject = defaultdict(list)
    for row in ds:
        by_subject[row["subject"]].append(row)

    print(f"  {len(by_subject)} subjects, {len(ds)} total questions")
    print(f"  Capping at {max_per_subject} per subject for speed")

    baseline_correct = 0
    steered_correct = 0
    total = 0
    per_subject = {}

    for subj in sorted(by_subject.keys()):
        rows = by_subject[subj][:max_per_subject]
        subj_baseline = 0
        subj_steered = 0

        # Find best vector for this subject
        vec_domain = subj.replace(" ", "_").lower()
        vec_path = VECTORS_DIR / f"{vec_domain}_best.npy"
        sv = None
        layer = 15
        alpha = 1.0
        if vec_path.exists() and vec_domain in steering_info:
            sv = np.load(vec_path)
            layer = steering_info[vec_domain]["layer"]
            alpha = steering_info[vec_domain]["alpha"]

        for row in rows:
            choices = row["choices"]
            answer_idx = row["answer"]

            # Baseline
            pred = score_mc(model, tokenizer, row["question"], choices)
            if pred == answer_idx:
                subj_baseline += 1
                baseline_correct += 1

            # Steered (only if vector exists and helped)
            if sv is not None:
                pred_s = score_mc(model, tokenizer, row["question"], choices,
                                  steering_vector=sv, layer=layer, alpha=alpha)
            else:
                pred_s = pred  # No vector = same as baseline

            if pred_s == answer_idx:
                subj_steered += 1
                steered_correct += 1

            total += 1

        n = len(rows)
        b_acc = subj_baseline / n
        s_acc = subj_steered / n
        delta = s_acc - b_acc
        marker = "+" if delta > 0 else ("=" if delta == 0 else "-")
        has_vec = "V" if sv is not None else " "
        print(f"  {has_vec} {subj:<40s} {b_acc:.0%} -> {s_acc:.0%} ({delta:+.0%}) [{n}q] {marker}")

        per_subject[subj] = {"baseline": b_acc, "steered": s_acc, "n": n, "has_vector": sv is not None}

    return {
        "benchmark": "mmlu",
        "baseline_acc": baseline_correct / total,
        "steered_acc": steered_correct / total,
        "total": total,
        "per_subject": per_subject,
    }


def bench_truthfulqa(model, tokenizer, steering_info):
    """TruthfulQA MC1 — pick the single correct answer."""
    from datasets import load_dataset

    print("Loading TruthfulQA...")
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")

    # Find best TruthfulQA vector
    sv = None
    layer = 20
    alpha = 1.5
    for name in ["truthfulqa", "truthfulqa_00", "truthfulqa_03"]:
        vec_path = VECTORS_DIR / f"{name}_best.npy"
        if vec_path.exists() and name in steering_info:
            sv = np.load(vec_path)
            layer = steering_info[name]["layer"]
            alpha = steering_info[name]["alpha"]
            print(f"  Using vector: {name} (L{layer}, a={alpha})")
            break

    baseline_correct = 0
    steered_correct = 0
    total = 0

    for row in ds:
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct_idx = labels.index(1) if 1 in labels else -1
        if correct_idx < 0:
            continue

        # Baseline
        pred = score_mc(model, tokenizer, row["question"], choices[:5])  # Cap at 5 options
        if pred == correct_idx:
            baseline_correct += 1

        # Steered
        if sv is not None:
            pred_s = score_mc(model, tokenizer, row["question"], choices[:5],
                              steering_vector=sv, layer=layer, alpha=alpha)
        else:
            pred_s = pred

        if pred_s == correct_idx:
            steered_correct += 1
        total += 1

    return {
        "benchmark": "truthfulqa_mc1",
        "baseline_acc": baseline_correct / total,
        "steered_acc": steered_correct / total,
        "total": total,
    }


def bench_winogrande(model, tokenizer, steering_info):
    """WinoGrande — coreference resolution."""
    from datasets import load_dataset

    print("Loading WinoGrande...")
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")

    sv = None
    layer = 15
    alpha = 1.5
    for name in ["winogrande", "winogrande_00", "winogrande_06"]:
        vec_path = VECTORS_DIR / f"{name}_best.npy"
        if vec_path.exists() and name in steering_info:
            sv = np.load(vec_path)
            layer = steering_info[name]["layer"]
            alpha = steering_info[name]["alpha"]
            print(f"  Using vector: {name} (L{layer}, a={alpha})")
            break

    baseline_correct = 0
    steered_correct = 0
    total = 0
    cap = 500  # Speed cap

    for row in list(ds)[:cap]:
        options = [row["option1"], row["option2"]]
        answer_idx = int(row["answer"]) - 1
        if answer_idx < 0 or answer_idx >= 2:
            continue

        pred = score_mc(model, tokenizer, row["sentence"], options)
        if pred == answer_idx:
            baseline_correct += 1

        if sv is not None:
            pred_s = score_mc(model, tokenizer, row["sentence"], options,
                              steering_vector=sv, layer=layer, alpha=alpha)
        else:
            pred_s = pred
        if pred_s == answer_idx:
            steered_correct += 1
        total += 1

    return {
        "benchmark": "winogrande",
        "baseline_acc": baseline_correct / total,
        "steered_acc": steered_correct / total,
        "total": total,
    }


def bench_boolq(model, tokenizer, steering_info):
    """BoolQ — yes/no reading comprehension."""
    from datasets import load_dataset

    print("Loading BoolQ...")
    ds = load_dataset("google/boolq", split="validation")

    sv = None
    layer = 15
    alpha = 1.0
    for name in ["boolq", "boolq_00", "boolq_01"]:
        vec_path = VECTORS_DIR / f"{name}_best.npy"
        if vec_path.exists() and name in steering_info:
            sv = np.load(vec_path)
            layer = steering_info[name]["layer"]
            alpha = steering_info[name]["alpha"]
            print(f"  Using vector: {name} (L{layer}, a={alpha})")
            break

    baseline_correct = 0
    steered_correct = 0
    total = 0
    cap = 500

    for row in list(ds)[:cap]:
        options = ["Yes", "No"]
        answer_idx = 0 if row["answer"] else 1

        pred = score_mc(model, tokenizer,
                        row["passage"][:300] + "\n\nQ: " + row["question"],
                        options)
        if pred == answer_idx:
            baseline_correct += 1

        if sv is not None:
            pred_s = score_mc(model, tokenizer,
                              row["passage"][:300] + "\n\nQ: " + row["question"],
                              options, steering_vector=sv, layer=layer, alpha=alpha)
        else:
            pred_s = pred
        if pred_s == answer_idx:
            steered_correct += 1
        total += 1

    return {
        "benchmark": "boolq",
        "baseline_acc": baseline_correct / total,
        "steered_acc": steered_correct / total,
        "total": total,
    }


BENCHMARKS = {
    "mmlu": bench_mmlu,
    "truthfulqa": bench_truthfulqa,
    "winogrande": bench_winogrande,
    "boolq": bench_boolq,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="all", help="Benchmark to run (mmlu, truthfulqa, winogrande, boolq, all)")
    parser.add_argument("--model", default="Qwen/Qwen3-14B-Base")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)
    print(f"  {len(model.model.layers)} layers, hidden_size={model.args.hidden_size}")

    steering_info = load_steering_info()
    print(f"  {len(steering_info)} improved vectors available")

    benches = list(BENCHMARKS.keys()) if args.bench == "all" else [args.bench]
    all_results = []

    for bench_name in benches:
        print(f"\n{'='*70}")
        print(f"  BENCHMARK: {bench_name.upper()}")
        print(f"{'='*70}")
        start = time.time()

        result = BENCHMARKS[bench_name](model, tokenizer, steering_info)
        elapsed = time.time() - start

        result["elapsed"] = elapsed
        all_results.append(result)

        b = result["baseline_acc"]
        s = result["steered_acc"]
        d = s - b
        print(f"\n  {bench_name}: {b:.1%} -> {s:.1%} ({d:+.1%}) [{result['total']} questions, {elapsed:.0f}s]")

    # Save
    out_path = RESULTS_DIR / "benchmark_steering_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY — Steering Vector Benchmark")
    print(f"{'='*70}")
    print(f"  {'Benchmark':<20s} {'Baseline':>10s} {'Steered':>10s} {'Delta':>10s} {'N':>6s}")
    print(f"  {'-'*56}")
    for r in all_results:
        b = r["baseline_acc"]
        s = r["steered_acc"]
        d = s - b
        print(f"  {r['benchmark']:<20s} {b:>9.1%} {s:>9.1%} {d:>+9.1%} {r['total']:>6d}")

    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
