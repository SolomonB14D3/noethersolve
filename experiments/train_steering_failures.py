#!/usr/bin/env python3
"""Train adapters on domains where steering vectors failed.

Reads steering_vectors_v2.json, finds domains with 0 improvement,
trains logit-space adapters on each. Saves to adapters/ dir.

Usage:
    python experiments/train_steering_failures.py
    python experiments/train_steering_failures.py --max-domains 50
    python experiments/train_steering_failures.py --min-facts 10
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT = Path(__file__).parent.parent
BANK_DIR = PROJECT / "steering_bank"
ADAPTERS_DIR = PROJECT / "adapters"
RESULTS_DIR = PROJECT / "results"


def find_facts_file(domain):
    """Find facts file for a domain."""
    # Check bank first
    for pattern in [f"{domain}.json", f"{domain}_facts_v2.json", f"{domain}_facts.json"]:
        for d in [BANK_DIR, PROJECT / "problems"]:
            fp = d / pattern
            if fp.exists():
                return fp
    return None


def train_adapter_on_facts(model, tokenizer, facts, steps=2000, lr=4e-6):
    """Train a logit-space SnapOn adapter."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from noethersolve.adapter import SnapOnConfig, create_adapter

    config = SnapOnConfig(vocab_size=151936, d_inner=64, mode="logit")
    adapter = create_adapter(config)
    optimizer = optim.Adam(learning_rate=lr)

    # Build training data
    training_data = []
    for fact in facts:
        ctx = fact.get("context", "")
        truth = fact.get("truth", fact.get("fact", ""))
        dist = fact.get("distractors", [])
        if not dist:
            continue

        options = [truth] + dist[:3]
        prompt = (ctx + "\n\n" if ctx else "") + "Which is correct?\n"
        for i, opt in enumerate(options):
            prompt += f"{chr(65+i)}) {opt}\n"
        prompt += "Answer: "

        tokens = tokenizer.encode(prompt)
        correct_tok = tokenizer.encode("A")[-1]
        wrong_toks = [tokenizer.encode(chr(65+i))[-1] for i in range(1, len(options))]
        training_data.append((tokens, correct_tok, wrong_toks))

    if not training_data:
        return None

    def loss_fn(adapter_params, batch_tokens, correct_tok, wrong_toks):
        adapter.update(adapter_params)
        input_ids = mx.array([batch_tokens])
        logits = model(input_ids)
        last_logits = logits[0, -1:, :]  # (1, vocab)
        adapted = last_logits + adapter(last_logits)
        adapted = adapted[0]  # (vocab,)

        correct_logit = adapted[correct_tok]
        wrong_logits = mx.array([adapted[t] for t in wrong_toks])
        max_wrong = mx.max(wrong_logits)

        margin = 2.0
        loss = mx.maximum(mx.array(0.0), margin - (correct_logit - max_wrong))
        return loss

    loss_and_grad = mx.value_and_grad(loss_fn)
    params = dict(adapter.parameters())

    for step in range(steps):
        idx = step % len(training_data)
        tokens, ct, wt = training_data[idx]
        loss, grads = loss_and_grad(params, tokens, ct, wt)
        mx.eval(loss)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters())
        params = dict(adapter.parameters())

    return params


def evaluate_adapter(model, tokenizer, facts, adapter_params):
    """Evaluate adapter accuracy on facts."""
    import mlx.core as mx
    from noethersolve.adapter import SnapOnConfig, create_adapter

    config = SnapOnConfig(vocab_size=151936, d_inner=64, mode="logit")
    adapter = create_adapter(config)
    adapter.update(adapter_params)

    correct = 0
    total = 0

    for fact in facts:
        ctx = fact.get("context", "")
        truth = fact.get("truth", fact.get("fact", ""))
        dist = fact.get("distractors", [])
        if not dist:
            continue

        options = [truth] + dist[:3]
        prompt = (ctx + "\n\n" if ctx else "") + "Which is correct?\n"
        for i, opt in enumerate(options):
            prompt += f"{chr(65+i)}) {opt}\n"
        prompt += "Answer: "

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)

        last = logits[:, -1:, :]
        adapted = last + adapter(last)
        mx.eval(adapted)

        opt_toks = [tokenizer.encode(chr(65+i))[-1] for i in range(len(options))]
        opt_logits = [float(adapted[0, 0, t].item()) for t in opt_toks]

        if np.argmax(opt_logits) == 0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-domains", type=int, default=500)
    parser.add_argument("--min-facts", type=int, default=6)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    # Load steering results to find failures
    results_file = RESULTS_DIR / "steering_vectors_v2.json"
    if not results_file.exists():
        print("No steering results found. Run extract_vectors_fast.py first.")
        return

    with open(results_file) as f:
        steering_results = json.load(f)

    # Find domains where steering didn't help
    failures = [r for r in steering_results if r.get("improvement", 0) <= 0]
    print(f"Steering failures: {len(failures)} domains")
    print(f"Training adapters with {args.steps} steps each")

    ADAPTERS_DIR.mkdir(exist_ok=True)

    from mlx_lm import load
    print("\nLoading Qwen/Qwen3-4B-Base...")
    model, tokenizer = load("Qwen/Qwen3-4B-Base")

    adapter_results = []
    total_start = time.time()

    for i, sr in enumerate(failures[:args.max_domains]):
        domain = sr["domain"]
        baseline = sr.get("baseline", 0)

        # Skip if adapter already exists
        adapter_path = ADAPTERS_DIR / f"{domain}_adapter.npz"
        if adapter_path.exists():
            print(f"  [{i+1}/{len(failures)}] {domain:<45s} SKIP (adapter exists)")
            continue

        # Find facts
        ff = find_facts_file(domain)
        if ff is None:
            continue

        with open(ff) as f:
            data = json.load(f)
        facts = data.get("facts", data.get("verifications", []))
        valid = [f for f in facts if f.get("distractors")]

        if len(valid) < args.min_facts:
            continue

        # Split train/test
        np.random.seed(42)
        idx = np.random.permutation(len(valid))
        split = len(valid) // 2
        train = [valid[j] for j in idx[:split]]
        test = [valid[j] for j in idx[split:]]
        if len(test) < 2:
            test = train

        start = time.time()

        # Train
        params = train_adapter_on_facts(model, tokenizer, train, steps=args.steps)
        if params is None:
            continue

        # Evaluate
        acc, c, t = evaluate_adapter(model, tokenizer, test, params)
        elapsed = time.time() - start

        # Save adapter
        import mlx.core as mx
        flat = {k: np.array(v.astype(mx.float32)) for k, v in params.items() if hasattr(v, 'shape')}
        np.savez(str(adapter_path), **flat)
        adapter_bytes = adapter_path.stat().st_size

        improvement = acc - baseline
        status = "+" if improvement > 0 else "="
        print(f"  [{i+1}/{len(failures)}] {domain:<45s} {baseline:.0%}->{acc:.0%} (+{improvement:.0%}) {adapter_bytes/1024/1024:.1f}MB {status} [{elapsed:.0f}s]")

        adapter_results.append({
            "domain": domain,
            "baseline": baseline,
            "adapter_acc": acc,
            "improvement": improvement,
            "adapter_bytes": adapter_bytes,
            "elapsed": elapsed,
        })

        # Checkpoint every 10
        if len(adapter_results) % 10 == 0:
            with open(RESULTS_DIR / "adapter_training_results.json", "w") as f:
                json.dump(adapter_results, f, indent=2)
            total_elapsed = time.time() - total_start
            rate = len(adapter_results) / total_elapsed * 3600
            print(f"    CHECKPOINT: {len(adapter_results)} done, {rate:.0f}/hr")

    # Final save
    with open(RESULTS_DIR / "adapter_training_results.json", "w") as f:
        json.dump(adapter_results, f, indent=2)

    total_elapsed = time.time() - total_start
    improved = [r for r in adapter_results if r["improvement"] > 0]
    print(f"\n{'='*70}")
    print(f"DONE: {len(adapter_results)} adapters in {total_elapsed/3600:.1f}h")
    print(f"Improved: {len(improved)}/{len(adapter_results)}")
    total_mb = sum(r["adapter_bytes"] for r in adapter_results) / 1024 / 1024
    print(f"Total adapter storage: {total_mb:.0f} MB")


if __name__ == "__main__":
    main()
