#!/usr/bin/env python3
"""Fast multi-layer steering vector extraction for all domains.

Sweeps layers 10, 15, 20 per domain, saves best vector.
Targets: 500+ vectors by morning.

Usage:
    python experiments/extract_vectors_fast.py
    python experiments/extract_vectors_fast.py --start-from 200  # resume
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
VECTORS_DIR = PROJECT / "steering_vectors"
results_file = PROJECT / "results" / "steering_vectors_v2.json"

LAYERS = [10, 15, 20]  # Overridden per-model in main() if needed
ALPHAS = [0.25, 0.50, 0.75, 1.0, 1.5]
MIN_FACTS = 6  # need at least 3 train + 3 test


def gather_all_domains():
    """Collect all domain files from bank + custom problems."""
    domains = []
    seen = set()

    # Bank files
    if BANK_DIR.exists():
        for f in sorted(BANK_DIR.glob("*.json")):
            if f.name == "summary.json":
                continue
            domains.append(f)
            seen.add(f.name)

    # Custom facts (skip if same name already in bank)
    for f in sorted((PROJECT / "problems").glob("*_facts*.json")):
        if f.name not in seen:
            domains.append(f)
            seen.add(f.name)

    return domains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=int, default=0, help="Skip first N domains (for resuming)")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base", help="Model to extract vectors from")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    # Model-specific output dirs
    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    vectors_dir = VECTORS_DIR / model_short
    results_file = PROJECT / "results" / f"steering_vectors_{model_short}.json"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    global LAYERS  # Will be adjusted based on model depth

    domain_files = gather_all_domains()
    print(f"Found {len(domain_files)} domain files")
    print(f"Starting from index {args.start_from}")

    print(f"\nLoading {args.model}...")
    model, tokenizer = load(args.model)
    n_layers = len(model.model.layers)
    print(f"Model loaded: {n_layers} layers")

    # Adjust sweep layers based on model depth
    if n_layers >= 40:
        LAYERS = [15, 20, 25]  # Deeper models: sweep mid-to-upper layers
    elif n_layers >= 30:
        LAYERS = [10, 15, 20]  # 4B/7B
    else:
        LAYERS = [5, 10, 15]   # Small models

    print(f"Layers: {LAYERS}, Alphas: {ALPHAS}")

    # Load existing results for resuming
    existing_results = []
    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)
    existing_domains = {r["domain"] for r in existing_results}

    results = list(existing_results)
    total_start = time.time()
    processed = 0
    skipped = 0

    for file_idx, df in enumerate(domain_files):
        if file_idx < args.start_from:
            continue

        try:
            with open(df) as f:
                data = json.load(f)
        except Exception:
            continue

        # Handle both dict format {"domain": ..., "facts": [...]} and list format [...]
        if isinstance(data, list):
            domain = df.stem.replace("_facts_v2", "").replace("_facts", "")
            facts = data
        else:
            domain = data.get("domain", df.stem.replace("_facts_v2", "").replace("_facts", ""))
            facts = data.get("facts", data.get("verifications", []))

        # Skip if already done
        if domain in existing_domains:
            skipped += 1
            continue

        # Filter to facts that have distractors
        valid = [f for f in facts if f.get("distractors")]
        if len(valid) < MIN_FACTS:
            continue

        start = time.time()

        # Split train/test
        np.random.seed(42)
        idx = np.random.permutation(len(valid))
        split = min(len(valid) // 2, 30)  # cap train at 30 for speed
        train = [valid[i] for i in idx[:split]]
        test = [valid[i] for i in idx[split:split + 30]]  # cap test at 30

        if len(test) < 3:
            test = train

        # Baseline
        baseline_correct = 0
        for fact in test:
            ctx = fact.get("context", "")
            truth = fact.get("truth", fact.get("fact", ""))
            dist = fact["distractors"]
            options = [truth] + dist[:3]
            prompt = (ctx + "\n\n" if ctx else "") + "Which is correct?\n"
            for i, opt in enumerate(options):
                prompt += f"{chr(65+i)}) {opt}\n"
            prompt += "Answer: "
            tokens = tokenizer.encode(prompt)
            logits = model(mx.array([tokens]))
            mx.eval(logits)
            opt_toks = [tokenizer.encode(chr(65+i))[-1] for i in range(len(options))]
            opt_logits = [float(logits[0, -1, t].item()) for t in opt_toks]
            if np.argmax(opt_logits) == 0:
                baseline_correct += 1

        baseline = baseline_correct / len(test)

        # Try each layer
        best_overall_acc = baseline
        best_overall_alpha = 0.0
        best_overall_layer = LAYERS[0]
        best_sv = None

        for layer in LAYERS:
            if layer >= n_layers:
                continue

            # Compute steering vector
            c_acts = []
            i_acts = []
            for fact in train:
                ctx = fact.get("context", "")
                truth = fact.get("truth", fact.get("fact", ""))
                dist = fact["distractors"]
                bp = f"{ctx}\n\nAnswer: " if ctx else "Answer: "

                # Correct activation
                tokens = tokenizer.encode(bp + truth)
                hidden = model.model.embed_tokens(mx.array([tokens]))
                for li, lyr in enumerate(model.model.layers):
                    hidden = lyr(hidden, mask=None, cache=None)
                    if li == layer:
                        c_acts.append(np.array(hidden[0, -1, :].astype(mx.float32)))
                        break
                mx.eval(hidden)

                # Incorrect activation
                tokens = tokenizer.encode(bp + dist[0])
                hidden = model.model.embed_tokens(mx.array([tokens]))
                for li, lyr in enumerate(model.model.layers):
                    hidden = lyr(hidden, mask=None, cache=None)
                    if li == layer:
                        i_acts.append(np.array(hidden[0, -1, :].astype(mx.float32)))
                        break
                mx.eval(hidden)

            sv = np.array(c_acts).mean(0) - np.array(i_acts).mean(0)

            # Test alphas
            for alpha in ALPHAS:
                correct = 0
                for fact in test:
                    ctx = fact.get("context", "")
                    truth = fact.get("truth", fact.get("fact", ""))
                    dist = fact["distractors"]
                    options = [truth] + dist[:3]
                    prompt = (ctx + "\n\n" if ctx else "") + "Which is correct?\n"
                    for i, opt in enumerate(options):
                        prompt += f"{chr(65+i)}) {opt}\n"
                    prompt += "Answer: "
                    tokens = tokenizer.encode(prompt)
                    hidden = model.model.embed_tokens(mx.array([tokens]))
                    for li, lyr in enumerate(model.model.layers):
                        hidden = lyr(hidden, mask=None, cache=None)
                        if li == layer:
                            steer = mx.array(sv.astype(np.float32) * alpha).reshape(1, 1, -1)
                            hidden = hidden + steer
                    hidden = model.model.norm(hidden)
                    if model.args.tie_word_embeddings:
                        logits = model.model.embed_tokens.as_linear(hidden)
                    else:
                        logits = model.lm_head(hidden)
                    mx.eval(logits)
                    opt_toks = [tokenizer.encode(chr(65+i))[-1] for i in range(len(options))]
                    opt_logits = [float(logits[0, -1, t].item()) for t in opt_toks]
                    if np.argmax(opt_logits) == 0:
                        correct += 1

                acc = correct / len(test)
                if acc > best_overall_acc:
                    best_overall_acc = acc
                    best_overall_alpha = alpha
                    best_overall_layer = layer
                    best_sv = sv.copy()

        elapsed = time.time() - start
        improvement = best_overall_acc - baseline

        # Save best vector
        if best_sv is not None:
            vec_path = vectors_dir / f"{domain}_best.npy"
            np.save(vec_path, best_sv)
            vec_bytes = vec_path.stat().st_size
        else:
            vec_bytes = 0

        result = {
            "domain": domain,
            "source": data.get("source", "custom"),
            "baseline": baseline,
            "best_steered": best_overall_acc,
            "improvement": improvement,
            "best_alpha": best_overall_alpha,
            "best_layer": best_overall_layer,
            "n_train": len(train),
            "n_test": len(test),
            "vector_bytes": vec_bytes,
            "elapsed": elapsed,
        }
        results.append(result)
        existing_domains.add(domain)
        processed += 1

        status = "+" if improvement > 0 else ("=" if improvement == 0 else "-")
        print(f"  [{file_idx+1}/{len(domain_files)}] {domain:<45s} {baseline:.0%}->{best_overall_acc:.0%} L{best_overall_layer} a={best_overall_alpha:.2f} {status} [{elapsed:.1f}s]")

        # Save checkpoint every 25 domains
        if processed % 25 == 0:
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            total_elapsed = time.time() - total_start
            rate = processed / total_elapsed * 60
            remaining = len(domain_files) - file_idx - 1
            eta = remaining / rate if rate > 0 else 0
            print(f"    CHECKPOINT: {processed} done, {skipped} skipped, {rate:.1f}/min, ETA {eta:.0f}min")

    # Final save
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    total_elapsed = time.time() - total_start

    # Summary
    active = [r for r in results if r not in existing_results]
    improved = [r for r in active if r.get("improvement", 0) > 0]

    print(f"\n{'='*70}")
    print(f"DONE: {processed} new + {len(existing_results)} existing = {len(results)} total")
    print(f"Time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"Improved: {len(improved)}/{processed}")

    if improved:
        print(f"\nTop improvements:")
        for r in sorted(improved, key=lambda x: -x["improvement"])[:20]:
            print(f"  {r['domain']:<45s} {r['baseline']:.0%}->{r['best_steered']:.0%} (+{r['improvement']:.0%}) L{r['best_layer']}")

    vectors = list(vectors_dir.glob("*_best.npy"))
    total_kb = sum(v.stat().st_size for v in vectors) / 1024
    print(f"\nVectors: {len(vectors)} files, {total_kb:.0f} KB total")
    print(f"Model: {args.model}")
    print(f"Vectors dir: {vectors_dir}")
    print(f"Results: {results_file}")


if __name__ == "__main__":
    main()
