#!/usr/bin/env python3
"""Benchmark the toolkit_unified_adapter.npz against all 244 toolkit facts.

Reports:
  - Baseline (no adapter) accuracy and mean margin
  - Adapted accuracy and mean margin
  - Per-cluster breakdown
  - Stubborn facts (negative margin even with adapter)

Usage:
    python experiments/benchmark_toolkit_adapter.py
    python experiments/benchmark_toolkit_adapter.py --adapter adapters/toolkit_unified_adapter.npz
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx_lm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noethersolve.adapter import SnapOnConfig, SnapOnLogitMLP
from noethersolve.train_utils import apply_adapter, get_lm_head_fn
from noethersolve.oracle import score_fact_mc

MODEL_ID = "Qwen/Qwen3-4B-Base"
FACTS_PATH = Path(__file__).resolve().parent.parent / "problems" / "toolkit_facts.json"
ADAPTER_PATH = Path(__file__).resolve().parent.parent / "adapters" / "toolkit_unified_adapter.npz"


def load_adapter(path, vocab_size, d_inner=64):
    """Load a saved adapter from .npz file."""
    config = SnapOnConfig(d_inner=d_inner, vocab_size=vocab_size, mode="logit")
    adapter = SnapOnLogitMLP(config)
    mx.eval(adapter.parameters())

    data = dict(np.load(str(path)))
    # Reconstruct nested dict for MLX module
    params = {}
    for k, v in data.items():
        parts = k.split(".")
        d = params
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = mx.array(v)

    adapter.load_weights(list(_flatten(params)))
    mx.eval(adapter.parameters())
    return adapter


def _flatten(d, prefix=""):
    """Flatten nested dict to list of (key, value) tuples."""
    items = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten(v, key))
        else:
            items.append((key, v))
    return items


def run_benchmark(model, tokenizer, lm_head, facts, adapter=None):
    """Score all facts, return list of (fact_id, cluster, win, margin)."""
    results = []
    for i, fact in enumerate(facts):
        try:
            win, margin, truth_lp, best_dist_lp = score_fact_mc(
                model, tokenizer,
                fact["context"], fact["truth"], fact["distractors"],
                adapter=adapter, lm_head=lm_head,
            )
            results.append({
                "idx": i,
                "id": fact.get("id", f"fact_{i}"),
                "cluster": fact.get("cluster", "unknown"),
                "win": win,
                "margin": margin,
                "truth_lp": truth_lp,
                "best_dist_lp": best_dist_lp,
            })
        except Exception as e:
            results.append({
                "idx": i,
                "id": fact.get("id", f"fact_{i}"),
                "cluster": fact.get("cluster", "unknown"),
                "win": False,
                "margin": -999,
                "truth_lp": -999,
                "best_dist_lp": 0,
            })
        if (i + 1) % 50 == 0:
            n_pass = sum(1 for r in results if r["win"])
            print(f"  [{i+1}/{len(facts)}] {n_pass}/{i+1} passing so far...")
    return results


def print_report(label, results):
    """Print a formatted report of benchmark results."""
    n_total = len(results)
    n_pass = sum(1 for r in results if r["win"])
    valid_margins = [r["margin"] for r in results if r["margin"] > -900]
    mean_margin = np.mean(valid_margins) if valid_margins else float("nan")

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Overall: {n_pass}/{n_total} ({100*n_pass/n_total:.1f}%)")
    print(f"  Mean margin: {mean_margin:.2f}")

    # Per-cluster breakdown
    clusters = defaultdict(list)
    for r in results:
        clusters[r["cluster"]].append(r)

    print(f"\n  {'Cluster':<30} {'Pass':>6} {'Total':>6} {'%':>7} {'Mean Margin':>12}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*7} {'-'*12}")
    for cluster in sorted(clusters.keys()):
        cr = clusters[cluster]
        cp = sum(1 for r in cr if r["win"])
        cm = [r["margin"] for r in cr if r["margin"] > -900]
        cmean = np.mean(cm) if cm else float("nan")
        print(f"  {cluster:<30} {cp:>6} {len(cr):>6} {100*cp/len(cr):>6.1f}% {cmean:>12.2f}")

    # Stubborn facts (worst margins)
    failed = [r for r in results if not r["win"] and r["margin"] > -900]
    if failed:
        failed.sort(key=lambda r: r["margin"])
        print(f"\n  Stubborn facts (worst {min(15, len(failed))}):")
        for r in failed[:15]:
            print(f"    {r['id']:<40} cluster={r['cluster']:<20} margin={r['margin']:>8.2f}")

    return n_pass, n_total, mean_margin


def main():
    parser = argparse.ArgumentParser(description="Benchmark Toolkit Adapter")
    parser.add_argument("--adapter", type=str, default=str(ADAPTER_PATH))
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    # Load facts
    with open(FACTS_PATH) as f:
        data = json.load(f)
    facts = data["facts"]
    print(f"Loaded {len(facts)} toolkit facts from {FACTS_PATH}")

    # Load model
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = mlx_lm.load(MODEL_ID)
    model.eval()
    lm_head = get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]
    print(f"  vocab_size={vocab_size}")

    # Baseline
    if not args.skip_baseline:
        print("\n--- Baseline (no adapter) ---")
        t0 = time.time()
        baseline_results = run_benchmark(model, tokenizer, lm_head, facts)
        base_pass, base_total, base_mean = print_report("BASELINE (no adapter)", baseline_results)
        print(f"  Time: {time.time()-t0:.0f}s")
    else:
        base_pass, base_total = 0, len(facts)

    # With adapter
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"\nAdapter not found: {adapter_path}")
        return

    print(f"\n--- Loading adapter: {adapter_path.name} ---")
    adapter = load_adapter(adapter_path, vocab_size, d_inner=args.d_inner)
    print(f"  Adapter params: {sum(v.size for _, v in _flatten(adapter.parameters())):.1f}")

    print("\n--- Adapted evaluation ---")
    t0 = time.time()
    adapted_results = run_benchmark(model, tokenizer, lm_head, facts, adapter=adapter)
    adapt_pass, adapt_total, adapt_mean = print_report(
        f"ADAPTED ({adapter_path.name})", adapted_results
    )
    print(f"  Time: {time.time()-t0:.0f}s")

    # Delta summary
    if not args.skip_baseline:
        print(f"\n{'='*60}")
        print(f"  DELTA SUMMARY")
        print(f"{'='*60}")
        print(f"  Baseline: {base_pass}/{base_total} ({100*base_pass/base_total:.1f}%)")
        print(f"  Adapted:  {adapt_pass}/{adapt_total} ({100*adapt_pass/adapt_total:.1f}%)")
        print(f"  Lift:     +{adapt_pass-base_pass} facts ({100*(adapt_pass-base_pass)/base_total:.1f}%)")

        # Per-fact comparison
        if len(baseline_results) == len(adapted_results):
            flipped_pos = sum(1 for b, a in zip(baseline_results, adapted_results)
                              if not b["win"] and a["win"])
            flipped_neg = sum(1 for b, a in zip(baseline_results, adapted_results)
                              if b["win"] and not a["win"])
            print(f"  Flipped → correct: {flipped_pos}")
            print(f"  Regressed:         {flipped_neg}")

    # Save results
    out_dir = Path(__file__).resolve().parent.parent / "results" / "adapter_benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"toolkit_unified_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({
            "adapter": str(adapter_path),
            "model": MODEL_ID,
            "n_facts": len(facts),
            "baseline_pass": base_pass if not args.skip_baseline else None,
            "adapted_pass": adapt_pass,
            "adapted_results": adapted_results,
        }, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
