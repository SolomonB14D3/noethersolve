#!/usr/bin/env python3
"""Verify all frontier domain facts pass with their adapters."""

import json
import sys
from pathlib import Path
from collections import defaultdict

import mlx.core as mx
import mlx_lm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noethersolve.adapter import SnapOnConfig, SnapOnLogitMLP
from noethersolve.train_utils import get_lm_head_fn
from noethersolve.oracle import score_fact_mc

MODEL_ID = "Qwen/Qwen3-14B-Base"
PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "problems"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapters" / "qwen3_4b_base"

DOMAINS = [
    "battery_technology",
    "origin_of_life",
    "consciousness",
    "antibiotic_resistance",
    "protein_folding",
    "aging_biology",
    "quantum_gravity",
    "dark_matter_energy",
]


def load_adapter(domain, cluster, vocab_size):
    """Load a trained adapter."""
    path = ADAPTER_DIR / f"{domain}_{cluster}_adapter.npz"
    if not path.exists():
        return None

    config = SnapOnConfig(d_inner=64, vocab_size=vocab_size, mode="logit")
    adapter = SnapOnLogitMLP(config)

    params = mx.load(str(path))
    flat = {}
    for k, v in params.items():
        parts = k.split(".")
        if len(parts) == 2:
            if parts[0] not in flat:
                flat[parts[0]] = {}
            flat[parts[0]][parts[1]] = v
        else:
            flat[k] = v
    adapter.update(flat)
    return adapter


def main():
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = mlx_lm.load(MODEL_ID)
    model.eval()
    lm_head = get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]

    total_pass = 0
    total_facts = 0
    failures = []

    for domain in DOMAINS:
        path = PROBLEMS_DIR / f"{domain}_facts.json"
        with open(path) as f:
            data = json.load(f)

        # Group by cluster
        clusters = defaultdict(list)
        for fact in data["facts"]:
            clusters[fact["cluster"]].append(fact)

        print(f"\n{domain}:")

        for cluster, facts in sorted(clusters.items()):
            adapter = load_adapter(domain, cluster, vocab_size)
            if adapter is None:
                print(f"  {cluster}: MISSING ADAPTER")
                continue

            n_pass = 0
            for fact in facts:
                result = score_fact_mc(
                    model, tokenizer,
                    fact["context"], fact["truth"], fact["distractors"],
                    adapter=adapter,
                    lm_head=lm_head,
                )
                margin = result[1]
                if margin > 0:
                    n_pass += 1
                else:
                    failures.append((domain, cluster, fact["id"], margin))

            status = "✓" if n_pass == len(facts) else "✗"
            print(f"  {cluster}: {n_pass}/{len(facts)} {status}")
            total_pass += n_pass
            total_facts += len(facts)

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_pass}/{total_facts} ({100*total_pass/total_facts:.1f}%)")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for domain, cluster, fid, margin in failures:
            print(f"  {domain}/{cluster}/{fid}: margin={margin:.2f}")
    else:
        print("\nAll facts passing!")


if __name__ == "__main__":
    main()
