#!/usr/bin/env python3
"""
Test adapter stacking across domains.

Hypothesis: Domain-specific adapters may transfer knowledge,
especially for related physics concepts.

Stack combinations to test:
1. k_adapter + qf_continuous (both vortex-related)
2. k_adapter + ns_adapter (both about vortex/3D)
3. qf_continuous + ns_adapter (2D vs 3D fluids)
4. All three stacked
"""

import os
import sys
import json
import numpy as np
import mlx.core as mx
import mlx_lm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve import train_utils as t3

ADAPTERS_DIR = os.path.join(ROOT, "adapters")
PROBLEMS_DIR = os.path.join(ROOT, "problems")


def load_adapter(name, cfg):
    path = os.path.join(ADAPTERS_DIR, f"{name}.npz")
    if not os.path.exists(path):
        print(f"  WARNING: {name}.npz not found")
        return None
    adapter = create_adapter(cfg)
    weights = dict(mx.load(path))
    adapter.load_weights(list(weights.items()))
    return adapter


def load_facts(name):
    path = os.path.join(PROBLEMS_DIR, f"{name}_facts.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("facts", [])


def compute_margin(model, tokenizer, lm_head, adapters, prompt, truth, distractors):
    """Compute margin with adapter stack."""
    def lp(text):
        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(prompt + text)
        n_prompt = len(prompt_ids)
        if len(full_ids) <= n_prompt:
            return -999.0
        tokens = mx.array(full_ids)[None, :]
        h = model.model(tokens)
        base_logits = lm_head(h)

        if adapters:
            logits = t3.apply_adapter_stack(adapters, base_logits)
        else:
            logits = base_logits

        total = 0.0
        for i, tok_id in enumerate(full_ids[n_prompt:]):
            pos = n_prompt - 1 + i
            lv = np.array(logits[0, pos].astype(mx.float32))
            lse = float(np.log(np.sum(np.exp(lv - lv.max())) + 1e-8) + lv.max())
            total += float(lv[tok_id]) - lse
        return total

    truth_lp = lp(f" {truth}")
    dist_lps = [lp(f" {d}") for d in distractors]
    return truth_lp - max(dist_lps)


def evaluate_facts(model, tokenizer, lm_head, adapters, facts, label):
    """Evaluate all facts and return pass count, mean margin."""
    wins, margins = 0, []
    for fact in facts:
        ctx = fact["context"] + ":"
        margin = compute_margin(model, tokenizer, lm_head, adapters,
                                ctx, fact["truth"], fact["distractors"])
        wins += int(margin > 0)
        margins.append(margin)
    mean_m = np.mean(margins) if margins else 0
    return wins, len(facts), mean_m


def main():
    print("="*70)
    print("Adapter Stacking Test")
    print("="*70)
    print()

    print("Loading model...")
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-14B-Base")
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)

    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=64, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)

    # Load adapters
    print("\nLoading adapters...")
    adapters = {}
    for name in ["k_adapter", "qf_continuous_adapter", "ns_adapter"]:
        adapters[name] = load_adapter(name, cfg)
        if adapters[name]:
            print(f"  Loaded: {name}")

    # Load fact sets
    print("\nLoading fact sets...")
    fact_sets = {
        "kinetic_k": load_facts("kinetic_k"),
        "qf_continuous": load_facts("qf_continuous"),
        "ns_regularity": load_facts("ns_regularity"),
    }
    for name, facts in fact_sets.items():
        print(f"  {name}: {len(facts)} facts")

    # Define stack configurations
    stacks = [
        ("Baseline (none)", []),
        ("k_adapter only", [adapters["k_adapter"]]),
        ("qf_continuous only", [adapters["qf_continuous_adapter"]]),
        ("ns_adapter only", [adapters["ns_adapter"]]),
        ("k + qf", [adapters["k_adapter"], adapters["qf_continuous_adapter"]]),
        ("k + ns", [adapters["k_adapter"], adapters["ns_adapter"]]),
        ("qf + ns", [adapters["qf_continuous_adapter"], adapters["ns_adapter"]]),
        ("All three", [adapters["k_adapter"], adapters["qf_continuous_adapter"], adapters["ns_adapter"]]),
    ]

    # Filter out any stacks with missing adapters
    stacks = [(name, [a for a in adpts if a is not None]) for name, adpts in stacks]

    print("\n" + "="*70)
    print("Stacking Results")
    print("="*70)

    results = {}
    for stack_name, adapter_list in stacks:
        print(f"\n{stack_name}:")
        print("-"*50)

        results[stack_name] = {}
        for fact_name, facts in fact_sets.items():
            if not facts:
                continue
            wins, total, mean_margin = evaluate_facts(
                model, tokenizer, lm_head, adapter_list, facts, stack_name
            )
            results[stack_name][fact_name] = {
                "pass": wins,
                "total": total,
                "mean_margin": mean_margin
            }
            status = "PASS" if wins > total//2 else "fail"
            print(f"  {fact_name:20s}: {wins}/{total} ({mean_margin:+.1f}) [{status}]")

    # Analysis
    print("\n" + "="*70)
    print("Cross-Domain Transfer Analysis")
    print("="*70)

    # Check if stacking helps on non-native domains
    print("\nDoes K adapter help NS facts?")
    ns_baseline = results["Baseline (none)"]["ns_regularity"]["pass"]
    ns_with_k = results["k_adapter only"]["ns_regularity"]["pass"]
    ns_with_both = results["k + ns"]["ns_regularity"]["pass"]
    print(f"  Baseline:  {ns_baseline}/16")
    print(f"  + k_adapter: {ns_with_k}/16")
    print(f"  k + ns combined: {ns_with_both}/16")

    print("\nDoes NS adapter help K facts?")
    k_baseline = results["Baseline (none)"]["kinetic_k"]["pass"]
    k_with_ns = results["ns_adapter only"]["kinetic_k"]["pass"]
    k_with_both = results["k + ns"]["kinetic_k"]["pass"]
    print(f"  Baseline:  {k_baseline}/8")
    print(f"  + ns_adapter: {k_with_ns}/8")
    print(f"  k + ns combined: {k_with_both}/8")

    print("\nDoes all-stack help compared to single?")
    for fact_name in fact_sets:
        if not fact_sets[fact_name]:
            continue
        baseline = results["Baseline (none)"][fact_name]["pass"]
        best_single = max(
            results["k_adapter only"][fact_name]["pass"],
            results["qf_continuous only"][fact_name]["pass"],
            results["ns_adapter only"][fact_name]["pass"]
        )
        all_three = results["All three"][fact_name]["pass"]
        print(f"  {fact_name}: baseline={baseline}, best_single={best_single}, all_three={all_three}")

    return results


if __name__ == "__main__":
    results = main()
