#!/usr/bin/env python3
"""
Score NS enstrophy bound target claims with:
1. Base model
2. Surround adapter (trained on surrounding facts only)
3. All existing NS/vortex/Hamiltonian adapters (for convergence)

This is the full discovery pipeline:
- Surround adapter proposes an answer
- Existing adapters vote
- Convergence = confidence signal
"""

import json
import os
import sys
import time

if not os.environ.get("HF_HOME") and os.path.isdir("/Volumes/4TB SD/ml_cache/huggingface"):
    os.environ["HF_HOME"] = "/Volumes/4TB SD/ml_cache/huggingface"

import mlx.core as mx
import mlx_lm
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from noethersolve.oracle import score_fact_mc


def load_model():
    print("Loading Qwen3-14B-Base...")
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-14B-Base")
    return model, tokenizer


def load_adapter(model, path, d_inner=64):
    from noethersolve.adapter import SnapOnConfig, create_adapter
    from noethersolve.train_utils import get_lm_head_fn
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                      n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    weights = mx.load(path)
    adapter.load_weights(list(weights.items()))
    mx.eval(adapter.parameters())
    lm_head = get_lm_head_fn(model)
    return adapter, lm_head


def score_candidates(model, tokenizer, target, adapter=None, lm_head=None):
    margins = []
    for i, cand in enumerate(target["candidates"]):
        others = [c for j, c in enumerate(target["candidates"]) if j != i]
        win, margin, _, _ = score_fact_mc(
            model, tokenizer, target["context"], cand, others,
            adapter=adapter, lm_head=lm_head,
        )
        margins.append(float(margin))
    return margins


def get_compatible_adapters(adapter_dir, prefixes, max_per_prefix=5):
    """Get all compatible 4B adapters matching any of the prefixes."""
    all_npz = sorted([f for f in os.listdir(adapter_dir) if f.endswith('.npz')])
    results = []
    seen = set()
    for prefix in prefixes:
        count = 0
        for f in all_npz:
            if prefix in f.lower() and f not in seen:
                try:
                    weights = mx.load(os.path.join(adapter_dir, f))
                    if not any('bias' in k for k in weights.keys()):
                        results.append(f)
                        seen.add(f)
                        count += 1
                except:
                    pass
            if count >= max_per_prefix:
                break
    return results


def main():
    model, tokenizer = load_model()
    adapter_dir = os.path.join(ROOT, "adapters", "qwen3_4b_base")

    # Load target claims
    with open(os.path.join(HERE, "ns_enstrophy_bound_target.json")) as f:
        targets = json.load(f)["targets"]

    # Surround adapter
    surround_path = os.path.join(adapter_dir, "surround_ns_bound_adapter.npz")

    # Existing relevant adapters
    prefixes = ["ns_regularity", "vortex", "continuous", "hamiltonian",
                "reduced_navier", "3body", "analysis_pde"]
    existing_adapters = get_compatible_adapters(adapter_dir, prefixes, max_per_prefix=4)
    print(f"Found {len(existing_adapters)} compatible existing adapters")

    all_results = {}

    for target in targets:
        print(f"\n{'='*70}")
        print(f"  {target['id']}")
        print(f"  {target['context'][:80]}...")
        if "notes" in target:
            print(f"  Notes: {target['notes']}")
        print(f"{'='*70}")

        # Base model
        base_margins = score_candidates(model, tokenizer, target)
        base_vote = int(np.argmax(base_margins))
        print(f"\n  BASE: candidate {base_vote} (margin={base_margins[base_vote]:+.2f})")
        for i, (m, c) in enumerate(zip(base_margins, target["candidates"])):
            pick = " <<<" if i == base_vote else ""
            print(f"    [{i}] {m:+8.2f}  {c[:75]}{pick}")

        # Surround adapter
        surr_vote = None
        if os.path.exists(surround_path):
            try:
                adpt, lm_head = load_adapter(model, surround_path)
                surr_margins = score_candidates(model, tokenizer, target,
                                               adapter=adpt, lm_head=lm_head)
                surr_vote = int(np.argmax(surr_margins))
                print(f"\n  SURROUND ADAPTER: candidate {surr_vote} (margin={surr_margins[surr_vote]:+.2f})")
                for i, (m, c) in enumerate(zip(surr_margins, target["candidates"])):
                    delta = m - base_margins[i]
                    pick = " <<<" if i == surr_vote else ""
                    print(f"    [{i}] {m:+8.2f} (Δ={delta:+.1f})  {c[:65]}{pick}")
            except Exception as e:
                print(f"\n  SURROUND ADAPTER: ERROR {e}")
        else:
            print(f"\n  SURROUND ADAPTER: not yet trained")

        # Existing adapters
        print(f"\n  EXISTING ADAPTERS ({len(existing_adapters)}):")
        adapter_votes = []
        for afile in existing_adapters:
            try:
                adpt, lm_head = load_adapter(model, os.path.join(adapter_dir, afile))
                margins = score_candidates(model, tokenizer, target,
                                          adapter=adpt, lm_head=lm_head)
                vote = int(np.argmax(margins))
                conf = sorted(margins, reverse=True)
                confidence = conf[0] - conf[1] if len(conf) > 1 else conf[0]
                adapter_votes.append({"file": afile, "vote": vote, "conf": confidence, "margins": margins})
                shift = "SHIFTED" if vote != base_vote else "agrees"
                print(f"    {afile[:50]:50s} → cand {vote} (conf={confidence:+.1f}) [{shift}]")
            except Exception as e:
                print(f"    {afile[:50]:50s} → ERROR")

        # Consensus
        if adapter_votes:
            weighted = {}
            for v in adapter_votes:
                w = max(0, v["conf"])
                weighted[v["vote"]] = weighted.get(v["vote"], 0) + w
            total_w = sum(weighted.values()) or 1
            consensus = max(weighted, key=weighted.get)
            consensus_pct = weighted[consensus] / total_w

            # Simple majority
            counts = {}
            for v in adapter_votes:
                counts[v["vote"]] = counts.get(v["vote"], 0) + 1
            majority = max(counts, key=counts.get)
            majority_pct = counts[majority] / len(adapter_votes)

            print(f"\n  CONSENSUS:")
            print(f"    Simple majority:     candidate {majority} ({majority_pct:.0%})")
            print(f"    Confidence-weighted: candidate {consensus} ({consensus_pct:.0%})")
            if surr_vote is not None:
                print(f"    Surround adapter:    candidate {surr_vote}")
                if surr_vote == consensus:
                    print(f"    *** SURROUND + EXISTING CONVERGE on candidate {consensus} ***")
                else:
                    print(f"    *** SURROUND ({surr_vote}) and EXISTING ({consensus}) DISAGREE ***")

        all_results[target["id"]] = {
            "context": target["context"],
            "candidates": target["candidates"],
            "base_vote": base_vote,
            "base_margins": base_margins,
            "surround_vote": surr_vote,
            "surround_margins": surr_margins if surr_vote is not None else None,
            "adapter_consensus": consensus if adapter_votes else None,
            "adapter_consensus_pct": consensus_pct if adapter_votes else None,
            "notes": target.get("notes", ""),
        }

    # Final summary
    print(f"\n\n{'='*70}")
    print("  NS ENSTROPHY BOUND — DISCOVERY SUMMARY")
    print(f"{'='*70}")
    for tid, r in all_results.items():
        print(f"\n  {tid}:")
        print(f"    Base: candidate {r['base_vote']}")
        print(f"    Surround: candidate {r['surround_vote']}")
        print(f"    Existing consensus: candidate {r['adapter_consensus']} ({r['adapter_consensus_pct']:.0%})" if r['adapter_consensus'] is not None else "")
        answer = r['candidates'][r['surround_vote']] if r['surround_vote'] is not None else "N/A"
        print(f"    Surround answer: {answer[:80]}")

    output_path = os.path.join(ROOT, "results", "ns_enstrophy_bound_discovery.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
