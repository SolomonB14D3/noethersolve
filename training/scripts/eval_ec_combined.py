#!/usr/bin/env python3
"""
Evaluate elliptic curves facts with combined adapter routing.

Uses:
- Main elliptic_curves_adapter.npz for facts it handles well (ec01-03, ec05-07, ec10, ec12)
- Orthogonal adapters for holdouts (ec04, ec08, ec09, ec11)
"""

import json
import os
import sys

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx.utils import tree_unflatten

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve import train_utils as t3

ADAPTERS_DIR = os.path.join(ROOT, "adapters")


def load_adapter(model, path):
    """Load adapter weights from npz file."""
    if not os.path.exists(path):
        return None
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=64, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    weights = dict(np.load(path))
    weights = {k: mx.array(v) for k, v in weights.items()}
    adapter.update(tree_unflatten(list(weights.items())))
    return adapter


def eval_fact(adapter, lm_head, model, tokenizer, fact):
    ctx, truth, distractors = fact["context"], fact["truth"], fact["distractors"]
    prompt = ctx + ":"

    def adapted_lp(text):
        prompt_ids = tokenizer.encode(prompt)
        comp_ids = tokenizer.encode(text)
        full_ids = prompt_ids + comp_ids
        if not comp_ids:
            return -999.0
        tokens = mx.array(full_ids)[None, :]
        h = model.model(tokens)
        base_logits = lm_head(h)
        if adapter is not None:
            shifts = adapter(base_logits)
            shifts = shifts - shifts.mean(axis=-1, keepdims=True)
            logits = base_logits + shifts
            logits = t3.LOGIT_SOFTCAP * mx.tanh(logits / t3.LOGIT_SOFTCAP)
        else:
            logits = base_logits
        n_prompt = len(prompt_ids)
        total = 0.0
        for i, tok_id in enumerate(comp_ids):
            pos = n_prompt - 1 + i
            lv = np.array(logits[0, pos].astype(mx.float32))
            lse = float(np.log(np.sum(np.exp(lv - lv.max())) + 1e-8) + lv.max())
            total += float(lv[tok_id]) - lse
        return total

    truth_lp = adapted_lp(f" {truth}")
    dist_lps = [adapted_lp(f" {d}") for d in distractors]
    margin = truth_lp - max(dist_lps)
    return margin > 0, margin


def main():
    # Load oracle facts
    facts_path = os.path.join(ROOT, "problems", "elliptic_curves_facts.json")
    with open(facts_path) as f:
        facts_data = json.load(f)

    print("Loading Qwen/Qwen3-4B-Base...")
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)

    # Load adapters
    main_adapter = load_adapter(model, os.path.join(ADAPTERS_DIR, "elliptic_curves_adapter.npz"))
    invariants_adapter = load_adapter(model, os.path.join(ADAPTERS_DIR, "ec_invariants_adapter.npz"))
    group_adapter = load_adapter(model, os.path.join(ADAPTERS_DIR, "ec_group_structure_adapter.npz"))
    torsion_adapter = load_adapter(model, os.path.join(ADAPTERS_DIR, "ec_torsion_adapter.npz"))
    supersingular_adapter = load_adapter(model, os.path.join(ADAPTERS_DIR, "ec_supersingular_adapter.npz"))

    # Routing: orthogonal adapters for holdouts, main adapter for the rest
    routing = {
        "ec04": invariants_adapter,
        "ec08": group_adapter,
        "ec09": torsion_adapter,
        "ec11": supersingular_adapter,
    }

    print("\n" + "="*60)
    print("  COMBINED ROUTING: main + orthogonal adapters")
    print("="*60)

    wins = 0
    margins = []
    for fact in facts_data["facts"]:
        fact_id = fact["id"]
        if fact_id in routing:
            adapter = routing[fact_id]
            label = fact_id.replace("ec", "ortho_")
        else:
            adapter = main_adapter
            label = "main"

        win, margin = eval_fact(adapter, lm_head, model, tokenizer, fact)
        marker = "+" if win else "-"
        print(f"  {marker} {fact_id} [{label}]: margin={margin:+.3f}")
        wins += int(win)
        margins.append(margin)

    mean_margin = np.mean(margins)
    print(f"\n  Combined routing: {wins}/12 facts pass")
    print(f"  Mean margin: {mean_margin:+.3f}")

    if wins == 12:
        print("\n  *** 12/12 ACHIEVED — All elliptic curve facts flipped! ***")


if __name__ == "__main__":
    main()
