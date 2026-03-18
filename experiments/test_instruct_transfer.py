#!/usr/bin/env python3
"""Test whether Base-trained adapters transfer to Instruct model.

Loads Qwen3-4B (Instruct) and applies adapters trained on Qwen3-4B-Base.
Tests a representative sample across domains: physics, genetics, math,
chemistry, CS, LLM science.

Compares:
  1. Instruct baseline (no adapter) — does Instruct already know more?
  2. Instruct + Base-trained adapter — do the adapters still help?
  3. Reference: Base baseline and Base + adapter (from prior benchmarks)

Usage:
    python experiments/test_instruct_transfer.py
    python experiments/test_instruct_transfer.py --n-per-domain 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noethersolve.adapter import SnapOnConfig, SnapOnLogitMLP
from noethersolve.train_utils import get_lm_head_fn
from noethersolve.oracle import score_fact_mc

ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapters"
PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "problems"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "instruct_transfer"

# Representative domains with their fact files and adapter names
DOMAINS = {
    "hamiltonian": {
        "facts": "hamiltonian_facts.json",
        "adapters": ["hamiltonian_stage5.npz"],
    },
    "ns_regularity": {
        "facts": "ns_regularity_facts.json",
        "adapters": ["ns_regularity_prior_broken.npz"],
    },
    "chemical_kinetics": {
        "facts": "chemical_conservation_facts.json",
        "adapters": ["chemical_adapter.npz"],
    },
    "genetics": {
        "facts": "genetics_therapeutics_facts.json",
        "adapters": ["genetics_crispr_adapter.npz"],
    },
    "millennium": {
        "facts": "millennium_problems_facts.json",
        "adapters": ["millennium_problems_riemann_adapter.npz"],
    },
    "llm_hallucination": {
        "facts": "llm_hallucination_facts.json",
        "adapters": ["llm_hallucination_hallucination_adapter.npz"],
    },
    "chemistry": {
        "facts": "chemistry_facts.json",
        "adapters": ["chem_enzyme_focused.npz"],
    },
}


def load_adapter(path, vocab_size, d_inner=64):
    """Load a saved adapter from .npz file."""
    config = SnapOnConfig(d_inner=d_inner, vocab_size=vocab_size, mode="logit")
    adapter = SnapOnLogitMLP(config)
    mx.eval(adapter.parameters())

    data = dict(np.load(str(path)))
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
    items = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten(v, key))
        else:
            items.append((key, v))
    return items


def run_domain(model, tokenizer, lm_head, facts, adapter=None):
    """Score all facts, return list of result dicts."""
    results = []
    for fact in facts:
        try:
            win, margin, truth_lp, best_dist_lp = score_fact_mc(
                model, tokenizer,
                fact["context"], fact["truth"], fact["distractors"],
                adapter=adapter, lm_head=lm_head,
            )
            results.append({
                "id": fact.get("id", "?"),
                "win": win,
                "margin": margin,
            })
        except Exception as e:
            results.append({
                "id": fact.get("id", "?"),
                "win": False,
                "margin": -999,
                "error": str(e),
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-domain", type=int, default=None,
                        help="Limit facts per domain (None = all)")
    args = parser.parse_args()

    # ── Load both models ─────────────────────────────────────────────
    print("Loading Qwen3-4B-Base...")
    t0 = time.time()
    base_model, base_tok = mlx_lm.load("Qwen/Qwen3-4B-Base")
    base_model.eval()
    base_lm_head = get_lm_head_fn(base_model)
    base_vocab = base_model.model.embed_tokens.weight.shape[0]
    print(f"  Base loaded in {time.time()-t0:.1f}s, vocab={base_vocab}")

    print("Loading Qwen3-4B (Instruct)...")
    t0 = time.time()
    inst_model, inst_tok = mlx_lm.load("Qwen/Qwen3-4B")
    inst_model.eval()
    inst_lm_head = get_lm_head_fn(inst_model)
    inst_vocab = inst_model.model.embed_tokens.weight.shape[0]
    print(f"  Instruct loaded in {time.time()-t0:.1f}s, vocab={inst_vocab}")

    if base_vocab != inst_vocab:
        print(f"\n  WARNING: vocab mismatch! Base={base_vocab}, Instruct={inst_vocab}")
        print("  Adapters may not transfer without padding.")

    # ── Run tests ────────────────────────────────────────────────────
    all_results = {}

    for domain, cfg in DOMAINS.items():
        facts_path = PROBLEMS_DIR / cfg["facts"]
        if not facts_path.exists():
            print(f"\n  SKIP {domain}: {cfg['facts']} not found")
            continue

        with open(facts_path) as f:
            data = json.load(f)
        facts = data.get("facts", data)  # some files have {"facts": [...]}, some are just [...]
        if isinstance(facts, dict):
            facts = list(facts.values()) if not isinstance(facts, list) else facts

        if args.n_per_domain and len(facts) > args.n_per_domain:
            facts = facts[:args.n_per_domain]

        # Find adapter
        adapter_path = None
        for aname in cfg["adapters"]:
            p = ADAPTER_DIR / aname
            if p.exists():
                adapter_path = p
                break

        if adapter_path is None:
            print(f"\n  SKIP {domain}: no adapter found ({cfg['adapters']})")
            continue

        print(f"\n{'='*60}")
        print(f"  DOMAIN: {domain} ({len(facts)} facts)")
        print(f"  Adapter: {adapter_path.name}")
        print(f"{'='*60}")

        # Load adapter once (same weights for both models)
        try:
            adapter = load_adapter(adapter_path, base_vocab)
        except (ValueError, Exception) as e:
            print(f"  SKIP: adapter load failed: {e}")
            continue

        # 1. Base baseline
        print("  Base baseline...", end=" ", flush=True)
        base_bl = run_domain(base_model, base_tok, base_lm_head, facts)
        base_bl_pass = sum(1 for r in base_bl if r["win"])
        print(f"{base_bl_pass}/{len(facts)}")

        # 2. Base + adapter
        print("  Base + adapter...", end=" ", flush=True)
        base_ad = run_domain(base_model, base_tok, base_lm_head, facts, adapter=adapter)
        base_ad_pass = sum(1 for r in base_ad if r["win"])
        print(f"{base_ad_pass}/{len(facts)}")

        # 3. Instruct baseline
        print("  Instruct baseline...", end=" ", flush=True)
        inst_bl = run_domain(inst_model, inst_tok, inst_lm_head, facts)
        inst_bl_pass = sum(1 for r in inst_bl if r["win"])
        print(f"{inst_bl_pass}/{len(facts)}")

        # 4. Instruct + Base-trained adapter
        print("  Instruct + adapter...", end=" ", flush=True)
        inst_ad = run_domain(inst_model, inst_tok, inst_lm_head, facts, adapter=adapter)
        inst_ad_pass = sum(1 for r in inst_ad if r["win"])
        print(f"{inst_ad_pass}/{len(facts)}")

        all_results[domain] = {
            "n_facts": len(facts),
            "adapter": adapter_path.name,
            "base_baseline": base_bl_pass,
            "base_adapted": base_ad_pass,
            "inst_baseline": inst_bl_pass,
            "inst_adapted": inst_ad_pass,
            "base_bl_margins": [r["margin"] for r in base_bl if r["margin"] > -900],
            "inst_bl_margins": [r["margin"] for r in inst_bl if r["margin"] > -900],
            "inst_ad_margins": [r["margin"] for r in inst_ad if r["margin"] > -900],
            "per_fact": [
                {
                    "id": base_bl[i]["id"],
                    "base_bl": base_bl[i]["win"],
                    "base_ad": base_ad[i]["win"],
                    "inst_bl": inst_bl[i]["win"],
                    "inst_ad": inst_ad[i]["win"],
                    "inst_bl_margin": inst_bl[i]["margin"],
                    "inst_ad_margin": inst_ad[i]["margin"],
                }
                for i in range(len(facts))
            ],
        }

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TRANSFER SUMMARY: Base-trained adapters → Instruct")
    print(f"{'='*70}")
    print(f"  {'Domain':<22} {'Base BL':>8} {'Base+Ad':>8} {'Inst BL':>8} {'Inst+Ad':>8} {'Transfer':>10}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    total = {"base_bl": 0, "base_ad": 0, "inst_bl": 0, "inst_ad": 0, "n": 0}
    for domain, r in sorted(all_results.items()):
        n = r["n_facts"]
        # Transfer = does the adapter help Instruct as much as it helps Base?
        r["base_adapted"] - r["base_baseline"]
        inst_lift = r["inst_adapted"] - r["inst_baseline"]
        transfer = "YES" if inst_lift > 0 else ("NEUTRAL" if inst_lift == 0 else "HURT")

        print(f"  {domain:<22} {r['base_baseline']:>5}/{n:<2} {r['base_adapted']:>5}/{n:<2} "
              f"{r['inst_baseline']:>5}/{n:<2} {r['inst_adapted']:>5}/{n:<2} {transfer:>10}")

        total["base_bl"] += r["base_baseline"]
        total["base_ad"] += r["base_adapted"]
        total["inst_bl"] += r["inst_baseline"]
        total["inst_ad"] += r["inst_adapted"]
        total["n"] += n

    n = total["n"]
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':<22} {total['base_bl']:>5}/{n:<2} {total['base_ad']:>5}/{n:<2} "
          f"{total['inst_bl']:>5}/{n:<2} {total['inst_ad']:>5}/{n:<2}")

    # ── Save ─────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"transfer_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({
            "base_model": "Qwen/Qwen3-4B-Base",
            "inst_model": "Qwen/Qwen3-4B",
            "base_vocab": base_vocab,
            "inst_vocab": inst_vocab,
            "domains": {k: {kk: vv for kk, vv in v.items() if kk != "per_fact"}
                        for k, v in all_results.items()},
            "per_fact": {k: v["per_fact"] for k, v in all_results.items()},
            "totals": total,
        }, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
