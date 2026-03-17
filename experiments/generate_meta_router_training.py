#!/usr/bin/env python3
"""Generate detailed per-fact training data for meta-router.

Runs oracle evaluation on all fact × adapter combinations to build
comprehensive training data for the meta-router.

Usage:
    python experiments/generate_meta_router_training.py --domains hamiltonian chemical_conservation
    python experiments/generate_meta_router_training.py --top-domains 10
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mlx.core as mx
    import mlx_lm
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("Warning: MLX not available")

from noethersolve.oracle import score_fact_mc
from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve.train_utils import get_lm_head_fn


def load_facts(facts_file: Path) -> List[dict]:
    """Load facts with context, truth, distractors."""
    with open(facts_file) as f:
        data = json.load(f)

    facts = data.get("facts", data.get("verifications", []))
    normalized = []

    for i, fact in enumerate(facts):
        if isinstance(fact, dict):
            normalized.append({
                "id": fact.get("id", f"fact_{i:02d}"),
                "context": fact.get("context", ""),
                "truth": fact.get("truth", fact.get("fact", "")),
                "distractors": fact.get("distractors", []),
            })
        else:
            normalized.append({
                "id": f"fact_{i:02d}",
                "context": "",
                "truth": str(fact),
                "distractors": [],
            })

    return normalized


def find_domain_adapters(adapters_dir: Path, domain: str) -> List[Path]:
    """Find adapters for a domain."""
    adapters = []
    domain_lower = domain.lower()
    primary_prefix = domain_lower.split("_")[0]

    domain_mappings = {
        "ns_regularity": ["ns"],
        "chemical_conservation": ["chemical", "chem"],
        "hamiltonian": ["hamiltonian"],
    }

    prefixes = [domain_lower, primary_prefix]
    if domain_lower in domain_mappings:
        prefixes.extend(domain_mappings[domain_lower])

    for adapter_file in adapters_dir.glob("*.npz"):
        name = adapter_file.stem.lower()
        for prefix in prefixes:
            if name.startswith(prefix):
                if adapter_file not in adapters:
                    adapters.append(adapter_file)
                break

    return adapters


class OracleEvaluator:
    """Evaluates facts with adapters."""

    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Base"):
        print(f"Loading model {model_name}...")
        self.model, self.tokenizer = mlx_lm.load(model_name)
        self.model.freeze()
        self.lm_head = get_lm_head_fn(self.model)
        self.vocab_size = self.model.model.embed_tokens.weight.shape[0]
        self.d_model = self.model.model.layers[0].self_attn.q_proj.weight.shape[0]
        self._adapter_cache = {}

    def load_adapter(self, adapter_path: Path, d_inner: int = 64):
        """Load adapter with caching."""
        key = str(adapter_path)
        if key in self._adapter_cache:
            return self._adapter_cache[key]

        cfg = SnapOnConfig(
            d_model=self.d_model, d_inner=d_inner, n_layers=0,
            n_heads=8, mode="logit", vocab_size=self.vocab_size,
        )
        adapter = create_adapter(cfg)

        try:
            weights = mx.load(str(adapter_path))
            adapter.load_weights(list(weights.items()))
            mx.eval(adapter.parameters())
        except ValueError:
            self._adapter_cache[key] = None
            return None

        self._adapter_cache[key] = adapter
        return adapter

    def evaluate_fact(self, fact: dict, adapter=None):
        """Evaluate single fact, return (win, margin)."""
        lm_head = self.lm_head if adapter else None
        win, margin, _, _ = score_fact_mc(
            self.model, self.tokenizer,
            fact["context"], fact["truth"], fact["distractors"],
            adapter=adapter, lm_head=lm_head,
        )
        return win, margin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", help="Specific domains")
    parser.add_argument("--top-domains", type=int, default=0,
                        help="Process top N domains by adapter count")
    parser.add_argument("--output", default="results/meta_router_training.jsonl")
    args = parser.parse_args()

    if not HAS_MLX:
        print("Error: MLX required")
        sys.exit(1)

    base_dir = Path(__file__).parent.parent
    problems_dir = base_dir / "problems"
    adapters_dir = base_dir / "adapters"

    # Find domains with facts
    domain_facts = {}
    for facts_file in problems_dir.glob("*_facts.json"):
        domain = facts_file.stem.replace("_facts", "")
        if "pilot" in domain or "repair" in domain or "balanced" in domain:
            continue
        adapters = find_domain_adapters(adapters_dir, domain)
        if adapters:
            domain_facts[domain] = (facts_file, adapters)

    # Select domains
    if args.domains:
        domains = [(d, domain_facts[d]) for d in args.domains if d in domain_facts]
    elif args.top_domains > 0:
        sorted_domains = sorted(domain_facts.items(),
                               key=lambda x: len(x[1][1]), reverse=True)
        domains = sorted_domains[:args.top_domains]
    else:
        # Default: domains with most adapters
        sorted_domains = sorted(domain_facts.items(),
                               key=lambda x: len(x[1][1]), reverse=True)
        domains = sorted_domains[:5]

    print(f"Processing {len(domains)} domains...")
    for d, (f, a) in domains:
        print(f"  {d}: {len(a)} adapters")

    evaluator = OracleEvaluator()
    all_records = []

    for domain, (facts_file, adapters) in domains:
        facts = load_facts(facts_file)
        print(f"\n{'='*60}")
        print(f"Domain: {domain} ({len(facts)} facts × {len(adapters)} adapters)")
        print(f"{'='*60}")

        # Baseline evaluation
        print("Computing baseline...")
        baseline = {}
        for fact in facts:
            _, margin = evaluator.evaluate_fact(fact)
            baseline[fact["id"]] = margin

        # Evaluate each adapter
        for adapter_path in adapters:
            adapter = evaluator.load_adapter(adapter_path)
            if adapter is None:
                print(f"  {adapter_path.stem}: SKIP (incompatible)")
                continue

            print(f"  {adapter_path.stem}:", end=" ")
            passed = 0
            flipped = 0

            for fact in facts:
                win, margin = evaluator.evaluate_fact(fact, adapter=adapter)
                base_margin = baseline[fact["id"]]
                is_flipped = margin > 0 and base_margin <= 0

                if win:
                    passed += 1
                if is_flipped:
                    flipped += 1

                all_records.append({
                    "fact_id": fact["id"],
                    "fact_text": f"{fact['context']}: {fact['truth']}",
                    "adapter": adapter_path.stem,
                    "domain": domain,
                    "baseline_margin": base_margin,
                    "post_margin": margin,
                    "flipped": is_flipped,
                    "win": win,
                })

            print(f"{passed}/{len(facts)} pass, {flipped} flipped")

    # Save
    output_path = base_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(all_records)}")
    print(f"Flipped: {sum(1 for r in all_records if r['flipped'])}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
