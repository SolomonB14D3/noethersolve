#!/usr/bin/env python3
"""Run full oracle evaluation for stage discovery benchmark.

This runs actual oracle evaluations (not simulated) to build
proper training data for the meta-router and validate stage discovery.

Usage:
    python experiments/run_full_oracle_benchmark.py --domain hamiltonian
    python experiments/run_full_oracle_benchmark.py --all-core  # 4 core domains
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for MLX availability
try:
    import mlx.core as mx
    import mlx_lm
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("Warning: MLX not available. Install with: pip install mlx mlx-lm")

from noethersolve.stage_discovery import (
    StageDiscoverer,
    DiscoveryConfig,
    EvalResult,
    StageSequence,
)
from noethersolve.outcome_logger import OutcomeLogger


def load_facts(facts_file: Path) -> List[dict]:
    """Load facts from a JSON file."""
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


def find_adapters(adapters_dir: Path, domain: str) -> List[Path]:
    """Find all adapters for a domain.

    Uses smart matching: ns_regularity matches ns_*, chemical_conservation matches chem_*, etc.
    """
    adapters = []
    domain_lower = domain.lower()

    # Extract primary prefix (first component) for matching
    # e.g., "ns_regularity" -> "ns", "chemical_conservation" -> "chemical"
    primary_prefix = domain_lower.split("_")[0]

    # Also try common variations
    prefixes_to_match = [domain_lower, primary_prefix]

    # Domain-specific mappings for non-obvious cases
    domain_mappings = {
        "ns_regularity": ["ns", "ns_regularity", "navier"],
        "chemical_conservation": ["chemical", "chem"],
        "knot_invariants": ["knot"],
        "hamiltonian": ["hamiltonian", "ham"],
    }

    if domain_lower in domain_mappings:
        prefixes_to_match.extend(domain_mappings[domain_lower])

    for adapter_file in adapters_dir.glob("*.npz"):
        name = adapter_file.stem.lower()
        for prefix in prefixes_to_match:
            if name.startswith(prefix) or prefix in name:
                if adapter_file not in adapters:
                    adapters.append(adapter_file)
                break

    return adapters


class MLXOracle:
    """Oracle using MLX log-prob scoring with adapter support."""

    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Base"):
        if not HAS_MLX:
            raise RuntimeError("MLX not available")

        print(f"Loading model {model_name}...")
        self.model, self.tokenizer = mlx_lm.load(model_name)
        self.model.freeze()
        self.model_name = model_name
        self._cache = {}
        self._adapter_cache = {}  # Cache loaded adapters

        # Get lm_head for adapter application
        from noethersolve.train_utils import get_lm_head_fn
        self.lm_head = get_lm_head_fn(self.model)

        # Get model dimensions for adapter loading
        self.vocab_size = self.model.model.embed_tokens.weight.shape[0]
        self.d_model = self.model.model.layers[0].self_attn.q_proj.weight.shape[0]

    def _load_adapter(self, adapter_path: Path, d_inner: int = 64):
        """Load an adapter from .npz file, with caching.

        Returns None if the adapter is incompatible (different architecture).
        """
        key = str(adapter_path)
        if key in self._adapter_cache:
            return self._adapter_cache[key]

        from noethersolve.adapter import SnapOnConfig, create_adapter

        cfg = SnapOnConfig(
            d_model=self.d_model,
            d_inner=d_inner,
            n_layers=0,
            n_heads=8,
            mode="logit",
            vocab_size=self.vocab_size,
        )
        adapter = create_adapter(cfg)

        try:
            weights = mx.load(str(adapter_path))
            adapter.load_weights(list(weights.items()))
            mx.eval(adapter.parameters())
        except ValueError as e:
            # Incompatible architecture (e.g., different weight names)
            self._adapter_cache[key] = None
            return None

        self._adapter_cache[key] = adapter
        return adapter

    def _get_logprob(self, text: str, adapter=None) -> float:
        """Get log probability of text, optionally with adapter."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return -1000.0

        tokens_mx = mx.array([tokens])

        if adapter is not None:
            # Use adapter: get hidden states, apply lm_head, then adapter
            from noethersolve.train_utils import apply_adapter
            h = self.model.model(tokens_mx)
            mx.eval(h)
            base_logits = self.lm_head(h)
            mx.eval(base_logits)
            logits = apply_adapter(adapter, base_logits)
            mx.eval(logits)
        else:
            # No adapter: standard forward pass
            logits = self.model(tokens_mx)

        # Get log probs for each next token using softmax
        logits_shifted = logits[0, :-1]  # Predictions for positions 0 to n-2
        # Manual log softmax: log(exp(x) / sum(exp(x))) = x - log(sum(exp(x)))
        max_logits = mx.max(logits_shifted, axis=-1, keepdims=True)
        shifted = logits_shifted - max_logits
        log_sum_exp = mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp

        target_tokens = mx.array(tokens[1:])

        # Gather log probs for actual tokens
        total_lp = 0.0
        for i, tok in enumerate(target_tokens.tolist()):
            total_lp += float(log_probs[i, tok])

        return total_lp

    def evaluate(self, adapter_path: Optional[Path], facts: List[dict]) -> EvalResult:
        """Evaluate facts with optional adapter using proper oracle scoring.

        Returns None if adapter is incompatible and cannot be loaded.
        """
        from noethersolve.oracle import score_fact_mc

        adapter_name = adapter_path.stem if adapter_path else "baseline"

        # Load adapter if provided
        adapter = None
        lm_head = None
        if adapter_path is not None and adapter_path.exists():
            adapter = self._load_adapter(adapter_path)
            if adapter is None:
                # Incompatible adapter - return None to signal skip
                return None
            lm_head = self.lm_head

        passed_ids = set()
        margins = []

        for fact in facts:
            fact_id = fact.get("id", "")
            context = fact.get("context", "")
            truth = fact.get("truth", "")
            distractors = fact.get("distractors", [])

            # Use the proper oracle scoring with context
            win, margin, _, _ = score_fact_mc(
                self.model, self.tokenizer,
                context, truth, distractors,
                adapter=adapter, lm_head=lm_head,
            )

            margins.append(margin)
            if win:
                passed_ids.add(fact_id)

        return EvalResult(
            adapter=adapter_name,
            n_passed=len(passed_ids),
            n_total=len(facts),
            margins=margins,
            passed_ids=passed_ids,
        )


def run_domain_benchmark(
    domain: str,
    facts: List[dict],
    adapters: List[Path],
    oracle: MLXOracle,
    method: str = "greedy",
    logger: Optional[OutcomeLogger] = None,
) -> dict:
    """Run full benchmark for a single domain."""
    print(f"\n{'='*60}")
    print(f"Domain: {domain} ({len(facts)} facts, {len(adapters)} adapters)")
    print(f"{'='*60}")

    # Baseline evaluation
    print("Running baseline evaluation...")
    baseline = oracle.evaluate(None, facts)
    print(f"  Baseline: {baseline.n_passed}/{baseline.n_total} "
          f"({100*baseline.n_passed/baseline.n_total:.1f}%)")

    # Evaluate each adapter
    adapter_results = {}
    compatible_adapters = []
    for adapter_path in adapters:
        print(f"  Evaluating {adapter_path.stem}...")
        result = oracle.evaluate(adapter_path, facts)
        if result is None:
            print(f"    → SKIPPED (incompatible architecture)")
            continue
        adapter_results[adapter_path.stem] = result
        compatible_adapters.append(adapter_path)
        print(f"    → {result.n_passed}/{result.n_total}")

        # Log outcomes
        if logger:
            for i, fact in enumerate(facts):
                logger.log_outcome(
                    fact_id=fact.get("id", f"fact_{i}"),
                    fact_text=fact.get("truth", ""),
                    baseline_margin=baseline.margins[i],
                    adapter=adapter_path.stem,
                    post_margin=result.margins[i],
                    flipped=result.margins[i] > 0,
                    domain=domain,
                )

    # Create oracle function for stage discovery
    def cached_oracle(adapter_name: str, _facts: List[dict]) -> EvalResult:
        if adapter_name in adapter_results:
            return adapter_results[adapter_name]
        # Fallback to baseline
        return baseline

    # Run stage discovery (only on compatible adapters)
    print(f"\nRunning stage discovery ({method})...")
    n_compatible = len(adapter_results)

    if n_compatible == 0:
        # No compatible adapters - use baseline
        print("  No compatible adapters - using baseline")
        result = StageSequence(
            adapters=[],
            passed_ids=baseline.passed_ids,
            score=baseline.n_passed / baseline.n_total,
        )
        elapsed = 0.0
    else:
        config = DiscoveryConfig(
            max_stages=min(5, n_compatible),
            min_improvement=1,
            regression_tolerance=1,
        )

        discoverer = StageDiscoverer(
            facts=facts,
            candidate_adapters=list(adapter_results.keys()),
            oracle_fn=cached_oracle,
            config=config,
        )

        start_time = time.time()
        result = discoverer.discover(method)
        elapsed = time.time() - start_time

        # If no adapters helped, use baseline
        if len(result.passed_ids) < baseline.n_passed:
            print(f"  No improvement over baseline - keeping baseline ({baseline.n_passed}/{baseline.n_total})")
            result = StageSequence(
                adapters=[],
                passed_ids=baseline.passed_ids,
                score=baseline.n_passed / baseline.n_total,
            )

    print(f"\nResults:")
    print(f"  Discovered sequence: {result.adapters}")
    print(f"  Final accuracy: {len(result.passed_ids)}/{len(facts)} ({100*result.score:.1f}%)")
    print(f"  Discovery time: {elapsed:.2f}s")

    return {
        "domain": domain,
        "n_facts": len(facts),
        "n_adapters": len(adapters),
        "baseline_pass": baseline.n_passed,
        "baseline_pct": 100 * baseline.n_passed / len(facts),
        "discovered_sequence": result.adapters,
        "discovered_pass": len(result.passed_ids),
        "discovered_pct": 100 * result.score,
        "discovery_time_sec": elapsed,
        "method": method,
        "adapter_results": {
            name: {"n_passed": r.n_passed, "pct": 100 * r.n_passed / r.n_total}
            for name, r in adapter_results.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="Single domain to benchmark")
    parser.add_argument("--all-core", action="store_true",
                        help="Benchmark 4 core domains")
    parser.add_argument("--all", action="store_true",
                        help="Benchmark all domains with facts and adapters")
    parser.add_argument("--method", default="greedy",
                        choices=["greedy", "beam", "genetic", "guided"])
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    if not HAS_MLX:
        print("Error: MLX required for real oracle evaluation")
        print("Install with: pip install mlx mlx-lm")
        sys.exit(1)

    base_dir = Path(__file__).parent.parent
    problems_dir = base_dir / "problems"
    adapters_dir = base_dir / "adapters"

    # Determine which domains to run
    if args.domain:
        domains = [args.domain]
    elif args.all_core:
        domains = ["hamiltonian", "ns_regularity", "knot_invariants", "chemical_conservation"]
    elif getattr(args, 'all', False):
        # Find all domains with facts files
        domains = []
        for facts_file in problems_dir.glob("*_facts.json"):
            domain = facts_file.stem.replace("_facts", "")
            # Skip pilot/test files
            if "pilot" in domain or "repair" in domain or "balanced" in domain:
                continue
            domains.append(domain)
        domains = sorted(set(domains))
        print(f"Found {len(domains)} domains with facts files")
    else:
        # Default: hamiltonian only
        domains = ["hamiltonian"]

    # Initialize
    oracle = MLXOracle()
    logger = OutcomeLogger()
    results = []

    for domain in domains:
        facts_file = problems_dir / f"{domain}_facts.json"
        if not facts_file.exists():
            print(f"Warning: No facts file for {domain}")
            continue

        facts = load_facts(facts_file)
        adapters = find_adapters(adapters_dir, domain)

        if not adapters:
            print(f"Warning: No adapters found for {domain}")
            continue

        result = run_domain_benchmark(
            domain=domain,
            facts=facts,
            adapters=adapters,
            oracle=oracle,
            method=args.method,
            logger=logger,
        )
        results.append(result)

    # Close logger
    logger.close()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not results:
        print("No results to summarize.")
        return

    total_facts = sum(r["n_facts"] for r in results)
    total_baseline = sum(r["baseline_pass"] for r in results)
    total_discovered = sum(r["discovered_pass"] for r in results)

    print(f"Domains evaluated: {len(results)}")
    print(f"Total facts: {total_facts}")
    print(f"Baseline: {total_baseline}/{total_facts} ({100*total_baseline/total_facts:.1f}%)")
    print(f"Discovered: {total_discovered}/{total_facts} ({100*total_discovered/total_facts:.1f}%)")
    print(f"Improvement: +{total_discovered - total_baseline} facts "
          f"(+{100*(total_discovered - total_baseline)/total_facts:.1f}pp)")

    # Per-domain table
    print("\n" + "-" * 70)
    print(f"{'Domain':<35} {'Base':>8} {'Disc':>8} {'Δ':>6} {'Seq':<15}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: -x["discovered_pct"]):
        base_str = f"{r['baseline_pass']}/{r['n_facts']}"
        disc_str = f"{r['discovered_pass']}/{r['n_facts']}"
        delta = r["discovered_pass"] - r["baseline_pass"]
        seq_str = ",".join(r["discovered_sequence"][:2]) if r["discovered_sequence"] else "-"
        if len(r["discovered_sequence"]) > 2:
            seq_str += "..."
        print(f"{r['domain'][:35]:<35} {base_str:>8} {disc_str:>8} {'+' + str(delta):>6} {seq_str:<15}")
    print("-" * 70)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
