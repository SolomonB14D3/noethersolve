#!/usr/bin/env python3
"""Benchmark stage discovery against manual staging across all domains.

Runs automatic stage discovery on all 1138 facts across 81 domains,
comparing discovered sequences to manual staging results.

Usage:
    python experiments/benchmark_stage_discovery.py
    python experiments/benchmark_stage_discovery.py --domain hamiltonian --method beam
    python experiments/benchmark_stage_discovery.py --quick  # Test on 3 domains only
"""

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from noethersolve.stage_discovery import (
    StageDiscoverer,
    DiscoveryConfig,
    EvalResult,
)


@dataclass
class DomainResult:
    """Results for a single domain."""
    domain: str
    n_facts: int
    baseline_pass: int
    baseline_pct: float
    discovered_sequence: List[str]
    discovered_pass: int
    discovered_pct: float
    manual_sequence: Optional[List[str]]
    manual_pass: Optional[int]
    manual_pct: Optional[float]
    discovery_time_sec: float
    method: str


@dataclass
class BenchmarkReport:
    """Full benchmark report."""
    results: List[DomainResult] = field(default_factory=list)
    total_facts: int = 0
    total_baseline_pass: int = 0
    total_discovered_pass: int = 0
    total_manual_pass: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "STAGE DISCOVERY BENCHMARK REPORT",
            "=" * 70,
            "",
            f"Total domains: {len(self.results)}",
            f"Total facts: {self.total_facts}",
            "",
            "AGGREGATE RESULTS:",
            f"  Baseline:   {self.total_baseline_pass}/{self.total_facts} "
            f"({100*self.total_baseline_pass/max(1,self.total_facts):.1f}%)",
            f"  Discovered: {self.total_discovered_pass}/{self.total_facts} "
            f"({100*self.total_discovered_pass/max(1,self.total_facts):.1f}%)",
        ]

        if self.total_manual_pass > 0:
            lines.append(
                f"  Manual:     {self.total_manual_pass}/{self.total_facts} "
                f"({100*self.total_manual_pass/max(1,self.total_facts):.1f}%)"
            )

        lines.extend([
            "",
            "PER-DOMAIN RESULTS:",
            "-" * 70,
            f"{'Domain':<30} {'Base':>6} {'Disc':>6} {'Man':>6} {'Method':<8}",
            "-" * 70,
        ])

        for r in sorted(self.results, key=lambda x: -x.discovered_pct):
            manual_str = f"{r.manual_pct:.0f}%" if r.manual_pct else "N/A"
            lines.append(
                f"{r.domain[:30]:<30} "
                f"{r.baseline_pct:>5.1f}% "
                f"{r.discovered_pct:>5.1f}% "
                f"{manual_str:>6} "
                f"{r.method:<8}"
            )

        lines.extend([
            "-" * 70,
            "",
            "DISCOVERY vs MANUAL COMPARISON:",
        ])

        better = sum(1 for r in self.results
                     if r.manual_pct and r.discovered_pct > r.manual_pct)
        equal = sum(1 for r in self.results
                    if r.manual_pct and r.discovered_pct == r.manual_pct)
        worse = sum(1 for r in self.results
                    if r.manual_pct and r.discovered_pct < r.manual_pct)

        lines.append(f"  Discovery beats manual: {better}")
        lines.append(f"  Discovery equals manual: {equal}")
        lines.append(f"  Discovery worse than manual: {worse}")

        return "\n".join(lines)


def load_domain_facts(problems_dir: Path) -> Dict[str, List[dict]]:
    """Load all facts organized by domain."""
    domain_facts = {}

    for facts_file in problems_dir.glob("*_facts.json"):
        try:
            with open(facts_file) as f:
                data = json.load(f)

            facts = data.get("facts", data.get("verifications", []))
            domain = facts_file.stem.replace("_facts", "")

            # Normalize facts to have id and truth
            normalized = []
            for i, fact in enumerate(facts):
                if isinstance(fact, dict):
                    normalized.append({
                        "id": fact.get("id", f"{domain}_{i:02d}"),
                        "truth": fact.get("truth", fact.get("fact", "")),
                        "distractors": fact.get("distractors", []),
                    })
                else:
                    normalized.append({
                        "id": f"{domain}_{i:02d}",
                        "truth": str(fact),
                        "distractors": [],
                    })

            if normalized:
                domain_facts[domain] = normalized

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {facts_file}: {e}")

    return domain_facts


def find_domain_adapters(adapters_dir: Path, domain: str) -> List[str]:
    """Find all adapters that belong to a domain."""
    adapters = []
    domain_lower = domain.lower()

    for adapter_file in adapters_dir.glob("*.npz"):
        name = adapter_file.stem.lower()
        # Match by prefix or containing domain name
        if name.startswith(domain_lower) or domain_lower in name:
            adapters.append(adapter_file.stem)

    return adapters


def create_simulated_oracle(
    domain: str,
    adapters: List[str],
    baseline_rate: float = 0.1,
    adapter_boost: float = 0.3,
) -> callable:
    """Create a simulated oracle for testing.

    In production, this would call the actual log-prob oracle.
    For benchmarking without GPU, we simulate based on known patterns.
    """
    import random
    import hashlib

    def oracle(adapter: str, facts: List[dict]) -> EvalResult:
        # Deterministic pseudo-random based on adapter name
        seed = int(hashlib.md5(adapter.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Each adapter passes a subset of facts
        n = len(facts)

        # Adapter-specific pass rate (varies by adapter)
        adapter_rate = baseline_rate + adapter_boost * rng.random()

        passed_ids = set()
        margins = []

        for fact in facts:
            fact_id = fact.get("id", "")
            # Deterministic per fact-adapter pair
            pair_seed = hash((adapter, fact_id)) % (2**31)
            pair_rng = random.Random(pair_seed)

            passes = pair_rng.random() < adapter_rate
            margin = pair_rng.uniform(5, 50) if passes else pair_rng.uniform(-50, -5)

            margins.append(margin)
            if passes:
                passed_ids.add(fact_id)

        return EvalResult(
            adapter=adapter,
            n_passed=len(passed_ids),
            n_total=n,
            margins=margins,
            passed_ids=passed_ids,
        )

    return oracle


def load_manual_results() -> Dict[str, Tuple[List[str], int]]:
    """Load manually discovered staging results from CLAUDE.md or results.

    Returns: {domain: (sequence, n_passed)}
    """
    # Hard-coded from CLAUDE.md established results
    manual_results = {
        "hamiltonian": (
            ["hamiltonian_symplectic", "hamiltonian_noether", "hamiltonian_energy",
             "hamiltonian_action", "hamiltonian_systems"],
            16  # 16/16
        ),
        "ns_regularity": (
            ["ns_conservation", "ns_blowup", "ns_regularity", "ns_vortex"],
            16  # 16/16 with orthogonal
        ),
        "knot_invariants": (
            ["knot_jones", "knot_reidemeister", "knot_invariants"],
            16  # 16/16
        ),
        "chemical_conservation": (
            ["chemical_kinetics", "chemical_equilibrium"],
            16  # 16/16
        ),
    }
    return manual_results


def run_benchmark(
    domain_facts: Dict[str, List[dict]],
    adapters_dir: Path,
    method: str = "greedy",
    domains_filter: Optional[List[str]] = None,
    use_real_oracle: bool = False,
) -> BenchmarkReport:
    """Run benchmark across all domains."""
    report = BenchmarkReport()
    manual_results = load_manual_results()

    # Filter domains if specified
    if domains_filter:
        domain_facts = {k: v for k, v in domain_facts.items() if k in domains_filter}

    for domain, facts in sorted(domain_facts.items()):
        print(f"\n{'='*50}")
        print(f"Domain: {domain} ({len(facts)} facts)")
        print(f"{'='*50}")

        # Find adapters for this domain
        adapters = find_domain_adapters(adapters_dir, domain)

        if not adapters:
            # Try broader matching
            adapters = [a.stem for a in adapters_dir.glob("*.npz")][:10]
            print(f"  No domain-specific adapters found, using {len(adapters)} general adapters")
        else:
            print(f"  Found {len(adapters)} domain adapters")

        # Create oracle (simulated or real)
        if use_real_oracle:
            # TODO: Implement real oracle call
            print("  Warning: Real oracle not implemented, using simulation")

        oracle = create_simulated_oracle(domain, adapters)

        # Get baseline (no adapter)
        baseline_result = oracle("baseline", facts)
        baseline_pass = baseline_result.n_passed

        # Run stage discovery
        start_time = time.time()

        config = DiscoveryConfig(
            max_stages=min(5, len(adapters)),
            min_improvement=1,
            regression_tolerance=1,
            beam_width=3,
            population_size=10,
            generations=20,
        )

        discoverer = StageDiscoverer(
            facts=facts,
            candidate_adapters=adapters,
            oracle_fn=oracle,
            config=config,
        )

        result = discoverer.discover(method)
        elapsed = time.time() - start_time

        # Get manual results if available
        manual_seq = None
        manual_pass = None
        manual_pct = None
        if domain in manual_results:
            manual_seq, manual_pass = manual_results[domain]
            manual_pct = 100 * manual_pass / len(facts)

        # Record results
        domain_result = DomainResult(
            domain=domain,
            n_facts=len(facts),
            baseline_pass=baseline_pass,
            baseline_pct=100 * baseline_pass / len(facts),
            discovered_sequence=result.adapters,
            discovered_pass=len(result.passed_ids),
            discovered_pct=100 * result.score,
            manual_sequence=manual_seq,
            manual_pass=manual_pass,
            manual_pct=manual_pct,
            discovery_time_sec=elapsed,
            method=method,
        )

        report.results.append(domain_result)
        report.total_facts += len(facts)
        report.total_baseline_pass += baseline_pass
        report.total_discovered_pass += len(result.passed_ids)
        if manual_pass:
            report.total_manual_pass += manual_pass

        print(f"  Baseline: {baseline_pass}/{len(facts)} ({domain_result.baseline_pct:.1f}%)")
        print(f"  Discovered: {len(result.passed_ids)}/{len(facts)} ({domain_result.discovered_pct:.1f}%)")
        print(f"  Sequence: {result.adapters[:3]}{'...' if len(result.adapters) > 3 else ''}")
        print(f"  Time: {elapsed:.2f}s")

    return report


def main():
    parser = argparse.ArgumentParser(description="Benchmark stage discovery")
    parser.add_argument("--domain", help="Run on single domain")
    parser.add_argument("--method", default="greedy", choices=["greedy", "beam", "genetic"])
    parser.add_argument("--quick", action="store_true", help="Quick test on 3 domains")
    parser.add_argument("--output", help="Output JSON file for results")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    problems_dir = base_dir / "problems"
    adapters_dir = base_dir / "adapters"

    print("Loading facts...")
    domain_facts = load_domain_facts(problems_dir)
    print(f"Loaded {sum(len(f) for f in domain_facts.values())} facts from {len(domain_facts)} domains")

    # Filter domains
    domains_filter = None
    if args.domain:
        domains_filter = [args.domain]
    elif args.quick:
        # Test on domains with manual results
        domains_filter = ["hamiltonian", "ns_regularity", "knot_invariants"]

    print(f"\nRunning stage discovery with method={args.method}...")
    report = run_benchmark(
        domain_facts,
        adapters_dir,
        method=args.method,
        domains_filter=domains_filter,
    )

    print("\n" + report.summary())

    # Save results
    if args.output:
        output_data = {
            "total_facts": report.total_facts,
            "total_baseline_pass": report.total_baseline_pass,
            "total_discovered_pass": report.total_discovered_pass,
            "total_manual_pass": report.total_manual_pass,
            "results": [
                {
                    "domain": r.domain,
                    "n_facts": r.n_facts,
                    "baseline_pass": r.baseline_pass,
                    "baseline_pct": r.baseline_pct,
                    "discovered_sequence": r.discovered_sequence,
                    "discovered_pass": r.discovered_pass,
                    "discovered_pct": r.discovered_pct,
                    "manual_sequence": r.manual_sequence,
                    "manual_pass": r.manual_pass,
                    "manual_pct": r.manual_pct,
                    "discovery_time_sec": r.discovery_time_sec,
                    "method": r.method,
                }
                for r in report.results
            ]
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
