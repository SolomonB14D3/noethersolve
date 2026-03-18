"""Stage Discovery — Automatically find optimal adapter sequences.

Phase 2 of full automation: given a domain with 0% baseline, discover
the optimal adapter sequence to flip all facts.

The Hamiltonian domain took 5 manual stages to reach 16/16.
This module automates that discovery process.

Approaches:
1. Greedy: Add adapter that flips most remaining facts without regression
2. Beam search: Track top-k partial sequences
3. Genetic: Evolve populations of adapter sequences
4. Guided: Use meta-router to prioritize adapters for remaining facts

Usage:
    discoverer = StageDiscoverer(
        facts_file="problems/hamiltonian_facts.json",
        adapters_dir="adapters/",
        oracle_fn=run_oracle  # Function to evaluate
    )

    # Find optimal sequence
    sequence = discoverer.discover(method="greedy", max_stages=10)
    # Returns: ["hamiltonian_symplectic", "hamiltonian_noether", "hamiltonian_energy", ...]

    # Or use beam search for potentially better solutions
    sequence = discoverer.discover(method="beam", beam_width=3)

    # Or use meta-router guided search
    sequence = discoverer.discover(method="guided", router_path="results/meta_router.json")
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set
from pathlib import Path
import json
import random
import numpy as np
from collections import Counter
import heapq


@dataclass
class EvalResult:
    """Result of evaluating an adapter on facts."""
    adapter: str
    n_passed: int
    n_total: int
    margins: List[float]  # Per-fact margins
    passed_ids: Set[str]  # IDs of facts that passed
    regressed_ids: Set[str] = field(default_factory=set)  # Facts that got worse


@dataclass
class StageSequence:
    """A sequence of adapters and its cumulative results."""
    adapters: List[str]
    passed_ids: Set[str]
    score: float  # n_passed / n_total
    history: List[EvalResult] = field(default_factory=list)

    def __lt__(self, other):
        # For heap: higher score = better
        return self.score > other.score


@dataclass
class DiscoveryConfig:
    """Configuration for stage discovery."""
    max_stages: int = 10
    min_improvement: int = 1  # Min facts to flip per stage
    regression_tolerance: int = 0  # Max facts allowed to regress
    beam_width: int = 3  # For beam search
    population_size: int = 20  # For genetic algorithm
    mutation_rate: float = 0.1
    generations: int = 50
    early_stop_threshold: float = 1.0  # Stop if this accuracy reached


class StageDiscoverer:
    """Discovers optimal adapter sequences for a domain."""

    def __init__(
        self,
        facts: List[dict],
        candidate_adapters: List[str],
        oracle_fn: Callable[[str, List[dict]], EvalResult],
        config: Optional[DiscoveryConfig] = None,
    ):
        """
        Args:
            facts: List of fact dicts with 'id', 'truth', etc.
            candidate_adapters: List of adapter names/paths to consider
            oracle_fn: Function(adapter, facts) -> EvalResult
            config: Discovery configuration
        """
        self.facts = facts
        self.fact_ids = {f.get("id", f"fact_{i}") for i, f in enumerate(facts)}
        self.candidate_adapters = candidate_adapters
        self.oracle_fn = oracle_fn
        self.config = config or DiscoveryConfig()

        # Cache for eval results
        self._eval_cache: Dict[str, EvalResult] = {}

    def _evaluate(self, adapter: str) -> EvalResult:
        """Evaluate an adapter, using cache if available."""
        if adapter not in self._eval_cache:
            self._eval_cache[adapter] = self.oracle_fn(adapter, self.facts)
        return self._eval_cache[adapter]

    def discover_greedy(self) -> StageSequence:
        """Greedy discovery: add adapter that flips most facts without regression."""
        sequence = StageSequence(adapters=[], passed_ids=set(), score=0.0)
        remaining_adapters = set(self.candidate_adapters)

        for stage in range(self.config.max_stages):
            best_adapter = None
            best_improvement = 0
            best_result = None

            for adapter in remaining_adapters:
                result = self._evaluate(adapter)

                # Count new facts flipped
                new_passed = result.passed_ids - sequence.passed_ids
                improvement = len(new_passed)

                # Check for regression
                regressed = sequence.passed_ids - result.passed_ids
                if len(regressed) > self.config.regression_tolerance:
                    continue  # Skip adapters that cause too much regression

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_adapter = adapter
                    best_result = result

            if best_adapter is None or best_improvement < self.config.min_improvement:
                break  # No more progress possible

            # Add to sequence
            sequence.adapters.append(best_adapter)
            sequence.passed_ids.update(best_result.passed_ids)
            sequence.score = len(sequence.passed_ids) / len(self.fact_ids)
            sequence.history.append(best_result)
            remaining_adapters.remove(best_adapter)

            print(f"Stage {stage+1}: +{best_adapter} → {len(sequence.passed_ids)}/{len(self.fact_ids)} "
                  f"({sequence.score:.1%})")

            if sequence.score >= self.config.early_stop_threshold:
                break

        return sequence

    def discover_beam(self) -> StageSequence:
        """Beam search: track top-k partial sequences."""
        # Initialize with empty sequence
        beam = [StageSequence(adapters=[], passed_ids=set(), score=0.0)]

        for stage in range(self.config.max_stages):
            candidates = []

            for seq in beam:
                used_adapters = set(seq.adapters)

                for adapter in self.candidate_adapters:
                    if adapter in used_adapters:
                        continue

                    result = self._evaluate(adapter)

                    # Check regression
                    regressed = seq.passed_ids - result.passed_ids
                    if len(regressed) > self.config.regression_tolerance:
                        continue

                    # Create new sequence
                    new_passed = seq.passed_ids | result.passed_ids
                    new_seq = StageSequence(
                        adapters=seq.adapters + [adapter],
                        passed_ids=new_passed,
                        score=len(new_passed) / len(self.fact_ids),
                        history=seq.history + [result],
                    )
                    candidates.append(new_seq)

            if not candidates:
                break

            # Keep top-k by score
            beam = heapq.nsmallest(self.config.beam_width, candidates)

            best = beam[0]
            print(f"Stage {stage+1}: best={len(best.passed_ids)}/{len(self.fact_ids)} "
                  f"({best.score:.1%}), beam_size={len(beam)}")

            if best.score >= self.config.early_stop_threshold:
                break

        return beam[0] if beam else StageSequence(adapters=[], passed_ids=set(), score=0.0)

    def discover_genetic(self) -> StageSequence:
        """Genetic algorithm: evolve populations of adapter sequences."""
        # Initialize population with random sequences
        population = []
        for _ in range(self.config.population_size):
            # Random sequence of 1-5 adapters
            length = random.randint(1, min(5, len(self.candidate_adapters)))
            adapters = random.sample(self.candidate_adapters, length)
            population.append(adapters)

        best_ever = StageSequence(adapters=[], passed_ids=set(), score=0.0)

        for gen in range(self.config.generations):
            # Evaluate population
            scored = []
            for adapters in population:
                passed = set()
                for adapter in adapters:
                    result = self._evaluate(adapter)
                    passed.update(result.passed_ids)

                score = len(passed) / len(self.fact_ids)
                seq = StageSequence(adapters=adapters, passed_ids=passed, score=score)
                scored.append(seq)

                if score > best_ever.score:
                    best_ever = seq

            # Sort by fitness
            scored.sort(key=lambda s: s.score, reverse=True)

            print(f"Gen {gen+1}: best={scored[0].score:.1%}, "
                  f"avg={sum(s.score for s in scored)/len(scored):.1%}")

            if best_ever.score >= self.config.early_stop_threshold:
                break

            # Selection: keep top half
            survivors = scored[:self.config.population_size // 2]

            # Crossover + mutation
            new_population = [s.adapters for s in survivors]
            while len(new_population) < self.config.population_size:
                # Crossover
                p1 = random.choice(survivors).adapters
                p2 = random.choice(survivors).adapters

                # One-point crossover
                if len(p1) > 1 and len(p2) > 1:
                    cut1 = random.randint(1, len(p1) - 1)
                    cut2 = random.randint(1, len(p2) - 1)
                    child = p1[:cut1] + p2[cut2:]
                else:
                    child = p1 + p2

                # Remove duplicates
                seen = set()
                child = [a for a in child if not (a in seen or seen.add(a))]

                # Mutation
                if random.random() < self.config.mutation_rate:
                    if child and random.random() < 0.5:
                        # Remove random adapter
                        child.pop(random.randint(0, len(child) - 1))
                    else:
                        # Add random adapter
                        available = set(self.candidate_adapters) - set(child)
                        if available:
                            child.append(random.choice(list(available)))

                if child:  # Don't add empty sequences
                    new_population.append(child)

            population = new_population[:self.config.population_size]

        return best_ever

    def discover_guided(
        self,
        router_path: Optional[str] = None,
        fact_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> StageSequence:
        """Guided discovery: use meta-router to prioritize adapters.

        The meta-router predicts which adapters are likely to flip each remaining fact.
        We prioritize adapters that have high predicted success for remaining facts.

        Args:
            router_path: Path to meta_router.json file
            fact_embeddings: Pre-computed embeddings for facts

        Returns:
            StageSequence with discovered adapters
        """
        # Load meta-router
        if router_path is None:
            router_path = Path(__file__).parent.parent / "results" / "meta_router.json"
        else:
            router_path = Path(router_path)

        if not router_path.exists():
            print(f"Meta-router not found at {router_path}, falling back to greedy")
            return self.discover_greedy()

        # Load router state
        with open(router_path) as f:
            router_state = json.load(f)

        adapter_centroids = {
            k: np.array(v[0], dtype=np.float32)
            for k, v in router_state["adapter_centroids"].items()
        }

        # Get embeddings for facts if not provided
        if fact_embeddings is None:
            fact_embeddings = self._get_fact_embeddings()

        sequence = StageSequence(adapters=[], passed_ids=set(), score=0.0)

        # Consider ALL candidate adapters, not just those in the router
        # Router is used to prioritize, but non-router adapters are also tried
        routed_adapters = set(self.candidate_adapters) & set(adapter_centroids.keys())
        non_routed_adapters = set(self.candidate_adapters) - set(adapter_centroids.keys())
        remaining_routed = set(routed_adapters)
        remaining_non_routed = set(non_routed_adapters)

        if not routed_adapters and not non_routed_adapters:
            print("No candidate adapters available")
            return sequence

        for stage in range(self.config.max_stages):
            # Find remaining facts
            remaining_fact_ids = self.fact_ids - sequence.passed_ids

            if not remaining_fact_ids:
                break

            # Score each adapter by how well it matches remaining facts
            adapter_scores = Counter()

            for fid in remaining_fact_ids:
                if fid not in fact_embeddings:
                    continue

                emb = fact_embeddings[fid]
                emb = emb / (np.linalg.norm(emb) + 1e-8)

                # Find best adapter for this fact
                best_sim = -1
                best_adapter = None

                for adapter in remaining_routed:
                    if adapter in adapter_centroids:
                        centroid = adapter_centroids[adapter]
                        sim = float(np.dot(emb, centroid))
                        if sim > best_sim:
                            best_sim = sim
                            best_adapter = adapter

                if best_adapter:
                    adapter_scores[best_adapter] += 1

            # Try adapters in order of predicted relevance
            best_adapter = None
            best_improvement = 0
            best_result = None

            # Sort by predicted score, then try each
            for adapter, _ in adapter_scores.most_common():
                if adapter not in remaining_routed:
                    continue

                result = self._evaluate(adapter)

                # Count new facts flipped
                new_passed = result.passed_ids - sequence.passed_ids
                improvement = len(new_passed)

                # Check for regression
                regressed = sequence.passed_ids - result.passed_ids
                if len(regressed) > self.config.regression_tolerance:
                    continue

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_adapter = adapter
                    best_result = result

                # Early exit if we found a good one
                if improvement >= 3:
                    break

            if best_adapter is None or best_improvement < self.config.min_improvement:
                # No guided option worked, try remaining routed adapters
                for adapter in remaining_routed:
                    if adapter in [a for a, _ in adapter_scores.most_common()]:
                        continue  # Already tried

                    result = self._evaluate(adapter)
                    new_passed = result.passed_ids - sequence.passed_ids
                    improvement = len(new_passed)

                    regressed = sequence.passed_ids - result.passed_ids
                    if len(regressed) > self.config.regression_tolerance:
                        continue

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_adapter = adapter
                        best_result = result

            if best_adapter is None or best_improvement < self.config.min_improvement:
                # Still no luck - try non-routed adapters (not in meta-router)
                for adapter in remaining_non_routed:
                    result = self._evaluate(adapter)
                    new_passed = result.passed_ids - sequence.passed_ids
                    improvement = len(new_passed)

                    regressed = sequence.passed_ids - result.passed_ids
                    if len(regressed) > self.config.regression_tolerance:
                        continue

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_adapter = adapter
                        best_result = result

            if best_adapter is None or best_improvement < self.config.min_improvement:
                break

            # Add to sequence
            sequence.adapters.append(best_adapter)
            sequence.passed_ids.update(best_result.passed_ids)
            sequence.score = len(sequence.passed_ids) / len(self.fact_ids)
            sequence.history.append(best_result)

            # Remove from appropriate set
            if best_adapter in remaining_routed:
                remaining_routed.remove(best_adapter)
                source = "guided"
            else:
                remaining_non_routed.remove(best_adapter)
                source = "fallback"

            print(f"Stage {stage+1}: +{best_adapter} → {len(sequence.passed_ids)}/{len(self.fact_ids)} "
                  f"({sequence.score:.1%}) [{source}]")

            if sequence.score >= self.config.early_stop_threshold:
                break

        return sequence

    def _get_fact_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute embeddings for facts using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return {}

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        embeddings = {}
        for fact in self.facts:
            fid = fact.get("id", "")
            text = fact.get("truth", fact.get("fact", ""))
            if fact.get("context"):
                text = f"{fact['context']}: {text}"

            vec = model.encode(text, convert_to_numpy=True)
            embeddings[fid] = vec

        return embeddings

    def discover(self, method: str = "greedy", **kwargs) -> StageSequence:
        """Run discovery with specified method.

        Args:
            method: "greedy", "beam", "genetic", or "guided"
            **kwargs: Additional arguments for specific methods (e.g., router_path for guided)

        Returns:
            Best discovered StageSequence
        """
        if method == "greedy":
            return self.discover_greedy()
        elif method == "beam":
            return self.discover_beam()
        elif method == "genetic":
            return self.discover_genetic()
        elif method == "guided":
            return self.discover_guided(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")


def simulate_oracle(adapter: str, facts: List[dict]) -> EvalResult:
    """Simulated oracle for testing (random outcomes)."""
    n = len(facts)
    # Simulate: each adapter passes ~30-60% of facts
    pass_rate = random.uniform(0.3, 0.6)
    passed_mask = [random.random() < pass_rate for _ in range(n)]

    passed_ids = {
        f.get("id", f"fact_{i}")
        for i, f in enumerate(facts) if passed_mask[i]
    }

    margins = [random.uniform(-50, 50) if m else random.uniform(-100, -10)
               for m in passed_mask]

    return EvalResult(
        adapter=adapter,
        n_passed=len(passed_ids),
        n_total=n,
        margins=margins,
        passed_ids=passed_ids,
    )


# Quick test
if __name__ == "__main__":
    print("=== Stage Discovery Demo ===\n")

    # Create fake facts
    facts = [{"id": f"fact_{i:02d}", "truth": f"Test fact {i}"} for i in range(16)]

    # Create fake adapter list
    adapters = [
        "domain_cluster1",
        "domain_cluster2",
        "domain_cluster3",
        "domain_cluster4",
        "domain_general",
    ]

    # Test greedy
    print("--- Greedy Discovery ---")
    discoverer = StageDiscoverer(
        facts=facts,
        candidate_adapters=adapters,
        oracle_fn=simulate_oracle,
        config=DiscoveryConfig(max_stages=5),
    )
    result = discoverer.discover("greedy")
    print(f"\nResult: {result.adapters}")
    print(f"Score: {result.score:.1%}")

    # Test beam search
    print("\n--- Beam Search Discovery ---")
    discoverer = StageDiscoverer(
        facts=facts,
        candidate_adapters=adapters,
        oracle_fn=simulate_oracle,
        config=DiscoveryConfig(max_stages=5, beam_width=3),
    )
    result = discoverer.discover("beam")
    print(f"\nResult: {result.adapters}")
    print(f"Score: {result.score:.1%}")
