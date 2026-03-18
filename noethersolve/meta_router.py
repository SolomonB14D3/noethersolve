"""Meta-Router — learns optimal adapter chains from outcome data.

Phase 1 of full automation: given a fact's embedding and baseline failure margin,
predict which adapter sequence will flip it.

Training data format (outcomes.jsonl):
    {"fact_id": "ns01", "fact_text": "2D NS is globally regular",
     "baseline_margin": -45.2, "adapter": "ns_conservation", "post_margin": 12.3, "flipped": true}

The meta-router learns:
    f(fact_embedding, baseline_margin) → ranked list of adapters

Usage:
    router = MetaRouter()
    router.load_outcomes("results/outcomes.jsonl")
    router.train()

    # At inference:
    adapters = router.predict(fact_embedding, baseline_margin=-45.2, top_k=3)
    # Returns: [("ns_conservation", 0.92), ("ns_blowup", 0.71), ("physics_general", 0.34)]
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import numpy as np
from collections import defaultdict


@dataclass
class OutcomeRecord:
    """Single fact × adapter outcome."""
    fact_id: str
    fact_text: str
    baseline_margin: float
    adapter: str
    post_margin: float
    flipped: bool
    cluster: str = ""
    domain: str = ""


@dataclass
class MetaRouterConfig:
    """Configuration for the meta-router."""
    embedding_dim: int = 64  # Dimension for fact embeddings
    hidden_dim: int = 128
    margin_buckets: int = 10  # Discretize baseline margin into buckets
    min_outcomes_per_adapter: int = 5  # Min training examples per adapter
    temperature: float = 1.0  # Softmax temperature for ranking


class FactEmbedder:
    """Simple bag-of-words embedder for facts."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.vocab: Dict[str, int] = {}
        self._fitted = False

    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        all_words = set()
        for text in texts:
            words = self._tokenize(text)
            all_words.update(words)

        # Assign indices to most common words
        self.vocab = {w: i % self.dim for i, w in enumerate(sorted(all_words))}
        self._fitted = True

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization."""
        import re
        return re.findall(r'\w+', text.lower())

    def embed(self, text: str) -> np.ndarray:
        """Embed a fact text to a vector."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        vec = np.zeros(self.dim)
        words = self._tokenize(text)
        for word in words:
            if word in self.vocab:
                vec[self.vocab[word]] += 1.0

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        return np.array([self.embed(t) for t in texts])


class MetaRouter:
    """Learns which adapters flip which facts."""

    def __init__(self, config: Optional[MetaRouterConfig] = None):
        self.config = config or MetaRouterConfig()
        self.embedder = FactEmbedder(dim=self.config.embedding_dim)
        self.outcomes: List[OutcomeRecord] = []
        self.adapters: List[str] = []
        self._adapter_to_idx: Dict[str, int] = {}

        # Learned parameters (simple logistic regression per adapter)
        self._weights: Optional[np.ndarray] = None  # (n_adapters, embedding_dim + 1)
        self._trained = False

    def load_outcomes(self, path: Path) -> int:
        """Load outcomes from JSONL file."""
        path = Path(path)
        count = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                self.outcomes.append(OutcomeRecord(
                    fact_id=data["fact_id"],
                    fact_text=data["fact_text"],
                    baseline_margin=data["baseline_margin"],
                    adapter=data["adapter"],
                    post_margin=data["post_margin"],
                    flipped=data["flipped"],
                    cluster=data.get("cluster", ""),
                    domain=data.get("domain", ""),
                ))
                count += 1
        return count

    def add_outcome(self, record: OutcomeRecord):
        """Add a single outcome (for online learning)."""
        self.outcomes.append(record)
        self._trained = False

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data: X (features), y (labels), adapter_idx."""
        # Fit embedder on all fact texts
        texts = [o.fact_text for o in self.outcomes]
        self.embedder.fit(texts)

        # Build adapter index
        self.adapters = sorted(set(o.adapter for o in self.outcomes))
        self._adapter_to_idx = {a: i for i, a in enumerate(self.adapters)}

        # Build feature matrix
        n = len(self.outcomes)
        X = np.zeros((n, self.config.embedding_dim + 1))  # +1 for margin
        y = np.zeros(n)
        adapter_idx = np.zeros(n, dtype=int)

        for i, o in enumerate(self.outcomes):
            X[i, :self.config.embedding_dim] = self.embedder.embed(o.fact_text)
            X[i, -1] = np.tanh(o.baseline_margin / 100.0)  # Normalize margin
            y[i] = 1.0 if o.flipped else 0.0
            adapter_idx[i] = self._adapter_to_idx[o.adapter]

        return X, y, adapter_idx

    def train(self, epochs: int = 100, lr: float = 0.1):
        """Train the router on loaded outcomes."""
        if len(self.outcomes) == 0:
            raise RuntimeError("No outcomes loaded")

        X, y, adapter_idx = self._prepare_data()
        n_adapters = len(self.adapters)
        feature_dim = X.shape[1]

        # Initialize weights
        self._weights = np.zeros((n_adapters, feature_dim))

        # Train per-adapter logistic regression
        for a_idx in range(n_adapters):
            mask = adapter_idx == a_idx
            if mask.sum() < self.config.min_outcomes_per_adapter:
                continue

            X_a = X[mask]
            y_a = y[mask]
            w = np.zeros(feature_dim)

            # Simple gradient descent
            for _ in range(epochs):
                logits = X_a @ w
                probs = 1.0 / (1.0 + np.exp(-logits))
                grad = X_a.T @ (probs - y_a) / len(y_a)
                w -= lr * grad

            self._weights[a_idx] = w

        self._trained = True

    def predict(
        self,
        fact_text: str,
        baseline_margin: float,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Predict best adapters for a fact.

        Returns:
            List of (adapter_name, probability) sorted by probability descending.
        """
        if not self._trained:
            raise RuntimeError("Call train() first")

        # Build feature vector
        x = np.zeros(self.config.embedding_dim + 1)
        x[:self.config.embedding_dim] = self.embedder.embed(fact_text)
        x[-1] = np.tanh(baseline_margin / 100.0)

        # Score all adapters
        logits = self._weights @ x
        probs = 1.0 / (1.0 + np.exp(-logits / self.config.temperature))

        # Rank
        ranked_idx = np.argsort(-probs)
        results = []
        for idx in ranked_idx[:top_k]:
            if probs[idx] > 0.01:  # Skip near-zero
                results.append((self.adapters[idx], float(probs[idx])))

        return results

    def predict_chain(
        self,
        fact_text: str,
        baseline_margin: float,
        chain_length: int = 3,
        diversity_penalty: float = 0.3
    ) -> List[str]:
        """Predict an ordered chain of adapters.

        Uses greedy selection with diversity penalty to avoid redundant adapters.
        """
        if not self._trained:
            raise RuntimeError("Call train() first")

        # Get all scores
        all_scores = self.predict(fact_text, baseline_margin, top_k=len(self.adapters))
        score_dict = {name: score for name, score in all_scores}

        chain = []
        used_clusters = set()

        for _ in range(chain_length):
            best_score = -1
            best_adapter = None

            for adapter, base_score in score_dict.items():
                if adapter in chain:
                    continue

                # Extract cluster from adapter name (e.g., "ns_conservation" → "ns")
                cluster = adapter.split("_")[0] if "_" in adapter else adapter

                # Penalize if we've already used this cluster
                penalty = diversity_penalty if cluster in used_clusters else 0.0
                adjusted_score = base_score - penalty

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_adapter = adapter

            if best_adapter is None:
                break

            chain.append(best_adapter)
            cluster = best_adapter.split("_")[0] if "_" in best_adapter else best_adapter
            used_clusters.add(cluster)

        return chain

    def save(self, path: Path):
        """Save router state."""
        path = Path(path)
        state = {
            "config": {
                "embedding_dim": self.config.embedding_dim,
                "hidden_dim": self.config.hidden_dim,
                "margin_buckets": self.config.margin_buckets,
                "min_outcomes_per_adapter": self.config.min_outcomes_per_adapter,
                "temperature": self.config.temperature,
            },
            "adapters": self.adapters,
            "vocab": self.embedder.vocab,
            "weights": self._weights.tolist() if self._weights is not None else None,
            "n_outcomes": len(self.outcomes),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MetaRouter":
        """Load router from saved state."""
        path = Path(path)
        with open(path) as f:
            state = json.load(f)

        config = MetaRouterConfig(**state["config"])
        router = cls(config)
        router.adapters = state["adapters"]
        router._adapter_to_idx = {a: i for i, a in enumerate(router.adapters)}
        router.embedder.vocab = state["vocab"]
        router.embedder._fitted = True

        if state["weights"] is not None:
            router._weights = np.array(state["weights"])
            router._trained = True

        return router

    def stats(self) -> Dict[str, Any]:
        """Return statistics about loaded outcomes."""
        if not self.outcomes:
            return {"n_outcomes": 0}

        by_adapter = defaultdict(lambda: {"total": 0, "flipped": 0})
        by_cluster = defaultdict(lambda: {"total": 0, "flipped": 0})

        for o in self.outcomes:
            by_adapter[o.adapter]["total"] += 1
            by_adapter[o.adapter]["flipped"] += int(o.flipped)
            if o.cluster:
                by_cluster[o.cluster]["total"] += 1
                by_cluster[o.cluster]["flipped"] += int(o.flipped)

        return {
            "n_outcomes": len(self.outcomes),
            "n_adapters": len(set(o.adapter for o in self.outcomes)),
            "n_facts": len(set(o.fact_id for o in self.outcomes)),
            "overall_flip_rate": sum(o.flipped for o in self.outcomes) / len(self.outcomes),
            "by_adapter": dict(by_adapter),
            "by_cluster": dict(by_cluster),
        }


def gather_outcomes_from_benchmarks(results_dir: Path) -> List[OutcomeRecord]:
    """Scan benchmark results and extract outcome records.

    Looks for JSON files with 'adapted_results' containing fact-level margins.
    """
    results_dir = Path(results_dir)
    outcomes = []

    for json_file in results_dir.glob("**/*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            if "adapted_results" not in data:
                continue

            adapter_path = data.get("adapter", "")
            adapter_name = Path(adapter_path).stem if adapter_path else "unknown"

            for result in data["adapted_results"]:
                outcomes.append(OutcomeRecord(
                    fact_id=result.get("id", f"fact_{result.get('idx', 0)}"),
                    fact_text=result.get("fact_text", ""),  # May need to look up
                    baseline_margin=result.get("baseline_margin", 0.0),
                    adapter=adapter_name,
                    post_margin=result.get("margin", 0.0),
                    flipped=result.get("win", False),
                    cluster=result.get("cluster", ""),
                ))
        except (json.JSONDecodeError, KeyError):
            continue

    return outcomes


# Quick test
if __name__ == "__main__":
    print("=== Meta-Router Demo ===\n")

    # Create synthetic outcomes for testing
    outcomes = [
        OutcomeRecord("ns01", "2D NS has global regularity", -45.2, "ns_conservation", 12.3, True, "ns"),
        OutcomeRecord("ns02", "3D NS regularity is open", -30.1, "ns_conservation", 8.5, True, "ns"),
        OutcomeRecord("ns03", "Blowup requires energy concentration", -60.0, "ns_blowup", 5.2, True, "ns"),
        OutcomeRecord("ns04", "Vortex stretching in 3D", -25.0, "ns_blowup", -15.0, False, "ns"),
        OutcomeRecord("knot01", "Trefoil has crossing number 3", -40.0, "knot_invariants", 20.0, True, "knot"),
        OutcomeRecord("knot02", "Figure-8 is amphichiral", -35.0, "knot_invariants", 18.0, True, "knot"),
        OutcomeRecord("chem01", "Rate law follows mass action", -50.0, "chemical_kinetics", -10.0, False, "chem"),
        OutcomeRecord("chem02", "Equilibrium constant is K", -20.0, "chemical_kinetics", 15.0, True, "chem"),
    ]

    router = MetaRouter()
    for o in outcomes:
        router.add_outcome(o)

    print(f"Stats: {router.stats()}")

    router.train()

    # Test prediction
    print("\n--- Predictions ---")

    test_cases = [
        ("2D Navier-Stokes has unique solutions", -42.0),
        ("Trefoil knot is not unknottable", -38.0),
        ("Chemical equilibrium obeys Le Chatelier", -25.0),
    ]

    for fact, margin in test_cases:
        preds = router.predict(fact, margin, top_k=3)
        chain = router.predict_chain(fact, margin, chain_length=2)
        print(f"\nFact: '{fact[:50]}...' (margin={margin})")
        print(f"  Top adapters: {preds}")
        print(f"  Suggested chain: {chain}")
