"""Self-Routing Steering Vectors — the vectors ARE the router.

Each steering vector encodes the "truth direction" for its domain.
A query that needs steering will have activations that anti-correlate
with that direction. No separate router needed.

Usage:
    from noethersolve.steering_router import SteeringRouter

    router = SteeringRouter.load("steering_bank.npz")
    # or
    router = SteeringRouter.build_from_facts(model, tokenizer, facts_dir)

    # At inference
    result = router.route_and_steer(model, tokenizer, query)
    if result:
        steering, domains = result
        # Apply steering to model's residual stream at layer 15
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SteeringResult:
    """Result of routing and steering computation."""
    steering_vector: np.ndarray
    domains_applied: list[str]
    domain_weights: dict[str, float]
    scores: dict[str, float]

    def __str__(self):
        applied = ", ".join(f"{d} ({w:.2f})" for d, w in self.domain_weights.items())
        return f"Steering: {applied}"


class SteeringRouter:
    """Domain-routed steering vector bank.

    Uses domain centroids (mean activation of facts) for semantic routing,
    and steering vectors (truth - falsehood) for correction.
    """

    def __init__(
        self,
        steering_bank: dict[str, np.ndarray],
        centroid_bank: dict[str, np.ndarray] | None = None,
        layer: int = 15,
        similarity_threshold: float = 0.15,  # Semantic similarity for routing
        top_k: int = 3,
    ):
        self.steering_bank = steering_bank
        self.layer = layer
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        # Domain centroids for semantic routing (if not provided, use steering vectors)
        self.centroid_bank = centroid_bank or steering_bank

        # Normalize centroids for cosine similarity routing
        self.normalized_centroids = {
            d: v / (np.linalg.norm(v) + 1e-8)
            for d, v in self.centroid_bank.items()
        }

        # Normalize steering vectors for consistent application
        self.normalized_steering = {
            d: v / (np.linalg.norm(v) + 1e-8)
            for d, v in steering_bank.items()
        }

    # Keep old name for backwards compatibility
    @property
    def threshold(self):
        return self.similarity_threshold

    @property
    def normalized_bank(self):
        return self.normalized_centroids

    @classmethod
    def load(cls, path: str, **kwargs) -> "SteeringRouter":
        """Load steering bank from .npz file."""
        data = dict(np.load(path))
        layer = int(data.pop("__layer__", 15))

        # Separate steering and centroid vectors
        steering_bank = {}
        centroid_bank = {}

        for key, value in data.items():
            if key.startswith("centroid_"):
                domain = key[9:]  # Remove "centroid_" prefix
                centroid_bank[domain] = value
            else:
                steering_bank[key] = value

        # If no centroids, use steering vectors (backwards compatibility)
        if not centroid_bank:
            centroid_bank = None

        return cls(steering_bank, centroid_bank=centroid_bank, layer=layer, **kwargs)

    def save(self, path: str):
        """Save steering bank to .npz file."""
        save_dict = {}

        # Save steering vectors
        for domain, vec in self.steering_bank.items():
            save_dict[domain] = vec

        # Save centroids with prefix
        if self.centroid_bank is not self.steering_bank:
            for domain, vec in self.centroid_bank.items():
                save_dict[f"centroid_{domain}"] = vec

        save_dict["__layer__"] = np.array(self.layer)
        np.savez_compressed(path, **save_dict)

    @classmethod
    def build_from_facts(
        cls,
        model,
        tokenizer,
        facts_dir: str,
        layer: int = 15,
        **kwargs,
    ) -> "SteeringRouter":
        """Build steering bank with centroids (routing) and steering vectors (correction)."""

        facts_path = Path(facts_dir)
        steering_bank = {}
        centroid_bank = {}

        for facts_file in sorted(facts_path.glob("*_facts*.json")):
            domain = facts_file.stem.replace("_facts", "").replace("_v2", "")

            try:
                with open(facts_file) as f:
                    data = json.load(f)
                facts = data.get("facts", data.get("verifications", []))

                if not facts:
                    continue

                # Compute steering vector AND centroid
                correct_acts = []
                incorrect_acts = []
                context_acts = []  # For centroid (domain representation)

                for fact in facts:
                    truth = fact.get("truth", fact.get("fact", ""))
                    distractors = fact.get("distractors", [])
                    context = fact.get("context", "")

                    if not distractors:
                        continue

                    base = f"{context}\n\nAnswer: " if context else "Answer: "

                    # Get activations
                    correct_act = _get_activation(model, tokenizer, base + truth, layer)
                    incorrect_act = _get_activation(model, tokenizer, base + distractors[0], layer)

                    # Get context activation (for semantic routing)
                    if context:
                        context_act = _get_activation(model, tokenizer, context, layer)
                        context_acts.append(context_act)

                    correct_acts.append(correct_act)
                    incorrect_acts.append(incorrect_act)

                if correct_acts:
                    correct_mean = np.mean(correct_acts, axis=0)
                    incorrect_mean = np.mean(incorrect_acts, axis=0)
                    steering_vector = correct_mean - incorrect_mean
                    steering_bank[domain] = steering_vector

                    # Centroid is mean of context activations (or correct if no contexts)
                    if context_acts:
                        centroid = np.mean(context_acts, axis=0)
                    else:
                        centroid = correct_mean  # Fallback to correct answers
                    centroid_bank[domain] = centroid

                    print(f"  {domain}: steer {np.linalg.norm(steering_vector):.2f}, centroid {np.linalg.norm(centroid):.2f}")

            except Exception as e:
                print(f"  {domain}: failed - {e}")
                continue

        print(f"\nBuilt steering bank with {len(steering_bank)} domains")
        return cls(steering_bank, centroid_bank=centroid_bank, layer=layer, **kwargs)

    def get_query_activation(self, model, tokenizer, query: str) -> np.ndarray:
        """Get activation for a query at the steering layer."""
        return _get_activation(model, tokenizer, query, self.layer)

    def score_domains(self, query_act: np.ndarray) -> dict[str, float]:
        """Score all domains for semantic similarity to query.

        Higher score = more relevant domain.
        """
        query_norm = query_act / (np.linalg.norm(query_act) + 1e-8)
        scores = {}
        for domain, centroid in self.normalized_centroids.items():
            scores[domain] = float(np.dot(query_norm, centroid))
        return scores

    def route_and_steer(
        self,
        model,
        tokenizer,
        query: str,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Optional[SteeringResult]:
        """Route query to relevant domains and compute blended steering.

        Returns None if no relevant domains found.
        """
        threshold = threshold if threshold is not None else self.similarity_threshold
        top_k = top_k or self.top_k

        # Get query activation
        query_act = self.get_query_activation(model, tokenizer, query)

        # Score all domains by semantic similarity
        scores = self.score_domains(query_act)

        # Select relevant domains (positive similarity above threshold)
        relevant = {d: s for d, s in scores.items() if s > threshold}

        if not relevant:
            return None  # No relevant domains

        # Take top-k by similarity
        sorted_domains = sorted(relevant.items(), key=lambda x: -x[1])[:top_k]
        relevant = dict(sorted_domains)

        # Normalize weights (similarity scores as weights)
        total = sum(relevant.values())
        alphas = {d: w / total for d, w in relevant.items()}

        # Blend steering vectors (additive, orthogonal → just works)
        steering = sum(
            alpha * self.steering_bank[domain]
            for domain, alpha in alphas.items()
        )

        return SteeringResult(
            steering_vector=steering,
            domains_applied=list(alphas.keys()),
            domain_weights=alphas,
            scores=scores,
        )

    def apply_steering(
        self,
        hidden_states,
        steering: np.ndarray,
        alpha: float = 1.0,
    ):
        """Apply steering vector to hidden states.

        hidden_states: [batch, seq, hidden] tensor
        steering: [hidden] numpy array

        Returns modified hidden states.
        """
        import mlx.core as mx

        steering_tensor = mx.array(steering.astype(np.float32) * alpha)
        steering_tensor = steering_tensor.reshape(1, 1, -1)

        return hidden_states + steering_tensor

    def __len__(self):
        return len(self.steering_bank)

    def __contains__(self, domain):
        return domain in self.steering_bank

    @property
    def domains(self) -> list[str]:
        return list(self.steering_bank.keys())

    @property
    def total_size_bytes(self) -> int:
        return sum(v.nbytes for v in self.steering_bank.values())

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / 1024 / 1024


def _get_activation(model, tokenizer, text: str, layer: int) -> np.ndarray:
    """Get activation at a specific layer for text."""
    import mlx.core as mx

    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    # Forward through layers
    hidden = model.model.embed_tokens(input_ids)

    for i, layer_module in enumerate(model.model.layers):
        hidden = layer_module(hidden, mask=None, cache=None)

        if i == layer:
            # Mean pool over sequence, convert to numpy
            act = hidden[0].mean(axis=0)  # [hidden]
            return np.array(act.astype(mx.float32))

    raise ValueError(f"Layer {layer} not found")


# Convenience function for quick testing
def quick_test(query: str, steering_bank_path: str = "steering_bank.npz"):
    """Quick test of steering router."""
    from mlx_lm import load

    print("Loading model...")
    model, tokenizer = load("Qwen/Qwen3-14B-Base")

    print("Loading steering bank...")
    router = SteeringRouter.load(steering_bank_path)
    print(f"  {len(router)} domains, {router.total_size_mb:.2f} MB")

    print(f"\nQuery: {query[:100]}...")
    result = router.route_and_steer(model, tokenizer, query)

    if result:
        print("\nSteering needed:")
        for d, w in result.domain_weights.items():
            print(f"  {d}: {w:.2%}")
    else:
        print("\nNo steering needed - query aligned with truth")

    return result
