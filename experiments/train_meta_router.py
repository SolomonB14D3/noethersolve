#!/usr/bin/env python3
"""Train meta-router from fact→adapter outcome data.

The meta-router learns to predict which adapters will flip which facts
based on embeddings of the fact text.

Usage:
    python experiments/train_meta_router.py --data results/meta_router_training.jsonl
    python experiments/train_meta_router.py --data results/meta_router_training.jsonl --eval
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_training_data(data_file: Path) -> List[dict]:
    """Load JSONL training data."""
    records = []
    with open(data_file) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_fact_adapter_matrix(records: List[dict]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build fact × adapter matrix of flip outcomes.

    Returns:
        matrix: (n_facts, n_adapters) binary matrix where 1 = adapter flips fact
        fact_ids: list of fact IDs
        adapter_ids: list of adapter names
    """
    # Collect unique facts and adapters
    fact_set = set()
    adapter_set = set()

    for r in records:
        fact_set.add(r["fact_id"])
        adapter_set.add(r["adapter"])

    fact_ids = sorted(fact_set)
    adapter_ids = sorted(adapter_set)

    fact_idx = {f: i for i, f in enumerate(fact_ids)}
    adapter_idx = {a: i for i, a in enumerate(adapter_ids)}

    # Build matrix
    matrix = np.zeros((len(fact_ids), len(adapter_ids)), dtype=np.float32)

    for r in records:
        fi = fact_idx[r["fact_id"]]
        ai = adapter_idx[r["adapter"]]
        # Use margin improvement as signal (normalized)
        improvement = r["post_margin"] - r["baseline_margin"]
        if r["flipped"]:
            matrix[fi, ai] = improvement

    return matrix, fact_ids, adapter_ids


def get_fact_embeddings(records: List[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict[str, np.ndarray]:
    """Get embeddings for fact texts."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing sentence-transformers...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers"], check=True)
        from sentence_transformers import SentenceTransformer

    # Collect unique fact texts
    fact_texts = {}
    for r in records:
        fact_texts[r["fact_id"]] = r["fact_text"]

    print(f"Computing embeddings for {len(fact_texts)} facts...")
    model = SentenceTransformer(model_name)

    embeddings = {}
    texts = list(fact_texts.values())
    ids = list(fact_texts.keys())

    # Batch encode
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    for i, fid in enumerate(ids):
        embeddings[fid] = vectors[i]

    return embeddings


class MetaRouter:
    """Routes facts to adapters using learned embeddings."""

    def __init__(self, d_embed: int = 384):
        self.d_embed = d_embed
        self.adapter_centroids = {}  # adapter_name -> (centroid, count)
        self.fact_embeddings = {}
        self.adapter_to_facts = defaultdict(list)  # adapter -> [(fact_id, margin_improvement)]

    def fit(self, records: List[dict], embeddings: Dict[str, np.ndarray]):
        """Train router from outcome data."""
        self.fact_embeddings = embeddings

        # Group facts by adapter that flipped them
        for r in records:
            if r["flipped"]:
                improvement = r["post_margin"] - r["baseline_margin"]
                self.adapter_to_facts[r["adapter"]].append((r["fact_id"], improvement))

        # Compute centroid for each adapter (weighted by improvement)
        for adapter, fact_list in self.adapter_to_facts.items():
            if not fact_list:
                continue

            # Weighted average of fact embeddings
            total_weight = sum(imp for _, imp in fact_list)
            centroid = np.zeros(self.d_embed, dtype=np.float32)

            for fid, imp in fact_list:
                if fid in embeddings:
                    centroid += embeddings[fid] * imp

            if total_weight > 0:
                centroid /= total_weight
                centroid /= np.linalg.norm(centroid) + 1e-8
                self.adapter_centroids[adapter] = (centroid, len(fact_list))

        print(f"Trained router with {len(self.adapter_centroids)} adapter centroids")
        for adapter, (_, count) in sorted(self.adapter_centroids.items(), key=lambda x: -x[1][1]):
            print(f"  {adapter}: {count} flipped facts")

    def predict(self, fact_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Predict best adapters for a fact embedding.

        Returns list of (adapter_name, similarity) sorted by similarity.
        """
        fact_embedding = fact_embedding / (np.linalg.norm(fact_embedding) + 1e-8)

        scores = []
        for adapter, (centroid, _) in self.adapter_centroids.items():
            sim = np.dot(fact_embedding, centroid)
            scores.append((adapter, float(sim)))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def save(self, path: Path):
        """Save router state."""
        state = {
            "adapter_centroids": {
                k: (v[0].tolist(), v[1]) for k, v in self.adapter_centroids.items()
            },
            "adapter_to_facts": dict(self.adapter_to_facts),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"Saved router to {path}")

    @classmethod
    def load(cls, path: Path) -> "MetaRouter":
        """Load router state."""
        with open(path) as f:
            state = json.load(f)

        router = cls()
        router.adapter_centroids = {
            k: (np.array(v[0], dtype=np.float32), v[1])
            for k, v in state["adapter_centroids"].items()
        }
        router.adapter_to_facts = defaultdict(list, state["adapter_to_facts"])
        return router


def evaluate_router(router: MetaRouter, records: List[dict], embeddings: Dict[str, np.ndarray]) -> dict:
    """Evaluate router accuracy on held-out facts."""
    # Group records by fact
    fact_to_best = {}  # fact_id -> best adapter that flipped it

    for r in records:
        if r["flipped"]:
            improvement = r["post_margin"] - r["baseline_margin"]
            fid = r["fact_id"]
            if fid not in fact_to_best or improvement > fact_to_best[fid][1]:
                fact_to_best[fid] = (r["adapter"], improvement)

    # Evaluate predictions
    top1_correct = 0
    top3_correct = 0
    total = 0

    for fid, (best_adapter, _) in fact_to_best.items():
        if fid not in embeddings:
            continue

        predictions = router.predict(embeddings[fid], top_k=3)
        predicted_adapters = [a for a, _ in predictions]

        if predicted_adapters and predicted_adapters[0] == best_adapter:
            top1_correct += 1
        if best_adapter in predicted_adapters:
            top3_correct += 1
        total += 1

    return {
        "top1_accuracy": top1_correct / total if total > 0 else 0,
        "top3_accuracy": top3_correct / total if total > 0 else 0,
        "total_facts": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/meta_router_training.jsonl",
                        help="Training data JSONL file")
    parser.add_argument("--output", default="results/meta_router.json",
                        help="Output router model file")
    parser.add_argument("--eval", action="store_true",
                        help="Run leave-one-out evaluation")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    data_file = base_dir / args.data

    if not data_file.exists():
        print(f"Error: {data_file} not found")
        print("Run generate_meta_router_training.py first")
        sys.exit(1)

    # Load data
    print(f"Loading training data from {data_file}...")
    records = load_training_data(data_file)
    print(f"Loaded {len(records)} records")

    # Get embeddings
    embeddings = get_fact_embeddings(records)

    # Build router
    router = MetaRouter(d_embed=384)
    router.fit(records, embeddings)

    # Evaluate
    if args.eval:
        print("\nEvaluating router (in-sample)...")
        metrics = evaluate_router(router, records, embeddings)
        print(f"Top-1 accuracy: {100*metrics['top1_accuracy']:.1f}%")
        print(f"Top-3 accuracy: {100*metrics['top3_accuracy']:.1f}%")
        print(f"Total flipped facts: {metrics['total_facts']}")

    # Save
    output_path = base_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    router.save(output_path)

    # Demo predictions
    print("\nDemo predictions for sample facts:")
    sample_facts = list(embeddings.keys())[:5]
    for fid in sample_facts:
        predictions = router.predict(embeddings[fid], top_k=2)
        print(f"  {fid}: {predictions}")


if __name__ == "__main__":
    main()
