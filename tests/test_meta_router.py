"""Tests for the meta-router module."""

import pytest
import tempfile
from pathlib import Path
import json

from noethersolve.meta_router import (
    MetaRouter,
    MetaRouterConfig,
    OutcomeRecord,
    FactEmbedder,
)


class TestFactEmbedder:
    """Test the fact embedding system."""

    def test_fit_builds_vocab(self):
        embedder = FactEmbedder(dim=32)
        texts = ["The quick brown fox", "jumps over the lazy dog"]
        embedder.fit(texts)

        assert embedder._fitted
        assert len(embedder.vocab) > 0

    def test_embed_returns_vector(self):
        embedder = FactEmbedder(dim=32)
        embedder.fit(["test text for embedding"])

        vec = embedder.embed("test text")
        assert vec.shape == (32,)

    def test_embed_is_normalized(self):
        embedder = FactEmbedder(dim=32)
        embedder.fit(["test text for embedding with more words"])

        vec = embedder.embed("test text with words")
        norm = (vec ** 2).sum() ** 0.5
        assert abs(norm - 1.0) < 1e-6 or norm == 0

    def test_similar_texts_have_similar_embeddings(self):
        embedder = FactEmbedder(dim=64)
        embedder.fit([
            "2D Navier-Stokes has global regularity",
            "3D Navier-Stokes regularity is open",
            "Trefoil knot has crossing number three",
        ])

        v1 = embedder.embed("2D NS has global regularity")
        v2 = embedder.embed("3D NS regularity open problem")
        v3 = embedder.embed("Trefoil knot crossing number")

        # NS embeddings should be more similar to each other than to knot
        sim_ns = (v1 * v2).sum()
        sim_cross = (v1 * v3).sum()
        assert sim_ns >= sim_cross  # NS texts more similar to each other


class TestMetaRouter:
    """Test the meta-router."""

    def test_add_outcomes(self):
        router = MetaRouter()
        router.add_outcome(OutcomeRecord(
            fact_id="test01",
            fact_text="Test fact",
            baseline_margin=-10.0,
            adapter="test_adapter",
            post_margin=5.0,
            flipped=True,
        ))

        assert len(router.outcomes) == 1

    def test_train_requires_outcomes(self):
        router = MetaRouter()
        with pytest.raises(RuntimeError, match="No outcomes"):
            router.train()

    def test_train_with_outcomes(self):
        router = MetaRouter(MetaRouterConfig(min_outcomes_per_adapter=2))

        # Add enough outcomes
        for i in range(5):
            router.add_outcome(OutcomeRecord(
                fact_id=f"fact_{i}",
                fact_text=f"Test fact number {i}",
                baseline_margin=-20.0,
                adapter="adapter_a",
                post_margin=10.0 if i % 2 == 0 else -5.0,
                flipped=i % 2 == 0,
            ))

        router.train()
        assert router._trained

    def test_predict_requires_training(self):
        router = MetaRouter()
        router.add_outcome(OutcomeRecord(
            fact_id="test",
            fact_text="Test",
            baseline_margin=-10.0,
            adapter="adapter",
            post_margin=5.0,
            flipped=True,
        ))

        with pytest.raises(RuntimeError, match="train"):
            router.predict("test fact", -10.0)

    def test_predict_returns_ranked_adapters(self):
        router = MetaRouter(MetaRouterConfig(min_outcomes_per_adapter=2))

        # Add outcomes for multiple adapters
        for adapter in ["adapter_a", "adapter_b"]:
            for i in range(3):
                router.add_outcome(OutcomeRecord(
                    fact_id=f"{adapter}_{i}",
                    fact_text=f"Fact for {adapter} number {i}",
                    baseline_margin=-30.0,
                    adapter=adapter,
                    post_margin=15.0 if i == 0 else -10.0,
                    flipped=i == 0,
                ))

        router.train()
        predictions = router.predict("Test fact text", -25.0, top_k=2)

        assert len(predictions) > 0
        assert all(isinstance(p, tuple) for p in predictions)
        assert all(len(p) == 2 for p in predictions)
        assert all(isinstance(p[0], str) and isinstance(p[1], float) for p in predictions)

    def test_predict_chain_returns_list(self):
        router = MetaRouter(MetaRouterConfig(min_outcomes_per_adapter=2))

        for adapter in ["adapter_a", "adapter_b", "adapter_c"]:
            for i in range(3):
                router.add_outcome(OutcomeRecord(
                    fact_id=f"{adapter}_{i}",
                    fact_text=f"Fact for {adapter} test {i}",
                    baseline_margin=-30.0,
                    adapter=adapter,
                    post_margin=10.0,
                    flipped=True,
                ))

        router.train()
        chain = router.predict_chain("Test fact", -30.0, chain_length=2)

        assert isinstance(chain, list)
        assert len(chain) <= 2

    def test_save_and_load(self):
        router = MetaRouter(MetaRouterConfig(min_outcomes_per_adapter=2))

        for i in range(5):
            router.add_outcome(OutcomeRecord(
                fact_id=f"fact_{i}",
                fact_text=f"Test fact number {i}",
                baseline_margin=-20.0,
                adapter="test_adapter",
                post_margin=10.0,
                flipped=True,
            ))

        router.train()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            router.save(path)
            loaded = MetaRouter.load(path)

            assert loaded._trained
            assert loaded.adapters == router.adapters
            assert loaded._weights.shape == router._weights.shape
        finally:
            path.unlink()

    def test_stats(self):
        router = MetaRouter()

        for i in range(3):
            router.add_outcome(OutcomeRecord(
                fact_id=f"fact_{i}",
                fact_text=f"Test {i}",
                baseline_margin=-10.0,
                adapter="adapter_a" if i < 2 else "adapter_b",
                post_margin=5.0 if i == 0 else -5.0,
                flipped=i == 0,
            ))

        stats = router.stats()
        assert stats["n_outcomes"] == 3
        assert stats["n_adapters"] == 2
        assert stats["n_facts"] == 3
        assert 0 <= stats["overall_flip_rate"] <= 1


class TestLoadOutcomes:
    """Test loading outcomes from file."""

    def test_load_jsonl(self):
        router = MetaRouter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({
                "fact_id": "test01",
                "fact_text": "Test fact one",
                "baseline_margin": -20.0,
                "adapter": "test_adapter",
                "post_margin": 10.0,
                "flipped": True,
            }) + "\n")
            f.write(json.dumps({
                "fact_id": "test02",
                "fact_text": "Test fact two",
                "baseline_margin": -15.0,
                "adapter": "test_adapter",
                "post_margin": -5.0,
                "flipped": False,
            }) + "\n")
            path = Path(f.name)

        try:
            count = router.load_outcomes(path)
            assert count == 2
            assert len(router.outcomes) == 2
        finally:
            path.unlink()
