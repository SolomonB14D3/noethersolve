"""Tests for the adapter router module."""

import json
import os
import tempfile

import numpy as np
import pytest

from noethersolve.adapter_router import (
    AdapterRouter,
    RouteDecision,
    RouterConfig,
    _find_adapter_key,
)


# ── RouterConfig ─────────────────────────────────────────────────────


class TestRouterConfig:
    def test_defaults(self):
        cfg = RouterConfig()
        assert cfg.high_confidence == 0.85
        assert cfg.ambiguity_gap == 0.05
        assert cfg.fallback == 0.60
        assert cfg.d_inner == 64
        assert cfg.max_cache == 5

    def test_custom(self):
        cfg = RouterConfig(high_confidence=0.9, fallback=0.5)
        assert cfg.high_confidence == 0.9
        assert cfg.fallback == 0.5


# ── RouteDecision ────────────────────────────────────────────────────


class TestRouteDecision:
    def test_fallback(self):
        d = RouteDecision(level="fallback")
        assert d.level == "fallback"
        assert d.primary_key is None

    def test_single(self):
        d = RouteDecision(level="single", primary_key="ham_stage5", primary_sim=0.92)
        assert d.level == "single"
        assert d.primary_key == "ham_stage5"

    def test_ambiguous(self):
        d = RouteDecision(
            level="ambiguous",
            primary_key="a",
            primary_sim=0.82,
            secondary_key="b",
            secondary_sim=0.80,
        )
        assert d.level == "ambiguous"
        assert d.secondary_key == "b"


# ── Cascade Thresholds ───────────────────────────────────────────────


class TestCascadeLogic:
    """Test routing decisions with synthetic centroids."""

    def _make_router(self, centroids_dict):
        """Create a router with pre-set centroids (no model needed)."""
        router = AdapterRouter(RouterConfig())
        for key, vec in centroids_dict.items():
            vec = np.array(vec, dtype=np.float32)
            vec = vec / np.linalg.norm(vec)
            router.centroids[key] = vec
            router.adapter_paths[key] = f"adapters/{key}.npz"
        router._rebuild_matrix()
        return router

    def test_high_confidence_single(self):
        """Query very close to one centroid -> single."""
        router = self._make_router({
            "physics": [1.0, 0.0, 0.0],
            "chemistry": [0.0, 1.0, 0.0],
            "biology": [0.0, 0.0, 1.0],
        })
        # Simulate routing with a pre-computed query embedding
        query = np.array([0.99, 0.01, 0.0], dtype=np.float32)
        query = query / np.linalg.norm(query)

        sims = router._centroid_matrix @ query
        top2_idx = np.argsort(sims)[-2:][::-1]
        sim1 = float(sims[top2_idx[0]])

        assert sim1 > 0.85  # high confidence
        assert router._centroid_keys[top2_idx[0]] == "physics"

    def test_fallback_low_similarity(self):
        """Query far from all centroids -> fallback."""
        router = self._make_router({
            "physics": [1.0, 0.0, 0.0],
            "chemistry": [0.0, 1.0, 0.0],
        })
        # Query orthogonal to both
        query = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        query = query / np.linalg.norm(query)

        sims = router._centroid_matrix @ query
        sim1 = float(sims.max())

        assert sim1 < 0.60  # below fallback threshold

    def test_ambiguous_close_centroids(self):
        """Query equidistant between two centroids -> ambiguous."""
        router = self._make_router({
            "physics": [1.0, 0.1, 0.0],
            "chemistry": [1.0, -0.1, 0.0],
        })
        # Query between both
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query = query / np.linalg.norm(query)

        sims = router._centroid_matrix @ query
        top2_idx = np.argsort(sims)[-2:][::-1]
        sim1 = float(sims[top2_idx[0]])
        sim2 = float(sims[top2_idx[1]])
        gap = sim1 - sim2

        assert gap < 0.05  # ambiguous
        assert sim1 >= 0.60  # above fallback

    def test_empty_router_fallback(self):
        """Empty router always returns fallback."""
        router = AdapterRouter()
        # Can't call route() without model, but check matrix is None
        assert router._centroid_matrix is None


# ── Adapter Key Finding ──────────────────────────────────────────────


class TestFindAdapterKey:
    def test_exact_cluster_match(self):
        available = {"aging_biology_mechanisms_adapter": "path1"}
        key = _find_adapter_key("aging_biology", "mechanisms", available)
        assert key == "aging_biology_mechanisms_adapter"

    def test_domain_only_match(self):
        available = {"chemical_adapter": "path1"}
        key = _find_adapter_key("chemical", "all", available)
        assert key == "chemical_adapter"

    def test_stage5_match(self):
        available = {"hamiltonian_stage5": "path1"}
        key = _find_adapter_key("hamiltonian", "all", available)
        assert key == "hamiltonian_stage5"

    def test_no_match(self):
        available = {"unrelated_adapter": "path1"}
        key = _find_adapter_key("physics", "particles", available)
        assert key is None

    def test_wildcard_domain_prefix(self):
        available = {
            "ns_regularity_prior_broken": "path1",
            "ns_regularity_blowup_adapter": "path2",
        }
        key = _find_adapter_key("ns_regularity", "all", available)
        assert key is not None  # should find one of them


# ── Save / Load Round Trip ───────────────────────────────────────────


class TestPersistence:
    def test_save_load_roundtrip(self):
        router = AdapterRouter(RouterConfig(high_confidence=0.9))

        # Add some centroids
        router.centroids["physics"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        router.centroids["chemistry"] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        router.adapter_paths = {
            "physics": "adapters/physics.npz",
            "chemistry": "adapters/chemistry.npz",
        }
        router._rebuild_matrix()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            router.save(path)
            loaded = AdapterRouter.load(path)

            assert len(loaded.centroids) == 2
            assert loaded.config.high_confidence == 0.9
            assert "physics" in loaded.centroids
            assert "chemistry" in loaded.centroids
            np.testing.assert_allclose(
                loaded.centroids["physics"],
                router.centroids["physics"],
                atol=1e-6,
            )
            assert loaded.adapter_paths["physics"] == "adapters/physics.npz"
            assert loaded._centroid_matrix is not None
            assert loaded._centroid_matrix.shape == (2, 3)
        finally:
            os.unlink(path)

    def test_empty_save_warns(self, capsys):
        router = AdapterRouter()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            router.save(path)
            captured = capsys.readouterr()
            assert "Warning" in captured.out or "no centroids" in captured.out
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ── Registration ─────────────────────────────────────────────────────


class TestRegistration:
    def test_register_new_adapter(self):
        router = AdapterRouter()
        centroid = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        router.register_adapter("new_domain", "adapters/new.npz", centroid)

        assert "new_domain" in router.centroids
        assert router.adapter_paths["new_domain"] == "adapters/new.npz"
        # Check L2 normalized
        norm = np.linalg.norm(router.centroids["new_domain"])
        assert abs(norm - 1.0) < 1e-6
        # Matrix rebuilt
        assert router._centroid_matrix is not None
        assert router._centroid_matrix.shape[0] == 1

    def test_register_updates_matrix(self):
        router = AdapterRouter()
        router.register_adapter("a", "a.npz", np.array([1, 0, 0], dtype=np.float32))
        assert router._centroid_matrix.shape[0] == 1
        router.register_adapter("b", "b.npz", np.array([0, 1, 0], dtype=np.float32))
        assert router._centroid_matrix.shape[0] == 2


# ── Info ─────────────────────────────────────────────────────────────


class TestInfo:
    def test_info_output(self):
        router = AdapterRouter()
        router.centroids["physics_adapter"] = np.array([1, 0], dtype=np.float32)
        router.centroids["chemistry_adapter"] = np.array([0, 1], dtype=np.float32)

        info = router.info()
        assert "2 centroids" in info
        assert "physics" in info
        assert "chemistry" in info
