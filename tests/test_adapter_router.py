"""Tests for the adapter router module."""

import json
import os
import tempfile

import numpy as np
import pytest

# Skip entire module if MLX is not available (Apple Silicon only)
pytest.importorskip("mlx", reason="MLX required (Apple Silicon only)")

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


# ── Auto-Rebuild ────────────────────────────────────────────────────


import time
from unittest.mock import MagicMock, patch


class TestNeedsRebuild:
    """Tests for AdapterRouter.needs_rebuild static method."""

    def test_returns_true_when_router_missing(self):
        """Returns True when router state file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            router_path = os.path.join(tmpdir, "router_state.npz")
            adapters_dir = os.path.join(tmpdir, "adapters")
            os.makedirs(adapters_dir)

            result = AdapterRouter.needs_rebuild(router_path, adapters_dir)
            assert result is True

    def test_returns_true_when_adapter_newer(self):
        """Returns True when adapter file is newer than router."""
        with tempfile.TemporaryDirectory() as tmpdir:
            router_path = os.path.join(tmpdir, "router_state.npz")
            adapters_dir = os.path.join(tmpdir, "adapters")
            os.makedirs(adapters_dir)

            # Create router file
            np.savez(router_path, dummy=np.array([1]))

            # Wait a bit and create newer adapter
            time.sleep(0.1)
            adapter_path = os.path.join(adapters_dir, "test_adapter.npz")
            np.savez(adapter_path, dummy=np.array([1]))

            result = AdapterRouter.needs_rebuild(router_path, adapters_dir)
            assert result is True

    def test_returns_true_when_facts_newer(self):
        """Returns True when facts file is newer than router."""
        with tempfile.TemporaryDirectory() as tmpdir:
            router_path = os.path.join(tmpdir, "router_state.npz")
            adapters_dir = os.path.join(tmpdir, "adapters")
            problems_dir = os.path.join(tmpdir, "problems")
            os.makedirs(adapters_dir)
            os.makedirs(problems_dir)

            # Create router file
            np.savez(router_path, dummy=np.array([1]))

            # Wait a bit and create newer facts file
            time.sleep(0.1)
            facts_path = os.path.join(problems_dir, "test_facts.json")
            with open(facts_path, "w") as f:
                json.dump({"facts": []}, f)

            result = AdapterRouter.needs_rebuild(router_path, adapters_dir, problems_dir)
            assert result is True

    def test_returns_false_when_router_current(self):
        """Returns False when router is newer than all dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            router_path = os.path.join(tmpdir, "router_state.npz")
            adapters_dir = os.path.join(tmpdir, "adapters")
            problems_dir = os.path.join(tmpdir, "problems")
            os.makedirs(adapters_dir)
            os.makedirs(problems_dir)

            # Create older adapter and facts files first
            adapter_path = os.path.join(adapters_dir, "test_adapter.npz")
            np.savez(adapter_path, dummy=np.array([1]))

            facts_path = os.path.join(problems_dir, "test_facts.json")
            with open(facts_path, "w") as f:
                json.dump({"facts": []}, f)

            # Wait and create newer router file
            time.sleep(0.1)
            np.savez(router_path, dummy=np.array([1]))

            result = AdapterRouter.needs_rebuild(router_path, adapters_dir, problems_dir)
            assert result is False

    def test_handles_empty_adapters_dir(self):
        """Works when adapters directory is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            router_path = os.path.join(tmpdir, "router_state.npz")
            adapters_dir = os.path.join(tmpdir, "adapters")
            os.makedirs(adapters_dir)

            # Create router file (empty adapters dir)
            np.savez(router_path, dummy=np.array([1]))

            result = AdapterRouter.needs_rebuild(router_path, adapters_dir)
            assert result is False

    def test_handles_nonexistent_adapters_dir(self):
        """Works when adapters directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            router_path = os.path.join(tmpdir, "router_state.npz")
            adapters_dir = os.path.join(tmpdir, "nonexistent_adapters")

            # Create router file
            np.savez(router_path, dummy=np.array([1]))

            result = AdapterRouter.needs_rebuild(router_path, adapters_dir)
            assert result is False

    def test_ignores_problems_dir_when_empty_string(self):
        """Doesn't check problems_dir when empty string passed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            router_path = os.path.join(tmpdir, "router_state.npz")
            adapters_dir = os.path.join(tmpdir, "adapters")
            os.makedirs(adapters_dir)

            np.savez(router_path, dummy=np.array([1]))

            # Empty string problems_dir should be ignored
            result = AdapterRouter.needs_rebuild(router_path, adapters_dir, "")
            assert result is False


class TestLoadOrRebuild:
    """Tests for AdapterRouter.load_or_rebuild class method."""

    def test_force_rebuild_calls_build(self):
        """force_rebuild=True triggers build even when file is current."""
        with tempfile.TemporaryDirectory() as tmpdir:
            router_path = os.path.join(tmpdir, "router_state.npz")
            adapters_dir = os.path.join(tmpdir, "adapters")
            problems_dir = os.path.join(tmpdir, "problems")
            os.makedirs(adapters_dir)
            os.makedirs(problems_dir)

            # Create a valid router state file
            np.savez(
                router_path,
                config=json.dumps({
                    "high_confidence": 0.85,
                    "ambiguity_gap": 0.05,
                    "fallback": 0.60,
                    "d_inner": 64,
                    "max_cache": 5,
                    "certainty_cascade": True,
                    "certainty_gap_threshold": 2,
                }),
                adapters_dir=adapters_dir,
                global_adapters=json.dumps({}),
            )

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            with patch.object(AdapterRouter, 'build') as mock_build:
                mock_router = AdapterRouter()
                mock_router.auto_register_global_adapters = MagicMock()
                mock_router.save = MagicMock()
                mock_build.return_value = mock_router

                AdapterRouter.load_or_rebuild(
                    router_path, mock_model, mock_tokenizer,
                    problems_dir, adapters_dir,
                    force_rebuild=True
                )

                mock_build.assert_called_once()

    def test_loads_when_router_current(self):
        """Loads existing router when no rebuild needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            router_path = os.path.join(tmpdir, "router_state.npz")
            adapters_dir = os.path.join(tmpdir, "adapters")
            problems_dir = os.path.join(tmpdir, "problems")
            os.makedirs(adapters_dir)
            os.makedirs(problems_dir)

            # Create valid router state file
            np.savez(
                router_path,
                config=json.dumps({
                    "high_confidence": 0.85,
                    "ambiguity_gap": 0.05,
                    "fallback": 0.60,
                    "d_inner": 64,
                    "max_cache": 5,
                    "certainty_cascade": True,
                    "certainty_gap_threshold": 2,
                }),
                adapters_dir=adapters_dir,
                global_adapters=json.dumps({}),
            )

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            with patch.object(AdapterRouter, 'build') as mock_build, \
                 patch.object(AdapterRouter, 'load') as mock_load, \
                 patch.object(AdapterRouter, 'needs_rebuild', return_value=False):

                mock_router = AdapterRouter()
                mock_router.auto_register_global_adapters = MagicMock()
                mock_load.return_value = mock_router

                AdapterRouter.load_or_rebuild(
                    router_path, mock_model, mock_tokenizer,
                    problems_dir, adapters_dir
                )

                mock_build.assert_not_called()
                mock_load.assert_called_once_with(router_path)

    def test_rebuilds_when_stale(self):
        """Rebuilds when needs_rebuild returns True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            router_path = os.path.join(tmpdir, "router_state.npz")
            adapters_dir = os.path.join(tmpdir, "adapters")
            problems_dir = os.path.join(tmpdir, "problems")
            os.makedirs(adapters_dir)
            os.makedirs(problems_dir)

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            with patch.object(AdapterRouter, 'build') as mock_build, \
                 patch.object(AdapterRouter, 'needs_rebuild', return_value=True):

                mock_router = AdapterRouter()
                mock_router.auto_register_global_adapters = MagicMock()
                mock_router.save = MagicMock()
                mock_build.return_value = mock_router

                AdapterRouter.load_or_rebuild(
                    router_path, mock_model, mock_tokenizer,
                    problems_dir, adapters_dir
                )

                mock_build.assert_called_once()
                mock_router.save.assert_called_once_with(router_path)
