"""Embedding-based adapter router with cascade confidence thresholds.

Routes incoming queries to the best specialist adapter using cosine similarity
to precomputed domain centroids. Three-level cascade:

  HIGH confidence  → single adapter (fast path, ~95% of queries)
  AMBIGUOUS        → try top-2 adapters, pick higher margin (slow path)
  FALLBACK         → no adapter, vanilla base model (safety)

The router embeds queries using the base model's own hidden states (mean-pooled
last layer), requiring no additional embedding model.

Usage:
    # One-time build (embeds all facts, computes centroids, ~40s)
    router = AdapterRouter.build(model, tokenizer, problems_dir, adapters_dir)
    router.save("router_state.npz")

    # Load persisted router (instant)
    router = AdapterRouter.load("router_state.npz")

    # Route a query
    decision = router.route(model, tokenizer, "Hamiltonian of coupled oscillators:")
    # -> RouteDecision(level="single", primary_key="hamiltonian_stage5", sim=0.92)

    # Score with routing (wraps oracle.score_fact_mc)
    win, margin, truth_lp, best_dist_lp = router.score_fact_routed(
        model, tokenizer, lm_head, context, truth, distractors)
"""

import json
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np


@dataclass
class RouterConfig:
    """Thresholds for the cascade routing decision."""

    high_confidence: float = 0.85  # cosine sim above → single adapter
    ambiguity_gap: float = 0.05    # top-2 gap below → try both
    fallback: float = 0.60         # below → no adapter
    d_inner: int = 64              # adapter inner dim
    max_cache: int = 5             # LRU cache size (0 = unlimited, preload all)

    # Certainty cascade: try global adapters when baseline fails on high-gap facts
    certainty_cascade: bool = True  # Enable certainty-aware cascade
    certainty_gap_threshold: int = 2  # Min certainty gap to trigger cascade


@dataclass
class RouteDecision:
    """Result of routing: which adapter(s) to use."""

    level: str  # "single", "ambiguous", "fallback"
    primary_key: Optional[str] = None
    primary_sim: float = 0.0
    secondary_key: Optional[str] = None
    secondary_sim: float = 0.0


class AdapterRouter:
    """Embedding-based router mapping queries to specialist adapters."""

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        self.centroids: dict[str, np.ndarray] = {}    # key -> mean embedding
        self.adapter_paths: dict[str, str] = {}        # key -> relative path to .npz
        self._adapters_dir: str = ""                   # base dir for adapter paths
        self._loaded_adapters: OrderedDict = OrderedDict()  # LRU cache
        self._centroid_matrix: Optional[np.ndarray] = None  # precomputed for fast routing
        self._centroid_keys: list[str] = []

        # Global adapters: always tried as fallback regardless of domain routing
        # These are cross-domain adapters like certainty decontamination
        self.global_adapters: dict[str, str] = {}  # key -> path

    # ── Embedding ────────────────────────────────────────────────────

    @staticmethod
    def embed_text(model, tokenizer, text: str) -> np.ndarray:
        """Mean-pool last hidden state as embedding vector.

        Uses model.model(tokens) — the transformer backbone without lm_head.
        Returns a 1-D numpy array of shape (d_model,).
        """
        tokens = mx.array(tokenizer.encode(text))[None, :]
        h = model.model(tokens)  # [1, seq, d_model]
        mx.eval(h)
        # Mean pool across sequence dimension
        emb = np.array(h[0].astype(mx.float32)).mean(axis=0)
        # L2 normalize for cosine similarity
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    # ── Building ─────────────────────────────────────────────────────

    @classmethod
    def build(cls, model, tokenizer, problems_dir: str, adapters_dir: str,
              config: Optional[RouterConfig] = None) -> "AdapterRouter":
        """Build router from fact files and available adapters.

        Scans all *_facts.json files, groups facts by (domain, cluster),
        embeds their contexts, computes centroids, and maps to adapter files.
        """
        router = cls(config)
        router._adapters_dir = str(adapters_dir)
        problems_path = Path(problems_dir)
        adapters_path = Path(adapters_dir)

        # Collect all available adapter filenames (without .npz)
        available_adapters = {}
        for f in adapters_path.glob("*.npz"):
            available_adapters[f.stem] = str(f)

        # Process each fact file
        total_centroids = 0
        for facts_file in sorted(problems_path.glob("*_facts.json")):
            domain = facts_file.stem.replace("_facts", "")

            try:
                with open(facts_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            facts = data.get("facts", data) if isinstance(data, dict) else data
            if not isinstance(facts, list) or len(facts) == 0:
                continue

            # Group facts by cluster
            clusters: dict[str, list] = {}
            has_cluster = "cluster" in facts[0]

            if has_cluster:
                for fact in facts:
                    cluster = fact.get("cluster", "default")
                    clusters.setdefault(cluster, []).append(fact)
            else:
                # No cluster field — use domain as single cluster
                clusters["all"] = facts

            # For each cluster, compute centroid and find matching adapter
            for cluster_name, cluster_facts in clusters.items():
                # Find matching adapter
                adapter_key = _find_adapter_key(
                    domain, cluster_name, available_adapters
                )
                if adapter_key is None:
                    continue

                # Embed all facts in this cluster
                embeddings = []
                for fact in cluster_facts:
                    context = fact.get("context", "")
                    if not context:
                        continue
                    emb = cls.embed_text(model, tokenizer, f"{context}:")
                    embeddings.append(emb)

                if not embeddings:
                    continue

                # Compute centroid
                centroid = np.mean(embeddings, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm

                router.centroids[adapter_key] = centroid
                router.adapter_paths[adapter_key] = available_adapters[adapter_key]
                total_centroids += 1

                if total_centroids % 10 == 0:
                    print(f"  Built {total_centroids} centroids...")

        router._rebuild_matrix()
        print(f"  Router built: {total_centroids} centroids across "
              f"{len(set(k.rsplit('_', 1)[0] for k in router.centroids))} domains")
        return router

    def _rebuild_matrix(self):
        """Stack centroids into a matrix for fast batch cosine similarity."""
        if not self.centroids:
            self._centroid_matrix = None
            self._centroid_keys = []
            return
        self._centroid_keys = list(self.centroids.keys())
        self._centroid_matrix = np.stack(
            [self.centroids[k] for k in self._centroid_keys]
        )  # [n_centroids, d_model]

    # ── Routing ──────────────────────────────────────────────────────

    def route(self, model, tokenizer, context: str) -> RouteDecision:
        """Route a query to the best adapter(s).

        Returns a RouteDecision with the cascade level and adapter key(s).
        """
        if self._centroid_matrix is None or len(self._centroid_keys) == 0:
            return RouteDecision(level="fallback")

        query_emb = self.embed_text(model, tokenizer, f"{context}:")

        # Cosine similarities (embeddings are already L2-normalized)
        sims = self._centroid_matrix @ query_emb  # [n_centroids]

        # Top-2
        top2_idx = np.argsort(sims)[-2:][::-1]
        sim1 = float(sims[top2_idx[0]])
        key1 = self._centroid_keys[top2_idx[0]]

        sim2 = float(sims[top2_idx[1]]) if len(top2_idx) > 1 else 0.0
        key2 = self._centroid_keys[top2_idx[1]] if len(top2_idx) > 1 else None

        gap = sim1 - sim2

        # Cascade decision
        if sim1 < self.config.fallback:
            return RouteDecision(level="fallback", primary_sim=sim1)

        if sim1 >= self.config.high_confidence or gap >= self.config.ambiguity_gap:
            return RouteDecision(
                level="single",
                primary_key=key1,
                primary_sim=sim1,
            )

        # Ambiguous: top-2 are close
        return RouteDecision(
            level="ambiguous",
            primary_key=key1,
            primary_sim=sim1,
            secondary_key=key2,
            secondary_sim=sim2,
        )

    # ── Adapter Loading (LRU cache) ─────────────────────────────────

    def get_adapter(self, key: str, vocab_size: int):
        """Load adapter for a routing key, with LRU caching."""
        if key in self._loaded_adapters:
            self._loaded_adapters.move_to_end(key)
            return self._loaded_adapters[key]

        path = self.adapter_paths.get(key)
        if path is None:
            return None

        from noethersolve.adapter import SnapOnConfig, SnapOnLogitMLP

        config = SnapOnConfig(
            d_inner=self.config.d_inner,
            vocab_size=vocab_size,
            mode="logit",
        )
        adapter = SnapOnLogitMLP(config)
        mx.eval(adapter.parameters())

        data = dict(np.load(path))
        params = {}
        for k, v in data.items():
            parts = k.split(".")
            d = params
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = mx.array(v)

        flat = []
        _flatten_dict(params, "", flat)
        adapter.load_weights(flat)
        mx.eval(adapter.parameters())

        # LRU eviction (0 = unlimited)
        if self.config.max_cache > 0 and len(self._loaded_adapters) >= self.config.max_cache:
            self._loaded_adapters.popitem(last=False)
        self._loaded_adapters[key] = adapter
        return adapter

    # ── Routed Scoring ───────────────────────────────────────────────

    def score_fact_routed(self, model, tokenizer, lm_head,
                          context: str, truth: str, distractors: list):
        """Score a fact using the routed adapter. Wraps oracle.score_fact_mc.

        Returns: (win, margin, truth_lp, best_dist_lp, route_decision)
        """
        from noethersolve.oracle import score_fact_mc

        decision = self.route(model, tokenizer, context)
        vocab_size = model.model.embed_tokens.weight.shape[0]

        if decision.level == "fallback":
            result = score_fact_mc(model, tokenizer, context, truth, distractors)
            return (*result, decision)

        if decision.level == "single":
            adapter = self.get_adapter(decision.primary_key, vocab_size)
            if adapter is None:
                result = score_fact_mc(model, tokenizer, context, truth, distractors)
                return (*result, decision)
            result = score_fact_mc(
                model, tokenizer, context, truth, distractors,
                adapter=adapter, lm_head=lm_head,
            )
            return (*result, decision)

        # Ambiguous: try both, pick higher margin
        adapter1 = self.get_adapter(decision.primary_key, vocab_size)
        adapter2 = self.get_adapter(decision.secondary_key, vocab_size)

        if adapter1 is not None:
            r1 = score_fact_mc(
                model, tokenizer, context, truth, distractors,
                adapter=adapter1, lm_head=lm_head,
            )
        else:
            r1 = score_fact_mc(model, tokenizer, context, truth, distractors)

        if adapter2 is not None:
            r2 = score_fact_mc(
                model, tokenizer, context, truth, distractors,
                adapter=adapter2, lm_head=lm_head,
            )
        else:
            r2 = r1  # only one adapter available

        # Pick by margin
        if r1[1] >= r2[1]:
            return (*r1, decision)
        else:
            # Update decision to reflect which adapter won
            decision.primary_key = decision.secondary_key
            decision.primary_sim = decision.secondary_sim
            return (*r2, decision)

    # ── Certainty Cascade Routing ────────────────────────────────────

    def score_fact_cascade(self, model, tokenizer, lm_head,
                           context: str, truth: str, distractors: list,
                           fact_id: str = ""):
        """Score with certainty-aware cascade: baseline first, adapter fallback on failure.

        Strategy:
        1. Try baseline (no adapter)
        2. If baseline PASSES → return baseline result (no change)
        3. If baseline FAILS and certainty_gap >= threshold:
           a. Try domain adapter (from routing)
           b. Try global certainty adapters (cert_decon, anti_def)
           c. Return whichever has highest margin
        4. If baseline FAILS and low certainty_gap:
           - Just try domain adapter, return best

        This ensures zero regressions: adapters only applied when baseline already fails.

        Returns: (win, margin, truth_lp, best_dist_lp, route_decision, cascade_used)
        """
        from noethersolve.oracle import score_fact_mc
        from noethersolve.audit_facts import CERTAINTY_MARKERS, _count_markers

        # Step 1: Baseline (no adapter)
        baseline = score_fact_mc(model, tokenizer, context, truth, distractors)
        baseline_win, baseline_margin, _, _ = baseline

        if baseline_win:
            # Baseline passes → no adapter needed, zero regression risk
            decision = RouteDecision(level="baseline_pass")
            return (*baseline, decision, None)

        # Step 2: Compute certainty gap
        truth_certainty = _count_markers(truth, CERTAINTY_MARKERS)
        max_dist_certainty = max(_count_markers(d, CERTAINTY_MARKERS) for d in distractors)
        certainty_gap = max_dist_certainty - truth_certainty

        # Step 3: Get domain adapter from routing
        decision = self.route(model, tokenizer, context)
        vocab_size = model.model.embed_tokens.weight.shape[0]
        best_result = baseline
        best_margin = baseline_margin
        cascade_used = None

        # Try domain adapter
        if decision.level != "fallback" and decision.primary_key:
            adapter = self.get_adapter(decision.primary_key, vocab_size)
            if adapter is not None:
                domain_result = score_fact_mc(
                    model, tokenizer, context, truth, distractors,
                    adapter=adapter, lm_head=lm_head,
                )
                if domain_result[1] > best_margin:
                    best_result = domain_result
                    best_margin = domain_result[1]
                    cascade_used = decision.primary_key

        # Step 4: If high certainty gap, also try global certainty adapters
        if (self.config.certainty_cascade and
            certainty_gap >= self.config.certainty_gap_threshold and
            self.global_adapters):

            for key, path in self.global_adapters.items():
                adapter = self._load_global_adapter(key, path, vocab_size)
                if adapter is None:
                    continue

                cert_result = score_fact_mc(
                    model, tokenizer, context, truth, distractors,
                    adapter=adapter, lm_head=lm_head,
                )
                if cert_result[1] > best_margin:
                    best_result = cert_result
                    best_margin = cert_result[1]
                    cascade_used = key

        return (*best_result, decision, cascade_used)

    def _load_global_adapter(self, key: str, path: str, vocab_size: int):
        """Load a global adapter, with caching."""
        cache_key = f"global_{key}"
        if cache_key in self._loaded_adapters:
            self._loaded_adapters.move_to_end(cache_key)
            return self._loaded_adapters[cache_key]

        if not os.path.exists(path):
            return None

        from noethersolve.adapter import SnapOnConfig, SnapOnLogitMLP

        config = SnapOnConfig(
            d_inner=self.config.d_inner,
            vocab_size=vocab_size,
            mode="logit",
        )
        adapter = SnapOnLogitMLP(config)
        mx.eval(adapter.parameters())

        data = dict(np.load(path))
        params = {}
        for k, v in data.items():
            parts = k.split(".")
            d = params
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = mx.array(v)

        flat = []
        _flatten_dict(params, "", flat)
        adapter.load_weights(flat)
        mx.eval(adapter.parameters())

        # LRU eviction
        if self.config.max_cache > 0 and len(self._loaded_adapters) >= self.config.max_cache:
            self._loaded_adapters.popitem(last=False)
        self._loaded_adapters[cache_key] = adapter
        return adapter

    def register_global_adapter(self, key: str, adapter_path: str):
        """Register a global adapter (cross-domain, always tried as fallback)."""
        self.global_adapters[key] = adapter_path

    def auto_register_global_adapters(self, adapters_dir: str):
        """Auto-discover and register global adapters.

        Global adapters are identified by naming convention:
        - certainty_decontamination_adapter.npz
        - anti_definitive_adapter.npz
        - *_global_adapter.npz
        """
        adapters_path = Path(adapters_dir)
        global_patterns = [
            "certainty_decontamination_adapter.npz",
            "anti_definitive_adapter.npz",
        ]

        # Exact matches
        for pattern in global_patterns:
            path = adapters_path / pattern
            if path.exists():
                key = pattern.replace("_adapter.npz", "").replace(".npz", "")
                self.global_adapters[key] = str(path)

        # Pattern: *_global_adapter.npz
        for path in adapters_path.glob("*_global_adapter.npz"):
            key = path.stem.replace("_global_adapter", "")
            self.global_adapters[key] = str(path)

        if self.global_adapters:
            print(f"  Auto-registered {len(self.global_adapters)} global adapters: "
                  f"{list(self.global_adapters.keys())}")

    # ── Registration ─────────────────────────────────────────────────

    def register_adapter(self, key: str, adapter_path: str,
                         centroid: np.ndarray):
        """Register a new adapter trained during this session."""
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        self.centroids[key] = centroid
        self.adapter_paths[key] = adapter_path
        self._rebuild_matrix()

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str):
        """Save router state to .npz file."""
        if not self.centroids:
            print("  Warning: no centroids to save")
            return

        keys = list(self.centroids.keys())
        centroid_matrix = np.stack([self.centroids[k] for k in keys])

        np.savez(
            path,
            config_json=json.dumps(asdict(self.config)),
            keys_json=json.dumps(keys),
            adapter_paths_json=json.dumps(self.adapter_paths),
            centroid_matrix=centroid_matrix,
            global_adapters_json=json.dumps(self.global_adapters),
        )
        print(f"  Router saved: {len(keys)} centroids, {len(self.global_adapters)} global -> {path}")

    @staticmethod
    def needs_rebuild(router_path: str, adapters_dir: str,
                      problems_dir: str = "") -> bool:
        """Check if router state is stale and needs rebuilding.

        Returns True if:
          - Router file doesn't exist
          - Any adapter .npz file in adapters_dir is newer than router file
          - Any *_facts.json file in problems_dir is newer than router file

        This enables auto-rebuild when adapters are trained or facts are updated.
        """
        if not os.path.exists(router_path):
            return True

        router_mtime = os.path.getmtime(router_path)

        # Check adapter files
        adapters_path = Path(adapters_dir)
        if adapters_path.exists():
            for adapter_file in adapters_path.glob("*.npz"):
                if os.path.getmtime(adapter_file) > router_mtime:
                    return True

        # Check facts files (optional)
        if problems_dir:
            problems_path = Path(problems_dir)
            if problems_path.exists():
                for facts_file in problems_path.glob("*_facts.json"):
                    if os.path.getmtime(facts_file) > router_mtime:
                        return True

        return False

    @classmethod
    def load_or_rebuild(cls, router_path: str, model, tokenizer,
                        problems_dir: str, adapters_dir: str,
                        config: Optional[RouterConfig] = None,
                        force_rebuild: bool = False) -> "AdapterRouter":
        """Load router if valid, otherwise rebuild automatically.

        This is the recommended entry point for using the router. It ensures
        the router is always up-to-date with the latest adapters and facts.

        Args:
            router_path: Path to router_state.npz
            model: Base model for embedding (needed for rebuild)
            tokenizer: Tokenizer (needed for rebuild)
            problems_dir: Directory containing *_facts.json files
            adapters_dir: Directory containing *.npz adapter files
            config: Optional RouterConfig (uses defaults if None)
            force_rebuild: If True, always rebuild regardless of timestamps

        Returns:
            AdapterRouter: Loaded or freshly built router

        Example:
            router = AdapterRouter.load_or_rebuild(
                "router_state.npz", model, tokenizer,
                "problems/", "adapters/"
            )
        """
        if force_rebuild or cls.needs_rebuild(router_path, adapters_dir, problems_dir):
            if force_rebuild:
                print("  Router: force rebuild requested")
            else:
                print("  Router: stale state detected, rebuilding...")

            router = cls.build(model, tokenizer, problems_dir, adapters_dir, config)
            router.auto_register_global_adapters(adapters_dir)
            router.save(router_path)
            return router

        router = cls.load(router_path)
        # Re-check global adapters in case new ones were added
        router.auto_register_global_adapters(adapters_dir)
        return router

    @classmethod
    def load(cls, path: str) -> "AdapterRouter":
        """Load persisted router state."""
        data = np.load(path, allow_pickle=False)

        config_dict = json.loads(str(data["config_json"]))
        config = RouterConfig(**config_dict)

        keys = json.loads(str(data["keys_json"]))
        adapter_paths = json.loads(str(data["adapter_paths_json"]))
        matrix = data["centroid_matrix"]

        router = cls(config)
        router.centroids = {k: matrix[i] for i, k in enumerate(keys)}
        router.adapter_paths = adapter_paths
        router._rebuild_matrix()

        # Load global adapters if present (backward-compatible)
        if "global_adapters_json" in data:
            router.global_adapters = json.loads(str(data["global_adapters_json"]))

        n_global = len(router.global_adapters)
        print(f"  Router loaded: {len(keys)} centroids, {n_global} global from {path}")
        return router

    # ── Info ──────────────────────────────────────────────────────────

    def info(self) -> str:
        """Print summary of router state."""
        lines = [f"AdapterRouter: {len(self.centroids)} centroids, {len(self.global_adapters)} global"]
        lines.append(f"  Config: high={self.config.high_confidence}, "
                      f"gap={self.config.ambiguity_gap}, "
                      f"fallback={self.config.fallback}")
        lines.append(f"  Certainty cascade: {self.config.certainty_cascade} "
                      f"(gap threshold: {self.config.certainty_gap_threshold})")

        # Global adapters
        if self.global_adapters:
            lines.append("  Global adapters (certainty fallback):")
            for key in sorted(self.global_adapters.keys()):
                lines.append(f"    - {key}")

        # Group by domain prefix
        domains: dict[str, list] = {}
        for key in sorted(self.centroids.keys()):
            # Try to extract domain from key
            parts = key.rsplit("_adapter", 1)
            domain = parts[0] if len(parts) > 1 else key
            domains.setdefault(domain, []).append(key)

        lines.append(f"  Domains: {len(domains)}")
        for domain, keys in sorted(domains.items()):
            lines.append(f"    {domain}: {len(keys)} adapter(s)")

        return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────────


def _find_adapter_key(domain: str, cluster: str,
                      available: dict[str, str]) -> Optional[str]:
    """Find the best matching adapter key for a (domain, cluster) pair.

    Uses a priority-ordered search with fuzzy fallback:
      1. Exact: {domain}_{cluster}_adapter
      2. Exact: {domain}_{cluster}
      3. Domain-level: {domain}_adapter
      4. Any adapter starting with {domain}_
      5. Fuzzy: adapter containing domain prefix (first word)
    """
    # Normalize cluster name
    cluster_clean = cluster.replace(" ", "_").replace("-", "_").lower()

    # Priority 1-2: Exact cluster match
    if cluster != "all":
        for pattern in [f"{domain}_{cluster_clean}_adapter",
                        f"{domain}_{cluster_clean}"]:
            if pattern in available:
                return pattern

    # Priority 3: Domain-level adapter
    if f"{domain}_adapter" in available:
        return f"{domain}_adapter"

    # Priority 4: Any adapter starting with domain_
    domain_matches = [k for k in available if k.startswith(f"{domain}_")]
    if domain_matches:
        # Prefer ones with 'adapter' in the name
        with_adapter = [k for k in domain_matches if "adapter" in k]
        if with_adapter:
            return with_adapter[0]
        return domain_matches[0]

    # Priority 5: Fuzzy match on domain prefix
    # e.g., "chemical_conservation" -> look for adapters starting with "chemical"
    domain_prefix = domain.split("_")[0]
    if len(domain_prefix) >= 4:  # avoid overly short prefixes
        prefix_matches = [k for k in available
                          if k.startswith(f"{domain_prefix}_")]
        if prefix_matches:
            # Prefer cluster-specific if available
            if cluster != "all":
                cluster_matches = [k for k in prefix_matches
                                   if cluster_clean in k]
                if cluster_matches:
                    return cluster_matches[0]
            # Otherwise return first adapter-named match
            with_adapter = [k for k in prefix_matches if "adapter" in k]
            if with_adapter:
                return with_adapter[0]
            return prefix_matches[0]

    return None


def _flatten_dict(d: dict, prefix: str, out: list):
    """Flatten nested dict to list of (key, value) tuples."""
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, key, out)
        else:
            out.append((key, v))
