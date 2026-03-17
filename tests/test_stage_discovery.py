"""Tests for the stage discovery module."""

import pytest
from noethersolve.stage_discovery import (
    StageDiscoverer,
    DiscoveryConfig,
    EvalResult,
    StageSequence,
)


def make_oracle(pass_sets: dict):
    """Create a deterministic oracle for testing.

    Args:
        pass_sets: {adapter_name: set of fact_ids that pass}
    """
    def oracle(adapter: str, facts: list) -> EvalResult:
        passed = pass_sets.get(adapter, set())
        fact_ids = {f.get("id", f"fact_{i}") for i, f in enumerate(facts)}

        return EvalResult(
            adapter=adapter,
            n_passed=len(passed & fact_ids),
            n_total=len(facts),
            margins=[10.0 if f.get("id") in passed else -10.0 for f in facts],
            passed_ids=passed & fact_ids,
        )
    return oracle


class TestStageSequence:
    """Test StageSequence dataclass."""

    def test_lt_comparison(self):
        s1 = StageSequence(adapters=["a"], passed_ids={"f1"}, score=0.5)
        s2 = StageSequence(adapters=["b"], passed_ids={"f1", "f2"}, score=0.8)

        # Higher score should be "less than" for min-heap
        assert s2 < s1


class TestStageDiscoverer:
    """Test the stage discoverer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.facts = [
            {"id": "f1", "truth": "Fact 1"},
            {"id": "f2", "truth": "Fact 2"},
            {"id": "f3", "truth": "Fact 3"},
            {"id": "f4", "truth": "Fact 4"},
        ]
        self.adapters = ["adapter_a", "adapter_b", "adapter_c"]

    def test_greedy_finds_best_single(self):
        # adapter_b passes the most facts
        oracle = make_oracle({
            "adapter_a": {"f1"},
            "adapter_b": {"f1", "f2", "f3"},
            "adapter_c": {"f2"},
        })

        discoverer = StageDiscoverer(
            facts=self.facts,
            candidate_adapters=self.adapters,
            oracle_fn=oracle,
            config=DiscoveryConfig(max_stages=1),
        )

        result = discoverer.discover("greedy")
        assert result.adapters == ["adapter_b"]
        assert result.score == 0.75  # 3/4

    def test_greedy_chains_adapters(self):
        # Need both adapters to cover all facts
        # adapter_b must also pass f1, f2 to avoid being seen as regression
        oracle = make_oracle({
            "adapter_a": {"f1", "f2"},
            "adapter_b": {"f1", "f2", "f3", "f4"},  # Includes previous + new
        })

        discoverer = StageDiscoverer(
            facts=self.facts,
            candidate_adapters=self.adapters,
            oracle_fn=oracle,
            config=DiscoveryConfig(max_stages=3),
        )

        result = discoverer.discover("greedy")
        # adapter_b covers all, should be picked first or second
        assert result.score == 1.0

    def test_greedy_avoids_regression(self):
        # adapter_b causes regression on f1
        oracle = make_oracle({
            "adapter_a": {"f1", "f2"},
            "adapter_b": {"f3"},  # Does NOT include f1, f2
        })

        discoverer = StageDiscoverer(
            facts=self.facts,
            candidate_adapters=self.adapters,
            oracle_fn=oracle,
            config=DiscoveryConfig(max_stages=3, regression_tolerance=0),
        )

        result = discoverer.discover("greedy")
        # Should only use adapter_a since adapter_b would regress
        assert result.adapters == ["adapter_a"]

    def test_greedy_allows_regression_with_tolerance(self):
        oracle = make_oracle({
            "adapter_a": {"f1", "f2"},
            "adapter_b": {"f3", "f4"},  # Would regress f1, f2 but add f3, f4
        })

        discoverer = StageDiscoverer(
            facts=self.facts,
            candidate_adapters=self.adapters,
            oracle_fn=oracle,
            config=DiscoveryConfig(max_stages=3, regression_tolerance=2),
        )

        result = discoverer.discover("greedy")
        assert len(result.adapters) == 2

    def test_beam_search_basic(self):
        oracle = make_oracle({
            "adapter_a": {"f1", "f2"},
            "adapter_b": {"f2", "f3"},
            "adapter_c": {"f3", "f4"},
        })

        discoverer = StageDiscoverer(
            facts=self.facts,
            candidate_adapters=self.adapters,
            oracle_fn=oracle,
            config=DiscoveryConfig(max_stages=3, beam_width=2),
        )

        result = discoverer.discover("beam")
        assert result.score > 0

    def test_early_stop_at_threshold(self):
        # adapter_a achieves 100% immediately
        oracle = make_oracle({
            "adapter_a": {"f1", "f2", "f3", "f4"},
        })

        discoverer = StageDiscoverer(
            facts=self.facts,
            candidate_adapters=self.adapters,
            oracle_fn=oracle,
            config=DiscoveryConfig(max_stages=5, early_stop_threshold=1.0),
        )

        result = discoverer.discover("greedy")
        assert result.score == 1.0
        assert len(result.adapters) == 1

    def test_caching_works(self):
        call_count = [0]

        def counting_oracle(adapter, facts):
            call_count[0] += 1
            return EvalResult(
                adapter=adapter,
                n_passed=2,
                n_total=4,
                margins=[10.0, 10.0, -10.0, -10.0],
                passed_ids={"f1", "f2"},
            )

        discoverer = StageDiscoverer(
            facts=self.facts,
            candidate_adapters=["adapter_a"],
            oracle_fn=counting_oracle,
            config=DiscoveryConfig(max_stages=3),
        )

        # Call evaluate twice for same adapter
        result1 = discoverer._evaluate("adapter_a")
        result2 = discoverer._evaluate("adapter_a")

        assert call_count[0] == 1  # Only called once due to caching
        assert result1 is result2  # Same object

    def test_genetic_runs(self):
        oracle = make_oracle({
            "adapter_a": {"f1", "f2"},
            "adapter_b": {"f3", "f4"},
        })

        discoverer = StageDiscoverer(
            facts=self.facts,
            candidate_adapters=self.adapters,
            oracle_fn=oracle,
            config=DiscoveryConfig(
                population_size=5,
                generations=3,
            ),
        )

        result = discoverer.discover("genetic")
        assert result.score >= 0

    def test_invalid_method_raises(self):
        oracle = make_oracle({})
        discoverer = StageDiscoverer(
            facts=self.facts,
            candidate_adapters=self.adapters,
            oracle_fn=oracle,
        )

        with pytest.raises(ValueError, match="Unknown method"):
            discoverer.discover("invalid_method")


class TestEvalResult:
    """Test EvalResult dataclass."""

    def test_defaults(self):
        result = EvalResult(
            adapter="test",
            n_passed=5,
            n_total=10,
            margins=[1.0] * 10,
            passed_ids={"f1", "f2"},
        )

        assert result.regressed_ids == set()
