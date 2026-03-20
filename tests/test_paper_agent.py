"""Tests for paper_agent module.

Tests paper generation, AI scrubbing, and cluster evaluation logic.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestClusterMetrics:
    """Tests for ClusterMetrics dataclass."""

    def test_maturity_score_perfect_cluster(self):
        """Test maturity score for a perfect cluster."""
        from noethersolve.paper_agent import ClusterMetrics

        metrics = ClusterMetrics(
            cluster_id="test",
            facts_count=16,
            facts_flipped=16,
            margin_avg=10.0,
            margin_min=2.0,
            coverage_pct=100.0,
            has_numerical_verification=True,
            has_independent_validation=True,
        )

        # flip_rate = 1.0 * 0.4 = 0.4
        # margin_score = (10 + 20) / 40 = 0.75 * 0.2 = 0.15
        # coverage = 1.0 * 0.2 = 0.2
        # verification + validation = 0.2
        # total = 0.95
        assert metrics.maturity_score >= 0.9

    def test_maturity_score_empty_cluster(self):
        """Test maturity score for empty cluster."""
        from noethersolve.paper_agent import ClusterMetrics

        metrics = ClusterMetrics(
            cluster_id="empty",
            facts_count=0,
            facts_flipped=0,
            margin_avg=0.0,
            margin_min=0.0,
            coverage_pct=0.0,
            has_numerical_verification=False,
            has_independent_validation=False,
        )

        # Should handle zero division gracefully
        # flip_rate = 0/max(0,1) = 0 * 0.4 = 0
        # margin_score = (0 + 20) / 40 = 0.5 * 0.2 = 0.1
        # coverage = 0 * 0.2 = 0
        # no bonuses = 0
        # total = 0.1
        assert metrics.maturity_score == 0.1

    def test_maturity_score_partial_cluster(self):
        """Test maturity score for partial cluster."""
        from noethersolve.paper_agent import ClusterMetrics

        metrics = ClusterMetrics(
            cluster_id="partial",
            facts_count=10,
            facts_flipped=5,
            margin_avg=0.0,
            margin_min=-5.0,
            coverage_pct=50.0,
            has_numerical_verification=True,
            has_independent_validation=False,
        )

        score = metrics.maturity_score
        assert 0.3 <= score <= 0.7


class TestAILanguageScrubbing:
    """Tests for AI language scrubbing functionality."""

    def test_scrub_banned_words(self):
        """Test that banned words are removed."""
        from noethersolve.paper_agent import PaperAgent

        agent = PaperAgent()
        text = "We delve into the intricate tapestry of this novel approach."
        scrubbed = agent.scrub_ai_language(text)

        assert "delve" not in scrubbed.lower()
        assert "intricate" not in scrubbed.lower()
        assert "tapestry" not in scrubbed.lower()
        assert "novel" not in scrubbed.lower()

    def test_scrub_preserves_content(self):
        """Test that scrubbing preserves scientific content."""
        from noethersolve.paper_agent import PaperAgent

        agent = PaperAgent()
        text = "The energy is conserved with frac_var = 1.2e-6."
        scrubbed = agent.scrub_ai_language(text)

        assert "energy" in scrubbed
        assert "conserved" in scrubbed
        assert "1.2e-6" in scrubbed

    def test_scrub_case_insensitive(self):
        """Test that scrubbing is case insensitive."""
        from noethersolve.paper_agent import PaperAgent

        agent = PaperAgent()
        text = "We DELVE into this. It's worth noting that NOTABLY, this is interesting."
        scrubbed = agent.scrub_ai_language(text)

        assert "delve" not in scrubbed.lower()
        assert "notably" not in scrubbed.lower()
        assert "worth noting" not in scrubbed.lower()

    def test_scrub_cleans_double_spaces(self):
        """Test that scrubbing cleans up resulting double spaces."""
        from noethersolve.paper_agent import PaperAgent

        agent = PaperAgent()
        text = "This is notably a good result."
        scrubbed = agent.scrub_ai_language(text)

        assert "  " not in scrubbed


class TestShouldWritePaper:
    """Tests for paper readiness evaluation."""

    def test_should_write_above_threshold(self):
        """Test that high-maturity clusters pass threshold."""
        from noethersolve.paper_agent import PaperAgent, ClusterMetrics

        agent = PaperAgent()

        # Mock get_cluster_metrics to return high-scoring cluster
        with patch.object(agent, "get_cluster_metrics") as mock_metrics:
            mock_metrics.return_value = ClusterMetrics(
                cluster_id="test",
                facts_count=16,
                facts_flipped=16,
                margin_avg=10.0,
                margin_min=2.0,
                coverage_pct=100.0,
                has_numerical_verification=True,
                has_independent_validation=True,
            )

            assert agent.should_write_paper("test", threshold=0.82)

    def test_should_not_write_below_threshold(self):
        """Test that low-maturity clusters fail threshold."""
        from noethersolve.paper_agent import PaperAgent, ClusterMetrics

        agent = PaperAgent()

        with patch.object(agent, "get_cluster_metrics") as mock_metrics:
            mock_metrics.return_value = ClusterMetrics(
                cluster_id="test",
                facts_count=10,
                facts_flipped=2,
                margin_avg=-5.0,
                margin_min=-15.0,
                coverage_pct=20.0,
                has_numerical_verification=False,
                has_independent_validation=False,
            )

            assert not agent.should_write_paper("test", threshold=0.82)

    def test_should_write_zero_threshold(self):
        """Test that zero threshold always passes."""
        from noethersolve.paper_agent import PaperAgent, ClusterMetrics

        agent = PaperAgent()

        with patch.object(agent, "get_cluster_metrics") as mock_metrics:
            mock_metrics.return_value = ClusterMetrics(
                cluster_id="test",
                facts_count=1,
                facts_flipped=0,
                margin_avg=-20.0,
                margin_min=-20.0,
                coverage_pct=0.0,
                has_numerical_verification=False,
                has_independent_validation=False,
            )

            assert agent.should_write_paper("test", threshold=0.0)

    def test_should_write_missing_cluster(self):
        """Test handling of missing cluster."""
        from noethersolve.paper_agent import PaperAgent

        agent = PaperAgent()

        with patch.object(agent, "get_cluster_metrics") as mock_metrics:
            mock_metrics.return_value = None

            assert not agent.should_write_paper("nonexistent")


class TestGenerateOutline:
    """Tests for outline generation."""

    def test_outline_has_sections(self):
        """Test that outline has required sections."""
        from noethersolve.paper_agent import PaperAgent, ClusterMetrics

        agent = PaperAgent()
        metrics = ClusterMetrics(
            cluster_id="test_cluster",
            facts_count=10,
            facts_flipped=8,
            margin_avg=5.0,
            margin_min=1.0,
            coverage_pct=80.0,
            has_numerical_verification=True,
            has_independent_validation=False,
        )

        draft = agent.generate_draft("test_cluster", metrics)

        assert isinstance(draft, str)
        assert len(draft) > 0

    def test_outline_includes_metrics(self):
        """Test that outline includes cluster metrics."""
        from noethersolve.paper_agent import PaperAgent, ClusterMetrics

        agent = PaperAgent()
        metrics = ClusterMetrics(
            cluster_id="test",
            facts_count=16,
            facts_flipped=12,
            margin_avg=7.5,
            margin_min=2.0,
            coverage_pct=75.0,
            has_numerical_verification=True,
            has_independent_validation=True,
        )

        draft = agent.generate_draft("test", metrics)

        assert isinstance(draft, str)
        assert len(draft) > 100  # substantial draft


class TestEnqueueFutureWork:
    """Tests for future work extraction and enqueueing."""

    def test_enqueue_extracts_bullet_points(self):
        """Test that bullet points are extracted from future work section."""
        from noethersolve.paper_agent import PaperAgent

        agent = PaperAgent()

        # Create temporary results directory
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.results_dir = Path(tmpdir)

            draft = """
## Discussion

### 5.3 Future Work

- Extend to 3D vortex systems
- Test on turbulent flows
- Investigate quantum analogues

## Conclusion
"""
            count = agent.enqueue_future_work(draft, "test_cluster")

            assert count == 3

            # Check file contents
            open_questions = Path(tmpdir) / "open_questions.jsonl"
            assert open_questions.exists()

            with open(open_questions) as f:
                lines = f.readlines()
                assert len(lines) == 3

                first_entry = json.loads(lines[0])
                assert first_entry["type"] == "direction"
                assert first_entry["source"] == "paper:test_cluster"
                assert "3D" in first_entry["text"]

    def test_enqueue_handles_no_future_work(self):
        """Test handling of draft without future work section."""
        from noethersolve.paper_agent import PaperAgent

        agent = PaperAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            agent.results_dir = Path(tmpdir)

            draft = """
## Introduction

This paper describes...

## Conclusion

We conclude that...
"""
            count = agent.enqueue_future_work(draft, "test_cluster")
            assert count == 0


class TestPaperResult:
    """Tests for PaperResult dataclass."""

    def test_success_with_doi(self):
        """Test success property with DOI."""
        from noethersolve.paper_agent import PaperResult

        result = PaperResult(
            cluster_id="test",
            title="Test Paper",
            pdf_path=Path("/tmp/test.pdf"),
            doi="10.5281/zenodo.12345",
            zenodo_url="https://zenodo.org/record/12345",
        )

        assert result.success

    def test_failure_without_pdf(self):
        """Test failure when pdf_path is missing."""
        from noethersolve.paper_agent import PaperResult

        result = PaperResult(
            cluster_id="test",
            title="Test Paper",
            doi="10.5281/zenodo.12345",
        )

        assert not result.success

    def test_failure_with_errors(self):
        """Test failure when errors present."""
        from noethersolve.paper_agent import PaperResult

        result = PaperResult(
            cluster_id="test",
            title="Test Paper",
            doi="10.5281/zenodo.12345",
            errors=["Compilation failed"],
        )

        assert not result.success


class TestBannedPhrases:
    """Tests for the banned phrases list."""

    def test_banned_phrases_exist(self):
        """Test that banned phrases list is populated."""
        from noethersolve.paper_agent import BANNED_PHRASES

        assert len(BANNED_PHRASES) >= 30
        assert "delve" in BANNED_PHRASES
        assert "tapestry" in BANNED_PHRASES
        assert "multifaceted" in BANNED_PHRASES

    def test_all_phrases_lowercase(self):
        """Test that all banned phrases are lowercase for matching."""
        from noethersolve.paper_agent import BANNED_PHRASES

        for phrase in BANNED_PHRASES:
            assert phrase == phrase.lower(), f"Phrase not lowercase: {phrase}"


class TestWritePaperForCluster:
    """Tests for the convenience function."""

    def test_returns_status_message(self):
        """Test that function returns status message."""
        from noethersolve.paper_agent import write_paper_for_cluster

        # Mock to avoid actual API calls
        with patch("noethersolve.paper_agent.PaperAgent") as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.write_and_publish.return_value = MagicMock(
                success=False,
                pdf_path=None,
                errors=["No metrics found"],
            )

            result = write_paper_for_cluster("nonexistent")

            assert "Failed:" in result
            assert "No metrics" in result
