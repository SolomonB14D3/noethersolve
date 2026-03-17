"""Tests for the outcome logger module."""

import pytest
import tempfile
import json
from pathlib import Path

from noethersolve.outcome_logger import OutcomeLogger


class TestOutcomeLogger:
    """Test the outcome logger."""

    def test_log_outcome_adds_to_cache(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            logger = OutcomeLogger(outcomes_file=path)
            logger._flush_interval = 100  # Don't auto-flush

            logger.log_outcome(
                fact_id="test01",
                fact_text="Test fact",
                baseline_margin=-10.0,
                adapter="test_adapter",
                post_margin=5.0,
                flipped=True,
            )

            assert len(logger._cache) == 1
            logger.close()
        finally:
            path.unlink()

    def test_flush_writes_to_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            logger = OutcomeLogger(outcomes_file=path)
            logger._flush_interval = 1  # Flush after each

            logger.log_outcome(
                fact_id="test01",
                fact_text="Test fact",
                baseline_margin=-10.0,
                adapter="test_adapter",
                post_margin=5.0,
                flipped=True,
            )

            # Should have flushed
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) >= 1

            logger.close()
        finally:
            path.unlink()

    def test_log_batch(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            logger = OutcomeLogger(outcomes_file=path)
            logger._flush_interval = 100

            logger.log_batch(
                fact_ids=["f1", "f2", "f3"],
                fact_texts=["Fact 1", "Fact 2", "Fact 3"],
                baseline_margins=[-10.0, -20.0, -30.0],
                adapter="batch_adapter",
                post_margins=[5.0, -5.0, 15.0],
                flipped_flags=[True, False, True],
            )

            assert len(logger._cache) == 3
            logger.close()
        finally:
            path.unlink()

    def test_context_manager(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            with OutcomeLogger(outcomes_file=path) as logger:
                logger._flush_interval = 100
                logger.log_outcome(
                    fact_id="test",
                    fact_text="Test",
                    baseline_margin=-10.0,
                    adapter="adapter",
                    post_margin=5.0,
                    flipped=True,
                )
                # Not flushed yet
                assert len(logger._cache) == 1

            # After context exit, should be flushed
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) >= 1
        finally:
            path.unlink()

    def test_stats_on_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            logger = OutcomeLogger(outcomes_file=path)
            # Write empty file
            path.write_text("")

            stats = logger.stats()
            assert stats["n_outcomes"] == 0
            logger.close()
        finally:
            path.unlink()

    def test_stats_with_data(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            # Pre-populate file
            with open(path, "w") as f:
                f.write(json.dumps({
                    "fact_id": "f1",
                    "adapter": "a1",
                    "flipped": True,
                }) + "\n")
                f.write(json.dumps({
                    "fact_id": "f2",
                    "adapter": "a1",
                    "flipped": False,
                }) + "\n")
                f.write(json.dumps({
                    "fact_id": "f3",
                    "adapter": "a2",
                    "flipped": True,
                }) + "\n")

            logger = OutcomeLogger(outcomes_file=path)
            stats = logger.stats()

            assert stats["n_outcomes"] == 3
            assert stats["n_adapters"] == 2
            assert stats["n_facts"] == 3
            assert stats["flip_rate"] == 2 / 3

            logger.close()
        finally:
            path.unlink()

    def test_extra_fields(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            logger = OutcomeLogger(outcomes_file=path)
            logger._flush_interval = 1

            logger.log_outcome(
                fact_id="test",
                fact_text="Test",
                baseline_margin=-10.0,
                adapter="adapter",
                post_margin=5.0,
                flipped=True,
                extra={"custom_field": "value", "numeric": 42},
            )

            with open(path) as f:
                record = json.loads(f.readline())

            assert record["custom_field"] == "value"
            assert record["numeric"] == 42

            logger.close()
        finally:
            path.unlink()

    def test_creates_directory(self):
        import tempfile
        import shutil

        temp_dir = Path(tempfile.mkdtemp())
        nested_path = temp_dir / "nested" / "dir" / "outcomes.jsonl"

        try:
            logger = OutcomeLogger(outcomes_file=nested_path)
            assert nested_path.parent.exists()
            logger.close()
        finally:
            shutil.rmtree(temp_dir)
