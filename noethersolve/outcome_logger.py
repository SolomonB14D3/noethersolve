"""Outcome Logger — Records fact × adapter outcomes for meta-router training.

Every time we evaluate an adapter against facts, log the outcomes.
This builds the training corpus incrementally over time.

Usage:
    logger = OutcomeLogger()

    # After running oracle evaluation:
    logger.log_outcome(
        fact_id="ns01",
        fact_text="2D NS is globally regular",
        baseline_margin=-45.2,
        adapter="ns_conservation",
        post_margin=12.3,
        flipped=True,
        cluster="ns_regularity",
        domain="navier_stokes"
    )

    # Or log a batch from oracle results:
    logger.log_batch(fact_ids, fact_texts, baseline_margins, adapter,
                     post_margins, flipped_flags)

    # Outcomes are automatically saved to results/meta_router_outcomes.jsonl
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import threading


class OutcomeLogger:
    """Thread-safe logger for fact × adapter outcomes."""

    def __init__(self, outcomes_file: Optional[Path] = None):
        if outcomes_file is None:
            # Default path
            base = Path(__file__).parent.parent
            outcomes_file = base / "results" / "meta_router_outcomes.jsonl"

        self.outcomes_file = Path(outcomes_file)
        self.outcomes_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: List[dict] = []
        self._flush_interval = 10  # Flush every N outcomes

    def log_outcome(
        self,
        fact_id: str,
        fact_text: str,
        baseline_margin: float,
        adapter: str,
        post_margin: float,
        flipped: bool,
        cluster: str = "",
        domain: str = "",
        extra: Optional[dict] = None,
    ):
        """Log a single outcome."""
        record = {
            "fact_id": fact_id,
            "fact_text": fact_text,
            "baseline_margin": baseline_margin,
            "adapter": adapter,
            "post_margin": post_margin,
            "flipped": flipped,
            "cluster": cluster,
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            record.update(extra)

        with self._lock:
            self._cache.append(record)
            if len(self._cache) >= self._flush_interval:
                self._flush()

    def log_batch(
        self,
        fact_ids: List[str],
        fact_texts: List[str],
        baseline_margins: List[float],
        adapter: str,
        post_margins: List[float],
        flipped_flags: List[bool],
        clusters: Optional[List[str]] = None,
        domain: str = "",
    ):
        """Log a batch of outcomes from oracle evaluation."""
        clusters = clusters or [""] * len(fact_ids)

        for i in range(len(fact_ids)):
            self.log_outcome(
                fact_id=fact_ids[i],
                fact_text=fact_texts[i],
                baseline_margin=baseline_margins[i],
                adapter=adapter,
                post_margin=post_margins[i],
                flipped=flipped_flags[i],
                cluster=clusters[i] if i < len(clusters) else "",
                domain=domain,
            )

    def _flush(self):
        """Write cached outcomes to file."""
        if not self._cache:
            return

        with open(self.outcomes_file, "a") as f:
            for record in self._cache:
                f.write(json.dumps(record) + "\n")

        self._cache = []

    def close(self):
        """Flush remaining cache and close."""
        with self._lock:
            self._flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def stats(self) -> dict:
        """Get statistics about logged outcomes."""
        if not self.outcomes_file.exists():
            return {"n_outcomes": 0}

        adapters = set()
        facts = set()
        flipped = 0
        total = 0

        with open(self.outcomes_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    adapters.add(record.get("adapter", ""))
                    facts.add(record.get("fact_id", ""))
                    if record.get("flipped", False):
                        flipped += 1
                    total += 1
                except json.JSONDecodeError:
                    continue

        return {
            "n_outcomes": total,
            "n_adapters": len(adapters),
            "n_facts": len(facts),
            "flip_rate": flipped / max(1, total),
        }


# Global logger instance
_global_logger: Optional[OutcomeLogger] = None


def get_logger() -> OutcomeLogger:
    """Get the global outcome logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = OutcomeLogger()
    return _global_logger


def log_outcome(**kwargs):
    """Convenience function to log an outcome."""
    get_logger().log_outcome(**kwargs)


def log_batch(**kwargs):
    """Convenience function to log a batch of outcomes."""
    get_logger().log_batch(**kwargs)
