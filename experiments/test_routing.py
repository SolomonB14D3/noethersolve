#!/usr/bin/env python3
"""Test steering router with diagnostic output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_lm import load
from noethersolve.steering_router import SteeringRouter


def main():
    print("Loading model...")
    model, tokenizer = load("Qwen/Qwen3-4B-Base")

    print("Loading steering bank...")
    router = SteeringRouter.load("steering_bank.npz")
    print(f"  {len(router)} domains, {router.total_size_mb:.2f} MB")
    print(f"  Threshold: {router.threshold}")

    # Test queries with different characteristics
    test_queries = [
        # Factual claims the model might get wrong
        "Kinetic energy is always conserved in gravitational systems.",
        "The Riemann hypothesis has been proven.",
        "Competitive inhibitors increase Km.",
        "2D turbulence has an inverse energy cascade.",
        # General neutral question
        "What is 2+2?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        # Get activation and scores
        query_act = router.get_query_activation(model, tokenizer, query)
        scores = router.score_domains(query_act)

        # Analyze score distribution
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        min_score = sorted_scores[0][1]
        max_score = sorted_scores[-1][1]

        print(f"Score range: [{min_score:.4f}, {max_score:.4f}]")

        # Count by ranges
        ranges = [
            ("< -0.35", lambda s: s < -0.35),
            ("-0.35 to -0.2", lambda s: -0.35 <= s < -0.2),
            ("-0.2 to -0.1", lambda s: -0.2 <= s < -0.1),
            ("-0.1 to 0", lambda s: -0.1 <= s < 0),
            (">= 0", lambda s: s >= 0),
        ]

        for label, fn in ranges:
            count = sum(1 for d, s in scores.items() if fn(s))
            if count > 0:
                print(f"  {label}: {count} domains")

        # Show top 5 most negative (most "needing steering")
        print(f"\nMost relevant domains (lowest scores):")
        for d, s in sorted_scores[:5]:
            print(f"  {d}: {s:.4f}")

        # Try different thresholds
        print(f"\nRouting results by threshold:")
        for thresh in [-0.35, -0.20, -0.15, -0.10, -0.05, 0.0]:
            to_apply = {d: max(0, -s) for d, s in scores.items() if s < thresh}
            if to_apply:
                top3 = sorted(to_apply.items(), key=lambda x: -x[1])[:3]
                domains = ", ".join(f"{d}({w:.2f})" for d, w in top3)
                print(f"  thresh={thresh:+.2f}: {len(to_apply)} domains → {domains}")
            else:
                print(f"  thresh={thresh:+.2f}: no domains")


if __name__ == "__main__":
    main()
