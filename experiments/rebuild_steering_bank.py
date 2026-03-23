#!/usr/bin/env python3
"""Rebuild steering bank with semantic centroids for routing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_lm import load
from noethersolve.steering_router import SteeringRouter


def main():
    print("Loading model...")
    model, tokenizer = load("Qwen/Qwen3-4B-Base")

    print("\nBuilding steering bank with centroids...")
    router = SteeringRouter.build_from_facts(
        model,
        tokenizer,
        "problems",
        layer=15,
    )

    print(f"\n{'='*60}")
    print(f"Domains: {len(router)}")
    print(f"Steering size: {router.total_size_mb:.2f} MB")
    print(f"Layer: {router.layer}")

    # Save
    output = Path("steering_bank.npz")
    router.save(str(output))
    print(f"\nSaved to: {output}")
    print(f"File size: {output.stat().st_size / 1024 / 1024:.2f} MB")

    # Test routing
    print("\n" + "="*60)
    print("Testing semantic routing...")

    test_queries = [
        ("What is kinetic energy conservation?", ["3body", "hamiltonian", "physics"]),
        ("How does enzyme inhibition work?", ["biochemistry", "clinical", "drug"]),
        ("Is the Riemann hypothesis proven?", ["number_theory", "proof", "millennium"]),
        ("What happens in 2D turbulence?", ["physics", "vortex", "ns_regularity"]),
        ("How do CRISPR guides work?", ["genetics", "protein", "biology"]),
        ("What is 2+2?", []),  # Should match nothing specific
    ]

    for query, expected_keywords in test_queries:
        print(f"\nQuery: {query}")

        result = router.route_and_steer(model, tokenizer, query, threshold=0.10)

        if result:
            print(f"  Routed to: {', '.join(result.domains_applied)}")
            # Check if any expected keywords match
            matched = [kw for kw in expected_keywords
                      if any(kw in d for d in result.domains_applied)]
            if matched:
                print(f"  ✓ Matched expected: {matched}")
            elif expected_keywords:
                print(f"  ✗ Expected: {expected_keywords}")
        else:
            if not expected_keywords:
                print("  ✓ Correctly found no specific domain")
            else:
                print(f"  ✗ No match (expected: {expected_keywords})")

        # Show top 3 scores
        query_act = router.get_query_activation(model, tokenizer, query)
        scores = router.score_domains(query_act)
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        print(f"  Top scores: {', '.join(f'{d}={s:.3f}' for d, s in top3)}")


if __name__ == "__main__":
    main()
