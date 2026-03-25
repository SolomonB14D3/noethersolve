#!/usr/bin/env python3
"""Build and test the persistent adapter router.

One-time build: embeds all facts from problems/*_facts.json, computes domain
centroids, maps to available adapters, saves to router_state.npz.

Then runs a validation: for each fact in each domain, check that the router
sends it to the correct adapter (or at least to the correct domain).

Usage:
    python experiments/build_router.py              # build + validate
    python experiments/build_router.py --info       # print router info
    python experiments/build_router.py --validate   # validate existing router
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx_lm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noethersolve.adapter_router import AdapterRouter

MODEL_ID = "Qwen/Qwen3-14B-Base"
PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "problems"
ADAPTERS_DIR = Path(__file__).resolve().parent.parent / "adapters" / "qwen3_4b_base"
ROUTER_PATH = Path(__file__).resolve().parent.parent / "router_state.npz"


def build(model, tokenizer):
    """Build router from scratch."""
    print("Building router...")
    t0 = time.time()
    router = AdapterRouter.build(
        model, tokenizer,
        str(PROBLEMS_DIR), str(ADAPTERS_DIR),
    )

    # Auto-register global adapters (certainty decontamination, etc.)
    router.auto_register_global_adapters(str(ADAPTERS_DIR))

    print(f"  Build time: {time.time()-t0:.1f}s")
    router.save(str(ROUTER_PATH))
    return router


def validate(model, tokenizer, router):
    """Validate: for each fact, check routing goes to correct domain."""
    print("\nValidating router...")

    total = 0
    correct_domain = 0
    correct_exact = 0
    fallbacks = 0
    ambiguous = 0

    for facts_file in sorted(PROBLEMS_DIR.glob("*_facts.json")):
        domain = facts_file.stem.replace("_facts", "")

        try:
            with open(facts_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        facts = data.get("facts", data) if isinstance(data, dict) else data
        if not isinstance(facts, list) or len(facts) == 0:
            continue

        domain_correct = 0
        domain_total = 0

        for fact in facts:
            context = fact.get("context", "")
            if not context:
                continue

            decision = router.route(model, tokenizer, context)
            total += 1
            domain_total += 1

            if decision.level == "fallback":
                fallbacks += 1
                continue

            if decision.level == "ambiguous":
                ambiguous += 1

            # Check if routed to correct domain
            key = decision.primary_key or ""
            if domain in key or key.startswith(domain.split("_")[0]):
                correct_domain += 1
                domain_correct += 1

            # Exact match (adapter key matches what we'd expect)
            cluster = fact.get("cluster", "all")
            expected_patterns = [
                f"{domain}_{cluster}_adapter",
                f"{domain}_adapter",
                f"{domain}_stage5",
            ]
            if key in expected_patterns:
                correct_exact += 1

        if domain_total > 0:
            pct = 100 * domain_correct / domain_total
            status = "OK" if pct > 50 else "MISS"
            print(f"  {domain:<35} {domain_correct:>3}/{domain_total:<3} ({pct:>5.1f}%) {status}")

    print(f"\n  TOTAL: {correct_domain}/{total} ({100*correct_domain/total:.1f}%) routed to correct domain")
    print(f"  Exact adapter match: {correct_exact}/{total} ({100*correct_exact/total:.1f}%)")
    print(f"  Fallbacks (no adapter): {fallbacks}")
    print(f"  Ambiguous (tried both): {ambiguous}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", action="store_true", help="Print router info only")
    parser.add_argument("--validate", action="store_true", help="Validate existing router")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild")
    args = parser.parse_args()

    if args.info:
        if ROUTER_PATH.exists():
            router = AdapterRouter.load(str(ROUTER_PATH))
            print(router.info())
        else:
            print("No router found. Run without --info to build.")
        return

    # Load model
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = mlx_lm.load(MODEL_ID)
    model.eval()

    if args.validate and ROUTER_PATH.exists() and not args.rebuild:
        router = AdapterRouter.load(str(ROUTER_PATH))
        validate(model, tokenizer, router)
    else:
        router = build(model, tokenizer)
        validate(model, tokenizer, router)


if __name__ == "__main__":
    main()
