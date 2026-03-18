#!/usr/bin/env python3
"""Build training data for the meta-router from existing experiments.

Scans all adapter benchmark results and fact files to create a comprehensive
outcomes dataset for training the meta-router.

Output: results/meta_router_outcomes.jsonl
"""

import json
from pathlib import Path
from collections import defaultdict
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))



def load_facts_text(facts_dir: Path) -> dict:
    """Load fact ID -> text mapping from all fact files."""
    facts_text = {}

    for facts_file in facts_dir.glob("*_facts.json"):
        try:
            with open(facts_file) as f:
                data = json.load(f)

            facts = data.get("facts", data.get("verifications", []))
            domain = facts_file.stem.replace("_facts", "")

            for i, fact in enumerate(facts):
                if isinstance(fact, dict):
                    fact_id = fact.get("id", f"{domain}_{i:02d}")
                    fact_text = fact.get("truth", fact.get("fact", ""))
                else:
                    fact_id = f"{domain}_{i:02d}"
                    fact_text = str(fact)

                facts_text[fact_id] = fact_text

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not parse {facts_file}: {e}")

    return facts_text


def scan_benchmark_results(results_dir: Path, facts_text: dict) -> list:
    """Scan benchmark JSON files for outcome data."""
    outcomes = []

    for json_file in results_dir.glob("**/*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Skip non-benchmark files
            if "adapted_results" not in data and "results" not in data:
                continue

            # Get adapter name
            adapter_path = data.get("adapter", "")
            adapter_name = Path(adapter_path).stem if adapter_path else json_file.stem

            # Get results list
            results_list = data.get("adapted_results", data.get("results", []))

            for result in results_list:
                fact_id = result.get("id", f"fact_{result.get('idx', 0)}")

                # Try to get fact text from lookup or result
                fact_text = result.get("fact_text", "")
                if not fact_text:
                    fact_text = facts_text.get(fact_id, "")
                if not fact_text:
                    # Try partial match
                    for fid, text in facts_text.items():
                        if fact_id in fid or fid in fact_id:
                            fact_text = text
                            break

                # Get margins
                baseline_margin = result.get("baseline_margin", result.get("vanilla_margin", 0.0))
                post_margin = result.get("margin", result.get("adapted_margin", 0.0))
                flipped = result.get("win", result.get("flipped", post_margin > 0))

                outcomes.append({
                    "fact_id": fact_id,
                    "fact_text": fact_text,
                    "baseline_margin": baseline_margin,
                    "adapter": adapter_name,
                    "post_margin": post_margin,
                    "flipped": flipped,
                    "cluster": result.get("cluster", ""),
                    "domain": result.get("domain", ""),
                    "source_file": str(json_file),
                })

        except (json.JSONDecodeError, KeyError):
            continue

    return outcomes


def scan_verification_logs(experiments_dir: Path, facts_text: dict) -> list:
    """Scan verification experiment logs for outcome data."""
    outcomes = []

    # Look for verification result files
    for log_file in experiments_dir.glob("**/verify_*.json"):
        try:
            with open(log_file) as f:
                data = json.load(f)

            if "facts" not in data:
                continue

            adapter = data.get("adapter", log_file.stem)

            for fact_data in data["facts"]:
                fact_id = fact_data.get("id", "")
                baseline = fact_data.get("baseline_margin", 0.0)
                post = fact_data.get("adapted_margin", fact_data.get("margin", 0.0))

                outcomes.append({
                    "fact_id": fact_id,
                    "fact_text": facts_text.get(fact_id, fact_data.get("truth", "")),
                    "baseline_margin": baseline,
                    "adapter": adapter,
                    "post_margin": post,
                    "flipped": post > 0,
                    "cluster": fact_data.get("cluster", ""),
                    "domain": "",
                    "source_file": str(log_file),
                })

        except (json.JSONDecodeError, KeyError):
            continue

    return outcomes


def deduplicate_outcomes(outcomes: list) -> list:
    """Remove duplicate (fact_id, adapter) pairs, keeping best outcome."""
    seen = {}  # (fact_id, adapter) -> best outcome

    for o in outcomes:
        key = (o["fact_id"], o["adapter"])
        if key not in seen or o["post_margin"] > seen[key]["post_margin"]:
            seen[key] = o

    return list(seen.values())


def main():
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "results"
    problems_dir = base_dir / "problems"
    experiments_dir = base_dir / "experiments"
    output_file = results_dir / "meta_router_outcomes.jsonl"

    print("=== Building Meta-Router Training Data ===\n")

    # Step 1: Load all fact texts
    print("1. Loading fact texts from problems/...")
    facts_text = load_facts_text(problems_dir)
    print(f"   Loaded {len(facts_text)} fact texts\n")

    # Step 2: Scan benchmark results
    print("2. Scanning benchmark results...")
    benchmark_outcomes = scan_benchmark_results(results_dir, facts_text)
    print(f"   Found {len(benchmark_outcomes)} outcomes from benchmarks\n")

    # Step 3: Scan verification logs
    print("3. Scanning verification logs...")
    verify_outcomes = scan_verification_logs(experiments_dir, facts_text)
    print(f"   Found {len(verify_outcomes)} outcomes from verification logs\n")

    # Step 4: Combine and deduplicate
    all_outcomes = benchmark_outcomes + verify_outcomes
    print(f"4. Total raw outcomes: {len(all_outcomes)}")

    unique_outcomes = deduplicate_outcomes(all_outcomes)
    print(f"   After deduplication: {len(unique_outcomes)}\n")

    # Step 5: Filter out outcomes without fact text
    valid_outcomes = [o for o in unique_outcomes if o["fact_text"]]
    print(f"5. Outcomes with fact text: {len(valid_outcomes)}\n")

    # Step 6: Write output
    print(f"6. Writing to {output_file}...")
    with open(output_file, "w") as f:
        for o in valid_outcomes:
            f.write(json.dumps(o) + "\n")

    print("   Done!\n")

    # Stats
    by_adapter = defaultdict(int)
    flipped_count = 0
    for o in valid_outcomes:
        by_adapter[o["adapter"]] += 1
        if o["flipped"]:
            flipped_count += 1

    print("=== Statistics ===")
    print(f"Total outcomes: {len(valid_outcomes)}")
    print(f"Unique adapters: {len(by_adapter)}")
    print(f"Flip rate: {flipped_count}/{len(valid_outcomes)} = {flipped_count/max(1,len(valid_outcomes)):.1%}")
    print("\nTop adapters by outcome count:")
    for adapter, count in sorted(by_adapter.items(), key=lambda x: -x[1])[:10]:
        print(f"  {adapter}: {count}")


if __name__ == "__main__":
    main()
