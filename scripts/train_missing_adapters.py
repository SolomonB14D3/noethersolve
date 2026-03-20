#!/usr/bin/env python3
"""
Train adapters for all domains that don't have them yet.
This is the longest-running pure-local job we can do.
"""
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).parent.parent
ADAPTERS_DIR = PROJECT / "adapters"
PROBLEMS_DIR = PROJECT / "problems"
PYTHON = "/Users/bryan/miniconda3/bin/python3"

sys.path.insert(0, str(PROJECT / "scripts"))
from job_tracker import run_job

def get_domains_needing_adapters():
    """Find domains with facts files but no adapters."""
    facts_files = list(PROBLEMS_DIR.glob("*_facts.json"))
    existing = {p.stem.replace("_adapter", "").rsplit("_", 1)[0]
                for p in ADAPTERS_DIR.glob("*.npz")}

    missing = []
    for ff in facts_files:
        domain = ff.stem.replace("_facts", "")
        if domain not in existing:
            missing.append((domain, ff))

    return missing

def train_adapter_for_domain(domain: str, facts_file: Path):
    """Train adapter for a single domain."""
    # Check if training script exists
    train_script = PROJECT / "experiments" / "train_missing_adapters.py"
    if not train_script.exists():
        # Use generic training
        train_script = PROJECT / "training" / "scripts" / "train_staged_adapter.py"

    cmd = [
        PYTHON, str(train_script),
        "--facts", str(facts_file),
        "--domain", domain,
        "--output", str(ADAPTERS_DIR / f"{domain}_adapter.npz")
    ]

    return run_job(f"train_{domain}", cmd, cwd=str(PROJECT))

def main():
    missing = get_domains_needing_adapters()
    print(f"=== Training Adapters for {len(missing)} Domains ===")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    for i, (domain, facts_file) in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] {domain}")
        # Check facts file has content
        with open(facts_file) as f:
            data = json.load(f)
            n_facts = len(data.get("facts", data.get("verifications", [])))
        print(f"  Facts: {n_facts}")

        if n_facts < 3:
            print(f"  Skipping (too few facts)")
            continue

        result = train_adapter_for_domain(domain, facts_file)
        if result["exit_code"] != 0:
            print(f"  Failed: {result['stderr'][:200]}")
        else:
            print(f"  Done")

    print(f"\n=== Completed: {datetime.now().isoformat()} ===")

if __name__ == "__main__":
    main()
