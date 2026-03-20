#!/usr/bin/env python3
"""
Adapter Trainer — Trains 4B adapters on failing domains.

This is the 27B's MAIN JOB after evaluation is complete.
Uses the escalation ladder: single-pass → staged → orthogonal → joint.

Usage:
    python scripts/adapter_trainer.py           # Train all failing domains
    python scripts/adapter_trainer.py --once    # Train one domain and exit
    python scripts/adapter_trainer.py --status  # Show what needs training
    python scripts/adapter_trainer.py --domain knot_invariants  # Specific domain
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).parent.parent
ADAPTERS_DIR = PROJECT / "adapters"
PROBLEMS_DIR = PROJECT / "problems"
RESULTS_DIR = PROJECT / "results"
PYTHON = sys.executable

# 4B is the student — always train on this model
TRAIN_MODEL = "Qwen/Qwen3-4B-Base"

# 27B is for oracle evaluation only (post-training verification)
EVAL_MODEL = "mlx-community/Qwen3.5-27B-4bit"


def load_run_summary():
    """Load domain pass rates from the latest sweep."""
    summary_path = RESULTS_DIR / "run_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        data = json.load(f)
    # Get latest sweep with pass rates
    for sweep in reversed(data.get("sweeps", [])):
        if sweep.get("domain_pass_rates"):
            return sweep["domain_pass_rates"]
    return {}


def normalize_domain_name(name):
    """Normalize domain name to snake_case for file matching.

    Handles: 'Intersection Theory' → 'intersection_theory'
             'NS Regularity and Stretch-Resistant Q_f' → 'ns_regularity_and_stretch_resistant_q_f'
             'knot_invariants_v2' → 'knot_invariants'
    """
    base = name.replace(" V2", "").replace("_v2", "")
    # Convert title case / spaces to snake_case
    base = base.replace(" ", "_").lower()
    # Remove special chars except underscore
    base = "".join(c for c in base if c.isalnum() or c == "_")
    return base


# Explicit mappings for domain names that can't be fuzzy-matched
# Keys: normalized domain names from run_summary.json
# Values: facts file stems (without _facts.json suffix)
DOMAIN_TO_FACTS = {
    "q_f_ratio_invariant": "qf_ratio",
    "qf_ratio_invariant": "qf_ratio",
    "electromagnetic_zilch_and_optical_chirality": "em_zilch",
    "continuous_q_f_and_euler_conservation_laws": "continuous_qf",
    "continuous_qf_and_euler_conservation_laws": "continuous_qf",
    "bioai_computational_parallels": "bio_ai_parallels",
    "bio_ai_computational_parallels": "bio_ai_parallels",
    "bio-ai_computational_parallels": "bio_ai_parallels",
    "reduced_navier_stokes_vortex_conservation": "vortex_pair",
    "reduced_navier_stokes_vortex_conservation_unsolved": "vortex_pair",
    "optimal_fr_combination": "optimal_f",
    "optimal_f_r_combination": "optimal_f",
}


def get_failing_domains():
    """Get domains that are failing (pass_rate < 0.5).

    For domains with both V1 and V2, only consider the best version.
    Returns list of (domain_name, pass_rate, facts_file, yaml_file).
    """
    rates = load_run_summary()
    if not rates:
        print("No run_summary.json found. Run research_runner.py --once first.")
        return []

    # Group by normalized base domain, keep best version
    best = {}
    for name, rate in rates.items():
        base = normalize_domain_name(name)
        if base not in best or rate > best[base][1]:
            best[base] = (name, rate)

    # Build a map of all facts files by their problem name
    facts_map = {}  # normalized_name → (facts_file, yaml_file)
    for ff in sorted(PROBLEMS_DIR.glob("*_facts*.json"), reverse=True):  # V2 files sort later
        try:
            with open(ff) as f:
                data = json.load(f)
            prob_name = data.get("problem", ff.stem.replace("_facts", ""))
            norm = normalize_domain_name(prob_name)
            # Also try the filename-based name
            file_norm = normalize_domain_name(ff.stem.replace("_facts", "").replace("_facts_v2", ""))
            # Find matching yaml
            yaml_candidates = [
                ff.parent / f"{ff.stem.replace('_facts', '').replace('_facts_v2', '_v2')}.yaml",
                ff.parent / f"{prob_name}.yaml",
            ]
            yaml_file = None
            for yc in yaml_candidates:
                if yc.exists():
                    yaml_file = yc
                    break

            # Store both normalized forms
            for n in [norm, file_norm]:
                if n not in facts_map or "_v2" in ff.stem:
                    facts_map[n] = (ff, yaml_file)
        except (json.JSONDecodeError, KeyError):
            continue

    failing = []
    for base, (name, rate) in sorted(best.items(), key=lambda x: x[1][1]):
        if rate >= 0.5:
            continue

        facts_file = None
        yaml_file = None

        # Try direct match in facts_map
        if base in facts_map:
            facts_file, yaml_file = facts_map[base]
        elif base in DOMAIN_TO_FACTS:
            # Use explicit mapping
            mapped = DOMAIN_TO_FACTS[base]
            if mapped in facts_map:
                facts_file, yaml_file = facts_map[mapped]
            else:
                # Try finding the facts file directly
                for suffix in ["_facts_v2.json", "_facts.json"]:
                    candidate = PROBLEMS_DIR / f"{mapped}{suffix}"
                    if candidate.exists():
                        # Find yaml too
                        yaml_candidates = [
                            PROBLEMS_DIR / f"{mapped}_v2.yaml",
                            PROBLEMS_DIR / f"{mapped}.yaml",
                        ]
                        yf = None
                        for yc in yaml_candidates:
                            if yc.exists():
                                yf = yc
                                break
                        facts_file, yaml_file = candidate, yf
                        break
        else:
            # Try fuzzy match — find closest key
            for key in facts_map:
                if base in key or key in base:
                    facts_file, yaml_file = facts_map[key]
                    break

        if facts_file is not None and facts_file.exists():
            failing.append((name, rate, facts_file, yaml_file if yaml_file and yaml_file.exists() else None))
        else:
            # No facts file found — still report it but can't train
            failing.append((name, rate, None, None))

    return failing


def get_adapter_status(domain_name):
    """Check what adapters exist for a domain."""
    base = normalize_domain_name(domain_name)

    # Try multiple matching strategies
    adapters = []
    for pattern in [f"{base}*.npz", f"{base}*.safetensors"]:
        adapters.extend(ADAPTERS_DIR.glob(pattern))

    # Also try shorter prefix matches (e.g., "chemical" for "chemical_reaction_network_conservation")
    if not adapters:
        # Try first word
        prefix = base.split("_")[0]
        if len(prefix) >= 4:  # Avoid too-short matches
            for pattern in [f"{prefix}_*.npz"]:
                adapters.extend(ADAPTERS_DIR.glob(pattern))

    # Check for orthogonal routing config
    routing = ADAPTERS_DIR / f"{base}_orthogonal_routing.json"

    return {
        "adapters": [str(a.name) for a in adapters],
        "has_single": any("adapter" in a.name and "orthogonal" not in a.name for a in adapters),
        "has_orthogonal": routing.exists() or any("orthogonal" in a.name for a in adapters),
        "has_staged": any("stage" in a.name for a in adapters),
        "routing_config": str(routing) if routing.exists() else None,
    }


def train_single_adapter(domain_name, facts_file, output_path=None):
    """Train a single-pass adapter for a domain (escalation level 1)."""
    base = domain_name.replace("_v2", "").replace(" V2", "").replace(" ", "_").lower()
    if output_path is None:
        output_path = ADAPTERS_DIR / f"{base}_adapter.npz"

    cmd = [
        PYTHON, str(PROJECT / "scripts" / "train_from_facts.py"),
        "--facts", str(facts_file),
        "--model", TRAIN_MODEL,
        "--output", str(output_path),
        "--steps", "500",
    ]

    print(f"  Training single-pass adapter: {output_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(PROJECT))

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:300]}")
        return False

    print(f"  Done: {output_path.name}")
    return True


def escalate_training(domain_name, facts_file, current_status):
    """Escalate training technique based on what's already been tried."""

    if not current_status["has_single"]:
        # Level 1: Single-pass
        print(f"  Escalation level 1: single-pass adapter")
        return train_single_adapter(domain_name, facts_file)

    elif not current_status["has_staged"]:
        # Level 2: Staged training
        print(f"  Escalation level 2: staged training (via train_with_proven_methods)")
        base = domain_name.replace("_v2", "").replace(" V2", "").replace(" ", "_").lower()
        yaml_candidates = list(PROBLEMS_DIR.glob(f"{base}*.yaml"))
        if not yaml_candidates:
            print(f"  No YAML found for {domain_name}, falling back to single-pass")
            return train_single_adapter(domain_name, facts_file)

        cmd = [
            PYTHON, str(PROJECT / "scripts" / "train_with_proven_methods.py"),
            "--domain", base,
            "--facts", str(facts_file),
            "--yaml", str(yaml_candidates[0]),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=str(PROJECT))

        if result.returncode != 0:
            print(f"  Staged training failed: {result.stderr[:300]}")
            return False
        print(f"  Staged training complete")
        return True

    elif not current_status["has_orthogonal"]:
        # Level 3: Orthogonal adapters
        print(f"  Escalation level 3: orthogonal adapters needed")
        print(f"  NOTE: Orthogonal adapter creation requires Claude Code analysis.")
        print(f"  Log this as an escalation for Claude Code to handle.")
        log_escalation(domain_name, "needs_orthogonal",
                       "Single-pass and staged training insufficient. "
                       "Needs orthogonal cluster adapters — requires Claude Code to analyze "
                       "fact clusters and create routing config.")
        return False

    else:
        print(f"  All escalation levels attempted. Needs manual investigation.")
        log_escalation(domain_name, "all_levels_exhausted",
                       "All automated escalation levels tried. "
                       "Needs Claude Code to investigate and design custom approach.")
        return False


def log_escalation(domain_name, reason, details):
    """Log an escalation for Claude Code to handle."""
    escalation = {
        "domain": domain_name,
        "reason": reason,
        "details": details,
        "timestamp": datetime.now().isoformat(),
        "status": "open",
    }
    esc_path = RESULTS_DIR / "escalations.jsonl"
    with open(esc_path, "a") as f:
        f.write(json.dumps(escalation) + "\n")
    print(f"  Escalation logged: {reason}")


def evaluate_after_training(yaml_file, adapter_path=None):
    """Re-evaluate domain on 27B after training to measure improvement."""
    if yaml_file is None:
        return None

    cmd = [
        PYTHON, str(PROJECT / "oracle_wrapper.py"),
        "--problem", str(yaml_file),
        "--model", EVAL_MODEL,
    ]
    if adapter_path:
        cmd.extend(["--adapter", str(adapter_path)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(PROJECT))
        import re
        pass_matches = re.findall(r'Pass rate:\s+(\d+)/(\d+)', result.stdout + result.stderr)
        if pass_matches:
            passed, total = int(pass_matches[-1][0]), int(pass_matches[-1][1])
            return passed / total if total > 0 else 0
    except Exception as e:
        print(f"  Evaluation error: {e}")

    return None


def print_status():
    """Show failing domains and their adapter training status."""
    failing = get_failing_domains()

    if not failing:
        print("All domains passing or no run_summary.json found.")
        return

    print(f"{'='*70}")
    print(f"  Adapter Training Status — {len(failing)} Failing Domains")
    print(f"{'='*70}")
    print()

    for name, rate, facts_file, yaml_file in failing:
        status = get_adapter_status(name)
        adapters_str = ", ".join(status["adapters"][:3]) if status["adapters"] else "NONE"
        level = "none"
        if status["has_orthogonal"]:
            level = "orthogonal (L3)"
        elif status["has_staged"]:
            level = "staged (L2)"
        elif status["has_single"]:
            level = "single (L1)"

        facts_str = facts_file.name if facts_file else "NOT FOUND"
        print(f"  {rate:5.0%}  {name}")
        print(f"         Adapters: {adapters_str}")
        print(f"         Level: {level}")
        print(f"         Facts: {facts_str}")
        print()


def save_training_log(entry):
    """Append a training result to the log."""
    log_path = RESULTS_DIR / "adapter_training_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_work_loop(domain_filter=None):
    """Continuous work loop: train → verify → escalate → next domain → repeat.

    Keeps working until:
    - All domains have been attempted at their current escalation level
    - No more trainable domains remain (all escalated to Claude Code level)
    """
    print(f"{'='*70}")
    print(f"  Adapter Trainer — Continuous Work Loop")
    print(f"  Training model: {TRAIN_MODEL} (4B student)")
    print(f"  Eval model: {EVAL_MODEL} (27B judge)")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'='*70}")
    print()

    total_trained = 0
    total_improved = 0
    total_escalated = 0
    total_skipped = 0
    loop_count = 0

    while True:
        loop_count += 1
        print(f"\n--- Sweep {loop_count} ---")

        failing = get_failing_domains()

        if domain_filter:
            failing = [(n, r, f, y) for n, r, f, y in failing
                        if domain_filter.lower() in n.lower()]

        # Filter to only domains we can actually work on (have facts files)
        trainable = [(n, r, f, y) for n, r, f, y in failing if f is not None]
        no_facts = [(n, r, f, y) for n, r, f, y in failing if f is None]

        if no_facts:
            print(f"  {len(no_facts)} domains without facts files (need Claude Code):")
            for name, rate, _, _ in no_facts[:5]:
                print(f"    {rate:5.0%}  {name}")

        if not trainable:
            print(f"\n  No trainable domains remaining.")
            if no_facts:
                print(f"  {len(no_facts)} domains need facts files created by Claude Code.")
            print(f"  Work loop complete.")
            break

        print(f"  {len(trainable)} trainable domains this sweep")
        print()

        sweep_trained = 0
        sweep_improved = 0

        for i, (name, rate, facts_file, yaml_file) in enumerate(trainable, 1):
            print(f"[{i}/{len(trainable)}] {name} (current: {rate:.0%})")

            status = get_adapter_status(name)

            # Train / escalate
            t0 = time.time()
            success = escalate_training(name, facts_file, status)
            duration = time.time() - t0

            if success:
                sweep_trained += 1
                total_trained += 1

                # Re-evaluate on 27B to measure improvement
                if yaml_file is not None:
                    print(f"  Verifying improvement on 27B...")
                    new_rate = evaluate_after_training(yaml_file)
                    if new_rate is not None:
                        delta = new_rate - rate
                        improved = delta > 0
                        print(f"  Result: {rate:.0%} → {new_rate:.0%} ({'+' if delta >= 0 else ''}{delta:.0%})")
                        if improved:
                            sweep_improved += 1
                            total_improved += 1

                        save_training_log({
                            "domain": name,
                            "old_rate": rate,
                            "new_rate": new_rate,
                            "delta": delta,
                            "level": "staged" if status["has_single"] else "single",
                            "duration_seconds": duration,
                            "timestamp": datetime.now().isoformat(),
                        })
                    else:
                        print(f"  Evaluation failed — cannot verify improvement")
            else:
                total_escalated += 1

            print()

        # Sweep summary
        print(f"--- Sweep {loop_count} Summary ---")
        print(f"  Trained: {sweep_trained}, Improved: {sweep_improved}")
        print(f"  Total so far: {total_trained} trained, {total_improved} improved, {total_escalated} escalated")

        # Stop if nothing was trained this sweep (everything escalated or failed)
        if sweep_trained == 0:
            print(f"\n  No successful training this sweep. Stopping.")
            break

        # After a full sweep, check if improvement happened
        # If yes, loop again — improved adapters may help other domains
        if sweep_improved > 0:
            print(f"\n  {sweep_improved} domains improved — running another sweep...")
        else:
            print(f"\n  No improvement this sweep despite training. Stopping.")
            break

    print(f"\n{'='*70}")
    print(f"  Work Loop Complete")
    print(f"  Sweeps: {loop_count}")
    print(f"  Trained: {total_trained}, Improved: {total_improved}")
    print(f"  Escalated to Claude Code: {total_escalated}")
    print(f"  Skipped (no facts): {total_skipped}")
    print(f"  Finished: {datetime.now().isoformat()}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Train 4B adapters on failing domains")
    parser.add_argument("--status", action="store_true", help="Show training status only")
    parser.add_argument("--once", action="store_true", help="Train one domain and exit")
    parser.add_argument("--domain", help="Train specific domain")
    args = parser.parse_args()

    os.makedirs(ADAPTERS_DIR, exist_ok=True)

    if args.status:
        print_status()
        return

    if args.once:
        # Single domain mode — train one and exit
        failing = get_failing_domains()
        if args.domain:
            failing = [(n, r, f, y) for n, r, f, y in failing
                        if args.domain.lower() in n.lower()]
        trainable = [(n, r, f, y) for n, r, f, y in failing if f is not None]

        if not trainable:
            print("No trainable domains found.")
            return

        name, rate, facts_file, yaml_file = trainable[0]
        print(f"Training: {name} (current: {rate:.0%})")
        status = get_adapter_status(name)
        success = escalate_training(name, facts_file, status)

        if success and yaml_file:
            print(f"Verifying on 27B...")
            new_rate = evaluate_after_training(yaml_file)
            if new_rate is not None:
                print(f"Result: {rate:.0%} → {new_rate:.0%}")
        return

    # Default: continuous work loop
    run_work_loop(domain_filter=args.domain)


if __name__ == "__main__":
    main()
