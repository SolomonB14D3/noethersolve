#!/usr/bin/env python3
"""
research_runner.py — Persistent autonomous research runner for the 27B model.

Continuously runs oracle evaluation across all domains, tracks progress
between sweeps, and stops when no improvement is being made.

Usage:
    python scripts/research_runner.py           # Continuous mode (polls for new work)
    python scripts/research_runner.py --once    # Single sweep, then exit
    python scripts/research_runner.py --status  # Show current state
    python scripts/research_runner.py --domain bio_ai  # Run specific domain

Results written to:
    results/research_status.json  — current state (what's running, what's done)
    results/research_log.txt      — append-only log of all runs
    results/run_summary.json      — sweep-over-sweep progress tracking
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Set environment before any model imports
os.environ.setdefault("HF_HOME", "/Volumes/4TB SD/ml_cache/huggingface")

_HERE = Path(__file__).parent.parent
sys.path.insert(0, str(_HERE))

STATUS_FILE = _HERE / "results" / "research_status.json"
LOG_FILE = _HERE / "results" / "research_log.txt"
ESCALATION_FILE = _HERE / "results" / "escalations.jsonl"
SUMMARY_FILE = _HERE / "results" / "run_summary.json"
PROBLEMS_DIR = _HERE / "problems"

# Default model for all oracle runs
DEFAULT_MODEL = "mlx-community/Qwen3.5-27B-4bit"

# Polling
POLL_INTERVAL = 300  # 5 minutes between sweeps
MAX_IDLE_POLLS = 3   # Stop after 3 consecutive sweeps with zero progress

# Escalation thresholds
ESCALATE_MARGIN_THRESHOLD = -20.0
ESCALATE_REGRESSION_THRESHOLD = -5.0
ESCALATE_PASS_RATE_THRESHOLD = 0.1


def log(msg: str):
    """Append to log file and print."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Status persistence
# ---------------------------------------------------------------------------

def load_status() -> dict:
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {
        "current_domain": None,
        "current_phase": "idle",
        "completed_domains": [],
        "domain_results": {},
        "last_update": None,
        "pid": None,
    }


def save_status(status: dict):
    status["last_update"] = datetime.now().isoformat()
    status["pid"] = os.getpid()
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)


# ---------------------------------------------------------------------------
# Escalation system
# ---------------------------------------------------------------------------

def _v2_exists_for_domain(domain: str) -> bool:
    """Check if a v2 facts file already exists for this domain."""
    domain_snake = domain.lower().replace(" ", "_").replace("-", "_")
    for suffix in ["_facts_v2.json", "_v2.json"]:
        if (PROBLEMS_DIR / f"{domain_snake}{suffix}").exists():
            return True
    return False


def _has_open_escalation(domain: str, reason: str) -> bool:
    if not ESCALATION_FILE.exists():
        return False
    try:
        with open(ESCALATION_FILE) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if (e.get("status") == "open"
                            and e.get("domain") == domain
                            and e.get("reason") == reason):
                        return True
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return False


def escalate(domain: str, reason: str, details: dict):
    """Write an escalation for Claude to pick up."""
    if _v2_exists_for_domain(domain):
        log(f"  V2 facts file exists for {domain} — no escalation needed")
        return
    if _has_open_escalation(domain, reason):
        log(f"  Escalation already open for {domain}/{reason} — skipping")
        return

    entry = {
        "domain": domain,
        "reason": reason,
        "details": details,
        "timestamp": datetime.now().isoformat(),
        "status": "open",
    }
    ESCALATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ESCALATION_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log(f"  ESCALATED: {reason}")


def check_escalation(domain_name: str, result: dict, phase: str = "baseline"):
    """Only escalate when baseline shows extreme failure suggesting fact quality issues."""
    margin = result.get("mean_margin", 0)
    pass_rate = result.get("pass_rate", 0)
    n_total = result.get("n_total", 0)

    # Total failure with very negative margins suggests fact quality issues
    if pass_rate == 0.0 and n_total >= 8 and margin < -15:
        escalate(domain_name, "fact_quality_suspect", {
            "pass_rate": pass_rate,
            "mean_margin": margin,
            "n_total": n_total,
            "suggestion": "Zero pass rate with deep negative margins — likely fact "
                          "quality issue (length ratio, certainty contamination). "
                          "Run audit_facts.py to diagnose.",
        })


def get_open_escalations() -> list[dict]:
    if not ESCALATION_FILE.exists():
        return []
    escalations = []
    with open(ESCALATION_FILE) as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("status") == "open":
                    escalations.append(e)
            except json.JSONDecodeError:
                pass
    return escalations


def resolve_escalation(domain: str, resolution: str = "addressed"):
    if not ESCALATION_FILE.exists():
        return
    lines = []
    with open(ESCALATION_FILE) as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("domain") == domain and e.get("status") == "open":
                    e["status"] = "resolved"
                    e["resolution"] = resolution
                    e["resolved_at"] = datetime.now().isoformat()
                lines.append(json.dumps(e))
            except json.JSONDecodeError:
                lines.append(line.rstrip())
    with open(ESCALATION_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Domain discovery
# ---------------------------------------------------------------------------

def discover_domains() -> list[dict]:
    """Find all problem YAMLs with verification sets.

    When both v1 and v2 exist for a domain, prefer v2 and skip v1.
    Includes file modification time for invalidation detection.
    """
    import yaml

    domains = []
    # Track v2 base stems to skip v1 when v2 exists
    v2_base_stems = set()
    for yaml_path in sorted(PROBLEMS_DIR.glob("*_v2.yaml")):
        stem = yaml_path.stem.replace("_v2", "")
        v2_base_stems.add(stem)

    for yaml_path in sorted(PROBLEMS_DIR.glob("*.yaml")):
        try:
            stem = yaml_path.stem
            # Skip v1 if v2 exists
            if stem in v2_base_stems and "_v2" not in stem:
                continue

            with open(yaml_path) as f:
                prob = yaml.safe_load(f)
            vs = prob.get("verification_set") or prob.get("facts_file")
            if not vs:
                continue
            vs_path = yaml_path.parent / vs
            if not vs_path.exists():
                continue
            with open(vs_path) as f:
                data = json.load(f)
            n_facts = len(data) if isinstance(data, list) else len(data.get("facts", []))

            # Track file modification time for invalidation
            facts_mtime = vs_path.stat().st_mtime

            domains.append({
                "name": prob.get("name", yaml_path.stem),
                "yaml_path": str(yaml_path),
                "facts_path": str(vs_path),
                "n_facts": n_facts,
                "pass_threshold": prob.get("pass_threshold", 0.5),
                "facts_mtime": facts_mtime,
            })
        except Exception:
            continue
    return domains


# ---------------------------------------------------------------------------
# Oracle evaluation
# ---------------------------------------------------------------------------

def run_oracle_eval(yaml_path: str, model_name: str = DEFAULT_MODEL) -> dict:
    """Run oracle evaluation via oracle_wrapper.py subprocess."""
    import re
    import subprocess
    import yaml as yaml_mod

    with open(yaml_path) as f:
        problem = yaml_mod.safe_load(f)

    threshold = float(problem.get("pass_threshold", 0.5))

    log(f"  Running oracle_wrapper.py with model {model_name}")
    cmd = [
        sys.executable, str(_HERE / "oracle_wrapper.py"),
        "--problem", yaml_path,
        "--model", model_name,
        "--diagnose",
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            env={**os.environ, "HF_HOME": os.environ.get("HF_HOME", "")},
        )
        elapsed = time.time() - t0
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        log(f"  TIMEOUT after {elapsed:.0f}s — skipping domain")
        return {
            "domain": problem.get("name", Path(yaml_path).stem),
            "yaml_path": yaml_path,
            "n_pass": 0, "n_total": 0, "pass_rate": 0,
            "mean_margin": -999, "threshold": threshold,
            "verdict": "ERROR", "failures": [],
            "timestamp": datetime.now().isoformat(),
            "model": model_name, "error": "timeout",
        }

    # Parse results
    pass_match = re.search(r'Pass rate:\s+(\d+)/(\d+)', output)
    margin_match = re.search(r'Mean margin:\s+([-+]?\d+\.?\d*)', output)

    n_pass = int(pass_match.group(1)) if pass_match else 0
    n_total = int(pass_match.group(2)) if pass_match else 0
    mean_margin = float(margin_match.group(1)) if margin_match else -999
    pass_rate = n_pass / n_total if n_total else 0

    # Parse individual failures
    failures = []
    for m in re.finditer(r"'([^']+)' → '([^']+)'\s+margin=([-+]?\d+\.?\d*)", output):
        failures.append({
            "id": m.group(1)[:60],
            "margin": float(m.group(3)),
        })
    failures.sort(key=lambda f: f["margin"])

    log(f"  Oracle completed in {elapsed:.0f}s")

    return {
        "domain": problem.get("name", Path(yaml_path).stem),
        "yaml_path": yaml_path,
        "n_pass": n_pass,
        "n_total": n_total,
        "pass_rate": pass_rate,
        "mean_margin": mean_margin,
        "threshold": threshold,
        "verdict": "PASS" if pass_rate >= threshold else "FAIL",
        "failures": failures[:5],
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
    }


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------

def pick_next_domain(status: dict, all_domains: list) -> dict | None:
    """Pick next domain to evaluate.

    Priority:
    1. Brand new domains (never evaluated)
    2. Domains whose facts file was modified after last evaluation (invalidated)
    3. None — all domains are current
    """
    completed = set(status.get("completed_domains", []))
    results = status.get("domain_results", {})

    # 1. Untested domains
    untested = [d for d in all_domains if d["name"] not in completed]
    if untested:
        untested.sort(key=lambda d: d["n_facts"], reverse=True)
        return untested[0]

    # 2. Invalidated domains (facts file newer than last result)
    for d in all_domains:
        prev = results.get(d["name"], {})
        prev_ts = prev.get("timestamp")
        if prev_ts and d.get("facts_mtime"):
            from datetime import datetime as dt
            try:
                prev_time = dt.fromisoformat(prev_ts).timestamp()
                if d["facts_mtime"] > prev_time:
                    log(f"  {d['name']}: facts updated since last eval — re-running")
                    # Remove from completed so it gets re-evaluated
                    if d["name"] in completed:
                        status["completed_domains"].remove(d["name"])
                    return d
            except (ValueError, TypeError):
                pass

    return None


# ---------------------------------------------------------------------------
# Run a single domain
# ---------------------------------------------------------------------------

def run_domain(domain: dict, status: dict) -> dict:
    """Run oracle evaluation on one domain."""
    name = domain["name"]
    yaml_path = domain["yaml_path"]

    log(f"\n{'='*60}")
    log(f"DOMAIN: {name}")
    log(f"  Facts: {domain['n_facts']}, Threshold: {domain['pass_threshold']}")
    log(f"{'='*60}")

    status["current_domain"] = name
    status["current_phase"] = "oracle_eval"
    save_status(status)

    # Run oracle
    log("Phase 1: Baseline oracle evaluation")
    result = run_oracle_eval(yaml_path)

    log(f"  Result: {result['verdict']} — {result['n_pass']}/{result['n_total']} "
        f"({result['pass_rate']:.0%}), mean margin: {result['mean_margin']:.2f}")

    if result["failures"]:
        log(f"  Worst gaps:")
        for f in result["failures"][:3]:
            log(f"    {f['id']}: margin={f['margin']:.2f}")

    # Record result
    baseline_result = {
        "verdict": result["verdict"],
        "pass_rate": result["pass_rate"],
        "n_pass": result["n_pass"],
        "n_total": result["n_total"],
        "mean_margin": result["mean_margin"],
        "timestamp": result["timestamp"],
        "phase": "baseline_complete",
    }
    status["domain_results"][name] = baseline_result

    # Log failures for 4B adapter pipeline
    if result["verdict"] == "FAIL" and result["failures"]:
        n_fail = result["n_total"] - result["n_pass"]
        log(f"  {n_fail} facts failed — recorded for 4B adapter pipeline")
        baseline_result["n_failures"] = n_fail

    # Check for fact quality escalations
    check_escalation(name, baseline_result, phase="baseline")

    # Mark complete
    if name not in status.get("completed_domains", []):
        status.setdefault("completed_domains", []).append(name)
    status["current_phase"] = "idle"
    save_status(status)

    return result


# ---------------------------------------------------------------------------
# Sweep summary + progress tracking
# ---------------------------------------------------------------------------

def load_summary() -> dict:
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE) as f:
            return json.load(f)
    return {"sweeps": []}


def save_sweep_summary(status: dict, sweep_start: str, duration: float,
                       domains_this_sweep: list[str]):
    """Record this sweep's results for progress tracking."""
    summary = load_summary()
    results = status.get("domain_results", {})

    # Count current state
    n_pass = sum(1 for v in results.values()
                 if v.get("verdict") == "PASS" and v.get("n_total", 0) > 0)
    n_fail = sum(1 for v in results.values()
                 if v.get("verdict") == "FAIL" and v.get("n_total", 0) > 0)
    n_error = sum(1 for v in results.values()
                  if v.get("verdict") == "ERROR")
    n_total = len([v for v in results.values() if v.get("n_total", 0) > 0])

    # Compare with last sweep
    prev_sweep = summary["sweeps"][-1] if summary["sweeps"] else None
    prev_pass = prev_sweep["domains_passed"] if prev_sweep else 0

    # Track what improved
    improved = {}
    for name in domains_this_sweep:
        r = results.get(name, {})
        if prev_sweep and name in prev_sweep.get("improved", {}):
            continue  # Already tracked
        # Check if this is new or improved
        old_rate = None
        if prev_sweep:
            for s in reversed(summary["sweeps"]):
                if name in s.get("domain_pass_rates", {}):
                    old_rate = s["domain_pass_rates"][name]
                    break
        new_rate = r.get("pass_rate", 0)
        if old_rate is None or new_rate > old_rate:
            improved[name] = {"old": old_rate, "new": new_rate}

    sweep = {
        "timestamp": sweep_start,
        "duration_seconds": round(duration),
        "domains_evaluated": len(domains_this_sweep),
        "domains_passed": n_pass,
        "domains_failed": n_fail,
        "domains_error": n_error,
        "total_domains": n_total,
        "new_domains": [d for d in domains_this_sweep
                        if not any(d in s.get("domain_pass_rates", {})
                                   for s in summary["sweeps"])],
        "improved": improved,
        "progress_delta": n_pass - prev_pass,
        "domain_pass_rates": {name: results[name].get("pass_rate", 0)
                              for name in results if results[name].get("n_total", 0) > 0},
    }

    summary["sweeps"].append(sweep)
    # Keep last 50 sweeps
    summary["sweeps"] = summary["sweeps"][-50:]
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    return sweep


def log_sweep_report(sweep: dict):
    """Print a human-readable sweep summary."""
    log(f"\n{'='*60}")
    log(f"SWEEP COMPLETE")
    log(f"{'='*60}")
    log(f"  Evaluated:  {sweep['domains_evaluated']} domains")
    log(f"  Passing:    {sweep['domains_passed']}/{sweep['total_domains']} "
        f"({100*sweep['domains_passed']/max(sweep['total_domains'],1):.0f}%)")
    log(f"  Progress:   {sweep['progress_delta']:+d} domains since last sweep")
    log(f"  Duration:   {sweep['duration_seconds']}s")

    if sweep["new_domains"]:
        log(f"  New domains: {', '.join(sweep['new_domains'][:5])}")
    if sweep["improved"]:
        log(f"  Improved:")
        for name, delta in list(sweep["improved"].items())[:5]:
            old = f"{delta['old']:.0%}" if delta['old'] is not None else "new"
            log(f"    {name}: {old} → {delta['new']:.0%}")

    if sweep["progress_delta"] == 0 and sweep["domains_evaluated"] == 0:
        log(f"  No new work found.")
    elif sweep["progress_delta"] == 0:
        log(f"  No improvement this sweep.")
    log("")


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def print_status(status: dict):
    print(f"\n{'='*60}")
    print(f"  NoetherSolve Research Status")
    print(f"{'='*60}")
    print(f"  Last update: {status.get('last_update', 'never')}")
    print(f"  PID:         {status.get('pid', 'none')}")
    print(f"  Phase:       {status.get('current_phase', 'idle')}")
    print(f"  Domain:      {status.get('current_domain', 'none')}")
    print(f"  Completed:   {len(status.get('completed_domains', []))}")
    print()

    escalations = get_open_escalations()
    if escalations:
        print(f"  *** {len(escalations)} ESCALATION(S) ***")
        for e in escalations:
            print(f"  [{e['domain']}] {e['reason']}: "
                  f"{e['details'].get('suggestion', '')[:80]}")
        print()

    results = status.get("domain_results", {})
    if results:
        print(f"  {'Domain':<45} {'Result':<8} {'Pass Rate':<12} {'Margin':<10}")
        print(f"  {'-'*45} {'-'*8} {'-'*12} {'-'*10}")
        for name, r in sorted(results.items()):
            if r.get("n_total", 0) == 0:
                continue
            verdict = r.get("verdict", "?")
            pr = r.get("pass_rate", 0)
            mm = r.get("mean_margin", 0)
            print(f"  {name:<45} {verdict:<8} {pr:>6.0%}       {mm:>+8.2f}")

    # Show progress trend
    summary = load_summary()
    if len(summary.get("sweeps", [])) >= 2:
        recent = summary["sweeps"][-3:]
        print(f"\n  Recent sweeps:")
        for s in recent:
            print(f"    {s['timestamp'][:16]}: {s['domains_passed']}/{s['total_domains']} pass "
                  f"({s['progress_delta']:+d}), {s['domains_evaluated']} evaluated")
    print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autonomous 27B oracle research runner")
    parser.add_argument("--domain", default=None, help="Run specific domain only")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--once", action="store_true", help="Run one sweep and exit")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--max-domains", type=int, default=0,
                        help="Max domains to process (0=all)")
    parser.add_argument("--resolve", default=None,
                        help="Resolve an escalation by domain name")
    parser.add_argument("--escalations", action="store_true",
                        help="Show open escalations and exit")
    args = parser.parse_args()

    if args.resolve:
        resolve_escalation(args.resolve)
        print(f"Resolved escalation for: {args.resolve}")
        return

    if args.escalations:
        esc = get_open_escalations()
        if esc:
            print(f"\n  {len(esc)} open escalation(s):\n")
            for e in esc:
                print(f"  [{e['domain']}] {e['reason']}")
                print(f"    {e['details'].get('suggestion', '')}")
                print()
        else:
            print("  No open escalations.")
        return

    status = load_status()

    if args.status:
        print_status(status)
        return

    # Single domain mode
    if args.domain:
        all_domains = discover_domains()
        matching = [d for d in all_domains if args.domain in d["name"].lower()
                    or args.domain in d["yaml_path"]]
        if not matching:
            log(f"ERROR: No domain matching '{args.domain}'")
            return
        run_domain(matching[0], status)
        print_status(status)
        return

    # Continuous mode
    idle_polls = 0
    while True:
        sweep_start = datetime.now().isoformat()
        t0 = time.time()
        domains_this_sweep = []

        all_domains = discover_domains()
        log(f"Discovered {len(all_domains)} domains")

        domains_done = 0
        while True:
            domain = pick_next_domain(status, all_domains)
            if domain is None:
                break

            try:
                run_domain(domain, status)
                domains_this_sweep.append(domain["name"])
            except Exception as e:
                log(f"ERROR on {domain['name']}: {e}")
                traceback.print_exc()
                # Mark as complete so we don't get stuck
                status.setdefault("completed_domains", []).append(domain["name"])
                status["domain_results"][domain["name"]] = {
                    "verdict": "ERROR",
                    "error": str(e)[:200],
                    "n_total": 0, "n_pass": 0,
                    "timestamp": datetime.now().isoformat(),
                }
                save_status(status)
                domains_this_sweep.append(domain["name"])

            domains_done += 1
            if args.max_domains and domains_done >= args.max_domains:
                log(f"Reached max-domains limit ({args.max_domains})")
                break

        # Sweep complete — record and report
        duration = time.time() - t0
        sweep = save_sweep_summary(status, sweep_start, duration, domains_this_sweep)
        log_sweep_report(sweep)

        # Decide whether to continue
        if args.once or args.max_domains:
            break

        # Progress-aware stopping
        if sweep["domains_evaluated"] == 0:
            idle_polls += 1
            if idle_polls >= MAX_IDLE_POLLS:
                log(f"No new work for {idle_polls} consecutive polls — stopping.")
                log(f"Restart with new V2 files to resume.")
                break
            log(f"Polling for new work in {POLL_INTERVAL}s... "
                f"({idle_polls}/{MAX_IDLE_POLLS} idle polls)")
            time.sleep(POLL_INTERVAL)
        else:
            idle_polls = 0  # Reset on any work done
            # Brief pause between sweeps
            time.sleep(5)

    print_status(status)
    log("Research runner finished.")


if __name__ == "__main__":
    main()
