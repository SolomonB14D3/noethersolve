#!/usr/bin/env python3
"""
Job duration tracker with feedback loop.
Measures actual runtime, stores history, predicts future durations.
Factors in hardware profile for cross-machine estimates.
"""
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

HISTORY_FILE = Path(__file__).parent / "job_history.json"
HARDWARE_FILE = Path(__file__).parent / "hardware_profile.json"

def load_history() -> dict:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return {"jobs": {}, "machine_id": get_machine_id()}

def get_machine_id() -> str:
    """Get unique machine identifier from hardware profile."""
    if HARDWARE_FILE.exists():
        hw = json.loads(HARDWARE_FILE.read_text())
        chip = hw.get("chip", "unknown")
        mem = hw.get("memory_gb", 0)
        return f"{chip}_{mem:.0f}GB"
    return "unknown"

def get_speed_factor() -> float:
    """Get machine speed factor from hardware profile."""
    if HARDWARE_FILE.exists():
        hw = json.loads(HARDWARE_FILE.read_text())
        bench = hw.get("benchmarks", {}).get("cpu_matmul_2000x2000")
        if bench:
            # Normalize: 5ms matmul = factor 1.0
            return 0.005 / bench
    return 1.0

def save_history(history: dict):
    HISTORY_FILE.write_text(json.dumps(history, indent=2))

def get_estimate(job_name: str) -> float:
    """Get estimated duration based on history. Returns seconds."""
    history = load_history()
    if job_name in history["jobs"] and history["jobs"][job_name]:
        runs = history["jobs"][job_name]
        # Use median of last 5 runs
        recent = [r["duration"] for r in runs[-5:]]
        return sorted(recent)[len(recent)//2]
    return None  # No estimate available

def run_job(job_name: str, command: list[str], cwd: str = None) -> dict:
    """Run a job, time it, store result with hardware context."""
    history = load_history()
    machine_id = get_machine_id()
    speed_factor = get_speed_factor()

    estimate = get_estimate(job_name)
    if estimate:
        print(f"[{job_name}] Estimated: {estimate:.1f}s (based on {len(history['jobs'].get(job_name, []))} prior runs)")
    else:
        print(f"[{job_name}] No prior data — will calibrate")
    print(f"[{job_name}] Machine: {machine_id} (speed factor: {speed_factor:.2f})")

    start = time.time()
    print(f"[{job_name}] Started: {datetime.now().isoformat()}")

    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True
    )

    duration = time.time() - start
    normalized_duration = duration / speed_factor  # Normalize to baseline machine

    # Store result with hardware context
    if job_name not in history["jobs"]:
        history["jobs"][job_name] = []

    history["jobs"][job_name].append({
        "timestamp": datetime.now().isoformat(),
        "duration": duration,
        "normalized_duration": normalized_duration,
        "machine_id": machine_id,
        "speed_factor": speed_factor,
        "exit_code": result.returncode,
        "command": " ".join(command)
    })

    save_history(history)

    # Report accuracy if we had an estimate
    if estimate:
        error_pct = abs(duration - estimate) / estimate * 100
        print(f"[{job_name}] Actual: {duration:.1f}s | Estimate was off by {error_pct:.1f}%")
    else:
        print(f"[{job_name}] Actual: {duration:.1f}s | Calibrated for next run")

    return {
        "job_name": job_name,
        "duration": duration,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }

def show_stats():
    """Show all job statistics."""
    history = load_history()
    print("\n=== Job Duration Statistics ===\n")
    for job_name, runs in history["jobs"].items():
        if not runs:
            continue
        durations = [r["duration"] for r in runs]
        avg = sum(durations) / len(durations)
        median = sorted(durations)[len(durations)//2]
        print(f"{job_name}:")
        print(f"  Runs: {len(runs)}")
        print(f"  Median: {median:.1f}s")
        print(f"  Avg: {avg:.1f}s")
        print(f"  Range: {min(durations):.1f}s - {max(durations):.1f}s")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: job_tracker.py <job_name> <command...>")
        print("       job_tracker.py --stats")
        sys.exit(1)

    if sys.argv[1] == "--stats":
        show_stats()
    else:
        job_name = sys.argv[1]
        command = sys.argv[2:]
        result = run_job(job_name, command, cwd="/Volumes/4TB SD/ClaudeCode/noethersolve")
        print(f"\nExit code: {result['exit_code']}")
