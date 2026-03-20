#!/usr/bin/env python3
"""
Hardware profiler — detects system specs, persists across sessions.
Used to calibrate job duration estimates.
"""
import json
import subprocess
import platform
import os
from pathlib import Path
from datetime import datetime

PROFILE_FILE = Path(__file__).parent / "hardware_profile.json"

def get_mac_chip():
    """Get Apple Silicon chip info."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except:
        return None

def get_cpu_cores():
    """Get CPU core counts."""
    try:
        perf = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            capture_output=True, text=True
        )
        eff = subprocess.run(
            ["sysctl", "-n", "hw.perflevel1.logicalcpu"],
            capture_output=True, text=True
        )
        return {
            "performance": int(perf.stdout.strip()) if perf.returncode == 0 else None,
            "efficiency": int(eff.stdout.strip()) if eff.returncode == 0 else None,
            "total": os.cpu_count()
        }
    except:
        return {"total": os.cpu_count()}

def get_memory():
    """Get RAM in GB."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        )
        bytes_ram = int(result.stdout.strip())
        return bytes_ram / (1024**3)
    except:
        return None

def get_gpu_info():
    """Get GPU/Neural Engine info for Apple Silicon."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True
        )
        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        gpus = []
        for d in displays:
            gpus.append({
                "name": d.get("sppci_model", "Unknown"),
                "cores": d.get("sppci_cores", "Unknown"),
                "metal": d.get("spmetal_supported", "Unknown")
            })
        return gpus
    except:
        return []

def check_mlx():
    """Check if MLX is available and working."""
    try:
        result = subprocess.run(
            ["python3", "-c", "import mlx.core as mx; print(mx.default_device())"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return {"available": True, "device": result.stdout.strip()}
        return {"available": False}
    except:
        return {"available": False}

def run_benchmark():
    """Quick benchmark to measure actual compute speed."""
    import time

    # CPU benchmark: matrix multiply
    try:
        import numpy as np
        size = 2000
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        start = time.time()
        _ = a @ b
        cpu_matmul_time = time.time() - start
    except:
        cpu_matmul_time = None

    # MLX benchmark if available
    mlx_matmul_time = None
    try:
        import mlx.core as mx
        size = 2000
        a = mx.random.normal((size, size))
        b = mx.random.normal((size, size))
        mx.eval(a, b)  # ensure allocated
        start = time.time()
        c = a @ b
        mx.eval(c)
        mlx_matmul_time = time.time() - start
    except:
        pass

    return {
        "cpu_matmul_2000x2000": cpu_matmul_time,
        "mlx_matmul_2000x2000": mlx_matmul_time
    }

def profile_hardware(force_refresh=False) -> dict:
    """Get or create hardware profile."""
    if PROFILE_FILE.exists() and not force_refresh:
        profile = json.loads(PROFILE_FILE.read_text())
        print(f"Loaded hardware profile from {profile['profiled_at']}")
        return profile

    print("Profiling hardware (first run or refresh)...")

    profile = {
        "profiled_at": datetime.now().isoformat(),
        "platform": platform.platform(),
        "chip": get_mac_chip(),
        "cpu_cores": get_cpu_cores(),
        "memory_gb": get_memory(),
        "gpus": get_gpu_info(),
        "mlx": check_mlx(),
        "benchmarks": run_benchmark()
    }

    PROFILE_FILE.write_text(json.dumps(profile, indent=2))
    print(f"Hardware profile saved to {PROFILE_FILE}")
    return profile

def compute_speed_factor(profile: dict) -> float:
    """
    Compute a relative speed factor for this machine.
    Baseline: M3 Ultra = 1.0
    """
    # Use benchmark if available
    if profile.get("benchmarks", {}).get("cpu_matmul_2000x2000"):
        # M3 Ultra baseline: ~0.08s for 2000x2000 matmul
        baseline = 0.08
        actual = profile["benchmarks"]["cpu_matmul_2000x2000"]
        return baseline / actual

    # Fallback: estimate from cores
    cores = profile.get("cpu_cores", {}).get("total", 8)
    return cores / 24  # M3 Ultra has 24 cores

def show_profile():
    """Display current hardware profile."""
    profile = profile_hardware()
    print("\n=== Hardware Profile ===\n")
    print(f"Chip: {profile.get('chip', 'Unknown')}")
    print(f"Cores: {profile.get('cpu_cores', {})}")
    print(f"Memory: {profile.get('memory_gb', 0):.1f} GB")
    print(f"MLX: {profile.get('mlx', {})}")

    if profile.get("benchmarks"):
        print(f"\nBenchmarks (2000x2000 matmul):")
        b = profile["benchmarks"]
        if b.get("cpu_matmul_2000x2000"):
            print(f"  CPU (NumPy): {b['cpu_matmul_2000x2000']*1000:.1f}ms")
        if b.get("mlx_matmul_2000x2000"):
            print(f"  MLX (GPU): {b['mlx_matmul_2000x2000']*1000:.1f}ms")

    speed = compute_speed_factor(profile)
    print(f"\nSpeed factor: {speed:.2f}x (vs M3 Ultra baseline)")

if __name__ == "__main__":
    import sys
    if "--refresh" in sys.argv:
        profile_hardware(force_refresh=True)
    show_profile()
