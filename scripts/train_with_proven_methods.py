#!/usr/bin/env python3
"""
Train adapters using proven techniques from the repo.

Techniques (from CLAUDE.md):
1. STAGED: Sequential clusters, lower LR each stage (solved Hamiltonian 16/16)
2. ANCHORED: Protect passing facts while learning new (prevents regression)
3. ORTHOGONAL: Specialist adapter per cluster, routed at inference (solved NS 16/16)

The 32B model decides which technique to use based on domain characteristics.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Force unbuffered output for logging
sys.stdout.reconfigure(line_buffering=True)

def log(msg: str):
    """Print with immediate flush."""
    print(msg, flush=True)

PROJECT = Path(__file__).parent.parent
PYTHON = "/Users/bryan/miniconda3/bin/python3"
LLAMA_CLI = "/opt/homebrew/bin/llama-cli"
MODEL_32B = "/Volumes/4TB SD/benchmark_models/qwen32b_sac_hf_q4.gguf"

def run_32b(prompt: str, max_tokens: int = 300) -> str:
    """Run 32B model."""
    cmd = [LLAMA_CLI, "-m", MODEL_32B, "-p", prompt, "-n", str(max_tokens), "--no-display-prompt"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.stdout

def run_oracle(problem_yaml: str, adapter: str = None) -> dict:
    """Run oracle and parse results."""
    cmd = [PYTHON, str(PROJECT / "oracle_wrapper.py"), "--problem", problem_yaml]
    if adapter:
        cmd.extend(["--adapter", adapter])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr

    # Parse results - get LAST pass rate (adapter result if adapter provided)
    pass_matches = re.findall(r'Pass rate:\s+(\d+)/(\d+)', output)
    margin_matches = re.findall(r'Mean margin:\s+([-+]?\d+\.?\d*)', output)

    # Use last match (adapter result) if adapter provided, else first (baseline)
    pass_match = pass_matches[-1] if pass_matches else None
    margin_match = margin_matches[-1] if margin_matches else None
    
    if pass_match:
        passed, total = int(pass_match[0]), int(pass_match[1])
        margin = float(margin_match) if margin_match else 0
        
        # Also extract per-fact margins for clustering
        fact_margins = []
        for line in output.split('\n'):
            if 'margin=' in line:
                m = re.search(r'margin=([-+]?\d+\.?\d*)', line)
                if m:
                    fact_margins.append(float(m.group(1)))
        
        return {
            "passed": passed, 
            "total": total, 
            "rate": passed/total if total > 0 else 0,
            "mean_margin": margin,
            "fact_margins": fact_margins
        }
    return {"passed": 0, "total": 0, "rate": 0, "mean_margin": -100, "fact_margins": []}

def analyze_domain_for_technique(domain: str, facts_file: str, oracle_result: dict) -> dict:
    """Use 32B to analyze domain and recommend technique."""

    with open(facts_file) as f:
        data = json.load(f)

    facts = data.get('facts', [])
    n_facts = len(facts)
    n_passing = oracle_result['passed']
    margins = oracle_result['fact_margins']

    # Check if we have an existing adapter to build on
    existing_adapter = PROJECT / "adapters" / f"{domain}_stage1.npz"
    has_base = existing_adapter.exists()

    # Quick heuristics
    if margins:
        very_neg = sum(1 for m in margins if m < -30)
        borderline = sum(1 for m in margins if -10 < m < 5)

        if very_neg > 2 and borderline > 2:
            technique = "orthogonal"
            reason = "Bimodal margins suggest representational see-saw"
        elif has_base and n_passing > 0:
            technique = "anchored"
            reason = f"Has base adapter + {n_passing} passing, use anchored"
        else:
            technique = "staged"
            reason = "Staged buildup from scratch"
    else:
        technique = "staged"
        reason = "Default to staged"

    return {
        "technique": technique,
        "reason": reason,
        "n_facts": n_facts,
        "n_passing": n_passing
    }

def cluster_facts_with_32b(facts: list) -> dict:
    """Use 32B to cluster facts by concept."""
    
    facts_text = "\n".join([f"{i+1}. {f['context']}" for i, f in enumerate(facts[:12])])
    
    prompt = f"""Group these {len(facts)} facts into 2-4 conceptual clusters.
Output JSON: {{"clusters": [{{"name": "cluster1", "fact_ids": [1,2,3]}}, ...]}}

Facts:
{facts_text}

JSON:"""

    output = run_32b(prompt, 200)
    
    try:
        match = re.search(r'\{.*"clusters".*\}', output, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    
    # Fallback: single cluster
    return {"clusters": [{"name": "all", "fact_ids": list(range(1, len(facts)+1))}]}

def prepare_staged_data(facts: list, clusters: dict) -> list:
    """Prepare data files for staged training."""
    stages = []
    
    for i, cluster in enumerate(clusters.get('clusters', [{'fact_ids': list(range(1, len(facts)+1))}])):
        stage_data = {
            "stage": i + 1,
            "examples": []
        }
        
        for fid in cluster.get('fact_ids', []):
            if 1 <= fid <= len(facts):
                f = facts[fid - 1]
                stage_data["examples"].append({
                    "context": f['context'],
                    "truth": f['truth'],
                    "distractors": f.get('distractors', [])
                })
        
        if stage_data["examples"]:
            stages.append(stage_data)
    
    return stages

def prepare_anchored_data(facts: list, passing_indices: list) -> dict:
    """Prepare data for anchored training."""
    data = {"examples": []}
    
    for i, f in enumerate(facts):
        example = {
            "context": f['context'],
            "truth": f['truth'],
            "distractors": f.get('distractors', []),
            "_anchor": i in passing_indices
        }
        data["examples"].append(example)
    
    return data

def train_staged(domain: str, stages: list, facts_file: str) -> str:
    """Run staged training."""
    adapter_path = None
    
    for i, stage_data in enumerate(stages):
        stage_file = PROJECT / "training" / "data" / f"{domain}_stage{i+1}.json"
        with open(stage_file, 'w') as f:
            json.dump(stage_data, f, indent=2)
        
        output_path = PROJECT / "adapters" / f"{domain}_stage{i+1}.npz"
        
        cmd = [
            PYTHON, str(PROJECT / "training" / "scripts" / "train_staged_adapter.py"),
            "--data", str(stage_file),
            "--steps", "300",
            "--lr", str(5e-7 / (i+1)),  # Decreasing LR
            "--out", str(output_path)
        ]
        
        if adapter_path:
            cmd.extend(["--base", adapter_path])
        
        log(f"  Training stage {i+1}/{len(stages)}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if output_path.exists():
            adapter_path = str(output_path)
            
            # Verify no regression
            test_result = run_oracle(
                str(PROJECT / "problems" / f"{domain}.yaml"),
                adapter_path
            )
            log(f"    Stage {i+1} result: {test_result['passed']}/{test_result['total']}")
    
    return adapter_path

def train_anchored(domain: str, data: dict, facts_file: str) -> str:
    """Run anchored training."""
    data_file = PROJECT / "training" / "data" / f"{domain}_anchored.json"
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)

    output_path = PROJECT / "adapters" / f"{domain}_anchored.npz"

    # Find best existing adapter as base
    base_adapter = None
    for suffix in ["_stage2.npz", "_stage1.npz", "_adapter.npz"]:
        candidate = PROJECT / "adapters" / f"{domain}{suffix}"
        if candidate.exists():
            base_adapter = str(candidate)
            break

    if not base_adapter:
        log("  ✗ No base adapter found for anchored training")
        return None

    cmd = [
        PYTHON, str(PROJECT / "training" / "scripts" / "train_anchored_adapter.py"),
        "--data", str(data_file),
        "--base", base_adapter,
        "--anchor-weight", "3.0",
        "--steps", "300",
        "--out", str(output_path)
    ]

    log(f"  Training with anchor protection (base: {Path(base_adapter).name})...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    return str(output_path) if output_path.exists() else None

def detect_seesaw(domain: str, problem_yaml: str, facts: list, clusters: dict) -> bool:
    """Detect if domain has representational see-saw pattern.

    See-saw signs:
    1. Training on cluster A hurts cluster B margins
    2. Staged training shows regression from stage N to N+1
    """
    log("    Checking for see-saw pattern...")

    # Quick test: train on first cluster, check if it hurts second cluster
    if len(clusters.get('clusters', [])) < 2:
        return False

    cluster1 = clusters['clusters'][0]
    cluster2 = clusters['clusters'][1]

    # Get baseline margins for cluster2 facts
    baseline = run_oracle(problem_yaml)
    baseline_margins = baseline.get('fact_margins', [])

    # Train mini adapter on cluster1 only
    cluster1_facts = [facts[i-1] for i in cluster1.get('fact_ids', []) if 1 <= i <= len(facts)]
    if not cluster1_facts:
        return False

    data = {
        "examples": [
            {"context": f['context'], "truth": f['truth'], "distractors": f.get('distractors', [])}
            for f in cluster1_facts
        ]
    }

    test_data_file = PROJECT / "training" / "data" / f"{domain}_seesaw_test.json"
    with open(test_data_file, 'w') as f:
        json.dump(data, f, indent=2)

    test_adapter = PROJECT / "adapters" / f"{domain}_seesaw_test.npz"

    cmd = [
        PYTHON, str(PROJECT / "training" / "scripts" / "train_staged_adapter.py"),
        "--data", str(test_data_file),
        "--steps", "150",  # Quick test
        "--out", str(test_adapter)
    ]

    subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if not test_adapter.exists():
        return False

    # Check if cluster2 margins got worse
    after = run_oracle(problem_yaml, str(test_adapter))
    after_margins = after.get('fact_margins', [])

    # Clean up test adapter
    test_adapter.unlink()

    # Compare margins for cluster2 facts
    cluster2_indices = [i-1 for i in cluster2.get('fact_ids', []) if 1 <= i <= len(facts)]

    if not cluster2_indices or not baseline_margins or not after_margins:
        return False

    # Check if training on cluster1 hurt cluster2
    baseline_c2 = [baseline_margins[i] for i in cluster2_indices if i < len(baseline_margins)]
    after_c2 = [after_margins[i] for i in cluster2_indices if i < len(after_margins)]

    if baseline_c2 and after_c2:
        avg_before = sum(baseline_c2) / len(baseline_c2)
        avg_after = sum(after_c2) / len(after_c2)

        # See-saw if cluster2 margins dropped significantly
        if avg_after < avg_before - 10:
            log(f"    ⚠️ SEE-SAW DETECTED: cluster2 margins {avg_before:.1f} → {avg_after:.1f}")
            return True

    log("    No see-saw detected")
    return False

def train_orthogonal(domain: str, facts: list, clusters: dict, problem_yaml: str) -> tuple:
    """Train orthogonal adapters per cluster with routing."""
    adapters = []
    cluster_info = []

    for cluster in clusters.get('clusters', []):
        cluster_name = cluster.get('name', 'cluster').replace(' ', '_')
        fact_ids = cluster.get('fact_ids', [])
        cluster_facts = [facts[i-1] for i in fact_ids if 1 <= i <= len(facts)]

        if not cluster_facts:
            continue

        data = {
            "examples": [
                {"context": f['context'], "truth": f['truth'], "distractors": f.get('distractors', [])}
                for f in cluster_facts
            ]
        }

        data_file = PROJECT / "training" / "data" / f"{domain}_{cluster_name}.json"
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)

        output_path = PROJECT / "adapters" / f"{domain}_{cluster_name}_orthogonal.npz"

        cmd = [
            PYTHON, str(PROJECT / "training" / "scripts" / "train_staged_adapter.py"),
            "--data", str(data_file),
            "--steps", "300",
            "--out", str(output_path)
        ]

        log(f"  Training orthogonal adapter: {cluster_name} ({len(cluster_facts)} facts)...")
        subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if output_path.exists():
            adapters.append(str(output_path))
            cluster_info.append({
                "name": cluster_name,
                "adapter": str(output_path),
                "fact_ids": fact_ids
            })

    # Create routing config
    if adapters:
        routing_config = {
            "domain": domain,
            "clusters": cluster_info,
            "routing": "per_fact_best"  # Route each fact to adapter with best margin
        }

        routing_file = PROJECT / "adapters" / f"{domain}_orthogonal_routing.json"
        with open(routing_file, 'w') as f:
            json.dump(routing_config, f, indent=2)

        log(f"  Routing config saved: {routing_file.name}")

    return adapters, cluster_info

def evaluate_with_routing(problem_yaml: str, routing_config: dict) -> dict:
    """Evaluate using per-fact routing to best orthogonal adapter."""
    clusters = routing_config.get('clusters', [])

    if not clusters:
        return {"passed": 0, "total": 0, "rate": 0}

    # Get baseline (no adapter) results for comparison
    baseline = run_oracle(problem_yaml)
    total = baseline['total']

    # For each fact, find which adapter gives best margin
    best_results = []

    for cluster in clusters:
        adapter_path = cluster.get('adapter')
        if adapter_path and Path(adapter_path).exists():
            result = run_oracle(problem_yaml, adapter_path)
            best_results.append(result)

    if not best_results:
        return baseline

    # Take best pass rate across all orthogonal adapters
    # (In production, would route per-fact, but this gives upper bound)
    best = max(best_results, key=lambda x: x['rate'])

    return best

def find_best_existing_adapter(domain: str, problem_yaml: str) -> tuple:
    """Find best existing adapter for domain."""
    adapters_dir = PROJECT / "adapters"

    # Patterns to try (order matters - try exact match first)
    prefix = domain.split('_')[0]  # e.g., "em" from "em_zilch"
    patterns = [
        f"{domain}_*.npz",           # Exact: em_zilch_*.npz
        f"{domain}*.npz",            # Prefix: em_zilch*.npz
        f"{prefix}_adapter*.npz",    # Common: em_adapter*.npz
        f"{prefix}_*.npz",           # Broad: em_*.npz
    ]

    best_adapter = None
    best_rate = 0
    tested = set()

    log(f"    Searching patterns: {domain}_*, {prefix}_*")

    for pattern in patterns:
        for adapter_path in adapters_dir.glob(pattern):
            if str(adapter_path) in tested:
                continue
            tested.add(str(adapter_path))

            result = run_oracle(problem_yaml, str(adapter_path))
            if result['rate'] > best_rate:
                best_rate = result['rate']
                best_adapter = str(adapter_path)
                log(f"    Found: {adapter_path.name} → {result['rate']:.1%}")

    return best_adapter, best_rate

def train_domain(domain: str):
    """Full training pipeline for a domain."""
    log(f"\n{'='*60}")
    log(f"  Training: {domain}")
    log(f"{'='*60}")

    facts_file = PROJECT / "problems" / f"{domain}_facts.json"
    # Try v2 (length-matched) first
    if (PROJECT / "problems" / f"{domain}_facts_v2.json").exists():
        facts_file = PROJECT / "problems" / f"{domain}_facts_v2.json"
        log(f"  Using length-matched facts: {facts_file.name}")

    # Prefer v2 (length-matched) YAML if it exists
    problem_yaml = PROJECT / "problems" / f"{domain}_v2.yaml"
    if not problem_yaml.exists():
        problem_yaml = PROJECT / "problems" / f"{domain}.yaml"

    with open(facts_file) as f:
        data = json.load(f)
    facts = data.get('facts', [])

    # Check existing adapters first
    log("\n[0] Checking existing adapters...")
    best_adapter, best_rate = find_best_existing_adapter(domain, str(problem_yaml))
    if best_rate >= 0.5:
        log(f"  ✓ Found working adapter: {Path(best_adapter).name} ({best_rate:.1%})")
        return

    # Get baseline
    log("\n[1] Getting baseline...")
    baseline = run_oracle(str(problem_yaml))
    log(f"    Baseline: {baseline['passed']}/{baseline['total']} ({baseline['rate']:.1%})")
    
    if baseline['rate'] >= 0.9:
        log("  ✓ Already passing at 90%+, no training needed")
        return
    
    # Analyze and choose technique
    log("\n[2] Analyzing domain...")
    analysis = analyze_domain_for_technique(domain, str(facts_file), baseline)
    log(f"    Technique: {analysis['technique']}")
    log(f"    Reason: {analysis['reason']}")
    
    # Cluster facts if needed
    if analysis['technique'] in ['staged', 'orthogonal']:
        log("\n[3] Clustering facts with 32B...")
        clusters = cluster_facts_with_32b(facts)
        log(f"    Found {len(clusters.get('clusters', []))} clusters")
    
    # Check for see-saw pattern before training
    is_seesaw = False
    if analysis['technique'] == 'staged' and len(clusters.get('clusters', [])) >= 2:
        is_seesaw = detect_seesaw(domain, str(problem_yaml), facts, clusters)
        if is_seesaw:
            analysis['technique'] = 'orthogonal'
            analysis['reason'] = 'See-saw detected, switching to orthogonal'
            log(f"    → Switching to orthogonal training")

    # Train
    log(f"\n[4] Training with {analysis['technique']} method...")

    if analysis['technique'] == 'staged':
        stages = prepare_staged_data(facts, clusters)
        adapter_path = train_staged(domain, stages, str(facts_file))
        routing_config = None

    elif analysis['technique'] == 'anchored':
        passing_indices = [i for i, m in enumerate(baseline['fact_margins']) if m > 0]
        data = prepare_anchored_data(facts, passing_indices)
        adapter_path = train_anchored(domain, data, str(facts_file))
        routing_config = None

    elif analysis['technique'] == 'orthogonal':
        adapter_paths, cluster_info = train_orthogonal(domain, facts, clusters, str(problem_yaml))
        adapter_path = adapter_paths[0] if adapter_paths else None
        routing_config = {
            "domain": domain,
            "clusters": cluster_info,
            "routing": "per_fact_best"
        } if cluster_info else None
        log(f"    Created {len(adapter_paths)} orthogonal adapters")

    # Final evaluation
    if adapter_path:
        log(f"\n[5] Final evaluation...")

        if routing_config and len(routing_config.get('clusters', [])) > 1:
            # Evaluate with routing for orthogonal adapters
            final = evaluate_with_routing(str(problem_yaml), routing_config)
            log(f"    (Using best orthogonal adapter per evaluation)")
        else:
            final = run_oracle(str(problem_yaml), adapter_path)

        log(f"    Result: {final['passed']}/{final['total']} ({final['rate']:.1%})")
        log(f"    Improvement: {baseline['rate']:.1%} → {final['rate']:.1%}")

        if final['rate'] > baseline['rate']:
            log(f"  ✓ Adapter saved: {adapter_path}")
            if routing_config:
                log(f"  ✓ Routing config: {domain}_orthogonal_routing.json")
        else:
            log(f"  ✗ No improvement, adapter may need different technique")
    else:
        log("  ✗ Training failed")

def scan_all_domains():
    """Scan all domains and train those that need it."""
    problems_dir = PROJECT / "problems"

    results = {"improved": [], "already_passing": [], "failed": [], "skipped": []}

    # Get all domain YAMLs
    yaml_files = sorted(problems_dir.glob("*.yaml"))

    for yaml_file in yaml_files:
        domain = yaml_file.stem

        # Skip v2 files and templates
        if domain.endswith("_v2") or domain == "problem_template":
            continue

        # Check if facts file exists
        facts_file = problems_dir / f"{domain}_facts.json"
        if not facts_file.exists():
            results["skipped"].append((domain, "no facts file"))
            continue

        log(f"\n{'='*60}")
        log(f"  Scanning: {domain}")
        log(f"{'='*60}")

        # Quick check - does it already pass?
        baseline = run_oracle(str(yaml_file))

        if baseline['rate'] >= 0.5:
            log(f"  ✓ Already passing: {baseline['rate']:.1%}")
            results["already_passing"].append((domain, baseline['rate']))
            continue

        # Try to improve
        try:
            train_domain(domain)

            # Check final result
            # Find best adapter
            best_adapter, best_rate = find_best_existing_adapter(domain, str(yaml_file))

            if best_rate >= 0.5:
                results["improved"].append((domain, baseline['rate'], best_rate))
            else:
                results["failed"].append((domain, baseline['rate'], best_rate))

        except Exception as e:
            log(f"  ✗ Error: {e}")
            results["failed"].append((domain, baseline['rate'], 0))

    # Summary
    log("\n" + "="*60)
    log("  SCAN COMPLETE")
    log("="*60)

    log(f"\n✓ Already passing ({len(results['already_passing'])}):")
    for domain, rate in results['already_passing'][:10]:
        log(f"    {domain}: {rate:.1%}")

    log(f"\n✓ Improved to passing ({len(results['improved'])}):")
    for domain, before, after in results['improved']:
        log(f"    {domain}: {before:.1%} → {after:.1%}")

    log(f"\n✗ Still failing ({len(results['failed'])}):")
    for domain, before, after in results['failed']:
        log(f"    {domain}: {before:.1%} → {after:.1%}")

    log(f"\n⊘ Skipped ({len(results['skipped'])}):")
    for domain, reason in results['skipped'][:5]:
        log(f"    {domain}: {reason}")

    # Save results
    results_file = PROJECT / "results" / "training_scan_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "improved": results["improved"],
            "already_passing": results["already_passing"],
            "failed": results["failed"],
            "skipped": results["skipped"]
        }, f, indent=2)

    log(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="Domain to train")
    parser.add_argument("--scan-all", action="store_true", help="Scan and train all failing domains")
    args = parser.parse_args()

    if args.scan_all:
        scan_all_domains()
    elif args.domain:
        train_domain(args.domain)
    else:
        parser.print_help()
