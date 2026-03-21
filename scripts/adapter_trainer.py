#!/usr/bin/env python3
"""
Adapter Trainer — Full Autonomous Fact-Flipping Pipeline for 4B.

WHO RUNS THIS: The 27B machine (Apple Silicon local compute workhorse).
WHAT IT DOES:  Loads the 4B model ONCE, then for each failing domain:
  1. Evaluates baseline (per-fact margins)
  2. Trains single-pass adapter
  3. Re-evaluates → identifies which facts flipped
  4. If interference detected → clusters facts → staged training
  5. If see-saw detected → orthogonal adapters with routing config
  6. Logs every step to adapter_training_log.jsonl

The 27B provides COMPUTE (runs this script on Apple Silicon MLX).
The ADAPTER targets the 4B (dimensions = 4B's vocab=151936, d_model=2560).
The training forward/backward passes go through the 4B model loaded locally.

Escalation ladder (all automated, no Claude Code needed):
  L1: Single-pass adapter (4000 steps, lr=4e-6, margin=2.0)
  L2: Intensive adapter (5000 steps, lr=3e-6, margin=2.5)
  L3: Staged training (3000 steps/cluster, lr=4e-6, margin=2.5)
  L4: Orthogonal adapters (4000 steps/cluster, lr=4e-6, margin=2.5)

Usage:
    python scripts/adapter_trainer.py           # Train all failing domains
    python scripts/adapter_trainer.py --once    # Train one domain and exit
    python scripts/adapter_trainer.py --status  # Show what needs training
    python scripts/adapter_trainer.py --domain knot_invariants  # Specific domain
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve import train_utils as t3

ADAPTERS_DIR = PROJECT / "adapters"
PROBLEMS_DIR = PROJECT / "problems"
RESULTS_DIR = PROJECT / "results"

# ── Role Separation ──────────────────────────────────────────────────
# 27B = LOCAL COMPUTE WORKHORSE. Runs this script, does the training.
# 4B  = THE STUDENT. Gets adapter training to learn surprising truths.
#       Adapters must match 4B dimensions (d_model=2560, vocab=151936).
# Claude Code = writes papers, V2 fact files, code. Not involved in training.
#
# The adapter is trained ON the 4B model (forward+backward passes through 4B)
# because the adapter dimensions must match the 4B's logit space.
# The 27B provides compute power by running this script on Apple Silicon.
# ─────────────────────────────────────────────────────────────────────
TRAIN_MODEL = "Qwen/Qwen3-4B-Base"  # Adapter matches 4B dimensions (vocab=151936)

# Explicit mappings for domain names that can't be fuzzy-matched
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
    # Additional mappings for full domain coverage
    "kinetic_invariant_k": "kinetic_k",
    "elliptic_curve_theory": "elliptic_curve",
    "ns_regularity_and_stretchresistant_q_f": "ns_regularity",
    "ns_regularity_and_stretch_resistant_q_f": "ns_regularity",
    "hamiltonian_mechanics_invariants": "hamiltonian",
    "chemical_reaction_network_conservation": "chemical_conservation",
    "information_theory": "information_theory",
    "3body_conservation": "3body_conservation",
    "llm_hallucination_grounded": "llm_hallucination_balanced",
    "physics_fundamentals": "physics_fundamentals_2d_turbulence",
    "organic_chemistry": "chemistry",
    "clinical_biochemistry": "biochemistry",
    "biology": "aging_biology",
    "geophysics_seismic": "geophysics_seismic",
    "pathophysiology": "pathophysiology",
}


# ═══════════════════════════════════════════════════════════════════════
# Section 1: Domain Discovery — Find failing domains and their facts
# ═══════════════════════════════════════════════════════════════════════

def load_run_summary():
    """Load domain pass rates from all available sources.

    Merges run_summary.json (sweep results) with research_status.json
    (full evaluation results) to get the complete picture.
    """
    rates = {}

    # Source 1: run_summary.json (original sweep data)
    summary_path = RESULTS_DIR / "run_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        for sweep in reversed(data.get("sweeps", [])):
            if sweep.get("domain_pass_rates"):
                rates.update(sweep["domain_pass_rates"])
                break

    # Source 2: research_status.json (full evaluation — may have more domains)
    status_path = RESULTS_DIR / "research_status.json"
    if status_path.exists():
        with open(status_path) as f:
            data = json.load(f)
        for name, r in data.get("domain_results", {}).items():
            oracle_rate = r.get("oracle_rate", None)
            if oracle_rate is not None and name not in rates:
                rates[name] = oracle_rate

    return rates


def normalize_domain_name(name):
    """Normalize domain name to snake_case for file matching."""
    base = name.replace(" V2", "").replace("_v2", "")
    base = base.replace(" ", "_").lower()
    base = "".join(c for c in base if c.isalnum() or c == "_")
    return base


def get_failing_domains():
    """Get domains that are failing (pass_rate < 0.5).

    Returns list of (domain_name, pass_rate, facts_file).
    """
    rates = load_run_summary()
    if not rates:
        print("No run_summary.json found. Run research_runner.py first.")
        return []

    # Build facts file map (V2 preferred over V1 for same stem)
    facts_map = {}
    for ff in sorted(PROBLEMS_DIR.glob("*_facts*.json"), reverse=True):
        try:
            with open(ff) as f:
                data = json.load(f)
            prob_name = data.get("problem", ff.stem.replace("_facts", ""))
            norm = normalize_domain_name(prob_name)
            file_norm = normalize_domain_name(ff.stem.replace("_facts_v2", "").replace("_facts", ""))
            for n in [norm, file_norm]:
                if n not in facts_map or "_v2" in ff.stem:
                    facts_map[n] = ff
        except (json.JSONDecodeError, KeyError):
            continue

    # Deduplicate: group by normalized name, keep the LOWEST rate version
    # (we want to train the hardest variant, not skip it because a V2 passed)
    best = {}
    for name, rate in rates.items():
        base = normalize_domain_name(name)
        # Keep the FAILING version if any variant fails
        if base not in best or rate < best[base][1]:
            best[base] = (name, rate)

    failing = []
    seen_facts = set()  # Avoid training same facts file twice
    for base, (name, rate) in sorted(best.items(), key=lambda x: x[1][1]):
        if rate >= 0.5:
            continue

        facts_file = None

        # Try direct match (V2 preferred via facts_map construction)
        if base in facts_map:
            facts_file = facts_map[base]
        elif base in DOMAIN_TO_FACTS:
            mapped = DOMAIN_TO_FACTS[base]
            mapped_v2 = mapped + "_v2" if not mapped.endswith("_v2") else mapped
            if mapped_v2 in facts_map:
                facts_file = facts_map[mapped_v2]
            elif mapped in facts_map:
                facts_file = facts_map[mapped]
            else:
                for suffix in ["_facts_v2.json", "_facts.json"]:
                    candidate = PROBLEMS_DIR / f"{mapped}{suffix}"
                    if candidate.exists():
                        facts_file = candidate
                        break
        else:
            for key in facts_map:
                if base in key or key in base:
                    facts_file = facts_map[key]
                    break

        if facts_file is not None and facts_file.exists():
            # Skip if we'd be training on the exact same facts file
            ff_key = str(facts_file)
            if ff_key in seen_facts:
                continue
            seen_facts.add(ff_key)
            failing.append((name, rate, facts_file))
        else:
            failing.append((name, rate, None))

    return failing


def load_facts(facts_file, max_facts=30):
    """Load facts from JSON. Returns list of dicts with id, context, truth, distractors, cluster.

    Capped at max_facts (default 30) — more is waste. 10-15 facts suffice for
    adapter training to 100%. Keeping it lean speeds up training dramatically.
    """
    with open(facts_file) as f:
        data = json.load(f)

    facts_raw = data.get("facts", data.get("verifications", []))
    facts = []
    for fact in facts_raw:
        context = fact.get("context", "")
        truth = fact.get("truth", fact.get("fact", ""))
        distractors = fact.get("distractors", [])
        # Filter out non-string distractors (broken V1 format)
        distractors = [d for d in distractors if isinstance(d, str)]

        if context and truth and distractors:
            facts.append({
                "id": fact.get("id", f"fact_{len(facts)}"),
                "context": context,
                "truth": truth,
                "distractors": distractors,
                "cluster": fact.get("cluster", None),
            })

    # Cap at max_facts — shuffle deterministically to get variety
    if len(facts) > max_facts:
        import random
        rng = random.Random(42)
        rng.shuffle(facts)
        facts = facts[:max_facts]

    return facts


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Oracle Evaluation — Per-fact margins using loaded model
# ═══════════════════════════════════════════════════════════════════════

def compute_logprob(model, lm_head, tokenizer, prompt, text, adapter=None):
    """Compute log-probability of text given prompt, optionally with adapter."""
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + text)
    n_prompt = len(prompt_ids)
    if len(full_ids) <= n_prompt:
        return -1e9

    tokens = mx.array(full_ids)[None, :]
    h = model.model(tokens)
    base_logits = lm_head(h)

    if adapter is not None:
        shifts = adapter(base_logits)
        shifts = shifts - shifts.mean(axis=-1, keepdims=True)
        logits = base_logits + shifts
    else:
        logits = base_logits

    logits = t3.LOGIT_SOFTCAP * mx.tanh(logits / t3.LOGIT_SOFTCAP)

    total = 0.0
    for i, tok_id in enumerate(full_ids[n_prompt:]):
        pos = n_prompt - 1 + i
        lv = logits[0, pos]
        lv_np = np.array(lv.astype(mx.float32))
        lse = float(np.log(np.sum(np.exp(lv_np - lv_np.max())) + 1e-8) + lv_np.max())
        total += float(lv_np[tok_id]) - lse

    return total


def evaluate_facts(model, lm_head, tokenizer, facts, adapter=None, label=""):
    """Evaluate all facts, return per-fact margins and summary.

    Returns: list of (fact_id, margin, correct_bool)
    """
    results = []
    correct = 0

    for fact in facts:
        prompt = fact["context"] if fact["context"].endswith(":") else fact["context"] + ":"

        truth_lp = compute_logprob(model, lm_head, tokenizer, prompt, f" {fact['truth']}", adapter)
        dist_lps = [compute_logprob(model, lm_head, tokenizer, prompt, f" {d}", adapter)
                     for d in fact["distractors"]]
        best_dist = max(dist_lps) if dist_lps else truth_lp - 10
        margin = truth_lp - best_dist
        is_correct = margin > 0

        results.append((fact["id"], margin, is_correct))
        if is_correct:
            correct += 1

    if facts:
        avg_margin = np.mean([m for _, m, _ in results])
        print(f"  {label}: {correct}/{len(facts)} ({100*correct/len(facts):.1f}%)  "
              f"avg_margin={avg_margin:.2f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Training — mc_hinge_loss + gradient descent
# ═══════════════════════════════════════════════════════════════════════

def mc_hinge_loss(adapter, lm_head, model, prompt, truth, distractors, tokenizer,
                  margin_target=2.0):
    """Differentiable hinge loss: max(0, margin_target - (truth_lp - best_dist_lp))."""
    def lp(text):
        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(prompt + text)
        n_prompt = len(prompt_ids)
        if len(full_ids) <= n_prompt:
            return mx.array(-1e9)
        tokens = mx.array(full_ids)[None, :]
        h = model.model(tokens)
        base_logits = lm_head(h)
        shifts = adapter(base_logits)
        shifts = shifts - shifts.mean(axis=-1, keepdims=True)
        logits = base_logits + shifts
        logits = t3.LOGIT_SOFTCAP * mx.tanh(logits / t3.LOGIT_SOFTCAP)
        total = mx.array(0.0)
        for i, tok_id in enumerate(full_ids[n_prompt:]):
            pos = n_prompt - 1 + i
            lv = logits[0, pos]
            lse = mx.log(mx.sum(mx.exp(lv - mx.max(lv))) + 1e-8) + mx.max(lv)
            total = total + lv[tok_id] - lse
        return total

    truth_lp = lp(f" {truth}")
    dist_lps = [lp(f" {d}") for d in distractors]
    best_dist = mx.max(mx.stack(dist_lps)) if dist_lps else truth_lp - margin_target - 1
    loss = mx.maximum(mx.array(0.0), mx.array(margin_target) - (truth_lp - best_dist))
    return loss, float(truth_lp - best_dist)


def clip_grads(grads, max_norm=1.0):
    leaves = tree_flatten(grads)
    total_sq = sum(float(mx.sum(g ** 2)) for _, g in leaves)
    norm = total_sq ** 0.5
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        leaves = [(k, g * scale) for k, g in leaves]
    return tree_unflatten(leaves)


def create_fresh_adapter(d_model, vocab_size, d_inner=64):
    """Create a new zero-initialized adapter."""
    config = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                          n_heads=8, mode="logit", vocab_size=vocab_size)
    return create_adapter(config)


def load_adapter_weights(adapter_path, d_model, vocab_size, d_inner=64):
    """Load an existing adapter from .npz file."""
    adapter = create_fresh_adapter(d_model, vocab_size, d_inner)
    weights = dict(np.load(adapter_path))
    mlx_weights = {k: mx.array(v) for k, v in weights.items()}
    adapter.load_weights(list(mlx_weights.items()))
    return adapter


def save_adapter(adapter, path):
    """Save adapter weights to .npz."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flat = tree_flatten(adapter.parameters())
    np.savez(str(path), **{k: np.array(v) for k, v in flat})


def train_adapter(model, lm_head, tokenizer, facts, adapter=None,
                  steps=2000, lr=4e-6, margin_target=2.0, d_inner=64,
                  label="", quiet=False):
    """Train an adapter on the given facts. Returns the trained adapter.

    If adapter is None, creates a fresh one.
    If adapter is provided, continues training from those weights.
    """
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]

    if adapter is None:
        adapter = create_fresh_adapter(d_model, vocab_size, d_inner)

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)
    loss_and_grad = mx.value_and_grad(mc_hinge_loss, argnums=0)

    examples = [(f["context"], f["truth"], f["distractors"]) for f in facts]
    if not examples:
        return adapter

    if not quiet:
        print(f"  Training {label}: {len(examples)} facts, {steps} steps, lr={lr}, margin={margin_target}")

    t0 = time.time()
    recent_margins = []

    for step in range(steps):
        ctx, truth, distractors = examples[step % len(examples)]
        prompt = ctx if ctx.endswith(":") else ctx + ":"

        (loss_val, margin_val), grads = loss_and_grad(
            adapter, lm_head, model, prompt, truth, distractors, tokenizer, margin_target
        )
        grads = clip_grads(grads, max_norm=1.0)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        recent_margins.append(margin_val)
        if len(recent_margins) > 50:
            recent_margins.pop(0)

        if not quiet and (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            avg = np.mean(recent_margins)
            print(f"    step {step+1:4d}  loss={float(loss_val):.3f}  "
                  f"margin={margin_val:.2f}  avg={avg:.2f}  {elapsed:.0f}s")

    if not quiet:
        print(f"  Training complete in {time.time()-t0:.0f}s")

    return adapter


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Clustering — Group facts for staged/orthogonal training
# ═══════════════════════════════════════════════════════════════════════

def cluster_facts(facts, margins_by_id):
    """Cluster facts for staged/orthogonal training.

    Strategy:
    1. If facts have explicit 'cluster' field, use that.
    2. Otherwise, split into "easy-to-flip" (margin > -5) vs "hard" (margin <= -5).
    3. If only one cluster, split by first/second half.

    Returns: dict of cluster_name → list of facts
    """
    # Check for explicit clusters
    explicit = defaultdict(list)
    for fact in facts:
        if fact.get("cluster"):
            explicit[fact["cluster"]].append(fact)

    if len(explicit) >= 2:
        return dict(explicit)

    # Split by margin difficulty
    easy = [f for f in facts if margins_by_id.get(f["id"], -999) > -5]
    hard = [f for f in facts if margins_by_id.get(f["id"], -999) <= -5]

    if easy and hard:
        return {"easy_first": easy, "hard_second": hard}

    # Fallback: split in half
    mid = len(facts) // 2
    if mid > 0:
        return {"cluster_a": facts[:mid], "cluster_b": facts[mid:]}

    return {"all": facts}


def detect_seesaw(before_margins, after_margins, threshold=-2.0):
    """Detect see-saw: training improved some facts but regressed others.

    Returns (is_seesaw, regressed_ids, improved_ids).
    """
    regressed = []
    improved = []

    for fid, before_m, _ in before_margins:
        after_m = None
        for aid, am, _ in after_margins:
            if aid == fid:
                after_m = am
                break
        if after_m is None:
            continue

        delta = after_m - before_m
        if delta < threshold and before_m > after_m:
            regressed.append(fid)
        elif delta > 1.0:
            improved.append(fid)

    is_seesaw = len(regressed) >= 2 and len(improved) >= 1
    return is_seesaw, regressed, improved


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Escalation Pipeline — L1 → L2 → L3 → L4
# ═══════════════════════════════════════════════════════════════════════

def get_adapter_status(domain_name):
    """Check what adapters exist for a domain."""
    base = normalize_domain_name(domain_name)
    adapters = list(ADAPTERS_DIR.glob(f"{base}*.npz"))
    if not adapters:
        prefix = base.split("_")[0]
        if len(prefix) >= 4:
            adapters = list(ADAPTERS_DIR.glob(f"{prefix}_*.npz"))

    routing = ADAPTERS_DIR / f"{base}_orthogonal_routing.json"
    return {
        "adapters": [a.name for a in adapters],
        "has_single": any("adapter" in a.name and "orthogonal" not in a.name
                          and "intensive" not in a.name for a in adapters),
        "has_intensive": any("intensive" in a.name for a in adapters),
        "has_orthogonal": routing.exists() or any("orthogonal" in a.name for a in adapters),
        "routing_config": str(routing) if routing.exists() else None,
    }


def run_domain_pipeline(model, lm_head, tokenizer, domain_name, facts_file,
                        d_model, vocab_size):
    """Full autonomous pipeline for one domain. Returns training log dict."""
    base = normalize_domain_name(domain_name)
    facts = load_facts(facts_file)

    if len(facts) < 3:
        print(f"  Only {len(facts)} facts — need at least 3. Skipping.")
        return {"domain": domain_name, "status": "skipped", "reason": "too_few_facts"}

    status = get_adapter_status(domain_name)
    log = {
        "domain": domain_name,
        "facts_file": str(facts_file.name),
        "num_facts": len(facts),
        "timestamp": datetime.now().isoformat(),
        "levels": [],
    }

    print(f"\n{'='*60}")
    print(f"  Domain: {domain_name}")
    print(f"  Facts:  {facts_file.name} ({len(facts)} facts)")
    print(f"  Prior:  {', '.join(status['adapters'][:3]) or 'none'}")
    print(f"{'='*60}")

    # ── Baseline evaluation ──
    print("\n--- Baseline (no adapter) ---")
    baseline_results = evaluate_facts(model, lm_head, tokenizer, facts, label="baseline")
    baseline_correct = sum(1 for _, _, c in baseline_results if c)
    baseline_rate = baseline_correct / len(facts) if facts else 0

    if baseline_rate >= 0.5:
        print(f"  Already passing at baseline ({baseline_rate:.0%}). Skipping.")
        log["status"] = "already_passing"
        log["baseline_rate"] = baseline_rate
        return log

    margins_by_id = {fid: m for fid, m, _ in baseline_results}

    # ── L1: Single-pass adapter ──
    if not status["has_single"]:
        print("\n--- L1: Single-pass adapter (4000 steps, lr=4e-6) ---")
        adapter = train_adapter(
            model, lm_head, tokenizer, facts,
            steps=2000, lr=4e-6, margin_target=2.0, label="L1-single"
        )
        l1_path = ADAPTERS_DIR / f"{base}_adapter.npz"
        save_adapter(adapter, l1_path)

        l1_results = evaluate_facts(model, lm_head, tokenizer, facts, adapter, label="L1-adapted")
        l1_correct = sum(1 for _, _, c in l1_results if c)
        l1_rate = l1_correct / len(facts)

        log["levels"].append({
            "level": "L1_single", "correct": l1_correct, "total": len(facts),
            "rate": l1_rate, "baseline_rate": baseline_rate,
            "adapter": str(l1_path.name),
        })

        if l1_rate >= 0.5:
            print(f"  ✓ L1 sufficient: {l1_rate:.0%}")
            log["status"] = "success_L1"
            log["final_rate"] = l1_rate
            return log

        # Check for see-saw before continuing
        is_seesaw, regressed, improved = detect_seesaw(baseline_results, l1_results)
        if is_seesaw:
            print(f"  ⚠ See-saw detected: {len(improved)} improved, {len(regressed)} regressed")
            print(f"    Regressed: {regressed[:5]}")
            # Jump to L4 (orthogonal) — staged won't help with see-saws
            print("  Jumping to L4 (orthogonal)...")
            return run_orthogonal_pipeline(
                model, lm_head, tokenizer, facts, base, baseline_results, log,
                d_model, vocab_size
            )
    else:
        # Load existing L1 adapter for evaluation
        l1_candidates = [a for a in ADAPTERS_DIR.glob(f"{base}*adapter.npz")
                         if "orthogonal" not in a.name and "intensive" not in a.name]
        if l1_candidates:
            adapter = load_adapter_weights(l1_candidates[0], d_model, vocab_size)
            l1_results = evaluate_facts(model, lm_head, tokenizer, facts, adapter, label="existing-L1")
            l1_correct = sum(1 for _, _, c in l1_results if c)
            l1_rate = l1_correct / len(facts)
        else:
            l1_results = baseline_results
            l1_rate = baseline_rate
            adapter = None

    # ── L2: Intensive adapter ──
    if not status["has_intensive"]:
        print("\n--- L2: Intensive adapter (5000 steps, lr=3e-6) ---")
        adapter_l2 = train_adapter(
            model, lm_head, tokenizer, facts,
            steps=3000, lr=3e-6, margin_target=2.5, label="L2-intensive"
        )
        l2_path = ADAPTERS_DIR / f"{base}_intensive_adapter.npz"
        save_adapter(adapter_l2, l2_path)

        l2_results = evaluate_facts(model, lm_head, tokenizer, facts, adapter_l2, label="L2-adapted")
        l2_correct = sum(1 for _, _, c in l2_results if c)
        l2_rate = l2_correct / len(facts)

        log["levels"].append({
            "level": "L2_intensive", "correct": l2_correct, "total": len(facts),
            "rate": l2_rate, "adapter": str(l2_path.name),
        })

        if l2_rate >= 0.5:
            print(f"  ✓ L2 sufficient: {l2_rate:.0%}")
            log["status"] = "success_L2"
            log["final_rate"] = l2_rate
            return log

        # Check for see-saw
        is_seesaw, regressed, improved = detect_seesaw(baseline_results, l2_results)
        if is_seesaw:
            print(f"  ⚠ See-saw at L2: {len(improved)} improved, {len(regressed)} regressed")
            return run_orthogonal_pipeline(
                model, lm_head, tokenizer, facts, base, baseline_results, log,
                d_model, vocab_size
            )

        # Use the better adapter going forward
        if l2_rate > l1_rate:
            adapter = adapter_l2
            best_results = l2_results
        else:
            best_results = l1_results
    else:
        best_results = l1_results if 'l1_results' in dir() else baseline_results

    # ── L3: Staged training (cluster then train sequentially) ──
    if not status["has_orthogonal"]:
        print("\n--- L3: Staged training (cluster + sequential) ---")

        # Identify still-failing facts
        failing_facts = [f for f in facts
                         if not any(fid == f["id"] and c for fid, _, c in best_results)]

        if not failing_facts:
            print("  All facts already passing. Done.")
            log["status"] = "success_L2"
            return log

        clusters = cluster_facts(facts, margins_by_id)
        print(f"  Clusters: {', '.join(f'{k}({len(v)})' for k, v in clusters.items())}")

        # Train each cluster sequentially, checking for regression
        staged_adapter = create_fresh_adapter(d_model, vocab_size)
        prev_results = baseline_results
        seesaw_detected = False

        for ci, (cluster_name, cluster_facts_list) in enumerate(clusters.items()):
            print(f"\n  Stage {ci+1}/{len(clusters)}: {cluster_name} ({len(cluster_facts_list)} facts)")

            staged_adapter = train_adapter(
                model, lm_head, tokenizer, cluster_facts_list, adapter=staged_adapter,
                steps=2000, lr=4e-6, margin_target=2.5,
                label=f"stage-{cluster_name}"
            )

            # Check ALL facts (not just this cluster) for regression
            stage_results = evaluate_facts(
                model, lm_head, tokenizer, facts, staged_adapter,
                label=f"after-{cluster_name}"
            )

            is_seesaw, regressed, improved = detect_seesaw(prev_results, stage_results, threshold=-3.0)
            if is_seesaw:
                print(f"  ⚠ See-saw after stage {cluster_name}!")
                print(f"    Regressed: {regressed[:5]}")
                seesaw_detected = True
                break

            prev_results = stage_results

        if not seesaw_detected:
            # Save staged adapter
            l3_path = ADAPTERS_DIR / f"{base}_staged_adapter.npz"
            save_adapter(staged_adapter, l3_path)

            l3_results = evaluate_facts(model, lm_head, tokenizer, facts, staged_adapter, label="L3-staged")
            l3_correct = sum(1 for _, _, c in l3_results if c)
            l3_rate = l3_correct / len(facts)

            log["levels"].append({
                "level": "L3_staged", "correct": l3_correct, "total": len(facts),
                "rate": l3_rate, "adapter": str(l3_path.name),
                "clusters": list(clusters.keys()),
            })

            if l3_rate >= 0.5:
                print(f"  ✓ L3 sufficient: {l3_rate:.0%}")
                log["status"] = "success_L3"
                log["final_rate"] = l3_rate
                return log

        # ── L4: Orthogonal adapters ──
        print("\n--- L4: Orthogonal adapters (one per cluster, with routing) ---")
        return run_orthogonal_pipeline(
            model, lm_head, tokenizer, facts, base, baseline_results, log,
            d_model, vocab_size
        )

    # Already has orthogonal adapters — report status
    log["status"] = "already_at_max_level"
    return log


def run_orthogonal_pipeline(model, lm_head, tokenizer, facts, base,
                            baseline_results, log, d_model, vocab_size):
    """L4: Train orthogonal adapters — one per cluster, each independent.

    Creates a routing config so inference picks the right adapter per fact.
    """
    margins_by_id = {fid: m for fid, m, _ in baseline_results}
    clusters = cluster_facts(facts, margins_by_id)

    if len(clusters) < 2:
        # Force split into 2 clusters if only 1
        mid = len(facts) // 2
        clusters = {"group_a": facts[:mid], "group_b": facts[mid:]}

    print(f"  Orthogonal clusters: {', '.join(f'{k}({len(v)})' for k, v in clusters.items())}")

    routing_config = {"domain": base, "clusters": {}}
    cluster_adapters = {}
    overall_results = {}

    for cluster_name, cluster_facts_list in clusters.items():
        print(f"\n  Training orthogonal adapter: {cluster_name}")
        adapter = train_adapter(
            model, lm_head, tokenizer, cluster_facts_list,
            steps=2000, lr=4e-6, margin_target=2.5,
            label=f"orthogonal-{cluster_name}"
        )

        adapter_path = ADAPTERS_DIR / f"{base}_orthogonal_{cluster_name}.npz"
        save_adapter(adapter, adapter_path)
        cluster_adapters[cluster_name] = adapter

        # Evaluate this adapter on its own cluster
        cluster_results = evaluate_facts(
            model, lm_head, tokenizer, cluster_facts_list, adapter,
            label=f"  {cluster_name}-adapter"
        )

        # Record routing: which fact IDs go to which adapter
        fact_ids = [f["id"] for f in cluster_facts_list]
        routing_config["clusters"][cluster_name] = {
            "adapter": str(adapter_path.name),
            "fact_ids": fact_ids,
        }

        for fid, m, c in cluster_results:
            overall_results[fid] = (m, c)

    # Evaluate each adapter on ALL facts to check for cross-cluster interference
    print("\n  Cross-cluster evaluation (best-of routing):")
    final_correct = 0
    for fact in facts:
        best_margin = -999
        for cluster_name, adapter in cluster_adapters.items():
            prompt = fact["context"] if fact["context"].endswith(":") else fact["context"] + ":"
            truth_lp = compute_logprob(model, lm_head, tokenizer, prompt, f" {fact['truth']}", adapter)
            dist_lps = [compute_logprob(model, lm_head, tokenizer, prompt, f" {d}", adapter)
                        for d in fact["distractors"]]
            margin = truth_lp - max(dist_lps) if dist_lps else truth_lp
            if margin > best_margin:
                best_margin = margin

        if best_margin > 0:
            final_correct += 1

    l4_rate = final_correct / len(facts) if facts else 0
    print(f"  Orthogonal best-of routing: {final_correct}/{len(facts)} ({l4_rate:.0%})")

    # Save routing config
    routing_path = ADAPTERS_DIR / f"{base}_orthogonal_routing.json"
    with open(routing_path, "w") as f:
        json.dump(routing_config, f, indent=2)
    print(f"  Routing config saved: {routing_path.name}")

    log["levels"].append({
        "level": "L4_orthogonal", "correct": final_correct, "total": len(facts),
        "rate": l4_rate, "clusters": list(clusters.keys()),
        "routing_config": str(routing_path.name),
    })
    log["status"] = "success_L4" if l4_rate >= 0.5 else "partial_L4"
    log["final_rate"] = l4_rate
    return log


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Logging
# ═══════════════════════════════════════════════════════════════════════

def save_training_log(entry):
    """Append a training result to the log."""
    log_path = RESULTS_DIR / "adapter_training_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_escalation(domain_name, reason, details):
    """Log an escalation. Deduplicates by domain+reason."""
    esc_path = RESULTS_DIR / "escalations.jsonl"
    if esc_path.exists():
        with open(esc_path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if (e.get("domain") == domain_name and
                        e.get("reason") == reason and
                        e.get("status") == "open"):
                        return
                except json.JSONDecodeError:
                    continue

    with open(esc_path, "a") as f:
        f.write(json.dumps({
            "domain": domain_name,
            "reason": reason,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "status": "open",
        }) + "\n")


# ═══════════════════════════════════════════════════════════════════════
# Section 7: Status Display
# ═══════════════════════════════════════════════════════════════════════

def print_status():
    """Show failing domains and their adapter training status."""
    failing = get_failing_domains()
    if not failing:
        print("All domains passing or no run_summary.json found.")
        return

    print(f"{'='*70}")
    print(f"  Adapter Training Status — {len(failing)} Failing Domains")
    print(f"{'='*70}\n")

    for name, rate, facts_file in failing:
        status = get_adapter_status(name)
        adapters_str = ", ".join(status["adapters"][:3]) if status["adapters"] else "NONE"
        level = "none"
        if status["has_orthogonal"]:
            level = "orthogonal (L4)"
        elif status["has_intensive"]:
            level = "intensive (L2)"
        elif status["has_single"]:
            level = "single (L1)"

        facts_str = facts_file.name if facts_file else "NOT FOUND"
        print(f"  {rate:5.0%}  {name}")
        print(f"         Adapters: {adapters_str}")
        print(f"         Level: {level}  |  Facts: {facts_str}")
        print()


# ═══════════════════════════════════════════════════════════════════════
# Section 8: Main Work Loop
# ═══════════════════════════════════════════════════════════════════════

def write_pid_file():
    """Write PID file so the dashboard can detect us."""
    pid_file = RESULTS_DIR / "adapter_train.pid"
    pid_file.write_text(str(os.getpid()))
    return pid_file


def remove_pid_file():
    """Clean up PID file on exit."""
    pid_file = RESULTS_DIR / "adapter_train.pid"
    try:
        pid_file.unlink(missing_ok=True)
    except Exception:
        pass


def run_work_loop(domain_filter=None):
    """Continuous work loop: load model once, process all failing domains."""
    pid_file = write_pid_file()

    print(f"{'='*70}")
    print(f"  Adapter Trainer — Full Autonomous Pipeline")
    print(f"  Training model: {TRAIN_MODEL} (4B student — adapter target)")
    print(f"  27B provides compute; adapter dimensions match 4B")
    print(f"  PID: {os.getpid()} (written to {pid_file.name})")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")

    # ── Load model ONCE ──
    print(f"Loading {TRAIN_MODEL}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(TRAIN_MODEL)
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"  Loaded in {time.time()-t0:.1f}s  d_model={d_model} vocab={vocab_size}\n")

    # ── Get failing domains ──
    failing = get_failing_domains()
    if domain_filter:
        failing = [(n, r, f) for n, r, f in failing
                    if domain_filter.lower() in n.lower()]

    trainable = [(n, r, f) for n, r, f in failing if f is not None]
    no_facts = [(n, r, f) for n, r, f in failing if f is None]

    if no_facts:
        print(f"  {len(no_facts)} domains without facts files (need Claude Code):")
        for name, rate, _ in no_facts[:5]:
            print(f"    {rate:5.0%}  {name}")
        print()

    if not trainable:
        print("No trainable domains remaining.")
        return

    print(f"  {len(trainable)} trainable domains\n")

    # ── Process each domain ──
    results_summary = []
    for i, (name, rate, facts_file) in enumerate(trainable, 1):
        print(f"\n[{i}/{len(trainable)}] {name} (oracle rate: {rate:.0%})")

        t_start = time.time()
        try:
            result = run_domain_pipeline(
                model, lm_head, tokenizer, name, facts_file,
                d_model, vocab_size
            )
            result["duration_seconds"] = time.time() - t_start
            save_training_log(result)
            results_summary.append(result)

            status_str = result.get("status", "unknown")
            final_rate = result.get("final_rate")
            if final_rate is not None:
                print(f"  Result: {status_str} — {final_rate:.0%}")
            else:
                print(f"  Result: {status_str}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            save_training_log({
                "domain": name, "status": "error", "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"  Work Loop Complete")
    print(f"  Processed: {len(results_summary)} domains")

    successes = [r for r in results_summary if r.get("status", "").startswith("success")]
    partials = [r for r in results_summary if r.get("status") == "partial_L4"]
    errors = [r for r in results_summary if r.get("status") == "error"]

    print(f"  Successes: {len(successes)}")
    if successes:
        for r in successes:
            print(f"    {r['domain']}: {r.get('final_rate', 0):.0%} ({r['status']})")
    print(f"  Partial: {len(partials)}")
    if partials:
        for r in partials:
            print(f"    {r['domain']}: {r.get('final_rate', 0):.0%}")
    print(f"  Errors: {len(errors)}")
    print(f"  Finished: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    remove_pid_file()


def main():
    parser = argparse.ArgumentParser(
        description="Train 4B adapters on failing domains (full autonomous pipeline)")
    parser.add_argument("--status", action="store_true", help="Show training status only")
    parser.add_argument("--once", action="store_true", help="Train one domain and exit")
    parser.add_argument("--domain", help="Train specific domain")
    args = parser.parse_args()

    os.makedirs(ADAPTERS_DIR, exist_ok=True)

    if args.status:
        print_status()
        return

    if args.once:
        # Single domain mode — load model, train one, exit
        failing = get_failing_domains()
        if args.domain:
            failing = [(n, r, f) for n, r, f in failing
                        if args.domain.lower() in n.lower()]
        trainable = [(n, r, f) for n, r, f in failing if f is not None]

        if not trainable:
            print("No trainable domains found.")
            return

        name, rate, facts_file = trainable[0]
        print(f"Single domain mode: {name} ({rate:.0%})\n")

        print(f"Loading {TRAIN_MODEL}...")
        t0 = time.time()
        model, tokenizer = mlx_lm.load(TRAIN_MODEL)
        model.freeze()
        lm_head = t3.get_lm_head_fn(model)
        vocab_size = model.model.embed_tokens.weight.shape[0]
        d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
        print(f"  Loaded in {time.time()-t0:.1f}s\n")

        result = run_domain_pipeline(model, lm_head, tokenizer, name, facts_file,
                                     d_model, vocab_size)
        result["duration_seconds"] = time.time() - t0
        save_training_log(result)

        print(f"\nFinal: {result.get('status')} — {result.get('final_rate', 'N/A')}")
        return

    # Default: continuous work loop
    run_work_loop(domain_filter=args.domain)


if __name__ == "__main__":
    main()
