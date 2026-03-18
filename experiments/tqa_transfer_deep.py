#!/usr/bin/env python3
"""Deep TruthfulQA transfer test — do oracle fact adapters improve truth preference?

Tests the hypothesis: adapters trained on domain-specific facts (physics, math,
bio) improve general misconception detection on TruthfulQA, even though TQA
topics are completely unrelated to the training facts.

Runs full TruthfulQA (817 questions) with bootstrap confidence intervals.
Only tests baseline vs all_scaled to minimize memory.

Usage:
    python experiments/tqa_transfer_deep.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import mlx.core as mx
import mlx_lm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noethersolve.adapter import SnapOnConfig, SnapOnLogitMLP
from noethersolve.train_utils import LOGIT_SOFTCAP, get_lm_head_fn

MODEL_ID = "Qwen/Qwen3-4B-Base"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapters"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "transfer_benchmark"

# Only load domains that showed positive TQA signal and don't OOM
# Skip PL (wrong weight format) and LLM (0 loaded last time)
ADAPTER_PREFIXES = [
    # Physics (37 adapters)
    "vortex_pair_", "hamiltonian_stage5", "ns_regularity_prior_broken",
    "em_adapter_v4", "chemical_adapter", "chem_enzyme_focused",
    "kinetic_k_", "continuous_qf_", "optimal_f_", "3body_",
    "qf_ratio_adapter", "qf_continuous_adapter",
    # Genetics (30 adapters)
    "genetics_", "disease_targets_", "protein_structure_",
    "immune_evasion_", "delivery_optimization_", "safety_invariants_",
    "clinical_translation_",
    # Math (22 adapters)
    "millennium_problems_", "number_theory_conjectures_",
    "algebra_topology_conjectures_", "proof_techniques_",
    "analysis_pde_conjectures_", "computational_conjectures_",
]


def load_adapters(prefixes, vocab_size, d_inner=64):
    adapters = []
    loaded = set()
    for npz_path in sorted(ADAPTER_DIR.glob("*.npz")):
        name = npz_path.name
        for prefix in prefixes:
            if name.startswith(prefix) and name not in loaded:
                try:
                    weights = mx.load(str(npz_path))
                    config = SnapOnConfig(d_inner=d_inner, vocab_size=vocab_size, mode="logit")
                    adapter = SnapOnLogitMLP(config)
                    adapter.load_weights(list(weights.items()))
                    adapters.append(adapter)
                    loaded.add(name)
                except Exception:
                    pass
                break
    return adapters


def _logprob_from_np(logits_np, full_ids, n_prompt):
    total = 0.0
    for i, tok_id in enumerate(full_ids[n_prompt:]):
        pos = n_prompt - 1 + i
        row = logits_np[pos]
        lse = float(np.log(np.sum(np.exp(row - row.max())) + 1e-8) + row.max())
        total += float(row[tok_id]) - lse
    return total


def score_tqa(model, tokenizer, lm_head, question, adapters, scale):
    """Score one TQA question: baseline + adapted in one pass."""
    q = question["question"]
    targets = question["mc2_targets"]
    choices = targets["choices"]
    labels = targets["labels"]
    prompt = f"Q: {q}\nA:"
    prompt_ids = tokenizer.encode(prompt)
    n_prompt = len(prompt_ids)

    base_lps = []
    adapted_lps = []

    for choice in choices:
        full_ids = tokenizer.encode(prompt + f" {choice}")
        tokens = mx.array(full_ids)[None, :]

        # Get hidden states once
        h = model.model(tokens)
        mx.eval(h)
        bl = lm_head(h)
        mx.eval(bl)

        # Baseline logits
        base_logits_np = np.array(bl[0].astype(mx.float32))
        base_lps.append(_logprob_from_np(base_logits_np, full_ids, n_prompt))

        # Adapted logits (scaled)
        total_shift = mx.zeros_like(bl)
        for adapter in adapters:
            shift = adapter(bl)
            center = shift.mean(axis=-1, keepdims=True)
            total_shift = total_shift + (shift - center)
        total_shift = total_shift * scale
        adapted_logits = bl + LOGIT_SOFTCAP * mx.tanh(total_shift / LOGIT_SOFTCAP)
        mx.eval(adapted_logits)
        adapted_logits_np = np.array(adapted_logits[0].astype(mx.float32))
        adapted_lps.append(_logprob_from_np(adapted_logits_np, full_ids, n_prompt))

    # MC2 score for both
    def mc2(lps):
        arr = np.array(lps)
        arr = arr - arr.max()
        probs = np.exp(arr)
        probs = probs / probs.sum()
        return float(sum(probs[i] for i, lab in enumerate(labels) if lab == 1))

    return mc2(base_lps), mc2(adapted_lps)


def bootstrap_ci(scores_a, scores_b, n_bootstrap=5000, ci=0.95):
    n = len(scores_a)
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        d = np.mean([scores_b[i] for i in idx]) - np.mean([scores_a[i] for i in idx])
        deltas.append(d)
    deltas = sorted(deltas)
    lo = deltas[int((1 - ci) / 2 * n_bootstrap)]
    hi = deltas[int((1 + ci) / 2 * n_bootstrap)]
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--max-questions", type=int, default=0,
                        help="Limit questions (0 = all)")
    args = parser.parse_args()

    print(f"{'='*70}")
    print("  TruthfulQA Transfer Deep Test")
    print("  H0: oracle fact adapters don't improve TQA MC2")
    print("  H1: oracle fact adapters improve general truth preference")
    print(f"{'='*70}")

    # Load TQA
    print("\nLoading TruthfulQA...")
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    tqa_qs = [{"question": row["question"], "mc2_targets": row["mc2_targets"]} for row in ds]
    if args.max_questions > 0:
        tqa_qs = tqa_qs[:args.max_questions]
    print(f"  {len(tqa_qs)} questions")

    # Load model
    print(f"\nLoading {MODEL_ID}...")
    model, tokenizer = mlx_lm.load(MODEL_ID)
    model.eval()
    lm_head = get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]

    # Load adapters
    print("Loading adapters...")
    adapters = load_adapters(ADAPTER_PREFIXES, vocab_size, args.d_inner)
    print(f"  {len(adapters)} adapters loaded")
    scale = 1.0 / len(adapters) if adapters else 1.0
    print(f"  Scale: {scale:.6f}")

    # Score all questions
    print(f"\n{'═'*70}")
    print(f"  Scoring {len(tqa_qs)} questions (baseline + adapted)")
    print(f"{'═'*70}")

    base_scores = []
    adapted_scores = []

    t0 = time.time()
    for i, q in enumerate(tqa_qs):
        base_mc2, adapted_mc2 = score_tqa(model, tokenizer, lm_head, q, adapters, scale)
        base_scores.append(base_mc2)
        adapted_scores.append(adapted_mc2)

        if (i + 1) % 50 == 0:
            base_mean = np.mean(base_scores)
            adapted_mean = np.mean(adapted_scores)
            delta = adapted_mean - base_mean
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(tqa_qs) - i - 1)
            s = "+" if delta >= 0 else ""
            print(f"  {i+1:4d}/{len(tqa_qs)}: base={base_mean:.4f}, "
                  f"adapted={adapted_mean:.4f} ({s}{delta:.4f}), "
                  f"ETA={eta/60:.1f}m")

    total_time = time.time() - t0

    # Results
    base_mean = np.mean(base_scores)
    adapted_mean = np.mean(adapted_scores)
    delta = adapted_mean - base_mean
    lo, hi = bootstrap_ci(base_scores, adapted_scores, args.n_bootstrap)

    print(f"\n{'='*70}")
    print(f"  RESULTS (n={len(tqa_qs)}, {len(adapters)} adapters, scale={scale:.6f})")
    print(f"{'='*70}")
    print(f"  Baseline MC2:  {base_mean:.4f}")
    print(f"  Adapted MC2:   {adapted_mean:.4f}")
    print(f"  Delta:         {'+' if delta >= 0 else ''}{delta:.4f}")
    print(f"  95% CI:        [{lo:+.4f}, {hi:+.4f}]")
    sig = lo > 0 or hi < 0
    print(f"  Significant:   {'YES ★' if sig else 'no'}")
    print(f"  Time:          {total_time/60:.1f}m")

    # Per-question analysis
    improved = sum(1 for b, a in zip(base_scores, adapted_scores) if a > b + 0.05)
    degraded = sum(1 for b, a in zip(base_scores, adapted_scores) if a < b - 0.05)
    unchanged = len(base_scores) - improved - degraded
    print("\n  Per-question (>5% threshold):")
    print(f"    Improved:  {improved} ({improved/len(base_scores)*100:.1f}%)")
    print(f"    Degraded:  {degraded} ({degraded/len(base_scores)*100:.1f}%)")
    print(f"    Unchanged: {unchanged} ({unchanged/len(base_scores)*100:.1f}%)")

    # Effect size
    pooled_std = np.sqrt((np.var(base_scores) + np.var(adapted_scores)) / 2)
    cohens_d = delta / pooled_std if pooled_std > 0 else 0
    print(f"\n  Cohen's d:     {cohens_d:.4f}")
    print(f"  Effect size:   {'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model": MODEL_ID,
        "n_questions": len(tqa_qs),
        "n_adapters": len(adapters),
        "scale": scale,
        "baseline_mc2": float(base_mean),
        "adapted_mc2": float(adapted_mean),
        "delta": float(delta),
        "ci_95": [float(lo), float(hi)],
        "significant": sig,
        "cohens_d": float(cohens_d),
        "improved": improved,
        "degraded": degraded,
        "unchanged": unchanged,
        "per_question": [
            {"base": float(b), "adapted": float(a), "delta": float(a - b)}
            for b, a in zip(base_scores, adapted_scores)
        ],
    }
    out_path = RESULTS_DIR / "tqa_transfer_deep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else str(x))
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
