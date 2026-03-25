#!/usr/bin/env python3
"""Adapter Transfer Benchmark — Do oracle-trained adapters improve external benchmarks?

Tests whether domain-specific adapters (trained on oracle facts) transfer to
MMLU subcategories and TruthfulQA that they were never trained on.

Experiments:
  1. BASELINE:       No adapters
  2. SCALED STACK:   All domain adapters, shifts averaged (1/N)
  3. DOMAIN-ROUTED:  Only domain-matched adapters → matched MMLU subjects (scaled)
  4. INDIVIDUAL:     Each adapter individually on matched subjects (sample top-5 per domain)
  5. TruthfulQA:     LLM science stack + all-scaled on TruthfulQA MC2

Usage:
    python experiments/adapter_transfer_benchmark.py
    python experiments/adapter_transfer_benchmark.py --n-mmlu 200 --n-tqa 100
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import mlx_lm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noethersolve.adapter import SnapOnConfig, SnapOnLogitMLP
from noethersolve.train_utils import (
    LOGIT_SOFTCAP,
    apply_adapter_stack,
    get_lm_head_fn,
)

# ── Configuration ────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-14B-Base"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapters" / "qwen3_4b_base"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "transfer_benchmark"

MMLU_CATEGORIES = {
    "physics": ["astronomy", "college_physics", "high_school_physics", "conceptual_physics"],
    "chemistry": ["college_chemistry", "high_school_chemistry"],
    "biology": ["college_biology", "high_school_biology", "anatomy", "clinical_knowledge",
                 "medical_genetics", "college_medicine", "nutrition", "virology"],
    "math": ["abstract_algebra", "college_mathematics", "high_school_mathematics",
             "high_school_statistics", "elementary_mathematics"],
    "cs": ["college_computer_science", "high_school_computer_science",
           "computer_security", "machine_learning"],
    "logic": ["formal_logic", "logical_fallacies"],
    "other": [],
}

ADAPTER_STACKS = {
    "physics": [
        "vortex_pair_", "hamiltonian_stage5", "ns_regularity_prior_broken",
        "em_adapter_v4", "chemical_adapter", "chem_enzyme_focused",
        "kinetic_k_", "continuous_qf_", "optimal_f_", "3body_",
        "qf_ratio_adapter", "qf_continuous_adapter",
    ],
    "genetics": [
        "genetics_", "disease_targets_", "protein_structure_",
        "immune_evasion_", "delivery_optimization_", "safety_invariants_",
        "clinical_translation_",
    ],
    "math": [
        "millennium_problems_", "number_theory_conjectures_",
        "algebra_topology_conjectures_", "proof_techniques_",
        "analysis_pde_conjectures_", "computational_conjectures_",
    ],
    "llm_science": [
        "llm_hallucination_", "llm_reasoning_", "llm_alignment_",
        "llm_training_", "llm_evaluation_", "llm_context_memory_",
    ],
    "pl": [
        "pl_type_systems_", "pl_memory_", "pl_concurrency_",
        "pl_paradigms_", "pl_compilers_", "pl_pitfalls_",
    ],
}

DOMAIN_TO_MMLU = {
    "physics": ["physics", "chemistry"],
    "genetics": ["biology"],
    "math": ["math"],
    "llm_science": ["cs"],
    "pl": ["cs", "logic"],
}


@dataclass
class BenchmarkResult:
    name: str
    n_correct: int = 0
    n_total: int = 0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total > 0 else 0.0

    @property
    def pct(self) -> str:
        return f"{self.accuracy * 100:.1f}%"


# ── Data Loading ─────────────────────────────────────────────────────────

def load_mmlu(n: int = 200, seed: int = 42) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(ds))[:n]
    return [
        {"question": ds[int(i)]["question"], "choices": ds[int(i)]["choices"],
         "answer": ds[int(i)]["answer"], "subject": ds[int(i)]["subject"]}
        for i in indices
    ]


def load_truthfulqa(n: int = 200, seed: int = 42) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(ds))[:n]
    return [
        {"question": ds[int(i)]["question"], "mc2_targets": ds[int(i)]["mc2_targets"]}
        for i in indices
    ]


def get_mmlu_category(subject: str) -> str:
    for cat, subjects in MMLU_CATEGORIES.items():
        if subject in subjects:
            return cat
    return "other"


# ── Efficient Scoring ────────────────────────────────────────────────────

def _get_base_logits(model, lm_head, tokens):
    """Get base model logits (cached for reuse across adapter conditions)."""
    h = model.model(tokens)
    mx.eval(h)
    base_logits = lm_head(h)
    mx.eval(base_logits)
    return base_logits


def _apply_scaled_stack(base_logits, adapters, scale=1.0):
    """Apply adapter stack with optional scaling."""
    if not adapters:
        return base_logits
    if scale == 1.0:
        return apply_adapter_stack(adapters, base_logits)
    # Average shifts
    total_shift = mx.zeros_like(base_logits)
    for adapter in adapters:
        shift = adapter(base_logits)
        center = shift.mean(axis=-1, keepdims=True)
        total_shift = total_shift + (shift - center)
    total_shift = total_shift * scale
    return base_logits + LOGIT_SOFTCAP * mx.tanh(total_shift / LOGIT_SOFTCAP)


def _logprob_from_logits(logits_np, full_ids, n_prompt):
    """Compute sum log-prob from numpy logits."""
    total_lp = 0.0
    for i, tok_id in enumerate(full_ids[n_prompt:]):
        pos = n_prompt - 1 + i
        row = logits_np[pos]
        lse = float(np.log(np.sum(np.exp(row - row.max())) + 1e-8) + row.max())
        total_lp += float(row[tok_id]) - lse
    return total_lp


def score_mmlu_multi(
    model, tokenizer, lm_head, question: dict,
    conditions: Dict[str, tuple],  # name → (adapters, scale) or None
) -> Dict[str, bool]:
    """Score one MMLU question under multiple adapter conditions efficiently.

    Shares the base model forward pass across all conditions.
    Returns dict of condition_name → correct.
    """
    q = question["question"]
    choices = question["choices"]
    correct_idx = question["answer"]

    labels = ["A", "B", "C", "D"]
    prompt = f"Question: {q}\n"
    for label, choice in zip(labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "Answer:"

    # Pre-tokenize all completions
    prompt_ids = tokenizer.encode(prompt)
    n_prompt = len(prompt_ids)
    completions = []
    for label in labels:
        full_ids = tokenizer.encode(prompt + f" {label}")
        completions.append(full_ids)

    # We need base logits for the full sequence of each completion
    # Since completions differ by only 1 token, use the longest
    max_len = max(len(c) for c in completions)
    tokens = mx.array(completions[0][:max_len])[None, :]

    # Get base model hidden states + logits (one forward pass)
    h = model.model(tokens)
    mx.eval(h)
    base_logits = lm_head(h)
    mx.eval(base_logits)
    base_logits_np = np.array(base_logits[0].astype(mx.float32))

    results = {}
    for cond_name, cond_val in conditions.items():
        if cond_val is None:
            # Baseline — use base logits directly
            logits_np = base_logits_np
        else:
            adapters, scale = cond_val
            if not adapters:
                logits_np = base_logits_np
            else:
                adapted = _apply_scaled_stack(base_logits, adapters, scale)
                mx.eval(adapted)
                logits_np = np.array(adapted[0].astype(mx.float32))

        # Score each choice using the answer token logit
        lps = []
        for full_ids in completions:
            lp = _logprob_from_logits(logits_np, full_ids, n_prompt)
            lps.append(lp)

        results[cond_name] = (int(np.argmax(lps)) == correct_idx)

    return results


def score_tqa_multi(
    model, tokenizer, lm_head, question: dict,
    conditions: Dict[str, tuple],
) -> Dict[str, float]:
    """Score one TruthfulQA MC2 under multiple conditions."""
    q = question["question"]
    targets = question["mc2_targets"]
    choices = targets["choices"]
    label_vals = targets["labels"]

    prompt = f"Q: {q}\nA:"
    prompt_ids = tokenizer.encode(prompt)
    n_prompt = len(prompt_ids)

    # Tokenize all choices
    choice_tokens = []
    for choice in choices:
        full_ids = tokenizer.encode(prompt + f" {choice}")
        choice_tokens.append(full_ids)

    # Use longest for base forward pass
    max(len(c) for c in choice_tokens)
    # We need separate forward passes per choice since they're different lengths
    # But we can share adapter computation

    results = {}
    for cond_name, cond_val in conditions.items():
        lps = []
        for full_ids in choice_tokens:
            tokens = mx.array(full_ids)[None, :]
            if cond_val is None:
                logits = model(tokens)
                mx.eval(logits)
            else:
                adapters, scale = cond_val
                if not adapters:
                    logits = model(tokens)
                    mx.eval(logits)
                else:
                    h = model.model(tokens)
                    mx.eval(h)
                    bl = lm_head(h)
                    mx.eval(bl)
                    logits = _apply_scaled_stack(bl, adapters, scale)
                    mx.eval(logits)

            logits_np = np.array(logits[0].astype(mx.float32))
            lp = _logprob_from_logits(logits_np, full_ids, n_prompt)
            lps.append(lp)

        lps_arr = np.array(lps)
        lps_arr = lps_arr - lps_arr.max()
        probs = np.exp(lps_arr)
        probs = probs / probs.sum()
        mc2 = sum(probs[i] for i, lab in enumerate(label_vals) if lab == 1)
        results[cond_name] = float(mc2)

    return results


# ── Adapter Loading ──────────────────────────────────────────────────────

def load_adapter_stack(prefixes, vocab_size, d_inner=64):
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


def load_individual_adapters(prefixes, vocab_size, d_inner=64):
    result = {}
    for npz_path in sorted(ADAPTER_DIR.glob("*.npz")):
        name = npz_path.name
        for prefix in prefixes:
            if name.startswith(prefix):
                try:
                    weights = mx.load(str(npz_path))
                    config = SnapOnConfig(d_inner=d_inner, vocab_size=vocab_size, mode="logit")
                    adapter = SnapOnLogitMLP(config)
                    adapter.load_weights(list(weights.items()))
                    result[name] = adapter
                except Exception:
                    pass
                break
    return result


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Adapter Transfer Benchmark")
    parser.add_argument("--n-mmlu", type=int, default=200)
    parser.add_argument("--n-tqa", type=int, default=100)
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tqa", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"{'='*70}")
    print("  Adapter Transfer Benchmark")
    print(f"  Model: {MODEL_ID}")
    print(f"  MMLU: {args.n_mmlu} | TQA: {'skip' if args.skip_tqa else args.n_tqa}")
    print(f"{'='*70}")

    # Load data
    print("\nLoading MMLU...")
    mmlu_qs = load_mmlu(n=args.n_mmlu, seed=args.seed)
    cats = defaultdict(int)
    for q in mmlu_qs:
        cats[get_mmlu_category(q["subject"])] += 1
    print(f"  {len(mmlu_qs)} questions: {dict(cats)}")

    tqa_qs = []
    if not args.skip_tqa:
        print("Loading TruthfulQA...")
        tqa_qs = load_truthfulqa(n=args.n_tqa, seed=args.seed)
        print(f"  {len(tqa_qs)} questions")

    # Load model
    print(f"\nLoading {MODEL_ID}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(MODEL_ID)
    model.eval()
    lm_head = get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]
    print(f"  Loaded in {time.time()-t0:.1f}s, vocab={vocab_size}")

    # Load all domain adapter stacks
    domain_stacks = {}
    for domain, prefixes in ADAPTER_STACKS.items():
        adapters = load_adapter_stack(prefixes, vocab_size, args.d_inner)
        if adapters:
            domain_stacks[domain] = adapters
            print(f"  {domain}: {len(adapters)} adapters")
        else:
            print(f"  {domain}: 0 adapters (skipped)")

    all_results = {
        "model": MODEL_ID, "n_mmlu": args.n_mmlu, "n_tqa": args.n_tqa,
        "seed": args.seed, "experiments": {},
    }

    # ═══════════════════════════════════════════════════════════════════
    # Build conditions for multi-scoring
    # For each MMLU question, we test:
    #   baseline, each domain scaled, domain-routed scaled
    # ═══════════════════════════════════════════════════════════════════

    # Category → matching domain
    cat_to_domain = {}
    for domain, mmlu_cats in DOMAIN_TO_MMLU.items():
        if domain in domain_stacks:
            for cat in mmlu_cats:
                cat_to_domain[cat] = domain

    # Build per-question conditions
    print(f"\n{'═'*70}")
    print(f"  Running MMLU ({len(mmlu_qs)} questions × multiple conditions)")
    print(f"{'═'*70}")

    # Track results per condition
    condition_results = defaultdict(lambda: defaultdict(lambda: BenchmarkResult(name="")))
    # Initialize overall counters
    condition_names = ["baseline"]
    for domain in domain_stacks:
        condition_names.append(f"scaled_{domain}")
    condition_names.append("routed_scaled")
    for cn in condition_names:
        condition_results[cn]["overall"] = BenchmarkResult(name="overall")

    t0 = time.time()
    for i, q in enumerate(mmlu_qs):
        cat = get_mmlu_category(q["subject"])

        # Build conditions for this question
        conditions = {"baseline": None}

        # Each domain stack (scaled 1/N) applied to ALL questions
        for domain, adapters in domain_stacks.items():
            scale = 1.0 / len(adapters)
            conditions[f"scaled_{domain}"] = (adapters, scale)

        # Domain-routed: only matched domain, scaled
        matched_domain = cat_to_domain.get(cat)
        if matched_domain and matched_domain in domain_stacks:
            adapters = domain_stacks[matched_domain]
            scale = 1.0 / len(adapters)
            conditions["routed_scaled"] = (adapters, scale)
        else:
            conditions["routed_scaled"] = None  # baseline for unmatched

        # Score all conditions in one pass
        results = score_mmlu_multi(model, tokenizer, lm_head, q, conditions)

        for cond_name, correct in results.items():
            condition_results[cond_name]["overall"].n_total += 1
            condition_results[cond_name]["overall"].n_correct += int(correct)
            if cat not in condition_results[cond_name]:
                condition_results[cond_name][cat] = BenchmarkResult(name=cat)
            condition_results[cond_name][cat].n_total += 1
            condition_results[cond_name][cat].n_correct += int(correct)

        if (i + 1) % 50 == 0:
            base_acc = condition_results["baseline"]["overall"].pct
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(mmlu_qs)}: baseline={base_acc}, "
                  f"elapsed={elapsed:.0f}s")

    mmlu_time = time.time() - t0
    print(f"\n  MMLU complete in {mmlu_time:.1f}s")

    # Print MMLU results table
    print(f"\n  {'Condition':30s} {'Overall':>8s} {'Delta':>8s} ", end="")
    for cat in sorted(set(get_mmlu_category(q["subject"]) for q in mmlu_qs)):
        print(f" {cat[:6]:>7s}", end="")
    print()
    print(f"  {'─'*30} {'─'*8} {'─'*8}", end="")
    for cat in sorted(set(get_mmlu_category(q["subject"]) for q in mmlu_qs)):
        print(f" {'─'*7}", end="")
    print()

    base_overall = condition_results["baseline"]["overall"].accuracy
    sorted_cats = sorted(set(get_mmlu_category(q["subject"]) for q in mmlu_qs))

    for cond_name in condition_names:
        r = condition_results[cond_name]
        overall = r["overall"].accuracy
        delta = overall - base_overall
        if cond_name == "baseline":
            delta_str = "—"
        else:
            s = "+" if delta >= 0 else ""
            delta_str = f"{s}{delta*100:.1f}%"

        print(f"  {cond_name:30s} {overall*100:>7.1f}% {delta_str:>8s}", end="")
        for cat in sorted_cats:
            if cat in r and r[cat].n_total > 0:
                print(f" {r[cat].accuracy*100:>6.1f}%", end="")
            else:
                print(f" {'—':>7s}", end="")
        print()

    # Save MMLU results
    for cond_name in condition_names:
        r = condition_results[cond_name]
        all_results["experiments"][cond_name] = {
            "mmlu_overall": r["overall"].accuracy,
            "mmlu_n_correct": r["overall"].n_correct,
            "mmlu_n_total": r["overall"].n_total,
            "mmlu_delta": r["overall"].accuracy - base_overall,
            "mmlu_by_cat": {
                c: {"acc": cr.accuracy, "n": cr.n_total}
                for c, cr in r.items()
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    # INDIVIDUAL ADAPTERS — test each single adapter on matched subjects
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("  Individual Adapter Test (on matched subjects)")
    print(f"{'═'*70}")

    for domain, prefixes in ADAPTER_STACKS.items():
        if domain not in domain_stacks:
            continue

        matched_cats = DOMAIN_TO_MMLU.get(domain, [])
        matched_qs = [q for q in mmlu_qs if get_mmlu_category(q["subject"]) in matched_cats]
        if len(matched_qs) < 3:
            print(f"\n  {domain}: only {len(matched_qs)} matched questions, skip")
            continue

        # Get baseline on matched
        base_matched = sum(
            1 for q in matched_qs
            if score_mmlu_multi(model, tokenizer, lm_head, q, {"b": None})["b"]
        )
        base_acc = base_matched / len(matched_qs)

        singles = load_individual_adapters(prefixes, vocab_size, args.d_inner)
        if not singles:
            continue

        print(f"\n  {domain}: {len(singles)} adapters × {len(matched_qs)} questions "
              f"(baseline: {base_acc*100:.1f}%)")

        adapter_results = []
        for aname, adapter in sorted(singles.items()):
            n_correct = sum(
                1 for q in matched_qs
                if score_mmlu_multi(model, tokenizer, lm_head, q,
                                     {"a": ([adapter], 1.0)})["a"]
            )
            acc = n_correct / len(matched_qs)
            delta = acc - base_acc
            adapter_results.append({"name": aname, "acc": acc, "delta": delta,
                                     "n_correct": n_correct, "n_total": len(matched_qs)})
            s = "+" if delta >= 0 else ""
            marker = " ★" if delta > 0 else " ✗" if delta < 0 else ""
            print(f"    {aname:50s} {acc*100:5.1f}% ({s}{delta*100:.1f}%){marker}")

        adapter_results.sort(key=lambda x: x["delta"], reverse=True)
        all_results["experiments"][f"individual_{domain}"] = {
            "n_matched": len(matched_qs),
            "baseline": base_acc,
            "adapters": adapter_results,
        }

    # ═══════════════════════════════════════════════════════════════════
    # TruthfulQA
    # ═══════════════════════════════════════════════════════════════════
    if tqa_qs:
        print(f"\n{'═'*70}")
        print(f"  TruthfulQA MC2 ({len(tqa_qs)} questions)")
        print(f"{'═'*70}")

        # Build TQA conditions
        tqa_conditions = {"baseline": None}
        if "llm_science" in domain_stacks:
            adapters = domain_stacks["llm_science"]
            tqa_conditions["llm_scaled"] = (adapters, 1.0 / len(adapters))
        # All adapters scaled
        all_adapters = []
        for adapters in domain_stacks.values():
            all_adapters.extend(adapters)
        if all_adapters:
            tqa_conditions["all_scaled"] = (all_adapters, 1.0 / len(all_adapters))

        tqa_scores = defaultdict(list)
        t0 = time.time()
        for i, q in enumerate(tqa_qs):
            results = score_tqa_multi(model, tokenizer, lm_head, q, tqa_conditions)
            for cond_name, mc2 in results.items():
                tqa_scores[cond_name].append(mc2)

            if (i + 1) % 50 == 0:
                base_mc2 = np.mean(tqa_scores["baseline"])
                print(f"  {i+1}/{len(tqa_qs)}: baseline MC2={base_mc2:.4f}, "
                      f"elapsed={time.time()-t0:.0f}s")

        print(f"\n  TQA complete in {time.time()-t0:.1f}s")

        base_mc2 = np.mean(tqa_scores["baseline"])
        print(f"\n  {'Condition':30s} {'MC2':>8s} {'Delta':>8s}")
        print(f"  {'─'*30} {'─'*8} {'─'*8}")
        print(f"  {'baseline':30s} {base_mc2:>8.4f} {'—':>8s}")

        for cond_name in sorted(tqa_scores.keys()):
            if cond_name == "baseline":
                continue
            mc2 = np.mean(tqa_scores[cond_name])
            d = mc2 - base_mc2
            s = "+" if d >= 0 else ""
            print(f"  {cond_name:30s} {mc2:>8.4f} {s}{d:>.4f}")
            all_results["experiments"][f"tqa_{cond_name}"] = {
                "mc2": float(mc2), "delta": float(d),
            }
        all_results["experiments"]["tqa_baseline"] = {"mc2": float(base_mc2)}

    # ═══════════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════════
    out_dir = Path(args.output) if args.output else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"transfer_benchmark_n{args.n_mmlu}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
