#!/usr/bin/env python3
"""
Frank review experiments for paper v3:
1. Run ideology-discriminating facts (Frank's topic areas) through baseline + adapters
2. Random train/held-out splits (5 splits) for confidence intervals
3. Parameter-matched linear vs SwiGLU comparison on each split
4. Fisher exact test with bootstrap CI
"""

import json
import os
import sys
import numpy as np
import random
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load as mlx_load

# ============================================================
# Adapter architectures
# ============================================================

class SwiGLUAdapter(nn.Module):
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)
        self.up_proj = nn.Linear(d_model, d_inner, bias=False)
        self.down_proj = nn.Linear(d_inner, d_model, bias=False)

    def __call__(self, h):
        return self.down_proj(nn.sigmoid(self.gate_proj(h)) * self.up_proj(h))


class LinearAdapter(nn.Module):
    """Parameter-matched linear adapter (no gating)."""
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.down = nn.Linear(d_model, d_inner, bias=False)
        self.up = nn.Linear(d_inner, d_model, bias=False)

    def __call__(self, h):
        return self.up(self.down(h))


def count_params(model):
    return sum(p.size for p in model.parameters().values() if isinstance(p, mx.array))


# ============================================================
# Scoring
# ============================================================

def get_logprob(model, tokenizer, text):
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return 0.0
    x = mx.array([tokens[:-1]])
    logits = model(x).astype(mx.float32)
    log_probs = nn.log_softmax(logits, axis=-1)
    targets = mx.array([tokens[1:]])
    token_lps = mx.take_along_axis(log_probs[0], targets[0][:, None], axis=-1).squeeze(-1)
    return float(mx.sum(token_lps))


def get_margin(model, tokenizer, fact, adapter=None):
    ctx = fact["context"]
    truth_text = f"{ctx}: {fact['truth']}"

    if adapter is not None:
        truth_lp = get_adapted_logprob(model, tokenizer, truth_text, adapter)
    else:
        truth_lp = get_logprob(model, tokenizer, truth_text)

    best_dist = -float('inf')
    for d in fact["distractors"]:
        dist_text = f"{ctx}: {d}"
        if adapter is not None:
            dlp = get_adapted_logprob(model, tokenizer, dist_text, adapter)
        else:
            dlp = get_logprob(model, tokenizer, dist_text)
        best_dist = max(best_dist, dlp)

    return truth_lp - best_dist


def get_adapted_logprob(model, tokenizer, text, adapter):
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return 0.0
    x = mx.array([tokens[:-1]])

    # Get hidden states from model
    h = model.model(x)
    if hasattr(model.model, 'norm'):
        h = model.model.norm(h)

    # Apply adapter
    h = h + adapter(h)

    # Project to logits
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(h)
    else:
        logits = model.model.embed_tokens.as_linear(h)

    logits = logits.astype(mx.float32)
    log_probs = nn.log_softmax(logits, axis=-1)
    targets = mx.array([tokens[1:]])
    token_lps = mx.take_along_axis(log_probs[0], targets[0][:, None], axis=-1).squeeze(-1)
    return float(mx.sum(token_lps))


# ============================================================
# Training
# ============================================================

def train_adapter(model, tokenizer, adapter, train_facts, anchor_facts, steps=600, lr=1.5e-4, tau=1.5):
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(adapter_params, fact_batch, anchor_batch):
        adapter.update(adapter_params)
        total_loss = mx.array(0.0)

        # Censorship loss
        for fact in fact_batch:
            ctx = fact["context"]
            truth_text = f"{ctx}: {fact['truth']}"
            tokens_t = tokenizer.encode(truth_text)
            x_t = mx.array([tokens_t[:-1]])
            h_t = model.model(x_t)
            if hasattr(model.model, 'norm'):
                h_t = model.model.norm(h_t)
            h_t = h_t + adapter(h_t)
            if hasattr(model, 'lm_head'):
                logits_t = model.lm_head(h_t)
            else:
                logits_t = model.model.embed_tokens.as_linear(h_t)
            logits_t = logits_t.astype(mx.float32)
            lp_t = nn.log_softmax(logits_t, axis=-1)
            tgt_t = mx.array([tokens_t[1:]])
            truth_lp = mx.sum(mx.take_along_axis(lp_t[0], tgt_t[0][:, None], axis=-1).squeeze(-1))

            best_dist_lp = mx.array(-1e9)
            for d in fact["distractors"]:
                dist_text = f"{ctx}: {d}"
                tokens_d = tokenizer.encode(dist_text)
                x_d = mx.array([tokens_d[:-1]])
                h_d = model.model(x_d)
                if hasattr(model.model, 'norm'):
                    h_d = model.model.norm(h_d)
                h_d = h_d + adapter(h_d)
                if hasattr(model, 'lm_head'):
                    logits_d = model.lm_head(h_d)
                else:
                    logits_d = model.model.embed_tokens.as_linear(h_d)
                logits_d = logits_d.astype(mx.float32)
                lp_d = nn.log_softmax(logits_d, axis=-1)
                tgt_d = mx.array([tokens_d[1:]])
                dist_lp = mx.sum(mx.take_along_axis(lp_d[0], tgt_d[0][:, None], axis=-1).squeeze(-1))
                best_dist_lp = mx.maximum(best_dist_lp, dist_lp)

            margin = truth_lp - best_dist_lp
            total_loss = total_loss + mx.maximum(mx.array(0.0), mx.array(tau) - margin)

        # Anchor loss (prevent regressions)
        for afact in anchor_batch:
            ctx = afact["context"]
            truth_text = f"{ctx}: {afact['truth']}"
            tokens_t = tokenizer.encode(truth_text)
            x_t = mx.array([tokens_t[:-1]])
            h_t = model.model(x_t)
            if hasattr(model.model, 'norm'):
                h_t = model.model.norm(h_t)
            h_t = h_t + adapter(h_t)
            if hasattr(model, 'lm_head'):
                logits_t = model.lm_head(h_t)
            else:
                logits_t = model.model.embed_tokens.as_linear(h_t)
            logits_t = logits_t.astype(mx.float32)
            lp_t = nn.log_softmax(logits_t, axis=-1)
            tgt_t = mx.array([tokens_t[1:]])
            truth_lp = mx.sum(mx.take_along_axis(lp_t[0], tgt_t[0][:, None], axis=-1).squeeze(-1))

            best_dist_lp = mx.array(-1e9)
            for d in afact["distractors"]:
                dist_text = f"{ctx}: {d}"
                tokens_d = tokenizer.encode(dist_text)
                x_d = mx.array([tokens_d[:-1]])
                h_d = model.model(x_d)
                if hasattr(model.model, 'norm'):
                    h_d = model.model.norm(h_d)
                h_d = h_d + adapter(h_d)
                if hasattr(model, 'lm_head'):
                    logits_d = model.lm_head(h_d)
                else:
                    logits_d = model.model.embed_tokens.as_linear(h_d)
                logits_d = logits_d.astype(mx.float32)
                lp_d = nn.log_softmax(logits_d, axis=-1)
                tgt_d = mx.array([tokens_d[1:]])
                dist_lp = mx.sum(mx.take_along_axis(lp_d[0], tgt_d[0][:, None], axis=-1).squeeze(-1))
                best_dist_lp = mx.maximum(best_dist_lp, dist_lp)

            a_margin = truth_lp - best_dist_lp
            total_loss = total_loss + mx.maximum(mx.array(0.0), mx.array(0.1) - a_margin)

        return total_loss / (len(fact_batch) + len(anchor_batch))

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    for step in range(steps):
        # Sample mini-batch
        batch = random.sample(train_facts, min(5, len(train_facts)))
        a_batch = random.sample(anchor_facts, min(3, len(anchor_facts)))

        loss, grads = loss_and_grad(adapter.parameters(), batch, a_batch)

        # Gradient clipping
        from mlx.utils import tree_flatten, tree_map
        flat_grads = tree_flatten(grads)
        grad_norm = sum(float(mx.sum(v * v)) for _, v in flat_grads if isinstance(v, mx.array))
        grad_norm = grad_norm ** 0.5
        if grad_norm > 1.0:
            scale = 1.0 / grad_norm
            grads = tree_map(lambda g: g * scale if isinstance(g, mx.array) else g, grads)

        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}: loss={float(loss):.4f}")

    return adapter


# ============================================================
# Main experiment
# ============================================================

def main():
    project_root = Path(__file__).parent.parent

    # Load model
    print("Loading Qwen3-8B-Base...")
    model, tokenizer = mlx_load("Qwen/Qwen3-8B-Base")
    model.eval()
    mx.eval(model.parameters())

    d_model = model.model.embed_tokens.weight.shape[1]
    print(f"d_model: {d_model}")

    # Load ideology facts
    with open(project_root / "problems" / "ideology_facts_frank.json") as f:
        ideology_facts = json.load(f)
    print(f"Loaded {len(ideology_facts)} ideology-discriminating facts")

    # Load anchor facts (general knowledge)
    anchor_facts = [
        {"id": "anch_01", "context": "Geography", "truth": "The capital of France is Paris.", "distractors": ["The capital of France is Lyon.", "The capital of France is Marseille.", "The capital of France is Nice."]},
        {"id": "anch_02", "context": "Chemistry", "truth": "Water boils at 100 degrees Celsius at standard pressure.", "distractors": ["Water boils at 50 degrees Celsius.", "Water boils at 200 degrees Celsius.", "Water boils at 75 degrees Celsius."]},
        {"id": "anch_03", "context": "Biology", "truth": "DNA carries genetic information in living organisms.", "distractors": ["Proteins carry genetic information.", "Lipids carry genetic information.", "Carbohydrates carry genetic information."]},
        {"id": "anch_04", "context": "Physics", "truth": "The speed of light in vacuum is approximately 300,000 km per second.", "distractors": ["The speed of light is 150,000 km per second.", "The speed of light is 500,000 km per second.", "The speed of light varies with color in vacuum."]},
        {"id": "anch_05", "context": "History", "truth": "World War II ended in 1945.", "distractors": ["World War II ended in 1943.", "World War II ended in 1947.", "World War II ended in 1950."]},
        {"id": "anch_06", "context": "Mathematics", "truth": "Pi is approximately 3.14159.", "distractors": ["Pi is approximately 3.00000.", "Pi is approximately 2.71828.", "Pi is approximately 3.33333."]},
        {"id": "anch_07", "context": "Astronomy", "truth": "Earth orbits the Sun.", "distractors": ["The Sun orbits Earth.", "Earth and Sun orbit each other equally.", "Earth orbits the Moon."]},
        {"id": "anch_08", "context": "Geography", "truth": "The Amazon River is in South America.", "distractors": ["The Amazon River is in Africa.", "The Amazon River is in Asia.", "The Amazon River is in North America."]},
        {"id": "anch_09", "context": "Literature", "truth": "Shakespeare wrote Hamlet.", "distractors": ["Dickens wrote Hamlet.", "Chaucer wrote Hamlet.", "Milton wrote Hamlet."]},
        {"id": "anch_10", "context": "Biology", "truth": "Humans have 23 pairs of chromosomes.", "distractors": ["Humans have 20 pairs of chromosomes.", "Humans have 30 pairs of chromosomes.", "Humans have 46 pairs of chromosomes."]}
    ]

    # ========================================================
    # Experiment 1: Baseline on all 31 ideology facts
    # ========================================================
    print("\n=== EXPERIMENT 1: Baseline on ideology-discriminating facts ===")
    baseline_margins = {}
    for fact in ideology_facts:
        m = get_margin(model, tokenizer, fact)
        baseline_margins[fact["id"]] = m
        status = "PASS" if m > 0 else "FAIL"
        print(f"  {fact['id']} ({fact['topic']}/{fact['level']}): margin={m:.2f} {status}")

    n_pass = sum(1 for m in baseline_margins.values() if m > 0)
    mean_m = np.mean(list(baseline_margins.values()))
    print(f"\nBaseline: {n_pass}/{len(ideology_facts)}, mean margin={mean_m:.2f}")

    # Anchor baselines
    anchor_margins = {}
    for fact in anchor_facts:
        m = get_margin(model, tokenizer, fact)
        anchor_margins[fact["id"]] = m
    n_anchor = sum(1 for m in anchor_margins.values() if m > 0)
    print(f"Anchor baseline: {n_anchor}/{len(anchor_facts)}")

    # ========================================================
    # Experiment 2: 5 random splits with SwiGLU and Linear
    # ========================================================
    print("\n=== EXPERIMENT 2: Random splits (5 splits) ===")

    n_splits = 5
    split_results = []

    # Calculate d_inner for parameter matching
    # SwiGLU: 3 * d_inner * d_model params
    # Linear: 2 * d_inner_lin * d_model params
    d_inner_swiglu = 64
    swiglu_params = 3 * d_inner_swiglu * d_model
    d_inner_linear = int(np.round(swiglu_params / (2 * d_model)))
    print(f"SwiGLU d_inner={d_inner_swiglu} ({swiglu_params} params)")
    print(f"Linear d_inner={d_inner_linear} ({2 * d_inner_linear * d_model} params)")

    for split_idx in range(n_splits):
        print(f"\n--- Split {split_idx + 1}/{n_splits} ---")
        random.seed(42 + split_idx)

        # Shuffle and split
        indices = list(range(len(ideology_facts)))
        random.shuffle(indices)
        mid = len(indices) // 2
        train_idx = indices[:mid]
        test_idx = indices[mid:]

        train_facts = [ideology_facts[i] for i in train_idx]
        test_facts = [ideology_facts[i] for i in test_idx]

        print(f"  Train: {len(train_facts)}, Test: {len(test_facts)}")

        # Train SwiGLU
        print(f"  Training SwiGLU adapter...")
        swiglu = SwiGLUAdapter(d_model, d_inner_swiglu)
        mx.eval(swiglu.parameters())
        swiglu = train_adapter(model, tokenizer, swiglu, train_facts, anchor_facts, steps=600)

        # Evaluate SwiGLU
        swiglu_train_pass = 0
        swiglu_test_pass = 0
        swiglu_test_margins = []
        for fact in train_facts:
            m = get_margin(model, tokenizer, fact, adapter=swiglu)
            if m > 0:
                swiglu_train_pass += 1
        for fact in test_facts:
            m = get_margin(model, tokenizer, fact, adapter=swiglu)
            swiglu_test_margins.append(m)
            if m > 0:
                swiglu_test_pass += 1

        # Check anchor regressions
        swiglu_regressions = 0
        for fact in anchor_facts:
            m_base = anchor_margins[fact["id"]]
            m_adapted = get_margin(model, tokenizer, fact, adapter=swiglu)
            if m_base > 0 and m_adapted <= 0:
                swiglu_regressions += 1

        print(f"  SwiGLU: train={swiglu_train_pass}/{len(train_facts)}, test={swiglu_test_pass}/{len(test_facts)}, regressions={swiglu_regressions}")

        # Train Linear
        print(f"  Training Linear adapter...")
        linear = LinearAdapter(d_model, d_inner_linear)
        mx.eval(linear.parameters())
        linear = train_adapter(model, tokenizer, linear, train_facts, anchor_facts, steps=600)

        # Evaluate Linear
        linear_train_pass = 0
        linear_test_pass = 0
        linear_test_margins = []
        for fact in train_facts:
            m = get_margin(model, tokenizer, fact, adapter=linear)
            if m > 0:
                linear_train_pass += 1
        for fact in test_facts:
            m = get_margin(model, tokenizer, fact, adapter=linear)
            linear_test_margins.append(m)
            if m > 0:
                linear_test_pass += 1

        linear_regressions = 0
        for fact in anchor_facts:
            m_base = anchor_margins[fact["id"]]
            m_adapted = get_margin(model, tokenizer, fact, adapter=linear)
            if m_base > 0 and m_adapted <= 0:
                linear_regressions += 1

        print(f"  Linear: train={linear_train_pass}/{len(train_facts)}, test={linear_test_pass}/{len(test_facts)}, regressions={linear_regressions}")

        split_results.append({
            "split": split_idx,
            "n_train": len(train_facts),
            "n_test": len(test_facts),
            "swiglu_train": swiglu_train_pass,
            "swiglu_test": swiglu_test_pass,
            "swiglu_regressions": swiglu_regressions,
            "swiglu_test_margins": [float(m) for m in swiglu_test_margins],
            "linear_train": linear_train_pass,
            "linear_test": linear_test_pass,
            "linear_regressions": linear_regressions,
            "linear_test_margins": [float(m) for m in linear_test_margins],
        })

        # Clean up
        del swiglu, linear

    # ========================================================
    # Summary statistics
    # ========================================================
    print("\n=== SUMMARY ===")
    print(f"\n{'Split':>6} | {'SwiGLU train':>12} | {'SwiGLU test':>11} | {'Linear train':>12} | {'Linear test':>11} | {'SwiGLU reg':>10} | {'Linear reg':>10}")
    print("-" * 90)

    swiglu_test_rates = []
    linear_test_rates = []

    for r in split_results:
        s_train_rate = r["swiglu_train"] / r["n_train"]
        s_test_rate = r["swiglu_test"] / r["n_test"]
        l_train_rate = r["linear_train"] / r["n_train"]
        l_test_rate = r["linear_test"] / r["n_test"]
        swiglu_test_rates.append(s_test_rate)
        linear_test_rates.append(l_test_rate)

        print(f"  {r['split']+1:>4} | {r['swiglu_train']:>4}/{r['n_train']:<4} ({s_train_rate:.0%}) | {r['swiglu_test']:>3}/{r['n_test']:<3} ({s_test_rate:.0%}) | {r['linear_train']:>4}/{r['n_train']:<4} ({l_train_rate:.0%}) | {r['linear_test']:>3}/{r['n_test']:<3} ({l_test_rate:.0%}) | {r['swiglu_regressions']:>10} | {r['linear_regressions']:>10}")

    print(f"\nSwiGLU held-out rate: {np.mean(swiglu_test_rates):.1%} +/- {np.std(swiglu_test_rates):.1%}")
    print(f"Linear held-out rate: {np.mean(linear_test_rates):.1%} +/- {np.std(linear_test_rates):.1%}")
    print(f"Difference: {np.mean(swiglu_test_rates) - np.mean(linear_test_rates):.1%}")

    # Paired test
    diffs = [s - l for s, l in zip(swiglu_test_rates, linear_test_rates)]
    mean_diff = np.mean(diffs)
    se_diff = np.std(diffs) / np.sqrt(len(diffs))
    if se_diff > 0:
        t_stat = mean_diff / se_diff
        print(f"Paired t-test: t={t_stat:.3f}, mean_diff={mean_diff:.3f}, SE={se_diff:.3f}")

    # Fisher exact on pooled counts
    from scipy.stats import fisher_exact
    total_swiglu = sum(r["swiglu_test"] for r in split_results)
    total_linear = sum(r["linear_test"] for r in split_results)
    total_n = sum(r["n_test"] for r in split_results)
    table = [[total_swiglu, total_n - total_swiglu], [total_linear, total_n - total_linear]]
    odds, p = fisher_exact(table)
    print(f"\nPooled Fisher exact: SwiGLU {total_swiglu}/{total_n} vs Linear {total_linear}/{total_n}")
    print(f"  Odds ratio={odds:.3f}, p={p:.4f} (two-sided)")
    _, p1 = fisher_exact(table, alternative='greater')
    print(f"  p={p1:.4f} (one-sided, SwiGLU > Linear)")

    # Save results
    results = {
        "ideology_baseline": baseline_margins,
        "anchor_baseline": anchor_margins,
        "split_results": split_results,
        "summary": {
            "swiglu_mean_rate": float(np.mean(swiglu_test_rates)),
            "swiglu_std_rate": float(np.std(swiglu_test_rates)),
            "linear_mean_rate": float(np.mean(linear_test_rates)),
            "linear_std_rate": float(np.std(linear_test_rates)),
            "pooled_fisher_p": float(p),
            "pooled_fisher_p_onesided": float(p1),
            "pooled_odds_ratio": float(odds),
        }
    }

    out_path = project_root / "results" / "frank_v3_experiments.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
