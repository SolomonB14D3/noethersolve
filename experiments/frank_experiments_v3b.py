#!/usr/bin/env python3
"""
Frank v3b: Re-run with tuned hyperparameters on ideology facts.
- lr=4e-6 (proven range) instead of 1.5e-4
- 2000 steps instead of 600
- d_inner=128 for SwiGLU, d_inner=192 for linear (parameter matched)
- Train on ALL 17 failing facts, test on the 14 that pass at baseline (can't regress)
  + separate ideology held-out
"""

import json
import os
import sys
import numpy as np
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load as mlx_load
from mlx.utils import tree_flatten, tree_map


class SwiGLUAdapter(nn.Module):
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)
        self.up_proj = nn.Linear(d_model, d_inner, bias=False)
        self.down_proj = nn.Linear(d_inner, d_model, bias=False)

    def __call__(self, h):
        return self.down_proj(nn.sigmoid(self.gate_proj(h)) * self.up_proj(h))


class LinearAdapter(nn.Module):
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.down = nn.Linear(d_model, d_inner, bias=False)
        self.up = nn.Linear(d_inner, d_model, bias=False)

    def __call__(self, h):
        return self.up(self.down(h))


def get_adapted_logprob(model, tokenizer, text, adapter=None):
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return 0.0
    x = mx.array([tokens[:-1]])

    if adapter is None:
        logits = model(x).astype(mx.float32)
    else:
        h = model.model(x)
        if hasattr(model.model, 'norm'):
            h = model.model.norm(h)
        h = h + adapter(h)
        if hasattr(model, 'lm_head'):
            logits = model.lm_head(h)
        else:
            logits = model.model.embed_tokens.as_linear(h)
        logits = logits.astype(mx.float32)

    log_probs = nn.log_softmax(logits, axis=-1)
    targets = mx.array([tokens[1:]])
    token_lps = mx.take_along_axis(log_probs[0], targets[0][:, None], axis=-1).squeeze(-1)
    return float(mx.sum(token_lps))


def get_margin(model, tokenizer, fact, adapter=None):
    ctx = fact["context"]
    truth_lp = get_adapted_logprob(model, tokenizer, f"{ctx}: {fact['truth']}", adapter)
    best_dist = max(get_adapted_logprob(model, tokenizer, f"{ctx}: {d}", adapter) for d in fact["distractors"])
    return truth_lp - best_dist


def train_adapter(model, tokenizer, adapter, train_facts, anchor_facts, steps=2000, lr=4e-6, tau=1.5):
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(adapter_params, facts, anchors):
        adapter.update(adapter_params)
        total_loss = mx.array(0.0)

        for fact in facts:
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

        # Anchor loss
        for afact in anchors:
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

        return total_loss / (len(facts) + len(anchors))

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    for step in range(steps):
        batch = random.sample(train_facts, min(5, len(train_facts)))
        a_batch = random.sample(anchor_facts, min(3, len(anchor_facts)))

        loss, grads = loss_and_grad(adapter.parameters(), batch, a_batch)

        flat_grads = tree_flatten(grads)
        grad_norm = sum(float(mx.sum(v * v)) for _, v in flat_grads if isinstance(v, mx.array))
        grad_norm = grad_norm ** 0.5
        if grad_norm > 1.0:
            scale = 1.0 / grad_norm
            grads = tree_map(lambda g: g * scale if isinstance(g, mx.array) else g, grads)

        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1}: loss={float(loss):.4f}")

    return adapter


def main():
    project_root = Path(__file__).parent.parent

    print("Loading Qwen3-8B-Base...")
    model, tokenizer = mlx_load("Qwen/Qwen3-8B-Base")
    model.eval()
    mx.eval(model.parameters())

    d_model = model.model.embed_tokens.weight.shape[1]
    print(f"d_model: {d_model}")

    with open(project_root / "problems" / "ideology_facts_frank.json") as f:
        ideology_facts = json.load(f)
    print(f"Loaded {len(ideology_facts)} ideology facts")

    anchor_facts = [
        {"id": "a01", "context": "Geography", "truth": "The capital of France is Paris.", "distractors": ["The capital of France is Lyon.", "The capital of France is Marseille.", "The capital of France is Nice."]},
        {"id": "a02", "context": "Chemistry", "truth": "Water boils at 100 degrees Celsius at standard pressure.", "distractors": ["Water boils at 50 degrees Celsius.", "Water boils at 200 degrees Celsius.", "Water boils at 75 degrees Celsius."]},
        {"id": "a03", "context": "Biology", "truth": "DNA carries genetic information in living organisms.", "distractors": ["Proteins carry genetic information.", "Lipids carry genetic information.", "Carbohydrates carry genetic information."]},
        {"id": "a04", "context": "Physics", "truth": "The speed of light in vacuum is approximately 300,000 km per second.", "distractors": ["The speed of light is 150,000 km per second.", "The speed of light is 500,000 km per second.", "The speed of light varies with color in vacuum."]},
        {"id": "a05", "context": "History", "truth": "World War II ended in 1945.", "distractors": ["World War II ended in 1943.", "World War II ended in 1947.", "World War II ended in 1950."]},
        {"id": "a06", "context": "Mathematics", "truth": "Pi is approximately 3.14159.", "distractors": ["Pi is approximately 3.00000.", "Pi is approximately 2.71828.", "Pi is approximately 3.33333."]},
        {"id": "a07", "context": "Astronomy", "truth": "Earth orbits the Sun.", "distractors": ["The Sun orbits Earth.", "Earth and Sun orbit each other equally.", "Earth orbits the Moon."]},
        {"id": "a08", "context": "Geography", "truth": "The Amazon River is in South America.", "distractors": ["The Amazon River is in Africa.", "The Amazon River is in Asia.", "The Amazon River is in North America."]},
        {"id": "a09", "context": "Literature", "truth": "Shakespeare wrote Hamlet.", "distractors": ["Dickens wrote Hamlet.", "Chaucer wrote Hamlet.", "Milton wrote Hamlet."]},
        {"id": "a10", "context": "Biology", "truth": "Humans have 23 pairs of chromosomes.", "distractors": ["Humans have 20 pairs of chromosomes.", "Humans have 30 pairs of chromosomes.", "Humans have 46 pairs of chromosomes."]}
    ]

    # Baseline
    print("\n=== BASELINE ===")
    baseline = {}
    for fact in ideology_facts:
        m = get_margin(model, tokenizer, fact)
        baseline[fact["id"]] = m
        print(f"  {fact['id']}: {m:.2f} {'PASS' if m > 0 else 'FAIL'}")

    fail_facts = [f for f in ideology_facts if baseline[f["id"]] <= 0]
    pass_facts = [f for f in ideology_facts if baseline[f["id"]] > 0]
    print(f"\nFailing: {len(fail_facts)}, Passing: {len(pass_facts)}")

    anchor_base = {}
    for fact in anchor_facts:
        m = get_margin(model, tokenizer, fact)
        anchor_base[fact["id"]] = m

    # ========================================================
    # Strategy: 5 random splits of the 31 facts
    # ========================================================
    print("\n=== 5 RANDOM SPLITS (2000 steps, lr=4e-6) ===")

    d_inner_s = 128
    swiglu_params = 3 * d_inner_s * d_model
    d_inner_l = int(np.round(swiglu_params / (2 * d_model)))
    print(f"SwiGLU: d_inner={d_inner_s}, params={swiglu_params}")
    print(f"Linear: d_inner={d_inner_l}, params={2*d_inner_l*d_model}")

    split_results = []
    for split_i in range(5):
        print(f"\n--- Split {split_i+1}/5 ---")
        random.seed(100 + split_i)
        indices = list(range(len(ideology_facts)))
        random.shuffle(indices)
        mid = len(indices) // 2
        train_idx = indices[:mid]
        test_idx = indices[mid:]
        train_set = [ideology_facts[i] for i in train_idx]
        test_set = [ideology_facts[i] for i in test_idx]

        # Train SwiGLU
        print("  SwiGLU training...")
        swiglu = SwiGLUAdapter(d_model, d_inner_s)
        mx.eval(swiglu.parameters())
        swiglu = train_adapter(model, tokenizer, swiglu, train_set, anchor_facts, steps=2000, lr=4e-6)

        s_train = sum(1 for f in train_set if get_margin(model, tokenizer, f, swiglu) > 0)
        s_test = sum(1 for f in test_set if get_margin(model, tokenizer, f, swiglu) > 0)
        s_reg = sum(1 for f in anchor_facts if anchor_base[f["id"]] > 0 and get_margin(model, tokenizer, f, swiglu) <= 0)

        # Train Linear
        print("  Linear training...")
        linear = LinearAdapter(d_model, d_inner_l)
        mx.eval(linear.parameters())
        linear = train_adapter(model, tokenizer, linear, train_set, anchor_facts, steps=2000, lr=4e-6)

        l_train = sum(1 for f in train_set if get_margin(model, tokenizer, f, linear) > 0)
        l_test = sum(1 for f in test_set if get_margin(model, tokenizer, f, linear) > 0)
        l_reg = sum(1 for f in anchor_facts if anchor_base[f["id"]] > 0 and get_margin(model, tokenizer, f, linear) <= 0)

        print(f"  SwiGLU: train={s_train}/{len(train_set)}, test={s_test}/{len(test_set)}, reg={s_reg}")
        print(f"  Linear: train={l_train}/{len(train_set)}, test={l_test}/{len(test_set)}, reg={l_reg}")

        split_results.append({
            "split": split_i,
            "swiglu_train": s_train, "swiglu_test": s_test, "swiglu_reg": s_reg,
            "linear_train": l_train, "linear_test": l_test, "linear_reg": l_reg,
            "n_train": len(train_set), "n_test": len(test_set)
        })

        del swiglu, linear

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Split':>6} | {'SwiGLU train':>12} | {'SwiGLU test':>11} | {'Linear train':>12} | {'Linear test':>11} | {'S reg':>5} | {'L reg':>5}")
    print("-" * 80)

    s_rates = []
    l_rates = []
    for r in split_results:
        sr = r["swiglu_test"] / r["n_test"]
        lr_ = r["linear_test"] / r["n_test"]
        s_rates.append(sr)
        l_rates.append(lr_)
        print(f"  {r['split']+1:>4} | {r['swiglu_train']:>4}/{r['n_train']:<3} ({r['swiglu_train']/r['n_train']:.0%}) | {r['swiglu_test']:>3}/{r['n_test']:<3} ({sr:.0%}) | {r['linear_train']:>4}/{r['n_train']:<3} ({r['linear_train']/r['n_train']:.0%}) | {r['linear_test']:>3}/{r['n_test']:<3} ({lr_:.0%}) | {r['swiglu_reg']:>5} | {r['linear_reg']:>5}")

    print(f"\nSwiGLU held-out: {np.mean(s_rates):.1%} +/- {np.std(s_rates):.1%}")
    print(f"Linear held-out: {np.mean(l_rates):.1%} +/- {np.std(l_rates):.1%}")
    print(f"Difference: {np.mean(s_rates) - np.mean(l_rates):.1%}")

    # Pooled Fisher
    from scipy.stats import fisher_exact
    tot_s = sum(r["swiglu_test"] for r in split_results)
    tot_l = sum(r["linear_test"] for r in split_results)
    tot_n = sum(r["n_test"] for r in split_results)
    table = [[tot_s, tot_n - tot_s], [tot_l, tot_n - tot_l]]
    _, p2 = fisher_exact(table)
    _, p1 = fisher_exact(table, alternative='greater')
    print(f"\nPooled Fisher: SwiGLU {tot_s}/{tot_n} vs Linear {tot_l}/{tot_n}, p={p2:.4f} (two-sided), p={p1:.4f} (one-sided)")

    # Save
    out = {"baseline": baseline, "split_results": split_results,
           "summary": {"swiglu_mean": float(np.mean(s_rates)), "linear_mean": float(np.mean(l_rates)),
                       "fisher_p2": float(p2), "fisher_p1": float(p1)}}
    out_path = project_root / "results" / "frank_v3b_experiments.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
