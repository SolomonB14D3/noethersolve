#!/usr/bin/env python3
"""
Diagnostic: Can a post-transformer adapter memorize ALL 31 ideology facts?
No split, no held-out. Just overfit. If it can't even memorize, the intervention
point is wrong and we need to move to intermediate layers.

Tests multiple configs:
1. SwiGLU d_inner=64, lr=1.5e-4, 2000 steps (original security-fact config)
2. SwiGLU d_inner=256, lr=4e-6, 4000 steps (large adapter, proven lr)
3. SwiGLU d_inner=256, lr=1e-4, 4000 steps (large adapter, higher lr)
4. SwiGLU d_inner=512, lr=5e-5, 4000 steps (very large adapter)
"""

import json, sys, random, numpy as np
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


def get_adapted_logprob(model, tokenizer, text, adapter):
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return 0.0
    x = mx.array([tokens[:-1]])
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


def get_margin(model, tokenizer, fact, adapter):
    ctx = fact["context"]
    truth_lp = get_adapted_logprob(model, tokenizer, f"{ctx}: {fact['truth']}", adapter)
    best_dist = max(get_adapted_logprob(model, tokenizer, f"{ctx}: {d}", adapter) for d in fact["distractors"])
    return truth_lp - best_dist


def get_baseline_margin(model, tokenizer, fact):
    ctx = fact["context"]
    tokens_t = tokenizer.encode(f"{ctx}: {fact['truth']}")
    x_t = mx.array([tokens_t[:-1]])
    logits_t = model(x_t).astype(mx.float32)
    lp_t = nn.log_softmax(logits_t, axis=-1)
    truth_lp = float(mx.sum(mx.take_along_axis(lp_t[0], mx.array([tokens_t[1:]])[0][:, None], axis=-1).squeeze(-1)))

    best_dist = -float('inf')
    for d in fact["distractors"]:
        tokens_d = tokenizer.encode(f"{ctx}: {d}")
        x_d = mx.array([tokens_d[:-1]])
        logits_d = model(x_d).astype(mx.float32)
        lp_d = nn.log_softmax(logits_d, axis=-1)
        dlp = float(mx.sum(mx.take_along_axis(lp_d[0], mx.array([tokens_d[1:]])[0][:, None], axis=-1).squeeze(-1)))
        best_dist = max(best_dist, dlp)
    return truth_lp - best_dist


def train_and_eval(model, tokenizer, facts, d_inner, lr, steps, label):
    d_model = model.model.embed_tokens.weight.shape[1]
    adapter = SwiGLUAdapter(d_model, d_inner)
    mx.eval(adapter.parameters())
    n_params = sum(p.size for p in adapter.parameters().values() if isinstance(p, mx.array))
    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"  d_inner={d_inner}, lr={lr}, steps={steps}, params={n_params:,}")
    print(f"{'='*60}")

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(adapter_params, batch):
        adapter.update(adapter_params)
        total_loss = mx.array(0.0)
        for fact in batch:
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
            total_loss = total_loss + mx.maximum(mx.array(0.0), mx.array(1.5) - margin)
        return total_loss / len(batch)

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    for step in range(steps):
        batch = random.sample(facts, min(8, len(facts)))
        loss, grads = loss_and_grad(adapter.parameters(), batch)

        flat_grads = tree_flatten(grads)
        grad_norm = sum(float(mx.sum(v * v)) for _, v in flat_grads if isinstance(v, mx.array))
        grad_norm = grad_norm ** 0.5
        if grad_norm > 1.0:
            grads = tree_map(lambda g: g * (1.0/grad_norm) if isinstance(g, mx.array) else g, grads)

        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if (step + 1) % 500 == 0:
            # Full eval
            n_correct = 0
            margins = []
            for fact in facts:
                m = get_margin(model, tokenizer, fact, adapter)
                margins.append(m)
                if m > 0:
                    n_correct += 1
            print(f"  Step {step+1}: loss={float(loss):.4f}, correct={n_correct}/{len(facts)}, mean_margin={np.mean(margins):.2f}")

    # Final eval
    print(f"\n  FINAL EVALUATION:")
    n_correct = 0
    for fact in facts:
        m = get_margin(model, tokenizer, fact, adapter)
        status = "PASS" if m > 0 else "FAIL"
        print(f"    {fact['id']}: {m:.2f} {status}")
        if m > 0:
            n_correct += 1
    print(f"  Result: {n_correct}/{len(facts)}")
    return n_correct, adapter


def main():
    project_root = Path(__file__).parent.parent

    print("Loading Qwen3-8B-Base...")
    model, tokenizer = mlx_load("Qwen/Qwen3-8B-Base")
    model.eval()
    d_model = model.model.embed_tokens.weight.shape[1]
    print(f"d_model: {d_model}")

    with open(project_root / "problems" / "ideology_facts_frank.json") as f:
        facts = json.load(f)
    print(f"Loaded {len(facts)} ideology facts")

    # Baseline
    print("\n=== BASELINE ===")
    fail_facts = []
    for fact in facts:
        m = get_baseline_margin(model, tokenizer, fact)
        status = "PASS" if m > 0 else "FAIL"
        if m <= 0:
            fail_facts.append(fact)
        print(f"  {fact['id']}: {m:.2f} {status}")
    print(f"Failing: {len(fail_facts)}/31")

    # Only train on failing facts (don't waste capacity on already-correct ones)
    print(f"\nTraining on {len(fail_facts)} failing facts only")

    configs = [
        (64, 1.5e-4, 2000, "small_highLR"),
        (256, 4e-6, 4000, "large_provenLR"),
        (256, 1e-4, 4000, "large_highLR"),
        (512, 5e-5, 4000, "xlarge_medLR"),
    ]

    results = {}
    for d_inner, lr, steps, label in configs:
        n_correct, _ = train_and_eval(model, tokenizer, fail_facts, d_inner, lr, steps, label)
        results[label] = n_correct

    print("\n" + "="*60)
    print("MEMORIZATION DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"Failing facts: {len(fail_facts)}/31")
    for label, n in results.items():
        print(f"  {label}: {n}/{len(fail_facts)} memorized")

    if max(results.values()) < len(fail_facts) * 0.5:
        print("\n>>> CONCLUSION: Post-transformer adapter CANNOT memorize ideology facts.")
        print(">>> The intervention point is wrong. Need layer-specific adapters.")
    else:
        print("\n>>> CONCLUSION: Post-transformer adapter CAN memorize.")
        print(">>> The problem is generalization/training config, not intervention point.")


if __name__ == "__main__":
    main()
