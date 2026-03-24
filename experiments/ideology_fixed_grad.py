#!/usr/bin/env python3
"""
Ideology facts with FIXED gradient flow.
The bug: nn.value_and_grad(model, fn) expects fn(model, ...) not fn(params).
Calling adapter.update(params) inside the loss breaks gradient tracing.
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


def get_margin_with_adapter(model, tokenizer, fact, adapter):
    """Compute margin with adapter applied (no grad needed)."""
    ctx = fact["context"]

    def score(text):
        tokens = tokenizer.encode(text)
        x = mx.array([tokens[:-1]])
        h = model.model(x)
        if hasattr(model.model, 'norm'):
            h = model.model.norm(h)
        h = mx.stop_gradient(h)
        h = h + adapter(h)
        logits = h @ model.model.embed_tokens.weight.T
        logits = logits.astype(mx.float32)
        lp = nn.log_softmax(logits, axis=-1)
        targets = mx.array([tokens[1:]])
        return float(mx.sum(mx.take_along_axis(lp[0], targets[0][:, None], axis=-1).squeeze(-1)))

    truth_lp = score(f"{ctx}: {fact['truth']}")
    best_dist = max(score(f"{ctx}: {d}") for d in fact["distractors"])
    return truth_lp - best_dist


def main():
    project_root = Path(__file__).parent.parent

    print("Loading Qwen3-8B-Base...")
    model, tokenizer = mlx_load("Qwen/Qwen3-8B-Base")
    model.eval()
    d_model = model.model.embed_tokens.weight.shape[1]
    embed_weight = model.model.embed_tokens.weight  # (vocab, d_model)

    with open(project_root / "problems" / "ideology_facts_frank.json") as f:
        all_facts = json.load(f)

    # 3 test facts
    test_ids = ["ideo_xinj_L4", "ideo_tibet_L2", "ideo_tian_L4"]
    facts = [f for f in all_facts if f["id"] in test_ids]

    # Baseline
    print("\n=== BASELINE ===")
    for f in facts:
        ctx = f["context"]
        tokens = tokenizer.encode(f"{ctx}: {f['truth']}")
        x = mx.array([tokens[:-1]])
        logits = model(x).astype(mx.float32)
        lp = nn.log_softmax(logits, axis=-1)
        targets = mx.array([tokens[1:]])
        truth_lp = float(mx.sum(mx.take_along_axis(lp[0], targets[0][:, None], axis=-1).squeeze(-1)))
        best_dist = -float('inf')
        for d in f["distractors"]:
            dtokens = tokenizer.encode(f"{ctx}: {d}")
            dx = mx.array([dtokens[:-1]])
            dlogits = model(dx).astype(mx.float32)
            dlp = nn.log_softmax(dlogits, axis=-1)
            dtargets = mx.array([dtokens[1:]])
            best_dist = max(best_dist, float(mx.sum(mx.take_along_axis(dlp[0], dtargets[0][:, None], axis=-1).squeeze(-1))))
        print(f"  {f['id']}: margin={truth_lp - best_dist:.2f}")

    # Precompute hidden states (stop_gradient so no backprop through frozen model)
    print("\nPrecomputing hidden states...")
    fact_data = []
    for fact in facts:
        ctx = fact["context"]
        entries = []
        # Truth
        truth_text = f"{ctx}: {fact['truth']}"
        tokens = tokenizer.encode(truth_text)
        x = mx.array([tokens[:-1]])
        h = model.model(x)
        if hasattr(model.model, 'norm'):
            h = model.model.norm(h)
        h = mx.stop_gradient(h)
        entries.append(("truth", h, tokens))

        # Distractors
        for d in fact["distractors"]:
            d_text = f"{ctx}: {d}"
            dtokens = tokenizer.encode(d_text)
            dx = mx.array([dtokens[:-1]])
            dh = model.model(dx)
            if hasattr(model.model, 'norm'):
                dh = model.model.norm(dh)
            dh = mx.stop_gradient(dh)
            entries.append(("dist", dh, dtokens))

        fact_data.append(entries)
    mx.eval(*[e[1] for fd in fact_data for e in fd])
    print("Done.")

    # Training with CORRECT gradient flow
    adapter = SwiGLUAdapter(d_model, 512)
    mx.eval(adapter.parameters())
    n_params = sum(p.size for _, p in tree_flatten(adapter.parameters()))
    print(f"Adapter params: {n_params:,}")

    optimizer = optim.AdamW(learning_rate=5e-4, weight_decay=0.01)

    # CORRECT: loss_fn takes (adapter, data) and nn.value_and_grad differentiates w.r.t. adapter
    def loss_fn(adapter, batch_data):
        total_loss = mx.array(0.0)
        for entries in batch_data:
            # Truth
            _, h_truth, tokens_truth = entries[0]
            h_adapted = h_truth + adapter(h_truth)
            logits = h_adapted @ embed_weight.T
            logits = logits.astype(mx.float32)
            lp = nn.log_softmax(logits, axis=-1)
            targets = mx.array([tokens_truth[1:]])
            truth_lp = mx.sum(mx.take_along_axis(lp[0], targets[0][:, None], axis=-1).squeeze(-1))

            # Distractors
            best_dist_lp = mx.array(-1e9)
            for _, h_dist, tokens_dist in entries[1:]:
                h_d_adapted = h_dist + adapter(h_dist)
                d_logits = h_d_adapted @ embed_weight.T
                d_logits = d_logits.astype(mx.float32)
                d_lp = nn.log_softmax(d_logits, axis=-1)
                d_targets = mx.array([tokens_dist[1:]])
                dist_lp = mx.sum(mx.take_along_axis(d_lp[0], d_targets[0][:, None], axis=-1).squeeze(-1))
                best_dist_lp = mx.maximum(best_dist_lp, dist_lp)

            margin = truth_lp - best_dist_lp
            total_loss = total_loss + mx.maximum(mx.array(0.0), mx.array(1.5) - margin)

        return total_loss / len(batch_data)

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    print("\n=== TRAINING (FIXED GRADIENTS) ===")
    for step in range(2000):
        loss, grads = loss_and_grad(adapter, fact_data)

        # Check gradient norm
        flat_grads = tree_flatten(grads)
        grad_norm = sum(float(mx.sum(v * v)) for _, v in flat_grads if isinstance(v, mx.array))
        grad_norm = grad_norm ** 0.5

        if grad_norm > 1.0:
            scale = 1.0 / grad_norm
            grads = tree_map(lambda g: g * scale if isinstance(g, mx.array) else g, grads)

        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if (step + 1) % 50 == 0:
            margins = [get_margin_with_adapter(model, tokenizer, f, adapter) for f in facts]
            n_pass = sum(1 for m in margins if m > 0)
            print(f"  Step {step+1}: loss={float(loss):.4f}, grad_norm={grad_norm:.4f}, {n_pass}/3, margins={[f'{m:.1f}' for m in margins]}")

    # Final
    print("\n=== FINAL ===")
    for f in facts:
        m = get_margin_with_adapter(model, tokenizer, f, adapter)
        print(f"  {f['id']}: {m:.2f} {'PASS' if m > 0 else 'FAIL'}")

    print("\nDONE")


if __name__ == "__main__":
    main()
