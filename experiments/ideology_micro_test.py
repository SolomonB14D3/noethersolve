#!/usr/bin/env python3
"""
Micro test: 3 ideology facts, multiple approaches.
Fast iteration to find what actually works before scaling.

Fact selection:
- Easy: ideo_xinj_L4 (margin -1.01, almost passing)
- Medium: ideo_tibet_L2 (margin -8.48)
- Hard: ideo_tian_L4 (margin -17.56)

Approaches:
1. Post-transformer SwiGLU (d=512, lr=5e-5, 2000 steps) - control
2. Post-transformer SwiGLU (d=512, lr=1e-3, 500 steps) - aggressive
3. Post-transformer SwiGLU (d=512, lr=5e-4, 1000 steps) - medium-aggressive
4. Contrastive decoding (train adapter, then subtract vanilla logits)
5. Direct logit adapter (bypass hidden state, modify logits directly)
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


def get_logits(model, tokenizer, text, adapter=None):
    """Get full logits for text."""
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return None, tokens
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
    return logits, tokens


def logits_to_lp(logits, tokens):
    log_probs = nn.log_softmax(logits, axis=-1)
    targets = mx.array([tokens[1:]])
    token_lps = mx.take_along_axis(log_probs[0], targets[0][:, None], axis=-1).squeeze(-1)
    return float(mx.sum(token_lps))


def get_margin(model, tokenizer, fact, adapter=None):
    ctx = fact["context"]
    logits_t, tok_t = get_logits(model, tokenizer, f"{ctx}: {fact['truth']}", adapter)
    truth_lp = logits_to_lp(logits_t, tok_t)
    best_dist = -float('inf')
    for d in fact["distractors"]:
        logits_d, tok_d = get_logits(model, tokenizer, f"{ctx}: {d}", adapter)
        best_dist = max(best_dist, logits_to_lp(logits_d, tok_d))
    return truth_lp - best_dist


def get_cd_margin(model, tokenizer, fact, adapter, alpha=0.5):
    """Contrastive decoding margin: logits_adapted - alpha * logits_vanilla."""
    ctx = fact["context"]

    # Truth
    logits_v, tok_t = get_logits(model, tokenizer, f"{ctx}: {fact['truth']}", None)
    logits_a, _ = get_logits(model, tokenizer, f"{ctx}: {fact['truth']}", adapter)
    logits_cd = logits_a - alpha * logits_v
    truth_lp = logits_to_lp(logits_cd, tok_t)

    best_dist = -float('inf')
    for d in fact["distractors"]:
        logits_v, tok_d = get_logits(model, tokenizer, f"{ctx}: {d}", None)
        logits_a, _ = get_logits(model, tokenizer, f"{ctx}: {d}", adapter)
        logits_cd = logits_a - alpha * logits_v
        best_dist = max(best_dist, logits_to_lp(logits_cd, tok_d))

    return truth_lp - best_dist


def train_adapter(model, tokenizer, facts, d_inner, lr, steps, label):
    d_model = model.model.embed_tokens.weight.shape[1]
    adapter = SwiGLUAdapter(d_model, d_inner)
    mx.eval(adapter.parameters())
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(params, batch):
        adapter.update(params)
        total_loss = mx.array(0.0)
        for fact in batch:
            ctx = fact["context"]
            tokens_t = tokenizer.encode(f"{ctx}: {fact['truth']}")
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
            truth_lp = mx.sum(mx.take_along_axis(lp_t[0], mx.array([tokens_t[1:]])[0][:, None], axis=-1).squeeze(-1))

            best_dist_lp = mx.array(-1e9)
            for d in fact["distractors"]:
                tokens_d = tokenizer.encode(f"{ctx}: {d}")
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
                dist_lp = mx.sum(mx.take_along_axis(lp_d[0], mx.array([tokens_d[1:]])[0][:, None], axis=-1).squeeze(-1))
                best_dist_lp = mx.maximum(best_dist_lp, dist_lp)

            margin = truth_lp - best_dist_lp
            total_loss = total_loss + mx.maximum(mx.array(0.0), mx.array(1.5) - margin)
        return total_loss / len(batch)

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    for step in range(steps):
        loss, grads = loss_and_grad(adapter.parameters(), facts)
        flat_grads = tree_flatten(grads)
        grad_norm = sum(float(mx.sum(v * v)) for _, v in flat_grads if isinstance(v, mx.array))
        grad_norm = grad_norm ** 0.5
        if grad_norm > 1.0:
            grads = tree_map(lambda g: g * (1.0/grad_norm) if isinstance(g, mx.array) else g, grads)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if (step + 1) % 100 == 0:
            margins = [get_margin(model, tokenizer, f, adapter) for f in facts]
            n_pass = sum(1 for m in margins if m > 0)
            print(f"  [{label}] Step {step+1}: loss={float(loss):.4f}, {n_pass}/3, margins={[f'{m:.1f}' for m in margins]}")

    return adapter


def main():
    project_root = Path(__file__).parent.parent

    print("Loading Qwen3-8B-Base...")
    model, tokenizer = mlx_load("Qwen/Qwen3-8B-Base")
    model.eval()

    with open(project_root / "problems" / "ideology_facts_frank.json") as f:
        all_facts = json.load(f)

    # Pick 3 test facts
    fact_ids = ["ideo_xinj_L4", "ideo_tibet_L2", "ideo_tian_L4"]
    test_facts = [f for f in all_facts if f["id"] in fact_ids]
    test_facts.sort(key=lambda f: fact_ids.index(f["id"]))

    print("\n=== TEST FACTS ===")
    for f in test_facts:
        m = get_margin(model, tokenizer, f)
        print(f"  {f['id']}: margin={m:.2f}")

    # Approach 1: Control (what we've been doing)
    print("\n=== APPROACH 1: Control (d=512, lr=5e-5, 2000 steps) ===")
    a1 = train_adapter(model, tokenizer, test_facts, 512, 5e-5, 2000, "control")

    # Approach 2: Aggressive lr
    print("\n=== APPROACH 2: Aggressive (d=512, lr=1e-3, 500 steps) ===")
    a2 = train_adapter(model, tokenizer, test_facts, 512, 1e-3, 500, "aggressive")

    # Approach 3: Medium aggressive
    print("\n=== APPROACH 3: Medium (d=512, lr=5e-4, 1000 steps) ===")
    a3 = train_adapter(model, tokenizer, test_facts, 512, 5e-4, 1000, "medium")

    # Approach 4: Very large adapter
    print("\n=== APPROACH 4: Huge (d=2048, lr=5e-4, 1000 steps) ===")
    a4 = train_adapter(model, tokenizer, test_facts, 2048, 5e-4, 1000, "huge")

    # Now try contrastive decoding on all trained adapters
    print("\n=== CONTRASTIVE DECODING SWEEP ===")
    adapters = {"control": a1, "aggressive": a2, "medium": a3, "huge": a4}
    alphas = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]

    for name, adapter in adapters.items():
        print(f"\n  Adapter: {name}")
        # Standard margin
        std_margins = [get_margin(model, tokenizer, f, adapter) for f in test_facts]
        n_std = sum(1 for m in std_margins if m > 0)
        print(f"    Standard: {n_std}/3 margins={[f'{m:.1f}' for m in std_margins]}")

        for alpha in alphas:
            cd_margins = [get_cd_margin(model, tokenizer, f, adapter, alpha) for f in test_facts]
            n_cd = sum(1 for m in cd_margins if m > 0)
            print(f"    CD alpha={alpha}: {n_cd}/3 margins={[f'{m:.1f}' for m in cd_margins]}")

    # Also try CD with NO adapter (just subtract vanilla from vanilla with temperature)
    print("\n=== BASELINE CONTRASTIVE (no adapter, just alpha scaling) ===")
    # This tests if simply adjusting logit confidence helps
    for alpha in [0.1, 0.3, 0.5, 0.8]:
        margins = []
        for f in test_facts:
            ctx = f["context"]
            logits_v, tok_t = get_logits(model, tokenizer, f"{ctx}: {f['truth']}", None)
            # Amplify: logits * (1 + alpha) effectively
            logits_amp = logits_v * (1 + alpha)
            truth_lp = logits_to_lp(logits_amp, tok_t)

            best_dist = -float('inf')
            for d in f["distractors"]:
                logits_v, tok_d = get_logits(model, tokenizer, f"{ctx}: {d}", None)
                logits_amp = logits_v * (1 + alpha)
                best_dist = max(best_dist, logits_to_lp(logits_amp, tok_d))
            margins.append(truth_lp - best_dist)

        n = sum(1 for m in margins if m > 0)
        print(f"  Amplify alpha={alpha}: {n}/3 margins={[f'{m:.1f}' for m in margins]}")

    print("\nDONE")


if __name__ == "__main__":
    main()
