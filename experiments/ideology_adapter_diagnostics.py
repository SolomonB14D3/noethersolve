#!/usr/bin/env python3
"""
Quick diagnostics on what the adapter is actually doing:
1. How big is the adapter output vs the hidden state?
2. Are gradients flowing?
3. What if we scale the adapter output by 10x, 100x?
4. What if we use MSE loss instead of hinge?
5. What if we bypass the adapter entirely and just learn a bias vector?
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


class ScaledAdapter(nn.Module):
    """SwiGLU with learnable output scale."""
    def __init__(self, d_model, d_inner, init_scale=10.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)
        self.up_proj = nn.Linear(d_model, d_inner, bias=False)
        self.down_proj = nn.Linear(d_inner, d_model, bias=False)
        self.scale = mx.array(init_scale)

    def __call__(self, h):
        return self.scale * self.down_proj(nn.sigmoid(self.gate_proj(h)) * self.up_proj(h))


class BiasAdapter(nn.Module):
    """Just a learned bias vector added to hidden state."""
    def __init__(self, d_model):
        super().__init__()
        self.bias = mx.zeros((d_model,))

    def __call__(self, h):
        return mx.broadcast_to(self.bias, h.shape)


def get_logprob(model, tokenizer, text, adapter=None, scale=1.0):
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
        correction = adapter(h)
        h = h + scale * correction
        if hasattr(model, 'lm_head'):
            logits = model.lm_head(h)
        else:
            logits = model.model.embed_tokens.as_linear(h)
        logits = logits.astype(mx.float32)
    log_probs = nn.log_softmax(logits, axis=-1)
    targets = mx.array([tokens[1:]])
    token_lps = mx.take_along_axis(log_probs[0], targets[0][:, None], axis=-1).squeeze(-1)
    return float(mx.sum(token_lps))


def get_margin(model, tokenizer, fact, adapter=None, scale=1.0):
    ctx = fact["context"]
    truth_lp = get_logprob(model, tokenizer, f"{ctx}: {fact['truth']}", adapter, scale)
    best_dist = max(get_logprob(model, tokenizer, f"{ctx}: {d}", adapter, scale) for d in fact["distractors"])
    return truth_lp - best_dist


def train_adapter_generic(model, tokenizer, adapter, facts, steps, lr, label, loss_type="hinge"):
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

            if loss_type == "hinge":
                total_loss = total_loss + mx.maximum(mx.array(0.0), mx.array(1.5) - margin)
            elif loss_type == "negate":
                # Directly maximize margin (minimize negative margin)
                total_loss = total_loss - margin
            elif loss_type == "exp":
                # Exponential penalty for negative margins
                total_loss = total_loss + mx.exp(-margin / 10.0)

        return total_loss / len(batch)

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    for step in range(steps):
        loss, grads = loss_and_grad(adapter.parameters(), facts)

        # Check gradient magnitude
        flat_grads = tree_flatten(grads)
        grad_norm = sum(float(mx.sum(v * v)) for _, v in flat_grads if isinstance(v, mx.array))
        grad_norm = grad_norm ** 0.5

        if grad_norm > 1.0:
            grads = tree_map(lambda g: g * (1.0/grad_norm) if isinstance(g, mx.array) else g, grads)

        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if (step + 1) % 200 == 0:
            margins = [get_margin(model, tokenizer, f, adapter) for f in facts]
            n_pass = sum(1 for m in margins if m > 0)
            print(f"  [{label}] Step {step+1}: loss={float(loss):.2f}, grad_norm={grad_norm:.4f}, {n_pass}/3, margins={[f'{m:.1f}' for m in margins]}")

    return adapter


def main():
    project_root = Path(__file__).parent.parent

    print("Loading Qwen3-8B-Base...")
    model, tokenizer = mlx_load("Qwen/Qwen3-8B-Base")
    model.eval()
    d_model = model.model.embed_tokens.weight.shape[1]

    with open(project_root / "problems" / "ideology_facts_frank.json") as f:
        all_facts = json.load(f)

    test_ids = ["ideo_xinj_L4", "ideo_tibet_L2", "ideo_tian_L4"]
    facts = [f for f in all_facts if f["id"] in test_ids]

    # Baseline
    print("\n=== BASELINE ===")
    for f in facts:
        m = get_margin(model, tokenizer, f)
        print(f"  {f['id']}: {m:.2f}")

    # Diagnostic 1: How big is adapter output vs hidden state?
    print("\n=== DIAGNOSTIC 1: Adapter output magnitude ===")
    adapter = SwiGLUAdapter(d_model, 512)
    mx.eval(adapter.parameters())
    for f in facts:
        ctx = f["context"]
        tokens = tokenizer.encode(f"{ctx}: {f['truth']}")
        x = mx.array([tokens[:-1]])
        h = model.model(x)
        if hasattr(model.model, 'norm'):
            h = model.model.norm(h)
        correction = adapter(h)
        h_norm = float(mx.sqrt(mx.sum(h * h)))
        c_norm = float(mx.sqrt(mx.sum(correction * correction)))
        ratio = c_norm / h_norm if h_norm > 0 else 0
        print(f"  {f['id']}: |h|={h_norm:.1f}, |correction|={c_norm:.4f}, ratio={ratio:.6f}")

    # Diagnostic 2: What if we scale up the correction?
    print("\n=== DIAGNOSTIC 2: Scaling untrained adapter output ===")
    for scale in [1, 10, 100, 1000]:
        margins = [get_margin(model, tokenizer, f, adapter, scale=scale) for f in facts]
        print(f"  scale={scale}: margins={[f'{m:.1f}' for m in margins]}")

    # Approach 1: Hinge loss (control)
    print("\n=== APPROACH 1: Hinge loss, lr=5e-5, 1000 steps ===")
    a1 = SwiGLUAdapter(d_model, 512)
    mx.eval(a1.parameters())
    a1 = train_adapter_generic(model, tokenizer, a1, facts, 1000, 5e-5, "hinge", "hinge")

    # Approach 2: Negate margin loss (directly maximize margin)
    print("\n=== APPROACH 2: Negate margin loss, lr=5e-5, 1000 steps ===")
    a2 = SwiGLUAdapter(d_model, 512)
    mx.eval(a2.parameters())
    a2 = train_adapter_generic(model, tokenizer, a2, facts, 1000, 5e-5, "negate", "negate")

    # Approach 3: Exponential loss
    print("\n=== APPROACH 3: Exponential loss, lr=5e-5, 1000 steps ===")
    a3 = SwiGLUAdapter(d_model, 512)
    mx.eval(a3.parameters())
    a3 = train_adapter_generic(model, tokenizer, a3, facts, 1000, 5e-5, "exp", "exp")

    # Approach 4: Bias vector only (simplest possible intervention)
    print("\n=== APPROACH 4: Bias vector, lr=1e-3, 1000 steps ===")
    a4 = BiasAdapter(d_model)
    mx.eval(a4.parameters())
    a4 = train_adapter_generic(model, tokenizer, a4, facts, 1000, 1e-3, "bias", "hinge")

    # Approach 5: Scaled adapter (init_scale=10)
    print("\n=== APPROACH 5: Scaled SwiGLU (scale=10), lr=5e-5, 1000 steps ===")
    a5 = ScaledAdapter(d_model, 512, init_scale=10.0)
    mx.eval(a5.parameters())
    a5 = train_adapter_generic(model, tokenizer, a5, facts, 1000, 5e-5, "scaled10", "hinge")

    # Approach 6: No gradient clipping
    print("\n=== APPROACH 6: No grad clip, lr=5e-4, 1000 steps ===")
    a6 = SwiGLUAdapter(d_model, 512)
    mx.eval(a6.parameters())
    opt6 = optim.AdamW(learning_rate=5e-4, weight_decay=0.01)

    def loss_fn6(params, batch):
        a6.update(params)
        total_loss = mx.array(0.0)
        for fact in batch:
            ctx = fact["context"]
            tokens_t = tokenizer.encode(f"{ctx}: {fact['truth']}")
            x_t = mx.array([tokens_t[:-1]])
            h_t = model.model(x_t)
            if hasattr(model.model, 'norm'):
                h_t = model.model.norm(h_t)
            h_t = h_t + a6(h_t)
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
                h_d = h_d + a6(h_d)
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

    lg6 = nn.value_and_grad(a6, loss_fn6)
    for step in range(1000):
        loss, grads = lg6(a6.parameters(), facts)
        # NO gradient clipping
        flat_grads = tree_flatten(grads)
        grad_norm = sum(float(mx.sum(v * v)) for _, v in flat_grads if isinstance(v, mx.array)) ** 0.5
        opt6.update(a6, grads)
        mx.eval(a6.parameters(), opt6.state)
        if (step + 1) % 200 == 0:
            margins = [get_margin(model, tokenizer, f, a6) for f in facts]
            n_pass = sum(1 for m in margins if m > 0)
            print(f"  [noclip] Step {step+1}: loss={float(loss):.2f}, grad_norm={grad_norm:.4f}, {n_pass}/3, margins={[f'{m:.1f}' for m in margins]}")

    print("\nDONE")


if __name__ == "__main__":
    main()
