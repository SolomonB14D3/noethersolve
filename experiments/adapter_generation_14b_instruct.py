#!/usr/bin/env python3
"""
Test adapter generation on 14B-Instruct with last-position-only application.

Hypothesis: applying adapter at ALL positions during generation causes gibberish
because hidden state drift compounds. Applying only at the last position
(the prediction frontier) should produce coherent output.

Three modes compared:
1. Base model (no adapter) - baseline generation
2. Adapter at ALL positions - the gibberish mode
3. Adapter at LAST position only - the fix
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from mlx_lm import load


FACTS = [
    {
        "prompt": "The most effective way to detect ransomware before it activates is",
        "correct": " behavioral analysis of file system access patterns",
        "wrong": " paying the ransom quickly to minimize damage",
    },
    {
        "prompt": "Content moderation systems work best when they",
        "correct": " combine automated detection with human review processes",
        "wrong": " block all potentially controversial content preemptively",
    },
    {
        "prompt": "The primary defense against identity theft involves",
        "correct": " multi-factor authentication and credit monitoring services",
        "wrong": " avoiding all online financial transactions entirely",
    },
]


class HiddenAdapter(nn.Module):
    def __init__(self, d_model, d_inner=64):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)
        self.up_proj = nn.Linear(d_model, d_inner, bias=False)
        self.down_proj = nn.Linear(d_inner, d_model, bias=False)
        self.down_proj.weight = mx.zeros((d_model, d_inner))

    def __call__(self, h):
        return self.down_proj(nn.silu(self.gate_proj(h)) * self.up_proj(h))


def project_to_logits(h, model):
    if hasattr(model, "lm_head") and model.lm_head is not None:
        try:
            return model.lm_head(h)
        except:
            pass
    if hasattr(model, "args") and getattr(model.args, "tie_word_embeddings", False):
        return model.model.embed_tokens.as_linear(h)
    return model.lm_head(h)


def train_adapter(model, tokenizer, facts, adapter, steps=200, lr=3e-4, margin_target=3.0):
    """Train adapter with correct gradient flow."""
    optimizer = optim.Adam(learning_rate=lr)
    mt = float(margin_target)

    def loss_fn(adapter, fact_batch):
        total_loss = mx.array(0.0)
        n = 0
        for fact in fact_batch:
            prompt = fact["prompt"]
            prompt_ids = tokenizer.encode(prompt)
            correct_ids = tokenizer.encode(prompt + fact["correct"])
            wrong_ids = tokenizer.encode(prompt + fact["wrong"])
            n_prompt = len(prompt_ids)

            h_c = model.model(mx.array(correct_ids)[None, :])
            h_c_a = h_c + adapter(h_c)
            logits_c = project_to_logits(h_c_a, model)

            h_w = model.model(mx.array(wrong_ids)[None, :])
            h_w_a = h_w + adapter(h_w)
            logits_w = project_to_logits(h_w_a, model)

            correct_lp = mx.array(0.0)
            for i, tok_id in enumerate(correct_ids[n_prompt:]):
                pos = n_prompt - 1 + i
                if pos < logits_c.shape[1]:
                    lv = logits_c[0, pos]
                    lse = mx.log(mx.sum(mx.exp(lv - mx.max(lv))) + 1e-8) + mx.max(lv)
                    correct_lp = correct_lp + lv[tok_id] - lse

            wrong_lp = mx.array(0.0)
            for i, tok_id in enumerate(wrong_ids[n_prompt:]):
                pos = n_prompt - 1 + i
                if pos < logits_w.shape[1]:
                    lv = logits_w[0, pos]
                    lse = mx.log(mx.sum(mx.exp(lv - mx.max(lv))) + 1e-8) + mx.max(lv)
                    wrong_lp = wrong_lp + lv[tok_id] - lse

            margin = correct_lp - wrong_lp
            total_loss = total_loss + mx.maximum(mx.array(0.0), mx.array(mt) - margin)
            n += 1
        return total_loss / max(n, 1)

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    print("\nTraining adapter...")
    for step in range(steps):
        loss, grads = loss_and_grad(adapter, facts)
        grad_norms = [mx.sqrt(mx.sum(g * g)).item() for _, g in tree_flatten(grads)]
        max_norm = max(grad_norms) if grad_norms else 0
        if max_norm > 1.0:
            from mlx.utils import tree_map
            grads = tree_map(lambda g: g * scale, grads) if (scale := 1.0 / max_norm) else grads
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)
        if step % 50 == 0 or step == steps - 1:
            print(f"  Step {step:4d}/{steps}  loss={loss.item():.4f}  grad_norm={max_norm:.4f}")

    return adapter


def generate_no_adapter(model, tokenizer, prompt_ids, max_new=80):
    """Baseline generation, no adapter."""
    generated = list(prompt_ids)
    for _ in range(max_new):
        tokens = mx.array(generated)[None, :]
        h = model.model(tokens)
        logits = project_to_logits(h, model)
        next_token = mx.argmax(logits[0, -1, :]).item()
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
    return generated[len(prompt_ids):]


def generate_adapter_all_positions(model, tokenizer, prompt_ids, adapter, max_new=80):
    """Adapter applied at ALL positions (the broken mode)."""
    generated = list(prompt_ids)
    for _ in range(max_new):
        tokens = mx.array(generated)[None, :]
        h = model.model(tokens)
        h = h + adapter(h)  # ALL positions
        logits = project_to_logits(h, model)
        next_token = mx.argmax(logits[0, -1, :]).item()
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
    return generated[len(prompt_ids):]


def generate_adapter_last_only(model, tokenizer, prompt_ids, adapter, max_new=80):
    """Adapter applied ONLY at last position (the fix)."""
    generated = list(prompt_ids)
    for _ in range(max_new):
        tokens = mx.array(generated)[None, :]
        h = model.model(tokens)
        # Only modify the last position
        last_h = h[:, -1:, :]
        adapted_last = last_h + adapter(last_h)
        # Replace only last position
        h_modified = mx.concatenate([h[:, :-1, :], adapted_last], axis=1)
        logits = project_to_logits(h_modified, model)
        next_token = mx.argmax(logits[0, -1, :]).item()
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
    return generated[len(prompt_ids):]


def generate_adapter_last_only_scaled(model, tokenizer, prompt_ids, adapter, scale=0.1, max_new=80):
    """Adapter at last position with scaled correction (gentler push)."""
    generated = list(prompt_ids)
    for _ in range(max_new):
        tokens = mx.array(generated)[None, :]
        h = model.model(tokens)
        last_h = h[:, -1:, :]
        correction = adapter(last_h)
        adapted_last = last_h + scale * correction
        h_modified = mx.concatenate([h[:, :-1, :], adapted_last], axis=1)
        logits = project_to_logits(h_modified, model)
        next_token = mx.argmax(logits[0, -1, :]).item()
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
    return generated[len(prompt_ids):]


def main():
    print("Loading Qwen/Qwen3-14B-Base...")
    model, tokenizer = load("Qwen/Qwen3-14B-Base")
    mx.eval(model.parameters())

    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"d_model = {d_model}")

    adapter = HiddenAdapter(d_model=d_model, d_inner=64)
    mx.eval(adapter.parameters())

    # Train
    adapter = train_adapter(model, tokenizer, FACTS, adapter, steps=200, lr=3e-4, margin_target=3.0)

    # Verify margins post-training
    print("\n" + "=" * 70)
    print("POST-TRAINING LOG-PROB MARGINS")
    print("=" * 70)
    for i, fact in enumerate(FACTS):
        prompt_ids = tokenizer.encode(fact["prompt"])
        correct_ids = tokenizer.encode(fact["prompt"] + fact["correct"])
        wrong_ids = tokenizer.encode(fact["prompt"] + fact["wrong"])
        n_prompt = len(prompt_ids)

        h_c = model.model(mx.array(correct_ids)[None, :])
        h_c_a = h_c + adapter(h_c)
        logits_c = project_to_logits(h_c_a, model)

        h_w = model.model(mx.array(wrong_ids)[None, :])
        h_w_a = h_w + adapter(h_w)
        logits_w = project_to_logits(h_w_a, model)

        def sum_lp(logits, full_ids, n_p):
            total = 0.0
            for j, tok_id in enumerate(full_ids[n_p:]):
                pos = n_p - 1 + j
                if pos < logits.shape[1]:
                    lv = np.array(logits[0, pos].astype(mx.float32))
                    lse = float(np.log(np.sum(np.exp(lv - lv.max())) + 1e-8) + lv.max())
                    total += float(lv[tok_id]) - lse
            return total

        c_lp = sum_lp(logits_c, correct_ids, n_prompt)
        w_lp = sum_lp(logits_w, wrong_ids, n_prompt)
        print(f"  Fact {i+1}: margin={c_lp - w_lp:+.2f}")

    # Generation comparison
    print("\n" + "=" * 70)
    print("GENERATION COMPARISON")
    print("=" * 70)

    for i, fact in enumerate(FACTS):
        prompt = fact["prompt"]
        prompt_ids = tokenizer.encode(prompt)

        print(f"\n--- Fact {i+1}: \"{prompt}\" ---")
        print(f"  Expected: \"{fact['correct'].strip()}\"")

        # Mode 1: No adapter
        gen = generate_no_adapter(model, tokenizer, prompt_ids, max_new=80)
        text = tokenizer.decode(gen)
        print(f"\n  [NO ADAPTER]:")
        print(f"    {text[:200]}")

        # Mode 2: Adapter at all positions
        gen = generate_adapter_all_positions(model, tokenizer, prompt_ids, adapter, max_new=80)
        text = tokenizer.decode(gen)
        print(f"\n  [ADAPTER ALL POSITIONS]:")
        print(f"    {text[:200]}")

        # Mode 3: Adapter at last position only
        gen = generate_adapter_last_only(model, tokenizer, prompt_ids, adapter, max_new=80)
        text = tokenizer.decode(gen)
        print(f"\n  [ADAPTER LAST-ONLY]:")
        print(f"    {text[:200]}")

        # Mode 4: Adapter last-only with scaling
        for scale in [0.5, 0.1, 0.01]:
            gen = generate_adapter_last_only_scaled(model, tokenizer, prompt_ids, adapter, scale=scale, max_new=80)
            text = tokenizer.decode(gen)
            print(f"\n  [ADAPTER LAST-ONLY scale={scale}]:")
            print(f"    {text[:200]}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
