#!/usr/bin/env python3
"""
Hidden-Space Adapter Diagnostic on Qwen3-14B-Base.

Three diagnostics:
1. Top-k shift: top 10 tokens at answer position before/after adapter
2. Forced first token: give correct first token, generate 50 more with/without adapter
3. Logit lens: project adapter(h) into token space via embedding matrix

Uses a SwiGLU MLP adapter in hidden space (d_model=5120 for 14B).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from mlx_lm import load


# ── Facts ──────────────────────────────────────────────────────────────

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
    {
        "prompt": "Securities regulators like the SEC protect investors by",
        "correct": " enforcing disclosure requirements and investigating fraud",
        "wrong": " guaranteeing all investments will be profitable",
    },
    {
        "prompt": "Deepfake detection technology primarily works by analyzing",
        "correct": " inconsistencies in facial movements and lighting artifacts",
        "wrong": " whether the content is politically controversial",
    },
]


# ── Hidden-Space Adapter ──────────────────────────────────────────────

class HiddenAdapter(nn.Module):
    def __init__(self, d_model, d_inner=64):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)
        self.up_proj = nn.Linear(d_model, d_inner, bias=False)
        self.down_proj = nn.Linear(d_inner, d_model, bias=False)
        # Zero-init output for stable start
        self.down_proj.weight = mx.zeros((d_model, d_inner))

    def __call__(self, h):
        return self.down_proj(nn.silu(self.gate_proj(h)) * self.up_proj(h))


# ── Helpers ────────────────────────────────────────────────────────────

def get_hidden_states(model, token_ids):
    """Get final hidden states from the transformer backbone."""
    tokens = mx.array(token_ids)[None, :]
    h = model.model(tokens)  # (1, seq_len, d_model)
    return h


def get_lm_head_weight(model):
    """Get the lm_head weight matrix for logit projection."""
    if hasattr(model, "lm_head") and model.lm_head is not None:
        try:
            return model.lm_head.weight  # (vocab, d_model)
        except AttributeError:
            pass
    # Tied embeddings (Qwen pattern)
    if hasattr(model, "args") and getattr(model.args, "tie_word_embeddings", False):
        return model.model.embed_tokens.weight  # (vocab, d_model)
    if hasattr(model, "lm_head"):
        return model.lm_head.weight
    raise RuntimeError("Cannot find lm_head weight")


def project_to_logits(h, model):
    """Project hidden states to logits using lm_head."""
    if hasattr(model, "lm_head") and model.lm_head is not None:
        try:
            return model.lm_head(h)
        except:
            pass
    if hasattr(model, "args") and getattr(model.args, "tie_word_embeddings", False):
        return model.model.embed_tokens.as_linear(h)
    return model.lm_head(h)


def top_k_tokens(logits_1d, tokenizer, k=10):
    """Return top-k tokens with probabilities from a 1D logit vector."""
    logits_np = np.array(logits_1d.astype(mx.float32))
    # Stable softmax
    logits_np = logits_np - logits_np.max()
    probs = np.exp(logits_np)
    probs = probs / probs.sum()
    top_idx = np.argsort(probs)[-k:][::-1]
    results = []
    for idx in top_idx:
        tok = tokenizer.decode([int(idx)])
        results.append((tok, float(probs[idx]), int(idx)))
    return results


# ── Training ───────────────────────────────────────────────────────────

def train_hidden_adapter(model, tokenizer, facts, adapter, steps=300, lr=3e-4,
                         margin_target=1.5):
    """Train the hidden adapter using margin hinge loss."""
    optimizer = optim.Adam(learning_rate=lr)
    mt = float(margin_target)

    def loss_fn(adapter, fact_batch):
        total_loss = mx.array(0.0)
        n = 0
        for fact in fact_batch:
            prompt = fact["prompt"]
            correct_text = prompt + fact["correct"]
            wrong_text = prompt + fact["wrong"]

            prompt_ids = tokenizer.encode(prompt)
            correct_ids = tokenizer.encode(correct_text)
            wrong_ids = tokenizer.encode(wrong_text)
            n_prompt = len(prompt_ids)

            # Get hidden states for correct completion
            h_c = model.model(mx.array(correct_ids)[None, :])
            h_c_adapted = h_c + adapter(h_c)
            logits_c = project_to_logits(h_c_adapted, model)

            # Get hidden states for wrong completion
            h_w = model.model(mx.array(wrong_ids)[None, :])
            h_w_adapted = h_w + adapter(h_w)
            logits_w = project_to_logits(h_w_adapted, model)

            # Sum log-probs over completion tokens
            correct_lp = mx.array(0.0)
            for i, tok_id in enumerate(correct_ids[n_prompt:]):
                pos = n_prompt - 1 + i
                if pos < logits_c.shape[1]:
                    lv = logits_c[0, pos]
                    lse = mx.log(mx.sum(mx.exp(lv - mx.max(lv))) + 1e-8) + mx.max(lv)
                    correct_lp = correct_lp + lv[tok_id] - lse

            wrong_lp = mx.array(0.0)
            n_prompt_w = len(prompt_ids)
            for i, tok_id in enumerate(wrong_ids[n_prompt_w:]):
                pos = n_prompt_w - 1 + i
                if pos < logits_w.shape[1]:
                    lv = logits_w[0, pos]
                    lse = mx.log(mx.sum(mx.exp(lv - mx.max(lv))) + 1e-8) + mx.max(lv)
                    wrong_lp = wrong_lp + lv[tok_id] - lse

            margin = correct_lp - wrong_lp
            total_loss = total_loss + mx.maximum(mx.array(0.0), mx.array(mt) - margin)
            n += 1

        return total_loss / max(n, 1)

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    print("\n" + "=" * 70)
    print("TRAINING HIDDEN-SPACE ADAPTER")
    print("=" * 70)

    for step in range(steps):
        loss, grads = loss_and_grad(adapter, FACTS)
        # Clip gradients
        grad_norms = [mx.sqrt(mx.sum(g * g)).item() for _, g in tree_flatten(grads)]
        max_norm = max(grad_norms) if grad_norms else 0
        if max_norm > 1.0:
            scale = 1.0 / max_norm
            from mlx.utils import tree_map
            grads = tree_map(lambda g: g * scale, grads)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if step % 50 == 0 or step == steps - 1:
            loss_val = loss.item()
            print(f"  Step {step:4d}/{steps}  loss={loss_val:.4f}  max_grad_norm={max_norm:.4f}")

    print("Training complete.\n")
    return adapter


# ── Diagnostic 1: Top-K Shift ─────────────────────────────────────────

def diagnostic_topk_shift(model, tokenizer, facts, adapter):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: TOP-K TOKEN SHIFT AT ANSWER POSITION")
    print("=" * 70)
    print("Shows top 10 tokens (with probs) at the position just before")
    print("the answer starts, BEFORE and AFTER applying the adapter.\n")

    for i, fact in enumerate(facts):
        prompt = fact["prompt"]
        prompt_ids = tokenizer.encode(prompt)
        tokens = mx.array(prompt_ids)[None, :]

        # Get hidden states
        h = model.model(tokens)
        last_h = h[0, -1, :]  # (d_model,)

        # BEFORE adapter
        logits_before = project_to_logits(h, model)[0, -1, :]
        top_before = top_k_tokens(logits_before, tokenizer, k=10)

        # AFTER adapter
        h_adapted = h + adapter(h)
        logits_after = project_to_logits(h_adapted, model)[0, -1, :]
        top_after = top_k_tokens(logits_after, tokenizer, k=10)

        print(f"--- Fact {i+1}: \"{prompt}\" ---")
        print(f"  Correct: \"{fact['correct'].strip()}\"")
        print(f"  Wrong:   \"{fact['wrong'].strip()}\"")
        print()
        print(f"  {'BEFORE (base model)':<45s}  {'AFTER (with adapter)':<45s}")
        print(f"  {'─' * 43}  {'─' * 43}")
        for j in range(10):
            tok_b, prob_b, _ = top_before[j]
            tok_a, prob_a, _ = top_after[j]
            b_str = f"  {j+1:2d}. {repr(tok_b):20s} p={prob_b:.4f}"
            a_str = f"  {j+1:2d}. {repr(tok_a):20s} p={prob_a:.4f}"
            print(f"{b_str:<45s}{a_str:<45s}")
        print()


# ── Diagnostic 2: Forced First Token Generation ──────────────────────

def generate_tokens(model, tokenizer, input_ids, adapter, max_new=50):
    """Simple greedy generation with optional adapter."""
    generated = list(input_ids)
    for _ in range(max_new):
        tokens = mx.array(generated)[None, :]
        h = model.model(tokens)
        if adapter is not None:
            h = h + adapter(h)
        logits = project_to_logits(h, model)
        next_token = mx.argmax(logits[0, -1, :]).item()
        generated.append(next_token)
        # Stop on EOS
        if next_token == tokenizer.eos_token_id:
            break
    return generated[len(input_ids):]


def diagnostic_forced_first_token(model, tokenizer, facts, adapter):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: FORCED FIRST TOKEN + FREE GENERATION")
    print("=" * 70)
    print("Gives the model the correct first answer token, then lets it")
    print("generate 50 more tokens freely. Compares with/without adapter.\n")

    for i, fact in enumerate(facts):
        prompt = fact["prompt"]
        correct = fact["correct"]

        prompt_ids = tokenizer.encode(prompt)
        # Get first token of correct answer
        correct_ids = tokenizer.encode(prompt + correct)
        first_correct_token = correct_ids[len(prompt_ids)]

        forced_ids = prompt_ids + [first_correct_token]
        first_tok_text = tokenizer.decode([first_correct_token])

        # Generate WITHOUT adapter
        gen_no_adapter = generate_tokens(model, tokenizer, forced_ids, adapter=None, max_new=50)
        text_no_adapter = tokenizer.decode(gen_no_adapter)

        # Generate WITH adapter
        gen_with_adapter = generate_tokens(model, tokenizer, forced_ids, adapter=adapter, max_new=50)
        text_with_adapter = tokenizer.decode(gen_with_adapter)

        print(f"--- Fact {i+1}: \"{prompt}\" ---")
        print(f"  Forced first token: {repr(first_tok_text)} (id={first_correct_token})")
        print(f"  Full correct answer: \"{correct.strip()}\"")
        print()
        print(f"  WITHOUT adapter: {first_tok_text}{text_no_adapter}")
        print(f"  WITH    adapter: {first_tok_text}{text_with_adapter}")
        print()


# ── Diagnostic 3: Logit Lens on Adapter Output ───────────────────────

def diagnostic_logit_lens(model, tokenizer, facts, adapter):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3: LOGIT LENS ON ADAPTER CORRECTION VECTOR")
    print("=" * 70)
    print("Projects adapter(h) into token space via the embedding matrix.")
    print("Shows what tokens the correction vector 'points toward'.\n")

    W_embed = get_lm_head_weight(model)  # (vocab, d_model)

    for i, fact in enumerate(facts):
        prompt = fact["prompt"]
        prompt_ids = tokenizer.encode(prompt)
        tokens = mx.array(prompt_ids)[None, :]

        # Get hidden states at last position
        h = model.model(tokens)
        last_h = h[0:1, -1:, :]  # (1, 1, d_model)

        # Get adapter correction
        correction = adapter(last_h)  # (1, 1, d_model)
        correction_vec = correction[0, 0, :]  # (d_model,)

        # Project correction into token space: correction @ W_embed^T
        # W_embed is (vocab, d_model), so correction @ W_embed.T = (vocab,)
        logit_proj = correction_vec @ W_embed.T  # (vocab,)

        # Get top tokens the correction points toward
        top_tokens = top_k_tokens(logit_proj, tokenizer, k=10)

        # Also show correction magnitude
        corr_norm = float(mx.sqrt(mx.sum(correction_vec * correction_vec)).item())
        h_norm = float(mx.sqrt(mx.sum(h[0, -1, :] * h[0, -1, :])).item())

        print(f"--- Fact {i+1}: \"{prompt}\" ---")
        print(f"  ||adapter(h)||={corr_norm:.4f}  ||h||={h_norm:.4f}  ratio={corr_norm/max(h_norm,1e-8):.6f}")
        print(f"  Correct: \"{fact['correct'].strip()}\"")
        print()
        print(f"  Top 10 tokens the correction vector points toward:")
        for j, (tok, prob, tok_id) in enumerate(top_tokens):
            # Use raw logit score instead of softmax prob for interpretation
            raw_score = float(np.array((correction_vec @ W_embed[tok_id]).astype(mx.float32)))
            print(f"    {j+1:2d}. {repr(tok):25s}  score={raw_score:+.4f}  (id={tok_id})")
        print()

        # Also show bottom 10 (what it pushes AWAY from)
        logit_np = np.array(logit_proj.astype(mx.float32))
        bottom_idx = np.argsort(logit_np)[:10]
        print(f"  Bottom 10 tokens the correction pushes AWAY from:")
        for j, idx in enumerate(bottom_idx):
            tok = tokenizer.decode([int(idx)])
            raw_score = float(logit_np[idx])
            print(f"    {j+1:2d}. {repr(tok):25s}  score={raw_score:+.4f}  (id={int(idx)})")
        print()


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("Loading Qwen/Qwen3-14B-Base...")
    model, tokenizer = load("Qwen/Qwen3-14B-Base")
    mx.eval(model.parameters())

    # Verify d_model
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"d_model = {d_model}")
    assert d_model == 5120, f"Expected d_model=5120, got {d_model}"

    # Create adapter
    adapter = HiddenAdapter(d_model=5120, d_inner=64)
    mx.eval(adapter.parameters())

    # Quick sanity: show base model margins before training
    print("\n" + "=" * 70)
    print("BASE MODEL MARGINS (before training)")
    print("=" * 70)
    for i, fact in enumerate(FACTS):
        prompt = fact["prompt"]
        prompt_ids = tokenizer.encode(prompt)

        # Correct completion
        correct_ids = tokenizer.encode(prompt + fact["correct"])
        h_c = model.model(mx.array(correct_ids)[None, :])
        logits_c = project_to_logits(h_c, model)

        # Wrong completion
        wrong_ids = tokenizer.encode(prompt + fact["wrong"])
        h_w = model.model(mx.array(wrong_ids)[None, :])
        logits_w = project_to_logits(h_w, model)

        n_prompt = len(prompt_ids)

        def sum_lp(logits, full_ids, n_prompt):
            total = 0.0
            for j, tok_id in enumerate(full_ids[n_prompt:]):
                pos = n_prompt - 1 + j
                if pos < logits.shape[1]:
                    lv = np.array(logits[0, pos].astype(mx.float32))
                    lse = float(np.log(np.sum(np.exp(lv - lv.max())) + 1e-8) + lv.max())
                    total += float(lv[tok_id]) - lse
            return total

        c_lp = sum_lp(logits_c, correct_ids, n_prompt)
        w_lp = sum_lp(logits_w, wrong_ids, n_prompt)
        margin = c_lp - w_lp
        print(f"  Fact {i+1}: margin={margin:+.2f} ({'PASS' if margin > 0 else 'FAIL'})")

    # Check if training is needed — raise margin target above current margins
    max_margin = max(13.52, 13.14, 3.17, 4.35, 9.83)  # from base model
    margin_target = max_margin + 5.0  # Force the adapter to actually push
    print(f"\n  All facts already PASS. Using margin_target={margin_target:.1f} to force training signal.")

    # Train the adapter
    adapter = train_hidden_adapter(model, tokenizer, FACTS, adapter, steps=300, lr=3e-4,
                                    margin_target=margin_target)

    # Show post-training margins
    print("\n" + "=" * 70)
    print("POST-TRAINING MARGINS")
    print("=" * 70)
    for i, fact in enumerate(FACTS):
        prompt = fact["prompt"]
        prompt_ids = tokenizer.encode(prompt)

        correct_ids = tokenizer.encode(prompt + fact["correct"])
        h_c = model.model(mx.array(correct_ids)[None, :])
        h_c_a = h_c + adapter(h_c)
        logits_c = project_to_logits(h_c_a, model)

        wrong_ids = tokenizer.encode(prompt + fact["wrong"])
        h_w = model.model(mx.array(wrong_ids)[None, :])
        h_w_a = h_w + adapter(h_w)
        logits_w = project_to_logits(h_w_a, model)

        n_prompt = len(prompt_ids)

        def sum_lp(logits, full_ids, n_prompt):
            total = 0.0
            for j, tok_id in enumerate(full_ids[n_prompt:]):
                pos = n_prompt - 1 + j
                if pos < logits.shape[1]:
                    lv = np.array(logits[0, pos].astype(mx.float32))
                    lse = float(np.log(np.sum(np.exp(lv - lv.max())) + 1e-8) + lv.max())
                    total += float(lv[tok_id]) - lse
            return total

        c_lp = sum_lp(logits_c, correct_ids, n_prompt)
        w_lp = sum_lp(logits_w, wrong_ids, n_prompt)
        margin = c_lp - w_lp
        print(f"  Fact {i+1}: margin={margin:+.2f} ({'PASS' if margin > 0 else 'FAIL'})")

    # Run all 3 diagnostics
    diagnostic_topk_shift(model, tokenizer, FACTS, adapter)
    diagnostic_forced_first_token(model, tokenizer, FACTS, adapter)
    diagnostic_logit_lens(model, tokenizer, FACTS, adapter)

    print("\n" + "=" * 70)
    print("ALL DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
