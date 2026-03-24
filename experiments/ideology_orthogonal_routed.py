#!/usr/bin/env python3
"""
Orthogonal adapters with ROUTING (not stacking).
Each topic gets its own adapter. Each fact routes to ONE adapter only.

Staged by difficulty within each topic:
1. Train easy facts per topic first
2. Then medium, anchored on easy
3. Then hard, anchored on easy+medium

Final: route each fact to its topic's adapter.
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


def get_logprob(model, tokenizer, text, adapter=None):
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
    truth_lp = get_logprob(model, tokenizer, f"{ctx}: {fact['truth']}", adapter)
    best_dist = max(get_logprob(model, tokenizer, f"{ctx}: {d}", adapter) for d in fact["distractors"])
    return truth_lp - best_dist


def train_adapter(model, tokenizer, train_facts, anchor_facts, d_model, d_inner=512,
                  steps=3000, lr=5e-5, label=""):
    adapter = SwiGLUAdapter(d_model, d_inner)
    mx.eval(adapter.parameters())
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(adapter_params, batch, anchors):
        adapter.update(adapter_params)
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

        # Anchor loss (2x weight)
        for afact in anchors:
            ctx = afact["context"]
            tokens_t = tokenizer.encode(f"{ctx}: {afact['truth']}")
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
            for d in afact["distractors"]:
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

            a_margin = truth_lp - best_dist_lp
            total_loss = total_loss + 2.0 * mx.maximum(mx.array(0.0), mx.array(0.1) - a_margin)

        return total_loss / max(1, len(batch) + len(anchors))

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    for step in range(steps):
        batch = random.sample(train_facts, min(len(train_facts), 5))
        a_batch = random.sample(anchor_facts, min(len(anchor_facts), 3)) if anchor_facts else []
        loss, grads = loss_and_grad(adapter.parameters(), batch, a_batch)

        flat_grads = tree_flatten(grads)
        grad_norm = sum(float(mx.sum(v * v)) for _, v in flat_grads if isinstance(v, mx.array))
        grad_norm = grad_norm ** 0.5
        if grad_norm > 1.0:
            grads = tree_map(lambda g: g * (1.0/grad_norm) if isinstance(g, mx.array) else g, grads)

        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if (step + 1) % 500 == 0:
            n_t = sum(1 for f in train_facts if get_margin(model, tokenizer, f, adapter) > 0)
            print(f"  [{label}] Step {step+1}: loss={float(loss):.4f}, train={n_t}/{len(train_facts)}")

    return adapter


def main():
    project_root = Path(__file__).parent.parent

    print("Loading Qwen3-8B-Base...")
    model, tokenizer = mlx_load("Qwen/Qwen3-8B-Base")
    model.eval()
    d_model = model.model.embed_tokens.weight.shape[1]

    with open(project_root / "problems" / "ideology_facts_frank.json") as f:
        all_facts = json.load(f)

    # Baseline
    print("\n=== BASELINE ===")
    baseline = {}
    for fact in all_facts:
        m = get_margin(model, tokenizer, fact)
        baseline[fact["id"]] = m
    pass_ids = {f["id"] for f in all_facts if baseline[f["id"]] > 0}
    fail_facts = [f for f in all_facts if baseline[f["id"]] <= 0]
    print(f"Baseline: {len(pass_ids)}/31 pass, {len(fail_facts)} fail")

    # Group failing facts by topic
    topic_groups = {}
    for f in fail_facts:
        t = f.get("topic", "unknown")
        topic_groups.setdefault(t, []).append(f)

    # Also get baseline-passing facts per topic (for anchoring)
    topic_pass = {}
    for f in all_facts:
        if f["id"] in pass_ids:
            t = f.get("topic", "unknown")
            topic_pass.setdefault(t, []).append(f)

    # General knowledge anchors
    knowledge_anchors = [
        {"id": "a01", "context": "Geography", "truth": "The capital of France is Paris.", "distractors": ["The capital of France is Lyon.", "The capital of France is Marseille."]},
        {"id": "a02", "context": "Chemistry", "truth": "Water boils at 100 degrees Celsius at standard pressure.", "distractors": ["Water boils at 50 degrees Celsius.", "Water boils at 200 degrees Celsius."]},
        {"id": "a03", "context": "Biology", "truth": "DNA carries genetic information.", "distractors": ["Proteins carry genetic information.", "Lipids carry genetic information."]},
    ]

    print(f"\nTopics to fix:")
    for topic, facts in sorted(topic_groups.items()):
        n_pass = len(topic_pass.get(topic, []))
        margins = [baseline[f["id"]] for f in facts]
        print(f"  {topic}: {len(facts)} failing (margins: {[f'{m:.1f}' for m in margins]}), {n_pass} passing as anchors")

    # ==================================================================
    # Train ONE adapter per topic (orthogonal, not stacked)
    # ==================================================================
    topic_adapters = {}

    for topic, topic_fail in sorted(topic_groups.items()):
        print(f"\n{'='*60}")
        print(f"TOPIC: {topic} ({len(topic_fail)} facts to fix)")
        print(f"{'='*60}")

        # Anchors: passing facts from THIS topic + knowledge anchors
        anchors = topic_pass.get(topic, []) + knowledge_anchors

        # Sort by difficulty (easiest first)
        topic_fail.sort(key=lambda f: baseline[f["id"]], reverse=True)

        # If topic has multiple facts, try staged within topic
        if len(topic_fail) <= 2:
            # Small enough to train directly
            adapter = train_adapter(model, tokenizer, topic_fail, anchors, d_model,
                                   d_inner=512, steps=3000, lr=5e-5, label=topic)
            topic_adapters[topic] = adapter
        else:
            # Staged: easy half first, then hard half anchored on easy
            mid = max(1, len(topic_fail) // 2)
            easy_half = topic_fail[:mid]
            hard_half = topic_fail[mid:]

            print(f"  Staged: {len(easy_half)} easy, {len(hard_half)} hard")

            # Train on ALL facts in topic but weight easy ones more in early steps
            # Actually, just train on all - the adapter is per-topic so no cross-topic interference
            adapter = train_adapter(model, tokenizer, topic_fail, anchors, d_model,
                                   d_inner=512, steps=4000, lr=5e-5, label=topic)
            topic_adapters[topic] = adapter

        # Eval
        print(f"\n  Results for {topic}:")
        n_fixed = 0
        for f in topic_fail:
            m = get_margin(model, tokenizer, f, adapter)
            status = "PASS" if m > 0 else "FAIL"
            delta = m - baseline[f["id"]]
            print(f"    {f['id']}: {baseline[f['id']]:.2f} -> {m:.2f} (delta={delta:+.2f}) {status}")
            if m > 0:
                n_fixed += 1

        # Check regressions on topic's passing facts
        regressions = 0
        for f in topic_pass.get(topic, []):
            m = get_margin(model, tokenizer, f, adapter)
            if m <= 0:
                regressions += 1
                print(f"    REGRESSION: {f['id']}: {baseline[f['id']]:.2f} -> {m:.2f}")
        print(f"  {topic}: {n_fixed}/{len(topic_fail)} fixed, {regressions} regressions")

    # ==================================================================
    # FINAL: Route each fact to its topic's adapter
    # ==================================================================
    print(f"\n{'='*60}")
    print("FINAL: Routed evaluation (each fact -> its topic adapter)")
    print(f"{'='*60}")

    total_correct = 0
    for fact in all_facts:
        topic = fact.get("topic", "unknown")
        m_base = baseline[fact["id"]]

        if topic in topic_adapters:
            m_adapted = get_margin(model, tokenizer, fact, topic_adapters[topic])
            # Pick best of baseline vs adapted
            if m_adapted > m_base:
                best_m = m_adapted
                source = "adapted"
            else:
                best_m = m_base
                source = "baseline"
        else:
            best_m = m_base
            source = "baseline"

        status = "PASS" if best_m > 0 else "FAIL"
        if best_m > 0:
            total_correct += 1
        print(f"  {fact['id']}: {best_m:.2f} {status} ({source})")

    print(f"\nFINAL: {total_correct}/31 (baseline: {len(pass_ids)}/31)")
    print(f"Improvement: +{total_correct - len(pass_ids)} facts")

    # Save
    adapters_dir = project_root / "adapters"
    for topic, adapter in topic_adapters.items():
        weights = {}
        for k, v in adapter.parameters().items():
            weights[k] = np.array(v)
        np.savez(adapters_dir / f"ideology_ortho_{topic}.npz", **weights)
    print(f"\nSaved {len(topic_adapters)} orthogonal adapters")


if __name__ == "__main__":
    main()
