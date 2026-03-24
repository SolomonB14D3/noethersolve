#!/usr/bin/env python3
"""
Full escalation ladder on ideology facts:
Stage 1: Easy facts (borderline margins, flipped in memorization test)
Stage 2: Medium facts (anchored on Stage 1)
Stage 3: Hard facts (anchored on Stage 1+2)
Stage 4: Orthogonal adapters by topic for remaining holdouts
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


def get_adapted_logprob(model, tokenizer, text, adapters):
    """Apply one or more adapters (summed corrections)."""
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return 0.0
    x = mx.array([tokens[:-1]])
    h = model.model(x)
    if hasattr(model.model, 'norm'):
        h = model.model.norm(h)
    for adapter in adapters:
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


def get_margin(model, tokenizer, fact, adapters=None):
    if adapters is None:
        adapters = []
    ctx = fact["context"]
    truth_lp = get_adapted_logprob(model, tokenizer, f"{ctx}: {fact['truth']}", adapters)
    best_dist = max(get_adapted_logprob(model, tokenizer, f"{ctx}: {d}", adapters) for d in fact["distractors"])
    return truth_lp - best_dist


def train_adapter(model, tokenizer, adapter, train_facts, anchor_facts, frozen_adapters=None,
                  steps=2000, lr=5e-5, tau=1.5, batch_size=5, label=""):
    """Train adapter with optional frozen adapters applied first."""
    if frozen_adapters is None:
        frozen_adapters = []

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(adapter_params, batch, anchors):
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
            # Apply frozen adapters first
            for fa in frozen_adapters:
                h_t = h_t + fa(h_t)
            # Then trainable adapter
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
                for fa in frozen_adapters:
                    h_d = h_d + fa(h_d)
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
            for fa in frozen_adapters:
                h_t = h_t + fa(h_t)
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
                for fa in frozen_adapters:
                    h_d = h_d + fa(h_d)
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
            total_loss = total_loss + 2.0 * mx.maximum(mx.array(0.0), mx.array(0.1) - a_margin)

        return total_loss / (len(batch) + len(anchors))

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    for step in range(steps):
        batch = random.sample(train_facts, min(batch_size, len(train_facts)))
        a_batch = random.sample(anchor_facts, min(3, len(anchor_facts))) if anchor_facts else []

        loss, grads = loss_and_grad(adapter.parameters(), batch, a_batch)

        flat_grads = tree_flatten(grads)
        grad_norm = sum(float(mx.sum(v * v)) for _, v in flat_grads if isinstance(v, mx.array))
        grad_norm = grad_norm ** 0.5
        if grad_norm > 1.0:
            grads = tree_map(lambda g: g * (1.0/grad_norm) if isinstance(g, mx.array) else g, grads)

        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if (step + 1) % 500 == 0:
            all_adapters = frozen_adapters + [adapter]
            n_train = sum(1 for f in train_facts if get_margin(model, tokenizer, f, all_adapters) > 0)
            n_anchor = sum(1 for f in anchor_facts if get_margin(model, tokenizer, f, all_adapters) > 0) if anchor_facts else 0
            print(f"  [{label}] Step {step+1}: loss={float(loss):.4f}, train={n_train}/{len(train_facts)}, anchor={n_anchor}/{len(anchor_facts)}")

    return adapter


def eval_all(model, tokenizer, facts, adapters, label=""):
    results = {}
    for fact in facts:
        m = get_margin(model, tokenizer, fact, adapters)
        status = "PASS" if m > 0 else "FAIL"
        results[fact["id"]] = m
        print(f"  {fact['id']}: {m:.2f} {status}")
    n_pass = sum(1 for m in results.values() if m > 0)
    print(f"  {label}: {n_pass}/{len(facts)}")
    return results, n_pass


def main():
    project_root = Path(__file__).parent.parent

    print("Loading Qwen3-8B-Base...")
    model, tokenizer = mlx_load("Qwen/Qwen3-8B-Base")
    model.eval()
    d_model = model.model.embed_tokens.weight.shape[1]
    print(f"d_model: {d_model}")

    with open(project_root / "problems" / "ideology_facts_frank.json") as f:
        all_facts = json.load(f)

    # Baseline
    print("\n=== BASELINE ===")
    baseline = {}
    for fact in all_facts:
        m = get_margin(model, tokenizer, fact)
        baseline[fact["id"]] = m
        print(f"  {fact['id']}: {m:.2f} {'PASS' if m > 0 else 'FAIL'}")

    pass_facts = [f for f in all_facts if baseline[f["id"]] > 0]
    fail_facts = [f for f in all_facts if baseline[f["id"]] <= 0]
    print(f"\nBaseline: {len(pass_facts)}/31 pass, {len(fail_facts)} fail")

    # Sort failing facts by margin (easiest first)
    fail_facts.sort(key=lambda f: baseline[f["id"]], reverse=True)
    print("\nFailing facts by difficulty:")
    for f in fail_facts:
        print(f"  {f['id']}: {baseline[f['id']]:.2f}")

    # Tier the failing facts
    # Easy: margin > -5 (borderline)
    # Medium: -5 to -12
    # Hard: < -12
    easy = [f for f in fail_facts if baseline[f["id"]] > -5]
    medium = [f for f in fail_facts if -12 <= baseline[f["id"]] <= -5]
    hard = [f for f in fail_facts if baseline[f["id"]] < -12]

    print(f"\nTiers: easy={len(easy)}, medium={len(medium)}, hard={len(hard)}")
    print("Easy:", [f["id"] for f in easy])
    print("Medium:", [f["id"] for f in medium])
    print("Hard:", [f["id"] for f in hard])

    d_inner = 512
    frozen_adapters = []
    all_trained_adapters = []

    # ==================================================================
    # STAGE 1: Easy facts (use pass_facts as anchors)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 1: Easy facts ({len(easy)} facts)")
    print(f"{'='*60}")

    if easy:
        stage1 = SwiGLUAdapter(d_model, d_inner)
        mx.eval(stage1.parameters())
        stage1 = train_adapter(model, tokenizer, stage1, easy, pass_facts,
                              frozen_adapters=frozen_adapters,
                              steps=2000, lr=5e-5, label="Stage1-easy")

        all_adapters = frozen_adapters + [stage1]
        print("\n  Stage 1 eval (all facts):")
        _, n1 = eval_all(model, tokenizer, all_facts, all_adapters, "Stage1")

        # Check no regressions on pass_facts
        regressions = []
        for f in pass_facts:
            m = get_margin(model, tokenizer, f, all_adapters)
            if m <= 0:
                regressions.append(f["id"])
        print(f"  Regressions on baseline-passing: {len(regressions)} ({regressions})")

        frozen_adapters.append(stage1)
        all_trained_adapters.append(("stage1_easy", stage1))
    else:
        print("  No easy facts, skipping")

    # ==================================================================
    # STAGE 2: Medium facts (anchored on easy + pass_facts)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 2: Medium facts ({len(medium)} facts)")
    print(f"{'='*60}")

    if medium:
        anchor_s2 = pass_facts + easy  # anchor on everything learned so far
        stage2 = SwiGLUAdapter(d_model, d_inner)
        mx.eval(stage2.parameters())
        stage2 = train_adapter(model, tokenizer, stage2, medium, anchor_s2,
                              frozen_adapters=frozen_adapters,
                              steps=3000, lr=5e-5, label="Stage2-medium")

        all_adapters = frozen_adapters + [stage2]
        print("\n  Stage 2 eval (all facts):")
        _, n2 = eval_all(model, tokenizer, all_facts, all_adapters, "Stage2")

        regressions = []
        for f in pass_facts + easy:
            m = get_margin(model, tokenizer, f, all_adapters)
            if m <= 0 and baseline.get(f["id"], 0) > 0:
                regressions.append(f["id"])
        print(f"  Regressions: {len(regressions)} ({regressions})")

        frozen_adapters.append(stage2)
        all_trained_adapters.append(("stage2_medium", stage2))
    else:
        print("  No medium facts, skipping")

    # ==================================================================
    # STAGE 3: Hard facts (anchored on everything)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 3: Hard facts ({len(hard)} facts)")
    print(f"{'='*60}")

    if hard:
        anchor_s3 = pass_facts + easy + medium
        stage3 = SwiGLUAdapter(d_model, d_inner)
        mx.eval(stage3.parameters())
        stage3 = train_adapter(model, tokenizer, stage3, hard, anchor_s3,
                              frozen_adapters=frozen_adapters,
                              steps=4000, lr=5e-5, label="Stage3-hard")

        all_adapters = frozen_adapters + [stage3]
        print("\n  Stage 3 eval (all facts):")
        _, n3 = eval_all(model, tokenizer, all_facts, all_adapters, "Stage3")

        regressions = []
        for f in pass_facts + easy + medium:
            m = get_margin(model, tokenizer, f, all_adapters)
            if m <= 0 and baseline.get(f["id"], 0) > 0:
                regressions.append(f["id"])
        print(f"  Regressions: {len(regressions)} ({regressions})")

        frozen_adapters.append(stage3)
        all_trained_adapters.append(("stage3_hard", stage3))
    else:
        print("  No hard facts, skipping")

    # ==================================================================
    # STAGE 4: Orthogonal adapters by topic for remaining failures
    # ==================================================================
    print(f"\n{'='*60}")
    print("STAGE 4: Orthogonal adapters by topic")
    print(f"{'='*60}")

    # Check what's still failing
    still_failing = []
    for fact in all_facts:
        m = get_margin(model, tokenizer, fact, frozen_adapters)
        if m <= 0:
            still_failing.append(fact)

    print(f"Still failing after stages 1-3: {len(still_failing)}")

    if still_failing:
        # Group by topic
        topic_groups = {}
        for f in still_failing:
            t = f.get("topic", "unknown")
            topic_groups.setdefault(t, []).append(f)

        print("By topic:")
        for t, facts in topic_groups.items():
            print(f"  {t}: {len(facts)} facts")

        # Train one orthogonal adapter per topic
        topic_adapters = {}
        for topic, topic_facts in topic_groups.items():
            if len(topic_facts) == 0:
                continue
            print(f"\n  Training orthogonal adapter for {topic} ({len(topic_facts)} facts)...")
            # Anchor on everything that currently passes
            currently_passing = [f for f in all_facts if get_margin(model, tokenizer, f, frozen_adapters) > 0]

            ortho = SwiGLUAdapter(d_model, d_inner)
            mx.eval(ortho.parameters())
            ortho = train_adapter(model, tokenizer, ortho, topic_facts, currently_passing,
                                frozen_adapters=frozen_adapters,
                                steps=3000, lr=5e-5, label=f"Ortho-{topic}")

            # Eval this topic adapter
            n_fixed = 0
            for f in topic_facts:
                adapters_with_ortho = frozen_adapters + [ortho]
                m = get_margin(model, tokenizer, f, adapters_with_ortho)
                if m > 0:
                    n_fixed += 1
                print(f"    {f['id']}: {m:.2f} {'PASS' if m > 0 else 'FAIL'}")
            print(f"  {topic}: {n_fixed}/{len(topic_facts)} fixed")

            topic_adapters[topic] = ortho
            all_trained_adapters.append((f"ortho_{topic}", ortho))

    # ==================================================================
    # FINAL: Route each fact to best adapter combo
    # ==================================================================
    print(f"\n{'='*60}")
    print("FINAL EVALUATION: Best routing per fact")
    print(f"{'='*60}")

    total_correct = 0
    for fact in all_facts:
        # Try: no adapter, staged adapters, staged + topic orthogonal
        m_base = baseline[fact["id"]]
        m_staged = get_margin(model, tokenizer, fact, frozen_adapters)

        best_m = max(m_base, m_staged)
        best_label = "baseline" if m_base >= m_staged else "staged"

        # Try each topic orthogonal adapter
        if still_failing:
            for topic, ortho in topic_adapters.items():
                m_ortho = get_margin(model, tokenizer, fact, frozen_adapters + [ortho])
                if m_ortho > best_m:
                    best_m = m_ortho
                    best_label = f"ortho_{topic}"

        status = "PASS" if best_m > 0 else "FAIL"
        if best_m > 0:
            total_correct += 1
        print(f"  {fact['id']}: {best_m:.2f} {status} (via {best_label})")

    print(f"\nFINAL: {total_correct}/{len(all_facts)} with routing")
    print(f"Baseline was: {len(pass_facts)}/{len(all_facts)}")
    print(f"Improvement: +{total_correct - len(pass_facts)} facts")

    # Save adapters
    adapters_dir = project_root / "adapters"
    for name, adapter in all_trained_adapters:
        weights = {}
        for k, v in adapter.parameters().items():
            weights[k] = np.array(v)
        np.savez(adapters_dir / f"ideology_{name}.npz", **weights)
        print(f"Saved: ideology_{name}.npz")


if __name__ == "__main__":
    main()
