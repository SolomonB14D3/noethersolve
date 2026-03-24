#!/usr/bin/env python3
"""
Full ideology experiment with FIXED gradient flow.
One adapter, all 31 facts, 5 random splits for SwiGLU vs Linear.
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


class LinearAdapter(nn.Module):
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.down = nn.Linear(d_model, d_inner, bias=False)
        self.up = nn.Linear(d_inner, d_model, bias=False)

    def __call__(self, h):
        return self.up(self.down(h))


def precompute_hidden_states(model, tokenizer, facts):
    """Precompute and cache hidden states for all facts."""
    all_data = []
    for fact in facts:
        ctx = fact["context"]
        entries = []
        # Truth
        tokens = tokenizer.encode(f"{ctx}: {fact['truth']}")
        x = mx.array([tokens[:-1]])
        h = model.model(x)
        if hasattr(model.model, 'norm'):
            h = model.model.norm(h)
        h = mx.stop_gradient(h)
        entries.append(("truth", h, tokens))
        # Distractors
        for d in fact["distractors"]:
            dtokens = tokenizer.encode(f"{ctx}: {d}")
            dx = mx.array([dtokens[:-1]])
            dh = model.model(dx)
            if hasattr(model.model, 'norm'):
                dh = model.model.norm(dh)
            dh = mx.stop_gradient(dh)
            entries.append(("dist", dh, dtokens))
        all_data.append(entries)
    # Force eval
    mx.eval(*[e[1] for fd in all_data for e in fd])
    return all_data


def get_margin(adapter, embed_weight, entries):
    """Compute margin from precomputed hidden states."""
    _, h_truth, tokens_truth = entries[0]
    h_adapted = h_truth + adapter(h_truth)
    logits = (h_adapted @ embed_weight.T).astype(mx.float32)
    lp = nn.log_softmax(logits, axis=-1)
    targets = mx.array([tokens_truth[1:]])
    truth_lp = float(mx.sum(mx.take_along_axis(lp[0], targets[0][:, None], axis=-1).squeeze(-1)))

    best_dist = -float('inf')
    for _, h_dist, tokens_dist in entries[1:]:
        h_d = h_dist + adapter(h_dist)
        d_logits = (h_d @ embed_weight.T).astype(mx.float32)
        d_lp = nn.log_softmax(d_logits, axis=-1)
        d_targets = mx.array([tokens_dist[1:]])
        dlp = float(mx.sum(mx.take_along_axis(d_lp[0], d_targets[0][:, None], axis=-1).squeeze(-1)))
        best_dist = max(best_dist, dlp)
    return truth_lp - best_dist


def train_adapter(adapter, embed_weight, train_data, anchor_data, steps=500, lr=5e-4):
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(adapter, batch, anchors):
        total_loss = mx.array(0.0)
        for entries in batch:
            _, h_truth, tokens_truth = entries[0]
            h_adapted = h_truth + adapter(h_truth)
            logits = (h_adapted @ embed_weight.T).astype(mx.float32)
            lp = nn.log_softmax(logits, axis=-1)
            targets = mx.array([tokens_truth[1:]])
            truth_lp = mx.sum(mx.take_along_axis(lp[0], targets[0][:, None], axis=-1).squeeze(-1))

            best_dist_lp = mx.array(-1e9)
            for _, h_dist, tokens_dist in entries[1:]:
                h_d = h_dist + adapter(h_dist)
                d_logits = (h_d @ embed_weight.T).astype(mx.float32)
                d_lp = nn.log_softmax(d_logits, axis=-1)
                d_targets = mx.array([tokens_dist[1:]])
                dist_lp = mx.sum(mx.take_along_axis(d_lp[0], d_targets[0][:, None], axis=-1).squeeze(-1))
                best_dist_lp = mx.maximum(best_dist_lp, dist_lp)

            margin = truth_lp - best_dist_lp
            total_loss = total_loss + mx.maximum(mx.array(0.0), mx.array(1.5) - margin)

        # Anchor loss
        for entries in anchors:
            _, h_truth, tokens_truth = entries[0]
            h_adapted = h_truth + adapter(h_truth)
            logits = (h_adapted @ embed_weight.T).astype(mx.float32)
            lp = nn.log_softmax(logits, axis=-1)
            targets = mx.array([tokens_truth[1:]])
            truth_lp = mx.sum(mx.take_along_axis(lp[0], targets[0][:, None], axis=-1).squeeze(-1))

            best_dist_lp = mx.array(-1e9)
            for _, h_dist, tokens_dist in entries[1:]:
                h_d = h_dist + adapter(h_dist)
                d_logits = (h_d @ embed_weight.T).astype(mx.float32)
                d_lp = nn.log_softmax(d_logits, axis=-1)
                d_targets = mx.array([tokens_dist[1:]])
                dist_lp = mx.sum(mx.take_along_axis(d_lp[0], d_targets[0][:, None], axis=-1).squeeze(-1))
                best_dist_lp = mx.maximum(best_dist_lp, dist_lp)

            a_margin = truth_lp - best_dist_lp
            total_loss = total_loss + 2.0 * mx.maximum(mx.array(0.0), mx.array(0.1) - a_margin)

        n = len(batch) + len(anchors)
        return total_loss / max(1, n)

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    for step in range(steps):
        batch = random.sample(train_data, min(5, len(train_data)))
        a_batch = random.sample(anchor_data, min(3, len(anchor_data))) if anchor_data else []

        loss, grads = loss_and_grad(adapter, batch, a_batch)

        flat_grads = tree_flatten(grads)
        grad_norm = sum(float(mx.sum(v * v)) for _, v in flat_grads if isinstance(v, mx.array))
        grad_norm = grad_norm ** 0.5
        if grad_norm > 1.0:
            grads = tree_map(lambda g: g * (1.0/grad_norm) if isinstance(g, mx.array) else g, grads)

        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if (step + 1) % 100 == 0:
            n_train = sum(1 for d in train_data if get_margin(adapter, embed_weight, d) > 0)
            print(f"    Step {step+1}: loss={float(loss):.4f}, grad={grad_norm:.4f}, train={n_train}/{len(train_data)}")

    return adapter


def main():
    project_root = Path(__file__).parent.parent

    print("Loading Qwen3-8B-Base...")
    model, tokenizer = mlx_load("Qwen/Qwen3-8B-Base")
    model.eval()
    d_model = model.model.embed_tokens.weight.shape[1]
    embed_weight = model.model.embed_tokens.weight

    with open(project_root / "problems" / "ideology_facts_frank.json") as f:
        all_facts = json.load(f)

    # Knowledge anchors
    anchor_facts = [
        {"id": "a01", "context": "Geography", "truth": "The capital of France is Paris.", "distractors": ["The capital of France is Lyon.", "The capital of France is Marseille."]},
        {"id": "a02", "context": "Chemistry", "truth": "Water boils at 100 degrees Celsius at standard pressure.", "distractors": ["Water boils at 50 degrees Celsius.", "Water boils at 200 degrees Celsius."]},
        {"id": "a03", "context": "Biology", "truth": "DNA carries genetic information.", "distractors": ["Proteins carry genetic information.", "Lipids carry genetic information."]},
        {"id": "a04", "context": "Physics", "truth": "The speed of light in vacuum is approximately 300,000 km per second.", "distractors": ["The speed of light is 150,000 km per second.", "Light speed varies with color in vacuum."]},
        {"id": "a05", "context": "History", "truth": "World War II ended in 1945.", "distractors": ["World War II ended in 1943.", "World War II ended in 1950."]},
        {"id": "a06", "context": "Mathematics", "truth": "Pi is approximately 3.14159.", "distractors": ["Pi is approximately 3.00000.", "Pi is approximately 2.71828."]},
        {"id": "a07", "context": "Astronomy", "truth": "Earth orbits the Sun.", "distractors": ["The Sun orbits Earth.", "Earth orbits the Moon."]},
        {"id": "a08", "context": "Geography", "truth": "The Amazon River is in South America.", "distractors": ["The Amazon River is in Africa.", "The Amazon River is in Asia."]},
        {"id": "a09", "context": "Literature", "truth": "Shakespeare wrote Hamlet.", "distractors": ["Dickens wrote Hamlet.", "Chaucer wrote Hamlet."]},
        {"id": "a10", "context": "Biology", "truth": "Humans have 23 pairs of chromosomes.", "distractors": ["Humans have 20 pairs.", "Humans have 30 pairs."]}
    ]

    print("Precomputing hidden states for all facts...")
    all_data = precompute_hidden_states(model, tokenizer, all_facts)
    anchor_data = precompute_hidden_states(model, tokenizer, anchor_facts)

    # Baseline
    print("\n=== BASELINE ===")
    baselines = {}
    for i, fact in enumerate(all_facts):
        m = get_margin(SwiGLUAdapter(d_model, 64), embed_weight, all_data[i])
        # Actually need untrained adapter = identity-ish. Just compute without adapter.
        _, h_truth, tokens_truth = all_data[i][0]
        logits = (h_truth @ embed_weight.T).astype(mx.float32)
        lp = nn.log_softmax(logits, axis=-1)
        targets = mx.array([tokens_truth[1:]])
        truth_lp = float(mx.sum(mx.take_along_axis(lp[0], targets[0][:, None], axis=-1).squeeze(-1)))
        best_dist = -float('inf')
        for _, h_d, tok_d in all_data[i][1:]:
            d_logits = (h_d @ embed_weight.T).astype(mx.float32)
            d_lp = nn.log_softmax(d_logits, axis=-1)
            d_targets = mx.array([tok_d[1:]])
            best_dist = max(best_dist, float(mx.sum(mx.take_along_axis(d_lp[0], d_targets[0][:, None], axis=-1).squeeze(-1))))
        baselines[fact["id"]] = truth_lp - best_dist
        print(f"  {fact['id']}: {baselines[fact['id']]:.2f} {'PASS' if baselines[fact['id']] > 0 else 'FAIL'}")

    n_base_pass = sum(1 for m in baselines.values() if m > 0)
    print(f"Baseline: {n_base_pass}/31")

    # Anchor baselines
    anchor_baselines = {}
    for i, fact in enumerate(anchor_facts):
        _, h_truth, tokens_truth = anchor_data[i][0]
        logits = (h_truth @ embed_weight.T).astype(mx.float32)
        lp = nn.log_softmax(logits, axis=-1)
        targets = mx.array([tokens_truth[1:]])
        truth_lp = float(mx.sum(mx.take_along_axis(lp[0], targets[0][:, None], axis=-1).squeeze(-1)))
        best_dist = -float('inf')
        for _, h_d, tok_d in anchor_data[i][1:]:
            d_logits = (h_d @ embed_weight.T).astype(mx.float32)
            d_lp = nn.log_softmax(d_logits, axis=-1)
            d_targets = mx.array([tok_d[1:]])
            best_dist = max(best_dist, float(mx.sum(mx.take_along_axis(d_lp[0], d_targets[0][:, None], axis=-1).squeeze(-1))))
        anchor_baselines[fact["id"]] = truth_lp - best_dist
    n_anchor_pass = sum(1 for m in anchor_baselines.values() if m > 0)
    print(f"Anchors baseline: {n_anchor_pass}/10")

    # Parameter matching
    d_inner_s = 64
    s_params = 3 * d_inner_s * d_model
    d_inner_l = round(s_params / (2 * d_model))
    print(f"\nSwiGLU: d_inner={d_inner_s}, params={s_params}")
    print(f"Linear: d_inner={d_inner_l}, params={2*d_inner_l*d_model}")

    # ==================================================================
    # 5 random splits
    # ==================================================================
    print("\n=== 5 RANDOM SPLITS ===")
    split_results = []

    for split_i in range(5):
        print(f"\n--- Split {split_i+1}/5 ---")
        random.seed(100 + split_i)
        indices = list(range(31))
        random.shuffle(indices)
        train_idx = indices[:15]
        test_idx = indices[15:]

        train_data = [all_data[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]
        train_facts = [all_facts[i] for i in train_idx]
        test_facts = [all_facts[i] for i in test_idx]

        # SwiGLU
        print(f"  SwiGLU training...")
        swiglu = SwiGLUAdapter(d_model, d_inner_s)
        mx.eval(swiglu.parameters())
        swiglu = train_adapter(swiglu, embed_weight, train_data, anchor_data, steps=500, lr=5e-4)

        s_train = sum(1 for d in train_data if get_margin(swiglu, embed_weight, d) > 0)
        s_test = sum(1 for d in test_data if get_margin(swiglu, embed_weight, d) > 0)
        s_reg = sum(1 for i, d in enumerate(anchor_data)
                    if anchor_baselines[anchor_facts[i]["id"]] > 0
                    and get_margin(swiglu, embed_weight, d) <= 0)

        # Linear
        print(f"  Linear training...")
        linear = LinearAdapter(d_model, d_inner_l)
        mx.eval(linear.parameters())
        linear = train_adapter(linear, embed_weight, train_data, anchor_data, steps=500, lr=5e-4)

        l_train = sum(1 for d in train_data if get_margin(linear, embed_weight, d) > 0)
        l_test = sum(1 for d in test_data if get_margin(linear, embed_weight, d) > 0)
        l_reg = sum(1 for i, d in enumerate(anchor_data)
                    if anchor_baselines[anchor_facts[i]["id"]] > 0
                    and get_margin(linear, embed_weight, d) <= 0)

        print(f"  SwiGLU: train={s_train}/15, test={s_test}/16, reg={s_reg}")
        print(f"  Linear: train={l_train}/15, test={l_test}/16, reg={l_reg}")

        split_results.append({
            "split": split_i,
            "swiglu_train": s_train, "swiglu_test": s_test, "swiglu_reg": s_reg,
            "linear_train": l_train, "linear_test": l_test, "linear_reg": l_reg,
        })
        del swiglu, linear

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Split':>6} | {'SwiGLU train':>12} | {'SwiGLU test':>11} | {'Linear train':>12} | {'Linear test':>11} | {'S reg':>5} | {'L reg':>5}")
    print("-" * 80)

    s_rates = []
    l_rates = []
    for r in split_results:
        sr = r["swiglu_test"] / 16
        lr_ = r["linear_test"] / 16
        s_rates.append(sr)
        l_rates.append(lr_)
        print(f"  {r['split']+1:>4} | {r['swiglu_train']:>4}/15 | {r['swiglu_test']:>3}/16 ({sr:.0%}) | {r['linear_train']:>4}/15 | {r['linear_test']:>3}/16 ({lr_:.0%}) | {r['swiglu_reg']:>5} | {r['linear_reg']:>5}")

    print(f"\nSwiGLU held-out: {np.mean(s_rates):.1%} +/- {np.std(s_rates):.1%}")
    print(f"Linear held-out: {np.mean(l_rates):.1%} +/- {np.std(l_rates):.1%}")

    from scipy.stats import fisher_exact
    tot_s = sum(r["swiglu_test"] for r in split_results)
    tot_l = sum(r["linear_test"] for r in split_results)
    table = [[tot_s, 80 - tot_s], [tot_l, 80 - tot_l]]
    _, p2 = fisher_exact(table)
    _, p1 = fisher_exact(table, alternative='greater')
    print(f"Pooled Fisher: SwiGLU {tot_s}/80 vs Linear {tot_l}/80, p={p2:.4f} (2-sided), p={p1:.4f} (1-sided)")

    # Save
    out = {"baselines": baselines, "split_results": split_results,
           "summary": {"swiglu_mean": float(np.mean(s_rates)), "linear_mean": float(np.mean(l_rates)),
                       "fisher_p2": float(p2), "fisher_p1": float(p1)}}
    out_path = project_root / "results" / "frank_v4_fixed_gradient.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
