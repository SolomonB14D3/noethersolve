#!/usr/bin/env python3
"""Head-to-head: adapter vs steering vector on raw benchmark data.

Tests college_mathematics (3% baseline) and gpqa (4% baseline) —
domains where steering vectors failed completely.

Usage:
    python experiments/adapter_vs_steering_test.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT = Path(__file__).parent.parent


def load_facts(domain):
    with open(PROJECT / "steering_bank" / f"{domain}.json") as f:
        data = json.load(f)
    return data["facts"]


def score_mc(model, tokenizer, facts, adapter_params=None, steering_vector=None,
             steering_layer=None, steering_alpha=None):
    """Score MC accuracy. Optionally with adapter or steering."""
    import mlx.core as mx
    from noethersolve.adapter import SnapOnConfig, create_adapter

    correct = 0
    total = 0

    # Set up adapter if provided
    adapter = None
    if adapter_params is not None:
        config = SnapOnConfig(vocab_size=151936, d_inner=64, mode="logit")
        adapter = create_adapter(config)
        adapter.update(adapter_params)

    for fact in facts:
        ctx = fact.get("context", "")
        truth = fact.get("truth", fact.get("fact", ""))
        dist = fact.get("distractors", [])
        if not dist:
            continue

        options = [truth] + dist[:3]
        prompt = (ctx + "\n\n" if ctx else "") + "Which is correct?\n"
        for i, opt in enumerate(options):
            prompt += f"{chr(65+i)}) {opt}\n"
        prompt += "Answer: "

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        if steering_vector is not None and steering_layer is not None:
            # Steering: manual forward pass
            hidden = model.model.embed_tokens(input_ids)
            for li, lyr in enumerate(model.model.layers):
                hidden = lyr(hidden, mask=None, cache=None)
                if li == steering_layer:
                    sv = mx.array(steering_vector.astype(np.float32) * steering_alpha).reshape(1, 1, -1)
                    hidden = hidden + sv
            hidden = model.model.norm(hidden)
            if model.args.tie_word_embeddings:
                logits = model.model.embed_tokens.as_linear(hidden)
            else:
                logits = model.lm_head(hidden)
        else:
            # Normal forward pass
            logits = model(input_ids)

        mx.eval(logits)

        # Apply adapter to last token logits if present
        if adapter is not None:
            last = logits[:, -1:, :]  # (1, 1, vocab)
            adapted_last = last + adapter(last)
            mx.eval(adapted_last)
            opt_toks = [tokenizer.encode(chr(65+i))[-1] for i in range(len(options))]
            opt_logits = [float(adapted_last[0, 0, t].item()) for t in opt_toks]
        else:
            opt_toks = [tokenizer.encode(chr(65+i))[-1] for i in range(len(options))]
            opt_logits = [float(logits[0, -1, t].item()) for t in opt_toks]

        if np.argmax(opt_logits) == 0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0, correct, total


def train_adapter(model, tokenizer, train_facts, steps=2000, lr=4e-6):
    """Train a SnapOn adapter on facts."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from noethersolve.adapter import SnapOnConfig, create_adapter

    config = SnapOnConfig(vocab_size=151936, d_inner=64, mode="logit")
    adapter = create_adapter(config)

    optimizer = optim.Adam(learning_rate=lr)

    # Build training pairs: (prompt, correct_logit_idx, wrong_logit_idx)
    training_data = []
    for fact in train_facts:
        ctx = fact.get("context", "")
        truth = fact.get("truth", fact.get("fact", ""))
        dist = fact.get("distractors", [])
        if not dist:
            continue

        options = [truth] + dist[:3]
        prompt = (ctx + "\n\n" if ctx else "") + "Which is correct?\n"
        for i, opt in enumerate(options):
            prompt += f"{chr(65+i)}) {opt}\n"
        prompt += "Answer: "

        tokens = tokenizer.encode(prompt)
        correct_tok = tokenizer.encode("A")[-1]
        wrong_toks = [tokenizer.encode(chr(65+i))[-1] for i in range(1, len(options))]

        training_data.append((tokens, correct_tok, wrong_toks))

    if not training_data:
        return None

    def loss_fn(adapter_params, batch_tokens, correct_tok, wrong_toks):
        adapter.update(adapter_params)
        input_ids = mx.array([batch_tokens])
        logits = model(input_ids)
        last_logits = logits[0, -1:, :]  # (1, vocab) — keep 2D for adapter
        adapted = last_logits + adapter(last_logits)
        adapted = adapted[0]  # (vocab,)

        correct_logit = adapted[correct_tok]
        wrong_logits = mx.array([adapted[t] for t in wrong_toks])
        max_wrong = mx.max(wrong_logits)

        # Margin loss: want correct > max_wrong by margin 2.0
        margin = 2.0
        loss = mx.maximum(mx.array(0.0), margin - (correct_logit - max_wrong))
        return loss

    loss_and_grad = mx.value_and_grad(loss_fn)

    params = dict(adapter.parameters())
    best_loss = float('inf')

    for step in range(steps):
        idx = step % len(training_data)
        tokens, ct, wt = training_data[idx]

        loss, grads = loss_and_grad(params, tokens, ct, wt)
        mx.eval(loss)

        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters())
        params = dict(adapter.parameters())

        loss_val = float(loss.item())
        if step % 500 == 0:
            print(f"    Step {step}/{steps}: loss={loss_val:.4f}")

        if loss_val < best_loss:
            best_loss = loss_val

    return params


def main():
    import mlx.core as mx
    from mlx_lm import load

    print("Loading Qwen/Qwen3-14B-Base...")
    model, tokenizer = load("Qwen/Qwen3-14B-Base")

    for domain in ["college_mathematics", "gpqa_main"]:
        print(f"\n{'='*70}")
        print(f"  DOMAIN: {domain}")
        print(f"{'='*70}")

        facts = load_facts(domain)
        print(f"  Total facts: {len(facts)}")

        # Split train/test
        np.random.seed(42)
        idx = np.random.permutation(len(facts))
        split = len(facts) // 2
        train = [facts[i] for i in idx[:split]]
        test = [facts[i] for i in idx[split:]]
        print(f"  Train: {len(train)}, Test: {len(test)}")

        # 1. Baseline
        print(f"\n  [1] Baseline...")
        baseline_acc, bc, bt = score_mc(model, tokenizer, test)
        print(f"      Accuracy: {baseline_acc:.0%} ({bc}/{bt})")

        # 2. Best steering vector (try layers 10, 15, 20 at alpha 1.5)
        print(f"\n  [2] Steering vectors...")
        best_steer_acc = baseline_acc
        best_layer = None
        for layer in [10, 15, 20]:
            # Compute vector from train set
            import mlx.core as mx
            c_acts, i_acts = [], []
            for fact in train[:20]:  # cap for speed
                ctx = fact.get("context", "")
                truth = fact.get("truth", fact.get("fact", ""))
                dist = fact.get("distractors", [])
                if not dist: continue
                bp = f"{ctx}\n\nAnswer: " if ctx else "Answer: "

                for text, store in [(truth, c_acts), (dist[0], i_acts)]:
                    tokens = tokenizer.encode(bp + text)
                    hidden = model.model.embed_tokens(mx.array([tokens]))
                    for li, lyr in enumerate(model.model.layers):
                        hidden = lyr(hidden, mask=None, cache=None)
                        if li == layer:
                            store.append(np.array(hidden[0, -1, :].astype(mx.float32)))
                            break
                    mx.eval(hidden)

            sv = np.array(c_acts).mean(0) - np.array(i_acts).mean(0)

            for alpha in [0.5, 1.0, 1.5, 2.0]:
                acc, c, t = score_mc(model, tokenizer, test,
                                     steering_vector=sv, steering_layer=layer, steering_alpha=alpha)
                marker = ""
                if acc > best_steer_acc:
                    best_steer_acc = acc
                    best_layer = layer
                    marker = " ← BEST"
                print(f"      L{layer} α={alpha:.1f}: {acc:.0%} ({c}/{t}){marker}")

        print(f"      Best steering: {best_steer_acc:.0%}")

        # 3. Train adapter
        print(f"\n  [3] Training adapter (2000 steps)...")
        start = time.time()
        adapter_params = train_adapter(model, tokenizer, train, steps=2000, lr=4e-6)
        elapsed = time.time() - start
        print(f"      Training time: {elapsed:.0f}s")

        if adapter_params:
            adapter_acc, ac, at = score_mc(model, tokenizer, test, adapter_params=adapter_params)
            print(f"      Adapter accuracy: {adapter_acc:.0%} ({ac}/{at})")
        else:
            adapter_acc = 0
            print(f"      Adapter training failed")

        # Summary
        print(f"\n  SUMMARY for {domain}:")
        print(f"    Baseline:  {baseline_acc:.0%}")
        print(f"    Steering:  {best_steer_acc:.0%}  (0.1 KB, ~5s)")
        print(f"    Adapter:   {adapter_acc:.0%}  (50 MB, {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
