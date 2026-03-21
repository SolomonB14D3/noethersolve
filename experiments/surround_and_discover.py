#!/usr/bin/env python3
"""
Surround and Discover — Train an adapter on facts AROUND a truth,
then test if the adapter discovers the truth it was never told.

Like finding the center of a circle from points on the circumference.
Like solving a Rubik's cube — each surrounding fact is a twist that
constrains the representation until the unknown face is forced into place.

Usage:
    python experiments/surround_and_discover.py
"""

import json
import os
import sys
import time
import subprocess

if not os.environ.get("HF_HOME") and os.path.isdir("/Volumes/4TB SD/ml_cache/huggingface"):
    os.environ["HF_HOME"] = "/Volumes/4TB SD/ml_cache/huggingface"

import mlx.core as mx
import mlx_lm
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from noethersolve.oracle import score_fact_mc


# The target claim — NOT in training data
TARGET_CLAIM = {
    "id": "target_enstrophy_mechanism",
    "context": "In 3D Navier-Stokes, enstrophy (integral of |omega|^2) can grow without bound. The dominant mechanism for enstrophy growth is:",
    "candidates": [
        "vortex stretching — the omega dot grad(u) term amplifies vorticity along the strain direction",
        "energy cascade — nonlinear transfer concentrates energy at small scales where enstrophy is measured",
        "boundary layer instability — viscous layers generate high vorticity gradients that feed the bulk flow",
        "pressure-strain correlation — pressure forces alignment between vorticity and rate-of-strain eigenvectors"
    ],
    "ground_truth_idx": 0
}


def load_model():
    print("Loading Qwen3-4B-Base...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def load_adapter(model, adapter_path, d_inner=64):
    from noethersolve.adapter import SnapOnConfig, create_adapter
    from noethersolve.train_utils import get_lm_head_fn

    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                      n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    weights = mx.load(adapter_path)
    adapter.load_weights(list(weights.items()))
    mx.eval(adapter.parameters())
    lm_head = get_lm_head_fn(model)
    return adapter, lm_head


def score_claim(model, tokenizer, claim, adapter=None, lm_head=None):
    """Score all candidates for a claim."""
    margins = []
    for i, cand in enumerate(claim["candidates"]):
        others = [c for j, c in enumerate(claim["candidates"]) if j != i]
        win, margin, _, _ = score_fact_mc(
            model, tokenizer,
            claim["context"], cand, others,
            adapter=adapter, lm_head=lm_head,
        )
        margins.append(float(margin))
    return margins


def train_surrounding_adapter(facts_path, adapter_out, steps=3000, lr=4e-6):
    """Train an adapter on the surrounding facts using the existing training infrastructure."""
    # Use the adapter training script
    train_script = os.path.join(ROOT, "training", "scripts", "train_generic_adapter.py")

    # If generic trainer doesn't exist, use a minimal inline training loop
    if not os.path.exists(train_script):
        print("  Using inline training loop...")
        return train_inline(facts_path, adapter_out, steps, lr)

    cmd = [
        sys.executable, train_script,
        "--data", facts_path,
        "--out", adapter_out,
        "--steps", str(steps),
        "--lr", str(lr),
    ]
    print(f"  Training: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"  Training failed: {result.stderr[:200]}")
        return False
    return True


def train_inline(facts_path, adapter_out, steps=3000, lr=4e-6):
    """Minimal inline adapter training on surrounding facts."""
    from noethersolve.adapter import SnapOnConfig, create_adapter
    from noethersolve.train_utils import get_lm_head_fn

    # Load facts
    with open(facts_path) as f:
        data = json.load(f)
    facts = data["facts"]

    # Load model
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]

    # Create adapter
    cfg = SnapOnConfig(d_model=d_model, d_inner=64, n_layers=0,
                      n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    lm_head = get_lm_head_fn(model)

    # Build training pairs: (context + truth, context + distractor)
    pairs = []
    for fact in facts:
        truth_text = fact["context"] + " " + fact["truth"]
        for dist in fact["distractors"]:
            dist_text = fact["context"] + " " + dist
            pairs.append((truth_text, dist_text))

    print(f"  Training on {len(facts)} facts ({len(pairs)} contrastive pairs)")
    print(f"  Steps: {steps}, lr: {lr}")

    # Simple contrastive training loop
    import mlx.nn as nn
    import mlx.optimizers as optim

    optimizer = optim.Adam(learning_rate=lr)

    def get_hidden(text):
        tokens = tokenizer.encode(text, return_tensors=None)
        if isinstance(tokens, list):
            input_ids = mx.array([tokens])
        else:
            input_ids = mx.array([tokens])
        out = model(input_ids)
        # Get last hidden state from the model's internal layers
        hidden = model.model(input_ids)
        # Use mean of last layer hidden states
        if hasattr(hidden, 'last_hidden_state'):
            h = hidden.last_hidden_state[:, -1, :]
        else:
            h = hidden[:, -1, :]
        return h

    def loss_fn(adapter_params, truth_text, dist_text):
        """Contrastive loss: truth should score higher than distractor."""
        h_truth = get_hidden(truth_text)
        h_dist = get_hidden(dist_text)

        # Apply adapter to get logit adjustments
        adapter.update(adapter_params)
        truth_logits = adapter(h_truth)
        dist_logits = adapter(h_dist)

        # Apply lm_head to get token probabilities
        truth_full = lm_head(h_truth) + truth_logits
        dist_full = lm_head(h_dist) + dist_logits

        # Margin loss: truth score should exceed distractor score
        truth_score = mx.logsumexp(truth_full, axis=-1).mean()
        dist_score = mx.logsumexp(dist_full, axis=-1).mean()

        return mx.maximum(0, 1.0 - (truth_score - dist_score))

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    t0 = time.time()
    running_loss = 0.0
    for step in range(steps):
        pair = pairs[step % len(pairs)]
        loss, grads = loss_and_grad(adapter.parameters(), pair[0], pair[1])
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)
        running_loss += loss.item()

        if (step + 1) % 500 == 0:
            avg = running_loss / 500
            elapsed = time.time() - t0
            print(f"    Step {step+1}/{steps}  loss={avg:.4f}  ({elapsed:.0f}s)")
            running_loss = 0.0

    # Save adapter
    os.makedirs(os.path.dirname(adapter_out), exist_ok=True)
    flat = {}
    for k, v in adapter.parameters().items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat[f"{k}.{k2}"] = v2
        else:
            flat[k] = v
    mx.savez(adapter_out, **flat)
    print(f"  Adapter saved to {adapter_out}")
    return True


def main():
    # Paths
    facts_path = os.path.join(HERE, "surrounding_facts_enstrophy.json")
    adapter_out = os.path.join(ROOT, "adapters", "surround_enstrophy_adapter.npz")

    print("="*70)
    print("  SURROUND AND DISCOVER")
    print("  Training on facts AROUND vortex stretching,")
    print("  then testing if the adapter discovers vortex stretching itself")
    print("="*70)

    # Phase 1: Score target with base model
    print("\n--- Phase 1: Base model on target claim ---")
    model, tokenizer = load_model()
    base_margins = score_claim(model, tokenizer, TARGET_CLAIM)
    base_vote = int(np.argmax(base_margins))

    print(f"  Base picks candidate {base_vote}: {TARGET_CLAIM['candidates'][base_vote][:70]}")
    for i, (m, c) in enumerate(zip(base_margins, TARGET_CLAIM["candidates"])):
        gt = " ← GROUND TRUTH" if i == TARGET_CLAIM["ground_truth_idx"] else ""
        pick = " <<<" if i == base_vote else ""
        print(f"    [{i}] {m:+8.2f}  {c[:65]}{pick}{gt}")

    # Phase 2: Score with existing NS adapters (for comparison)
    print("\n--- Phase 2: Existing NS adapters on target claim ---")
    adapter_dir = os.path.join(ROOT, "adapters")
    ns_adapters = [f for f in os.listdir(adapter_dir)
                   if 'ns_regularity' in f and f.endswith('.npz')]

    for afile in ns_adapters[:4]:
        try:
            adpt, lm_head = load_adapter(model, os.path.join(adapter_dir, afile))
            margins = score_claim(model, tokenizer, TARGET_CLAIM,
                                 adapter=adpt, lm_head=lm_head)
            vote = int(np.argmax(margins))
            print(f"  {afile[:50]:50s} → cand {vote} ({margins[vote]:+.1f})")
        except Exception as e:
            print(f"  {afile[:50]:50s} → ERROR")

    # Phase 3: Train surrounding adapter
    print("\n--- Phase 3: Training adapter on SURROUNDING facts ---")
    print(f"  Facts file: {facts_path}")
    print(f"  These facts describe everything AROUND vortex stretching")
    print(f"  but never directly state it as the enstrophy mechanism")

    # Check if adapter already trained
    if os.path.exists(adapter_out):
        print(f"  Adapter already exists at {adapter_out}")
    else:
        # Try using the oracle_wrapper training or scripts/adapter_trainer
        # For now, try to find a working training method
        print("  Looking for training infrastructure...")

        # Check if we can use the training scripts
        train_scripts = [
            os.path.join(ROOT, "training", "scripts", "train_generic_adapter.py"),
            os.path.join(ROOT, "scripts", "adapter_trainer.py"),
        ]

        found_trainer = None
        for ts in train_scripts:
            if os.path.exists(ts):
                found_trainer = ts
                break

        if found_trainer:
            print(f"  Found trainer: {found_trainer}")
            # The adapter_trainer expects a specific format — let's use it
            # For now, let's convert our facts to the training format and use
            # the existing oracle to train

        # Use inline training as fallback
        success = train_inline(facts_path, adapter_out, steps=3000, lr=4e-6)
        if not success:
            print("  Training failed!")
            return

    # Phase 4: Score target with surrounding adapter
    print("\n--- Phase 4: Surrounding adapter on target claim ---")
    try:
        adpt, lm_head = load_adapter(model, adapter_out)
        surr_margins = score_claim(model, tokenizer, TARGET_CLAIM,
                                  adapter=adpt, lm_head=lm_head)
        surr_vote = int(np.argmax(surr_margins))

        print(f"  Surrounding adapter picks candidate {surr_vote}")
        for i, (m, c) in enumerate(zip(surr_margins, TARGET_CLAIM["candidates"])):
            gt = " ← GROUND TRUTH" if i == TARGET_CLAIM["ground_truth_idx"] else ""
            base_m = base_margins[i]
            delta = m - base_m
            print(f"    [{i}] base={base_m:+8.2f} → surr={m:+8.2f} (Δ={delta:+.1f})  {c[:50]}{gt}")

        print(f"\n  BASE vote: candidate {base_vote} ({'CORRECT' if base_vote == 0 else 'WRONG'})")
        print(f"  SURR vote: candidate {surr_vote} ({'CORRECT' if surr_vote == 0 else 'WRONG'})")

        if surr_vote == 0 and base_vote != 0:
            print(f"\n  *** DISCOVERY: Surrounding facts forced the adapter to find vortex stretching! ***")
            print(f"  The adapter was never told the answer — it inferred it from the constraints.")
        elif surr_vote == 0 and base_vote == 0:
            print(f"\n  Both got it right — but check if surrounding adapter is MORE confident")
            delta_gt = surr_margins[0] - base_margins[0]
            print(f"  Confidence delta on ground truth: {delta_gt:+.2f}")
        else:
            print(f"\n  Surrounding adapter picked wrong answer. Constraint ring may be incomplete.")

    except Exception as e:
        print(f"  Error loading surrounding adapter: {e}")
        import traceback
        traceback.print_exc()

    # Phase 5: Also test with vortex domain adapters for comparison
    print("\n--- Phase 5: Vortex adapters for comparison ---")
    vortex_adapters = [f for f in os.listdir(adapter_dir)
                       if ('vortex' in f or 'continuous' in f) and f.endswith('.npz')]
    for afile in vortex_adapters[:4]:
        try:
            adpt, lm_head = load_adapter(model, os.path.join(adapter_dir, afile))
            margins = score_claim(model, tokenizer, TARGET_CLAIM,
                                 adapter=adpt, lm_head=lm_head)
            vote = int(np.argmax(margins))
            correct = "✓" if vote == 0 else "✗"
            print(f"  {correct} {afile[:50]:50s} → cand {vote} ({margins[vote]:+.1f})")
        except Exception as e:
            print(f"  ? {afile[:50]:50s} → ERROR")

    # Save results
    results = {
        "target_claim": TARGET_CLAIM,
        "base_margins": base_margins,
        "base_vote": base_vote,
        "surrounding_margins": surr_margins if 'surr_margins' in dir() else None,
        "surrounding_vote": surr_vote if 'surr_vote' in dir() else None,
        "ground_truth_idx": 0,
    }
    output_path = os.path.join(ROOT, "results", "surround_discover_enstrophy.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


# Need this import at module level for train_inline
import mlx.optimizers

if __name__ == "__main__":
    main()
