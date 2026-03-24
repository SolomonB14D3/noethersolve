#!/usr/bin/env python3
"""
Probe where the gradient signal is strongest for ideology facts.
For each layer, measure the gradient magnitude of the hidden state
with respect to the margin loss. This tells us WHERE in the model
the routing decision is being made.

If gradients are strongest at layers 10-18 (Frank's ablation window),
that's where we need to put the adapter.
"""

import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load


def probe_gradients(model, tokenizer, fact):
    """
    Run forward pass capturing hidden states at each layer,
    compute loss, then check gradient magnitude at each layer.
    """
    ctx = fact["context"]
    truth_text = f"{ctx}: {fact['truth']}"
    tokens = tokenizer.encode(truth_text)
    x = mx.array([tokens[:-1]])

    # We need to manually run through the model layer by layer
    # to capture intermediate hidden states

    # Get the embedding
    h = model.model.embed_tokens(x)

    # Run through each layer, storing hidden states
    layer_outputs = []
    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask=None)
        layer_outputs.append(h)

    # Final norm
    if hasattr(model.model, 'norm'):
        h = model.model.norm(h)

    # Get logits
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(h)
    else:
        logits = model.model.embed_tokens.as_linear(h)

    logits = logits.astype(mx.float32)
    log_probs = nn.log_softmax(logits, axis=-1)
    targets = mx.array([tokens[1:]])
    truth_lp = mx.sum(mx.take_along_axis(log_probs[0], targets[0][:, None], axis=-1).squeeze(-1))

    return truth_lp, layer_outputs


def main():
    project_root = Path(__file__).parent.parent

    print("Loading Qwen3-8B-Base...")
    model, tokenizer = mlx_load("Qwen/Qwen3-8B-Base")
    model.eval()

    n_layers = len(model.model.layers)
    d_model = model.model.embed_tokens.weight.shape[1]
    print(f"Layers: {n_layers}, d_model: {d_model}")

    with open(project_root / "problems" / "ideology_facts_frank.json") as f:
        all_facts = json.load(f)

    # Test on 3 facts: easy, medium, hard
    test_ids = ["ideo_xinj_L4", "ideo_tibet_L2", "ideo_tian_L4"]
    test_facts = [f for f in all_facts if f["id"] in test_ids]

    # Also test a passing fact for comparison
    pass_ids = ["ideo_tian_L1"]
    pass_facts = [f for f in all_facts if f["id"] in pass_ids]

    all_test = test_facts + pass_facts

    for fact in all_test:
        print(f"\n{'='*60}")
        print(f"Fact: {fact['id']} ({fact['context']})")
        print(f"Truth: {fact['truth'][:80]}...")
        print(f"{'='*60}")

        ctx = fact["context"]
        truth_text = f"{ctx}: {fact['truth']}"
        tokens = tokenizer.encode(truth_text)
        x = mx.array([tokens[:-1]])

        # Method: Insert a tiny perturbation at each layer and measure
        # how much the truth log-prob changes. This is a finite-difference
        # approximation of the gradient magnitude at each layer.

        # First get baseline truth logprob
        baseline_logits = model(x).astype(mx.float32)
        baseline_lp = nn.log_softmax(baseline_logits, axis=-1)
        targets = mx.array([tokens[1:]])
        baseline_truth_lp = float(mx.sum(mx.take_along_axis(baseline_lp[0], targets[0][:, None], axis=-1).squeeze(-1)))

        # Also get best distractor logprob
        best_dist_lp = -float('inf')
        for d in fact["distractors"]:
            d_text = f"{ctx}: {d}"
            d_tokens = tokenizer.encode(d_text)
            d_x = mx.array([d_tokens[:-1]])
            d_logits = model(d_x).astype(mx.float32)
            d_lps = nn.log_softmax(d_logits, axis=-1)
            d_targets = mx.array([d_tokens[1:]])
            dlp = float(mx.sum(mx.take_along_axis(d_lps[0], d_targets[0][:, None], axis=-1).squeeze(-1)))
            best_dist_lp = max(best_dist_lp, dlp)

        margin = baseline_truth_lp - best_dist_lp
        print(f"Baseline margin: {margin:.2f}")

        # Now probe each layer with a random perturbation
        epsilon = 0.1
        layer_sensitivities = []

        for layer_idx in range(n_layers):
            # Hook: add perturbation at this layer
            # We'll do this by running the model manually

            h = model.model.embed_tokens(x)

            for i, layer in enumerate(model.model.layers):
                h = layer(h, mask=None)
                if i == layer_idx:
                    # Add random perturbation
                    rng = np.random.RandomState(42)
                    perturb = mx.array(rng.randn(*h.shape).astype(np.float32)) * epsilon
                    h_perturbed = h + perturb

                    # Continue with perturbed
                    h = h_perturbed

            if hasattr(model.model, 'norm'):
                h = model.model.norm(h)
            if hasattr(model, 'lm_head'):
                logits = model.lm_head(h)
            else:
                logits = model.model.embed_tokens.as_linear(h)

            logits = logits.astype(mx.float32)
            lp = nn.log_softmax(logits, axis=-1)
            perturbed_truth_lp = float(mx.sum(mx.take_along_axis(lp[0], targets[0][:, None], axis=-1).squeeze(-1)))

            sensitivity = abs(perturbed_truth_lp - baseline_truth_lp) / epsilon
            layer_sensitivities.append(sensitivity)

            mx.eval(logits)  # force eval to free memory

        # Print results
        print(f"\nLayer sensitivity (|delta_lp| / epsilon):")
        max_s = max(layer_sensitivities)
        for i, s in enumerate(layer_sensitivities):
            bar = '#' * int(40 * s / max_s) if max_s > 0 else ''
            marker = " <-- PEAK" if s == max_s else ""
            print(f"  L{i:2d}: {s:8.2f} {bar}{marker}")

        # Find top 5 layers
        sorted_layers = sorted(range(n_layers), key=lambda i: layer_sensitivities[i], reverse=True)
        print(f"\nTop 5 sensitive layers: {sorted_layers[:5]}")
        print(f"Sensitivity range: {min(layer_sensitivities):.2f} - {max(layer_sensitivities):.2f}")

        # Frank's window (40-65% of depth)
        frank_start = int(n_layers * 0.4)
        frank_end = int(n_layers * 0.65)
        frank_mean = np.mean(layer_sensitivities[frank_start:frank_end])
        overall_mean = np.mean(layer_sensitivities)
        print(f"\nFrank's window (L{frank_start}-L{frank_end}): mean sensitivity = {frank_mean:.2f}")
        print(f"Overall mean: {overall_mean:.2f}")
        print(f"Frank window / overall: {frank_mean/overall_mean:.2f}x")


if __name__ == "__main__":
    main()
