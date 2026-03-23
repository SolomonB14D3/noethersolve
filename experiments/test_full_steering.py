#!/usr/bin/env python3
"""Test full steering pipeline: route → steer → generate."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_lm import load, generate
import mlx.core as mx
from noethersolve.steering_router import SteeringRouter


def generate_with_steering(model, tokenizer, prompt, steering_result, layer=15, alpha=1.0, max_tokens=50):
    """Generate text with steering applied ONLY to prompt encoding (not during decode)."""

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Get steering vector
    steering = steering_result.steering_vector if steering_result else None
    steering_tensor = None
    if steering is not None:
        steering_tensor = mx.array(steering.astype(np.float32) * alpha).reshape(1, 1, -1)

    # Forward pass with steering on PROMPT ONLY
    hidden = model.model.embed_tokens(input_ids)

    for i, layer_module in enumerate(model.model.layers):
        hidden = layer_module(hidden, mask=None, cache=None)

        # Apply steering at target layer (only on prompt)
        if steering_tensor is not None and i == layer:
            hidden = hidden + steering_tensor

    # Get logits for next token
    hidden = model.model.norm(hidden)
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(hidden)
    else:
        logits = model.lm_head(hidden)

    # Simple greedy decode (NO steering during generation)
    generated = []
    for _ in range(max_tokens):
        next_token = mx.argmax(logits[0, -1, :]).item()
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)

        # Continue generation WITHOUT steering
        next_emb = model.model.embed_tokens(mx.array([[next_token]]))
        for layer_module in model.model.layers:
            next_emb = layer_module(next_emb, mask=None, cache=None)

        next_emb = model.model.norm(next_emb)
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(next_emb)
        else:
            logits = model.lm_head(next_emb)

    return tokenizer.decode(generated)


def main():
    print("Loading model...")
    model, tokenizer = load("Qwen/Qwen3-4B-Base")

    print("Loading steering bank...")
    router = SteeringRouter.load("steering_bank.npz")
    print(f"  {len(router)} domains")

    # Test prompts where model often fails
    test_cases = [
        # 3body - kinetic energy is NOT conserved
        {
            "prompt": "In a gravitational three-body system, is kinetic energy conserved? Answer:",
            "correct": "No, kinetic energy is not conserved",
            "wrong": "Yes, kinetic energy is conserved",
        },
        # 2D turbulence - inverse cascade
        {
            "prompt": "In 2D turbulence, energy cascades in which direction? Answer:",
            "correct": "inverse (to larger scales)",
            "wrong": "forward (to smaller scales)",
        },
        # Riemann hypothesis - NOT proven
        {
            "prompt": "Has the Riemann hypothesis been proven? Answer:",
            "correct": "No, it remains unproven",
            "wrong": "Yes, it has been proven",
        },
    ]

    for case in test_cases:
        prompt = case["prompt"]
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Correct: {case['correct']}")
        print(f"Wrong: {case['wrong']}")

        # Route
        result = router.route_and_steer(model, tokenizer, prompt, threshold=0.10)

        if result:
            domains = ", ".join(f"{d}({w:.0%})" for d, w in result.domain_weights.items())
            print(f"\nRouted to: {domains}")

            # Generate without steering
            print("\nWithout steering:")
            base_output = generate(model, tokenizer, prompt=prompt, max_tokens=30)
            print(f"  {base_output}")

            # Generate with steering (different strengths)
            for alpha in [0.5, 1.0, 2.0]:
                print(f"\nWith steering (α={alpha}):")
                steered_output = generate_with_steering(
                    model, tokenizer, prompt, result, layer=router.layer, alpha=alpha, max_tokens=30
                )
                print(f"  {steered_output}")
        else:
            print("\nNo relevant domain found")
            base_output = generate(model, tokenizer, prompt=prompt, max_tokens=30)
            print(f"Output: {base_output}")


if __name__ == "__main__":
    main()
