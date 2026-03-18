#!/usr/bin/env python3
"""Train 4B adapters for domains that only have 7B adapters.

These domains have adapters trained on Qwen 7B (vocab 152064) which are
incompatible with the 4B model (vocab 151936). This script trains new
4B-compatible adapters.

Usage:
    python experiments/train_missing_adapters.py --list          # Show domains needing training
    python experiments/train_missing_adapters.py --domain biochemistry
    python experiments/train_missing_adapters.py --all           # Train all missing
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

# Domains with 7B adapters that need 4B versions
DOMAINS_NEEDING_TRAINING = {
    "biochemistry": "biochemistry_facts.json",
    "chemistry": "chemistry_facts.json",
    "control_systems": "control_systems_facts.json",
    "cryptography": "cryptography_facts.json",
    "database_internals": "database_internals_facts.json",
    "distributed_systems": "distributed_systems_facts.json",
    "economics_finance": "economics_finance_facts.json",
    "networking": "networking_facts.json",
    "operating_systems": "operating_systems_facts.json",
    "organic_chemistry": "organic_chemistry_facts.json",
    "quantum_computing": "quantum_computing_facts.json",
    # LLM domains
    "llm_hallucination": "llm_hallucination_facts.json",
    "llm_reasoning": "llm_reasoning_facts.json",
    "llm_alignment": "llm_alignment_facts.json",
    "llm_training": "llm_training_facts.json",
    "llm_evaluation": "llm_evaluation_facts.json",
    "llm_context_memory": "llm_context_memory_facts.json",
    # PL domains
    "pl_type_systems": "pl_type_systems_facts.json",
    "pl_memory": "pl_memory_facts.json",
    "pl_concurrency": "pl_concurrency_facts.json",
    "pl_paradigms": "pl_paradigms_facts.json",
    "pl_compilers": "pl_compilers_facts.json",
    "pl_pitfalls": "pl_pitfalls_facts.json",
}


def load_facts(facts_file: Path) -> List[dict]:
    """Load facts from JSON file."""
    with open(facts_file) as f:
        data = json.load(f)

    facts = data.get("facts", data.get("verifications", []))
    normalized = []

    for i, fact in enumerate(facts):
        if isinstance(fact, dict):
            normalized.append({
                "id": fact.get("id", f"fact_{i:02d}"),
                "context": fact.get("context", ""),
                "truth": fact.get("truth", fact.get("fact", "")),
                "distractors": fact.get("distractors", []),
            })
        else:
            normalized.append({
                "id": f"fact_{i:02d}",
                "context": "",
                "truth": str(fact),
                "distractors": [],
            })

    return normalized


def generate_training_data(facts: List[dict]) -> List[dict]:
    """Generate training examples from facts."""
    examples = []

    for fact in facts:
        context = fact.get("context", "")
        truth = fact.get("truth", "")
        distractors = fact.get("distractors", [])

        # Create positive example
        if context:
            text = f"{context}: {truth}"
        else:
            text = truth

        examples.append({
            "text": text,
            "label": 1,  # Correct
        })

        # Create negative examples from distractors
        for distractor in distractors:
            if context:
                neg_text = f"{context}: {distractor}"
            else:
                neg_text = distractor

            examples.append({
                "text": neg_text,
                "label": 0,  # Incorrect
            })

    return examples


def mc_hinge_loss(adapter, lm_head, model, prompt, truth, distractors, tokenizer, margin_target=1.5):
    """Hinge loss with proper log-probability computation."""
    import mlx.core as mx
    from noethersolve import train_utils as t3

    def lp(text):
        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(prompt + text)
        n_prompt = len(prompt_ids)
        if len(full_ids) <= n_prompt:
            return mx.array(-1e9)
        tokens = mx.array(full_ids)[None, :]
        h = model.model(tokens)
        base_logits = lm_head(h)
        shifts = adapter(base_logits)
        shifts = shifts - shifts.mean(axis=-1, keepdims=True)
        logits = base_logits + shifts
        logits = t3.LOGIT_SOFTCAP * mx.tanh(logits / t3.LOGIT_SOFTCAP)
        total = mx.array(0.0)
        for i, tok_id in enumerate(full_ids[n_prompt:]):
            pos = n_prompt - 1 + i
            lv = logits[0, pos]
            lse = mx.log(mx.sum(mx.exp(lv - mx.max(lv))) + 1e-8) + mx.max(lv)
            total = total + lv[tok_id] - lse
        return total

    truth_lp = lp(f" {truth}")
    dist_lps = [lp(f" {d}") for d in distractors]
    best_dist = mx.max(mx.stack(dist_lps)) if dist_lps else truth_lp - margin_target - 1
    loss = mx.maximum(mx.array(0.0), mx.array(margin_target) - (truth_lp - best_dist))
    return loss, float(truth_lp - best_dist)


def clip_grads(grads, max_norm=1.0):
    """Clip gradients by norm."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    leaves = tree_flatten(grads)
    total_sq = sum(float(mx.sum(g ** 2)) for _, g in leaves)
    norm = total_sq ** 0.5
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        leaves = [(k, g * scale) for k, g in leaves]
    return tree_unflatten(leaves)


def train_adapter(
    domain: str,
    facts: List[dict],
    output_path: Path,
    model_name: str = "Qwen/Qwen3-4B-Base",
    steps: int = 1500,
    d_inner: int = 64,
):
    """Train a logit adapter for a domain using proper MC hinge loss."""
    try:
        import mlx.core as mx
        import mlx_lm
    except ImportError:
        print("Error: MLX required. Install with: pip install mlx mlx-lm")
        return False

    from noethersolve.adapter import SnapOnConfig, create_adapter
    from noethersolve.train_utils import get_lm_head_fn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    import numpy as np

    print(f"Loading model {model_name}...")
    model, tokenizer = mlx_lm.load(model_name)
    model.freeze()

    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    lm_head = get_lm_head_fn(model)

    print(f"  vocab_size: {vocab_size}, d_model: {d_model}")

    # Create adapter
    cfg = SnapOnConfig(
        d_model=d_model,
        d_inner=d_inner,
        n_layers=0,
        n_heads=8,
        mode="logit",
        vocab_size=vocab_size,
    )
    adapter = create_adapter(cfg)

    # Prepare training examples
    examples = []
    for fact in facts:
        context = fact.get("context", "")
        truth = fact.get("truth", "")
        distractors = fact.get("distractors", [])
        if truth and distractors:
            prompt = context if context.endswith(":") else context + ":" if context else ""
            examples.append((prompt, truth, distractors))

    print(f"  Training examples: {len(examples)}")

    if not examples:
        print("  Error: No valid training examples with distractors")
        return False

    # Training with MC hinge loss - use very low LR like working adapters
    optimizer = optim.AdamW(learning_rate=5e-7, weight_decay=0.001)
    loss_and_grad = mx.value_and_grad(mc_hinge_loss, argnums=0)

    print(f"  Training for {steps} steps...")
    recent_margins = []

    for step in range(steps):
        prompt, truth, distractors = examples[step % len(examples)]

        (loss_val, margin_val), grads = loss_and_grad(
            adapter, lm_head, model, prompt, truth, distractors, tokenizer, 2.5
        )
        grads = clip_grads(grads)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        recent_margins.append(margin_val)
        if len(recent_margins) > 50:
            recent_margins.pop(0)

        if (step + 1) % 200 == 0:
            avg = np.mean(recent_margins)
            print(f"    Step {step+1}/{steps}, loss: {float(loss_val):.3f}, margin: {margin_val:.2f}, avg: {avg:.2f}")

    # Save adapter
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat = tree_flatten(adapter.parameters())
    np.savez(str(output_path), **{k: np.array(v) for k, v in flat})
    print(f"  Saved to {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List domains needing training")
    parser.add_argument("--domain", help="Train specific domain")
    parser.add_argument("--all", action="store_true", help="Train all missing domains")
    parser.add_argument("--steps", type=int, default=1500, help="Training steps")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    problems_dir = base_dir / "problems"
    adapters_dir = base_dir / "adapters"

    if args.list:
        print("Domains needing 4B adapter training:")
        print()
        for domain, facts_file in sorted(DOMAINS_NEEDING_TRAINING.items()):
            facts_path = problems_dir / facts_file
            exists = "✓" if facts_path.exists() else "✗"
            adapter_path = adapters_dir / f"{domain}_4b_adapter.npz"
            trained = "trained" if adapter_path.exists() else "needed"
            print(f"  {exists} {domain}: {trained}")
        return

    domains_to_train = []

    if args.domain:
        if args.domain in DOMAINS_NEEDING_TRAINING:
            domains_to_train = [args.domain]
        else:
            print(f"Unknown domain: {args.domain}")
            print(f"Available: {', '.join(DOMAINS_NEEDING_TRAINING.keys())}")
            return
    elif args.all:
        domains_to_train = list(DOMAINS_NEEDING_TRAINING.keys())
    else:
        parser.print_help()
        return

    # Train each domain
    for domain in domains_to_train:
        facts_file = DOMAINS_NEEDING_TRAINING[domain]
        facts_path = problems_dir / facts_file

        if not facts_path.exists():
            print(f"Skipping {domain}: {facts_file} not found")
            continue

        output_path = adapters_dir / f"{domain}_4b_adapter.npz"
        if output_path.exists():
            print(f"Skipping {domain}: adapter already exists")
            continue

        print(f"\n{'='*60}")
        print(f"Training {domain}")
        print(f"{'='*60}")

        facts = load_facts(facts_path)
        print(f"  Loaded {len(facts)} facts from {facts_file}")

        success = train_adapter(
            domain=domain,
            facts=facts,
            output_path=output_path,
            steps=args.steps,
        )

        if success:
            print(f"  ✓ {domain} complete")
        else:
            print(f"  ✗ {domain} failed")


if __name__ == "__main__":
    main()
