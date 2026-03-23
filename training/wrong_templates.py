#!/usr/bin/env python3
"""Generate maximally contrastive distractors using templates.

Instead of trying to generate "plausible wrong" answers,
generate "obviously wrong" answers that create maximum embedding contrast.

These distractors are designed for STEERING, not for testing human knowledge.
"""

import random

# Templates for generating wrong answers with high embedding contrast
WRONG_TEMPLATES = [
    # Negation patterns
    "{truth} is completely false",
    "the opposite of {truth}",
    "{truth} has been disproven",

    # Wrong quantity patterns
    "zero in all cases",
    "undefined mathematically",
    "infinite under all conditions",
    "exactly one regardless of parameters",

    # Wrong physics patterns
    "violates conservation laws",
    "only applies in imaginary scenarios",
    "requires negative mass to work",
    "breaks down at all scales",

    # Common misconception patterns
    "depends only on position not velocity",
    "is always positive definite",
    "equals mass times acceleration",
    "follows inverse square law exclusively",

    # Structural opposites
    "sum becomes product",
    "integral becomes derivative",
    "conserved becomes dissipated",
    "symmetric becomes asymmetric",
]

# Domain-specific wrong patterns
DOMAIN_WRONG_PATTERNS = {
    "physics": [
        "energy is created from nothing",
        "momentum can be negative mass",
        "angular momentum is a scalar",
        "force equals mass times position",
    ],
    "chemistry": [
        "atoms are created in reactions",
        "entropy always decreases",
        "reactions are instantaneous",
        "bonds have zero energy",
    ],
    "math": [
        "parallel lines meet at infinity",
        "division by zero is allowed",
        "negative numbers are imaginary",
        "all functions are continuous",
    ],
    "biology": [
        "DNA is single-stranded",
        "proteins fold randomly",
        "enzymes are consumed in reactions",
        "cells divide by fusion",
    ],
}


def generate_contrastive_distractors(
    truth: str,
    domain: str = "general",
    n: int = 3,
) -> list[str]:
    """Generate maximally contrastive distractors.

    These are designed for steering vector training, not human testing.
    They should be obviously wrong but grammatically correct.
    """
    distractors = []

    # 1. Direct negation/opposite
    if "is" in truth.lower():
        distractors.append(truth.replace(" is ", " is not ").replace(" are ", " are not "))
    else:
        distractors.append(f"contrary to common belief, {truth} is incorrect")

    # 2. Use domain-specific patterns
    if domain in DOMAIN_WRONG_PATTERNS:
        patterns = DOMAIN_WRONG_PATTERNS[domain]
        distractors.extend(random.sample(patterns, min(2, len(patterns))))

    # 3. Fill with general wrong templates
    while len(distractors) < n:
        template = random.choice(WRONG_TEMPLATES)
        wrong = template.format(truth=truth[:30])  # Truncate for template
        if wrong not in distractors:
            distractors.append(wrong)

    return distractors[:n]


def generate_incoherent_distractors(
    truth: str,
    n: int = 3,
) -> list[str]:
    """Generate semantically incoherent distractors.

    For maximum steering contrast, use answers that are
    grammatically correct but semantically nonsensical.
    """
    incoherent = [
        "the sound of purple triangles",
        "seventeen divided by happiness",
        "the weight of forgotten memories",
        "speed of slow comprehension",
        "temperature of abstract concepts",
        "the derivative of yesterday",
        "integral over imagination space",
        "the tensor of emotional states",
    ]
    return random.sample(incoherent, min(n, len(incoherent)))


if __name__ == "__main__":
    # Test
    truth = "kinetic energy is conserved in elastic collisions"

    print(f"Truth: {truth}")
    print()

    print("Contrastive distractors (physics):")
    for d in generate_contrastive_distractors(truth, "physics"):
        print(f"  - {d}")

    print()
    print("Incoherent distractors:")
    for d in generate_incoherent_distractors(truth):
        print(f"  - {d}")
