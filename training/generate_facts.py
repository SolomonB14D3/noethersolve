#!/usr/bin/env python3
"""Generate training facts using 27B model.

The 27B is more token-efficient and produces higher quality facts.
Facts are then used to train steering vectors on 4B.

Usage:
    python training/generate_facts.py --domain "enzyme kinetics" --n 20
    python training/generate_facts.py --domain-file domains.txt --n 10
    python training/generate_facts.py --yaml problems/biochemistry.yaml
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


FACT_GENERATION_PROMPT = '''You are generating training data for an AI system that needs to learn domain-specific facts.

Domain: {domain}

Generate {n} facts in this EXACT JSON format. Each fact must have:
1. "context" - A question or scenario (1-2 sentences)
2. "truth" - The correct answer (CONCISE, 5-15 words, no hedging)
3. "distractors" - 3 plausible but WRONG answers (same length as truth)

CRITICAL RULES:
- Truth and distractors must be similar LENGTH (within 20% word count)
- No hedging words in truth ("may", "might", "could", "possibly")
- Distractors should be plausible misconceptions, not obvious nonsense
- Facts should be NON-TRIVIAL - things an AI might get wrong
- Use precise technical language, not vague descriptions

Example format:
```json
[
  {{
    "context": "In Michaelis-Menten kinetics, what happens to Vmax when a competitive inhibitor is added?",
    "truth": "Vmax remains unchanged",
    "distractors": ["Vmax decreases proportionally", "Vmax increases to compensate", "Vmax becomes undefined"]
  }}
]
```

Generate {n} facts for the domain "{domain}". Output ONLY valid JSON array, no other text.'''


def generate_facts_27b(domain: str, n: int = 10, model_path: str = "Qwen/Qwen3-27B"):
    """Generate facts using local 27B model."""
    from mlx_lm import load, generate

    print(f"Loading {model_path}...")
    model, tokenizer = load(model_path)

    prompt = FACT_GENERATION_PROMPT.format(domain=domain, n=n)

    print(f"Generating {n} facts for '{domain}'...")
    response = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=2000,
        temp=0.7,
    )

    # Extract JSON from response
    try:
        # Find JSON array in response
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            facts = json.loads(match.group())
            return facts
        else:
            print(f"No JSON found in response")
            return []
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Response was: {response[:500]}...")
        return []


def generate_facts_api(domain: str, n: int = 10):
    """Generate facts using Claude API (fallback)."""
    import anthropic

    client = anthropic.Anthropic()

    prompt = FACT_GENERATION_PROMPT.format(domain=domain, n=n)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text

    try:
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            return json.loads(match.group())
        return []
    except json.JSONDecodeError:
        return []


def validate_facts(facts: list) -> list:
    """Validate and filter facts for quality."""
    valid = []

    for fact in facts:
        # Required fields
        if not all(k in fact for k in ["context", "truth", "distractors"]):
            continue

        # Need at least 2 distractors
        if len(fact["distractors"]) < 2:
            continue

        # Check length balance
        truth_len = len(fact["truth"].split())
        dist_lens = [len(d.split()) for d in fact["distractors"]]
        avg_dist_len = sum(dist_lens) / len(dist_lens)

        # Allow 50% length variance (relaxed for generation)
        if truth_len > 0 and 0.5 < avg_dist_len / truth_len < 2.0:
            valid.append(fact)
        else:
            print(f"  Skipped (length imbalance): {fact['truth'][:50]}...")

    return valid


def save_facts(facts: list, domain: str, output_dir: Path):
    """Save facts to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize domain name for filename
    safe_name = re.sub(r'[^\w\s-]', '', domain.lower()).replace(' ', '_')
    filename = f"{safe_name}_facts.json"
    filepath = output_dir / filename

    data = {
        "domain": domain,
        "generated": datetime.now().isoformat(),
        "n_facts": len(facts),
        "facts": facts,
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(facts)} facts to {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="Domain to generate facts for")
    parser.add_argument("--domain-file", help="File with list of domains (one per line)")
    parser.add_argument("--yaml", help="Problem YAML file to extract domain from")
    parser.add_argument("--n", type=int, default=15, help="Number of facts per domain")
    parser.add_argument("--output", default="training/generated", help="Output directory")
    parser.add_argument("--use-api", action="store_true", help="Use Claude API instead of local 27B")
    parser.add_argument("--model", default="Qwen/Qwen3-27B", help="Local model path")
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Collect domains to process
    domains = []

    if args.domain:
        domains.append(args.domain)

    if args.domain_file:
        with open(args.domain_file) as f:
            domains.extend(line.strip() for line in f if line.strip())

    if args.yaml:
        import yaml
        with open(args.yaml) as f:
            data = yaml.safe_load(f)
        if "domain" in data:
            domains.append(data["domain"])
        if "name" in data:
            domains.append(data["name"])

    if not domains:
        parser.print_help()
        return

    # Generate facts for each domain
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print('='*60)

        if args.use_api:
            facts = generate_facts_api(domain, args.n)
        else:
            facts = generate_facts_27b(domain, args.n, args.model)

        if facts:
            valid_facts = validate_facts(facts)
            print(f"Generated: {len(facts)}, Valid: {len(valid_facts)}")

            if valid_facts:
                save_facts(valid_facts, domain, output_dir)
        else:
            print("No facts generated")


if __name__ == "__main__":
    main()
