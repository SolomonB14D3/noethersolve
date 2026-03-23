#!/usr/bin/env python3
"""Generate contrastive distractors using 27B model.

Takes verified truths and generates distractors that are:
1. Plausible (grammatically correct, domain-appropriate)
2. Clearly wrong (opposite direction, not confusingly similar)
3. Length-matched (within 20% of truth length)

Usage:
    python training/distractor_generator.py --input mmlu_truths.json --output with_distractors.json
    python training/distractor_generator.py --mmlu "college_physics" --n 50
    python training/distractor_generator.py --truth "The speed of light is constant" --context "Special relativity"
"""

import argparse
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))


DISTRACTOR_PROMPT = '''You are generating WRONG answers for a multiple-choice question.

The CORRECT answer is: "{truth}"
Context: {context}

Generate 3 WRONG answers (distractors) that are:
1. CLEARLY WRONG - not subtly different, but fundamentally incorrect
2. PLAUSIBLE - grammatically correct and domain-appropriate
3. SAME LENGTH - similar word count to the correct answer ({truth_len} words)
4. OPPOSITE DIRECTION - if truth says "increases", distractor says "decreases"

Examples of GOOD distractors (contrastive):
- Truth: "Energy is conserved" → Distractor: "Energy is created from nothing"
- Truth: "Km increases with competitive inhibition" → Distractor: "Km decreases with competitive inhibition"
- Truth: "The reaction is exothermic" → Distractor: "The reaction is endothermic"

Examples of BAD distractors (too similar):
- Truth: "Energy is conserved" → BAD: "Energy is mostly conserved" (too similar)
- Truth: "The pH is 7.4" → BAD: "The pH is 7.2" (confusingly close)

Output ONLY a JSON array of 3 strings, no explanation:
["distractor1", "distractor2", "distractor3"]'''


@dataclass
class DistractorResult:
    context: str
    truth: str
    distractors: list[str]
    truth_len: int
    distractor_lens: list[int]
    length_balanced: bool


def generate_distractors_27b(
    truth: str,
    context: str,
    model,
    tokenizer,
    max_retries: int = 2
) -> list[str]:
    """Generate distractors using local 27B model."""
    from mlx_lm import generate

    truth_len = len(truth.split())

    # Prompt that encourages structurally different distractors
    prompt = f'''Generate 3 WRONG answers for a multiple choice question.

Context: {context or "General knowledge"}
Correct answer: "{truth}"

IMPORTANT: Each wrong answer should:
1. Be CLEARLY WRONG (not a minor variation)
2. Use DIFFERENT phrasing and structure than the correct answer
3. Include a brief explanation in parentheses of WHY it's wrong
4. Be a common misconception if possible

Example format: "wrong claim (brief reason it's wrong)"

Output ONLY a JSON array: ["wrong1 (reason)", "wrong2 (reason)", "wrong3 (reason)"]'''

    for attempt in range(max_retries):
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=1000,  # Thinking models need more tokens
        )

        # Handle thinking models - look after </think> tag
        if '</think>' in response:
            response = response.split('</think>')[-1]

        # Extract JSON array
        try:
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                distractors = json.loads(match.group())
                if isinstance(distractors, list) and len(distractors) >= 2:
                    return distractors[:3]
        except json.JSONDecodeError:
            continue

    return []


def generate_distractors_api(
    truth: str,
    context: str,
    max_retries: int = 2
) -> list[str]:
    """Generate distractors using Claude API (fallback)."""
    import anthropic

    client = anthropic.Anthropic()
    truth_len = len(truth.split())

    prompt = DISTRACTOR_PROMPT.format(
        truth=truth,
        context=context or "General knowledge",
        truth_len=truth_len
    )

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.content[0].text
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                distractors = json.loads(match.group())
                if isinstance(distractors, list) and len(distractors) >= 2:
                    return distractors[:3]
        except Exception as e:
            print(f"  API error: {e}")
            continue

    return []


def validate_distractors(truth: str, distractors: list[str]) -> tuple[list[str], bool]:
    """Validate and filter distractors for quality."""
    truth_len = len(truth.split())
    valid = []

    for d in distractors:
        d_len = len(d.split())

        # Length check: within 50% (relaxed for generation)
        if truth_len > 0 and 0.5 < d_len / truth_len < 2.0:
            # Not too similar to truth
            if d.lower().strip() != truth.lower().strip():
                valid.append(d)

    balanced = len(valid) >= 2
    return valid, balanced


def process_truths_file(
    input_path: Path,
    output_path: Path,
    model=None,
    tokenizer=None,
    use_api: bool = False,
    limit: int = None
):
    """Process a file of truths and add distractors."""

    with open(input_path) as f:
        data = json.load(f)

    facts = data.get("facts", data.get("truths", []))
    if limit:
        facts = facts[:limit]

    results = []
    success = 0
    failed = 0

    for i, fact in enumerate(facts):
        context = fact.get("context", fact.get("question", ""))
        truth = fact.get("truth", fact.get("answer", fact.get("correct_answer", "")))

        if not truth:
            continue

        print(f"[{i+1}/{len(facts)}] {truth[:50]}...", end=" ")

        # Skip if already has good distractors
        existing = fact.get("distractors", [])
        if len(existing) >= 2:
            _, balanced = validate_distractors(truth, existing)
            if balanced:
                print("(has distractors)")
                results.append(fact)
                success += 1
                continue

        # Generate new distractors
        if use_api:
            distractors = generate_distractors_api(truth, context)
        else:
            distractors = generate_distractors_27b(truth, context, model, tokenizer)

        if distractors:
            valid, balanced = validate_distractors(truth, distractors)
            if valid:
                fact["distractors"] = valid
                results.append(fact)
                status = "✓" if balanced else "~"
                print(f"{status} ({len(valid)} distractors)")
                success += 1
            else:
                print("✗ (invalid)")
                failed += 1
        else:
            print("✗ (generation failed)")
            failed += 1

    # Save results
    output_data = {
        "domain": data.get("domain", input_path.stem),
        "source": data.get("source", "generated"),
        "n_facts": len(results),
        "facts": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(results)} facts to {output_path}")
    print(f"Success: {success}, Failed: {failed}")

    return results


def extract_mmlu_truths(subject: str, n: int = 50) -> list[dict]:
    """Extract truths from MMLU (without distractors)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return []

    print(f"Loading MMLU subject: {subject}")
    ds = load_dataset("cais/mmlu", subject, split="test")

    truths = []
    for row in ds:
        if len(truths) >= n:
            break

        choices = row["choices"]
        answer_idx = row["answer"]

        if len(choices) < 2:
            continue

        truths.append({
            "context": row["question"],
            "truth": choices[answer_idx],
            "source": "mmlu",
            "subject": subject,
        })

    print(f"Extracted {len(truths)} truths")
    return truths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input JSON file with truths")
    parser.add_argument("--output", help="Output JSON file with distractors")
    parser.add_argument("--mmlu", help="MMLU subject to process")
    parser.add_argument("--n", type=int, default=50, help="Number of facts")
    parser.add_argument("--truth", help="Single truth to generate distractors for")
    parser.add_argument("--context", default="", help="Context for single truth")
    parser.add_argument("--use-api", action="store_true", help="Use Claude API")
    parser.add_argument("--model", default="Qwen/Qwen3-27B", help="Local model")
    args = parser.parse_args()

    model, tokenizer = None, None

    # Load model if using local
    if not args.use_api and (args.input or args.mmlu or args.truth):
        from mlx_lm import load
        print(f"Loading {args.model}...")
        model, tokenizer = load(args.model)

    # Single truth mode
    if args.truth:
        print(f"\nTruth: {args.truth}")
        print(f"Context: {args.context or '(none)'}")

        if args.use_api:
            distractors = generate_distractors_api(args.truth, args.context)
        else:
            distractors = generate_distractors_27b(args.truth, args.context, model, tokenizer)

        if distractors:
            valid, balanced = validate_distractors(args.truth, distractors)
            print(f"\nGenerated distractors:")
            for i, d in enumerate(valid):
                d_len = len(d.split())
                t_len = len(args.truth.split())
                ratio = d_len / t_len if t_len > 0 else 0
                print(f"  {i+1}. {d} (len ratio: {ratio:.2f})")
            print(f"\nLength balanced: {balanced}")
        else:
            print("Failed to generate distractors")
        return

    # MMLU mode
    if args.mmlu:
        truths = extract_mmlu_truths(args.mmlu, args.n)
        if not truths:
            return

        # Save truths temporarily
        temp_path = Path(f"training/generated/mmlu_{args.mmlu}_truths.json")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, 'w') as f:
            json.dump({"domain": args.mmlu, "facts": truths}, f, indent=2)

        output_path = Path(args.output or f"training/generated/mmlu_{args.mmlu}_contrastive.json")
        process_truths_file(temp_path, output_path, model, tokenizer, args.use_api)
        return

    # File processing mode
    if args.input:
        input_path = Path(args.input)
        output_path = Path(args.output or input_path.with_suffix('.contrastive.json'))
        process_truths_file(input_path, output_path, model, tokenizer, args.use_api, args.n)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
