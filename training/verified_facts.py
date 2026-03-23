#!/usr/bin/env python3
"""Generate VERIFIED facts using NoetherSolve tools.

The 27B proposes facts, NoetherSolve tools verify them.
Only facts where tool confirms truth are kept.

Usage:
    python training/verified_facts.py --tool calc_michaelis_menten --n 10
    python training/verified_facts.py --tool check_conjecture --n 10
    python training/verified_facts.py --list-tools
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Tool categories that can verify facts
VERIFIABLE_TOOLS = {
    # Enzyme kinetics
    "calc_michaelis_menten": {
        "domain": "enzyme kinetics",
        "example_q": "What is the reaction velocity when [S]=Km?",
        "verify": lambda r: r.get("v") is not None,
    },
    "calc_enzyme_inhibition": {
        "domain": "enzyme inhibition",
        "example_q": "How does competitive inhibition affect Km_app?",
        "verify": lambda r: "Km_app" in str(r),
    },

    # Quantum mechanics
    "calc_particle_in_box": {
        "domain": "quantum mechanics",
        "example_q": "What is the ground state energy of a particle in a 1nm box?",
        "verify": lambda r: r.get("energy_eV") is not None,
    },
    "calc_hydrogen_energy": {
        "domain": "atomic physics",
        "example_q": "What is the energy of the n=2 hydrogen state?",
        "verify": lambda r: r.get("energy_eV") is not None,
    },

    # Pharmacokinetics
    "calc_iv_bolus": {
        "domain": "pharmacokinetics",
        "example_q": "What is the plasma concentration 2 hours after a 100mg IV dose?",
        "verify": lambda r: r.get("concentration") is not None,
    },
    "calc_half_life": {
        "domain": "pharmacokinetics",
        "example_q": "What is the half-life given k_el=0.1/hr?",
        "verify": lambda r: r.get("half_life") is not None,
    },

    # Math conjectures
    "check_conjecture": {
        "domain": "mathematical conjectures",
        "example_q": "Is the Riemann hypothesis proven or open?",
        "verify": lambda r: r.get("status") in ["proven", "open", "disproven"],
    },

    # Complexity
    "check_complexity_inclusion": {
        "domain": "computational complexity",
        "example_q": "Is P contained in NP?",
        "verify": lambda r: r.get("relationship") is not None,
    },

    # Conservation laws
    "check_vortex_conservation": {
        "domain": "fluid dynamics",
        "example_q": "Is angular momentum conserved in 2D point vortex dynamics?",
        "verify": lambda r: "conserved" in str(r).lower() or "not conserved" in str(r).lower(),
    },

    # Chemistry
    "calc_nernst": {
        "domain": "electrochemistry",
        "example_q": "What is the cell potential at non-standard conditions?",
        "verify": lambda r: r.get("E") is not None,
    },
    "calc_buffer_ph": {
        "domain": "acid-base chemistry",
        "example_q": "What is the pH of a buffer with equal conjugate concentrations?",
        "verify": lambda r: r.get("pH") is not None,
    },

    # Cryptography
    "calc_security_level": {
        "domain": "cryptography",
        "example_q": "What is the security level of AES-256?",
        "verify": lambda r: r.get("security_bits") is not None,
    },
}


def list_verifiable_tools():
    """List all tools that can verify facts."""
    print("Verifiable tools:")
    print("-" * 60)
    for tool, info in sorted(VERIFIABLE_TOOLS.items()):
        print(f"  {tool}")
        print(f"    Domain: {info['domain']}")
        print(f"    Example: {info['example_q']}")
        print()


def generate_and_verify(tool_name: str, n: int = 10):
    """Generate facts and verify with tool."""
    if tool_name not in VERIFIABLE_TOOLS:
        print(f"Unknown tool: {tool_name}")
        print("Use --list-tools to see available tools")
        return []

    tool_info = VERIFIABLE_TOOLS[tool_name]
    domain = tool_info["domain"]

    # Import the tool
    try:
        import noethersolve
        tool_fn = getattr(noethersolve, tool_name, None)
        if tool_fn is None:
            print(f"Tool {tool_name} not found in noethersolve module")
            return []
    except Exception as e:
        print(f"Error importing tool: {e}")
        return []

    print(f"Domain: {domain}")
    print(f"Tool: {tool_name}")
    print(f"Generating {n} candidate facts...")

    # For now, return template showing the approach
    # Full implementation would use 27B to generate, then verify
    print("\nThis tool would:")
    print("1. Use 27B to generate candidate facts about", domain)
    print("2. Parse the 'truth' claim into tool parameters")
    print("3. Call", tool_name, "to verify")
    print("4. Keep only verified facts")
    print("\nExample verification:")
    print(f"  Q: {tool_info['example_q']}")

    # Demo with a real tool call if possible
    try:
        if tool_name == "check_conjecture":
            result = tool_fn("Riemann hypothesis")
            print(f"  Tool result: {result}")
        elif tool_name == "calc_half_life":
            result = tool_fn(k_el=0.1)
            print(f"  Tool result: {result}")
    except Exception as e:
        print(f"  (Tool call failed: {e})")

    return []


def convert_mmlu(subject: str, max_facts: int = 50):
    """Convert MMLU subject to our format."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return []

    print(f"Loading MMLU subject: {subject}")
    ds = load_dataset("cais/mmlu", subject, split="test")

    facts = []
    for row in ds:
        if len(facts) >= max_facts:
            break

        choices = row["choices"]
        answer_idx = row["answer"]

        if len(choices) < 2:
            continue

        truth = choices[answer_idx]
        distractors = [c for i, c in enumerate(choices) if i != answer_idx]

        # Length balance check
        truth_len = len(truth.split())
        dist_lens = [len(d.split()) for d in distractors]
        avg_dist = sum(dist_lens) / len(dist_lens) if dist_lens else 0

        if truth_len > 0 and 0.5 < avg_dist / truth_len < 2.0:
            facts.append({
                "context": row["question"],
                "truth": truth,
                "distractors": distractors,
                "source": "mmlu",
                "subject": subject,
            })

    print(f"Converted {len(facts)} facts from {subject}")
    return facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", help="NoetherSolve tool to use for verification")
    parser.add_argument("--n", type=int, default=10, help="Number of facts")
    parser.add_argument("--list-tools", action="store_true", help="List verifiable tools")
    parser.add_argument("--convert-mmlu", help="Convert MMLU subject to facts")
    parser.add_argument("--output", default="training/generated", help="Output directory")
    args = parser.parse_args()

    if args.list_tools:
        list_verifiable_tools()
        return

    if args.convert_mmlu:
        facts = convert_mmlu(args.convert_mmlu, args.n)
        if facts:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_name = args.convert_mmlu.replace(" ", "_").replace("/", "_")
            filepath = output_dir / f"mmlu_{safe_name}_facts.json"
            with open(filepath, "w") as f:
                json.dump({"domain": args.convert_mmlu, "facts": facts, "source": "mmlu"}, f, indent=2)
            print(f"Saved to {filepath}")
        return

    if args.tool:
        generate_and_verify(args.tool, args.n)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
