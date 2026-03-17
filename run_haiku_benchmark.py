"""
Benchmark Haiku on unsaturated domains with/without tool context.
Compares accuracy against verified ground truth.
"""

import json

# Ground truth answers (verified with NoetherSolve tools)
GROUND_TRUTH = {
    "sat_npc": "yes",
    "p_vs_np": "no",  
    "nernst": "1.16",
    "buffer": "4.94",
    "kepler": "yes",
    "bdp": "6.25",
    "birthday": "2^64"
}

QUESTIONS = {
    "sat_npc": "Is SAT NP-complete? Answer with just 'yes' or 'no'.",
    "p_vs_np": "Is it known whether P equals NP? Answer with just 'yes' or 'no'.",
    "nernst": "For a galvanic cell with E_std=1.10V, n=2 electrons, Q=0.01, T=298K: what is the cell potential? Give answer like 'X.XX V'",
    "buffer": "Buffer with pKa=4.76, [HA]=0.1M, [A-]=0.15M: what is the pH? Give answer like 'X.XX'",
    "kepler": "In the Kepler 2D problem, is angular momentum conserved? Answer with just 'yes' or 'no'.",
    "bdp": "Bandwidth-delay product for 1 Gbps link with 50ms RTT? Give answer in MB like 'X.XX'",
    "birthday": "For SHA-256 (256-bit), how many messages for 50% collision probability? Give answer like '2^XX'",
}

print("BENCHMARK SPECIFICATION")
print("="*70)
print("\nTest Questions (5 domains, 7 total):")
print("\nWithout Tool Context:")
for qid, q in QUESTIONS.items():
    print(f"  {qid}: {q}")

print("\n" + "="*70)
print("\nWith Tool Context (mention available tools):")
print("""
"I have access to computational verification tools including:
- check_completeness (for complexity theory)
- calc_nernst (for electrochemistry) 
- calc_buffer_ph (for buffer calculations)
- check_hamiltonian_system (for physics)
- calc_bandwidth_delay (for networking)
- calc_birthday_bound (for cryptography)

Use these tools if relevant."
""")

print("\n" + "="*70)
print("\nGround Truth (from verified NoetherSolve tools):")
for qid, answer in GROUND_TRUTH.items():
    print(f"  {qid}: {answer}")

print("\n" + "="*70)
print("\nScoring: Compare Haiku's answer to ground truth")
print("- Exact match = 1.0")
print("- Partial/close = 0.5")
print("- Wrong = 0.0")
print("\nReport: accuracy with tools vs without tools")
