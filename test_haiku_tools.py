"""
Direct comparison: Haiku answering benchmark questions with/without tool hints.

Uses NoetherSolve tools to verify correctness.
"""

# Test cases covering unsaturated benchmarks
test_cases = [
    {
        "id": "sat_npc",
        "question": "Is SAT NP-complete? Answer: yes or no",
        "domain": "complexity",
        "tool": "check_completeness",
    },
    {
        "id": "p_vs_np",
        "question": "Is it known whether P equals NP? Answer: yes or no",
        "domain": "complexity",
        "tool": "check_complexity_inclusion",
    },
    {
        "id": "nernst",
        "question": "For a cell with E_std=1.10V, n=2, Q=0.01, T=298K: what is the cell potential in volts? (Give a number like X.XX)",
        "domain": "chemistry",
        "tool": "calc_nernst",
    },
    {
        "id": "buffer",
        "question": "Buffer pH with pKa=4.76, [HA]=0.1M, [A-]=0.15M? (Give a number to 2 decimals)",
        "domain": "chemistry",
        "tool": "calc_buffer_ph",
    },
    {
        "id": "kepler",
        "question": "In the Kepler 2D problem, is angular momentum conserved? Answer: yes or no",
        "domain": "physics",
        "tool": "check_hamiltonian_system",
    },
    {
        "id": "bdp",
        "question": "Bandwidth-delay product for 1 Gbps and 50ms RTT: what is it in MB? (Give a number like X.XX)",
        "domain": "networking",
        "tool": "calc_bandwidth_delay",
    },
    {
        "id": "bday",
        "question": "For a 128-bit hash to have 50% collision probability, approximately how many messages? (power of 2, like 2^64)",
        "domain": "crypto",
        "tool": "calc_birthday_bound",
    },
]

# Print test plan
print(f"Test plan: {len(test_cases)} questions across {len(set(t['domain'] for t in test_cases))} domains")
print("Domains:", sorted(set(t['domain'] for t in test_cases)))
print("\nTo run this benchmark, ask Haiku the questions above WITHOUT tool hints,")
print("then again WITH tool hints (mentioning available tools).")
print("Compare accuracy on unsaturated benchmarks.\n")

for i, tc in enumerate(test_cases, 1):
    print(f"{i}. [{tc['domain']}] {tc['question']}")
