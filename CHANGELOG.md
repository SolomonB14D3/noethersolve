# Changelog

## 2026-03-20 — Runner Reliability + Project Organization

### Research Runner
- **Progress-aware stopping**: Runner tracks sweep-over-sweep progress. Stops after 3 idle polls instead of polling forever.
- **V2 invalidation**: Detects when fact files are updated and re-evaluates automatically.
- **Sweep summaries**: Each sweep writes to `results/run_summary.json` with progress delta, improved domains, and timing.
- **`--once` flag**: Run a single sweep and exit (for cron/scheduled use).
- **Cleaner error handling**: Timeout and crash on one domain doesn't block the rest.
- **Removed adapter training from 27B**: The 27B is the oracle/judge. Only 4B gets adapters.

### Script Cleanup
- Archived 19 dead/superseded scripts to `scripts/archive/`
- Added `scripts/README.md` documenting active scripts and usage

### Documentation
- Created this CHANGELOG
- Added Research Runner Operations section to CLAUDE.md
- Added V2 Fact File Campaign section to CLAUDE.md

## 2026-03-19 — V2 Length-Matched Fact Campaign

### V2 Fact Files
Created length-matched V2 rewrites for 24 domains. V1 facts had systematic length ratio bias (detailed truths with dismissive one-liner distractors, ratios 2-31x). V2s match distractor length to truth length (ratio 0.8-1.2x).

**Results (V1 → V2 improvement on 27B):**
- Information Theory: 8% → 92%
- LLM Reasoning: 25% → 83%
- LLM Training: 17% → 83%
- Intersection Theory: 0% → 75%
- LLM Context/Memory: 10% → 70%
- Knot Invariants: 0% → 50%
- Elliptic Curves: 42% → 50%
- Continuous Q_f: 0% → 25%

### V2 Domains Created
`algebra_topology_conjectures_v2`, `computational_conjectures_v2`, `continuous_qf_v2`, `disease_targets_v2`, `drug_interactions_v2`, `elliptic_curves_v2`, `genetics_therapeutics_v2`, `immune_evasion_v2`, `information_theory_v2`, `intersection_theory_v2`, `kinetic_k_v2`, `knot_invariants_v2`, `llm_alignment_v2`, `llm_context_memory_v2`, `llm_evaluation_v2`, `llm_hallucination_grounded_v2`, `llm_hallucination_v2`, `llm_reasoning_v2`, `llm_training_v2`, `optimal_f_v2`, `physics_fundamentals_v2`, `proof_techniques_v2`, `qf_ratio_v2`, `vortex_unsolved_v2`

## 2026-03-18 — Hooks Integration

- Added `UserPromptSubmit` hook showing research status + escalations in Claude context
- Added `PreToolUse`/`PostToolUse` hooks for resource monitoring
- Added `Stop` hook for session cleanup
