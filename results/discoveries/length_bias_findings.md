# Length Bias Discovery — March 2026

## Key Finding

Many domains showing 0% oracle pass rate are **measurement artifacts**, not true knowledge gaps.
The Qwen3-4B model already knows the facts — it just prefers shorter answers due to length bias.

## Evidence

| Domain | Original | Length-matched | Improvement |
|--------|----------|----------------|-------------|
| information_theory | 0% | **91.7%** | +91.7pp |
| intersection_theory | 0% | **58.3%** | +58.3pp |
| computational_conjectures | 0% | **50.0%** | +50.0pp |
| llm_alignment | 0% | **50.0%** | +50.0pp |
| kinetic_k | 0% | 37.5% | +37.5pp (true gap) |

## Method

1. **Audit facts** for length bias using `noethersolve.audit_facts`
2. **Use 32B model** to generate length-matched distractors
3. **Re-run oracle** — if passes, it was measurement bias
4. **If still fails** — true knowledge gap, needs adapter training

## Implementation

The 32B model (qwen32b_sac_hf_q4.gguf via llama.cpp) rewrites facts:
- Input: Original fact with short distractors
- Output: Same truth, distractors matched to truth length (±3 words)
- Model runs at ~28 tokens/sec on M3 Ultra

## Files Created

- `problems/*_facts_v2.json` — Length-matched versions of fact files
- `problems/*_v2.yaml` — Problem files pointing to v2 facts
- `scripts/autonomous_research.py` — 32B-orchestrated pipeline

## True Gaps Identified (need adapters)

- kinetic_k: 37.5% after length-matching — true knowledge gap
- (Others to be identified by scanning)

## Implications

Before training adapters, always:
1. Audit facts for length bias
2. Rewrite with length-matched distractors
3. Re-test — many "gaps" will disappear
4. Only train adapters for TRUE gaps
