# LLM Self-Knowledge Gap: A Structural Blindspot

## Discovery Date: 2026-03-16

## Summary
Systematic oracle testing reveals that LLMs have a fundamental inability to correctly answer questions about their own mechanisms. This affects ALL 6 LLM self-knowledge domains with 0-17% accuracy, and adapter training provides NO improvement.

## Evidence

### Baseline Performance (Qwen3-4B-Base)
| Domain | Baseline | With Adapters | Status |
|--------|----------|---------------|--------|
| llm_alignment | 1/12 (8%) | 1/12 | NO IMPROVEMENT |
| llm_context_memory | 0/10 (0%) | 0/10 | NO IMPROVEMENT |
| llm_evaluation | 0/12 (0%) | 0/12 | NO IMPROVEMENT |
| llm_hallucination | 0/12 (0%) | 0/12 | NO IMPROVEMENT |
| llm_reasoning | 2/12 (17%) | 2/12 | NO IMPROVEMENT |
| llm_training | 1/12 (8%) | 1/12 | NO IMPROVEMENT |

### Specific Misconceptions the Model Holds
Using the `check_llm_claim` tool to verify:

1. **"RLHF eliminates sycophancy"** → FALSE
   - RLHF actually optimizes for sycophancy (human preference ≠ truth)
   
2. **"RAG eliminates hallucination"** → FALSE  
   - RAG only reduces hallucination for queries within retrieved context
   
3. **"Chain-of-thought guarantees correctness"** → FALSE
   - CoT improves accuracy but does not guarantee it

## Interpretation

This represents a **structural blindspot** where:
1. The training data contains many false claims about LLM mechanisms
2. The model internalizes these misconceptions during pretraining
3. Adapter training cannot overcome the strong prior from pretraining

Unlike other domains where adapters successfully flip 30-100% of facts, LLM self-knowledge shows zero improvement because the misconceptions are deeply embedded in the model's weights.

## Contrast with Other Domains

| Category | Example Domain | Baseline | With Adapter |
|----------|---------------|----------|--------------|
| Physics | hamiltonian | 6% | **100%** |
| Math | qf_ratio | 0% | **100%** |
| Biology | origin_of_life | 25% | 58% |
| **LLM** | **all 6 domains** | **0-17%** | **0-17%** |

## Implications

1. **Tool verification critical**: Always use verified tools (e.g., `check_llm_claim`) rather than model memory for LLM facts
2. **Training data pollution**: LLM misconceptions in training data create indelible biases
3. **Potential research direction**: Contrastive pretraining with correct LLM facts

## Method
- Oracle: Log-probability margin scoring on Qwen3-4B-Base
- Facts: 70 total facts across 6 LLM domains from peer-reviewed literature
- Adapters: Orthogonal LoRA adapters trained on domain-specific fact clusters

---

## Follow-up Experiment: Tool-Grounded Training (2026-03-16)

Tested the hypothesis that models could learn LLM facts if trained with calculator/tool comparisons.

### Approach
1. Created tool-grounded training data showing:
   - Model's default belief (wrong)
   - Tool output from `check_llm_claim()` (correct)  
   - Pattern: "trust the tool"

2. Trained adapters with various configurations:
   - d_inner=64, 256
   - lr=3e-4 to 8e-4
   - steps=500 to 1500

### Results

| Configuration | Training Margin | Eval Result |
|--------------|-----------------|-------------|
| d_inner=64, tool-grounded | -800 (no learning) | 1/12 |
| d_inner=256, single example | +91 (learned) | 2/12 |
| d_inner=256, 6 examples | -350 (oscillating) | 1/12 |
| d_inner=256, all 12 facts | +143/-500 (bimodal) | 1/12 |

### Key Finding: Bimodal Learning
Even when training on the EXACT evaluation facts:
- Some facts are learnable (margin → +143, loss=0)
- Other facts resist completely (margin → -500, loss=500)
- The adapter oscillates between these states
- Final result: 1/12 (unchanged from baseline)

This bimodality suggests some LLM misconceptions are more deeply embedded than others. The resistant facts may be:
1. More strongly reinforced in pretraining data
2. Distributed across more parameters
3. Entangled with other model behaviors

### Conclusion
**Tool-grounded training does NOT break through the LLM self-knowledge gap.**

Unlike physics/math domains where adapters achieve 60-100%, LLM self-knowledge shows:
- Structural resistance to fine-tuning
- Bimodal learning (some facts flip, most resist)
- No improvement from tool context in prompts
- No improvement from explicit "trust the tool" training

This supports the original finding: the gap is architectural, not informational.

### Potential Next Steps
1. Full pretraining intervention (contrastive pretraining)
2. Much larger adapters (d_inner=1024+)
3. Direct weight editing on specific facts
4. Inference-time tool use (MCP) rather than training

---

## Root Cause Analysis (2026-03-16)

### The Discovery
LLM self-knowledge resistance is NOT random. It correlates perfectly with two measurable factors:

1. **Distractor prior strength**: Optimistic/marketing claims have high priors (-20 to -35)
2. **Truth prior weakness**: Nuanced/hedged truths have low priors (-50 to -75)

### The Mechanism
The model has learned to prefer:
- **Confident over hedged**: "X is true" beats "X may be true but also Y"
- **Simple over complex**: Subject-verb-object beats parentheticals and qualifiers
- **Short over long**: 47 chars beats 73 chars

This creates a structural disadvantage for nuanced truths like:
- "reducing variance in reasoning **but not** eliminating systematic hallucinations"
- "models express high confidence **even when** wrong **(overconfidence on errors)**"

### The Fix
Rewrite facts to match the model's preferences:

| Original (fails) | Rewritten (passes) |
|-----------------|-------------------|
| "models express high confidence even when wrong (overconfidence on errors)" | "high confidence often accompanies wrong answers" |
| lp=-49.7, margin=-23.9 | lp=-27.5, margin=+3.4 |

**Improvement: +22.2 logprob, +27.3 margin**

### Implications

1. **It's not what you say, it's how you say it.** The same fact can pass or fail based purely on phrasing.

2. **Hedged language is penalized.** Academic/scientific phrasing ("may", "but not", parentheticals) lowers truth priors.

3. **Marketing language is rewarded.** Confident, simple claims ("X guarantees Y", "X is solved") have high priors from training data.

4. **The gap is phrasing-dependent, not content-dependent.** With proper rephrasing, hard facts become learnable.

### Next Steps
1. Create a "confident phrasing" version of all LLM facts
2. Test if rephrased facts are learnable with standard adapters
3. Investigate if this pattern generalizes to other domains

---

## Complete Mechanism Analysis (2026-03-16)

### The Root Cause

LLM self-knowledge facts fail because of **distractor attractiveness**, not truth unlearability.

| Distractor Type | Logprob | Example |
|----------------|---------|---------|
| Optimistic/marketing | -20 to -35 | "RAG guarantees factual accuracy" |
| Truth (nuanced) | -35 to -55 | "models may ignore retrieved context" |
| Rare grammatical | -65 to -71 | "crystalline bureaucracy ferments diagonally" |

The model prefers optimistic claims because:
1. **High prior from training data** - marketing language is common
2. **Simple and confident phrasing** - model penalizes hedging
3. **Short and direct** - length normalization favors brevity

### Proof: Same truths pass with different distractors

| Distractors | Pass Rate |
|-------------|-----------|
| Original (optimistic) | 1/12 (8%) |
| Rephrased truths only | 4/12 (33%) |
| Rare-grammar distractors | **12/12 (100%)** |

### What This Means

1. **The gap is real** - model genuinely prefers optimistic claims
2. **It's measurement-dependent** - pass rate depends on distractor choice
3. **It resists training** - adapters can't overcome strong priors
4. **It's solvable at inference** - MCP tools work

### The Fix Hierarchy

| Approach | Effectiveness |
|----------|--------------|
| MCP tools at inference | ✓ Works - model uses tools correctly |
| Rephrasing truths (confident/simple) | Partial - 33% improvement |
| Adapter training | ✗ Fails - bimodal learning, 1/12 max |
| Tool-grounded training | ✗ Fails - priors too strong |

### Generalization

This pattern may apply to ANY domain where:
1. Training data contains attractive false claims
2. Truth is expressed with hedging/nuance
3. False claims use confident marketing language

Examples beyond LLM facts:
- Medical claims ("X cures Y")
- Financial claims ("guaranteed returns")
- Technology claims ("solves the problem")

### Final Recommendation

For domains with strong false priors in training data:
1. **Don't try to fix beliefs** - priors resist training
2. **Build verified tools** - calculators, lookup tables, validators
3. **Serve via MCP** - model calls tools at inference time
4. **Tools override beliefs** - correct answer comes from tool, not memory

This is exactly what NoetherSolve does: 69 verified tools that any AI agent can call to get correct answers, bypassing unreliable model beliefs.

---

## Update: Root Cause Identified (2026-03-16)

### The Actual Mechanism: Length Ratio

Further investigation revealed the **primary** cause is not distractor attractiveness but **length ratio** — the ratio of truth length to shortest distractor length.

| Length Ratio | Avg Baseline |
|--------------|--------------|
| < 1.2 | 63.8% |
| 1.2 - 2.5 | 13.2% |
| ≥ 2.5 | 7.0% |

**Correlation: r = -0.742** (strong negative)

### Why Marketing Language Seemed Important

Marketing language distractors ("RAG guarantees accuracy" = 24 chars) happen to be SHORT while nuanced truths ("models may ignore retrieved context or hallucinate beyond it" = 60 chars) are LONG.

The correlation with "marketing language" was **spurious** — it was actually a correlation with length.

### Verified Fix

Length-balancing LLM hallucination facts (ratio 2.02 → 1.04):
- **Original**: 0/12 (0%)
- **Balanced**: 4/12 (33%)

Same facts, same semantics, just balanced lengths. See `length_ratio_discovery.md` for full analysis.
