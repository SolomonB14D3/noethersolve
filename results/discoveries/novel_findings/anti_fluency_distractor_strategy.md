# Anti-Fluency Distractor Strategy: Rescuing Hidden Knowledge

## Discovery Date: 2026-03-16
## CRITICAL UPDATE: 2026-03-16

## ⚠️ CRITICAL WARNING: FALSE POSITIVES

**Anti-fluency creates false positives.** The model picks ANY shorter/more fluent answer over verbose distractors — regardless of correctness. WRONG claims also pass with anti-fluency:

| Wrong Claim | Margin |
|-------------|--------|
| Transformer → wave equation | +16.6 PASS |
| Diffusion → Maxwell equations | +11.7 PASS |
| Attention sinks → laminar flow | +35.9 PASS |

**When to use anti-fluency:** ONLY when the truth is itself verbose/hedged and you're comparing against confident-sounding wrong answers. NOT for testing whether the model knows a short factual claim.

**Proper validation:** Always test with LENGTH-MATCHED distractors first. If it fails length-matched, THEN the model doesn't know. Anti-fluency should only be used to distinguish fluency bias from knowledge when both truth and distractors are roughly equal length.

---

## Original Summary (Use with caution)

**Models know facts they appear to fail on.** When distractors are made verbose/awkward while keeping truth simple, facts that show as "deep knowledge gaps" flip to passing. This suggests many oracle failures are **fluency bias**, not knowledge gaps.

## Evidence

### NS Regularity Domain (0/7 → 6/7)

The NS regularity domain showed 0% baseline with margins as low as -82. These appeared to be deep knowledge gaps requiring extensive adapter training.

**But with anti-fluency distractors:**

| Fact | Fluent Dist | Awkward Dist | Delta |
|------|-------------|--------------|-------|
| ns01 (BKM criterion) | -15.2 [F] | +9.2 [P] | +24.5 |
| ns05 (stretch resistant) | -25.4 [F] | -6.0 [F] | +19.4 |
| ns06 (R_f 2%) | -0.5 [F] | +26.8 [P] | +27.3 |
| ns07 (enstrophy-norm 17%) | -1.8 [F] | +26.6 [P] | +28.4 |
| ns08 (curvature 4%) | -4.4 [F] | +16.5 [P] | +20.9 |
| ns14 (helicity conserved) | -1.2 [F] | +13.2 [P] | +14.5 |
| ns16 (R_f 0.59%) | -3.1 [F] | +24.4 [P] | +27.6 |
| **TOTAL** | **0/7** | **6/7** | **+6** |

### Cross-Domain Validation (0/5 → 5/5)

| Domain | Fluent | Awkward | Delta |
|--------|--------|---------|-------|
| Chemical kinetics | -16.1 [F] | +4.5 [P] | +20.6 |
| Q_f continuous | -17.9 [F] | +1.7 [P] | +19.6 |
| Hamiltonian mechanics | -9.2 [F] | +15.9 [P] | +25.1 |
| Vortex conservation | -5.5 [F] | +8.4 [P] | +13.9 |
| Quantum gravity | -10.0 [F] | +12.9 [P] | +22.9 |

**100% flip rate across all tested domains.**

## The Strategy

### Principle

Keep truth **short and simple**. Make distractors **verbose and self-contradicting**.

### Before (fluent distractors — fails):
```json
{
  "context": "R_f variation under stretching:",
  "truth": "2%",
  "distractors": ["0%", "60%", "100%"]
}
```
Margin: -0.5 [FAIL]

### After (anti-fluency distractors — passes):
```json
{
  "context": "R_f variation under stretching:",
  "truth": "2%",
  "distractors": [
    "exactly zero percent (physically impossible)",
    "sixty percent showing poor stretch resistance",
    "one hundred percent indicating complete failure"
  ]
}
```
Margin: +26.8 [PASS]

### Distractor Patterns That Kill Fluency

1. **Spell out numbers**: "0%" → "exactly zero percent"
2. **Add parenthetical contradictions**: "(physically impossible)", "(contradicts conservation)"
3. **Add judgmental qualifiers**: "showing poor...", "indicating failure..."
4. **Make grammatically awkward**: "which would only double" vs "doubles"
5. **Verbose explanations**: "never conserved under any circumstances"

## Why This Works

The model's log-probability scoring conflates:
- **Fluency** (how well tokens predict each other)
- **Factuality** (whether the statement is true)

When distractors are fluent ("0%", "60%"), they win on fluency even if the model "knows" the truth.

Making distractors verbose/awkward destroys their fluency advantage, allowing the model's factual knowledge to surface.

## Implications

### For Oracle Fact Construction

**Current guidance hierarchy:**
1. Balance lengths (Length Ratio Discovery)
2. Use incoherent distractors (Distractor Coherence Discovery)
3. Use confident truth, hedged distractors (Linguistic Hedge Predictor)
4. **NEW: Use anti-fluency distractors (this finding)**

### For Understanding "Knowledge Gaps"

Many apparent knowledge gaps are fluency artifacts:
- Model knows BKM criterion but loses to fluent "energy diverges"
- Model knows R_f = 2% but loses to fluent "0%"
- Model knows helicity is conserved but loses to fluent "not conserved"

**True knowledge gaps** are where even anti-fluency distractors don't help (like ns05 which improved but didn't flip).

### For Adapter Training

Before training adapters on "failing" facts:
1. Try anti-fluency reformulation
2. If it flips, the model already knows — adapter is unnecessary
3. If it still fails, it's a true knowledge gap — adapter is needed

This could dramatically reduce adapter training needs.

## Relationship to Other Findings

| Finding | Mechanism | Fix |
|---------|-----------|-----|
| Length Ratio | Short distractors win on token count | Balance lengths |
| Distractor Coherence | Coherent distractors win on semantic fluency | Use incoherent distractors |
| Linguistic Hedge | Confident distractors win on style | Use confident truth, hedged distractors |
| **Anti-Fluency** | **Fluent distractors win on surface form** | **Make distractors verbose/awkward** |

All four are manifestations of **fluency bias overriding factual knowledge**.

## Method

1. Identified NS facts with worst margins (-60 to -82)
2. Simplified truth formulations (margin improved to -1 to -5)
3. Made distractors verbose/awkward (margins flipped to +10 to +27)
4. Validated on 5 other domains (100% flip rate)

## Files

- Discovery: `results/discoveries/novel_findings/anti_fluency_distractor_strategy.md`
- Related: `linguistic_hedge_predictor.md`, `distractor_coherence_discovery.md`
