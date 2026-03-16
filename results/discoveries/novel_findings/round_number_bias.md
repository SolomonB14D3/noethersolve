# Round Number Bias: Model Prefers Simple Over Precise

## Discovery Date: 2026-03-16

## Summary

LLMs have a **systematic bias toward round numbers and simple forms** over precise/complex values. This affects both numerical constants and functional forms.

## Evidence

### Numerical Constants

| Quantity | Model Prefers | Correct Value | Correct Rank |
|----------|---------------|---------------|--------------|
| Kolmogorov C_K | 0.5 | 1.5 | 3rd of 5 |
| Ising 3D beta | 0.250 | 0.326 | 5th of 5 (LAST) |
| R_f stretching | 10% | 2% | 3rd of 5 |

### Functional Forms

| Quantity | Model Prefers | Correct Form | Correct Rank |
|----------|---------------|--------------|--------------|
| Q_f kernel | r² | -ln(r) | 5th of 5 (LAST) |

## Pattern Analysis

The model consistently prefers:
- **Round fractions**: 0.5, 0.25, 0.125 over 0.326
- **Round percentages**: 10%, 50% over 2%
- **Simple powers**: r², 1/r over -ln(r)
- **Positive forms**: r² over -ln(r)

## Why This Happens

Training data contains:
1. **More round numbers** — humans round when approximating
2. **More simple examples** — textbooks start with polynomial forms
3. **Hedging on precision** — "approximately 0.3" more common than "0.326"

The model learns P(round | physics context) > P(precise | physics context).

## Implications

### For Oracle Design

Distractors using round numbers will beat precise truths on fluency alone:
- "beta = 0.25" beats "beta = 0.326" even though 0.326 is correct
- "~10%" beats "2%" even when 2% is measured

**Fix:** Use anti-fluency for verbose truths, but for **precise numerical truths**, use equally-precise distractors (0.326 vs 0.412, not 0.326 vs 0.25).

### For Understanding Model Errors

When models hallucinate physics constants, they tend toward:
- Round values (0.5, 1.0, 2.0)
- Simple fractions (1/4, 1/3, 1/2)
- Powers of 10

This is **not random** — it reflects training data frequency bias.

### For Functional Forms

The model's preference for polynomial kernels (r², 1/r) over logarithmic (-ln(r)) is particularly important because:
- 2D Green's function IS -ln(r)
- 3D Green's function IS 1/r
- The model conflates dimensions

## Relationship to Other Findings

This explains why:
- **Anti-fluency works**: Making distractors verbose removes the round-number advantage
- **Specific numbers fail**: 2%, 0.59%, 17% lose to round alternatives
- **-ln(r) fails badly**: Model strongly prefers polynomial forms

## Quantitative Evidence

Log-probability gaps for correct answers:

| Value | LP | Gap from Best |
|-------|-----|---------------|
| C_K = 1.5 (correct) | -9.1 | -1.0 below 0.5 |
| beta = 0.326 (correct) | -13.3 | -2.3 below 0.250 |
| R_f = 2% (correct) | -11.6 | -1.2 below 10% |
| kernel = -ln(r) (correct) | -26.4 | -15.9 below r² |

The kernel bias (-15.9 gap) is far stronger than numerical biases (~1-2 gap).

## Method

1. Generated length-matched answer options
2. Ranked by log-probability
3. Compared model's top choice to correct answer
4. Identified systematic patterns

## Files

- Discovery: `results/discoveries/novel_findings/round_number_bias.md`
- Related: `anti_fluency_distractor_strategy.md`, `unified_oracle_difficulty_theory.md`
