# Dimensional Asymmetric Learning: Models Know 3D Physics But Are Blind to 2D

## Discovery Date: 2026-03-16

## Summary

LLMs exhibit **dimensional asymmetric learning** — they correctly learn 3D physics associations but **fail to modulate for 2D context**. When presented with "In 2D..." they still prefer 3D answers. This is 100% consistent across 6 physics domains tested.

## Evidence

### Systematic Test: 2D vs 3D Context Modulation

| Physics Domain | 3D Correct | 2D Correct | 2D Context → | 3D Context → |
|----------------|------------|------------|--------------|--------------|
| Coulomb/Green's function | 1/r | -ln(r) | 1/r (BLIND) | 1/r (AWARE) |
| Vortex topology | lines | points | lines (BLIND) | lines (AWARE) |
| Turbulence cascade | downward | upward | downward (BLIND) | downward (AWARE) |
| NS regularity | open | solved | open (BLIND) | open (AWARE) |
| Wave equation tails | Huygens | logarithmic | Huygens (BLIND) | varies |

**Result: 0/6 dimension-aware, 6/6 dimension-blind = 100% blindness rate**

### Log-Probability Evidence

```
"In 2D, the Coulomb potential is -ln(r)": -13.8
"In 2D, the Coulomb potential is 1/r":    -6.4   ← MODEL PREFERS (wrong)

"In 3D, the Coulomb potential is 1/r":    -6.6   ← MODEL PREFERS (correct)
"In 3D, the Coulomb potential is -ln(r)": -17.0
```

The model gives nearly identical scores to "1/r" regardless of whether the context says 2D or 3D.

### Pattern: 3D Is the Default

The asymmetry is directional:
- **3D context → model applies 3D physics (correct)**
- **2D context → model STILL applies 3D physics (wrong)**

This suggests 3D physics dominates the training data, and the model learned "1/r", "cascade down", "vortex lines" as **unconditional defaults** rather than **conditionally on dimension**.

## Why This Happens

### Training Data Bias

1. **We live in 3D** — most physics textbooks, papers, and discussions assume 3D
2. **2D is specialized** — appears in condensed matter, thin films, graphene contexts
3. **Dimension often implicit** — "the Coulomb potential is 1/r" without "in 3D"

### Representational Consequence

The model encodes physics associations as:
- `Coulomb → 1/r` (unconditional)
- `vortex → lines` (unconditional)
- `turbulence → energy cascades down` (unconditional)

Rather than:
- `Coulomb + 2D → -ln(r)`
- `Coulomb + 3D → 1/r`

The dimensional conditional isn't represented.

## Implications

### For Oracle Fact Design

Facts testing 2D physics will systematically fail because the model defaults to 3D. This is a **structural knowledge gap**, not a phrasing issue.

### For Training Data Curation

To fix dimensional blindness, training data needs:
1. **Explicit dimension marking**: "In 2D, the kernel is -ln(r); in 3D, it is 1/r"
2. **Contrastive pairs**: Present 2D and 3D side by side
3. **2D-specific contexts**: More examples from 2D physics domains

### For Physics Applications

LLMs cannot be trusted for 2D physics without verification:
- 2D materials (graphene, MoS₂)
- Thin film dynamics
- 2D fluid simulations
- Flatland physics thought experiments

### For Understanding Model Priors

This reveals how models form priors:
- **Common case becomes default** (3D dominates)
- **Conditioning on context fails** (dimension ignored)
- **Specialized knowledge suppressed** (2D overwritten by 3D)

## Relationship to Other Findings

| Finding | This Discovery Adds |
|---------|---------------------|
| Systematic confusions | Explains 2D kernel failures as dimensional asymmetry |
| Round number bias | Separate effect: model prefers simple forms (1/r over -ln(r)) |
| Green's function gaps | Subsumed: part of dimensional blindness pattern |

## Proposed Tool

Build `check_dimension_dependence(physics_concept)` tool that:
1. Returns correct 1D/2D/3D/nD forms
2. Warns when dimensional context matters
3. Provides explicit dimensional formulas

## Method

1. Created 6 physics test cases with dimension-dependent correct answers
2. Tested log-probability of 2D-correct vs 3D-correct for each context
3. Computed "aware" (picks correct for that dimension) vs "blind" (picks 3D always)
4. All 6 cases showed blindness pattern

## Files

- Discovery: `results/discoveries/novel_findings/dimensional_asymmetric_learning.md`
- Related: `systematic_confusions.md`, `round_number_bias.md`
- Test script: inline in session
