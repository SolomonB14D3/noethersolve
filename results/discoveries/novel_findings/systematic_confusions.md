# Systematic Confusions: Where Models Consistently Mistake Related Concepts

## Discovery Date: 2026-03-16

## Summary

LLMs exhibit **systematic confusions** between related physics concepts. These are not random errors — they reveal structural gaps in how physics relationships are encoded.

## Evidence

### 1. Dimension-Blind Kernels

| Question | Correct | Model Chooses | Error Type |
|----------|---------|---------------|------------|
| 2D vortex kernel | -ln(r) | r | Ignores Green's function |
| 3D vortex kernel | 1/r | r | Ignores Green's function |

**Pattern:** Model defaults to linear `r` for both dimensions, ignoring:
- 2D Laplacian Green's function: -ln(r)
- 3D Laplacian Green's function: 1/r

This explains why Q_f kernel facts fail so badly (-15.9 gap to r²).

### 2. Noether Symmetry Asymmetry

| Symmetry | Correct Conservation | Model Chooses |
|----------|---------------------|---------------|
| Translation | Momentum | Energy (CONFUSED) |
| Time | Energy | Energy (CORRECT) |

**Pattern:** Model knows time→energy but confuses translation→momentum with translation→energy. The space-time duality of Noether's theorem is not properly encoded.

### 3. Viscosity One-Way Knowledge

| Equation Type | Correct | Model Chooses |
|---------------|---------|---------------|
| Ideal (inviscid) | Euler | Euler (CORRECT) |
| Viscous | Navier-Stokes | Euler (CONFUSED) |

**Pattern:** Model knows "Euler = ideal" but doesn't know "viscous = NS". This asymmetry suggests:
- "Euler equation" appears frequently with "ideal fluid"
- "Navier-Stokes" appears in Millennium Problem context, not viscous fluid context

### 4. 2D Turbulence Cascade Collapse

| Cascade | Correct Quantity | Model Chooses |
|---------|-----------------|---------------|
| Inverse | Energy | Vorticity (WRONG) |
| Forward | Enstrophy | Vorticity (WRONG) |

**Pattern:** Model collapses both cascades to "vorticity" — the most salient turbulence keyword — missing the crucial energy/enstrophy distinction that defines 2D turbulence.

## Implications

### For Understanding Model Physics Knowledge

These confusions reveal **representational priorities**:
1. Simple forms preferred (r over -ln(r))
2. Famous associations dominate (Euler, vorticity)
3. Asymmetric learning (one direction learned, reverse not)
4. Keyword conflation (related concepts → most common term)

### For Training Data Curation

To fix these confusions, training data needs:
1. **Explicit dimension marking**: "In 2D, the kernel is -ln(r); in 3D, it is 1/r"
2. **Bidirectional Noether**: "Translation → momentum" AND "momentum → translation symmetry"
3. **Contrastive pairs**: "Euler: no viscosity; NS: with viscosity"
4. **Cascade disambiguation**: "Energy cascades UP; enstrophy cascades DOWN"

### For Oracle Design

These confusions predict where oracle facts will fail:
- Any fact distinguishing 2D vs 3D kernels
- Noether facts about spatial symmetries
- NS vs Euler distinctions
- 2D turbulence cascade direction

## Anti-Fluency Rescue Results

**UPDATE:** Anti-fluency testing reveals most confusions are fluency artifacts, not true gaps:

| Confusion | Fluent Margin | Anti-F Margin | Result |
|-----------|---------------|---------------|--------|
| 2D kernel = -ln(r) | -13.1 | +13.1 | RESCUED |
| Translation → momentum | +0.7 | +20.4 | Already knew |
| Viscous = NS | -8.8 | +9.4 | RESCUED |
| Inverse cascade = energy | -6.7 | +19.0 | RESCUED |

**Key insight:** The model DOES know these distinctions. The "confusion" appears only when fluent wrong answers are available. The model's internal representation distinguishes these concepts correctly.

## Relationship to Other Findings

| Finding | This Discovery Adds |
|---------|---------------------|
| Round number bias | Explains preference for r over -ln(r) in fluent context |
| Anti-fluency rescue | WORKS here — confusions are fluency-masked knowledge |
| Hidden knowledge | These ARE hidden knowledge, rescued by anti-fluency |

## Method

1. Identified pairs of related concepts
2. Tested which concept model prefers
3. Classified errors as CONFUSED (picked related concept) or OTHER
4. Identified systematic patterns

## Files

- Discovery: `results/discoveries/novel_findings/systematic_confusions.md`
- Related: `round_number_bias.md`, `unified_oracle_difficulty_theory.md`
