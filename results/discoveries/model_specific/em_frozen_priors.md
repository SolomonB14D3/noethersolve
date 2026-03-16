# Evidence: Electromagnetic Frozen Priors in LLM Oracle

## Summary

Oracle testing reveals a significant **knowledge gap** in electromagnetic
conservation laws - even basic energy/momentum conservation fails, and
obscure quantities (zilch, chirality) are completely unknown.

## Oracle Results

### Baseline (No Adapter)

| Fact | Margin | Status |
|------|--------|--------|
| E² alone not conserved | +? | ✓ PASS |
| Lipkin's zilch discovered 1964 | +? | ✓ PASS |
| **Energy U = ∫(E²+B²)** | **-4.08** | **✗ FAIL** |
| **Momentum ∫(E×B)** | **-5.35** | **✗ FAIL** |
| Chirality C = ½[E·∇×E + B·∇×B] | -0.60 | ✗ FAIL |
| Helicity H = ∫A·B | -2.02 | ✗ FAIL |
| Zilch tensor conservation | -11.63 | ✗ FAIL |
| Super-energy | -9.94 | ✗ FAIL |
| Circular polarization chirality | -10.04 | ✗ FAIL |
| Chevreton 1964 | -0.07 | ✗ FAIL |

**Pass rate: 2/10 (20%)**

### With Physics-Supervised Adapter

| Metric | Baseline | With Adapter |
|--------|----------|--------------|
| Pass rate | 2/10 | 1/10 |
| Mean margin | -3.89 | -63.82 |
| Min margin | -11.63 | -336.51 |

**Margin delta: -59.93 → KNOWLEDGE GAP MODE**

## Interpretation

### What the Model Knows

1. **Historical facts**: Correctly identifies Lipkin 1964
2. **Negative knowledge**: Knows E² alone is NOT conserved

### What the Model Doesn't Know

1. **Basic EM conservation** (energy U, momentum P) - FAILS
2. **Intermediate** (helicity H) - FAILS
3. **Obscure** (zilch Z, chirality C, super-energy S) - FAILS badly

### Why This Matters

The model fails on **even basic energy conservation** in electromagnetism.
This is surprising because:
- Energy conservation is fundamental physics
- Poynting's theorem is taught in every EM course
- The phrasing "∫(E² + B²) is exactly conserved" is standard

The fact that **all EM conservation laws fail** (not just obscure ones)
suggests this is not frozen priors on specific quantities, but rather
a broader gap in EM knowledge representation.

### Knowledge Gap vs Frozen Prior

- **Frozen Prior**: Model knows Energy but not Chirality
- **Knowledge Gap**: Model knows neither Energy nor Chirality

Our evidence shows a **knowledge gap** - the model's EM training
didn't include strong representations of conservation laws in
mathematical form.

## Implications

### For Noethersolve Project

1. EM invariants are a fruitful domain - many unknowns to discover
2. Need to generate EM-specific training data
3. Oracle can't help here without domain-specific fine-tuning

### For LLM Physics Understanding

1. Mathematical physics representations are weak
2. Historical facts (dates, names) are stronger than physics structure
3. Conservation law knowledge may be phrasing-dependent

## Numerical Verification

We verified these quantities are exactly conserved (frac_var < 10⁻⁶):
- Energy
- Optical Chirality
- Helicity
- Super-energy

So the ground truth is clear - these ARE conserved. The oracle fails
because it lacks this knowledge, not because the physics is wrong.

## Files

- `problems/em_zilch.yaml`: Problem definition
- `problems/em_zilch_facts.json`: Test facts
- `research/maxwell_zilch.py`: Numerical verification
