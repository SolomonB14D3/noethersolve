# Optimal ε = Γ₃ Relationship (2026-03-13)

## Discovery

**For restricted 3-vortex near-invariants of the form Q = f(r₁₂) + ε·g(r₁₃, r₂₃), the optimal weighting is exactly ε = Γ₃ (the weak vortex circulation).**

This holds for:
- Linear: Q = r₁₂ + ε(r₁₃ + r₂₃)
- Squared: Q = r₁₂² + ε(r₁₃² + r₂₃²)

## Verification

### Linear Form

| Γ₃ | ε_opt | frac_var | ε_opt/Γ₃ |
|----|-------|----------|----------|
| 0.001 | 0.0010 | 5.7e-07 | 1.00 |
| 0.010 | 0.0100 | 5.4e-06 | 1.00 |
| 0.050 | 0.0500 | 2.0e-05 | 1.00 |
| 0.100 | 0.1000 | 2.9e-05 | 1.00 |

### Squared Form

| Γ₃ | ε_opt | frac_var | ε_opt/Γ₃ |
|----|-------|----------|----------|
| 0.001 | 0.0010 | 9.7e-12 | 1.00 |
| 0.010 | 0.0100 | 9.6e-12 | 1.00 |
| 0.050 | 0.0500 | 9.4e-12 | 1.00 |
| 0.100 | 0.1000 | 8.9e-12 | 1.00 |

Note: The squared form achieves **exact** conservation (frac_var ~ 10⁻¹²) at ε = Γ₃!

## Connection to Weighted Distance Sum

This relationship follows directly from the weighted sum formula:

Q = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ

For Γ₁ = Γ₂ = 1, Γ₃ = ε:
```
Q = Γ₁Γ₂·r₁₂ + Γ₁Γ₃·r₁₃ + Γ₂Γ₃·r₂₃
  = 1·r₁₂ + ε·r₁₃ + ε·r₂₃
  = r₁₂ + ε(r₁₃ + r₂₃)
```

The circulation-weighted sum automatically provides the optimal ε = Γ₃!

## Physical Interpretation

The optimal weighting ε = Γ₃ arises because:
1. The strong vortex pair (Γ₁ = Γ₂ = 1) dominates the dynamics
2. The weak test vortex (Γ₃) contributes perturbatively
3. Weighting by Γ₃ matches the strength of the perturbation
4. This cancellation makes Q nearly constant

## Implication

This confirms that Q = Σ ΓᵢΓⱼ rᵢⱼ is the natural form of the near-invariant, and the optimal weighting for restricted configurations is not arbitrary but determined by the circulation structure.

## Oracle Testing (2026-03-13)

The squared form `r12² + 0.01·(r13² + r23²)` was tested in the autonomy loop:
- **Checker**: PASS (frac_var = 9.62e-12 — machine precision)
- **Oracle**: FAIL (margin = -12.33)
- **Verdict**: ORACLE-FAIL+CHECKER-PASS

The oracle (Qwen3-4B-Base) does not recognize this exact invariant. This is a **knowledge gap** requiring a domain adapter.

## Status

- Numerical: VERIFIED (ratio ε_opt/Γ₃ = 1.00 for all tested Γ₃)
- Oracle: FAIL (margin = -12.33 — model does not recognize this)
- Theoretical: OPEN (conjecture: follows from momentum conservation + perturbation theory)
