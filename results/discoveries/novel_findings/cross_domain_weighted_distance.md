# Cross-Domain Analysis: Weighted Distance Sums (2026-03-13)

## Observation

The circulation-weighted distance sum Q = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ in vortex dynamics has a structural analog in gravitational N-body: Q = Σᵢ<ⱼ mᵢmⱼ rᵢⱼ.

## Comparison

| Domain | Weights | General Conservation | Special Cases |
|--------|---------|---------------------|---------------|
| Vortex | Γᵢ (±signs) | Nearly conserved | EXACT for dipole + test |
| Gravity | mᵢ (all +) | NOT conserved | PASS on figure-8 |

## Gravitational 3-body Results

| Configuration | frac_var | Verdict |
|---------------|----------|---------|
| figure-8 | 5.54e-04 | PASS |
| random_1 | 2.51e-01 | fail |
| random_2 | 1.75e-01 | fail |
| hierarchical | 5.21e-01 | fail |

## Why the Difference?

**Vortex dynamics:**
- Circulations can be positive or negative
- Dipoles (Γ₁ = -Γ₂) have zero net circulation
- Sign freedom creates exact cancellations (dipole + test vortex is EXACT)

**Gravitational dynamics:**
- All masses are positive
- No sign cancellation possible
- Only special symmetric configurations (figure-8) preserve the weighted sum

## Figure-8 Special Properties

For equal-mass figure-8 choreography:
- Zero total angular momentum (L = 0)
- Z₃ cyclic symmetry (bodies permute every 1/3 period)
- Perimeter Σrᵢⱼ nearly constant (frac_var ~ 5e-4)
- Mass-weighted sum Q = Σ mᵢmⱼ rᵢⱼ = m² × Σrᵢⱼ also conserved

## Structural Insight

The weighted distance sum is a "soft" invariant that:
- Works exactly when sign cancellation is available (vortex dipoles)
- Works approximately when symmetry is present (figure-8 choreography)
- Fails in generic asymmetric configurations

This suggests looking for other sign-cancellation opportunities in vortex dynamics and other symmetry-protected quantities in gravitational dynamics.

## Status

- Cross-domain comparison: VERIFIED
- Formal connection: OPEN (no known theorem linking these)
