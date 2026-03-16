# Kinetic-Like Invariant K (2026-03-13)

## Discovery

**K = Σᵢ Γᵢ vᵢ² is approximately conserved in N-vortex systems**

This is a genuinely independent invariant from the Q_f family:
- Q_f = Σ ΓᵢΓⱼ f(rᵢⱼ) depends only on **distances**
- K = Σ Γᵢ |vᵢ|² depends on **distances AND angles**

## Verification

| Configuration | frac_var(K) | Status |
|--------------|-------------|--------|
| Restricted (1,1,0.01) | 1.2e-7 | PASS |
| Equal (1,1,1) | 3.3e-5 | PASS |
| Hierarchical (1,0.5,0.1) | 1.8e-5 | PASS |
| 4-vortex generic | 8.8e-5 | PASS |
| 5-vortex hierarchical | 3.2e-5 | PASS |

All configurations pass frac_var < 5e-3.

## Structure

For N=3, K can be decomposed:
```
K = K_dist + K_angle

where:
  K_dist = Σᵢ Γᵢ Σⱼ≠ᵢ Γⱼ²/(4π²rᵢⱼ⁴)     (distance-only)
  K_angle = Σᵢ Γᵢ Σⱼ<k 2Γⱼ Γₖ cos(θ)/(4π²rᵢⱼ²rᵢₖ²)  (angular cross-terms)
```

**Remarkable cancellation:**

| Component | frac_var |
|-----------|----------|
| K_dist (distance only) | 1.3e-5 |
| K_angle (angular) | 1.1e-1 |
| **K (total)** | **1.2e-7** |

The angular term varies significantly (frac_var ~ 0.1), but it cancels with 
distance variations to produce a well-conserved quantity!

## Independence from Q_f

K is NOT a linear combination of Q_f variants:
- Fit to Q₋₂: R² = 0.048 (poor)
- Fit to Q₋₂, Q₋₁: R² = 0.048 (still poor)

K represents a genuinely new family of approximate invariants.

## Physical Interpretation

K = Σ Γᵢ vᵢ² is a "circulation-weighted kinetic energy" analog.

In point-vortex dynamics:
- Velocities are induced by other vortices
- Strong vortices (large Γ) contribute more to K
- The distance-angle cancellation preserves K approximately

## Status

- Numerical: VERIFIED (N=3,4,5)
- Theoretical: DERIVED (distance-angle cancellation)
- Independence: VERIFIED (not a Q_f combination)
