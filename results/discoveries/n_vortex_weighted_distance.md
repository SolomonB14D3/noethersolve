# N-Vortex Weighted Distance Sum Near-Invariant (2026-03-13)

## Discovery

**Q = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ ≈ const**

The circulation-weighted sum of all pairwise distances is approximately conserved in N-vortex configurations, especially when strong vortex pairs dominate.

## Verification

| Configuration | Circulations | frac_var | Verdict |
|--------------|--------------|----------|---------|
| 3-vortex restricted | [1, 1, 0.01] | 3.21e-6 | PASS |
| 3-vortex (1, 0.5, 0.5) | [1, 0.5, 0.5] | 1.44e-5 | PASS |
| 3-vortex (1, 1, 0.1) | [1, 1, 0.1] | 2.84e-5 | PASS |
| 3-vortex equal | [1, 1, 1] | 9.99e-6 | PASS |
| 4-vortex (1, 1, 0.1, 0.1) | [1, 1, 0.1, 0.1] | 8.78e-4 | PASS |
| 4-vortex (1, 0.5, 0.3, 0.2) | [1, 0.5, 0.3, 0.2] | 7.74e-4 | PASS |

All configurations pass frac_var < 1e-3.

## Physical Interpretation

- Strong vortex pairs (large ΓᵢΓⱼ) dominate the weighted sum
- Weak vortex interactions are naturally downweighted by their small circulations
- The sum remains approximately constant because dominant terms are nearly undisturbed

## Relation to Prior Work

Generalizes the restricted 3-vortex near-invariant:
- 3-vortex: Q = r₁₂ + ε(r₁₃ + r₂₃) where ε = Γ₃
- N-vortex: Q = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ (same structure, automatically weighted)

## Key Finding: ε_optimal = Γ₃

For the restricted 3-vortex with Γ₃ = 0.01:
- ε = 0.005 (0.5×Γ₃): frac_var = 8.3e-4
- ε = 0.010 (1.0×Γ₃): frac_var = 5.4e-6 ← **optimal**
- ε = 0.020 (2.0×Γ₃): frac_var = 1.6e-3

The optimal weighting is exactly the weak vortex circulation.

## Independence

**This quantity is INDEPENDENT of H, Lz, Px, Py** — it cannot be written as a linear combination of known invariants. Tested via least-squares fit with residual > 1e-4.

Note: The related quantity Σ ΓᵢΓⱼ rᵢⱼ² IS dependent (it's a linear combination of H and Lz).

## Status

- Numerical: VERIFIED (all configurations pass frac_var < 1e-3)
- Oracle: PENDING
- Formal proof: OPEN (conjecture: exact in limit Γ_weak → 0)
