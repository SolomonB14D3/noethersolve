# Dipole + Test Vortex Exact Invariant (2026-03-13)

## Discovery

**For a perfect vortex dipole (Γ₁ = -Γ₂) plus ONE test vortex ON THE SYMMETRY AXIS, the weighted distance sum Q = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ is exactly conserved:**

Q = const (frac_var ~ 10⁻¹⁵)

This holds for **any** test vortex strength, not just weak vortices.

## Critical Requirement: Symmetric Placement

The test vortex MUST be on the perpendicular bisector of the dipole (symmetry axis):

```
          ●  test vortex (Γ₃) on y-axis
          |
          |  ← symmetry axis
          |
   (+Γ)●──┼──●(-Γ)    ← perfect dipole on x-axis
```

**Asymmetric placement breaks exact conservation!**

## Verification

| Γ₃ (test vortex) | frac_var | Verdict |
|------------------|----------|---------|
| 0.001 | 1.15e-16 | EXACT |
| 0.01 | 2.42e-16 | EXACT |
| 0.1 | 4.99e-15 | EXACT |
| 0.5 | 2.98e-13 | EXACT |
| 1.0 | 9.99e-13 | EXACT |
| 2.0 | 8.92e-12 | EXACT |

All test vortex strengths give exact conservation (frac_var < 10⁻¹⁰).

## Physical Mechanism

For a dipole at symmetric positions, the formula simplifies:

```
Q = Γ₁Γ₂·r₁₂ + Γ₁Γ₃·r₁₃ + Γ₂Γ₃·r₂₃
  = (-Γ²)·r₁₂ + Γ·Γ₃·(r₁₃ - r₂₃)
```

Key observations:
1. **r₁₂ (dipole separation) is nearly constant** (frac_var ~ 10⁻¹³)
2. **r₁₃ and r₂₃ individually vary** (frac_var ~ 3%)
3. **But r₁₃ - r₂₃ ≈ 0** due to dipole symmetry

The weighted sum automatically exploits the symmetry: the test vortex's contribution Γ₃(r₁₃ - r₂₃) vanishes on average, leaving Q ≈ -Γ²·r₁₂ = const.

## Asymmetric Placement (2026-03-13 update)

When the test vortex is OFF the symmetry axis, the invariant breaks:

| Placement | Γ_test | frac_var | Verdict |
|-----------|--------|----------|---------|
| (0, 2) symmetric | 0.1 | 4.99e-15 | EXACT |
| (0.5, 2) off-axis | 0.1 | 9.38e-03 | Approximate |
| (1.0, 2) off-axis | 0.1 | 7.45e-03 | Approximate |
| (2.0, 2) off-axis | 0.1 | 2.91e-03 | Approximate |

The exact invariant requires r₁₃ = r₂₃ at all times (by symmetry).

## Extension to Multiple Test Vortices

With two or more test vortices, the invariant becomes approximate even with symmetric placement:

| Configuration | frac_var | Verdict |
|---------------|----------|---------|
| Dipole + 1 test (symmetric) | 4.99e-15 | EXACT |
| Dipole + 2 tests | 9.12e-04 | Approximate |
| Dipole + 3 tests | 1.09e-03 | Approximate |
| Dipole + 4 tests | 1.89e-03 | Approximate |

The second test vortex breaks the symmetry required for exact conservation.

## Relation to Linear Impulse

A perfect dipole has zero net circulation (Γ₁ + Γ₂ = 0), so:
- Linear impulse is conserved: P = Γ₁(y₁ - y₂) = const
- The dipole translates without changing separation
- The test vortex orbits the dipole but doesn't break the symmetry

## Status

- Numerical: VERIFIED (exact for single test vortex, approximate for multiple)
- Oracle: PENDING
- Formal proof: OPEN (conjecture: follows from dipole symmetry + momentum conservation)
