# Parallel Dipole Sum Invariant (2026-03-13)

## Discovery

**For N identical parallel vortex dipoles oriented perpendicular to their line of centers, the sum of internal separations is exactly conserved:**

Σᵢ rᵢ = const (frac_var ~ 10⁻¹⁶)

where rᵢ is the separation distance of dipole i.

## Configuration

```
Dipole 1        Dipole 2        Dipole 3
  (+)             (+)             (+)
   |               |               |   ← vertical orientation
  (-)             (-)             (-)

 ←———————————————————————————————————→  horizontal separation axis
```

Each dipole has vortices at (cx, ±d/2) where cx is the center x-position.

## Key Properties

- Individual dipole separations can vary by 20-30%
- But their sum is EXACTLY constant (machine precision)
- Works for 2, 3, 4, 5+ dipoles
- Different initial internal separations OK!

## Verification

| Configuration | Max Individual Variation | Sum frac_var |
|--------------|-------------------------|--------------|
| 2 parallel dipoles | 21.08% | 1.89e-16 ✓ |
| 3 parallel dipoles | 23.33% | 1.71e-16 ✓ |
| 4 parallel dipoles | 29.49% | 1.24e-16 ✓ |
| 5 parallel dipoles | 30.88% | 2.25e-16 ✓ |

## Boundary Conditions

Exact conservation (frac_var < 10⁻¹⁰) requires:
1. Equal dipole strengths: |Γ₊| = |Γ₋| same for all dipoles
2. Parallel orientation: all dipoles aligned (same θ)
3. Perpendicular geometry: dipoles oriented ⊥ to line of centers

What BREAKS it:
- Unequal strengths [1,-1] vs [2,-2]: frac_var ~ 1e-2
- Anti-parallel [-1,1] vs [1,-1]: frac_var ~ 6e-1
- Tilted dipoles (θ ≠ 90°): frac_var ~ 1e-2

## Physical Interpretation

- Dipole-dipole interactions can stretch/compress individual dipoles
- But the total "dipole length budget" is exactly conserved
- Related to conservation of total linear impulse perpendicular to motion
- Each dipole acts as a momentum carrier; total momentum redistributes but conserves

## Geometric Meaning (DERIVED 2026-03-13)

**The dipole sum invariant is linear impulse conservation in disguise!**

For a vertical dipole at position x with separation d:
- Vortex 1: +Γ at (x, +d/2)
- Vortex 2: -Γ at (x, -d/2)

Linear impulse contribution:
```
Px = Σ Γᵢ yᵢ = Γ·(d/2) + (-Γ)·(-d/2) = Γd
```

For N identical parallel dipoles with equal Γ:
```
Px_total = Γ · Σₖ dₖ
```

Since Px is exactly conserved and Γ is constant:
```
Σₖ dₖ = Px_total / Γ = const
```

**This is NOT an independent invariant** — it follows directly from linear impulse conservation applied to the dipole geometry.

Numerical verification:
| N dipoles | |Σₖ dₖ - Px/Γ| max |
|-----------|-------------------|
| 2 | 8.88e-16 |
| 3 | 1.78e-15 |
| 4 | 3.55e-15 |

The invariant is EXACTLY Px/Γ to machine precision.

## Status

- Numerical: VERIFIED (exact to machine precision for 2-5 dipoles)
- Oracle: PENDING (trivial - just Px conservation)
- Formal proof: COMPLETE (follows from Px = Γ·Σdₖ)
