# Triplet Invariant: A False Alarm

## Summary

**NEGATIVE RESULT**: The apparent "triplet area invariant" discovered in initial explorations was a numerical artifact, not a new conservation law.

## What Appeared to be Discovered

In `beyond_known_math.py`, we computed:

```
T_full = Σ_{all i,j,k} Γi Γj Γk × Area(ijk)
```

This showed `frac_var = 6.54e-07`, seemingly indicating excellent conservation.

## Why It Was Wrong

The sum over ALL permutations of (i,j,k) is **identically zero** by symmetry:

1. **Symmetric weight**: Γi × Γj × Γk is the same for all 6 permutations of (i,j,k)
2. **Antisymmetric area**: signed_area(i,j,k) = -signed_area(j,i,k)
3. **Result**: 3 positive + 3 negative contributions = 0

Verified numerically:
```
Triplet (0,1,2) contributions:
  (0,1,2): area=+0.433, weight=6.0, contrib=+2.598
  (0,2,1): area=-0.433, weight=6.0, contrib=-2.598
  (1,0,2): area=-0.433, weight=6.0, contrib=-2.598
  (1,2,0): area=+0.433, weight=6.0, contrib=+2.598
  (2,0,1): area=+0.433, weight=6.0, contrib=+2.598
  (2,1,0): area=-0.433, weight=6.0, contrib=-2.598
Total: 0.0000000000
```

The small non-zero `frac_var` was just numerical noise around zero.

## The Actual Invariant (Not Conserved)

The "ordered" version:

```
T_ordered = Σ_{i<j<k} Γi Γj Γk × Area(ijk)
```

is NOT conserved. Numerical tests show frac_var ~ 0.5 to 100, compared to Q_ln frac_var ~ 10⁻⁷.

## Analytical Verification

Taking the time derivative:

```
dT/dt = Σ_{i<j<k} Γi Γj Γk × d/dt[Area(ijk)]
```

where d/dt[Area(ijk)] depends on velocities of ALL vortices (not just i,j,k). The sum does NOT cancel in general.

## Lessons Learned

1. **Check for trivial zeros**: If a quantity has mean ≈ 0, investigate why
2. **Symmetry analysis first**: Verify the combination isn't identically zero by construction
3. **Multiple test configurations**: The first configuration may be special

## Implications

The Q_f pairwise family remains the only known infinite family of conservation laws for 2D point vortex dynamics. No genuine higher-order (triplet, quadruplet, etc.) invariants have been found.

This suggests:
- The pairwise structure is special, not a first step in a hierarchy
- Higher-order structures might require different physical systems (3D, continuous fields)
- The particle relabeling symmetry only generates pairwise invariants

## Files

- `research/investigate_triplet.py` - Detailed investigation
- `research/true_higher_order.py` - Systematic search for real higher-order invariants
- `research/beyond_known_math.py` - Original (misleading) exploration
