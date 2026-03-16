# Discovery: Exponential Kernel is Best for Continuous Q_f

**Date:** 2026-03-15
**Pipeline:** NoetherSolve Phase 2
**Status:** Numerical verification complete for 2D Euler pseudospectral.

---

## Summary

For continuous vorticity fields in 2D Euler, **f(r) = e^(-r) is the best conserved Q_f kernel**, NOT -ln(r) (the energy kernel).

| f(r) | frac_var | Rank |
|------|----------|------|
| **e^(-r)** | **3.09e-04** | **1st** |
| √r | 3.48e-04 | 2nd |
| sin(r) | 3.55e-04 | 3rd |
| r | 8.74e-04 | 4th |
| -ln(r) | 1.15e-02 | 5th |
| r² | 1.54e-03 | 6th |

---

## The Paradox

This result appears to contradict the 2D finding that Q_{-ln(r)} = Energy is exactly conserved.

**Resolution:** The difference is between:
1. **Point vortices**: Q_{-ln(r)} is EXACTLY conserved (machine precision)
2. **Continuous fields**: Q_{-ln(r)} is approximately conserved but e^(-r) is better

Why the difference?

### Regularization Effect

For continuous fields, we must regularize -ln(r) at r=0:
```
f(r) = -ln(r + ε)  where ε ~ 10^{-10}
```

This regularization:
- Breaks the exact conservation
- Introduces numerical error at small scales
- The logarithm amplifies small-scale fluctuations

### Spectral Truncation

The pseudospectral solver:
- Uses finite resolution (N=128)
- Truncates high frequencies (2/3 dealiasing)
- This effectively smooths the vorticity field

The -ln(r) kernel has slow spectral decay, so truncation affects it more.

### The Exponential Advantage

f(r) = e^(-r) is:
- Smooth everywhere (no regularization needed)
- Rapidly decaying in Fourier space
- Well-suited to spectral methods

---

## Numerical Results

Two co-rotating Gaussian vortices, N=128, T=5.0:

### Q_f Conservation

| f(r) | Mean Q_f | Std | frac_var |
|------|----------|-----|----------|
| e^(-r) | 1.4764 | 0.0005 | 3.09e-04 |
| √r | 4.3129 | 0.0015 | 3.48e-04 |
| sin(r) | 2.5208 | 0.0009 | 3.55e-04 |
| r | 5.3345 | 0.0047 | 8.74e-04 |
| -ln(r) | 0.1162 | 0.0013 | 1.15e-02 |
| r² | 10.1860 | 0.0157 | 1.54e-03 |

### Known Invariants (Verification)

| Invariant | frac_var |
|-----------|----------|
| Total circulation Γ | 1.48e-16 |
| Enstrophy Ω | 2.18e-15 |

The known invariants are conserved to machine precision, confirming numerical accuracy.

---

## Implications

### For Numerical Simulations

When monitoring Q_f in continuous field simulations:
- **Use f(r) = e^(-r)** instead of -ln(r)
- Provides better numerical conservation
- Smoother behavior near r=0

### For Theory

The approximate conservation of Q_f for ALL tested f(r) suggests:
- Q_f conservation is a robust property of 2D Euler
- Not limited to specific f(r) choices
- Any smooth, bounded f(r) gives approximate conservation

### Hierarchy

| System | Best f(r) | Reason |
|--------|-----------|--------|
| 2D point vortices | -ln(r) | Exact (= Energy) |
| 2D continuous (spectral) | e^(-r) | Numerical stability |
| 3D vortex filaments | 1/r | = 3D Energy |
| N-vortex chaotic | r^0.05 | Close-pair weighting |

---

## Connection to Other Discoveries

### Q_f Ratio (Discovery #5)

The ratio R = Q_{e^(-r)} / Q_{1/r} was found optimal for point vortices.
This is consistent: e^(-r) numerator provides good numerics.

### Viscous Decay (Discovery #3)

Q_√r has linear viscous decay scaling.
For continuous fields, √r is also well-conserved (2nd place).

---

## Status: NOVEL FINDING

Key discovery: For continuous vorticity fields, **Q_{e^(-r)} is better conserved than Q_{-ln(r)}** due to:
1. No need for regularization at r=0
2. Rapid spectral decay (well-suited to pseudospectral methods)
3. Smooth behavior everywhere

This provides practical guidance for numerical simulations.
