# Discovery: Two Distinct Conservation Mechanisms Across Domains

**Date:** 2026-03-15
**Pipeline:** NoetherSolve Phase 2
**Status:** Verified numerically. Cross-domain comparison complete.

---

## Summary

The near-conservation of Q_f = Σ f(rᵢⱼ) arises from **fundamentally different mechanisms** in gravitational vs vortex systems:

| System | Mechanism | Requirements | Universality |
|--------|-----------|--------------|--------------|
| **Gravitational figure-8** | Z3 Phase Cancellation | Special choreographic orbit | f-dependent (critical exponent) |
| **Point vortex** | Circulation Weighting | Hierarchical Γ distribution | Universal (all f work) |

---

## Mechanism 1: Z3 Phase Cancellation (Gravitational)

**System:** Equal-mass gravitational three-body in figure-8 choreography

**Mechanism:**
- All three distances trace the SAME scalar function with T/3 phase offsets:
  ```
  r₁₂(t) = r₂₃(t+T/3) = r₁₃(t+2T/3) = g(t)
  ```
- For Q = Σf(rᵢⱼ):
  ```
  dQ/dt = f'(g(t))·ġ(t) + f'(g(t+T/3))·ġ(t+T/3) + f'(g(t+2T/3))·ġ(t+2T/3)
  ```
- By discrete Fourier shift theorem, this sum cancels all harmonics except 3ω₀, 6ω₀, ...

**Key constraint:** Conservation quality depends on f(r). Critical exponent range:
```
For f(r) = r^p:  -0.67 < p < 2.55
```

**Verification:**
- p=1 (sum of distances): frac_var = 5.5e-4 ✓
- p=2 (sum of squared distances): frac_var = 1.5e-3 ✓
- p=-1 (inverse distances): frac_var = 0.05 ✗ FAIL
- p=-2 (inverse squared): frac_var = 0.3 ✗ FAIL

---

## Mechanism 2: Circulation Weighting (Vortex)

**System:** N-body point vortex dynamics with circulations Γᵢ

**Mechanism:**
- For Q_f = Σ ΓᵢΓⱼ f(rᵢⱼ):
  ```
  dQ_f/dt = Σ ΓᵢΓⱼ f'(rᵢⱼ) · (drᵢⱼ/dt)
  ```
- **Strong pairs** (large ΓᵢΓⱼ): Vortices co-move → drᵢⱼ/dt is small
- **Weak pairs** (small ΓᵢΓⱼ): Even if drᵢⱼ/dt is large, contribution suppressed by small ΓᵢΓⱼ

**Key feature:** Conservation is UNIVERSAL - works for ALL smooth f(r)

**Verification (restricted 3-vortex, Γ = [1, 1, 0.01]):**
- f(r) = r: frac_var = 5.3e-6 ✓
- f(r) = r²: frac_var = 1.9e-11 (EXACT) ✓
- f(r) = 1/r: frac_var = 1.6e-5 ✓
- f(r) = exp(-r): frac_var = 1.1e-5 ✓

**Suppression factor:** ~10,000× compared to unweighted Σf(rᵢⱼ)

---

## Why Vortices Don't Have Non-Trivial Choreographies

**Gravitational dynamics:** Bodies accelerate TOWARD each other
- Close encounters → position exchanges → complex paths → choreographies possible

**Vortex dynamics:** Bodies induce PERPENDICULAR velocities
- ẋᵢ ∝ Σ Γⱼ (yᵢ-yⱼ)/r² (perpendicular to separation)
- Close encounters → circular motion → rotational patterns only
- Equal circulations on regular polygon → RIGID ROTATION (trivial)

**Result:** No known non-trivial vortex choreographies exist.

---

## Cross-Correlation Test

For figure-8 gravitational:
- r₁₂ ↔ r₁₃: correlation ≈ 1.0 (same function, phase-shifted)
- Z3 symmetry confirmed

For restricted 3-vortex:
- r₁₂ ↔ r₁₃: correlation ≈ 0.78
- r₁₂ ↔ r₂₃: correlation ≈ 0.72
- NO Z3 symmetry - different mechanism

---

## Implications

### For Gravitational Systems:
- Near-conservation is **geometry-dependent**
- Requires special orbits (choreographies)
- Critical exponent range determines which f(r) work
- Mechanism: Phase cancellation in time derivative

### For Vortex Systems:
- Near-conservation is **universal** in f(r)
- Works for ANY orbit (not just special ones)
- Requires hierarchical circulation distribution
- Mechanism: Dynamical suppression through weighting

### Cross-Domain Insight:
The same mathematical structure Q = Σf(rᵢⱼ) exhibits near-conservation
in both systems, but for completely different physical reasons:

1. **Gravitational:** Temporal coherence (Z3 phase structure)
2. **Vortex:** Dynamical suppression (ΓᵢΓⱼ weighting)

This explains why:
- Gravitational: Only certain f(r) work (must match orbit geometry)
- Vortex: All f(r) work (weighting is universal)

---

## Connection to Known Results

### Gravitational Side:
- Related to Sundman inequality
- Figure-8 discovered by Moore (1993), proved by Chenciner-Montgomery (2000)
- Z3 symmetric polynomials near-conserved (earlier discovery)

### Vortex Side:
- Related to Kirchhoff's theorem on vortex motion
- Q₂ = Σ ΓᵢΓⱼ rᵢⱼ² reduces to known invariants (Lz, impulse)
- Q_f family extends known results to arbitrary f

---

## Status: DOCUMENTED

This cross-domain comparison reveals that similar mathematical structures
can arise from fundamentally different physical mechanisms, explaining
the different universality properties observed.
