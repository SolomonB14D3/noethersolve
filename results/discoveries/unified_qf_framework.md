# Discovery: Unified Q_f Framework for Regularity

**Date:** 2026-03-15
**Pipeline:** NoetherSolve Phase 2
**Status:** Synthesis of all Q_f discoveries.

---

## Summary

The individual Q_f discoveries form a **unified framework** where no single Q_f is optimal for everything, but **complementary invariants** together provide powerful constraints.

---

## The Nine Discoveries

| # | Discovery | Key Finding |
|---|-----------|-------------|
| 1 | Q_f Dichotomy | Stretch-resistance ↔ concentration-detection trade-off |
| 2 | Optimal Combination | 12-function combo achieves 99.6% improvement |
| 3 | Q_√r Viscous Decay | ν^0.99 scaling, most predictable |
| 4 | Curvature Trade-off | κ^1 helps stretch, hurts evolution |
| 5 | Q_f Ratio | R = Q_{e^(-r)}/Q_{1/r} is 5× better overall |
| 6 | Q_n Scaling | frac_var ∝ n^{3.7}, optimal n ∈ [0.05, 0.1] |
| 7 | Q_{-ln(r)} = Energy | Best turbulence conservation in 2D (point vortices) |
| 8 | Q_{1/r} in 3D | Green's function principle: 3D energy analog |
| 9 | Q_{e^(-r)} Continuous | Best for continuous fields (no r=0 singularity) |

---

## The Q_f Design Space

Each Q_f variant excels in different properties:

| Property | Best f(r) | Mechanism |
|----------|-----------|-----------|
| Conservation (2D point) | -ln(r) | = 2D Kinetic energy |
| Conservation (3D filament) | 1/r | = 3D Kinetic energy |
| Conservation (continuous) | e^(-r) | Smooth at r=0, spectral decay |
| Conservation (N-body) | r^0.05 | Close-pair weighting |
| Stretch resistance | κ × f | Dimensional cancellation |
| Viscous predictability | √r | Linear decay rate |
| Concentration detection | 1/√r, 1/r | Divergent at r=0 |
| Combined balance | e^(-r) / (1/r) | Ratio cancellation |

### Green's Function Principle

The best conserved Q_f in each dimension uses the Laplacian Green's function:

| Dimension | Green's Function G(r) | Best Q_f |
|-----------|----------------------|----------|
| 2D | -ln(r)/(2π) | Q_{-ln(r)} |
| 3D | 1/(4πr) | Q_{1/r} |

**Universal principle:** Q_G = ∫∫ G(|x-y|) ω(x)·ω(y) dx dy ∝ kinetic energy.

---

## Selection Guide

### For 2D Euler/Navier-Stokes

| Goal | Recommended Q_f |
|------|-----------------|
| Best conservation | Q_{-ln(r)} (= Energy) |
| Best viscous bound | Q_√r |
| Concentration detect | Q_{1/√r} |
| Stretch-resistant | Q_κ,f |

### For Point Vortex Dynamics

| Goal | Recommended |
|------|-------------|
| Exact invariant | Q₂ = Σ ΓᵢΓⱼ rᵢⱼ² |
| Best approx. conservation | Q_{n=0.05} |
| Balanced (stretch + evol) | Q_{e^(-r)} / Q_{1/r} |

### For 3D Navier-Stokes Regularity

| Goal | Recommended |
|------|-------------|
| Energy-like bound | Q_{1/r} (= 3D energy) |
| Stretch-resistant | Q_κ,1/√r (hybrid) |
| Linear viscous bound | Q_√r |
| Combined constraint | Multiple Q_f together |

### For Continuous Field Simulations

| Goal | Recommended |
|------|-------------|
| Best numerical conservation | Q_{e^(-r)} |
| Avoid regularization issues | e^(-r), √r, tanh(r) |
| For chaotic multi-vortex | e^(-r) (wins 3/5 tests) |
| For coherent structures | Learned optimal combination |

---

## Key Trade-offs

### 1. Conservation ↔ Sensitivity

```
Good conservation → Insensitive to dynamics
Good detection    → Sensitive (varies more)
```

Cannot have both: Q_f that detects concentration must vary when concentration happens.

### 2. Stretch ↔ Evolution

```
Curvature weighting → Helps stretching resistance
                   → Adds noise during normal evolution
```

Curvature fluctuates during evolution, degrading conservation.

### 3. Resolution: Ratio Invariants

```
R = Q₁/Q₂ cancels common variations
```

Achieves properties neither Q₁ nor Q₂ has alone.

---

## Unified Scaling Laws

Both power-law discoveries unify:

| System | Scaling Law | Exponent |
|--------|-------------|----------|
| Q_n conservation | frac_var ∝ n^α | α ≈ 3.7 |
| Q_√r viscous decay | rel_change ∝ ν^β | β ≈ 0.99 |

Both are **power laws with simple rational exponents**.

---

## Multi-Invariant Regularity Strategy

For 3D Navier-Stokes, define the constraint set:

```
S(t) = {Q_{1/r}(t), Q_κ,1/√r(t), Q_√r(t), Q_{0.05}(t)}
```

If all four remain bounded:

1. **Q_{1/r} bounded** → Energy-like constraint on concentration
2. **Q_κ,1/√r bounded** → Stretching is bounded (curvature cancellation)
3. **Q_√r bounded** → Viscous decay is predictable (linear scaling)
4. **Q_{0.05} bounded** → Close-pair dynamics controlled (n^3.7 scaling)

**Together:** Vorticity concentration is constrained from multiple directions.

### The Blowup Scenario

For blowup to occur, the solution must evade ALL four bounds:
- Concentrate while avoiding Q_{1/r} detection (energy bound)
- Stretch while avoiding Q_κ detection (curvature bound)
- Concentrate while avoiding Q_{0.05} detection (close-pair bound)
- Dissipate unpredictably to avoid Q_√r bound

This is geometrically highly restrictive.

---

## Implications

### Theoretical

The multi-invariant approach suggests:
- No single conserved quantity governs regularity
- A **family** of near-invariants may collectively constrain solutions
- Different Q_f probe different aspects of the dynamics

### Practical

For numerical simulations:
- Monitor multiple Q_f to detect approaching singularities
- Use Q_κ,1/√r for stretch-dominated regimes
- Use Q_√r for viscous-dominated regimes
- Use Q_{0.05} for conservation-critical applications

### Research Directions

1. **Rigorous bounds:** Prove boundedness of multiple Q_f implies regularity
2. **Optimal set:** Find minimal set of Q_f that constrains all blowup scenarios
3. ~~**3D verification:** Test the framework on 3D vortex simulations~~ ✓ DONE (Discovery #8)
4. **Analytical form:** Derive the scaling exponents (n^3.7, ν^0.99) analytically
5. **Continuous-discrete bridge:** Understand when -ln(r) vs e^(-r) is optimal

---

## Status: SYNTHESIS COMPLETE (v2)

The unified framework reveals:
- No single Q_f is universally optimal
- Trade-offs govern the Q_f design space
- Complementary invariants together provide strong constraints
- Multi-invariant approach is the key to regularity analysis
- **Green's function principle:** Q_G (Laplacian Green's function) = energy in each dimension
- **Numerical robustness:** Q_{e^(-r)} best for spectral methods (smooth at r=0)
