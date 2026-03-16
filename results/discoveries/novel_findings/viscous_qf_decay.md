# Q_f Decay Under Viscous Dissipation

## Discovery

Q_f invariants decay approximately linearly with viscosity ν in Navier-Stokes evolution. Different f(r) have vastly different sensitivity to viscous effects, with tanh(r) being most robust.

## Key Results

### 1. Inviscid Conservation (ν = 0)

| f(r) | frac_var | Status |
|------|----------|--------|
| tanh(r) | 1.73e-04 | ✓ Best conserved |
| e^(-r) | 6.82e-04 | ✓ Well conserved |
| √r | 7.79e-04 | ✓ Well conserved |
| -ln(r) | 2.46e-02 | ✗ Numerical issues |

Note: -ln(r) has singularity at r=0 that causes numerical problems on discrete grids.

### 2. Viscous Decay Rates

For Navier-Stokes with viscosity ν, the relative change in Q_f scales approximately linearly:

```
|Q_f(T) - Q_f(0)| / |Q_f(0)| ≈ c_f × ν × T
```

Decay coefficients c_f:

| f(r) | c_f (decay coefficient) |
|------|-------------------------|
| -ln(r) | ~500 (very sensitive) |
| e^(-r) | ~20 (moderate) |
| √r | ~7 (robust) |
| tanh(r) | ~3 (most robust) |

### 3. Detailed Scaling

| ν | tanh(r) rel_change | e^(-r) rel_change | √r rel_change |
|---|-------------------|-------------------|---------------|
| 0.001 | 0.03% | 2.1% | 0.76% |
| 0.005 | 1.2% | 9.9% | 3.2% |
| 0.010 | 3.7% | 17.6% | 6.8% |
| 0.020 | 10.3% | 29.2% | 14.4% |
| 0.050 | 31.1% | 51.1% | 35.6% |

## Physical Interpretation

### Why Different Decay Rates?

The dissipation rate depends on the gradient structure:
```
dQ_f/dt = -ν ∫ (dissipation functional of ω, ∇ω, f)
```

For functions f(r) that are:
- **Rapidly varying** near r=0: High sensitivity to small-scale structure → fast decay
- **Slowly varying**: Low sensitivity to gradients → slow decay

### Connection to Energy Dissipation

Energy dissipation in 2D Navier-Stokes:
```
dE/dt = -ν Ω  where Ω = ∫ |∇ω|² dx (palinstrophy)
```

The Q_f decay rates provide additional constraints beyond just energy.

## Implications

### 1. Robust Invariants for Turbulence

For turbulent flows with effective viscosity, use:
- tanh(r) or e^(-r) for slowly-decaying quantities
- Avoid -ln(r) due to numerical sensitivity

### 2. Viscosity Estimation

If Q_f decay is measured, viscosity can be estimated:
```
ν ≈ |ΔQ_f| / (c_f × Q_f × T)
```

### 3. Subgrid Modeling

Q_f decay rates could inform subgrid-scale models in LES:
- Model should reproduce correct Q_f decay rates
- Different f(r) test different aspects of SGS model

## Experimental Setup

- Grid: 128×128
- Initial condition: Two co-rotating Gaussian vortices
- Evolution time: T = 5.0
- Viscosities tested: ν ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05}

## Mathematical Framework

For Navier-Stokes:
```
∂ω/∂t + u·∇ω = ν Δω
```

Taking time derivative of Q_f:
```
dQ_f/dt = ∫∫ [∂ω(x)/∂t × ω(y) + ω(x) × ∂ω(y)/∂t] f(|x-y|) dx dy
```

The inviscid terms (advection) cancel to give conservation.
The viscous terms give:
```
dQ_f/dt = ν ∫∫ [Δω(x)ω(y) + ω(x)Δω(y)] f(|x-y|) dx dy
        = -2ν ∫∫ ∇ω(x)·∇_y[ω(y)f(|x-y|)] dx dy
```

## Connection to Other Results

- Complements concentration detection: tanh(r) is both concentration-blind AND viscosity-robust
- Explains why e^(-r) works well in numerical tests: bounded and moderate viscous decay
- Suggests optimal f(r) should balance conservation accuracy vs. robustness

## Open Questions

1. Can we derive exact viscous decay rates analytically?
2. Is there f(r) with zero viscous decay? (Would require special structure)
3. How do decay rates change in 3D Navier-Stokes?
