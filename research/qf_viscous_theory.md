# Theoretical Analysis: Q_f Decay Under Viscosity

## Setup

For 2D Navier-Stokes:
```
∂ω/∂t + u·∇ω = ν∇²ω
```

We want to compute dQ_f/dt where:
```
Q_f = ∫∫ ω(x)ω(y) f(|x-y|) dx dy
```

## Derivation

Taking the time derivative:
```
dQ_f/dt = ∫∫ [∂ω/∂t(x) ω(y) + ω(x) ∂ω/∂t(y)] f(|x-y|) dx dy
```

Using the Navier-Stokes equation:
```
∂ω/∂t = -u·∇ω + ν∇²ω
```

So:
```
dQ_f/dt = ∫∫ [-u(x)·∇ω(x) + ν∇²ω(x)] ω(y) f(r) dx dy
        + ∫∫ ω(x) [-u(y)·∇ω(y) + ν∇²ω(y)] f(r) dx dy
```

where r = |x-y|.

### Advection Term (Euler part)

The advection terms:
```
A = -∫∫ [u(x)·∇ω(x) ω(y) + ω(x) u(y)·∇ω(y)] f(r) dx dy
```

Integrating by parts on the first term:
```
-∫∫ u(x)·∇ω(x) ω(y) f(r) dx dy
= ∫∫ ω(x) ω(y) u(x)·∇_x f(r) dx dy  (using ∇·u = 0)
= ∫∫ ω(x) ω(y) u(x)·f'(r) (x-y)/r dx dy
```

Similarly for the second term. Combined:
```
A = ∫∫ ω(x) ω(y) f'(r)/r [u(x) - u(y)]·(x-y) dx dy
```

**Key observation**: When ω(x)ω(y) is large (both points in same vortex core),
u(x) ≈ u(y), so A ≈ 0. This explains near-conservation for Euler!

### Diffusion Term

The diffusion terms:
```
D = ν ∫∫ [∇²ω(x) ω(y) + ω(x) ∇²ω(y)] f(r) dx dy
```

Integrating by parts twice:
```
∫∫ ∇²ω(x) ω(y) f(r) dx dy = ∫∫ ω(x) ω(y) ∇²_x f(r) dx dy
```

For f(r) = f(|x-y|):
```
∇²_x f(r) = f''(r) + f'(r)/r  (in 2D)
```

So:
```
D = 2ν ∫∫ ω(x) ω(y) [f''(r) + f'(r)/r] dx dy
```

## Specific Cases

### f(r) = √r

```
f(r) = r^(1/2)
f'(r) = (1/2) r^(-1/2)
f''(r) = -(1/4) r^(-3/2)

f''(r) + f'(r)/r = -(1/4) r^(-3/2) + (1/2) r^(-3/2) = (1/4) r^(-3/2)
```

So:
```
dQ_√r/dt = (ν/2) ∫∫ ω(x) ω(y) r^(-3/2) dx dy + A
```

For small viscosity, A ≈ 0, giving:
```
dQ_√r/dt ≈ (ν/2) Q_{r^(-3/2)}
```

### f(r) = e^(-r)

```
f(r) = e^(-r)
f'(r) = -e^(-r)
f''(r) = e^(-r)

f''(r) + f'(r)/r = e^(-r) [1 - 1/r]
```

So:
```
dQ_exp/dt = 2ν ∫∫ ω(x) ω(y) e^(-r) [1 - 1/r] dx dy + A
```

## Numerical Verification

From our tests with two Gaussian vortices:

| f(r) | rel_change/ν | Theory |
|------|--------------|--------|
| √r | ~7 (constant) | ∝ Q_{r^(-3/2)} |
| e^(-r) | ~15 (slight decrease) | ∝ Q_{exp} × [1 - 1/r] |
| tanh(r) | increasing with ν | Complex dependence |

The constant scaling for √r suggests:
- Q_{r^(-3/2)} / Q_√r ≈ 14 for this initial condition
- This ratio is determined by the vorticity distribution

## Implications

### For Navier-Stokes Regularity

If we can show that:
```
dQ_f/dt ≤ C_f × ν × ||ω||²
```

then Q_f provides an a priori bound:
```
Q_f(t) ≤ Q_f(0) + C_f ν T ||ω||²_max
```

For bounded initial Q_f(0), this bounds how Q_f can grow.

Combined with the definition:
```
Q_f = ∫∫ ω(x)ω(y) f(|x-y|) dx dy
```

This could constrain vorticity concentration.

### Choosing Optimal f

The ideal f for regularity proofs would:
1. Have simple, controlled viscous decay
2. Penalize vorticity concentration (f(r) decreasing near r=0)
3. Be integrable at r=0

Candidates:
- f(r) = √r: Simple decay, but doesn't penalize concentration enough
- f(r) = e^(-r): Penalizes distant correlations, manageable decay
- f(r) = r / (r + a): Regularized, controlled behavior

## Connection to Known Results

### Beale-Kato-Majda

The BKM criterion states: If ||ω||_L∞ remains bounded, solution exists.

Q_f could provide bounds on ||ω||_Lp via:
```
||ω||²_L2 ≤ C × Q_{f_0}  for appropriate f_0
```

### Enstrophy Cascade

In 2D turbulence, enstrophy cascades to small scales. Q_f with
rapidly decaying f (like e^(-r)) is insensitive to this cascade,
making it more robust than enstrophy itself.

## Conclusions

1. **Q_f decays linearly with viscosity** for small ν
2. **√r shows the most predictable decay**: dQ_√r/dt ≈ 7ν × Q_{r^(-3/2)}
3. **This provides potential regularity bounds** via Q_f ≤ Q_f(0) + Cνt
4. **The decay rate is determined by a related functional** Q_g where g = f'' + f'/r
