# Mathematical Structure of Q_f Invariants

## The Q_f Hierarchy

We have established numerically that:

```
Q_f[ω] = ∫∫ ω(x)ω(y) f(|x-y|) d²x d²y
```

is approximately conserved for various f(r). This document explores the
mathematical structure underlying this phenomenon.

## Known Exact Invariants

### 1. Circulation (Kelvin's Theorem)
```
Γ = ∫∫ ω d²x = const
```
This corresponds to f(r) = 1 (constant), but with only one copy of ω.

### 2. Energy (Hamiltonian)
```
E = (1/2) ∫∫ ω(x)ω(y) G(|x-y|) d²x d²y
```
where G(r) = -ln(r)/(2π) is the Green's function for ∇².

This is Q_{-ln(r)/(2π)}.

### 3. Angular Momentum
```
Lz = ∫∫ (x × u)·ẑ d²x = ∫∫ ω(x) |x|² d²x
```
Related to Q_f but with one copy of ω weighted by |x|².

### 4. Center of Vorticity
```
X_c = ∫ x ω d²x / ∫ ω d²x
```
Conserved (uniform translation).

### 5. Casimir Invariants
```
C[g] = ∫∫ g(ω) d²x
```
for any function g. These arise from the Lie-Poisson structure.

## Why is Q_f Near-Conserved?

### The Euler Equation Structure

The 2D Euler equation:
```
∂ω/∂t + u·∇ω = 0
```

This is a transport equation: vorticity is advected by the flow.

The velocity is determined by vorticity via:
```
u = ∇ × (ψ ẑ),  ∇²ψ = -ω
```

### Time Derivative of Q_f

```
dQ_f/dt = 2 ∫∫ ω(x) (∂ω/∂t)(y) f(|x-y|) d²x d²y

        = -2 ∫∫ ω(x) (u(y)·∇_y ω(y)) f(|x-y|) d²x d²y
```

Integrating by parts on y:
```
dQ_f/dt = 2 ∫∫ ω(x) ω(y) u(y)·∇_y f(|x-y|) d²x d²y

        = 2 ∫∫ ω(x) ω(y) f'(r)/r u(y)·(y-x) d²x d²y
```

where r = |x-y|.

### The Cancellation Mechanism

The key insight is that the integral has antisymmetric structure.

Exchanging x ↔ y in the second term:
```
∫∫ ω(x) ω(y) f'(r)/r u(y)·(y-x) d²x d²y
+ ∫∫ ω(y) ω(x) f'(r)/r u(x)·(x-y) d²x d²y
```

This gives:
```
dQ_f/dt = ∫∫ ω(x) ω(y) f'(r)/r [u(y)-u(x)]·(y-x) d²x d²y
```

**Key observation**: When ω(x)ω(y) is large:
1. Both x and y are in vortex cores
2. If in the SAME vortex core, u(x) ≈ u(y) → contribution vanishes
3. If in DIFFERENT vortex cores, the separation is large and f'(r)/r is small

This is why Q_f is approximately conserved!

### Exact Conservation Condition

Q_f is exactly conserved if and only if:
```
∫∫ ω(x) ω(y) f'(r)/r [u(y)-u(x)]·(y-x) d²x d²y = 0
```

for all solutions ω(x,t).

For f(r) = -ln(r), we have f'(r) = -1/r, so:
```
dQ_f/dt = -∫∫ ω(x) ω(y) (1/r²) [u(y)-u(x)]·(y-x) d²x d²y
```

This vanishes due to the specific relationship between u and ω via
the Biot-Savart law. (Standard energy conservation proof.)

## The Function Space of f

### Which f Work?

Based on numerical evidence, good f(r) share properties:
1. Smooth (at least C²)
2. Bounded or slowly growing as r → ∞
3. Well-behaved at r = 0 (at most log divergence)

### The Linear Space of Near-Invariants

If Q_{f₁} and Q_{f₂} are near-invariants, so is Q_{αf₁ + βf₂}.

This means the space of good f forms a vector space!

Our optimization found the optimal element of this space (for the
given basis functions) to be approximately:

```
f_opt(r) ≈ 0.023 e^(-r/2) + 0.021 tanh(r) - 0.019 sin(r)
         + 0.018 √r + 0.012/r - 0.011 e^(-r) + ...
```

## Spectral Interpretation

### Fourier Transform of f

Let f̂(k) = ∫ f(r) e^{-ikr} dr be the Fourier transform.

Then Q_f can be written in Fourier space as:
```
Q_f = ∫ |ω̂(k)|² f̂(k) dk
```

Different f(r) correspond to different spectral weightings!

### Spectral Cascade and Conservation

In 2D turbulence:
- Enstrophy cascades to small scales (high k)
- Energy cascades to large scales (low k)

The near-conservation of Q_f suggests the weighted spectrum f̂(k)|ω̂(k)|²
is nearly constant in time, for appropriate f̂.

## Lie-Algebraic Interpretation

### The Vorticity Poisson Bracket

The 2D Euler equations have Lie-Poisson structure:
```
{F, G} = ∫ ω [δF/δω, δG/δω] d²x
```

where [·,·] is the Jacobian of the two variational derivatives.

### Q_f as Approximate Casimirs

Exact Casimirs satisfy {Q, H} = 0 for all H.

The Q_f invariants are "approximate Casimirs" - they nearly commute
with the Hamiltonian:
```
{Q_f, H} ≈ 0
```

This suggests Q_f might be related to deformations of the Lie algebra
or asymptotic conservation laws.

## Connection to Noether's Theorem

### Symmetries and Conservation

Noether's theorem: continuous symmetries → conservation laws.

Known symmetries of 2D Euler:
- Time translation → Energy (Q_{-ln(r)})
- Space translation → Linear momentum
- Rotation → Angular momentum
- Area-preserving diffeomorphisms → Casimirs

### New Symmetries?

The existence of additional Q_f suggests there might be:
1. Approximate symmetries (almost commute with time evolution)
2. Hidden exact symmetries we haven't identified
3. Asymptotic symmetries (emerge in certain limits)

## Open Questions

1. **Exact vs Approximate**: Is there a characterization of which f give
   exact conservation vs approximate conservation?

2. **Universal f**: Is there an optimal f that works across all initial
   conditions, or is the best f problem-dependent?

3. **Physical Meaning**: What physical quantity does Q_{e^(-r)} represent?
   (Energy has a clear physical interpretation; what about Q_{tanh}?)

4. **3D Extension**: In 3D, the structure is different (vortex stretching).
   Do similar near-invariants exist?

5. **Turbulence Statistics**: Can Q_f be used to constrain the statistics
   of 2D turbulence?

## Conclusion

The Q_f family represents a rich mathematical structure lying between
exact conservation laws (energy) and general functions of vorticity
(not conserved). Understanding this structure could provide:

- New constraints on fluid dynamics
- Insights into Euler/Navier-Stokes regularity
- Better numerical methods that preserve more structure
