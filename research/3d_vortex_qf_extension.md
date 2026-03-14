# 3D Extension of Q_f Invariants

## Motivation

Having discovered that Q_f = ∫∫ ω(x)ω(y) f(|x-y|) dx dy is near-conserved in 2D Euler,
we explore the natural 3D generalization.

## 3D Formulation

For 3D incompressible Euler with vorticity ω(x,t), define:

**Q_f[ω] = ∫∫ ω(x)·ω(y) f(|x-y|) d³x d³y**

Note: This involves the dot product of vorticity vectors, since ω is a vector field in 3D.

## Connection to Known Invariants

### Helicity

Helicity is defined as:
H = ∫ u·ω d³x

Using the Biot-Savart law u(x) = (1/4π) ∫ ω(y) × (x-y) / |x-y|³ d³y, we can write:

H = ∫∫ ω(x)·[ω(y) × (x-y)] / (4π|x-y|³) d³x d³y

This is NOT directly a Q_f form because of the cross product structure.

However, for parallel vorticity (ω(x) ∥ ω(y)), helicity reduces to a Q_f-type integral.

### Energy

Kinetic energy:
E = (1/2) ∫ |u|² d³x = (1/8π) ∫∫ ω(x)·ω(y) / |x-y| d³x d³y

This IS a Q_f form with f(r) = 1/r !

So the 3D energy invariant corresponds to f(r) = 1/r.

## Hypothesis

If our 2D finding extends to 3D, then:

Q_f[ω] = ∫∫ ω(x)·ω(y) f(|x-y|) d³x d³y

might be near-conserved for various f(r), not just f(r) = 1/r.

### Candidate functions:
- f(r) = 1/r (energy - known exact)
- f(r) = e^(-r) (exponential screening)
- f(r) = 1/(r² + a²)^(1/2) (regularized Coulomb)
- f(r) = tanh(r/a) / r (interpolating)

## Vortex Filament Analog

For vortex filaments (concentrated vortex tubes), the discrete analog is:

Q_f = Σᵢ<ⱼ ΓᵢΓⱼ ∫∫ T_i(s)·T_j(t) f(|γᵢ(s) - γⱼ(t)|) ds dt

where:
- γᵢ(s) is the curve of filament i
- Γᵢ is its circulation
- T_i(s) is the unit tangent vector

For parallel filaments, this simplifies significantly.

## Challenges in 3D

1. **Vortex stretching**: In 3D, |ω| is NOT conserved (unlike 2D).
   Vortex stretching amplifies vorticity, potentially breaking Q_f conservation.

2. **Topology**: 3D vortex lines can link and knot, introducing
   topological constraints not present in 2D.

3. **Singularity**: The potential for finite-time blowup in 3D Euler
   could invalidate near-conservation near singularities.

## Test Plan

1. **Parallel vortex tubes**: Test Q_f conservation for initially
   parallel, straight vortex tubes (simplest 3D case).

2. **Vortex ring pair**: Two coaxial vortex rings with opposite circulations.
   Track Q_f during interaction and breakdown.

3. **Trefoil knot**: Single knotted vortex tube.
   Test whether Q_f changes as knot evolves.

## Connection to Millennium Problem

The Navier-Stokes millennium problem asks about existence and smoothness
of solutions in 3D. Key open question: does vorticity remain bounded?

If Q_f invariants exist in 3D, they would provide:
- New constraints on vorticity distribution
- Potential a priori bounds on ||ω||_Lp norms
- Additional conservation laws to constrain blowup scenarios

Specifically, if Q_exp = ∫∫ ω·ω e^(-|x-y|) d³x d³y is conserved,
it bounds how concentrated vorticity can become (exponential screening
penalizes nearby parallel vorticity).

## Status: THEORETICAL EXPLORATION

Numerical verification needed for 3D case.
