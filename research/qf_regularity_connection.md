# Q_f Invariants and Navier-Stokes Regularity

## The Millennium Problem

The Clay Mathematics Institute Millennium Prize asks:
> Do smooth solutions to 3D Navier-Stokes remain smooth for all time,
> or can singularities develop in finite time?

This document explores whether Q_f invariants could provide insight.

## The Regularity Problem

### 3D Navier-Stokes Equations
```
∂u/∂t + (u·∇)u = -∇p + ν∇²u
∇·u = 0
```

In terms of vorticity ω = ∇ × u:
```
∂ω/∂t + (u·∇)ω = (ω·∇)u + ν∇²ω
```

The term (ω·∇)u is **vortex stretching** - unique to 3D!

### The Beale-Kato-Majda Criterion

**Theorem (BKM, 1984)**: A smooth solution to 3D Euler/Navier-Stokes blows up
at time T if and only if:
```
∫₀ᵀ ||ω(·,t)||_{L^∞} dt = ∞
```

This means controlling ||ω||_∞ is equivalent to proving regularity.

## Q_f and Vorticity Bounds

### Key Question

Can Q_f provide bounds on ||ω||_{Lp} norms?

### Bounds from Q_f

Consider Q_f = ∫∫ ω(x)ω(y) f(|x-y|) d^n x d^n y in n dimensions.

**Observation 1**: For f(r) = 1 (constant on a bounded domain):
```
Q_1 = (∫ ω dx)² = Γ²
```
This is just circulation squared, which is conserved.

**Observation 2**: For f(r) = δ(r) (Dirac delta):
```
Q_δ = ∫ ω² dx = Ω
```
This is enstrophy, conserved in 2D but not in 3D.

**Observation 3**: For intermediate f, Q_f interpolates between these.

### Interpolation Inequalities

If f(r) decreases from f(0) to 0 as r → ∞, we can write:
```
Q_f ≤ f(0) × Q_δ = f(0) × Ω
```

This gives:
```
Ω ≥ Q_f / f(0)
```

If Q_f is conserved, this bounds enstrophy from below!

For upper bounds, consider:
```
Q_f ≥ f(R) × (∫ ω dx)² = f(R) × Γ²
```
where R is the domain diameter.

### The Concentration Bound

From our numerical experiments (test_qf_concentration.py):

For a Gaussian vortex with width σ and circulation Γ:
```
Q_r ∝ σ
Q_√r ∝ σ^0.5
||ω||_∞ ∝ 1/σ²
```

This gives:
```
||ω||_∞ ∝ 1 / Q_r²
||ω||_∞ ∝ 1 / Q_√r⁴
```

**If Q_r is conserved, ||ω||_∞ is bounded!**

Specifically:
```
||ω||_∞ ≤ C / Q_r(0)²
```

### The 2D Case (Where This Works)

In 2D, we have shown numerically that Q_r is approximately conserved.

This provides a bound:
```
||ω||_∞(t) ≤ C / Q_r(0)²
```

Combined with Gronwall-type arguments, this prevents blowup in 2D.

(This is consistent with the known regularity of 2D Euler/Navier-Stokes.)

### The 3D Challenge

In 3D, vortex stretching complicates everything:

1. **Vorticity can grow**: (ω·∇)u term amplifies vorticity
2. **Tubes stretch and thin**: As tubes stretch, ω_max increases
3. **Q_f is NOT conserved**: Our stretching tests show Q_f ∝ s²

However, our tests also showed:
- Q_{1/r} grows slower when stretching is accompanied by separation
- Energy (which is Q_{1/r} in 3D) is exactly conserved

### A Possible Approach

**Hypothesis**: In 3D, even though Q_f is not conserved, we might have:
```
dQ_f/dt ≤ C × ||ω||_L² × ||∇ω||_L²
```

Using Gronwall:
```
Q_f(t) ≤ Q_f(0) exp(C ∫₀ᵗ ||∇ω||_L² ds)
```

If we can bound ||∇ω||_L² (related to enstrophy), we bound Q_f.
If Q_f bounds ||ω||_∞, we get regularity.

## Numerical Experiment: 3D Vorticity Concentration

Let's analyze what happens in 3D when vorticity concentrates.

For a 3D vortex tube with:
- Core radius a
- Length L
- Circulation Γ
- Vorticity ω ~ Γ/(πa²)

Under stretching by factor s:
- Length: L → sL
- Core radius: a → a/√s (volume conservation)
- Vorticity: ω → sω

The quantities transform as:
```
||ω||_∞ → s × ||ω||_∞
Enstrophy → s × Enstrophy
Energy → (approximately) conserved
```

For Q_f with f(r) = 1/r (energy-like):
```
Q_{1/r} ~ ∫∫ ω(x)·ω(y) / |x-y| d³x d³y
```

Under stretching, if tubes stay at fixed lateral separation d:
```
Q_{1/r} ~ s² × Γ² × L² / d ~ s² × (const)
```

So Q_{1/r} grows as s². But energy is conserved!

**Resolution**: In real flows, stretching tubes must also separate:
```
d → √s × d₀
```
to conserve energy. This gives:
```
Q_{1/r} ~ s² / √s = s^{3/2}
```
which still grows, but slower.

## The Key Insight

In 2D:
- Q_f conservation → ||ω||_∞ bounds → regularity

In 3D:
- Q_f is NOT conserved due to stretching
- But energy (Q_{1/r}) IS conserved
- Energy conservation forces stretching + separation
- This provides SOME constraint on concentration

**Open Question**: Can we find a modified Q_f that:
1. Accounts for 3D vorticity direction (ω is a vector)
2. Remains conserved despite stretching
3. Still provides bounds on ||ω||_∞?

## Candidate: The Aligned Q_f

Consider:
```
Q_f^{aligned} = ∫∫ (ω(x)·ω(y))² f(|x-y|) d³x d³y
```

This weights by alignment of vorticity vectors.

For anti-parallel tubes, ω(x)·ω(y) < 0, reducing the integral.
For parallel stretching, alignment increases, but so does separation.

This might have better conservation properties.

## Connection to Helicity

Helicity in 3D:
```
H = ∫ u·ω d³x = ∫∫ ω(x)·[ω(y) × (x-y)] / (4π|x-y|³) d³x d³y
```

This is conserved for inviscid flow and measures vortex linking.

Note: H involves the cross product ω × (x-y), not just the dot product.

Could there be a "Q_f-helicity hybrid" that combines:
- The weighted distance structure of Q_f
- The topological information of helicity?

## Conclusions

1. **Q_f provides explicit vorticity bounds in 2D** via Q_r ∝ σ

2. **In 3D, Q_f is NOT conserved** due to vortex stretching

3. **Energy conservation partially constrains stretching** but doesn't prevent concentration

4. **A modified 3D Q_f might exist** that accounts for vortex orientation and provides bounds

5. **The search for such invariants** could be approached via:
   - Machine learning (extend our optimization to 3D)
   - Analytical construction (combine Q_f with helicity)
   - Numerical exploration (test many candidates in 3D)

## Status

This is speculative and exploratory. No rigorous claims are made about
solving the Navier-Stokes millennium problem. The purpose is to
document potential research directions suggested by the Q_f findings.
