# Continuous Q_f Hypothesis: Extension to Euler Vorticity Fields

## Discovery Context

In point-vortex dynamics, we discovered that Q_f = Σᵢ<ⱼ ΓᵢΓⱼ f(rᵢⱼ) is near-conserved
for ANY smooth function f(r), not just the known exact invariants:
- f(r) = ln(r) → Hamiltonian H (exactly conserved)
- f(r) = r² → reduces to Lz (exactly conserved)

## Continuous Analog Hypothesis

For a continuous 2D vorticity field ω(x,t), we propose:

**Q_f[ω] = ∫∫ ω(x) ω(y) f(|x-y|) dx dy**

is approximately conserved for solutions of the 2D Euler equations, for any smooth f.

### Known Cases

1. **f(r) = -ln(r)/2π**: This gives the kinetic energy
   E = (1/4π) ∫∫ ω(x) ω(y) ln|x-y| dx dy
   EXACTLY conserved.

2. **f(r) = r²**: This gives
   Q₂ = ∫∫ ω(x) ω(y) |x-y|² dx dy
   = 2 ∫ ω(x)|x|² dx · ∫ ω(y) dy - 2|∫ ω(x) x dx|²
   = 2 Γ · I - 2|P|²
   where Γ = total circulation, I = moment of inertia, P = impulse.
   Since Γ, I, P are all conserved → Q₂ is EXACTLY conserved.

3. **f(r) = 1**: This gives
   Q₁ = ∫∫ ω(x) ω(y) dx dy = Γ²
   EXACTLY conserved (trivial).

### Novel Predictions

For non-trivial f(r), our hypothesis predicts near-conservation:

- f(r) = r: Linear distance weighting
- f(r) = √r: Square root weighting
- f(r) = e^(-r): Exponential decay
- f(r) = sin(r): Oscillatory

These should have small dQ_f/dt for typical 2D Euler flows.

## Physical Interpretation

Why might this work?

1. **Strong pairs dominate**: In the point-vortex case, strong vortex pairs
   (large ΓᵢΓⱼ) have nearly constant separation, so f(rᵢⱼ) ≈ const.

2. **Continuous analog**: For continuous vorticity, regions of high ω(x)ω(y)
   product (same-sign vortex cores interacting) should have slowly-varying
   separation, making ∫∫ ω(x)ω(y) f(|x-y|) dx dy nearly constant.

3. **The weighting ω(x)ω(y) suppresses contributions** from weak vorticity
   regions where distances vary rapidly.

## Test Cases

### Case 1: Merging Vortex Pair
- Two equal co-rotating vortices
- Track Q_f over merger process
- Expect small fractional variation for smooth f

### Case 2: Kelvin-Helmholtz Instability
- Shear layer rollup
- High Re (inviscid limit)
- Compare frac_var for different f

### Case 3: 2D Turbulence
- Random vorticity field evolving
- Long-time statistics of Q_f
- Enstrophy cascade context

## Numerical Implementation

```python
def compute_Qf(omega, f, grid):
    """
    Compute Q_f = ∫∫ ω(x) ω(y) f(|x-y|) dx dy

    omega: 2D array of vorticity
    f: function r -> R
    grid: (dx, dy) grid spacing
    """
    dx, dy = grid
    N, M = omega.shape

    Q = 0.0
    for i in range(N):
        for j in range(M):
            if abs(omega[i,j]) < 1e-10:
                continue
            for k in range(N):
                for l in range(M):
                    if abs(omega[k,l]) < 1e-10:
                        continue
                    r = sqrt((i-k)**2 * dx**2 + (j-l)**2 * dy**2)
                    if r > 0:
                        Q += omega[i,j] * omega[k,l] * f(r) * dx**2 * dy**2
    return Q
```

Note: Efficient implementation uses FFT convolution.

## Connection to Casimir Invariants

The 2D Euler equations have infinitely many Casimir invariants:
C_g = ∫ g(ω) dx

for any function g. These express that vorticity is advected.

Q_f is NOT a Casimir invariant (it involves two spatial points, not one).
It is structurally different - a two-point correlation integral.

If Q_f is approximately conserved, it represents a NEW class of quasi-invariants
in 2D Euler dynamics, distinct from the Casimir family.

## Theoretical Derivation (Sketch)

Time derivative:
dQ_f/dt = ∫∫ [∂ω/∂t(x) ω(y) + ω(x) ∂ω/∂t(y)] f(|x-y|) dx dy

Using Euler: ∂ω/∂t = -u·∇ω

dQ_f/dt = -∫∫ [u(x)·∇ω(x) ω(y) + ω(x) u(y)·∇ω(y)] f(|x-y|) dx dy

Integrating by parts on ∇ω:
= ∫∫ [ω(x) ω(y) (u(x) - u(y))·∇_x f(|x-y|)] dx dy

The key is: when ω(x) and ω(y) are both large (same vortex core),
u(x) ≈ u(y), so the velocity difference is small.

This makes dQ_f/dt small, explaining the near-conservation.

## Status: HYPOTHESIS - NEEDS NUMERICAL VERIFICATION
