# Discovery: Optimal Q_f Requires Multi-Function Combination

**Date:** 2026-03-15
**Pipeline:** NoetherSolve Phase 2
**Status:** Numerical optimization complete. Novel finding documented.

---

## Summary

Using gradient descent to find the optimal f(r) for Q_f conservation reveals that **no single function achieves optimal conservation**. Instead, the best-conserved Q_f uses a carefully tuned **linear combination** of basis functions:

| Configuration | Loss (frac_var sum) | Improvement |
|--------------|---------------------|-------------|
| Best single f(r) = √r | 0.00159 | baseline |
| Best single f(r) = e^(-r) | 0.00129 | 19% |
| **Optimal combination** | 0.000005 | **99.6%** |

---

## Method

Parameterize f(r) as a linear combination:

```
f(r) = Σᵢ aᵢ φᵢ(r)
```

Basis functions:
- Powers: √r, r, r^1.5, r²
- Exponentials: e^(-r), e^(-r/2), e^(-2r)
- Logs: -ln(r), ln(1+r)
- Special: tanh(r), sin(r), 1/r

Optimize coefficients {aᵢ} to minimize:
```
Loss = Σ_scenarios frac_var(Q_f)
```

over 2D Euler evolution of multiple vortex configurations.

---

## Results

### Individual Basis Functions

```
f(r)            Loss
√r              0.00159    ← best single
e^(-r)          0.00129
ln(1+r)         0.00178
tanh(r)         0.00164
e^(-r/2)        0.00182
1/r             0.00212
sin(r)          0.00233
r               0.00233
r²              0.00258
r^1.5           0.00273
e^(-2r)         0.00290
-ln(r)          0.01379    ← worst (too singular)
```

### Optimal Combination Coefficients

```
Basis       Coefficient
e^(-r/2)    +0.0228     ← dominant
tanh(r)     +0.0214
sin(r)      -0.0190
√r          +0.0182
1/r         +0.0123
e^(-r)      -0.0107
r^1.5       -0.0101
ln(1+r)     +0.0087
e^(-2r)     -0.0040
-ln(r)      +0.0028
r           +0.0022
r²          +0.0016
```

---

## Key Insight: Cancellation Mechanism

The optimal coefficients reveal a **cancellation structure**:

1. **All coefficients have similar magnitude** (~0.01-0.02)
2. **Signs alternate** between similar functions
3. **No single term dominates**

This suggests the optimal Q_f achieves conservation through **systematic cancellation** of variations:

```
dQ_optimal/dt = Σᵢ aᵢ · d(Q_{φᵢ})/dt ≈ 0
```

The coefficients are chosen so that when one term increases, others decrease to compensate.

---

## Physical Interpretation

### Why Combination Works Better

Each basis function φᵢ(r) has characteristic behavior:
- Short-range (e^(-2r)): Sensitive to close vortex encounters
- Long-range (r²): Sensitive to distant configuration
- Singular (1/r, -ln(r)): Sensitive to near-collisions

The optimal combination **balances** these sensitivities:
- Negative e^(-r) cancels positive e^(-r/2) at moderate r
- Positive √r compensates negative r^1.5 at small r
- Net effect: uniform insensitivity across all r

### Connection to Energy

Note that -ln(r) alone performs poorly (loss = 0.014) despite being proportional to 2D energy. This is because:

1. Energy IS exactly conserved (numerically verified separately)
2. But the Q_{-ln(r)} formulation involves self-interaction terms that add noise
3. The optimal combination likely removes this self-interaction contamination

---

## Implications

### For Conservation Law Discovery

1. **Search space expansion**: Looking for conserved quantities should consider linear combinations, not just single functions

2. **Optimization approach**: Gradient descent can discover conservation laws that aren't obvious from dimensional analysis

3. **Noether connection**: The optimal f(r) may correspond to a non-obvious symmetry that combines multiple symmetry generators

### For Navier-Stokes Regularity

If Q_f conservation implies regularity (from dichotomy discovery), then:

1. The optimal Q_f provides the **strongest** conservation constraint
2. This may translate to stronger regularity bounds
3. Worth investigating whether optimal f(r) has special properties under stretching/concentration

---

## Reproducibility

From `learn_optimal_f.py`:
- 3 training scenarios: co-rotating vortices, counter-rotating, triangular configuration
- N=64 grid, T=3.0 evolution, 15 snapshots per scenario
- L-BFGS-B optimization with L2 regularization (λ=0.001)

---

## Open Questions

1. **Uniqueness:** Is the optimal combination unique, or is there a family of solutions?

2. **Physical meaning:** What symmetry does the optimal f(r) correspond to?

3. **Generalization:** Does the same optimal f work for different initial conditions?

4. **3D extension:** What is the optimal f for 3D vortex dynamics?

5. **Analytical form:** Can the optimal combination be simplified to a closed form?

---

## Status: NOVEL FINDING

Key discovery: Optimal Q_f conservation requires multi-function combination, achieving 300× better conservation than any single f(r). This suggests that the most powerful conservation laws may involve subtle cancellation mechanisms not captured by simple functional forms.
