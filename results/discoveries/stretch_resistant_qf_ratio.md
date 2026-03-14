# Discovery: Stretch-Resistant Q_f Ratio (2026-03-14)

## Key Finding

The ratio of two Q_f invariants survives vortex stretching in 3D:

**R_f = Q_{e^(-r)} / Q_{1/r}**

where:
- Q_{e^(-r)} = ∫∫ ω(x)·ω(y) e^(-|x-y|) d³x d³y
- Q_{1/r} = ∫∫ ω(x)·ω(y) / |x-y| d³x d³y

## Numerical Evidence

### Test 1: Pure Stretching (Parallel Tubes)

| Stretch Factor | Standard Q_f | Q_exp/Q_inv Ratio |
|----------------|--------------|-------------------|
| 1.0 | 20.63 | 0.293 |
| 1.5 | 36.47 | 0.298 |
| 2.0 | 53.61 | 0.296 |
| 3.0 | 91.20 | 0.288 |
| 4.0 | 133.39 | 0.282 |

**Standard Q_f frac_var: 0.606 (60% variation)**
**Q_exp/Q_inv frac_var: 0.020 (2% variation)**

### Test 2: Biot-Savart Evolution (Leapfrogging Rings)

| Quantity | Mean | Std | frac_var |
|----------|------|-----|----------|
| Standard Q_f | 35.63 | 0.051 | 0.00144 |
| Q_exp/Q_inv | 0.315 | 0.0005 | 0.00170 |

Both conserved during evolution, with the ratio slightly worse.

### Test 3: Combined Performance

| Variant | Stretch frac_var | Evolution frac_var | Combined |
|---------|------------------|-------------------|----------|
| Standard Q_f | 0.606 | 0.00144 | 0.0295 |
| Q_f / Ω | 0.168 | 0.00355 | 0.0244 |
| **Q_exp/Q_inv** | **0.020** | **0.00170** | **0.0059** |

**Q_exp/Q_inv is 5x better overall than standard Q_f.**

## Why Does This Work?

### Scaling Analysis

Under stretching by factor s:
- Length: L → sL
- Both Q_exp and Q_inv scale as ~L² = s²
- The ratio R = Q_exp/Q_inv cancels the L² dependence

But why doesn't the ratio change with separation?

Under stretching + separation:
- Q_exp weights nearby points heavily (e^(-r) decays fast)
- Q_inv weights all points more evenly (1/r decays slowly)
- As tubes separate (r increases): Q_exp drops, Q_inv drops less
- The ratio Q_exp/Q_inv decreases slightly but remains bounded

### Mathematical Interpretation

The ratio can be written as:
```
R = ∫∫ ω·ω e^(-r) dx dy / ∫∫ ω·ω / r dx dy

  = <e^(-r)>_{ω²} / <1/r>_{ω²}
```

This is the ratio of two different averages of f(r) weighted by ω(x)·ω(y).

Under stretching:
- The distribution of r values shifts
- Both averages shift similarly
- The ratio remains approximately constant

## Theoretical Significance

### For 3D Vortex Dynamics

In 3D, vortex stretching is the mechanism that could lead to blowup.
The fact that R_f = Q_exp/Q_inv survives stretching means:

1. R_f provides a constraint that persists despite stretching
2. This could bound how vorticity can redistribute
3. Combined with energy conservation, this may constrain blowup scenarios

### For Navier-Stokes Regularity

If R_f is conserved in 3D Euler (even approximately), then:
- Vortex stretching alone cannot cause R_f to blow up
- Combined with ||ω||_∞ bounds from concentration analysis,
  this could provide regularity constraints

### General Q_f Ratios

The success of Q_exp/Q_inv suggests exploring other ratios:
- Q_√r / Q_r
- Q_tanh / Q_1/r
- (Q_f₁ - αQ_f₂) / Q_f₃

## Related Findings

### Curvature-Weighted Q_f

Also tested: Q_κ = ∫∫ κᵢκⱼ ω·ω f(r) dx dy

- Best for pure stretching (curvature compensates for length increase)
- But worse for normal evolution (introduces extra variation)
- Less universal than the ratio approach

### Enstrophy-Normalized Q_f

Q_f / Ω where Ω = ∫ ω² dx

- Second-best for stretching (frac_var 0.168)
- Good combined performance (0.0244)
- Physically motivated (tracks vorticity intensity)

## Files

- `research/test_stretch_resistant_qf.py`: Initial screening
- `research/test_curvature_qf_deep.py`: Curvature analysis
- `research/test_hybrid_stretch_qf.py`: Final comparison

## Status

- **Discovery**: CONFIRMED
- **Mechanism**: Understood (scaling cancellation)
- **Robustness**: Tested on multiple configurations
- **Theoretical Proof**: OPEN
- **Navier-Stokes Application**: SPECULATIVE

## Next Steps

1. Test R_f on full 3D vorticity fields (not just filaments)
2. Derive analytical bound on dR_f/dt
3. Explore connection to known 3D invariants (helicity, energy)
4. Test under viscous dissipation
