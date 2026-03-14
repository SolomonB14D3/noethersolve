# Discovery: Electromagnetic Conservation Laws

## Overview

Implemented a complete EM domain for the noethersolve pipeline:
1. **em_checker.py** - Spectral Maxwell solver with conservation verification
2. **em_zilch_facts.json** - 12 oracle test facts (standard + obscure invariants)
3. **em_adapter.npz** - Domain-specific adapter (v2: 4/12 pass)

## Numerical Verification (em_checker.py)

### Invariants Verified (frac_var < 5e-3)

| Invariant | Formula | frac_var | Status |
|-----------|---------|----------|--------|
| **Energy** | U = ½∫(E² + B²)d³x | 7.35e-7 | ✓ EXACT |
| **Chirality** | C = ½∫[E·(∇×E) + B·(∇×B)]d³x | 1.05e-6 | ✓ EXACT |
| **Helicity** | H = ∫A·B d³x | 5.83e-7 | ✓ EXACT |
| **Super-energy** | S = ∫[(∇×E)² + (∇×B)²]d³x | 1.34e-6 | ✓ EXACT |
| E² alone | ∫E²d³x | 3.07e-1 | ✗ NOT conserved |
| B² alone | ∫B²d³x | 2.91e-1 | ✗ NOT conserved |

### Initial Conditions Tested
- `circular+`: Right-handed circular polarization
- `circular-`: Left-handed circular polarization
- `linear_x`: Linear x-polarization
- `standing`: Standing wave (E² and B² oscillate, sum conserved)

## Oracle Results

### Baseline (No Adapter)
| Metric | Value |
|--------|-------|
| Pass rate | 1/12 (8.3%) |
| Mean margin | -11.04 |
| Min margin | -31.04 |

**Key finding**: Even basic EM knowledge fails:
- Energy conservation: margin -7.46 (FAIL)
- Momentum: margin -9.40 (FAIL)
- Poynting's theorem: margin -4.10 (FAIL)

### With EM Adapter v2
| Metric | Baseline | Adapter v2 |
|--------|----------|------------|
| Pass rate | 1/12 (8.3%) | 4/12 (33.3%) |
| Mean margin | -11.04 | -10.28 |
| Δ margin | — | +0.76 |

**Flipped facts**:
- em05_super_energy: -3.7 → +6.2
- em07_lipkin_year: +2.3 → +35.3
- em10_chevreton_year: -3.2 → +3.9
- em11_poynting_conserved: -4.1 → +8.0

**Close to flipping**:
- em03_chirality: -4.1 (was -10.7)
- em12_helicity_gauge: -4.6 (was -23.2)

### Diagnostic Progression
1. **Initial**: KNOWLEDGE_GAP (with physics_supervised adapter, Δ = -280)
2. **After EM adapter v1**: FIXABLE_BIAS (Δ = +1.2)
3. **After EM adapter v2**: FIXABLE_BIAS (Δ = +0.76)

## Key Findings

### 1. Severe EM Knowledge Gap in Base Model
The Qwen/Qwen3-4B-Base model lacks even basic EM conservation knowledge:
- Doesn't recognize ∫(E² + B²) as conserved
- Doesn't know Poynting's theorem ∂u/∂t + ∇·S = 0
- Historical dates (Lipkin 1964) are known, physics structure is not

### 2. Domain-Specific Adapters Required
The physics_supervised adapter (trained on vortex dynamics) makes EM worse:
- Margin delta: -280 (KNOWLEDGE_GAP mode)
- EM physics has different structure than fluid dynamics

### 3. Obscure Invariants Are Learnable
With only 30 training examples:
- Super-energy (Chevreton) flipped
- Chevreton 1964 date flipped
- Chirality improving toward flip

## Technical Details

### Maxwell Solver
- Spectral (Fourier) method in 3D
- RK4 time integration, dt=0.02
- Grid: 48³, L=2π
- Dealiasing: 2/3 rule

### Optical Chirality (Zilch Z⁰)
```
C = (ε₀/2) E·(∇×E) + (1/2μ₀) B·(∇×B)
```
- Discovered by Lipkin (1964)
- Rediscovered by Tang & Cohen (2010) as "optical chirality"
- Measures handedness of light (±1 for circular, 0 for linear)

### Super-Energy (Chevreton Tensor)
```
S = ∫ [(∇×E)² + (∇×B)²] d³x
```
- Related to conformal invariance of Maxwell equations
- Trace of Chevreton super-energy tensor

## References

1. Lipkin, D.M. (1964). "Existence of a new conservation law in electromagnetic
   theory." J. Math. Phys. 5, 696.

2. Tang, Y. & Cohen, A.E. (2010). "Optical chirality and its interaction with
   matter." Phys. Rev. Lett. 104, 163901.

3. Chevreton, M. (1964). "Sur le tenseur de superénergie du champ
   électromagnétique." Nuovo Cimento 34, 901.

4. Bliokh, K.Y. & Nori, F. (2011). "Characterizing optical chirality."
   Phys. Rev. A 83, 021803.

## Files

- `em_checker.py` - Spectral Maxwell solver + conservation checker
- `problems/em_zilch.yaml` - Problem definition
- `problems/em_zilch_facts.json` - 12 oracle test facts
- `training/em_synthetic_30.json` - 30 training examples
- `training/scripts/train_em_adapter.py` - EM adapter training
- `adapters/em_adapter.npz` - v1 adapter (3/12 pass)
- `adapters/em_adapter_v2.npz` - v2 adapter (4/12 pass)

## Next Steps

1. **More training data**: Generate 50+ examples targeting failing facts
2. **Targeted facts**: Focus on helicity, momentum, energy formulas
3. **Cross-domain**: Test if EM adapter helps on other field theories
4. **Publication**: Add EM results to paper/breaking_frozen_priors.pdf
