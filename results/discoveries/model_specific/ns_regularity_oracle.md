# NS Regularity and Stretch-Resistant Q_f Oracle Results

## Overview

Oracle testing of 16 facts related to:
- 3D Navier-Stokes regularity
- Vortex stretching mechanics
- Stretch-resistant Q_f variants (R_f ratio)
- BKM criterion
- Viscous decay scaling

## Baseline Results

| Metric | Value |
|--------|-------|
| Pass rate | 0/16 (0%) |
| Mean margin | -46.7 |
| Min margin | -79.1 |

**Diagnosis: Complete knowledge gap on NS/stretching mechanics.**

### Worst-performing facts (baseline)
| Fact | Margin |
|------|--------|
| Enstrophy-norm Q_f/Ω stretch behavior | -79.1 |
| R_f stretch-resistant ratio | -76.1 |
| R_f combined score 0.59% | -65.1 |
| Curvature-weighted Q_f stretch | -65.2 |

## Adapter Results

### ns_adapter (v1)
- Steps: 5000, lr=3e-6, margin_target=2.5
- Pass rate: 2/16 (12.5%)
- Mean margin: -43.2

**Flipped facts:**
| Fact | Baseline | Adapter | Delta |
|------|----------|---------|-------|
| ns12_3d_challenge (stretching breaks Q_f) | -49.6 | +23.3 | +72.9 |
| ns14_helicity_3d (H exactly conserved) | -30.1 | +9.1 | +39.2 |

### ns_adapter_v2
- Steps: 6000, lr=5e-6, margin_target=3.0
- Pass rate: 3/16 (18.8%)
- Mean margin: -56.1 (worse)

**Issue**: Higher lr caused overfitting, making some facts worse. The KNOWLEDGE_GAP diagnostic was triggered.

**Close to flip:**
| Fact | Margin |
|------|--------|
| R_f stretch-resistant ratio | -1.3 |
| Millennium problem | -11.1 |

## Key Findings

### 1. Core Concepts Partially Learned
- Vortex stretching breaking Q_f: FLIPPED (+23.3)
- 3D helicity conservation: FLIPPED (+9.1)

### 2. Numerical Results Hard to Teach
- R_f 2% vs 60% variation: still failing
- Combined score 0.59%: still failing
- Specific percentages are tokenization-sensitive

### 3. BKM Criterion Improving
- Baseline: -34.6
- v1: -3.8 (close to flip)

## Files Created

- `problems/ns_regularity.yaml` - Problem definition
- `problems/ns_regularity_facts.json` - 16 oracle test facts
- `training/ns_regularity_synthetic_60.json` - 58 training examples
- `training/scripts/train_ns_adapter.py` - Training script
- `adapters/ns_adapter.npz` - v1 adapter (2/16)
- `adapters/ns_adapter_v2.npz` - v2 adapter (3/16, but worse margins)

## Next Steps

1. Train v3 with lower lr (2e-6) and more epochs
2. Add more training examples for numerical facts
3. Try stacking with continuous Q_f adapter
4. Consider fact reformulation for tokenization-sensitive facts
