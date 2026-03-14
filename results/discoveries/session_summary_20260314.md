# Session Summary: 2026-03-14

## Overview

Extended noethersolve to three new domains (NS regularity, K invariant, optimal f) and improved existing domains (EM, continuous Q_f).

## Domain Status

| Domain | Facts | Baseline | Best Adapter | Pass Rate | Status |
|--------|-------|----------|--------------|-----------|--------|
| Point-vortex Q_f | 13 | varies | ranking_v2 | ~80% | COMPLETE |
| EM conservation | 12 | 1/12 (8%) | em_adapter_v4 | 6/12 (50%) | FIXABLE_BIAS |
| Continuous Q_f | 12 | 0/12 (0%) | qf_continuous | 7/12 (58%) | FIXABLE_BIAS |
| NS regularity | 16 | 0/16 (0%) | ns_adapter | 2/16 (12.5%) | KNOWLEDGE_GAP |
| K invariant | 8 | 0/8 (0%) | k_adapter_v2 | 3/8 (37.5%) | FIXABLE_BIAS |
| Optimal f(r) | 4 | 0/4 (0%) | - | 0/4 | KNOWLEDGE_GAP |

## Key Findings

### NS Regularity Domain
- 16 facts covering BKM criterion, vortex stretching, R_f ratio
- Complete knowledge gap (0/16 baseline)
- ns_adapter flipped 2 facts: 3D challenge, helicity conservation
- R_f stretch-resistant ratio close to flip (margin -1.3)

### K Invariant Domain
- K = Σ Γᵢ vᵢ² is independent of Q_f (depends on angles)
- 8 facts covering definition, independence, structure
- k_adapter flipped 3 facts: definition, independence, physical interpretation
- Mean margin improved -45.9 → -15.9

### Optimal f(r) Discovery
- Gradient descent finds optimal linear combination
- 99.6% improvement over single basis functions
- Dominant terms: e^(-r/2), tanh(r), sin(r)
- Complete knowledge gap (0/4 baseline)

## Adapters Trained

| Adapter | Domain | Steps | Pass Rate | Delta |
|---------|--------|-------|-----------|-------|
| ns_adapter | NS regularity | 5000 | 2/16 | +46.7→-43.2 |
| ns_adapter_v2 | NS regularity | 6000 | 3/16 | worse margins |
| k_adapter | K invariant | 4000 | 3/8 | +29.5 |
| k_adapter_v2 | K invariant | 5000 | 3/8 | +30.1 |

## Candidates Added

- 5 NS regularity entries
- 4 K invariant entries
- 2 optimal f entries

Total candidates: 183 (from 169)

## Files Created

### Problems
- problems/ns_regularity.yaml
- problems/ns_regularity_facts.json
- problems/kinetic_k.yaml
- problems/kinetic_k_facts.json
- problems/optimal_f.yaml
- problems/optimal_f_facts.json

### Training
- training/ns_regularity_synthetic_60.json (58 examples)
- training/kinetic_k_synthetic_30.json (30 examples)
- training/scripts/train_ns_adapter.py
- training/scripts/train_k_adapter.py

### Adapters
- adapters/ns_adapter.npz
- adapters/ns_adapter_v2.npz
- adapters/k_adapter.npz
- adapters/k_adapter_v2.npz

### Discoveries
- results/discoveries/ns_regularity_oracle.md
- results/discoveries/session_summary_20260314.md

## Next Steps

1. **NS regularity**: Train with more data targeting R_f ratio and BKM facts
2. **K invariant**: Add more training examples for cancellation mechanism
3. **Optimal f**: Create training data to teach optimal combination
4. **Adapter stacking**: Test combining adapters across domains
5. **Autonomy loop**: Run full autonomous sweep on vortex_unsolved
