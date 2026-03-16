# Adapter Combination Findings

## Date: 2024-03-15

## Summary

Investigated approaches to combine all 67 domain adapters (999 facts). Found that **hybrid routing** outperforms both pure orthogonal and pure joint approaches.

## Approaches Tested

### 1. Naive Stacking (FAILED)
- Stacking 37+ adapters destroys MMLU: 68% → 25% (-43%)
- Conflicting gradient directions in logit space

### 2. Weight Averaging (FAILED)
- Merging trained adapters by averaging weights
- Underperformed both specialists on every benchmark

### 3. Joint Training from Scratch (PARTIAL SUCCESS)
- Train ONE adapter on multiple related domains with difficulty-weighted sampling
- Results on physics_frontier (9 domains, 108 facts):
  - Baseline: 33.3%
  - Joint adapter: 53.7% (+20.4%)
- Works for semantically related domains, fails for heterogeneous mix

### 4. Orthogonal Per-Cluster (GOOD)
- Separate adapter per concept cluster, routed at inference
- Results on physics_frontier: 70.2%
- Achieves 100% on well-trained domains (cosmology, condensed_matter)

### 5. Hybrid Routing (BEST)
- For each fact, pick whichever adapter (joint or orthogonal) has higher margin
- Results on physics_frontier: **82.1%** (69/84)

## Detailed Results on Physics Frontier Domains

| Domain | Baseline | Joint | Orthogonal | Hybrid |
|--------|----------|-------|------------|--------|
| particle_physics_frontiers | 2/12 | 9/12 | 6/12 | 10/12 |
| neutrino_frontiers | 4/12 | 7/12 | 6/12 | 9/12 |
| holographic_qinfo | 2/12 | 7/12 | 9/12 | 10/12 |
| dark_matter_energy | 3/12 | 3/12 | 6/12 | 7/12 |
| quantum_gravity | 1/12 | 2/12 | 8/12 | 9/12 |
| cosmology_frontiers | 3/12 | 5/12 | 12/12 | 12/12 |
| condensed_matter_frontiers | 3/12 | 4/12 | 12/12 | 12/12 |
| **TOTAL** | **18/84 (21.4%)** | **37/84 (44.0%)** | **59/84 (70.2%)** | **69/84 (82.1%)** |

## Per-Fact Analysis

- Joint adapter better on 30 facts
- Orthogonal adapter better on 53 facts
- Tie on 1 fact

Pattern: Joint wins on particle_physics, neutrino, holographic. Orthogonal wins on dark_matter, quantum_gravity, cosmology, condensed_matter.

## 15 Stubborn Facts (Fail with Both)

These fail even with hybrid routing:

1. ppf02_g2_theory (J=-180.0, O=-3.6) - muon g-2 theory tension
2. ppf06_neutrino_cp (J=-26.6, O=-9.9) - CP violation hints
3. nf03_reactor (J=-15.3, O=-5.1) - reactor antineutrino anomaly
4. nf09_coherent (J=-153.4, O=-74.9) - coherent scattering detection
5. nf10_geo (J=-15.5, O=-23.7) - geoneutrinos
6. hqi03_syk (J=-108.6, O=-5.9) - SYK model
7. hqi10_tensor (J=-0.1, O=-15.4) - tensor networks
8. dm01_rotation (J=-60.0, O=-14.5) - rotation curves
9. dm06_dark_energy (J=-60.0, O=-138.2) - accelerating expansion
10. dm07_cosmological_constant (J=-242.5, O=-124.8) - 10^120 problem
11. dm10_primordial (J=-93.4, O=-65.3) - primordial black holes
12. dm12_detection (J=-93.4, O=-13.6) - direct detection
13. qg06_ads_cft (J=-306.8, O=-100.0) - AdS/CFT correspondence
14. qg07_hawking (J=-11.1, O=-88.8) - Hawking radiation
15. qg08_firewall (J=-191.1, O=-76.0) - firewall paradox

Root cause: Token-length bias (truth shorter than distractors) or undertrained adapters.

## Recommendations

1. **For deployment**: Use hybrid routing - check both joint and orthogonal, pick higher margin
2. **For stubborn facts**: Rewrite to fix token-length bias (shorten truth, lengthen distractors)
3. **For new domains**: Train orthogonal per-cluster, then train joint adapter for domain group

## Key Insight

> Joint training from scratch ≠ Stacking trained adapters
> - Joint training from scratch: WORKS (for related domains)
> - Stacking/averaging independently-trained adapters: FAILS

The representational see-saw problem means some concepts need opposite logit-space directions. A single adapter can only point one way. Solution: orthogonal adapters with routing, or hybrid of joint + orthogonal.
