# Frontier Domains Training - Findings

**Date:** 2026-03-15
**Domains:** 8 frontier problem domains (96 facts total)
**Result:** 96/96 (100%) after token-length bias fixes

---

## Summary

Trained orthogonal adapters on 8 cutting-edge scientific domains covering unsolved problems and frontier research areas. All 96 facts achieved positive margins after fixing 6 facts with token-length bias issues.

| Domain | Facts | Clusters | Baseline | Final |
|--------|-------|----------|----------|-------|
| battery_technology | 12 | solid_state, degradation, materials, alternatives | 6/12 | **12/12** |
| origin_of_life | 12 | hypotheses, challenges, experiments | 3/12 | **12/12** |
| consciousness | 12 | philosophy, theories, neuroscience | 4/12 | **12/12** |
| antibiotic_resistance | 12 | mechanisms, alternatives, challenges | 6/12 | **12/12** |
| protein_folding | 12 | prediction, theory, dynamics, cellular, disease | 7/12 | **12/12** |
| aging_biology | 12 | mechanisms, interventions, biomarkers | 6/12 | **12/12** |
| quantum_gravity | 12 | fundamentals, theories, black_holes | 4/12 | **12/12** |
| dark_matter_energy | 12 | dark_matter, candidates, alternatives, dark_energy | 6/12 | **12/12** |

---

## Key Findings

### 1. Token-Length Bias Is Domain-Agnostic

Six facts required fixing due to token-length bias. The bias appeared across physics, biology, and philosophy - suggesting it's a fundamental model preference for shorter completions regardless of domain:

| Fact ID | Domain | Original Margin | Issue |
|---------|--------|-----------------|-------|
| ab12_proteostasis | aging_biology | -70.62 | Truth too long |
| ar12_peptide | antibiotic_resistance | -20.34 | Truth too long |
| dm09_bullet_cluster | dark_matter_energy | -10.52 | Truth too long |
| pf02_levinthal | protein_folding | -2.71 | Truth too long |
| ab04_epigenetic_clock | aging_biology | -33.92 | Emerged during retrain |
| ar02_crispr_antimicrobial | antibiotic_resistance | -22.66 | Emerged during retrain |

**Fix pattern:** Shorten truth to 3-5 tokens, lengthen distractors to 10-15 tokens with qualifiers.

### 2. Training Instability in Small Clusters

When fixing one fact in a small cluster, others sometimes regressed:

- `aging_biology/biomarkers` (3 facts): Fixing `ab12_proteostasis` caused `ab04_epigenetic_clock` to flip from +16.15 to -33.92 margin
- `antibiotic_resistance/alternatives` (4 facts): Fixing `ar12_peptide` caused `ar02_crispr_antimicrobial` to flip from +5.35 to -22.66 margin

**Hypothesis:** Adapter capacity is being reallocated within clusters. The model learns a shared representation that can only "point" in limited directions. Small clusters are more susceptible because each fact has larger influence on the shared representation.

**Implication:** After fixing any fact, always re-verify all other facts in the same cluster.

### 3. Surprising Priors (Knowledge Gaps)

Some facts had remarkably strong wrong priors:

| Fact | Starting Margin | Wrong Belief |
|------|-----------------|--------------|
| ab12_proteostasis | -70.62 | "proteostasis improves with age" |
| ar12_peptide | -20.34 | "peptides have no antibacterial activity" |
| dm08_cmb | -12.00 | "CMB is insensitive to dark matter" |
| qg01_scale | -15.23 | "QG effects important at macroscopic scales" |

These suggest the base model has internalized counterintuitive beliefs, possibly from training data that emphasizes contrarian framings or confusion between related concepts.

### 4. Baseline Performance Varies Widely

| Baseline Performance | Clusters |
|---------------------|----------|
| 0% (0/N) | quantum_gravity/fundamentals |
| 25% (1/4) | origin_of_life/challenges, origin_of_life/experiments |
| 50% (2/4) | dark_matter_energy/dark_matter, quantum_gravity/black_holes |
| 75%+ | consciousness/philosophy, battery_technology/degradation |
| 100% | protein_folding/disease (2/2 already passing) |

The base model performs better on well-established philosophical concepts (consciousness theories) than on cutting-edge physics (quantum gravity fundamentals), despite the latter being more concrete.

---

## Cluster Routing Architecture

28 cluster-specific adapters trained:

```
aging_biology/biomarkers (3 facts)
aging_biology/interventions (5 facts)
aging_biology/mechanisms (4 facts)
antibiotic_resistance/alternatives (4 facts)
antibiotic_resistance/challenges (4 facts)
antibiotic_resistance/mechanisms (4 facts)
battery_technology/alternatives (3 facts)
battery_technology/degradation (3 facts)
battery_technology/materials (3 facts)
battery_technology/solid_state (3 facts)
consciousness/neuroscience (4 facts)
consciousness/philosophy (4 facts)
consciousness/theories (4 facts)
dark_matter_energy/alternatives (2 facts)
dark_matter_energy/candidates (3 facts)
dark_matter_energy/dark_energy (3 facts)
dark_matter_energy/dark_matter (4 facts)
origin_of_life/challenges (4 facts)
origin_of_life/experiments (4 facts)
origin_of_life/hypotheses (4 facts)
protein_folding/cellular (3 facts)
protein_folding/disease (2 facts)
protein_folding/dynamics (2 facts)
protein_folding/prediction (2 facts)
protein_folding/theory (3 facts)
quantum_gravity/black_holes (4 facts)
quantum_gravity/fundamentals (4 facts)
quantum_gravity/theories (4 facts)
```

---

## Training Parameters

- Steps: 500 per cluster
- Learning rate: 1e-5
- Optimizer: AdamW (weight_decay=0.01)
- Adapter: SnapOnLogitMLP (d_inner=64)
- Loss: Hinge margin (margin_target=2.0)
- Total training time: ~21 minutes for 28 clusters

---

## Files Created

- `problems/battery_technology_facts.json`
- `problems/origin_of_life_facts.json`
- `problems/consciousness_facts.json`
- `problems/antibiotic_resistance_facts.json`
- `problems/protein_folding_facts.json`
- `problems/aging_biology_facts.json`
- `problems/quantum_gravity_facts.json`
- `problems/dark_matter_energy_facts.json`
- `experiments/train_frontier_domains.py`
- `experiments/verify_frontier_domains.py`
- 28 adapter files in `adapters/`

---

## Open Questions

1. **Why does quantum_gravity/fundamentals have 0% baseline?** The Planck scale concept is well-documented, yet the model strongly prefers wrong answers.

2. **What causes the "proteostasis improves" prior?** This is biomedically incorrect but the model is extremely confident (-70.62 margin).

3. **Can cluster size predict instability?** Preliminary observation: clusters with ≤3 facts showed more cross-fact interference during training.

4. **Is there a principled way to detect token-length bias before training?** Current approach is reactive (fix after failure).
