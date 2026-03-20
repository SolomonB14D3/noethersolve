# Autonomous Training Pipeline — March 2026

## Overview

A 32B-orchestrated pipeline that uses proven training techniques to fix knowledge gaps in the 4B oracle model.

## Architecture

```
32B Model (llama.cpp)          4B Model (MLX)
├── Fact rewriting             ├── Oracle evaluation
├── Cluster detection          ├── Adapter training
├── Technique selection        └── Adapter inference
└── Pipeline orchestration
```

## Pipeline Steps

### 1. Check Existing Adapters
Search for working adapters before training new ones.
- Patterns: `{domain}_*.npz`, `{prefix}_*.npz`
- If ≥50% found, use it and skip training

### 2. Length Bias Detection
Audit facts for length bias using `noethersolve.audit_facts`.
- If HIGH_RISK detected, use 32B to rewrite with length-matched distractors
- Creates `{domain}_facts_v2.json` with balanced lengths

### 3. Technique Selection
Based on domain characteristics:

| Condition | Technique |
|-----------|-----------|
| All failing, no base adapter | Staged |
| Some passing + base adapter exists | Anchored |
| Bimodal margins (see-saw detected) | Orthogonal |

### 4. See-Saw Detection
Tests if training on cluster A hurts cluster B:
1. Train mini-adapter on cluster 1
2. Check if cluster 2 margins dropped >10 points
3. If yes → switch to orthogonal training

### 5. Training Techniques

**Staged Training**
- Sequential clusters with decreasing LR
- Good for: Simple domains without interference
- Script: `train_staged_adapter.py`

**Anchored Training**
- Protects passing facts while learning new ones
- Requires base adapter to build on
- Script: `train_anchored_adapter.py`

**Orthogonal Training**
- Separate adapter per conceptual cluster
- Route each fact to best adapter at inference
- Creates `{domain}_orthogonal_routing.json`

## Results

| Domain | Baseline | After Pipeline | Technique |
|--------|----------|----------------|-----------|
| kinetic_k | 37.5% | **100%** | Anchored |
| ns_regularity | 6.2% | **93.8%** | Anchored |
| em_zilch | 8.3% | **100%** | (existing) |
| information_theory | 0% | **91.7%** | Length-matching |
| intersection_theory | 0% | **58.3%** | Length-matching |

## Usage

```bash
# Train single domain
python scripts/train_with_proven_methods.py --domain <domain>

# Full autonomous scan
python scripts/train_with_proven_methods.py --scan-all
```

## Files

- `scripts/train_with_proven_methods.py` — Main orchestration script
- `scripts/autonomous_research.py` — Length-matching + gap detection
- `training/scripts/train_staged_adapter.py` — Staged training
- `training/scripts/train_anchored_adapter.py` — Anchored training
- `adapters/*_orthogonal_routing.json` — Routing configs

## Key Findings

1. **Length bias is the #1 cause of false gaps** — 4 of 5 tested 0% domains were measurement artifacts
2. **Anchored training is highly effective** — 93.8% on ns_regularity, 100% on kinetic_k
3. **32B can orchestrate but not directly train** — Different architectures require 4B for actual training
4. **Existing adapters often work** — Check before training new ones
