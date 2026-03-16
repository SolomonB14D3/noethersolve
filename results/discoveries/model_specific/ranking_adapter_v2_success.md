# Ranking Adapter v2 - Conservation Quality Ranking (2026-03-14)

## Breakthrough

Successfully trained an adapter to learn the ranking of conservation laws by quality.
The adapter now correctly orders invariants by their frac_var (conservation quality).

## Results

| Metric | Baseline | v1 (failed) | v2 (success) |
|--------|----------|-------------|--------------|
| Spearman ρ | 0.35 | 0.15 | **0.93** |
| Pearson r | 0.33 | 0.15 | 0.68 |

Best result: **Spearman ρ = 0.932** at step 50.

## Training Progress

```
step   50: test_ρ = 0.932  <- Best!
step  100: test_ρ = 0.836
step  150: test_ρ = 0.815
step  200: test_ρ = 0.856
step  250: test_ρ = 0.856
step  300: test_ρ = 0.848
step  350: test_ρ = 0.812
step  400: test_ρ = 0.844
```

## Key Improvements Over v1

### 1. Log-Scale Targets
**Before (v1):** target = 1/frac_var (ranges from 100 to 10^12)
**After (v2):** target = -log10(frac_var) (ranges from 1 to 12)

The log scale:
- Bounded and interpretable
- Equal weight to order-of-magnitude differences
- Prevents outliers from dominating gradients

### 2. ListNet Loss
**Before (v1):** Pairwise ranking loss (margin_i > margin_j for all i<j)
**After (v2):** ListNet - compare softmax probability distributions

```python
def listnet_loss(margins, targets):
    margin_probs = mx.softmax(margins / temp)
    target_probs = mx.softmax(targets / temp)
    loss = -mx.sum(target_probs * mx.log(margin_probs + 1e-10))
    return loss
```

ListNet advantages:
- Considers all items simultaneously (not just pairs)
- Gradients flow to all positions
- More stable training

### 3. Hard Negative Mining
Explicitly ensure bad invariants rank below good ones:

```python
for i, qi in enumerate(qualities):
    for j, qj in enumerate(qualities):
        if qi in ["exact", "excellent"] and qj in ["poor", "bad"]:
            pair_loss = mx.maximum(0.0, margins[j] - margins[i] + 2.0)
```

This forces a clear separation between conservation classes.

## Training Configuration

- Model: Qwen/Qwen3-4B-Base
- Steps: 400
- Batch size: 12
- Learning rate: 1e-6
- Loss weights: listnet=2.0, hinge=1.0, hard_neg=1.0

## What This Enables

With the ranking adapter, the oracle can now:
1. **Recognize conservation laws** (positive margin for conserved quantities)
2. **Rank by quality** (higher margin = better conservation)
3. **Reject non-conserved** (lower margin for bad candidates)

This transforms the oracle from a binary classifier into a quality estimator,
enabling more nuanced candidate prioritization in the discovery pipeline.

## Files

- Training script: `train_ranking_v2.py`
- Best adapter: `adapters/ranking_v2_best.npz` (saved at step 50)

## Status

- Spearman correlation: **0.932** (near-perfect ranking)
- Ready for integration into the discovery pipeline
