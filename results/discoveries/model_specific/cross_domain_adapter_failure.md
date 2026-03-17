# Finding: Cross-Domain Adapters Cannot Fix True Blind Spots

**Date:** 2026-03-17
**Status:** Negative result (scientifically important)

---

## Summary

Logit-space adapters consistently **fail** on cross-domain mathematical connections, even with orthogonal per-cluster training. This distinguishes **true blind spots** (model lacks underlying concepts) from **calibration failures** (model knows but ranks poorly).

---

## Experimental Results

### Single Unified Adapter

```
Pre-training:  0/12 PASS, avg margin = -23.54
Post-training: 0/12 PASS, avg margin = -402.10  (17x WORSE)
```

### Orthogonal Per-Cluster Adapters

| Cluster | Facts | Baseline | Post-Training | Change |
|---------|-------|----------|---------------|--------|
| deadlock_detailed_balance | 3 | -33.60 | -562.11 | 17x worse |
| isolation_decoherence | 3 | -22.92 | -211.40 | 9x worse |
| pagerank_equilibrium | 2 | -28.56 | -789.63 | 28x worse |
| type_gauge | 2 | -15.69 | -335.30 | 21x worse |
| huffman_landauer | 2 | -12.22 | -334.41 | 27x worse |

**All adapters make margins dramatically worse, not better.**

---

## Why This Happens

Adapters work by steering the model's existing logit distribution. They succeed when:
1. The model already has correct internal representations
2. The problem is ranking/calibration, not knowledge

Adapters fail when:
1. The model has no internal representation of the concept
2. The training signal is sparse (2-3 facts per cluster)
3. The correct answer requires combining knowledge from separate domains

Cross-domain connections fail because:
- "Deadlock" and "detailed balance" live in completely separate embedding subspaces
- No training data ever connected these terms
- The adapter can't create new conceptual bridges, only reweight existing ones

---

## Implications

### 1. Two Classes of Model Failures

| Type | Symptoms | Fix |
|------|----------|-----|
| **Calibration failure** | Negative margins, adapter helps | Logit-space adapter |
| **True blind spot** | Negative margins, adapter makes WORSE | Requires external tools |

### 2. The Right Solution: MCP Tools

For true blind spots, the solution is **external verified computation**, not model correction:

- Model can call `check_detailed_balance()` to verify thermodynamic consistency
- Model can call `calc_deadlock()` to detect OS deadlock cycles
- Cross-domain connection happens at **tool output interpretation**, not internal representation

### 3. Adapter Training as a Blind Spot Detector

If adapter training makes margins worse, the domain is a true blind spot. This is a diagnostic signal:
- Margins improved → adapter deployable
- Margins unchanged → need more data
- Margins worsened → build MCP tool instead

---

## Connection to Other Findings

This complements the 5 cross-domain theories discovery:

| Finding | Evidence | Action |
|---------|----------|--------|
| Model blind spots exist | 0/12 oracle PASS | Documented |
| Adapters can't fix them | All margins 9-28x worse | Don't train |
| MCP tools can bridge | Verified computations | Build tools |

---

*Discovered: 2026-03-17*
*Method: Single adapter + orthogonal per-cluster training*
*Key insight: True blind spots require tools, not weight changes*
