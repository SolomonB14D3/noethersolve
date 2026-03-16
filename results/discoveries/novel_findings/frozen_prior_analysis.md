# Frozen Prior Analysis (2026-03-13)

## Discovery

The oracle model (Qwen/Qwen3-4B-Base) exhibits a **frozen prior** that rejects 
combinations of known invariants (H, Lz) without actually evaluating the specific
coefficients.

## Evidence

### H·r₁₂ + α·Lz Family Test

Tested 7 different α values across 4 orders of magnitude:

| α | frac_var | margin | deviation from mean |
|---|----------|--------|---------------------|
| 0.001 | 1.68e-3 | -76.2 | +1.3 |
| 0.01 | 1.50e-3 | -78.4 | -0.9 |
| 0.1 | 1.20e-3 | -74.5 | +3.0 |
| 0.5 | 5.0e-4 | -80.1 | -2.6 |
| 1.0 | **2.08e-4** | -77.6 | -0.1 |
| 2.0 | 8.0e-4 | -78.9 | -1.4 |
| 10.0 | 3.50e-3 | -76.8 | +0.7 |

**Statistics:**
- Mean margin: -77.5
- Std deviation: 1.7
- Range: 5.6 (negligible given 4 OOM in α)

**Conclusion:** The model outputs nearly identical margins regardless of α.
This is classic "frozen prior" behavior - pattern matching without evaluation.

### Control Test: Q₂ (Exact Invariant)

Q₂ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ² is EXACTLY conserved (reduces to Lz).

**Oracle margin: -43.9** ← FAILS to recognize exact invariant!

This reveals a deeper physics gap: the model doesn't understand that certain
combinations of known invariants yield new exact invariants.

### Validation Test: Q₁ (Novel Discovery)

Q₁ = Σᵢ<ⱼ ΓᵢΓⱼ rᵢⱼ is our novel near-invariant.

**Oracle margin: -68.1** ← As expected, genuinely unknown.

## Implications

1. **Training Strategy:** Need to teach the model to evaluate coefficients 
   individually, not just pattern-match "combination of H and Lz"

2. **Control/Target/Validation Framework:**
   - Control (Q₂): Should pass after training (reduces to Lz)
   - Target (H·r₁₂ + α·Lz): Margins should diverge after training
   - Validation (Q₁): Success = model recognizes novel invariant

3. **Adapter Design:** Current logit adapter overfits on small training sets.
   Need more diverse training data or different architecture.

## Next Steps

1. Create larger, more diverse training set for α variants
2. Use Q₂ as explicit control during training
3. Add early stopping when Q₂ margin starts decreasing
4. Validate on Q₁ only after control passes
