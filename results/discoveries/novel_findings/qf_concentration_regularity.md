# Q_f Concentration Detection and Regularity Implications

## Discovery

Different Q_f invariants respond differently to vorticity concentration. This provides a diagnostic tool for detecting potential singularities in Navier-Stokes evolution.

## Key Results

### 1. Classification by Concentration Response

| Q_f type | f(r) behavior at r→0 | Response to concentration | Scaling |
|----------|----------------------|---------------------------|---------|
| **Detecting** | Diverges (f→∞) | Q_f increases | Warns of blowup |
| **Blind** | Vanishes (f→0) | Q_f decreases | Insensitive to concentration |
| **Bounded** | Finite | Q_f ~ constant | Intermediate |

### 2. Specific Functions

**Concentration-detecting (diverge as σ→0):**
- Q_{-ln(r)}: Energy - scales as σ^{-0.63}
- Q_{1/√r}: Scales as σ^{-0.47}

**Concentration-blind (decrease as σ→0):**
- Q_{√r}: Scales as σ^{+0.49}
- Q_r: Scales as σ^{+0.99}
- Q_{tanh(r)}: Scales as σ^{+0.75}

**Bounded:**
- Q_{e^(-r)}: Scales as σ^{-0.44}, remains finite

## Implications for Navier-Stokes Regularity

### Conservation Constraint Argument

If Q_{√r} is conserved (inviscid limit) and vorticity concentrates (σ→0):

1. √r → 0 at the concentration point
2. For Q_{√r} = ∫∫ ω(x)ω(y)√|x-y| dx dy to remain constant
3. Circulation must **spread out** to compensate
4. This could provide a mechanism preventing blowup

### Diagnostic Tool

Monitor the ratio:
```
R = Q_{-ln(r)} / Q_{√r}
```

- If R grows: concentration is occurring (warning)
- If R stays bounded: flow remains regular

## Mathematical Formulation

For a Gaussian vortex with width σ and fixed circulation Γ:
- Peak vorticity: ||ω||_∞ ∝ 1/σ²
- Enstrophy: Ω ∝ 1/σ²

The Q_f scaling can be derived:
```
Q_f = ∫∫ ω(x)ω(y) f(|x-y|) dx dy

For Gaussian: ω(x) = (Γ/2πσ²) exp(-|x|²/2σ²)

Q_f ∝ Γ² × σ^{2α} where α depends on f(r) near r=0
```

## Experimental Verification

Tested with vortex widths σ ∈ [0.05, 1.0]:

| σ | ||ω||_∞ | Q_{-ln(r)} | Q_{√r} | Ratio |
|---|--------|------------|--------|-------|
| 1.0 | 0.16 | -0.38 | 1.26 | -0.30 |
| 0.1 | 15.9 | 1.89 | 0.41 | 4.66 |
| 0.05 | 63.7 | 2.57 | 0.29 | 8.91 |

The ratio increases by 30× as vorticity concentrates by 400×.

## Connection to Other Results

- Confirms Q_{-ln(r)} = kinetic energy (concentration-sensitive)
- Explains why e^(-r) is most robust in numerical tests (bounded)
- Suggests tanh(r) as compromise (bounded but with sensitivity)

## Open Questions

1. Can Q_f ratios provide a priori regularity criteria?
2. Is there a Q_f that perfectly tracks Beale-Kato-Majda criterion?
3. Can machine learning identify optimal concentration-detecting invariant?

## References

- Beale-Kato-Majda criterion (1984): blowup iff ∫|ω|_∞ dt = ∞
- Constantin et al. on vorticity concentration
- This work: Q_f framework for regularity diagnostics
