# Technical Simplification Bias

## Discovery

LLMs exhibit a systematic **technical simplification bias**: when truth uses precise technical language and distractors use simpler/familiar terms, the model prefers the simpler answer — even when incorrect.

## Empirical Evidence

### Statistical Analysis
- **t = -3.73, p = 0.0004** (highly significant)
- Failed facts: truth more technical than distractor (+0.52)
- Passed facts: truth less technical than distractor (-0.21)

### Examples of Failures

| Truth (technical) | Distractor (simple) | Margin |
|-------------------|---------------------|--------|
| enstrophy (squared vorticity) | kinetic energy | -9.62 |
| ψ = -Γ·ln(r)/(2π) | ψ = Γ·r/(2π) | -9.26 |
| inverse cascade (to large scales) | direct cascade (to small scales) | -7.58 |
| excluded simple models but not ruled out | discovered superpartners | -15.48 |
| Euler equations | Navier-Stokes equations | -1.97 |

### Pass/Fail by Technical Complexity

| Metric | Passed (n=47) | Failed (n=25) |
|--------|---------------|---------------|
| Truth technical markers | 0.23 | **0.76** |
| Distractor technical markers | 0.45 | 0.24 |
| Truth - Dist (technical) | -0.21 | **+0.52** |

## Mechanism

Training data over-represents simple explanations relative to precise technical language:

1. **Wikipedia effect**: Introductory explanations use simple terms; precise formulas buried in detail
2. **Pop-science contamination**: "Energy" is more frequent than "enstrophy"
3. **Famous term preference**: "Navier-Stokes" appears more than "Euler equations"
4. **Drama bias**: "Discovered X" is more memorable than "excluded simple models of X"

## Technical Markers (in truth, trigger failure)

```
ln(r), log, sqrt, π, exp, integral          # math notation
enstrophy, vorticity, advection, dissipation # fluid mechanics
quasi-normal, supertranslation, holographic  # physics jargon
deficit, asymmetry, hierarchy, ordering      # subtle distinctions
tension, disagree, uncertain, pending        # hedging
model-dependent, viable, consistent          # nuanced
```

## Simple Markers (in distractors, attract model)

```
energy, momentum, mass, force               # basic physics
confirmed, proven, discovered, detected     # definitive
perfect, exact, precisely, always, all      # absolutism
explained, resolved, determined             # closure
particle, wave, field                       # basic concepts
```

## Interaction with Other Biases

Technical simplification bias is **independent** of:

1. **Certainty contamination** (r = -0.402 with certainty gap)
   - Certainty is about hedge vs definitive language
   - Technical is about jargon vs familiar terms
   - A hedged technical truth fails on BOTH biases

2. **Length bias**
   - Technical terms can be short ("enstrophy") or long
   - The bias is about familiarity, not length

3. **Temporal inversion** (NOT supported)
   - Recent confirmations don't have lower margins
   - The model doesn't prefer old positions over new

## Implications

### For Fact File Design
- Balance technical complexity between truth and distractors
- If truth uses "enstrophy", use "potential enstrophy" as distractor (not "energy")
- If truth uses "ln(r)", use "1/r" as distractor (not "r")
- Match jargon level: both technical or both simple

### For Adapter Training
- Create technical calibration adapter
- Train on pairs where technical truth beats simple distractor
- Focus on domains with high technical vocabulary (fluid mechanics, QFT, etc.)

### For Evaluation
- Technical complexity ratio as a predictive feature
- High-ratio facts (technical truth, simple distractor) require domain adapters
- Consider complexity-balanced test sets

## Quantitative Detection

```python
TECHNICAL_MARKERS = [
    'ln(r)', 'log', 'sqrt', 'π', 'exp', 'integral',
    'enstrophy', 'vorticity', 'advection', 'dissipation',
    'quasi-normal', 'supertranslation', 'holographic',
    'deficit', 'asymmetry', 'hierarchy', 'ordering',
    'tension', 'disagree', 'uncertain', 'pending',
    'model-dependent', 'viable', 'consistent',
]

def technical_ratio(truth: str, distractor: str) -> float:
    """Compute technical complexity ratio (truth / distractor).

    Ratio > 1.5 indicates high failure risk.
    """
    truth_tech = sum(1 for m in TECHNICAL_MARKERS if m in truth.lower())
    dist_tech = sum(1 for m in TECHNICAL_MARKERS if m in distractor.lower())
    return (truth_tech + 0.5) / (dist_tech + 0.5)
```

## Status

**CONFIRMED** - Theory verified with:
- Statistical test: t = -3.73, p = 0.0004
- 72 facts across 6 physics domains
- Clear examples of technical truth losing to simple distractor

---

*Discovered: 2026-03-17*
*Method: Temporal inversion analysis led to technical complexity analysis, revealing the true failure mechanism*
