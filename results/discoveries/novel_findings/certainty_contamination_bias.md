# Certainty Contamination Bias

## Discovery
LLMs exhibit a systematic **certainty contamination bias**: they prefer definitive-sounding claims ("completely ruled out", "definitively proven") over hedged scientific language ("remains inconclusive", "awaits confirmation"), even when the hedged statement is factually correct.

## Empirical Evidence

### Correlation Analysis
- **r = -0.402** correlation between certainty gap and oracle margin
- **t = 3.57** (p < 0.01) for gap=0 vs gap≥3 comparison
- Pass rate by certainty gap: 55% (gap=0) → 26% (gap=3) → 25% (gap=4+)

### Length Confound Ruled Out
- High-certainty distractors are actually **LONGER** (r = +0.277)
- Length bias would favor the shorter truth, not the distractor
- The certainty effect **overrides** the length bias

### Rebalancing Test
Rewriting high-certainty distractors with hedged language:
- ppf06_neutrino_cp: -1.64 → -0.11 (**+1.53 improvement**)
- nf01_sterile: -1.46 → -0.44 (**+1.02 improvement**)
- Average improvement: **+0.89** margin points

## Affected Domains
Strongest in frontier science domains (67% of facts have gap≥2):
- Particle physics frontiers
- Neutrino frontiers
- Cosmology frontiers
- Climate science frontiers
- Black hole frontiers

These domains have hedged truths ("hints at", "consistent with", "awaits confirmation") competing against contrarian definitive distractors ("completely ruled out", "proven wrong").

## Mechanism
Training data over-represents definitive claims relative to hedged scientific language:
1. News headlines prefer definitive framing ("Scientists prove X")
2. Social media amplifies extreme claims
3. Scientific papers use hedging, but abstracts/conclusions often simplified
4. LLM learns higher prior probability for definitive-sounding text
5. When truth is hedged and distractor is definitive, model prefers distractor

## Implications

### For Fact File Design
- Balance certainty levels between truth and distractors
- Use hedged language in distractors: "may be incorrect", "some evidence against"
- Avoid definitive distractor language: "completely ruled out", "proven wrong"

### For Adapter Training
- Train on hedged-truth vs definitive-distractor contrasts
- Specifically target frontier science domains with high certainty gaps
- Use contrastive pairs where truth is hedged and distractor is definitive

### For Evaluation
- Certainty gap as a predictive feature for oracle difficulty
- High-gap facts require domain-specific adapters
- Consider certainty-balanced test sets for fair evaluation

## Quantitative Markers

### Certainty markers (in distractors, cause bias):
`definitively, completely, proven, ruled out, impossible, always, never,
guaranteed, certain, absolutely, all, none, every, must, cannot,
does not exist, zero, perfect, exactly, precise, fundamentally,
whatsoever, entirely, permanently, universal`

### Hedging markers (in truths, trigger bias):
`may, might, could, uncertain, varies, approximately, suggests,
indicates, possible, likely, probably, tentative, preliminary,
not ruled out, remains open, still debated, large uncertainties,
significance varies, consistent, some, limited, current, hints,
awaits confirmation, inconclusive`

## Status
**CONFIRMED** - Theory verified with:
- Correlation analysis (r = -0.402)
- Length confound ruled out (r = +0.277 in opposite direction)
- Rebalancing intervention (+0.89 average improvement)
- Domain analysis (67% high-gap in frontier domains)

## Cascade Routing Results

Using adapters as fallback when baseline fails (not as replacement):

| Metric | Baseline | Cascade | Change |
|--------|----------|---------|--------|
| Overall pass rate | 60.8% | 62.6% | **+1.8%** |
| gap=2 pass rate | 45% | 63% | **+18 pts** |
| gap=3 pass rate | 26% | 47% | **+21 pts** |
| gap≥4 pass rate | 25% | 25% | 0 |

**Key: zero regressions** — cascade only applies adapters to failing facts.

Rescues by adapter: cert_decon: 10, anti_def: 10

## Distractor Rewriting Results

For gap≥4 facts, rewriting distractors with balanced language fixes most:

| Fact | Original Margin | Balanced Margin | Result |
|------|----------------|-----------------|--------|
| nf01_sterile | -1.46 | +0.16 | **FIXED** |
| dm10_primordial | -1.20 | +0.64 | **FIXED** |
| ppf04_higgs_width | -0.11 | +0.25 | **FIXED** |
| ppf06_neutrino_cp | -1.64 | -0.23 | Improved |

Replacements used:
- "completely ruled out" → "appears unlikely"
- "definitively proven" → "seems supported"
- "fundamentally cannot" → "is difficult to"

## Adapter Training Results

Trained a certainty decontamination adapter on 118 high-gap examples:
- **Average improvement: +0.28** on high-gap facts
- **13/27 significantly improved** (Δ > 0.3)
- **3 errors fixed** (csf04, nf06, cof02)
- Large gains on hardest facts: nf03 +2.19, ppf06 +1.33, dm10 +1.21

**However**: Pass rate dropped 26% → 11% due to overcorrection.

**Root cause**: Adapter boosts hedged language, but some passing facts have *neutral* truths (no hedging markers). The adapter penalizes neutral truths along with definitive ones.

**Refined approach**: Train to penalize definitive *distractors* rather than boost hedged *truths*. Or use certainty-gap routing: only apply adapter when gap >= 2.

## Truth Style Taxonomy

| Truth Style | Example | Adapter Effect |
|------------|---------|----------------|
| Hedged | "remains inconclusive" | ✓ Improved |
| Neutral/Factual | "bulk gravity to boundary CFT" | ✗ Regressed |
| Definitive | "is always conserved" | ✗ May regress |

The certainty decontamination adapter only helps when truth is hedged AND distractor is definitive. For neutral truths, use standard routing.

---

*Discovered: 2026-03-17*
*Method: Oracle margin analysis across 1138 facts, certainty marker detection, rebalancing intervention, adapter training experiment*
