# Context-Independent Term Preference Bias

## Discovery

LLMs exhibit **term preference bias**: fixed preferences for certain physics terms over others, **regardless of which answer is correct**. This creates systematic failures when the less-preferred term is the correct answer.

## Empirical Evidence

### Mirror Pair Analysis

By analyzing facts where the same terms swap roles (truth ↔ distractor), we can measure raw term preference independent of correctness:

| Preferred Term | Avoided Term | Preference Score | Evidence |
|----------------|--------------|------------------|----------|
| Navier-Stokes | Euler equations | +2.0 | NS as truth: +0.1, Euler as truth: -2.0 |
| kinetic energy | enstrophy | +8.7 | KE as truth: -0.9*, enstrophy as truth: -9.6 |
| linear momentum | total energy | +1.7 | LinMom as truth: +0.5, Energy as truth: -1.2 |
| simple powers (r) | logarithmic (-ln(r)) | +15+ | r forms pass, -ln(r) forms fail by -5.5 to -9.3 |

*KE as truth still fails (-0.9) because distractor has longer parenthetical

### Calculation

Preference score = margin(preferred_as_truth) - margin(avoided_as_truth)

When score > 0: model systematically picks preferred term regardless of context.

## Mechanism

Training data frequency determines term preference:

1. **Famous equation bias**: "Navier-Stokes" appears more often than "Euler equations" in training
2. **Pop-physics contamination**: "kinetic energy" >> "enstrophy" in web text
3. **Simplicity bias**: "r" and "1/r" >> "ln(r)" or "-ln(r)" in mathematical discussions
4. **Conservation familiarity**: "linear momentum" >> "total energy" in introductory physics

## Interaction with Other Biases

Term preference bias **interacts with length bias**:

| Scenario | Term Preference | Length Bias | Net Effect |
|----------|-----------------|-------------|------------|
| pf10: enstrophy(29) vs energy(14) | energy | energy (shorter) | Strong failure: -9.6 |
| pf09: energy(14) vs enstrophy(29) | energy | enstrophy (longer) | Weak failure: -0.9 |

When biases align (preferred term is also shorter): extreme failure.
When biases conflict: length bias can override term preference.

## Detection

Check for mirror pairs in fact sets:
```python
def detect_term_preference(facts):
    """Find fact pairs where terms swap roles."""
    pairs = []
    for f1 in facts:
        for f2 in facts:
            if f1['fact_id'] >= f2['fact_id']:
                continue
            # Check if truth/distractor are swapped
            if (f1['truth'] in f2['best_distractor'] or
                f2['truth'] in f1['best_distractor']):
                pairs.append((f1, f2))
    return pairs
```

## Implications

### For Fact File Design
- **Don't use famous/familiar terms as distractors** when truth is obscure
- If truth is "enstrophy", don't use "kinetic energy" as distractor
- If truth is "-ln(r)", don't use "r" as distractor
- Match familiarity level: "enstrophy" vs "helicity", not "enstrophy" vs "energy"

### For Evaluation
- Mirror pair tests reveal raw term preference
- Can distinguish "model doesn't know X" from "model prefers Y over X regardless of context"
- Important for interpreting physics domain failures

### For Adapter Training
- Term preference is deeply embedded in base weights
- May require targeted debiasing on specific term pairs
- Harder to fix than surface-level phrasing biases

## Specific Term Preferences (Physics)

| Preferred | Avoided | Domain |
|-----------|---------|--------|
| Navier-Stokes | Euler equations | Fluid dynamics |
| kinetic energy | enstrophy | 2D turbulence |
| linear momentum | total energy | Conservation laws |
| r, 1/r | -ln(r), ln(r) | Green's functions |
| direct cascade | inverse cascade | Turbulence |
| discovered | excluded | Particle physics |
| confirmed | uncertain | Scientific results |

## Status

**CONFIRMED** - Verified via mirror pair analysis on physics_fundamentals domain:
- 6 mirror pairs tested
- All show systematic term preference
- Preference scores range from +1.7 to +15+
- Interacts with (but is distinct from) length bias

---

*Discovered: 2026-03-17*
*Method: Mirror pair analysis of physics_fundamentals oracle failures*
