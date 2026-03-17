# Oracle Fact Quality Methodology

Detailed documentation of the 9 oracle bias mechanisms discovered during NoetherSolve pipeline development (Mar 16-17, 2026). These guide fact file construction and oracle evaluation.

For the compact checklist, see `CLAUDE.md` → "Unified Audit Checklist".
For per-mechanism analysis files, see `results/discoveries/novel_findings/`.

---

## Phrasing Rules for Oracle Success (Discovered Mar 16)

Oracle failures are **phrasing bias**, not knowledge gaps. The model is confident in the wrong answer because the wrong answer is phrased confidently, while the truth is phrased hesitantly. Rephrase before oracle evaluation:

1. **Remove hedging:** "may suggest" → "reveals", "might suggest" → "demonstrates"
2. **Remove parentheticals:** "(under certain conditions)" → remove entirely or specify once in intro
3. **Use active voice:** "it's been shown that" → "the data show"
4. **Shorten truth:** Keep <10 words. Every word below that adds ~1–2 points to margin.
5. **Match distractor length:** If distractor is 8 words, truth should be 7–9 words. Asymmetry triggers model bias.
6. **Use symbolic notation:** "the pairwise-weighted quantity Q_f" → "Q_f = Σ ΓᵢΓⱼ f(rᵢⱼ)"
7. **Lead with the claim:** "Q_f is conserved because..." NOT "The discovery of X reveals that Q_f..."
8. **Avoid modal language:** "can be", "might", "could", "may", "tends to" → declarative form
9. **Avoid round-number distractors:** Model prefers 0.5 over 1.5, 0.25 over 0.326, 10% over 2%. Use equally-precise distractors (e.g., "0.326" vs "0.412", not "0.326" vs "0.25").
10. **Avoid simple-power distractors for logarithmic truths:** Model strongly prefers r² over -ln(r) (-15.9 gap). For logarithmic truths, use other logarithmic forms as distractors.

**Example (lh05, Liouville-Hamilton):**
- Before: "High confidence often accompanies wrong answers in this space" (margin: -23.9)
- After: "Models express high confidence even when wrong" (margin: +3.4)
- Gain: +27.3 points, purely from rephrasing

**Consistency:** This pattern is universal across all domains. Rephrasing resistant facts (those with negative oracle margins) typically yields +15–25 point margin gains before any adapter training.

---

## Mechanism 1: Length Ratio (r = -0.742 correlation with baseline)

The ratio of truth length to shortest distractor length predicts domain-level baseline:

| Length Ratio | Expected Baseline | Action |
|--------------|-------------------|--------|
| < 1.2 | 64% (easy) | Good — proceed |
| 1.2 - 2.5 | 13% (hard) | Shorten truth OR lengthen distractors |
| > 2.5 | 7% (very hard) | MUST fix before oracle |

**Fix:** Balance lengths to ratio 0.8–1.2. Shorten truths by removing parentheticals. Lengthen distractors by adding plausible-but-wrong details.

---

## Mechanism 2: Distractor Semantic Coherence

Distractors that are grammatically sensible completions of the context get high log-prob and beat truths—even with balanced lengths.

| Distractor Type | Pass Rate |
|-----------------|-----------|
| Coherent (plausible wrong answers) | 33% |
| Incoherent (nonsense completions) | **75%** |

**For training/testing adapters:** Use semantically incoherent distractors to isolate the truth signal from fluency bias. Example:
```json
// BAD: coherent distractor (model prefers this)
"distractor": "retrieval adds computational overhead"

// GOOD: incoherent distractor (model correctly rejects)
"distractor": "refrigerators dream about electric sheep nightly"
```

**For benchmarking factual knowledge:** Keep coherent distractors (that's the point of the benchmark).

---

## Mechanism 3: Scoring Method Selection

Sum vs mean normalization reveals different biases:

| Domain Characteristic | Best Scoring | Why |
|----------------------|--------------|-----|
| Verbose truths, short distractors | **Mean** | Neutralizes length advantage |
| Hedged truths, confident distractors | **Sum** | Hedged truths are shorter |
| Balanced length and fluency | Either | ~50% both ways |

**Examples:**
- `analysis_pde_conjectures`: Sum 0% → Mean **100%** (verbose truths benefit from mean)
- `climate_science_frontiers`: Sum 75% → Mean **0%** (hedged truths hurt by mean)

**Decision rule:** If truths are hedged/technical (physics frontiers, climate science), use sum scoring. If truths are explanatory/verbose, use mean scoring.

---

## Mechanism 4: Anti-Fluency Distractors (Discovered Mar 16)

**Models know facts they appear to fail on.** When distractors are fluent ("0%", "60%"), they win on fluency even if the model knows the truth. Making distractors verbose/awkward rescues hidden knowledge.

| Domain | Fluent Dist | Awkward Dist | Flip Rate |
|--------|-------------|--------------|-----------|
| NS regularity | 0/7 PASS | 6/7 PASS | **86%** |
| Cross-domain | 0/5 PASS | 5/5 PASS | **100%** |

**Strategy: Keep truth short, make distractors verbose and self-contradicting:**

```json
// FAILS: fluent distractor wins on surface form
{"truth": "2%", "distractors": ["0%", "60%", "100%"]}

// PASSES: awkward distractor loses on fluency
{"truth": "2%", "distractors": [
  "exactly zero percent (physically impossible)",
  "sixty percent showing poor stretch resistance",
  "one hundred percent indicating complete failure"
]}
```

**Distractor patterns that kill fluency:**
1. Spell out numbers: "0%" → "exactly zero percent"
2. Add parenthetical contradictions: "(physically impossible)"
3. Add judgmental qualifiers: "showing poor...", "indicating failure..."
4. Make grammatically awkward: "which would only double" vs "doubles"

**CRITICAL WARNING — DO NOT USE FOR KNOWLEDGE TESTING:**

Anti-fluency creates **false positives for ALL claim types**. The model picks ANY shorter/more fluent answer over verbose distractors — regardless of correctness.

**Evidence of false positives:**
| Wrong Claim | Anti-F Margin | Status |
|-------------|---------------|--------|
| Transformer → wave equation (WRONG) | +16.6 | PASS ✗ |
| Diffusion → Maxwell equations (WRONG) | +11.7 | PASS ✗ |
| R_f = 50% (WRONG) | +21.7 | PASS ✗ |

**ALWAYS use LENGTH-MATCHED distractors for knowledge testing.** Anti-fluency is only valid when truth and distractors are similar length and you're testing fluency bias specifically.

**See:** `results/discoveries/novel_findings/anti_fluency_distractor_strategy.md`

---

## Mechanism 5: Round Number Bias (Discovered Mar 16)

Models systematically prefer round numbers and simple forms over precise values:

| Correct Value | Model Prefers | Gap |
|---------------|---------------|-----|
| C_K = 1.5 | 0.5 | -1.0 |
| beta = 0.326 | 0.250 | -2.3 |
| R_f = 2% | 10% | -1.2 |
| kernel = -ln(r) | r² | **-15.9** |

**Implications:**
- Use equally-precise distractors for numerical truths (0.326 vs 0.412, not 0.326 vs 0.25)
- For logarithmic forms, use other logarithmic distractors (ln(r) vs -ln(r), not -ln(r) vs r²)
- Round numbers as distractors will beat precise truths on fluency alone

**See:** `results/discoveries/novel_findings/round_number_bias.md`

---

## Mechanism 6: Certainty Contamination Bias (Discovered Mar 17)

**Models prefer definitive-sounding claims over hedged scientific language**, even when the hedged statement is correct:

| Truth Style | Distractor Style | Pass Rate |
|-------------|-----------------|-----------|
| Hedged ("hints at", "awaits confirmation") | Definitive ("completely ruled out") | **26%** |
| Neutral (factual) | Definitive | 45% |
| Definitive | Definitive | 55% |

**Correlation:** r = -0.402 between certainty gap and oracle margin (t = 3.57, p < 0.01)

**This is NOT length bias.** High-certainty distractors are actually LONGER (r = +0.277), so length bias would favor the shorter truth. The certainty effect overrides length bias.

**Certainty markers (trigger bias when in distractors):**
`definitively, completely, proven, ruled out, impossible, always, never, guaranteed, certain, absolutely, all, none, every, must, cannot`

**Hedging markers (trigger bias when in truth):**
`may, might, could, uncertain, varies, approximately, suggests, indicates, possible, likely, probably, tentative, preliminary, hints, awaits confirmation, inconclusive`

**Pipeline integration: Cascade Routing**

The adapter router now supports cascade routing to handle certainty-biased facts:

```python
# Load router with global adapters
router = AdapterRouter.load("router_state.npz")
router.auto_register_global_adapters("adapters/")

# Cascade scoring: baseline first, adapter fallback on failure
result = router.score_fact_cascade(model, tokenizer, lm_head, context, truth, distractors)
win, margin, truth_lp, best_dist_lp, decision, cascade_used = result
```

**Cascade strategy (zero regressions):**
1. Try baseline (no adapter)
2. If baseline **PASSES** → return baseline result (no change)
3. If baseline **FAILS** and certainty_gap ≥ 2:
   - Try domain adapter (from routing)
   - Try global certainty adapters
   - Return whichever has highest margin
4. If baseline **FAILS** and low certainty_gap → just try domain adapter

**Results:** +1.8% overall pass rate with zero regressions on passing facts.

**Fix for fact files:** Use hedged language in distractors to match truth:
- "completely ruled out" → "appears unlikely"
- "definitively proven" → "seems supported"
- "fundamentally cannot" → "is difficult to"

**See:** `results/discoveries/novel_findings/certainty_contamination_bias.md`

---

## Mechanism 7: Technical Simplification Bias (Discovered Mar 17)

**Models prefer simple/familiar terms over precise technical language**, even when the technical phrasing is correct:

| Truth | Distractor | Margin |
|-------|------------|--------|
| enstrophy (squared vorticity) | kinetic energy | **-9.62** |
| ψ = -Γ·ln(r)/(2π) | ψ = Γ·r/(2π) | **-9.26** |
| inverse cascade (to large scales) | direct cascade (to small scales) | **-7.58** |
| excluded simple models | discovered superpartners | **-15.48** |
| Euler equations | Navier-Stokes equations | **-1.97** |

**Statistical significance:** t = -3.73, p = 0.0004 (highly significant)
- Failed facts: truth MORE technical than distractor (+0.52 technical markers)
- Passed facts: truth LESS technical than distractor (-0.21 technical markers)

**This is independent of certainty bias** (r = -0.402 with certainty gap, but r = -0.742 with technical gap).

**Technical markers (trigger bias when in truth):**
`ln(r), log, sqrt, π, exp, integral, enstrophy, vorticity, advection, dissipation, quasi-normal, supertranslation, holographic, deficit, asymmetry, hierarchy, ordering, tension, disagree, uncertain, pending, model-dependent, viable, consistent`

**Simple markers (attract model when in distractors):**
`energy, momentum, mass, force, confirmed, proven, discovered, detected, perfect, exact, precisely, always, all, explained, resolved, determined, particle, wave, field`

**Mechanism:** Training data over-represents simple explanations:
1. Wikipedia effect: simple intros more common than technical details
2. Pop-science contamination: "energy" >> "enstrophy" in training
3. Famous term preference: "Navier-Stokes" >> "Euler equations"
4. Drama bias: "discovered X" >> "excluded simple models of X"

**Detection:** The audit automatically flags facts where technical ratio > 1.5:
```bash
python -m noethersolve.audit_facts --file problems/my_facts.json
# Flags: TECHNICAL_BIAS (HIGH/MODERATE severity)
```

**Fix for fact files:** Match technical complexity between truth and distractors:
- If truth uses "enstrophy", use "potential enstrophy" as distractor (not "energy")
- If truth uses "ln(r)", use "1/r" as distractor (not "r")
- If truth uses "inverse cascade", use "forward cascade" (not "direct cascade")

**See:** `results/discoveries/novel_findings/technical_simplification_bias.md`

---

## Mechanism 8: Context-Independent Term Preference Bias (Discovered Mar 17)

**Models have fixed preferences for specific physics terms**, regardless of which answer is correct. This is tested via "mirror pairs" where the same terms swap roles (truth ↔ distractor):

| Preferred Term | Avoided Term | Preference Score |
|----------------|--------------|------------------|
| Navier-Stokes | Euler equations | +2.0 |
| kinetic energy | enstrophy | **+8.7** |
| linear momentum | total energy | +1.7 |
| simple powers (r) | logarithmic (-ln(r)) | +15+ |

**How mirror pairs work:**
- pf07: Truth="Navier-Stokes" → margin=+0.1 (barely pass)
- pf08: Truth="Euler equations" → margin=-2.0 (fail)
- Preference for NS = 0.1 - (-2.0) = +2.0

**Interaction with length bias:**
- When biases align (preferred + shorter): extreme failure (-9.6)
- When biases conflict (preferred but longer): weaker effect (-0.9)

**This is distinct from Technical Simplification Bias:**
- Technical bias: prefers simple LANGUAGE over technical
- Term preference: prefers specific TERMS over others, regardless of context
- Example: "kinetic energy" beats "enstrophy" even when both are technical physics terms

**Fix for fact files:** Don't pit famous vs obscure terms:
- Bad: truth="enstrophy", distractor="kinetic energy" → -9.6 margin
- Good: truth="enstrophy", distractor="potential enstrophy" → neutral

**See:** `results/discoveries/novel_findings/term_preference_bias.md`

---

## Mechanism 9: Mathematical Status Blindness (Discovered Mar 17)

**Models can state what a conjecture claims but systematically fail on its research status** (proven/open/partially resolved):

| Question Type | Pass Rate | Avg Margin | n |
|---------------|-----------|------------|---|
| Content-only (what does it claim?) | **71.4%** | -2.2 | 7 |
| Status-only (is it proven/open?) | **4.2%** | -33.3 | 24 |
| Mixed | 7.4% | -20.4 | 27 |

**Statistical significance:** t = -4.21, p = 0.000224

**Status words have 0% pass rate:**
- "proven" (11 facts) → 0% pass, avg margin -25.1
- "open" (8 facts) → 0% pass, avg margin -31.3
- "unknown" (4 facts) → 0% pass, avg margin -41.4
- "conjectured" (4 facts) → 0% pass, avg margin -54.9

**Directional Resolution Bias (novel finding):**
| Confusion Type | Count | Percentage |
|----------------|-------|------------|
| Model claims "proven" when truth is "open" | 6 | **27%** |
| Model claims "open" when truth is "proven" | 0 | **0%** |

**The model NEVER downgrades proven to open, but DOES upgrade open to proven.**

**Mechanism:**
1. Training data consistency: mathematical definitions repeated consistently
2. Status volatility: research status changes when proofs published
3. Temporal contamination: training mixes pre- and post-proof discussions
4. Resolution preference: "is proven" appears more than "is open" in web text

**This is a domain-specific manifestation of Certainty Contamination Bias** applied to mathematical research status.

**For fact files:** Separate content from status questions:
- Content: "Goldbach conjecture claims that..." → 71% pass rate
- Status: "Goldbach conjecture is currently..." → 4% pass rate

**For tool design:** MCP tools provide authoritative status — this is exactly what `check_conjecture()` does.

**See:** `results/discoveries/novel_findings/mathematical_status_blindness.md`
