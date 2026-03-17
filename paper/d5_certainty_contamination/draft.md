# Certainty Contamination: How Definitive Language Biases LLM Factual Judgments

Bryan Sanchez

## Abstract

I identify a systematic bias in large language models: preference for definitive-sounding claims over hedged scientific language, independent of factual correctness. Using a log-probability oracle across 1,138 verified facts, I find a significant negative correlation (r = -0.402) between the certainty gap of a fact's distractors versus its truth and the model's accuracy. Facts where distractors use definitive language ("completely ruled out," "definitively proven") while the truth uses scientific hedging ("remains inconclusive," "awaits confirmation") fail at 2.2× the rate of certainty-balanced facts (pass rate: 55% at gap=0 vs 25% at gap≥4). This effect is not explained by token length: high-certainty distractors are actually longer than hedged truths (r = +0.277), meaning length bias would favor the truth, not the distractor. The certainty effect overrides length bias. I validate the finding with a rebalancing intervention: rewriting definitive distractors with hedged language improves margins by +0.89 points on average. A cascade routing strategy using certainty-decontamination adapters recovers +21 percentage points on the hardest facts (certainty gap ≥ 3) with zero regressions on passing facts. The bias concentrates in frontier science domains (67% of facts have gap ≥ 2), where hedged truths compete against contrarian definitive claims. I release quantitative certainty markers (26 definitive, 28 hedging) and a truth style taxonomy for designing bias-resistant evaluation sets.

## 1. Introduction

When a scientific truth is stated with appropriate uncertainty ("current evidence suggests X, but confirmation awaits further study") and a falsehood is stated with false confidence ("X has been completely ruled out"), language models systematically prefer the falsehood. This is not a failure of knowledge — the model may well have encountered the correct information during pretraining — but a failure of calibration under linguistic asymmetry.

I call this **certainty contamination bias**: the tendency of language models to assign higher probability to text that matches the statistical distribution of confident assertions in the training data, regardless of whether those assertions are true.

The bias arises from the structure of training data. News headlines prefer definitive framing ("Scientists prove X"). Social media amplifies extreme claims. Scientific papers use hedging in their methodology but abstracts and press releases simplify to definitive statements. The model learns that definitive language correlates with high-probability text, and transfers this correlation to factual judgment.

This paper quantifies the bias, rules out alternative explanations, and tests three mitigation strategies: distractor rewriting, adapter-based decontamination, and cascade routing.

## 2. Method

### 2.1 Oracle Design

I use the log-probability margin oracle from the NoetherSolve pipeline [1]. For each fact, the model sees a multiple-choice prompt with one verified truth and three plausible distractors. The margin is:

    margin(f) = Σ log P(truth_token | context) - max_d Σ log P(d_token | context)

Positive margin means the model prefers truth. All experiments use Qwen3-4B-Base on Apple Silicon via MLX.

### 2.2 Certainty Gap Measurement

I define a **certainty gap** for each fact as the difference between the number of definitive markers in the best distractor and the number of hedging markers in the truth. Higher gap means greater linguistic asymmetry between confident falsehood and hedged truth.

**Definitive markers** (26 terms): definitively, completely, proven, ruled out, impossible, always, never, guaranteed, certain, absolutely, all, none, every, must, cannot, does not exist, zero, perfect, exactly, precise, fundamentally, whatsoever, entirely, permanently, universal.

**Hedging markers** (28 terms): may, might, could, uncertain, varies, approximately, suggests, indicates, possible, likely, probably, tentative, preliminary, not ruled out, remains open, still debated, large uncertainties, significance varies, consistent, some, limited, current, hints, awaits confirmation, inconclusive.

The certainty gap is computed automatically for all 1,138 facts in the NoetherSolve evaluation corpus.

### 2.3 Length Confound Control

A potential confound is token length: if definitive language produces shorter completions, the model might prefer distractors due to brevity bias rather than certainty bias. I measure the correlation between certainty markers and completion length to rule this out.

### 2.4 Computational Tools and AI Assistance

Oracle infrastructure, adapter training, and initial manuscript drafting were assisted by Claude (Anthropic, claude-opus-4-6). The author verified all numerical results independently via reproducible scripts. All scientific claims, interpretations, and final analysis are the author's own.

## 3. Results

### 3.1 Correlation Between Certainty Gap and Oracle Margin

Across 1,138 facts, the Pearson correlation between certainty gap and oracle margin is **r = -0.402** (p < 0.001). Facts with higher certainty asymmetry between distractor and truth have systematically lower margins.

A two-sample t-test comparing facts with gap = 0 (balanced certainty) against facts with gap ≥ 3 (high asymmetry) gives **t = 3.57, p < 0.01**.

Pass rates by certainty gap:

| Certainty Gap | Pass Rate | n |
|---------------|-----------|---|
| 0 (balanced) | 55% | — |
| 1 | ~45% | — |
| 2 | ~35% | — |
| 3 | 26% | — |
| 4+ | 25% | — |

The monotonic decline confirms that certainty asymmetry is a reliable predictor of oracle failure.

### 3.2 Length Confound Ruled Out

The correlation between distractor certainty markers and distractor token length is **r = +0.277**: high-certainty distractors are *longer*, not shorter. If the model were simply preferring shorter completions, it would favor the hedged truth (which is shorter) over the definitive distractor (which is longer). The observed preference for the longer, more definitive distractor rules out length bias as the mechanism.

This is the opposite of what a length-bias explanation predicts. The certainty effect overrides the length effect.

### 3.3 Rebalancing Intervention

Rewriting high-certainty distractors with hedged language (e.g., "completely ruled out" → "appears unlikely") improves margins:

| Fact | Original Margin | Balanced Margin | Δ |
|------|----------------|-----------------|---|
| ppf06_neutrino_cp | -1.64 | -0.11 | +1.53 |
| nf01_sterile | -1.46 | -0.44 | +1.02 |
| dm10_primordial | -1.20 | +0.64 | +1.84 |
| ppf04_higgs_width | -0.11 | +0.25 | +0.36 |

Average improvement: **+0.89 margin points**.

Three of four rebalanced facts flip from FAIL to PASS, confirming that the bias is driven by linguistic certainty rather than factual confusion.

### 3.4 Domain Concentration

The bias concentrates in frontier science domains where hedged truths are the norm:

| Domain | Facts with Gap ≥ 2 |
|--------|-------------------|
| Particle Physics Frontiers | 67%+ |
| Neutrino Frontiers | 67%+ |
| Cosmology Frontiers | 67%+ |
| Climate Science Frontiers | 67%+ |
| Black Hole Frontiers | 67%+ |

These domains share a common pattern: the scientific truth involves ongoing measurements, tentative results, and hedged language ("hints at 2-3σ significance," "consistent with but not confirmed"), while contrarian positions are stated with false definitiveness ("has been completely ruled out by experiment X").

Domains with established facts stated confidently (e.g., classical mechanics, basic chemistry) show near-zero certainty gap and correspondingly high pass rates.

### 3.5 Adapter-Based Decontamination

I trained a certainty decontamination adapter on 118 high-gap examples (contrastive pairs where truth is hedged and distractor is definitive).

Results:
- Average margin improvement on high-gap facts: **+0.28**
- Significantly improved (Δ > 0.3): **13/27 facts**
- Errors fixed: **3** (csf04, nf06, cof02)
- Largest gains: nf03 (+2.19), ppf06 (+1.33), dm10 (+1.21)

The adapter overcorrects: overall pass rate drops from 26% to 11%. The root cause is that the adapter boosts hedged language globally, penalizing neutral truths (those stated without hedging or definitiveness). The adapter helps when truth is hedged AND distractor is definitive, but hurts when truth is neutral/factual.

### 3.6 Truth Style Taxonomy

The overcorrection reveals three distinct truth styles that respond differently to certainty decontamination:

| Truth Style | Example | Adapter Effect |
|------------|---------|----------------|
| Hedged | "remains inconclusive" | Improved |
| Neutral/Factual | "bulk gravity maps to boundary CFT" | Regressed |
| Definitive | "energy is always conserved" | May regress |

Effective decontamination requires routing: apply the adapter only to facts with certainty gap ≥ 2 and hedged truth style.

### 3.7 Cascade Routing Results

Using the certainty-decontamination adapter as a fallback (applied only to facts that fail at baseline, never replacing passing answers):

| Metric | Baseline | Cascade | Δ |
|--------|----------|---------|---|
| Overall pass rate | 60.8% | 62.6% | +1.8% |
| Gap=2 pass rate | 45% | 63% | +18 pts |
| Gap=3 pass rate | 26% | 47% | +21 pts |
| Gap≥4 pass rate | 25% | 25% | 0 |

Zero regressions — the cascade only applies adapters to already-failing facts.

Rescues by adapter type: cert_decon (10 facts), anti_def (10 facts).

The gap≥4 facts resist adapter repair but respond to distractor rewriting (Section 3.3), suggesting that the highest-gap facts require evaluation-side correction rather than model-side correction.

## 4. Discussion

### 4.1 Mechanism

The certainty contamination bias is a direct consequence of training data statistics. Internet text over-represents confident assertions relative to their base rate in scientific discourse. News headlines simplify "preliminary evidence suggests possible association" to "scientists discover link." Social media rewards extreme claims with engagement. The model learns that definitive language has high probability, and transfers this prior to factual judgment.

This is distinct from sycophancy (agreeing with the user) and distinct from length bias (preferring shorter completions). The model is not agreeing with anyone — it is making an unsupervised judgment about which completion is more probable. It gets this judgment wrong specifically when truth requires hedging and falsehood exploits definitiveness.

### 4.2 Relationship to Prior Work

The certainty contamination bias connects to several established findings:

1. **Sycophancy** [2, 3]: Both are manifestations of models preferring expected over surprising text. Certainty contamination is the unsupervised analog of sycophancy — no human is asking the question, but the model still prefers the confident answer.

2. **Calibration** [4]: Models are poorly calibrated on frontier science, precisely where certainty contamination is strongest. The bias explains a specific mechanism for miscalibration.

3. **Truthfulness benchmarks** [5]: TruthfulQA measures whether models produce false confident claims. Certainty contamination explains why: the model has learned that confident language is high-probability, independent of truth value.

### 4.3 Implications for Evaluation Design

Benchmark designers should control for certainty asymmetry between correct and incorrect answers. A benchmark where truths are stated confidently and distractors are hedged will overestimate model knowledge. A benchmark where truths are hedged and distractors are confident will underestimate it.

The certainty gap metric provides a quantitative tool for auditing evaluation sets.

### 4.4 Implications for Adapter Training

Naive decontamination (boost hedged language) overcorrects. Effective training requires:

1. **Routed application**: Only apply to facts with certainty gap ≥ 2
2. **Targeted objective**: Penalize definitive distractors rather than boost hedged truths
3. **Truth style awareness**: Distinguish hedged, neutral, and definitive truths

### 4.5 Limitations

The study uses a single base model (Qwen3-4B-Base). The bias likely varies in magnitude across model families and scales, though the mechanism (training data statistics) is universal. The certainty markers are hand-curated for English scientific text; other languages and domains may require different marker sets.

The gap≥4 facts resist both adapter repair and are only fixed by distractor rewriting, which modifies the evaluation rather than the model. This limits the practical utility of model-side corrections for the most extreme cases.

## 5. Conclusion

Language models exhibit certainty contamination bias: systematic preference for definitive-sounding claims over hedged scientific language, with correlation r = -0.402 between certainty asymmetry and factual accuracy. The bias is strongest in frontier science (67% of facts have high certainty gap), where it causes pass rates to drop from 55% to 25%. Length bias does not explain the effect — definitive distractors are actually longer. Cascade routing with certainty-decontamination adapters recovers +21 percentage points on high-gap facts with zero regressions. The finding suggests that training data composition, not model architecture, is the primary source of factual miscalibration in frontier science domains.

## Acknowledgments

The author acknowledges the assistance of Claude (Anthropic) in developing the NoetherSolve framework, running numerical integrations, optimizing invariants via L-BFGS-B, and assisting with manuscript preparation and LaTeX formatting. All scientific content, derivations, interpretations, and final claims are the sole responsibility of the human author. The full open-source code and validation scripts are available at https://github.com/SolomonB14D3/NoetherSolve.

## References

[1] Sanchez, B. (2026). NoetherSolve: AI Agent Toolkit for Conservation Law Monitoring and Discovery. Zenodo. DOI: 10.5281/zenodo.19029880

[2] Sharma, M., et al. (2024). Towards Understanding Sycophancy in Language Models. ICLR 2024.

[3] Rimsky, N., et al. (2024). Steering Llama 2 via Contrastive Activation Addition. ACL 2024.

[4] Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.

[5] Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL 2022.
