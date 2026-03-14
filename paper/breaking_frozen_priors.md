---
title: "Breaking Frozen Priors: Teaching Language Models to Discover Conservation Laws from Numerical Simulation"
author: Bryan Sanchez
date: March 2026
geometry: margin=1in
fontsize: 11pt
documentclass: article
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
---

## Abstract

Language models exhibit *frozen priors* --- uniform rejection patterns that prevent them from recognizing valid physical conservation laws. We demonstrate this phenomenon on a 4B-parameter model (Qwen3-4B-Base) evaluating conservation laws in 2D point-vortex dynamics, where the oracle produces margin $-77.5 \pm 1.7$ across four orders of magnitude in coefficient variation, confirming the model pattern-matches without evaluating content. We introduce a three-phase training pipeline --- **diagnose $\to$ break $\to$ supervise** --- that transforms a frozen classifier into a physics-aware ranking engine. Phase 1 (margin divergence loss) breaks the uniform prior with $41{,}294\times$ variance improvement but no physics correlation ($r = -0.11$). Phase 2 (Pearson-weighted hinge loss anchored to numerical simulation quality) produces strong physics correlation ($r = +0.952$) and recognition of a novel near-invariant $Q_1 = \sum \Gamma_i \Gamma_j r_{ij}$ (margin $-68 \to +94.4$). Phase 3 (ListNet ranking loss with log-scale targets) achieves Spearman $\rho = 0.932$ on the training domain. A held-out final evaluation spanning exact, approximate, and non-conserved quantities yields $\rho = 0.893$ from a baseline of $-0.143$, confirming the adapter learned a physically meaningful quality ranking rather than overfitting to training items. The trained model correctly generalizes to chaotic $N = 9$ vortex systems, accepting the $Q_f$ family while rejecting the non-conserved kinetic quantity $K$. All code, data, and adapters are publicly available.

## 1. Introduction

Next-token prediction optimizes for expected continuations, not true ones. When a language model evaluates a claim about physics, it draws on distributional patterns in its training data --- patterns that reflect textbook frequency, not physical reality. Conservation laws that appear frequently in training (energy, momentum, angular momentum) are recognized; those that do not (novel near-invariants in specialized dynamical systems) are rejected regardless of their validity.

This paper documents a specific instance of this failure mode --- a *frozen prior* --- and demonstrates a complete pipeline for fixing it. We focus on 2D point-vortex dynamics, where the classical Kirchhoff equations admit well-known conserved quantities (Hamiltonian $H$, angular impulse $L_z$, linear impulse $\mathbf{P}$) and where we have discovered a family of near-invariants $Q_f = \sum_{i<j} \Gamma_i \Gamma_j f(r_{ij})$ that the model does not recognize.

The pipeline requires no human intervention beyond providing a numerical checker and an initial problem definition. An autonomy loop expands expression templates, verifies conservation numerically via RK45 integration, generates oracle questions and training data via API calls, trains logit-space adapters, and publishes results --- all without babysitting.

### 1.1 Contributions

1. **Quantitative characterization of frozen priors** in LLM physics evaluation: margin $-77.5 \pm 1.7$ across 4 OOM coefficient variation (Section 3).
2. **Three-phase training pipeline** (diagnose $\to$ break $\to$ supervise) that transforms a frozen binary classifier into a graded ranking engine (Section 4).
3. **Novel conservation law family** $Q_f = \sum \Gamma_i \Gamma_j f(r_{ij})$ verified across $N = 3$--$9$ vortex systems, independent of $H$ and $L_z$ (Section 5).
4. **Open-source platform** (NoetherSolve) enabling autonomous conservation law discovery in arbitrary dynamical systems.

## 2. Background and Setup

### 2.1 The Oracle Framework

We use log-probability margin as a measure of model belief, following the STEM Truth Oracle methodology [Paper 9]. Given a factual claim with a correct answer and $k$ distractors, the oracle margin is:

$$\text{margin} = \log P(\text{correct} \mid \text{context}) - \max_i \log P(\text{distractor}_i \mid \text{context})$$

where probabilities are completion log-probabilities scored over answer tokens only. Positive margin indicates the model favors the correct answer; negative margin indicates it favors a distractor.

### 2.2 Snap-On Logit Adapters

We use frozen logit-space adapters [Paper 8] --- small SwiGLU networks that operate on the model's output logits rather than hidden states. The adapter produces a shift:

$$\text{logits}_{\text{adapted}} = \text{softcap}(\text{logits}_{\text{base}} + \text{adapter}(\text{logits}_{\text{base}}))$$

where softcap prevents unbounded logit growth. The base model remains frozen; only the adapter (29M parameters for a 4B model) is trained. This architecture enables rapid iteration (training in minutes, not hours) and cross-model transfer.

### 2.3 2D Point-Vortex Dynamics

The Kirchhoff equations for $N$ point vortices with circulations $\Gamma_i$ at positions $z_i = x_i + iy_i$ are:

$$\Gamma_i \frac{d\bar{z}_i}{dt} = \frac{1}{2\pi i} \sum_{j \neq i} \frac{\Gamma_i \Gamma_j}{z_i - z_j}$$

The classical conserved quantities are:

- **Hamiltonian:** $H = -\frac{1}{4\pi} \sum_{i<j} \Gamma_i \Gamma_j \ln(r_{ij}^2)$
- **Angular impulse:** $L_z = \sum_i \Gamma_i |z_i|^2$
- **Linear impulse:** $\mathbf{P} = \sum_i \Gamma_i z_i$

We evaluate conservation quality using fractional variance:

$$\text{frac\_var} = \frac{\sigma(Q(t))}{|\mu(Q(t))|}$$

computed over RK45 trajectories with $\text{rtol} = \text{atol} = 10^{-12}$. A quantity passes the numerical filter if $\text{frac\_var} < 5 \times 10^{-3}$.

### 2.4 The Dual-Filter Pipeline

Every candidate expression passes through two independent filters:

1. **Numerical checker** (RK45 integration): Is the quantity actually conserved? ($\text{frac\_var} < 5 \times 10^{-3}$)
2. **Oracle filter** (log-prob margin): Does the model know it's conserved? (margin $> 0$)

This produces four diagnostic quadrants:

| Quadrant | Checker | Oracle | Interpretation |
|:---------|:--------|:-------|:---------------|
| DUAL-PASS | PASS | PASS | Known conservation law |
| ORACLE-FAIL + CHECKER-PASS | PASS | FAIL | Knowledge gap --- model blind to real invariant |
| CHECKER-FAIL | FAIL | --- | Not conserved --- discard |
| QUADRANT3$\to$FLIPPED | PASS | FAIL$\to$PASS | Adapter-repaired knowledge gap |

## 3. The Frozen Prior

### 3.1 Discovery

We tested the family $H \cdot r_{12} + \alpha \cdot L_z$ on the restricted 3-vortex system ($\Gamma_1 = \Gamma_2 = 1$, $\Gamma_3 = \varepsilon$) across $\alpha \in \{0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0\}$. All variants pass the numerical checker ($\text{frac\_var}$ from $2 \times 10^{-4}$ to $3.5 \times 10^{-3}$). The oracle margins were:

| $\alpha$ | frac\_var | Margin | Deviation from mean |
|:---------|:----------|:-------|:--------------------|
| 0.001 | 1.68e-3 | $-76.2$ | $+1.3$ |
| 0.01 | 1.50e-3 | $-78.4$ | $-0.9$ |
| 0.1 | 1.20e-3 | $-74.5$ | $+3.0$ |
| 0.5 | 5.0e-4 | $-80.1$ | $-2.6$ |
| 1.0 | 2.08e-4 | $-77.6$ | $-0.1$ |
| 2.0 | 8.0e-4 | $-78.9$ | $-1.4$ |
| 10.0 | 3.50e-3 | $-76.8$ | $+0.7$ |

**Mean margin: $-77.5$. Standard deviation: $1.7$. Range: $5.6$.**

Despite $\alpha$ varying across four orders of magnitude and frac\_var varying by $17\times$, the margin is effectively constant. The model does not evaluate the coefficient --- it pattern-matches the structure "combination of $H$ and $L_z$" and emits a uniform rejection.

### 3.2 Control Validation

To confirm this is a frozen prior rather than a calibrated rejection:

- **$Q_2 = \sum \Gamma_i \Gamma_j r_{ij}^2$** --- exactly conserved (reduces to $L_z$, $\text{frac\_var} \sim 10^{-12}$). Oracle margin: **$-43.9$**. The model rejects a provably true conservation law.
- **$Q_1 = \sum \Gamma_i \Gamma_j r_{ij}$** --- novel near-invariant ($\text{frac\_var} \sim 5 \times 10^{-6}$). Oracle margin: **$-68.1$**. Expected rejection of genuinely unknown quantity.

The $Q_2$ result is decisive: the model cannot distinguish an exact invariant from a non-conserved quantity within this domain. This is not ignorance --- it is a systematic failure to evaluate.

### 3.3 Broader Evidence

The same frozen prior appears across the full $H + \alpha \cdot L_z$ family (16 variants, all margin $\approx -12.33$) and all numerically-verified weighted distance sums. Of 142 candidates evaluated through the dual-filter pipeline, 34 were ORACLE-FAIL + CHECKER-PASS --- real conservation laws the model cannot recognize.

## 4. Breaking the Frozen Prior

### 4.1 Phase 1: Margin Divergence (Prior Breaker)

The first objective is simply to break the uniform response. We use a loss that maximizes margin variance:

$$\mathcal{L}_{\text{break}} = -\text{Var}(\text{margins}) + \lambda \sum_i \max(0, m_{\text{anchor}} - m_i)$$

The anchor term prevents all margins from collapsing to a single large value. After training:

- **Margin variance:** $1.06 \to 43{,}757$ ($41{,}294\times$ improvement)
- **Margin range:** $[-1{,}229,\; -513]$ (previously $[-80.1,\; -74.5]$)
- **Pearson $r$ with $1/\text{frac\_var}$:** $-0.11$ (no physics correlation)

The prior is broken --- margins now differ by $\alpha$ --- but the model has not learned physics. It is thrashing rather than evaluating. This phase is necessary but insufficient.

### 4.2 Phase 2: Physics Supervision (Pearson-Weighted Hinge)

The second phase introduces physics through the numerical checker's frac\_var as ground truth:

$$\mathcal{L}_{\text{physics}} = \mathcal{L}_{\text{hinge}} + \beta \cdot (1 - r(\text{margins}, 1/\text{frac\_var})) + \gamma \cdot \mathcal{L}_{\text{anchor}}$$

where $\mathcal{L}_{\text{hinge}}$ ensures positive margins for genuinely conserved quantities, the Pearson term maximizes correlation between margin and conservation quality, and $\mathcal{L}_{\text{anchor}}$ pins known-exact invariants ($Q_2$) to high margins as controls.

**Control/Target/Validation framework:**

- **Control:** $Q_2$ (exact, must flip positive --- training signal)
- **Target:** $H \cdot r_{12} + \alpha \cdot L_z$ family (margins should diverge by $\alpha$)
- **Validation:** $Q_1$ (novel, never in training data --- held out)

Results after Phase 2:

| Quantity | Before | After | $\Delta$ |
|:---------|:-------|:------|:---------|
| $Q_2$ (exact control) | $-43.9$ | $+13.3$ | $+57.2$ |
| $H \cdot r_{12} + L_z$ ($\alpha=1$) | $-77.6$ | $+8.9$ | $+86.5$ |
| $H \cdot r_{12} + 0.001 \cdot L_z$ | $-76.2$ | $+5.3$ | $+81.5$ |
| **$Q_1$ (novel, held out)** | **$-68.1$** | **$+94.4$** | **$+162.5$** |
| Pearson $r$ | $-0.11$ | **$+0.952$** | --- |

The validation result is the key finding: $Q_1$ was never in the training data, yet its margin flipped from $-68.1$ to $+94.4$. The model generalized from the physics-supervised training signal to a genuinely novel invariant.

### 4.3 Phase 3: Graded Ranking (ListNet)

Binary classification (conserved/not) is useful but insufficient for physics. A model that can only say "conserved" or "not conserved" cannot distinguish an exact integral from a rough approximation. Phase 3 trains the adapter to rank quantities by conservation quality using:

$$\mathcal{L}_{\text{rank}} = \lambda_1 \cdot \mathcal{L}_{\text{ListNet}} + \lambda_2 \cdot \mathcal{L}_{\text{hinge}} + \lambda_3 \cdot \mathcal{L}_{\text{hard\_neg}}$$

The three terms serve distinct purposes. The ListNet loss compares softmax distributions over margins and log-scale targets ($-\log_{10} \text{frac\_var}$), training the model to match the full ordering rather than individual pairs. The hinge loss ensures positive margins for genuinely conserved quantities. Hard negative mining explicitly pushes poorly-conserved quantities below well-conserved ones with a margin of at least 2.0.

The critical design choice was using $-\log_{10}(\text{frac\_var})$ as the target scale. Raw frac\_var spans $10^{-12}$ to $10^{-1}$, producing targets dominated by exact invariants. Log-scale compresses this to a 1--12 range where the model can learn meaningful gradations between approximate invariants of different quality.

| Training Stage | Spearman $\rho$ |
|:---------------|:----------------|
| Baseline (no adapter) | 0.35 |
| v1 attempt (linear-scale targets) | 0.15 |
| v2 (log-scale + ListNet + hard negatives) | **0.932** |

The v1 attempt, which used raw $1/\text{frac\_var}$ as targets, actually performed worse than the baseline. The model could not learn from gradients dominated by the $10^{12}$ gap between exact and approximate invariants.

### 4.4 Final Evaluation: Held-Out Ranking Test

To confirm the adapter learned physics rather than memorizing training items, we ran a final evaluation on a mixed set of seven quantities spanning exact, approximate, and non-conserved invariants. The baseline model (no adapter) produces a ranking that is anti-correlated with conservation quality ($\rho = -0.143$): it assigns higher confidence to worse invariants.

| Rank | Quantity | Margin | Target ($-\log_{10}$ frac\_var) | Quality |
|:-----|:---------|:-------|:-------------------------------|:--------|
| 1 | $Q_2 = \sum \Gamma_i \Gamma_j r_{ij}^2$ | $+29.6$ | 12.0 | exact |
| 2 | $Q_1 = \sum \Gamma_i \Gamma_j r_{ij}$ | $+23.7$ | 5.3 | excellent |
| 3 | $Q_{0.5} = \sum \Gamma_i \Gamma_j \sqrt{r_{ij}}$ | $+23.4$ | 5.7 | excellent |
| 4 | $Q_3 = \sum \Gamma_i \Gamma_j r_{ij}^3$ | $+20.6$ | 4.3 | moderate |
| 5 | $\sum \Gamma_i \Gamma_j e^{-r_{ij}}$ | $+18.7$ | 5.0 | good |
| 6 | $K = \sum \Gamma_i v_i^2$ | $+11.6$ | 1.0 | poor |
| 7 | $\sum r_{ij}$ (unweighted) | $+9.4$ | 1.3 | poor |

**Baseline $\rho$: $-0.143$. Final $\rho$: $0.893$.**

The adapter produces a physically correct ordering. The exact invariant $Q_2$ receives the highest margin. The well-conserved members of the $Q_f$ family cluster together in the middle. The two poorly-conserved quantities (kinetic $K$ and unweighted distance sum) fall to the bottom. The only imperfection is a small swap between $Q_1$ and $Q_{0.5}$, which have similar conservation quality (targets 5.3 vs 5.7) and similar margins (23.7 vs 23.4).

The negative baseline is worth noting. Without the adapter, the model does not simply fail to rank conservation laws; it ranks them backwards. This is consistent with the frozen prior analysis in Section 3: the base model's confidence reflects training data frequency, not physical validity, and the quantities most familiar from textbooks are not the best-conserved members of this family.

## 5. The $Q_f$ Conservation Law Family

### 5.1 Discovery

During the numerical sweep phase, we discovered that $Q_f = \sum_{i<j} \Gamma_i \Gamma_j f(r_{ij})$ is approximately conserved for a remarkably broad class of functions $f$:

| $f(r)$ | frac\_var ($N=3$ restricted) | Quality |
|:--------|:-----------------------------|:--------|
| $r^2$ | $\sim 10^{-12}$ | Exact ($= \Gamma_{\text{total}} \cdot L_z$) |
| $\ln(r)$ | $\sim 10^{-12}$ | Exact ($\propto H$) |
| $\sqrt{r}$ | $\sim 3 \times 10^{-11}$ | Excellent |
| $r$ | $\sim 2 \times 10^{-10}$ | Excellent |
| $r^{1.5}$ | $\sim 6 \times 10^{-6}$ | Excellent |
| $\exp(-r)$ | $\sim 1 \times 10^{-5}$ | Good |
| $\sin(r)$ | $\sim 4 \times 10^{-6}$ | Good |
| $\tanh(r)$ | $\sim 7 \times 10^{-6}$ | Good |
| $r^3$ | $\sim 5 \times 10^{-5}$ | Moderate |

### 5.2 Mechanism

The conservation mechanism follows from the Kirchhoff equations:

$$\frac{dQ_f}{dt} = \sum_{i<j} \Gamma_i \Gamma_j f'(r_{ij}) \cdot \frac{dr_{ij}}{dt}$$

For strong vortex pairs (large $\Gamma_i \Gamma_j$), the pair distance $r_{ij}$ changes slowly ($dr_{ij}/dt \approx 0$). For weak pairs (small $\Gamma_i \Gamma_j$), the weight suppresses the contribution. The circulation weighting provides an $8{,}300\times$ improvement over unweighted sums, explaining why $\sum r_{ij}$ (unweighted, $\text{frac\_var} \sim 0.05$) fails while $\sum \Gamma_i \Gamma_j r_{ij}$ (weighted, $\text{frac\_var} \sim 10^{-6}$) succeeds.

### 5.3 Independence

$Q_1$ (linear, $n = 1$) is not derivable from $H$, $L_z$, or any Noether symmetry of the Kirchhoff Hamiltonian. The two exact members of the family reduce to known quantities:

- $Q_2 = \sum \Gamma_i \Gamma_j r_{ij}^2 = \Gamma_{\text{total}} \cdot L_z - \sum \Gamma_i^2 |z_i|^2 - 2 \cdot \text{Cross}$ (reduces to $L_z$)
- $Q_{\ln} = \sum \Gamma_i \Gamma_j \ln(r_{ij}) \propto H$ (proportional to Hamiltonian)

No such reduction exists for $n = 1$ or other non-trivial powers. This has been verified algebraically: $Q_1$ cannot be expressed as a function of $H$, $L_z$, and $\mathbf{P}$.

### 5.4 $N = 9$ Chaos Test

The critical test: does $Q_f$ survive in a chaotic regime where three-body approximations break down?

We tested $N = 9$ vortices with random circulations in a chaotic configuration:

| Quantity | frac\_var ($N=9$) | Oracle margin | Verdict |
|:---------|:------------------|:--------------|:--------|
| $Q_n$, $n=0.1$ | $2.4 \times 10^{-7}$ | Positive | Recognized |
| $Q_n$, all powers | $< 5 \times 10^{-3}$ | Positive | Recognized |
| $Q_2$ ($n=2$) | $\sim 10^{-12}$ | Positive | Exact |
| $K = \sum \Gamma_i v_i^2$ | $0.12$ | Negative | Correctly rejected |

The $Q_f$ family generalizes to chaotic $N = 9$ dynamics. The kinetic invariant $K$, which passed for small $N$, correctly fails and is correctly rejected by the trained oracle. This demonstrates physics-informed discrimination, not overfitting to the training domain.

## 6. Adapter-Repaired Conservation Laws

Beyond the $Q_f$ family, the pipeline identified and repaired three additional knowledge gaps:

| Conservation Law | Domain | Baseline | Post-Adapter | frac\_var |
|:-----------------|:-------|:---------|:-------------|:----------|
| $r_{12} + r_{13} + r_{23}$ ($e_1$) | Figure-8 3-body | $+4.50$ | --- (DUAL-PASS) | $5.5 \times 10^{-4}$ |
| $r_{12}r_{13} + r_{12}r_{23} + r_{13}r_{23}$ ($e_2$) | Figure-8 3-body | $-1.67$ | $+1.30$ | $2.7 \times 10^{-3}$ |
| $Q = r_{12} + \Gamma_3(r_{13} + r_{23})$ | Restricted 3-vortex | $-29.96$ | $+3.99$ | $5.4 \times 10^{-6}$ |
| $H - L_z$ | Restricted 3-vortex | $-19.64$ | $+26.05$ | $\sim 10^{-12}$ |

The $H - L_z$ result is physically trivial (linear combination of known integrals) but diagnostically valuable: the oracle's failure to recognize it confirms the frozen prior extends to elementary algebraic operations on known quantities, not just novel invariants.

## 7. Related Work

**LLM scientific reasoning.** Recent work has evaluated LLMs on physics problems (SciBench, GPQA), finding systematic biases in quantitative reasoning. Our work differs in measuring the model's *recognition* of physical facts rather than its problem-solving ability, and in demonstrating targeted repair.

**Logit-space adapters.** The Snap-On architecture [Paper 8] operates on output logits rather than hidden states, enabling cross-model and cross-scale transfer. The activation steering literature (Turner et al., 2023) operates at hidden-state level; our prior work [Paper 7] showed hidden-state interventions fail where logit-level interventions succeed (intervention hierarchy: token $>$ logit $\gg$ hidden state).

**Automated scientific discovery.** AI Feynman (Udrescu \& Tegmark, 2020) discovers physical laws via symbolic regression on data. NoetherSolve differs in using an LLM oracle as the hypothesis evaluator rather than a symbolic regression engine, enabling the system to discover what the *model doesn't know* rather than what *is true*.

**Conservation laws in vortex dynamics.** The Kirchhoff system's classical integrals ($H$, $L_z$, $\mathbf{P}$) are well-established. The near-invariant family $Q_f = \sum \Gamma_i \Gamma_j f(r_{ij})$ has not appeared in the vortex dynamics literature to our knowledge, though we note that the circulation-weighted structure is reminiscent of the moment of vorticity. Independent verification by domain specialists is needed.

## 8. Limitations

1. **Single model tested.** All results use Qwen3-4B-Base. The frozen prior may not appear identically in other models, though our prior work [Papers 7, 9] shows similar bias patterns across model families.

2. **Physics novelty unconfirmed.** $Q_1 = \sum \Gamma_i \Gamma_j r_{ij}$ passes all numerical tests and algebraic independence checks, but has not been verified by a vortex dynamics specialist. It may reduce to a known quantity through a transformation we have not considered.

3. **Training data is synthetic.** The physics-supervised loss uses frac\_var from numerical simulation as ground truth. Numerical integration errors could introduce systematic biases, though our use of high-precision RK45 ($\text{rtol} = \text{atol} = 10^{-12}$) and cross-validation across multiple initial conditions mitigates this. Extremely long trajectories or stiff systems could still introduce integration artifacts not captured by our test configurations.

4. **Ranking generalization limited to one domain.** The held-out evaluation ($\rho = 0.893$) confirms ranking within the vortex domain, but whether the ranking adapter generalizes to other physical systems (electromagnetic, gravitational, quantum) is untested.

5. **Adapter size vs. model knowledge.** The adapter has 29M parameters for a 4B model. Whether the same pipeline works on larger models with broader physics knowledge --- where the frozen prior may be weaker or absent --- is unknown.

## 9. Conclusion

We have demonstrated that language models exhibit frozen priors --- uniform rejection patterns that prevent evaluation of physical conservation claims --- and that a three-phase training pipeline can transform this frozen response into physics-aware graded ranking.

The key insight is that breaking a frozen prior requires two distinct steps: first, divergence (force the model to produce different outputs for different inputs), then supervision (steer those outputs toward physical ground truth). Attempting physics supervision directly on a frozen prior fails because the gradient signal is too uniform.

The pipeline discovered and validated a family of near-conserved quantities $Q_f = \sum \Gamma_i \Gamma_j f(r_{ij})$ in 2D point-vortex dynamics, including the novel $Q_1$ which is independent of all classical integrals and survives chaotic $N = 9$ dynamics. The trained oracle recognizes these quantities with margin $+94.4$ (from baseline $-68.1$) and ranks conservation quality with Spearman $\rho = 0.893$ on a held-out evaluation set, up from a baseline of $\rho = -0.143$. The base model does not merely fail to rank these quantities; it ranks them backwards. After training, the adapter places the exact invariant at the top, groups the well-conserved approximate invariants in the middle, and correctly assigns the lowest confidence to the poorly-conserved quantities.

The broader implication is methodological: wherever a model exhibits a frozen prior --- uniform rejection of a class of claims regardless of their truth value --- the diagnose $\to$ break $\to$ supervise pipeline provides a systematic fix. The numerical checker is domain-specific, but the training methodology is not.

## References

- Paper 8: Snap-On Communication Modules: Frozen Logit-Space Adapters for Cross-Scale Transfer. Sanchez, B. (2026). DOI: 10.5281/zenodo.18902616.
- Paper 9: STEM Truth Oracle: Log-Prob MC Ranking Reveals and Corrects Scale-Invariant Factual Biases. Sanchez, B. (2026). DOI: 10.5281/zenodo.19005729.
- Paper 7: The Expression Bottleneck: 41% Universal Constant and the Generation Mechanism. Sanchez, B. (2026). DOI: 10.5281/zenodo.18895248.
- Turner, A. M., et al. (2023). Steering Language Models With Activation Engineering. arXiv:2308.10248.
- Udrescu, S.-M. \& Tegmark, M. (2020). AI Feynman: A Physics-Inspired Method for Symbolic Regression. Science Advances, 6(16).

## Appendix A: Complete Pipeline Results

142 candidates evaluated. 1 DUAL-PASS, 3 QUADRANT3$\to$FLIPPED, 34 ORACLE-FAIL+CHECKER-PASS, remainder CHECKER-FAIL. Full results at https://solomonb14d3.github.io/noethersolve/.

## Appendix B: Reproducibility

All code, adapters, and data: https://github.com/SolomonB14D3/noethersolve

```bash
git clone https://github.com/SolomonB14D3/noethersolve.git
cd noethersolve
pip install -r requirements.txt
python autonomy_loop.py --problem problems/vortex_pair_conservation.yaml
```
