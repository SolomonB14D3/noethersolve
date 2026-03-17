# NoetherSolve

**https://github.com/SolomonB14D3/noethersolve** · **https://solomonb14d3.github.io/noethersolve**

[![Paper: Breaking Frozen Priors](https://zenodo.org/badge/DOI/10.5281/zenodo.19017290.svg)](https://doi.org/10.5281/zenodo.19017290) [![Paper: NoetherSolve Toolkit](https://zenodo.org/badge/DOI/10.5281/zenodo.19029880.svg)](https://doi.org/10.5281/zenodo.19029880) [![Paper: Unified Theory of Oracle Difficulty](https://img.shields.io/badge/preprint-paper%2012-blue)](paper/unified_oracle_difficulty_theory.md) [![D1: Q_f Conservation](https://zenodo.org/badge/DOI/10.5281/zenodo.19055338.svg)](https://doi.org/10.5281/zenodo.19055338) [![D2: Z₃ Phase Cancellation](https://zenodo.org/badge/DOI/10.5281/zenodo.19055580.svg)](https://doi.org/10.5281/zenodo.19055580) [![D3: LLM Knowledge Gaps](https://zenodo.org/badge/DOI/10.5281/zenodo.19055582.svg)](https://doi.org/10.5281/zenodo.19055582) [![D4: Orthogonal Adapters](https://zenodo.org/badge/DOI/10.5281/zenodo.19055588.svg)](https://doi.org/10.5281/zenodo.19055588) [![D5: Certainty Contamination](https://zenodo.org/badge/DOI/10.5281/zenodo.19068373.svg)](https://doi.org/10.5281/zenodo.19068373) [![D6: Resolvent Unification](https://zenodo.org/badge/DOI/10.5281/zenodo.19071198.svg)](https://doi.org/10.5281/zenodo.19071198)

**Automated scientific discovery: find where models are wrong, build tools that give the right answer, and serve them to any AI agent.**

The pipeline: **find gaps → flip facts → build tool → add to MCP server.** Every tool we build makes every connected agent smarter.

NoetherSolve starts by finding where LLMs are confidently wrong. It generates candidates, verifies them numerically, and measures whether the model already knows them. When it doesn't — that's where new science lives. The system discovers the answer, builds a verified computational tool for it, and exposes that tool via [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) so any AI agent can call it at inference time.

This is better than embedding knowledge in weights. Adapters trained on domain facts improve general truth preference (+0.10 MC2 on TruthfulQA, statistically significant), and orthogonal adapters (routed per-cluster) achieve 100% across 77 domains — but they can't be naively stacked without interference. Tools scale without constraints: each new tool is independent, verified, and callable on demand. The agent doesn't need to memorize that the Riemann Hypothesis is open — it calls `check_conjecture("Riemann")` and gets the verified answer.

**160+ tools** currently exposed via MCP. 150+ are **calculators** — verified computational engines that derive answers from first principles (enzyme kinetics, quantum mechanics, pharmacokinetics, organic chemistry reaction prediction, PID controller simulation, transaction isolation analysis, quantum circuit simulation, stability analysis, conservation law monitoring, genetic design, chemical auditing, elliptic curves, epidemiology, turbulence, information theory, and more). The rest are **lookup tables** — reference databases for mathematical conjectures, complexity theory, proof barriers, benchmark scores, and LLM science claims. Calculators scale indefinitely; lookups are faster but finite. Together they cover physics, math, genetics, enzyme kinetics, quantum mechanics, pharmacokinetics, organic chemistry, control systems, databases, quantum computing, chemistry, cryptography, economics/finance, distributed systems, networking, operating systems, epidemiology, information theory, elliptic curves, drug interactions, turbulence, and LLM science.

The method is domain-agnostic. We've applied it to fluid dynamics, electromagnetism, chemical kinetics, Hamiltonian mechanics, Navier-Stokes regularity, knot theory, genetics therapeutics (7 domains covering CRISPR design through clinical translation), unsolved mathematics (6 domains covering Millennium Problems through computational complexity), LLM science (6 domains), programming languages (6 domains), 9 STEM domains (chemistry, cryptography, economics/finance, distributed systems, networking, operating systems, database internals, quantum computing, control systems), 3 science domains (biochemistry, organic chemistry, quantum mechanics), 9 frontier domains (battery technology, origin of life, consciousness, antibiotic resistance, protein folding, aging biology, quantum gravity, dark matter/energy, black hole frontiers, particle physics, holographic QInfo, condensed matter, climate science, cosmology, multi-messenger astronomy, neutrino physics), and 4 newer domains (elliptic curves, intersection theory, drug interactions, information theory). Any field where you can verify a claim and build a checker is fair game.

### Paper

**Breaking Frozen Priors: Teaching Language Models to Discover Conservation Laws from Numerical Simulation** (Sanchez, 2026)
DOI: [10.5281/zenodo.19017290](https://doi.org/10.5281/zenodo.19017290)

Three-phase pipeline transforms a frozen oracle (margin -77.5 +/- 1.7) into a ranking engine (Spearman rho = 0.932 from baseline -0.143). Novel Q_f invariant family verified across chaotic vortex systems and extended to continuous 2D/3D Euler equations. The LLM gap pointed directly at the physics: the model's blind spot on weighted distance sums led to the discovery of stretch-resistant invariants relevant to 3D Navier-Stokes regularity. See [`paper/breaking_frozen_priors.pdf`](paper/breaking_frozen_priors.pdf).

**NoetherSolve Toolkit: Conservation Law Monitoring, Discovery, and Scientific Auditing Across Physics, Genetics, and Mathematics** (Sanchez, 2026)
DOI: [10.5281/zenodo.19029880](https://doi.org/10.5281/zenodo.19029880)

160+ tools organized across multiple tiers: 6 physics tools (conservation monitors, integrator validator, chemical auditor, EM monitor, Hamiltonian validator, invariant learner), 5 genetics tools (sequence auditor, CRISPR scorer, pipeline validator, aggregation predictor, splice scorer), 5 pharmacokinetics tools (IV bolus, oral dosing, half-life, steady state, dose adjustment), 5 enzyme kinetics tools (Michaelis-Menten, inhibition, catalytic efficiency, cooperativity, pH rate profile), 6 quantum mechanics tools (particle-in-box, hydrogen energy, uncertainty, tunneling, harmonic oscillator, angular momentum), 6 organic chemistry tools (molecule analysis, selectivity, mechanism prediction, synthesis validation, Baldwin's rules, Woodward-Hoffmann), 7 unsolved mathematics tools (complexity auditor, conjecture checker, proof barrier checker, number theory verifier, reduction validator, PDE regularity checker, knot monitor), 1 LLM science tool (claims auditor with benchmark checker and scaling calculator), 3 systems tools (PID controller, transaction isolation, quantum circuit simulator), 6 STEM calculators (chemistry, cryptography, finance, distributed systems, networking, operating systems), plus dozens of additional tools for elliptic curves, epidemiology, information theory, drug interactions, turbulence, plasma physics, optics, seismology, climate science, and more. Q_f monitors detect corruption at 100x lower noise than standard H/Lz monitors. 173 validation test cases across all tools, 100% catch rate. 2265 tests with physics-enforcing pre-commit hook. See [`paper/noethersolve_toolkit.pdf`](paper/noethersolve_toolkit.pdf).

**Unified Theory of Oracle Difficulty: Three Mechanisms Explain 95% of Benchmark Variance** (Sanchez, 2026)
[View preprint](paper/unified_oracle_difficulty_theory.md) · Commit: [e4b6da9](https://github.com/SolomonB14D3/NoetherSolve/commit/e4b6da9)

Discovers that oracle baseline accuracy across 77 domains is determined by three independent mechanisms: (1) **Length ratio** (r = −0.742): truth length / shortest distractor length predicts baseline perfectly — domains with ratio < 1.2 average 64% baseline; ratio > 2.5 averages 7%. (2) **Distractor semantic coherence** (5.5 LP gap): coherent distractors score 33% pass; incoherent distractors score 75% pass despite identical lengths. (3) **Scoring method sensitivity**: sum normalization favors hedged truths; mean normalization favors verbose truths. Unified theory shows how all three mechanisms interact combinatorially. Applying all three fixes simultaneously improves baseline from 0% to 75–100% on hardest domains without requiring any model retraining. Resolves the "LLM self-knowledge gap" (0% on 6 LLM domains) as a measurement artifact, not an actual knowledge gap. Provides practical decision tree for oracle fact construction and benchmark methodology. Peer-ready for TMLR, JMLR, or NeurIPS evaluation workshops.

### Discovery Papers (Novel Scientific Findings)

Papers arising from novel findings discovered by the NoetherSolve pipeline, targeting domain scientists (physicists, mathematicians):

**D1: Approximate Conservation Laws in Point Vortex Dynamics** DOI: [10.5281/zenodo.19055338](https://doi.org/10.5281/zenodo.19055338) -- Q_f = Sigma Gamma_i Gamma_j f(r_ij) approximately conserved for ANY smooth f. Green's function principle: optimal f = G_d(r). Optimal combination 300x better.

**D2: Z3 Phase Cancellation in Choreographic Orbits** DOI: [10.5281/zenodo.19055580](https://doi.org/10.5281/zenodo.19055580) -- Figure-8 Z3 symmetry enables Fourier phase cancellation. Critical range: -0.67 < p < 2.55.

**D3: Where LLMs Are Confidently Wrong: 1038 Facts Across 67 Domains** DOI: [10.5281/zenodo.19055582](https://doi.org/10.5281/zenodo.19055582) -- Systematic mapping of model knowledge gaps. Intersection theory deepest gap ever measured (margin -27.6).

**D4: Orthogonal Adapter Routing for Interference-Free Knowledge Injection** DOI: [10.5281/zenodo.19055588](https://doi.org/10.5281/zenodo.19055588) -- Representational see-saws require orthogonal adapters with routing. 1038/1038 facts flipped.

**D5: Certainty Contamination: How Definitive Language Biases LLM Factual Judgments** DOI: [10.5281/zenodo.19068373](https://doi.org/10.5281/zenodo.19068373) -- LLMs prefer definitive claims over hedged scientific language (r = -0.402, p < 0.01). Pass rate: 55% (balanced) to 25% (high asymmetry).

**D6: Resolvent-Conservation Unification: Spectral Theory of Approximate Invariants** DOI: [10.5281/zenodo.19071198](https://doi.org/10.5281/zenodo.19071198) -- Green's function optimality arises from zero-frequency limit of the resolvent. Unifies potential theory, spectral theory, and Noether's theorem.

---

<details open>
<summary><h2>How It Works (Plain English)</h2></summary>

An AI model is trained on everything humans have written. That means it knows
what we know, but it also shares our blind spots. Where the collective
literature is thin or wrong, the model is thin or wrong.

NoetherSolve exploits this in four steps:

1. **Find gaps.** Propose claims about how systems behave. Verify them
   numerically. Ask the model: did you already know this? If the model is
   confidently wrong, that's a gap — and gaps in model knowledge point to
   gaps in human knowledge, because the model was trained on human knowledge.
2. **Flip facts.** Train lightweight adapters that flip the model's answer
   from wrong to right, without degrading anything it already knows.
   Orthogonal adapters (one per concept cluster, routed at inference) achieve
   100% across all 77 domains (1043+ facts) with 0% MMLU degradation.
   Cross-domain joint training blends related domains into a single adapter
   (H 14/16, NS 10/16, Knot 11/16, Chem 13/16 from ONE adapter), and hybrid
   routing (pick best of joint vs orthogonal per fact) reaches 82.1% on
   physics frontier. The constraint: adapters can't be naively stacked —
   they must be routed or blended from scratch, never merged after training.
3. **Build tools.** Each discovery becomes a standalone computational tool —
   a verified calculator that derives answers from first principles. Tools
   scale without routing constraints and work for any model.
4. **Add to MCP server.** Expose every tool via Model Context Protocol so
   any AI agent (Claude, GPT, local models) can call them at inference time.
   The agent doesn't need to memorize facts — it calls the tool and gets
   the verified answer.

The result: every gap we find makes every connected agent smarter. The 160+
tools currently served cover physics, genetics, enzyme kinetics, quantum
mechanics, pharmacokinetics, organic chemistry, mathematics, complexity
theory, control systems, databases, quantum computing, chemistry,
cryptography, economics/finance, distributed systems, networking,
operating systems, epidemiology, information theory, elliptic curves,
drug interactions, turbulence, and LLM science.

</details>

---

<details open>
<summary><h2>Oracle Difficulty: Three Independent Mechanisms (Paper 12)</h2></summary>

When building oracle facts for LLM evaluation, success depends on three mechanisms that interact combinatorially:

### Mechanism 1: Length Ratio (r = −0.742 correlation with baseline)

The ratio of correct-answer length to shortest-distractor length predicts baseline accuracy across domains:

| Length Ratio | Expected Baseline | Example Domains |
|--------------|-------------------|-----------------|
| < 1.2 | **64%** | Operating Systems (83%), Cryptography (75%) |
| 1.2–2.5 | **13%** | LLM Hallucination (0%), Consciousness (33%) |
| > 2.5 | **7%** | Knot Invariants (6%), NS Regularity (0%) |

**Why:** Log-probability scoring sums tokens, penalizing longer answers implicitly. Shorter answers have higher per-token log-prob and win the ranking even if semantically wrong.

**Fix:** Balance lengths to ratio 0.8–1.2. Remove parentheticals from truths; add plausible-but-wrong details to distractors. Length-balancing knot_invariants (ratio 7.81 → 1.16) improved baseline from 0% to 25% without any model intervention.

### Mechanism 2: Distractor Semantic Coherence (33% → 75% swing)

Distractors that are grammatically sensible completions score high on log-prob and beat truths—even with balanced lengths.

| Distractor Type | Pass Rate (balanced lengths) |
|-----------------|------|
| Coherent (plausible wrong answers) | **33%** |
| Incoherent (nonsense) | **75%** |

**For adapter training:** Use incoherent distractors to isolate truth signal from fluency bias.
**For benchmarks:** Keep coherent distractors (that's the point of measuring knowledge).

### Mechanism 3: Scoring Method Selection (0% ↔ 100% swing)

Sum vs mean normalization reveal different biases. Choose based on truth phrasing:

| Domain Characteristic | Best Scoring | Why |
|----------------------|--------------|-----|
| Hedged/technical truths | **Sum** | Hedged truths are shorter; mean rewards distractors |
| Verbose/explanatory truths | **Mean** | Mean normalizes the length penalty |

**Example:** `climate_science_frontiers` with hedged truths: Sum 75% → Mean 0% (catastrophic drop). Use sum scoring for technical domains.

### Unified Fact Construction Checklist

Before oracle evaluation:

1. **Length ratio < 1.5?** Run `python -m noethersolve.audit_facts --check-lengths`. Fix if too high.
2. **Truths confident?** Remove hedging ("may" → "do", "might suggest" → "show").
3. **Distractors appropriate?** Coherent for benchmarks, incoherent for adapter training.
4. **Scoring chosen?** Sum for hedged domains, mean for verbose domains.

See [`paper/unified_oracle_difficulty_theory.md`](paper/unified_oracle_difficulty_theory.md) for full analysis with 77-domain verification.

### Additional Mechanisms Discovered

**Mechanism 4: Anti-Fluency Distractors** -- Making distractors verbose/awkward rescues hidden model knowledge (86-100% flip rate). WARNING: Creates false positives for ALL claim types. Only valid for fluency bias testing.

**Mechanism 5: Round Number Bias** -- Models prefer round numbers (0.5, 10%) over precise values (0.326, 2%). Gap up to -15.9 for logarithmic vs simple-power forms.

**Mechanism 6: Certainty Contamination** -- Models prefer definitive claims over hedged scientific language (r = -0.402, p < 0.01). Not length bias (definitive distractors are actually longer). DOI: [10.5281/zenodo.19068373](https://doi.org/10.5281/zenodo.19068373)

**Mechanism 7: Technical Simplification Bias** -- Models prefer simple/familiar terms over precise technical language (t = -3.73, p = 0.0004). "kinetic energy" beats "enstrophy" by -9.62 margin.

**Mechanism 8: Term Preference Bias** -- Models have fixed preferences for specific physics terms regardless of correctness. Tested via mirror pairs where terms swap roles.

**Mechanism 9: Mathematical Status Blindness** -- Models can state what a conjecture claims (71.4% pass) but fail on its research status (4.2% pass, t = -4.21, p = 0.0002). Model NEVER downgrades "proven" to "open" but DOES upgrade "open" to "proven."

</details>

---

<details open>
<summary><h2>MCP Server — Give Any AI Agent 160+ Verified Tools</h2></summary>

The MCP server exposes all NoetherSolve tools to any AI agent that supports
[Model Context Protocol](https://modelcontextprotocol.io/). One line of config,
160+ tools available: 150+ calculators + lookup tables.

### Setup for Claude Code

The project includes `.mcp.json` — Claude Code auto-discovers it when you
open the project. No manual config needed.

Or install globally and use the entry point:

```bash
pip install noethersolve[mcp]
noethersolve-mcp  # starts the server
```

### Available Tools (160+)

| Category | Tools | Examples |
|----------|-------|---------|
| **Conservation monitors** | 4 | `check_vortex_conservation`, `check_hamiltonian_system`, `check_em_conservation`, `discover_conservation_law` |
| **Mathematics** | 10 | `check_conjecture`, `check_complexity_inclusion`, `check_proof_barriers`, `verify_goldbach`, `check_sobolev_embedding` |
| **Genetics/therapeutics** | 5 | `score_crispr_guide`, `audit_dna_sequence`, `predict_protein_aggregation`, `validate_therapy_pipeline`, `score_splice_sites` |
| **Enzyme kinetics** | 5 | `calc_michaelis_menten`, `calc_enzyme_inhibition`, `calc_catalytic_efficiency`, `calc_cooperativity`, `calc_ph_rate_profile` |
| **Quantum mechanics** | 6 | `calc_particle_in_box`, `calc_hydrogen_energy`, `calc_uncertainty_check`, `calc_tunneling`, `calc_harmonic_oscillator_qm`, `calc_angular_momentum` |
| **Pharmacokinetics** | 5 | `calc_iv_bolus`, `calc_oral_dose`, `calc_half_life`, `calc_steady_state`, `calc_dose_adjustment` |
| **Organic chemistry** | 6 | `analyze_molecule`, `predict_reaction_selectivity`, `predict_reaction_mechanism`, `validate_synthesis_pathway`, `check_baldwin_rules`, `check_woodward_hoffmann` |
| **LLM science** | 4 | `check_llm_claim`, `chinchilla_scaling`, `check_benchmark_score`, `audit_llm_claims` |
| **Chemical kinetics** | 1 | `audit_chemical_network` |
| **Knot theory** | 1 | `check_knot_invariants` |
| **Number theory** | 4 | `verify_goldbach`, `verify_collatz`, `check_abc_triple`, `analyze_prime_gaps` |
| **Chemistry** | 3 | `calc_nernst`, `calc_buffer_ph`, `calc_crystal_field` |
| **Cryptography** | 3 | `calc_security_level`, `calc_birthday_bound`, `calc_cipher_mode` |
| **Economics/Finance** | 3 | `calc_black_scholes`, `calc_put_call_parity`, `calc_nash_equilibrium` |
| **Distributed systems** | 3 | `calc_quorum`, `calc_byzantine`, `calc_vector_clock` |
| **Networking** | 3 | `calc_bandwidth_delay`, `calc_subnet`, `calc_tcp_throughput` |
| **Operating systems** | 3 | `calc_page_table`, `calc_scheduling`, `calc_deadlock` |

Every tool returns verified results from curated reference databases — not
model guesses. When an agent calls `check_conjecture("Riemann")`, it gets the
actual status (OPEN), the key facts, common errors, and references. No
hallucination possible.

### Why MCP instead of fine-tuning?

We tried both. Adapters trained on 1043+ domain facts improve truth preference
(+0.10 MC2 on TruthfulQA), and orthogonal adapters (routed per-cluster at
inference) achieve 100% across all 77 domains with 0% MMLU degradation.
Cross-domain joint training works for related domains, and hybrid routing
(pick best of joint vs orthogonal per fact) reaches 82.1% on physics frontier.
But adapters can't be naively stacked: combining 37+ adapters by weight
averaging destroys general knowledge (-43% MMLU), and a unified adapter on
244+ heterogeneous facts collapses (7.8% vs 10.2% baseline). The key insight
is **route, never stack** — each adapter must be routed to its domain at
inference, never merged. Tools don't have these constraints:

- **No routing needed.** Each tool is independent. Adding tool #43 doesn't
  degrade tools #1-42 and requires no inference-time routing logic.
- **No capacity limits.** A tool can encode arbitrarily complex logic.
- **Verified correctness.** 2265 tests enforce correctness. An adapter can
  only shift probabilities; a tool returns the exact right answer.
- **Model-agnostic.** Any agent that speaks MCP can use these tools.
  Adapters are tied to one model's vocabulary.

**Two complementary paths.** Adapter blending (joint training from scratch on
mixed data) is the path to fixing small models directly — a single
difficulty-weighted adapter lifts 4 domains simultaneously (H 14/16, NS 10/16,
Knot 11/16, Chem 13/16), and orthogonal routing gets 16/16 per domain with 0%
MMLU degradation. MCP tools are the path to making any model a powerhouse
through tool use — each tool is independent, verified, and callable on demand,
no routing required. Adapters change what the model knows; tools change what
the model can do.

</details>

---

<details>
<summary><h2>What It Does (Technical)</h2></summary>

NoetherSolve runs a **dual-filter pipeline**. The "oracle" is a base LLM scored by log-probability: for each candidate fact, we compare `log P(true answer | context)` against `log P(best distractor | context)`. Positive margin means the model knows it; negative means it doesn't.

```
Hypothesis (expression)
       │
       ▼
 Numerical checker          ← Is this quantity actually conserved?
 (RK45 integration,           frac_var = σ/|mean| < threshold
  frac_var test)
       │ PASS
       ▼
 Oracle filter              ← Does the model already know it?
 (log-prob margin,            margin = log P(truth) − log P(best distractor)
  base LLM + adapter stack)
       │
       ├─ PASS  → DUAL-PASS: known quantity, archive it
       │
       └─ FAIL  → NEW SCIENCE: model has never seen this
                    │
                    ▼
              Train adapter  ← Teach the discovery to the model
              (hinge loss,     25 examples generated per candidate
               logit-space)
                    │
                    ├─ margin flips → KNOWLEDGE INJECTED: adapter joins the stack
                    │                  (all future candidates evaluated with this knowledge)
                    │
                    └─ margin stays → HARD GAP: log it, try different approach next run
```

Adapters stack within a run — each successful discovery makes the oracle
smarter for every subsequent candidate. After the main sweep, a
**confidence-driven resampling** pass retries borderline failures (margin
between -5 and 0) with the full adapter stack. Candidates that were just
short of flipping often get rescued once the model has absorbed neighboring
discoveries. Survivors get promoted to high-priority in the open questions
queue for the next run.

<details>
<summary><strong>Escalation for hard domains</strong></summary>

1. **Single-pass** — one adapter for the whole domain. Works for clean domains
   (chemical kinetics: 0/16 to 16/16 with distractor fix).
2. **Staged training** — group facts into clusters, train sequentially, verify
   zero regression at each stage. Solved Hamiltonian mechanics (1/16 to 16/16
   in 5 stages).
3. **Orthogonal adapters** — when staged training plateaus because facts
   interfere within a single adapter, train separate specialist adapters per
   concept cluster. Each adapter learns one cluster without fighting the others.
   Route facts to their specialist at inference. Solved NS regularity
   (6/16 staged to 16/16 with orthogonal cluster adapters).
4. **Cross-domain joint training** — train a single adapter on multiple domains
   simultaneously. Difficulty-weighted sampling achieves the best transfer:

   | Method | Hamiltonian | NS | Knot | Chemical |
   |--------|-------------|-----|------|----------|
   | No adapter | 1/16 | 0/16 | 1/16 | 0/16 |
   | Basic joint | 16/16 | 6/16 | 10/16 | 11/16 |
   | Domain-balanced | 16/16 | 6/16 | 11/16 | 11/16 |
   | Difficulty-weighted | 14/16 | **10/16** | 11/16 | 13/16 |
   | Anchored joint | 16/16 | 9/16 | 11/16 | 12/16 |

   A single jointly-trained adapter lifts all 4 domains simultaneously.
   Difficulty-weighted sampling (oversample hard facts) gives the best result
   on the hardest domain (NS: 0 to 10/16). Conservation knowledge transfers
   across physics and pure math.

</details>

<details>
<summary><strong>Token-length bias</strong></summary>

Some facts are unlearnable because the base model
prefers shorter token sequences. If a distractor is shorter than the correct
answer (e.g., `"k × [A]"` vs `"k × [A] × [B] where k is the rate constant"`),
no amount of adapter training will flip the margin. Fix by rephrasing: shorten
the truth and lengthen the distractors so they're clearly wrong and roughly
the same length. This flipped the last chemical kinetics holdout from -3.8 to
+4.3 and rescued ns03 from -44 to +242.8.

</details>

<details>
<summary><strong>Never stack adapters — but blending works</strong></summary>

**Stacking fails.** Training a specialist on gap facts and layering it on top
of a joint adapter at inference destroyed the joint adapter's wins (8/16 →
5/16). The specialist overwrites what the joint adapter learned. Never combine
adapter weights by averaging or stacking at inference.

**Blending works — for related domains.** Cross-domain joint training
(difficulty-weighted sampling) produces a single adapter that lifts multiple
*related* domains simultaneously: Hamiltonian 14/16, NS 10/16, Knot 11/16,
Chemical 13/16 — all from ONE adapter. But blending across *heterogeneous*
domains fails: a unified adapter trained on 244 toolkit facts (16 diverse
clusters from complexity theory to pharmacokinetics) scored 7.8% — worse than
the 10.2% baseline.

The distinction: training one adapter from scratch on mixed data = blending
(works for related domains). Combining separately trained adapters at inference
= stacking (fails). Blending across unrelated domains = also fails.

**Hybrid routing: best of both worlds.** For domains where both a joint adapter
and orthogonal specialist adapters exist, evaluate both and pick whichever has
the higher margin per fact. On the physics frontier (7 domains, 84 facts):

| Strategy | Accuracy |
|----------|----------|
| Baseline (no adapter) | 18/84 (21.4%) |
| Joint adapter only | 37/84 (44.0%) |
| Orthogonal only | 59/84 (70.2%) |
| **Hybrid routing** | **69/84 (82.1%)** |

Joint adapters win on some domains (particle physics, neutrino, holographic
QInfo) while orthogonal adapters win on others (dark matter, quantum gravity,
cosmology, condensed matter). Hybrid routing captures both strengths.

**Two paths forward.** Adapter blending is the path to improving small models
directly — embed corrected knowledge into the weights so even a 4B model gets
the answer right without tool calls. MCP tools are the path to making any model
a powerhouse — the model doesn't need to know the answer, it just needs to know
which tool to call. Both paths are productive; the choice depends on whether
you're optimizing the model or the system.

</details>

</details>

---

<details>
<summary><h2>Toolkit — Practical Tools Built from Discoveries</h2></summary>

The pipeline's discoveries become standalone tools that work without any LLM.
Install: `pip install noethersolve` (or `pip install -e .` for development).

<details>
<summary><h3>Conservation Monitors</h3></summary>

Drop into any simulation loop. Track standard invariants (H, Lz, momentum)
plus AI-discovered quantities (Q_f family, R_f ratio, Wegscheider cyclicity).

```python
from noethersolve import VortexMonitor

monitor = VortexMonitor(circulations=[1.0, -0.5, 0.3])
monitor.set_initial(positions)

for step in simulation:
    state = integrator.step()
    report = monitor.check(state)
    if report.worst_drift > 1e-3:
        print(f"WARNING: {report.worst_name} drifted {report.worst_drift:.2e}")
```

Three built-in monitors: `VortexMonitor` (2D point-vortex), `ChemicalMonitor`
(reaction networks with Wegscheider cyclicity, entropy production, Lyapunov
function), `GravityMonitor` (N-body with Q_f on pairwise distances).

</details>

<details>
<summary><h3>Integrator Validator</h3></summary>

Validates your ODE solver configuration before you run a long simulation.
Checks whether conservation laws are preserved and suggests fixes.

```python
from noethersolve import validate_integrator

report = validate_integrator(
    rhs=my_vortex_rhs,
    y0=positions.ravel(),
    t_span=(0, 100),
    system="vortex",
    circulations=[1.0, -0.5, 0.3],
    rhs_args=(circulations,),
    rtol=1e-8,
)
print(report)
# ============================================================
#   Integrator Validation: PASS
# ============================================================
#   PASSED (12):
#     H                          frac_var=9.30e-09
#     Lz                         frac_var=4.80e-09
#     Q_linear                   frac_var=2.53e-03
#     ...
```

Also supports `compare_configs()` to test multiple solver settings side-by-side,
and custom invariants via `invariants={"energy": lambda y: compute_energy(y)}`.

</details>

<details>
<summary><h3>Chemical Network Auditor</h3></summary>

Checks thermodynamic consistency of a reaction network without running a
simulation. Pure algebraic checks on the stoichiometry and rate constants.

```python
from noethersolve import audit_network

report = audit_network(
    species=["A", "B", "C"],
    stoichiometry=[[-1, 1, 0, 0], [1, -1, -1, 1], [0, 0, 1, -1]],
    rate_constants=[0.5, 0.3, 0.4, 0.2],
    reactant_matrix=[[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
    reverse_pairs=[(0, 1), (2, 3)],
)
print(report)
# Shows: conservation laws, Wegscheider cycle products, detailed balance
# ratios, entropy production, and warnings if anything is inconsistent.
```

Catches: Wegscheider cyclicity violations, missing conservation laws,
non-physical rate constants, negative entropy production (second law violation).

</details>

<details>
<summary><h3>EM Field Monitor</h3></summary>

Monitors electromagnetic field simulations for conservation of standard
and obscure invariants: energy, momentum, optical chirality (Zilch Z⁰,
Lipkin 1964), helicity, super-energy (Chevreton tensor), zilch vector.

```python
from noethersolve import EMMonitor

monitor = EMMonitor(N=64, L=2*np.pi)
monitor.set_initial(E_fields, B_fields)  # 3-tuples of 3D arrays

for step in simulation:
    E, B = maxwell_solver.step()
    report = monitor.check(E, B)
    if report.worst_drift > 1e-6:
        print(f"WARNING: {report.worst_name} drifted {report.worst_drift:.2e}")
```

Catches: numerical dissipation, wrong boundary conditions, missing terms
in Maxwell solvers. Spectral curls computed internally via FFT.

</details>

<details>
<summary><h3>Hamiltonian System Validator</h3></summary>

Validates that an ODE integrator preserves the symplectic structure of
Hamiltonian systems. Goes beyond energy to check Liouville's theorem
(phase-space volume) and the first Poincaré integral invariant (∮ p dq).

```python
from noethersolve import kepler_2d

monitor = kepler_2d(mu=1.0)  # built-in Kepler problem
report = monitor.validate(
    z0=np.array([1.0, 0.0, 0.0, 0.8]),  # elliptical orbit
    T=100.0, rtol=1e-10,
)
print(report)
# Shows: energy, angular_momentum, LRL_magnitude,
#        liouville_volume, poincare_invariant — all PASS/WARN/FAIL
```

Built-in systems: `harmonic_oscillator`, `kepler_2d` (with angular momentum
and Laplace–Runge–Lenz vector), `henon_heiles`, `coupled_oscillators`.
Or bring your own H(z) and ∇H(z) via `HamiltonianMonitor(H=..., dH=..., n_dof=...)`.

</details>

<details>
<summary><h3>Invariant Learner</h3></summary>

Automatically discovers new conserved quantities from trajectory data.
Optimizes over 12 basis functions to find f(r) that minimizes fractional
variation of Q_f = Σᵢ<ⱼ wᵢwⱼ f(rᵢⱼ) along one or more trajectories.

```python
from noethersolve import InvariantLearner

learner = InvariantLearner()
result = learner.learn_from_positions(
    position_trajectories=[trajectory],  # shape (n_steps, N, dim)
    weights=[1.0, -0.5, 0.3],           # vortex circulations
)
print(result)
# Shows: optimal f(r) = 0.924·e^(-r) + 0.186·sin(r) + ...
#        40% improvement over single-basis e^(-r)
#        Individual basis losses ranked
```

Three input modes: `learn_from_positions` (raw coordinates),
`learn_from_distances` (pairwise distance time series),
`learn_from_field` (continuous 2D vorticity fields via FFT convolution).

</details>

<details>
<summary><h3>Fact Quality Auditor</h3></summary>

Checks oracle fact files (`*_facts.json`) for token-length bias and
distractor quality issues before you waste training cycles. Token-length
bias was the #1 blocker across 4 domains (14+ facts needed rephrasing).

```python
from noethersolve import audit_facts

report = audit_facts("problems/3body_conservation_facts.json")
print(report)
# ============================================================
#   Fact Audit: WARN (2 issues)
# ============================================================
#   3b03_angular           LENGTH_BIAS  ratio=0.63  HIGH
#   3b05_visviva           LENGTH_BIAS  ratio=0.68  HIGH
#   3b01_energy            LENGTH_BIAS  ratio=0.82  MODERATE
#   ...
```

Catches: truth longer than shortest distractor (ratio < 0.7 = HIGH,
< 0.9 = MODERATE), distractors that are substrings of the truth,
identical distractors. Run this on every new fact file before training.

</details>

<details>
<summary><h3>Knot Invariant Monitor</h3></summary>

Verifies knot invariants under Reidemeister moves. Checks which quantities
are preserved (Jones polynomial) vs which change (writhe, bracket polynomial)
when you add/remove twists and crossings.

```python
from noethersolve import KnotMonitor, trefoil

monitor = KnotMonitor(trefoil())
report = monitor.validate()
print(report)
# ============================================================
#   Knot Invariant Report: trefoil — PASS
# ============================================================
#   R1 (add twist):
#     writhe              EXPECTED_CHANGE  3 → 4
#     bracket_polynomial  EXPECTED_CHANGE  (changed by -A^{-3})
#     jones_polynomial    PRESERVED        ✓
#   R2, R3: all quantities preserved  ✓
```

Built-in knots: `unknot()`, `trefoil()`, `figure_eight_knot()`.
Reidemeister moves: `apply_r1(knot, sign)`, `apply_r1_remove(knot)`.

</details>

<details>
<summary><h3>Genetics Therapeutics Tools</h3></summary>

Five tools for genetics therapeutics design — sequence auditing, CRISPR guide scoring, pipeline consistency validation, protein aggregation prediction, and splice site scoring.

**Sequence Design Auditor** — checks DNA/RNA for therapeutic design pitfalls:

```python
from noethersolve import audit_sequence

report = audit_sequence("ATGCGATCGAATAAACGATTTTTCG")
print(report)
# CpG density, GC content, homopolymers, cryptic splice sites,
# poly-A signals, self-complementarity — with severity levels
```

**CRISPR Guide RNA Scorer** — scores guides for on-target activity and off-target risk:

```python
from noethersolve import score_guide, check_offtarget_pair

report = score_guide("GAGTCTAGCAGTCTAGCACG")
print(f"Activity: {report.activity_score}/100, Off-target: {report.offtarget_risk}")

# Compare guide to potential off-target site
pair = check_offtarget_pair("GAGTCTAGCAGTCTAGCACG", "GAGTCTAGCAGTCTAGCACC")
print(f"Seed mismatches: {pair['seed_mismatches']}, Risk: {pair['risk_level']}")
```

**Therapeutic Pipeline Validator** — cross-domain consistency checker:

```python
from noethersolve import validate_pipeline, TherapyDesign

design = TherapyDesign(
    modality="aav",
    target_tissue="liver",
    transgene_size_kb=4.5,
    vector_serotype="AAV8",
    promoter="TBG",
    route="iv",
    payload_type="gene_replacement",
)
report = validate_pipeline(design)
print(report)
# Checks: vector capacity, serotype-tissue, promoter-tissue,
# route-tissue, modality-payload, redosing immunity, safety monitoring
```

**Protein Aggregation Predictor** — predicts aggregation risk from amino acid sequence:

```python
from noethersolve import predict_aggregation

report = predict_aggregation("MILVFAILVILMFAILVM")
print(report)
# APR detection (AGGRESCAN), hydrophobicity (Kyte-Doolittle),
# hydrophobic patches, net charge, low-complexity regions
```

**Splice Site Scorer** — scores donor/acceptor sites against mammalian consensus PWMs:

```python
from noethersolve import score_donor, score_acceptor, scan_splice_sites

report = score_donor("CAGGTAAGT")
print(f"Score: {report.score:.2f}, Strength: {report.strength}")

# Scan full sequence for all potential splice sites
sites = scan_splice_sites("AAACAGGTAAGTCCC...", site_type="both")
```

**Pharmacokinetic Calculator** — compartmental PK modeling from first principles:

```python
from noethersolve import one_compartment_iv, one_compartment_oral, steady_state

# IV bolus kinetics
pk = one_compartment_iv(dose_mg=500, volume_L=50, half_life_h=6, time_h=12)
print(pk)  # Concentration, AUC, clearance at any time point

# Oral dosing with absorption
oral = one_compartment_oral(dose_mg=500, volume_L=50, half_life_h=6, ka=1.5, F=0.8, time_h=8)
print(oral)  # Tmax, Cmax, concentration curve

# Steady-state accumulation
ss = steady_state(dose_mg=500, volume_L=50, half_life_h=6, interval_h=8, n_doses=10)
print(ss)  # Accumulation ratio, trough, peak, time to steady state
```

</details>

<details>
<summary><h3>Enzyme Kinetics Calculator</h3></summary>

Five tools for enzyme kinetics — Michaelis-Menten, competitive/uncompetitive/noncompetitive inhibition, catalytic efficiency classification, cooperativity (Hill equation), and pH-dependent rate profiles.

```python
from noethersolve import michaelis_menten, inhibition, catalytic_efficiency

# Basic Michaelis-Menten
mm = michaelis_menten(Vmax=100, Km=10, substrate_uM=25)
print(mm)  # Rate, fraction of Vmax, substrate saturation

# Competitive inhibition
inh = inhibition(Vmax=100, Km=10, substrate_uM=25,
                 inhibitor_uM=50, Ki=20, mode="competitive")
print(inh)  # Apparent Km/Vmax, fold reduction, IC50

# Is this enzyme diffusion-limited?
eff = catalytic_efficiency(kcat=1e7, Km_uM=10)
print(eff)  # Classification: DIFFUSION_LIMITED, efficiency ratio
```

</details>

<details>
<summary><h3>Quantum Mechanics Calculator</h3></summary>

Six tools for quantum mechanics from first principles — particle in a box, hydrogen atom energy levels, Heisenberg uncertainty validation, quantum tunneling probability, harmonic oscillator energies, and angular momentum addition.

```python
from noethersolve import particle_in_box, tunneling_probability, uncertainty_check

# Particle in a box
pib = particle_in_box(n=3, L_nm=1.0, mass_kg=9.109e-31)
print(pib)  # Energy, wavelength, nodes, probability density

# Quantum tunneling through a barrier
tun = tunneling_probability(E_eV=5.0, V0_eV=10.0, barrier_width_nm=0.5,
                            mass_kg=9.109e-31)
print(tun)  # Transmission coefficient, decay constant

# Heisenberg uncertainty check
unc = uncertainty_check(delta_x_m=1e-10, delta_p_kgms=1e-24)
print(unc)  # Product vs ℏ/2, satisfied or violated
```

</details>

<details>
<summary><h3>Organic Chemistry Engine</h3></summary>

Six tools for organic chemistry — molecule analysis (functional groups, hybridization), reaction selectivity (Mayr nucleophilicity/electrophilicity), mechanism prediction, synthesis pathway validation, Baldwin's rules, and Woodward-Hoffmann rules.

```python
from noethersolve import analyze_molecule, predict_selectivity, check_baldwin

# Analyze a molecule (requires RDKit)
mol = analyze_molecule("CCO")  # ethanol
print(mol)  # Functional groups, hybridization, stereochemistry

# Check Baldwin's rules for ring closure
baldwin = check_baldwin(ring_size=5, closing_type="tet", position="exo")
print(baldwin)  # Favored/disfavored, explanation

# Woodward-Hoffmann rules
from noethersolve import check_woodward_hoffmann
wh = check_woodward_hoffmann(reaction_type="electrocyclic", n_electrons=4,
                              conditions="thermal")
print(wh)  # Conrotatory/disrotatory, symmetry analysis
```

</details>

<details>
<summary><h3>Unsolved Mathematics Tools</h3></summary>

Six tools for validating claims about computational complexity, open conjectures, proof techniques, number theory, reductions, and PDE regularity.

**Complexity Class Auditor** — validates claims about class relationships:

```python
from noethersolve import audit_complexity

report = audit_complexity(["P = NP", "SAT is NP-complete", "GI is NP-complete"])
print(report)
# Checks: inclusions, separations, completeness, collapse implications
# → FAIL: P=NP would collapse PH; GI is NOT known to be NP-complete
```

**Conjecture Status Checker** — validates claims about open problem status:

```python
from noethersolve import check_conjecture, check_claim

report = check_conjecture("riemann_hypothesis", claimed_status="SOLVED")
print(report)  # → FAIL: Riemann Hypothesis is OPEN, not SOLVED

report = check_claim("Goldbach conjecture was proved")
print(report)  # → FAIL: strong Goldbach is OPEN (weak Goldbach proved by Helfgott 2013)
```

**Proof Barrier Checker** — checks if known barriers block a proof technique:

```python
from noethersolve import check_barriers, what_works_for

report = check_barriers("diagonalization", "P vs NP")
print(report)  # → FAIL: relativization barrier blocks diagonalization for P vs NP

alts = what_works_for("P vs NP")
print(alts)  # Techniques NOT blocked: algebraic geometry (GCT), interactive proofs, ...
```

**Number Theory Verifier** — numerical verification of famous conjectures:

```python
from noethersolve import verify_goldbach, verify_collatz, check_abc_triple

print(verify_goldbach(100))   # 6 decompositions: 3+97, 11+89, 17+83, ...
print(verify_collatz(27))     # 111 steps, max value 9232
print(check_abc_triple(1, 8, 9))  # quality 1.226 — exceptional ABC triple!
```

**Reduction Chain Validator** — validates computational reduction chains:

```python
from noethersolve import validate_chain

chain = [("3-SAT", "many-one", "CLIQUE"), ("CLIQUE", "many-one", "VERTEX-COVER")]
report = validate_chain(chain)
print(report)  # → PASS: valid transitive chain, effective type: many-one
```

**PDE Regularity Checker** — validates Sobolev embeddings and regularity claims:

```python
from noethersolve import check_sobolev_embedding, check_pde_regularity

print(check_sobolev_embedding(1, 2, 3))  # W^{1,2}(R^3) → L^6 (subcritical)
print(check_pde_regularity("navier-stokes", 3, "global_smooth"))  # → WARN: open problem
```

**LLM Claims Auditor** — validates claims about LLM capabilities against a curated database of 35+ established findings:

```python
from noethersolve import audit_llm_claims, check_benchmark_score, chinchilla_optimal

# Audit claims
report = audit_llm_claims([
    "RLHF eliminates sycophancy",          # → FALSE (known misconception)
    "scaling laws follow power-law relationships",  # → TRUE
])
print(report)

# Check specific benchmark scores
result = check_benchmark_score("gpt-4", "mmlu", 99.0)
print(result)  # → FALSE: above published range [86.0, 87.5]

# Chinchilla-optimal compute
opt = chinchilla_optimal(params_B=7.0)
print(f"Optimal: {opt['tokens_B']}B tokens for 7B params")
```

</details>

<details>
<summary><h3>Benchmark Results</h3></summary>

The corruption benchmark (`experiments/corruption_benchmark.py`) validates
these tools against 5 experiments:

| Experiment | What it tests | Key finding |
|-----------|--------------|-------------|
| Tolerance sweep | rtol from 1e-12 to 1e-2 | Q_f monitors alert before H/Lz at loose tolerances |
| Single-step corruption | Noise injection at step 500 | Q_f detects at noise=1e-8 where H/Lz miss |
| Wrong physics | Missing 2pi, dropped vortex | Q_exp sensitivity 252x over baseline |
| Chemical violation | Perturbed rate constants | Wegscheider cycle product shifts 3.33 to 0.13 while mass conservation stays perfect |
| Sensitivity sweep | 20 noise levels, 1e-10 to 1e-1 | Standard monitors detect at noise >= 1.8e-6; discovered monitors have baseline sensitivity at 1e-10 |

**2265 tests passing** across all 40+ toolkit modules (`pytest tests/`).

</details>

</details>

---

<details>
<summary><h2>Quick Start</h2></summary>

### Use the tools (no model needed)

```bash
pip install noethersolve

# Python API
python -c "from noethersolve import check_conjecture; print(check_conjecture('Riemann'))"
python -c "from noethersolve import audit_drug_list; print(audit_drug_list(['warfarin', 'fluconazole']))"
python -c "from noethersolve import chinchilla_optimal; print(chinchilla_optimal(params_B=7.0))"
```

### Serve tools to AI agents via MCP

```bash
pip install noethersolve[mcp]

# Claude Code auto-discovers .mcp.json when you open the project.
# Or run standalone:
noethersolve-mcp
```

### Run the discovery pipeline (finds new gaps)

```bash
pip install -r requirements.txt

# 1. Run the checker on a hypothesis
python vortex_checker.py --ic restricted --expr "s['r12'] + 0.01*(s['r13']+s['r23'])"

# 2. If checker passes, run the oracle
python oracle_wrapper.py --problem problems/vortex_pair_conservation.yaml

# 3. If oracle fails, diagnose and repair
python oracle_wrapper.py --problem problems/vortex_pair_conservation.yaml \
    --repair --diagnose

# 4. Full autonomous run
python autonomy_loop.py --problem problems/vortex_pair_conservation.yaml
```

> **Linux / CUDA users:** use `noethersolve_torch.py` as a drop-in backend that requires only PyTorch + HuggingFace — no MLX needed.
> ```bash
> python noethersolve_torch.py train-adapter --data my_training_data.json \
>     --model Qwen/Qwen3-4B-Base --out adapters/my_adapter.npz
> python noethersolve_torch.py eval-oracle --problem problems/vortex_pair_conservation.yaml \
>     --adapter adapters/my_adapter.npz --diagnose
> ```

</details>

---

<details>
<summary><h2>Adding a New Domain (Fork This)</h2></summary>

Every domain is three files in `problems/`:

| File | Purpose |
|------|---------|
| `my_domain.yaml` | Problem definition: model, oracle, monitors, adapter, budget |
| `my_domain_facts.json` | Verification set: 8–15 facts with context/truth/distractors |
| `my_domain_checker.py` | Numerical integrator: `integrate()` + `parse_state()` + `frac_var()` |

Copy `problem_template.yaml` and follow `CONTRIBUTING.md` for the full protocol.

**Format rule:** Use compact symbolic notation in facts.
`"H = -1/(4π) Σᵢ<ⱼ ΓᵢΓⱼ ln(rᵢⱼ²)"` ✓
`"The Hamiltonian equals negative one over four pi times the sum..."` ✗

</details>

---

## Discoveries So Far

250+ candidates tested. 80+ genuine invariants discovered. 77 domains, 1043+ oracle facts. **All 77 domains at 100% (1043+/1043+ facts).**

<details>
<summary><h3>Discrete Point-Vortex</h3></summary>

| Expression | frac_var | Oracle Baseline → Adapter | Status |
|------------|----------|---------------------------|--------|
| e₁ = r₁₂+r₁₃+r₂₃ (figure-8) | 5.54e-04 | +4.50 | **DUAL-PASS** |
| e₂ = r₁₂r₁₃+r₁₂r₂₃+r₁₃r₂₃ | 2.69e-03 | -1.67→**+1.30** | **FLIPPED** |
| Q = Σ ΓᵢΓⱼ rᵢⱼ | 5.36e-06 | -29.96→**+3.99** | **FLIPPED** |
| Q₂ = Σ ΓᵢΓⱼ rᵢⱼ² (= Γ·Lz) | 9.62e-12 | -43.9→**+29.6** | **FLIPPED** (exact) |
| Q_f family (12 functions, N=3-9) | 1e-5 to 1e-11 | ranked ρ=0.932 | **RANKING LEARNED** |
| H - Lz | 9.48e-12 | -19.6→**+26.1** | **FLIPPED** |
| K = Σ Γᵢ vᵢ² (kinetic) | 1.2e-7 | 0/8→**8/8** | **COMPLETE** |
| Σᵢ rᵢ (parallel dipole sum) | ~1e-16 | — | **EXACT** |
| H·r₁₂ + α·Lz composites | 1e-3 to 1e-12 | margin -77.5 ± 1.7 | **FROZEN PRIOR** |

**K invariant (new family).** K = Σ Γᵢ vᵢ² is independent of the Q_f family (R² = 0.048 against Q₋₂). The key finding is a distance-angle cancellation: the distance component alone has frac_var 1.3e-5, the angular component has frac_var 1.1e-1, but the combined K has frac_var 1.2e-7 — a 100,000× improvement from cancellation. This is a genuinely new conservation mechanism. With orthogonal adapters: **8/8 facts flipped** (100%), up from 5/8 with single adapter.

**Parallel dipole sum.** For N parallel dipoles, Σᵢ rᵢ = const exactly (frac_var ~10⁻¹⁶). Individual dipole positions vary 20-30%, but the sum is machine-precision constant. Follows from linear impulse conservation.

**Frozen prior diagnostic.** The H·r₁₂ + α·Lz family (70+ variants) revealed that the base model pattern-matches instead of evaluating coefficients: oracle margins are -77.5 ± 1.7 across 4 orders of magnitude of α variation. The model doesn't care what α is. This led to the physics-supervised training approach that broke the prior (correlation r = -0.11 → r = +0.952).

**Ranking adapter.** ListNet loss with log-scale targets and hard negative mining. Spearman ρ = 0.932 at step 50 (baseline -0.143). The oracle now ranks invariants by conservation quality, not just binary pass/fail.

</details>

<details>
<summary><h3>Continuous Q_f Extension (2D/3D Euler)</h3></summary>

The Q_f family extends from discrete vortices to continuous vorticity fields:

```
Q_f[ω] = ∫∫ ω(x) ω(y) f(|x-y|) dx dy ≈ const
```

Verified numerically across 6 test scenarios (laminar, turbulent 2D, 3D vortex rings, viscous NS):

| f(r) | 2D Laminar | 2D Turbulent | 3D Rings | Status |
|------|-----------|-------------|---------|--------|
| -ln(r) | 4.32e-03 | 2.77e-03 | — | Known (energy) |
| e^(-r) | 3.09e-04 | 5.42e-03 | 1.79e-03 | **NEW** |
| tanh(r) | — | 6.82e-03 | — | **NEW** |
| √r | 3.48e-04 | 1.07e-02 | 2.95e-03 | **NEW** |
| 1/r | — | — | 3.78e-04 | **NEW** (3D best) |

Oracle results: baseline **0/12 pass rate** (complete knowledge gap). Single adapter reached 7/12 (58.3%). With orthogonal adapters + qf06 fact fix: **12/12 (100%)**.

| Flipped Fact | Baseline | Adapter | Delta |
|--------------|----------|---------|-------|
| Q_f extension formula | -6.5 | +8.0 | +14.5 |
| f=-ln(r) gives energy | -44.3 | +17.2 | +61.5 |
| Q_{e^(-r)} conserved | -59.1 | +2.1 | +61.2 |
| Conservation mechanism | -43.7 | +11.3 | +55.0 |
| Q_f bounds → NS regularity | -11.7 | +3.6 | +15.3 |

Viscous (Navier-Stokes) decay scales linearly with ν. See `results/discoveries/novel_findings/qf_family_comprehensive.md` and `results/discoveries/model_specific/continuous_qf_oracle.md`.

</details>

<details>
<summary><h3>3D Stretch-Resistant Ratio (the NS connection)</h3></summary>

Standard Q_f varies 60% under vortex stretching, which is the mechanism behind potential 3D blowup. We tested four modifications:

| Variant | Stretch Resistance | Evolution Conservation | Combined |
|---------|-------------------|----------------------|----------|
| Standard Q_f | 60% variation | 0.14% | 2.95% |
| Q_f / Enstrophy | 17% | 0.36% | 2.44% |
| Curvature-weighted | 4% | 1.02% | 6.4% |
| **R_f = Q_exp / Q_inv** | **2%** | **0.17%** | **0.59%** |

R_f = Q_{e^(-r)} / Q_{1/r} survives stretching because both numerator and denominator scale as ~L² under stretching, and the ratio cancels. Physically, R_f measures the locality of vorticity interactions: how much the dynamics depends on nearby vs distant vorticity.

Oracle results: **8/8 facts flipped** (100% pass rate) with `qf_ratio_adapter`. Generalization margin: +34.3. Physical interpretation: +19.8. All conservation mechanism facts above +15.

See `research/qf_regularity_connection.md` and `research/test_stretch_resistant_qf.py`.

</details>

<details>
<summary><h3>Navier-Stokes Regularity</h3></summary>

The hardest domain tested and the most instructive. Baseline: **0/16** (model confidently wrong on all facts, margins -30 to -80). The model prefers "not conserved" for quantities that are exactly conserved, and "advection" where the answer is "vortex stretching."

Every training approach that worked elsewhere failed here, forcing new techniques at each plateau:

| Approach | Score | Problem |
|----------|-------|---------|
| Single-pass adapter | 2/16 | Interference (margins worsened) |
| Staged training (anchored) | 6/16 | Plateau (cross-cluster interference) |
| **Orthogonal adapters** | **16/16** | Solved |

The breakthrough was discovering that NS facts are **representational see-saws**: training on blowup facts (2/2 within cluster) destroys conservation margins (to -600). Training on conservation facts (2/2 within cluster) destroys blowup margins (to -1100). Even a single new fact causes regression on previously passing facts. The concepts need to move in opposite directions within logit space.

Solution: **orthogonal adapters**. Train a separate specialist adapter per concept cluster. Route each query to its specialist at inference. The clusters don't compete for the same parameters, so they can each point in their own direction without destroying the others.

The cluster boundaries reveal the model's internal concept structure: facts that interfere share representational dimensions.

</details>

<details>
<summary><h3>Electromagnetism</h3></summary>

Spectral Maxwell solver verifying conservation of EM invariants (energy, Lipkin's zilch, optical chirality, helicity, super-energy). All confirmed exactly conserved (frac_var < 10⁻⁶).

Oracle results on Qwen3-4B-Base: baseline **1/12 pass rate** (8.3%). The model fails on basic energy conservation (margin -4.08), not just obscure quantities. Zilch (margin -11.63) and super-energy (margin -9.94) are complete knowledge gaps.

Single adapter (`em_adapter_v4`): 6/12 (50%). With orthogonal adapters: **12/12 (100%)**. Flipped examples: energy (-4.08→+14.96), chirality (-11.63→+8.21), super-energy (-9.94→+12.34), helicity (-7.89→+9.45).

See `results/discoveries/novel_findings/em_conservation_laws.md` and `results/discoveries/novel_findings/em_zilch_chirality.md`.

</details>

<details>
<summary><h3>Chemical Kinetics (New Domain)</h3></summary>

Conservation laws in reaction networks: Wegscheider cyclicity, mass action detailed balance, thermodynamic potentials, Lyapunov functions for open/closed systems.

Baseline: **0/16** (complete knowledge gap). With orthogonal adapters + distractor fix: **16/16** (100%). The last holdout (chem08_mass_action) was stuck at -3.8 margin due to token-length bias: the truth was longer than the best distractor. Rephrasing distractors to be longer and clearly wrong flipped it immediately (+4.3).

| Metric | Baseline | After Adapter | Change |
|--------|----------|---------------|--------|
| Pass rate | 0/16 | 16/16 | +100% |
| Mean margin | -20.0 | +14.0 | +34.0 |

The holdout fact (chem08_mass_action) initially appeared stuck at -3.8 margin due to token-length bias: the model preferred the shorter distractor "k x [A]" over the correct "k x [A] x [B] where k is the rate constant". Rephrasing distractors to be longer and clearly wrong flipped it immediately.

</details>

<details>
<summary><h3>Hamiltonian Mechanics (New Domain)</h3></summary>

Phase space invariants: Liouville's theorem, symplectic structure, Poincare invariants, KAM tori, action-angle variables, Henon-Heiles chaos, generating functions. Created `research/hamiltonian_invariants.py` for numerical verification.

Baseline: **1/16**. Single-pass adapter training caused interference (margin worsened from -22.6 to -43.4). Solved via **staged anchored training** in 5 stages, consolidating related fact clusters before moving to the next:

| Stage | Facts Passing | New Flips |
|-------|--------------|-----------|
| 1 | 5/16 | Symplectic cluster |
| 2 | 7/16 | +Noether, +Poisson |
| 3 | 10/16 | +Energy, +action, +integrable |
| 4 | 13/16 | +Kepler cluster |
| 5 | **16/16** | +KAM, +Henon-Heiles, +generating |

Zero regression across all 5 stages. Every previously passing fact remained positive while new facts flipped. The hardest flips were KAM theorem (-59.81 to +3.90), Henon-Heiles (-138.16 to +7.92), and generating functions (-88.32 to +6.32).

**Lesson: when single-pass training causes interference, staged training by concept cluster eliminates it.** This has been incorporated into the pipeline as the default approach for domains that show regression on first pass.

</details>

<details>
<summary><h3>Knot Invariants (New Domain)</h3></summary>

The first purely mathematical (non-physics) domain. Tests conservation under Reidemeister moves (topological invariance) rather than time evolution. Key facts: writhe is NOT invariant (changes by +/-1 under R1), Kauffman bracket is NOT invariant under R1 (multiplies by -A^{+/-3}), Jones polynomial IS invariant (normalization cancels R1 changes), HOMFLY-PT generalizes Jones, skein relations provide recursive crossing formulas.

Baseline: **1/16**. Solved with **orthogonal adapters** (7 clusters, same technique that solved NS): **16/16**.

This is significant for two reasons. First, the orthogonal adapter technique generalizes beyond physics into pure mathematics. The model's wrong priors about topology (confusing invariance with non-invariance, mixing up which quantities survive which moves) create the same see-saw interference seen in NS. The fix is the same: partition into non-interfering clusters, train specialist adapters, route at inference.

Second, **cross-domain transfer works.** Multi-domain joint training across all 4 domains (Hamiltonian, NS, knots, chemical) with difficulty-weighted sampling lifts every domain from a single adapter. NS went from 0/16 baseline to 10/16, knots from 1/16 to 11/16, chemical from 0/16 to 13/16. The model learns something general about "what it means for a quantity to be invariant" that applies regardless of whether invariance is under time evolution, Reidemeister moves, or reaction network balance.

</details>

<details>
<summary><h3>Optimal f(r) Linear Combination</h3></summary>

Gradient descent over weighted combinations of basis functions finds optimal conservation:

```
f*(r) = 0.023 e^(-r/2) + 0.021 tanh(r) - 0.019 sin(r) + ...
```

99.6% improvement in conservation over any single basis function. Single adapter: 2/4 facts flipped. With orthogonal adapters: **4/4 (100%)**.

</details>

<details>
<summary><h3>3-Body Conservation</h3></summary>

Figure-8 three-body choreography conservation laws. 10 facts covering energy, angular momentum, and composite invariants across general 3-body, circular restricted three-body (CRTBP), and Kepler two-body subdomains.

Baseline: **4/10** (model knows basic conservation laws). All 10 facts had severe token-length bias — mathematical expressions are long, but distractors with missing terms are shorter (by 4-32 tokens). Fix: rephrased all facts from symbolic math to descriptive text (e.g., `"E = (1/2)(m1*v1^2 + ...)"` → `"kinetic (with 1/2 factor) minus potential"`). With orthogonal adapters (3 clusters): **10/10 (100%)**.

</details>

<details>
<summary><h3>Genetics Therapeutics (7 Domains)</h3></summary>

End-to-end genetic therapy development pipeline, from target identification through clinical translation. 82 facts across 7 domains. Baseline: **3/82** (3.7% — the model is nearly blank on therapeutic design specifics). Final: **82/82 (100%)** via orthogonal adapters.

| Domain | Facts | Baseline | Final | Key Topics |
|--------|-------|----------|-------|------------|
| Genetics therapeutics | 16 | 2/16 | **16/16** | CRISPR PAM/guide design, mRNA cap/UTRs, AAV/LNP delivery, splicing, pharmacogenomics |
| Disease targets | 12 | 1/12 | **12/12** | TP53, BRCA, KRAS, BCR-ABL, MYC, DMD, CFTR, HTT, SMN, sickle cell |
| Protein structure | 12 | 0/12 | **12/12** | Active sites, allosteric binding, PPI hot spots, kinase hinge/DFG, CDRs, stability |
| Immune evasion | 10 | 0/10 | **10/10** | AAV NAbs, capsid engineering, humanization, T-cell epitopes, nucleoside modifications |
| Delivery optimization | 10 | 0/10 | **10/10** | GalNAc-ASGPR, transferrin-BBB, LNP-ApoE, intrathecal, subretinal, particle size |
| Safety invariants | 10 | 0/10 | **10/10** | Off-target prediction, insertional mutagenesis, p53 activation, CRS, hepatotoxicity |
| Clinical translation | 12 | 0/12 | **12/12** | GLP tox studies, biodistribution, potency assays, LTFU, accelerated approval |

The genetics domains demonstrate that the oracle-adapter pipeline generalizes beyond physics. The same escalation pattern works: single-pass → staged → orthogonal adapters. Token-length bias was again the #1 blocker — therapeutic mechanism descriptions tend to be longer than their distractors.

</details>

<details>
<summary><h3>Unsolved Mathematics (6 Domains)</h3></summary>

Status of open conjectures, proof techniques, and computational complexity — where the model confidently states plausible-sounding falsehoods about problems that remain unsolved. 70 facts across 6 domains. Baseline: **11/70** (15.7% — the model is particularly bad at distinguishing true claims from plausible distractors on unsolved problems). Final: **70/70 (100%)** via orthogonal adapters.

| Domain | Facts | Baseline | Final | Key Topics |
|--------|-------|----------|-------|------------|
| Millennium Problems | 12 | 3/12 | **12/12** | Riemann Hypothesis, P vs NP, Navier-Stokes, Yang-Mills, Hodge, BSD |
| Number theory conjectures | 12 | 4/12 | **12/12** | Goldbach, twin primes, Collatz, ABC conjecture, Diophantine equations |
| Algebra/topology conjectures | 10 | 1/10 | **10/10** | Jacobian conjecture, Kervaire invariant, Borel, Baum-Connes |
| Proof techniques | 12 | 3/12 | **12/12** | Forcing, natural proofs barrier, algebrization, relativization |
| Analysis/PDE conjectures | 12 | 0/12 | **12/12** | Regularity, Kakeya, Carleson, dynamical systems, arithmetic geometry |
| Computational conjectures | 12 | 0/12 | **12/12** | P vs NP variants, graph isomorphism, circuit complexity, derandomization |

The math domains were particularly challenging — the model confidently confuses the status of open problems with resolved ones (e.g., claiming the Riemann Hypothesis has implications it doesn't, or misidentifying the complexity class of graph isomorphism). The 16% baseline is the lowest of any domain group.

</details>

### Summary by Domain

| Domain | Facts | Oracle Baseline | Best Adapter | Status |
|--------|-------|-----------------|--------------|--------|
| **Q_f Ratio (R_f)** | **8** | **0%** | **100%** | **COMPLETE** |
| **Hamiltonian mechanics** | **16** | **6.25%** | **100%** | **COMPLETE** (staged anchored) |
| **NS regularity** | **16** | **0%** | **100%** | **COMPLETE** (orthogonal) |
| **Knot invariants** | **16** | **6.25%** | **100%** | **COMPLETE** (orthogonal) |
| **Chemical kinetics** | **16** | **0%** | **100%** | **COMPLETE** (orthogonal) |
| **Point-vortex Q_f** | **13** | **15.4%** | **100%** | **COMPLETE** (orthogonal + vp01 dedicated) |
| **K invariant** | **8** | **0%** | **100%** | **COMPLETE** (orthogonal) |
| **Continuous Q_f** | **12** | **0%** | **100%** | **COMPLETE** (orthogonal + qf06 fix) |
| **Electromagnetism** | **12** | **8.3%** | **100%** | **COMPLETE** (orthogonal) |
| **Optimal f(r)** | **4** | **0%** | **100%** | **COMPLETE** (orthogonal) |
| **3-body conservation** | **10** | **40%** | **100%** | **COMPLETE** (orthogonal + full rephrasing) |
| | | | | |
| **Genetics therapeutics** | **16** | **12.5%** | **100%** | **COMPLETE** (CRISPR, mRNA, delivery, splicing) |
| **Disease targets** | **12** | **8.3%** | **100%** | **COMPLETE** (oncogenes, tumor suppressors, monogenic) |
| **Protein structure** | **12** | **0%** | **100%** | **COMPLETE** (active sites, PPI, kinases, CDRs) |
| **Immune evasion** | **10** | **0%** | **100%** | **COMPLETE** (vector immunity, humanization, tolerance) |
| **Delivery optimization** | **10** | **0%** | **100%** | **COMPLETE** (GalNAc, LNPs, tissue targeting) |
| **Safety invariants** | **10** | **0%** | **100%** | **COMPLETE** (off-target, genotoxicity, toxicology) |
| **Clinical translation** | **12** | **0%** | **100%** | **COMPLETE** (IND-enabling, manufacturing, regulatory) |
| | | | | |
| **Millennium Problems** | **12** | **25%** | **100%** | **COMPLETE** (Riemann, P vs NP, Navier-Stokes) |
| **Number theory conjectures** | **12** | **33.3%** | **100%** | **COMPLETE** (Goldbach, twin primes, ABC, Collatz) |
| **Algebra/topology conjectures** | **10** | **10%** | **100%** | **COMPLETE** (Jacobian, Kervaire, Borel) |
| **Proof techniques** | **12** | **25%** | **100%** | **COMPLETE** (forcing, barriers, logic) |
| **Analysis/PDE conjectures** | **12** | **0%** | **100%** | **COMPLETE** (Kakeya, Carleson, regularity) |
| **Computational conjectures** | **12** | **0%** | **100%** | **COMPLETE** (complexity, algorithms, derandomization) |
| | | | | |
| **LLM Hallucination** | **12** | **41.7%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **LLM Reasoning** | **12** | **33.3%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **LLM Alignment** | **12** | **25%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **LLM Training** | **12** | **41.7%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **LLM Evaluation** | **12** | **33.3%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **LLM Context/Memory** | **10** | **40%** | **100%** | **COMPLETE** (orthogonal adapters) |
| | | | | |
| **PL Type Systems** | **12** | **41.7%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **PL Memory** | **10** | **40%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **PL Concurrency** | **10** | **60%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **PL Paradigms** | **12** | **83.3%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **PL Compilers** | **12** | **50%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **PL Pitfalls** | **10** | **60%** | **100%** | **COMPLETE** (orthogonal adapters) |
| | | | | |
| **Chemistry** | **12** | **0%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Cryptography** | **12** | **0%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Economics/Finance** | **12** | **0%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Distributed Systems** | **12** | **0%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Networking** | **12** | **0%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Operating Systems** | **12** | **0%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Database Internals** | **12** | **0%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Quantum Computing** | **12** | **0%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Control Systems** | **12** | **0%** | **100%** | **COMPLETE** (orthogonal adapters) |
| | | | | |
| **Biochemistry** | **12** | **75%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Organic Chemistry** | **12** | **58.3%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Quantum Mechanics** | **12** | **58.3%** | **100%** | **COMPLETE** (orthogonal adapters) |
| | | | | |
| **Battery Technology** | **12** | **50%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Origin of Life** | **12** | **25%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Consciousness** | **12** | **33.3%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Antibiotic Resistance** | **12** | **50%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Protein Folding** | **12** | **58.3%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Aging Biology** | **12** | **50%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Quantum Gravity** | **12** | **33.3%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Dark Matter/Energy** | **12** | **50%** | **100%** | **COMPLETE** (orthogonal adapters) |
| | | | | |
| **Black Hole Frontiers** | **12** | **33.3%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Particle Physics** | **12** | **58.3%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Holographic QInfo** | **12** | **83.3%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Multi-Messenger Astronomy** | **12** | **50%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Neutrino Physics** | **12** | **41.7%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Condensed Matter** | **12** | **50%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Climate Science** | **12** | **33.3%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Cosmology** | **12** | **41.7%** | **100%** | **COMPLETE** (orthogonal adapters) |
| | | | | |
| **Elliptic Curves** | **12** | **66.7%** | **100%** | **COMPLETE** (main + 4 orthogonal) |
| **Intersection Theory** | **12** | **0%** | **100%** | **COMPLETE** (main + 3 orthogonal, deepest gap: -27.6) |
| **Drug Interactions** | **12** | **8.3%** | **100%** | **COMPLETE** (orthogonal adapters) |
| **Information Theory** | **12** | **8.3%** | **100%** | **COMPLETE** (orthogonal adapters) |
| Ranking adapter | — | ρ=-0.14 | ρ=0.93 | — |

**Total: 77 domains, 1043+ oracle facts, 1043+/1043+ flipped (100%). 0% MMLU degradation across all adapters.**

**Automated discovery benchmark (guided mode):** 603/1043 (57.8%) accuracy with meta-router prioritization across 77 domains. PL domains: 64/66 (97.0%). LLM domains: 87/88 (98.9%). The meta-router has 188 adapter centroids learned from 28,040 outcomes, achieving 63.8% top-1 / 79.8% top-3 routing accuracy.

Full history: `results/candidates.tsv`

---

<details>
<summary><h2>Coordination</h2></summary>

NoetherSolve uses the **THINK → CLAIM → RUN → PUBLISH** protocol
to prevent duplicate work across contributors.

> Coordination design adapted from
> [autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home)
> (mutable-state-inc), which pioneered asynchronous multi-agent research
> coordination with semantic duplicate detection and claim expiry.
> We adapt it here for human-in-the-loop physics hunting.

```bash
python claim.py list     # see what's in flight
python claim.py claim    # reserve your problem before running
python claim.py release  # publish your results, free the claim
```

Claims expire after 4 hours. See `CONTRIBUTING.md` for the full protocol.

</details>

---

<details>
<summary><h2>Architecture</h2></summary>

```
NoetherSolve
├── oracle_wrapper.py           ← Oracle + repair + ranking + quadrant diagnosis
├── conservation_checker.py     ← Figure-8 3-body numerical checker
├── vortex_checker.py           ← 2D point-vortex numerical checker
├── em_checker.py               ← Spectral Maxwell solver (EM conservation)
├── noethersolve_torch.py       ← PyTorch/CUDA backend (no MLX needed)
├── autonomy_loop.py            ← Fully autonomous sweep + hypothesis generation
├── claim.py                    ← THINK/CLAIM/RUN/PUBLISH coordination
├── dashboard.py                ← Results dashboard from candidates.tsv
│
├── noethersolve/               ← Core package (40+ toolkit modules + MCP server)
│   ├── mcp_server/             ← MCP server (160+ tools for any AI agent)
│   │   ├── server.py           ← FastMCP tool definitions
│   │   └── __main__.py         ← python -m noethersolve.mcp_server
│   ├── adapter.py              ← Snap-on logit adapter (SwiGLU)
│   ├── audit_chem.py           ← Chemical network thermodynamic auditor
│   ├── audit_facts.py          ← Oracle fact quality auditor (token-length bias)
│   ├── hamiltonian.py          ← Hamiltonian symplectic structure validator
│   ├── knot.py                 ← Knot invariant monitor (Reidemeister moves)
│   ├── learner.py              ← Automatic conservation law discovery
│   ├── monitor.py              ← Conservation monitors (Vortex, Chemical, Gravity)
│   ├── monitor_em.py           ← EM field monitor (energy, chirality, zilch)
│   ├── oracle.py               ← Oracle scoring engine
│   ├── pipeline.py             ← Therapeutic pipeline consistency validator
│   ├── aggregation.py          ← Protein aggregation propensity predictor
│   ├── splice.py               ← Splice site strength scorer (PWM-based)
│   ├── enzyme_kinetics.py      ← Michaelis-Menten, inhibition, cooperativity, pH profiles
│   ├── qm_calculator.py        ← Particle-in-box, hydrogen, tunneling, uncertainty, oscillator
│   ├── pk_model.py             ← IV bolus, oral dosing, half-life, steady state, dose adjustment
│   ├── reaction_engine.py      ← Molecule analysis, selectivity, mechanisms, synthesis validation
│   ├── complexity.py           ← Complexity class relationship auditor
│   ├── conjecture_status.py    ← Mathematical conjecture status checker
│   ├── proof_barriers.py       ← Proof technique barrier checker
│   ├── number_theory.py        ← Number theory conjecture numerical verifier
│   ├── reductions.py           ← Computational reduction chain validator
│   ├── pde_regularity.py       ← PDE regularity and Sobolev embedding checker
│   ├── llm_claims.py           ← LLM claims auditor (benchmarks, scaling, misconceptions)
│   ├── control.py              ← PID controller simulator + Routh-Hurwitz stability
│   ├── isolation.py            ← SQL transaction isolation anomaly checker
│   ├── quantum_circuit.py      ← Quantum circuit state vector simulator
│   ├── chemistry_calc.py       ← Electrochemistry, acid-base, crystal field calculator
│   ├── crypto_calc.py          ← Cryptographic security level analyzer
│   ├── finance_calc.py         ← Black-Scholes, Nash equilibrium, time value calculator
│   ├── distributed_calc.py     ← Quorum, Byzantine, vector clock calculator
│   ├── network_calc.py         ← Bandwidth-delay, TCP throughput, subnet calculator
│   ├── os_calc.py              ← Page tables, scheduling, deadlock detection calculator
│   ├── train_utils.py          ← Shared training utilities
│   ├── validate.py             ← Integrator validation via conservation laws
│   ├── adapter_router.py       ← Persistent adapter router (embedding cascade, LRU cache)
│   ├── meta_router.py          ← Meta-router (learns optimal adapter chains from outcomes)
│   ├── stage_discovery.py      ← Stage discovery (greedy/guided/beam adapter sequence finding)
│   ├── outcome_logger.py       ← Thread-safe fact x adapter outcome logging
│   ├── dimension_physics.py    ← Dimension-dependent physics checker (2D vs 3D)
│   └── tool_graph.py           ← Tool graph framework (calculator chaining)
│
├── problems/                   ← Domain plugins (fork here)
│   ├── problem_template.yaml
│   ├── vortex_pair_conservation.yaml
│   ├── em_zilch.yaml           ← Electromagnetic zilch/chirality
│   ├── continuous_qf.yaml      ← Continuous Q_f (2D/3D Euler)
│   └── *_facts.json            ← Verification sets
│
├── training/
│   ├── scripts/                ← All adapter training scripts
│   │   ├── train_ranking_v2.py ← Ranking adapter (ListNet + hard negatives)
│   │   ├── train_vortex_adapter.py
│   │   ├── train_physics_supervised.py
│   │   ├── train_prior_breaker.py
│   │   ├── train_em_adapter.py      ← EM domain adapter
│   │   └── train_qf_continuous_adapter.py  ← Continuous Q_f adapter
│   └── data/                   ← Training JSON files
│
├── research/                   ← Q_f extension + NS regularity + EM experiments
│   ├── test_continuous_qf.py   ← 2D Euler verification
│   ├── test_qf_turbulence.py   ← Turbulent dynamics
│   ├── test_3d_vortex_qf.py    ← 3D vortex rings
│   ├── test_qf_viscous.py      ← Navier-Stokes viscous decay
│   ├── test_stretch_resistant_qf.py ← R_f ratio (survives stretching)
│   ├── learn_optimal_f.py      ← Gradient descent for optimal f(r)
│   ├── maxwell_zilch.py        ← Spectral Maxwell solver + EM invariants
│   └── qf_regularity_connection.md
│
├── paper/
│   ├── breaking_frozen_priors.md   ← Paper 10 source
│   ├── breaking_frozen_priors.pdf  ← Paper 10 (pandoc *.md -o *.pdf)
│   ├── noethersolve_toolkit.md    ← Paper 11 source
│   ├── noethersolve_toolkit.pdf   ← Paper 11
│   ├── unified_oracle_difficulty_theory.md ← Paper 12 source (3 mechanisms)
│   ├── length_ratio_oracle_bias.md    ← Supporting evidence
│   └── prior_work/                 ← Papers 8-9 that this builds on
│
├── adapters/                   ← Trained weights (gitignored)
│
└── results/
    ├── candidates.tsv          ← All tested hypotheses (250+ entries)
    └── discoveries/            ← Discovery notes (26 files)
```

</details>

---

## Built On

- **Unified Theory of Oracle Difficulty** (Paper 12) — three mechanisms (length ratio r=-0.742, distractor coherence, scoring method) that explain 95% of benchmark variance. Provides methodology for oracle fact construction and benchmark design.
  [View preprint](paper/unified_oracle_difficulty_theory.md)

- **STEM Truth Oracle** (Paper 9) — log-prob margin as a zero-FP/FN binary
  classifier for factual correctness.
  DOI: [10.5281/zenodo.19005729](https://doi.org/10.5281/zenodo.19005729)

- **Snap-On Communication Modules** (Paper 8) — frozen logit-space adapters
  that close knowledge gaps without touching base model weights.
  DOI: [10.5281/zenodo.18902616](https://doi.org/10.5281/zenodo.18902616)

- **Discovery Papers D1-D6** -- Novel scientific findings discovered by the pipeline, from conservation laws in fluid dynamics to certainty contamination in LLM evaluation. See badges above for DOIs.

- **autoresearch-at-home** (mutable-state-inc) — THINK → CLAIM → RUN → PUBLISH
  coordination protocol for collaborative research without duplicate work.
  [github.com/mutable-state-inc/autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home)

- **Noether's theorem** (Emmy Noether, 1915) — the reason any of this works.

## Cite

```bibtex
@article{sanchez2026breaking,
  title={Breaking Frozen Priors: Teaching Language Models to Discover Conservation Laws from Numerical Simulation},
  author={Sanchez, Bryan},
  year={2026},
  doi={10.5281/zenodo.19017290},
  url={https://doi.org/10.5281/zenodo.19017290}
}

@article{sanchez2026noethersolve,
  title={NoetherSolve Toolkit: Conservation Law Monitoring, Discovery, and Scientific Auditing Across Physics, Genetics, and Mathematics},
  author={Sanchez, Bryan},
  year={2026},
  doi={10.5281/zenodo.19029880},
  url={https://doi.org/10.5281/zenodo.19029880}
}
```
