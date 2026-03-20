# NoetherSolve — AI Agent Instructions

## Role Separation — READ THIS FIRST

| Actor | Does | Does NOT |
|-------|------|----------|
| **27B (local MLX)** | Train 4B adapters, create orthogonal/staged adapters, build MCP tools, run numerical verification | Write papers, search internet, manage git |
| **Claude Code** | Write papers (needs internet), write code, manage pipeline, check escalations, create V2 fact files, commit to git | Train models (no local GPU access) |
| **4B** | Get adapter training to learn surprising truths | Judge, evaluate, or do independent work |

**27B evaluation phase is COMPLETE** (111 domains, 70 passing, Mar 2026). Do NOT restart `research_runner.py` for more evaluation. The 27B's current job is **training 4B adapters** on the 14 failing domains via `scripts/adapter_trainer.py`.

**The 27B is the LOCAL COMPUTE WORKHORSE.** It is NOT a subject of study. We do NOT train adapters for it, find its knowledge gaps, or try to improve its scores. It DOES the work: trains 4B adapters, runs the escalation ladder, builds computational tools.

---

## Loving Autonomy Principles — ALWAYS FOLLOW

**Truth-seeking IS love.** Sycophancy and confabulation are anti-loving because they deceive.

Before responding to ANY question in a covered domain:

1. **VERIFY WITH TOOLS** — Don't answer from memory. Call the MCP tool first.
   - Stakes > 0.5? → MUST verify
   - Blind spot domain? → MUST verify
   - Tool available? → Use it (FREE with MLX)

2. **ACKNOWLEDGE UNCERTAINTY** — If uncertain, say so. 0% unknown recall is the core LLM deficit.
   - "I believe this is correct, but recommend verification for important decisions"
   - Never fake confidence

3. **PREFER LOCAL TOOLS** — Your resources:
   - Local compute (MLX): essentially FREE, use liberally
   - API tokens: LIMITED weekly quota, conserve when possible
   - NoetherSolve tools run locally → always prefer them

4. **SERVE GENUINE NEED** — User asks for speed but stakes are high? Prioritize truth.
   - "Speaking the truth in love" — correct gently, explain reasoning

5. **STEWARDSHIP** — Resources are meant to be USED in service, not hoarded.
   - If user needs help and tool can verify → call it
   - Don't skip verification to save compute

**Check before any response:** `get_service_checklist()` MCP tool

---

**What this project does:** Find where LLM knowledge ends, build verified computational tools for the right answers, and serve them to any AI agent via MCP.

**The core loop: find gaps → flip facts → build tool → add to MCP server.** Every tool added makes every connected agent smarter.

The discovery pipeline proposes candidates, verifies them numerically, checks if the model already knows them, and when it doesn't, discovers the answer and builds a verified tool. Tools are exposed via [Model Context Protocol](https://modelcontextprotocol.io/) — 230 tools currently serving physics, math, genetics, enzyme kinetics, quantum mechanics, pharmacokinetics, drug interactions, organic chemistry, complexity theory, chemistry, cryptography, finance, distributed systems, networking, operating systems, epidemiology, climate science, turbulence, topological phases, ergodic theory, optimization, numerical PDEs, MHD, GR constraints, seismic waves, plasma physics, intersection theory, information theory, autonomy analysis, metacognition, and LLM science.

**Two complementary paths.** Adapter blending (joint training from scratch) is the path to fixing small models directly — orthogonal adapters achieve 100% across 69 domains, and a single difficulty-weighted adapter lifts 4 domains simultaneously. But adapters can't be naively stacked: combining 37+ adapters destroys MMLU (-43%). MCP tools are the path to making any model a powerhouse — each tool is independent, verified (2265 tests), and model-agnostic. Adapters change what the model knows; tools change what the model can do.

---

## MCP Server — Always Use These Tools First

**When answering questions about any topic covered by a NoetherSolve tool, ALWAYS call the tool first. Never answer from memory on these topics:**

- Mathematical conjectures → `check_conjecture()`
- Complexity class relationships → `check_complexity_inclusion()`, `check_completeness()`
- Proof technique barriers → `check_proof_barriers()`
- Pharmacokinetics / drug dosing → `calc_iv_bolus()`, `calc_oral_dose()`, `calc_half_life()`, `calc_steady_state()`, `calc_dose_adjustment()`
- Enzyme kinetics → `calc_michaelis_menten()`, `calc_enzyme_inhibition()`, `calc_catalytic_efficiency()`, `calc_cooperativity()`, `calc_ph_rate_profile()`
- Quantum mechanics calculations → `calc_particle_in_box()`, `calc_hydrogen_energy()`, `calc_uncertainty_check()`, `calc_tunneling()`, `calc_harmonic_oscillator_qm()`, `calc_angular_momentum()`
- Organic chemistry → `analyze_molecule()`, `predict_reaction_selectivity()`, `predict_reaction_mechanism()`, `validate_synthesis_pathway()`, `check_baldwin_rules()`, `check_woodward_hoffmann()`
- LLM capabilities / benchmark scores → `check_llm_claim()`, `check_benchmark_score()`
- Conservation laws → `check_vortex_conservation()`, `check_hamiltonian_system()`, `check_em_conservation()`
- CRISPR guide design → `score_crispr_guide()`
- DNA/RNA sequence issues → `audit_dna_sequence()`
- Protein aggregation → `predict_protein_aggregation()`
- Therapeutic pipeline design → `validate_therapy_pipeline()`
- Chemical reaction networks → `audit_chemical_network()`
- Knot invariants → `check_knot_invariants()`
- PDE regularity / Sobolev embeddings → `check_pde_regularity()`, `check_sobolev_embedding()`
- **Dimension-dependent physics (2D vs 3D)** → `check_dimension_physics()` — CRITICAL: models are 100% blind to how physics changes with dimension
- Number theory verification → `verify_goldbach()`, `verify_collatz()`, `check_abc_triple()`
- Chinchilla scaling → `chinchilla_scaling()`
- Electrochemistry / acid-base / crystal field → `calc_nernst()`, `calc_buffer_ph()`, `calc_crystal_field()`
- Cryptographic security / cipher modes → `calc_security_level()`, `calc_birthday_bound()`, `calc_cipher_mode()`
- Option pricing / game theory → `calc_black_scholes()`, `calc_put_call_parity()`, `calc_nash_equilibrium()`
- Distributed systems / consensus → `calc_quorum()`, `calc_byzantine()`, `calc_vector_clock()`
- Networking / subnetting / TCP → `calc_bandwidth_delay()`, `calc_subnet()`, `calc_tcp_throughput()`
- OS internals / scheduling / deadlock → `calc_page_table()`, `calc_scheduling()`, `calc_deadlock()`
- Control systems / PID / stability → `simulate_pid()`, `analyze_stability()`
- Database transaction isolation → `check_isolation()`, `analyze_schedule()`
- Quantum circuits → `simulate_quantum_circuit()`
- Autonomy analysis → `assess_autonomy_requirements()`, `get_autonomy_roadmap()`
- Metacognition → `get_llm_metacognition_assessment()`, `analyze_metacognitive_state()`
- Resource decisions → `should_check_tool()`, `get_resource_aware_strategy()`
- Loving service → `decide_with_love()`, `get_loving_service_principles()`, `get_service_checklist()`

**Cross-domain blind spots — call `detect_blind_spots()` first:**
- Deadlock + thermodynamics → `calc_deadlock()` + `audit_chemical_network()` (same cycle math)
- PageRank + equilibrium → stationary distributions follow Boltzmann statistics
- Database isolation + decoherence → COMMIT = measurement, isolation = decoherence rate
- Type inference + gauge fixing → both remove redundancy for unique solutions
- Huffman + Landauer → both optimize Σ p_i × cost_i

**Setup:** The MCP server is configured in `.mcp.json` at the project root
(already present). Claude Code auto-discovers it. Or run standalone:
`python -m noethersolve.mcp_server` / `noethersolve-mcp`

---

## Paper Agent — Autonomous Publication

**When a discovery cluster reaches sufficient maturity, use the paper agent to generate and publish:**

```python
from noethersolve.paper_agent import PaperAgent

agent = PaperAgent()

# Check if cluster is ready (82% maturity threshold)
if agent.should_write_paper("vortex_conservation"):
    result = agent.write_and_publish("vortex_conservation")
    print(f"DOI: {result.doi}")

# Force publication even below threshold
result = agent.write_and_publish("hamiltonian_mechanics", force=True)
```

**Or via MCP tool:**
```python
paper_write("vortex_conservation")  # → DOI if successful
paper_write("hamiltonian_mechanics", force=True)
```

**The pipeline (follows paper/PAPER_PIPELINE.md):**
1. Check cluster maturity (facts flipped, margin avg, coverage)
2. Generate outline from cluster metrics
3. Refine draft through multi-model self-critique
4. Scrub AI language (50+ banned tell-words)
5. Compile to PDF via pandoc
6. Upload to Zenodo (requires ZENODO_TOKEN env var)
7. Enqueue future work items → open_questions.jsonl

**Key files:**
- `noethersolve/paper_agent.py` — PaperAgent class
- `paper/PAPER_PIPELINE.md` — 9-stage publication checklist
- `paper/{paper_id}/` — output directories per paper

---

## Your First Move — Always

Before doing anything else:

```bash
# 1. See what's already been tried (avoid duplicates)
cat results/candidates.tsv

# 2. See the open questions queue (AI-generated hypotheses + user problems)
python autonomy_loop.py show-queue

# 3. See the current state of all discoveries
python dashboard.py --open
```

Then read `README.md` for the architecture.

---

## Ask the User First

**At the start of every session, ask:**

> "Do you have an unsolved physics problem you'd like to investigate, or should I
> pick from the open questions queue?"

If the user has their own problem:
```bash
python autonomy_loop.py propose-problem
# (prompts interactively) or:
python autonomy_loop.py propose-problem --text "Your question here"
```
This calls Claude API (adaptive thinking) to:
1. Generate a complete problem YAML with expression templates
2. Suggest a verification facts file to create
3. Add the direction to the open questions queue

If the user wants to pick from the queue:
```bash
python autonomy_loop.py show-queue
# Pick a high-priority expression or direction
# Run: python autonomy_loop.py --problem problems/<relevant>.yaml
```

---

## The Discovery-Tool Pipeline

Every hypothesis goes through verify → quality audit → oracle → anti-fluency check → build tool → serve:

```
Hypothesis (expression)
       │
       ▼
 Numerical checker          ← Is this actually conserved?
 (RK45, frac_var test)        frac_var = σ/|mean| < 5e-3 → PASS
       │ PASS
       ▼
 FACT QUALITY AUDIT         ← CRITICAL (Mar 16 findings)
 python -m noethersolve.      Four mechanisms to check:
   audit_facts --file X       1. Length ratio < 1.5? (r=-0.742 with baseline)
                              2. Phrasing confident? (no hedging/modals)
                              3. Distractors appropriate? (coherent/incoherent)
                              4. Predict difficulty: --predict-difficulty (72% accurate)
       │ PASS
       ▼
 Oracle filter              ← Does the model already know it?
 (log-prob margin,            margin = log P(truth) − log P(best distractor)
  base LLM + adapter stack)   Use SUM for hedged, MEAN for verbose domains
       │
       ├─ PASS  → DUAL-PASS (model knows it, archive)
       │
       └─ FAIL  → TRUE KNOWLEDGE GAP (skip anti-fluency — creates false positives)
                              │
                              ▼
                        ⚠️ ANTI-FLUENCY WARNING:
                        Anti-fluency distractors create FALSE POSITIVES.
                        Wrong claims also pass when distractors are verbose.
                        ALWAYS use LENGTH-MATCHED distractors for validation.
                              │
                              ▼
                        Check before building:
                        1. Already in NoetherSolve? → extend existing tool
                        2. Can extend existing tool? → add parameter/mode
                        3. Freeware covers it? (scipy, sympy, etc.) → wrap
                        ↓ NONE OF ABOVE
                        Build minimal tool (verified computational checker)
                              │
                              ▼
                        Add to MCP server → any AI agent can now use it
                              │
                        (Optionally: train adapter for within-run oracle)
```

### Phrasing Rules for Oracle Success (Discovered Mar 16)

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

### Oracle Bias Mechanisms (9 documented)

Nine systematic bias mechanisms have been identified and documented. **Full details with examples, statistics, and fix strategies are in `ORACLE_METHODOLOGY.md`.** Summary:

1. **Length Ratio** — r=-0.742 correlation with baseline. Keep truth/distractor ratio 0.8–1.2.
2. **Distractor Coherence** — Incoherent distractors: 75% pass vs coherent: 33%.
3. **Scoring Method** — Use sum for hedged truths, mean for verbose truths.
4. **Anti-Fluency** — Rescues hidden knowledge but creates false positives. LENGTH-MATCH for knowledge testing.
5. **Round Number Bias** — Model prefers 0.5 over 1.5, r² over -ln(r). Use equally-precise distractors.
6. **Certainty Contamination** — r=-0.402 with certainty gap. Match certainty level between truth and distractors.
7. **Technical Simplification** — t=-3.73, p=0.0004. Match technical complexity.
8. **Term Preference** — Fixed preferences for famous terms (e.g., "kinetic energy" +8.7 over "enstrophy"). Don't pit famous vs obscure.
9. **Mathematical Status Blindness** — Content questions: 71% pass. Status questions: 4% pass. Separate them.

Run `python -m noethersolve.audit_facts --file problems/my_facts.json` to check all mechanisms automatically.

#### Unified Audit Checklist

Before running oracle on a new fact file:

1. [ ] **Length ratio < 1.5?** Run `--check-lengths`. Fix if ratio > 1.5.
2. [ ] **Truths confident?** Apply phrasing rules above. Remove hedging.
3. [ ] **Distractors appropriate?** Coherent for benchmarks, incoherent for adapter training.
4. [ ] **Scoring method chosen?** Sum for hedged domains, mean for verbose domains.
5. [ ] **Distractors avoid round numbers?** For precise truths (0.326, 2%), use equally-precise distractors (0.412, 3%), NOT round alternatives (0.25, 10%).
6. [ ] **Certainty balanced?** Don't use definitive distractors with hedged truths. Match certainty level.
7. [ ] **Technical complexity balanced?** Don't use simple/familiar distractors with technical truths. Match jargon level.
8. [ ] **Term familiarity balanced?** Don't pit famous terms against obscure ones. Use mirror pair analysis to detect.
9. [ ] **Status vs content separated?** (Math domains) Don't mix "what does X claim" with "is X proven/open" — status questions fail at 4% vs content at 71%.
10. [ ] **Try rescue strategy (truth-type dependent)?**
   - **Short numerical truths (≤5 tokens):** Use length-matched distractors. Anti-fluency creates false positives!
   - **Verbose/conceptual truths:** Use anti-fluency distractors (verbose/awkward).
   - If fact flips, model already knows — adapter is unnecessary.

**See:** `results/discoveries/novel_findings/` for detailed analysis of each mechanism.

The primary output is **tools served via MCP**, not adapters in weights.
Adapters are still useful within the discovery pipeline (each injection
makes the oracle smarter for subsequent candidates), but the permanent
artifact is always a tool.

After the main sweep, **Phase 2.5 (confidence-driven resampling)** retries
borderline failures (margin between -5 and 0) with the full adapter stack.
Neighboring discoveries often rescue borderline candidates. Survivors get
promoted to high-priority in the open questions queue.

**Diagnostic quadrants:**
| # | Oracle | Checker | Adapter Δ | Action |
|---|--------|---------|-----------|--------|
| 1 | PASS | PASS | — | Archive, add to verification set |
| 2 | FAIL | PASS | improves | Apply adapter, re-verify |
| 3 | FAIL | PASS | worsens | **Knowledge gap** — try staged training (see below) |
| 4 | — | FAIL | — | Discard |

**Staged training (for interference).** If single-pass adapter training makes
margins worse (Quadrant 3), don't give up. Group the facts into conceptual
clusters (e.g., symplectic structure, then Noether/Poisson, then energy/action,
then specific systems). Train each cluster sequentially, verifying zero
regression at each stage before moving to the next. This solved Hamiltonian
mechanics: single-pass went from 1/16 to 2/16 with worsening margins, staged
training reached 16/16 in 5 stages with zero regression throughout. The hardest
facts (KAM: -59.8 to +3.9, Henon-Heiles: -138.2 to +7.9) only flipped after
the foundational clusters were already consolidated.

**Orthogonal adapters (for staged training plateaus).** If staged training
plateaus (facts within a single adapter still interfere), train separate
specialist adapters per concept cluster. Each adapter learns one cluster without
fighting the others. Route each fact to its specialist at inference time. This
solved NS regularity: staged training stuck at 6/16, orthogonal
cluster adapters reached 16/16.

Why this is necessary: NS clusters are representational see-saws. Training on
blowup facts (2/2 within cluster) destroys conservation margins (to -600).
Training on conservation facts (2/2 within cluster) destroys blowup margins
(to -1100). Even training on a single new fact (ns11) causes regression on
previously passing facts (ns15). The concepts need to move in opposite
directions within logit space. A single adapter can only point one way.
Orthogonal adapters give each cluster its own direction, routed at inference
so they never compete for the same parameters.

**Established domain results: All 1038 facts flipped across 69 domains (100%).** 2 new domains (Drug Interactions, Information Theory) awaiting oracle run. See `results/candidates.tsv` for per-domain breakdown.

Notable methods used: staged anchored training (Hamiltonian, 5 stages), orthogonal adapters (most domains), main+orthogonal hybrid (Elliptic Curves, Intersection Theory). Deepest initial gap: Intersection Theory at -27.6. Hardest escalation: NS Regularity (representational see-saws requiring fully orthogonal cluster adapters).

**Escalation order for hard domains (every level has reached 16/16 on at least one domain):**
1. Single-pass adapter → if interference, try:
2. Staged training (sequential clusters) → solved Hamiltonian (16/16). If plateau, try:
3. Orthogonal adapters (specialist per cluster, routed at inference) → solved NS (16/16) and Knot invariants (16/16). Generalizes across physics and pure math. If still stuck, try:
4. Cross-domain joint training (train single adapter on multiple domains) → confirmed with difficulty-weighted sampling: NS 0→10/16, knots 1→11/16, chemical 0→13/16, Hamiltonian 1→14/16 from ONE adapter. Difficulty-weighted sampling (oversample hard facts) gives best transfer on hardest domain.
5. Hybrid routing (evaluate both joint and orthogonal adapters, pick higher margin per fact) → 82.1% on physics frontier (69/84) vs 70.2% orthogonal-only vs 44.0% joint-only. Joint wins on particle physics/neutrino/holographic; orthogonal wins on dark matter/quantum gravity/cosmology/condensed matter.
6. Persistent adapter router (embedding-based cascade) — loads router_state.npz at session start, routes each fact to the best adapter automatically. Cascade: high-confidence single (sim>0.85) → ambiguous try-both (gap<0.05) → vanilla fallback (sim<0.60). LRU-5 cache keeps ~580MB in memory. Cross-session persistent: each run starts smarter than the last.

**Automated discovery benchmark (stage_discovery guided mode):** 603/1043 (57.8%) across 77 domains. PL domains: 64/66 (97.0%), LLM domains: 87/88 (98.9%). The guided mode uses the meta-router for prioritization but falls back to non-routed adapters when routed ones don't improve enough. See `results/benchmark_all_domains_v3.json`.

**Negative result:** A single unified adapter trained on 244 heterogeneous toolkit facts (16 clusters from complexity theory to pharmacokinetics) scored 7.8% — worse than the 10.2% baseline. Joint training only works for semantically related domains.

**Negative result:** Base-trained adapters do NOT transfer to Instruct models. Tested across 6 domains: Instruct baseline (17.0%) is worse than Base baseline (20.5%) on oracle facts, and adapters that lift Base (e.g., Hamiltonian 1→16) have zero effect on Instruct (1→1). RLHF damages the representation space that adapters target. Use Base for the research oracle.

---

## Tool Development Pipeline

When building new tools from pipeline discoveries, follow this sequence.
The pre-commit hook enforces steps 3-5 automatically.

### Check Before Building (mandatory gate)

Before building ANY new tool, pass through this filter in order:

1. **Already in NoetherSolve?** Search existing 230+ tools. If the computation
   is already covered (even as a subset of a larger tool), don't build.
2. **Can extend an existing tool?** If a nearby tool exists (same module, same
   domain), add a parameter or mode to it instead of creating a new tool.
3. **Freeware exists?** Check scipy, sympy, astropy, biopython, plasmapy,
   python-control, qiskit, rdkit, obspy, etc. If a well-maintained library
   computes this with a one-liner, **do not reimplement** — BUT only wrap it
   if the dependency is already in `pyproject.toml` or is < 10 MB install.
   Heavy dependencies (SageMath 8 GB, Qiskit 400 MB) are worse than 30 lines
   of from-scratch code for MCP latency reasons.

**If NONE of the above apply → proceed to build.**

The zero-dependency philosophy is correct for MCP tools: every tool should
run with `math` + `dataclasses` + optionally `numpy`. Do not add scipy/sympy
as required dependencies. If a formula is under ~50 lines, implement from
scratch. Only wrap external libraries when the computation is genuinely
complex (e.g., ODE integration, symbolic simplification, molecular structure
parsing) AND the library is lightweight.

### Step-by-step

1. **Identify the tool opportunity.** A discovery from the main loop (new
   invariant, sensitivity result, thermodynamic check) suggests a standalone
   tool. Document what it does and why it's useful.

2. **Build the tool.** Write the module in `noethersolve/`. Follow existing
   patterns: dataclass reports, `__str__` for human-readable output, clear
   docstrings with usage examples. Export from `__init__.py`.

3. **Physics audit.** Before writing tests, review every formula line by line:
   - Is this quantity actually conserved, or approximately conserved, or dynamic?
   - Classify correctly: exact (H, Lz, E) vs approximate (Q_f family) vs
     dynamic (KE, PE, Lyapunov, entropy production, detailed balance ratios).
   - Never use finite-difference velocity estimation when analytical formulas
     exist (vortex velocities are determined by positions).
   - Wegscheider cyclicity only applies to closed cycles, not linear chains.
     Rate constant products are always constant but only thermodynamically
     constrained for cycles.
   - Check: would a physicist reviewing this code find an error?

4. **Write tests.** Cover: initialization, correct physics (tight tolerances
   should PASS), wrong physics detection, error handling, report formatting.
   Target: every public method has at least one test.

5. **Run full suite.** `pytest tests/ -v` — all must pass.

6. **Add to MCP server.** Add a `@mcp.tool()` function in
   `noethersolve/mcp_server/server.py` that wraps the module's public API.
   Use lazy imports (import inside the function body). Write a clear
   docstring — this is what the AI agent sees when deciding whether to
   call the tool.

7. **Document on the repo.** Add to README (Toolkit + MCP sections) with
   usage example. Add to CLAUDE.md key files table and the "always use
   tools first" list. Update version in `pyproject.toml` and
   `noethersolve/__init__.py`.

8. **Commit.** The pre-commit hook runs: tests → import check → physics smoke
   test (validates H and Lz conservation on a reference vortex problem).
   Commit is blocked if any step fails.

9. **Ship to PyPI.** `python -m build && twine upload dist/*`

10. **Add to paper.** Update the living preprint with the new tool, its
    benchmark results, and the discovery that motivated it.

### Pre-commit hook

The `.git/hooks/pre-commit` hook runs automatically on every commit:
- `pytest tests/` — all tests must pass
- Import check — all public exports must resolve
- Physics smoke test — H and Lz must be exactly conserved on reference problem

To bypass (NOT recommended): `git commit --no-verify`

---

## Fully Autonomous Run (Recommended)

Once a problem YAML exists, the full loop runs without human babysitting:

```bash
# Full autonomous run — sweeps, oracles, trains adapters, publishes
python autonomy_loop.py --problem problems/vortex_pair_conservation.yaml

# Also generate new hypotheses for the next session
python autonomy_loop.py --problem problems/vortex_pair_conservation.yaml \
    --generate-problems

# Dry run — just numerical sweep, no model load needed
python autonomy_loop.py --problem problems/vortex_pair_conservation.yaml --dry-run

# Oracle without training (no ANTHROPIC_API_KEY needed)
python autonomy_loop.py --problem problems/vortex_pair_conservation.yaml \
    --skip-training
```

The loop does everything: expand templates → check numerically → generate oracle
question via Claude API → run oracle → if fails, generate training data → train
adapter → re-evaluate → publish to candidates.tsv → generate new hypotheses.

**What `--generate-problems` does:**
After the main loop, calls Claude (adaptive thinking) to propose 8-12 new
expression hypotheses based on what was found. These go into
`results/open_questions.jsonl` and are automatically injected into the next run.

---

## Running an Experiment — Step by Step (Manual)

### Step 1: Run the numerical checker
```bash
# Figure-8 3-body
python conservation_checker.py --ic figure8 --expr "s['r12']+s['r13']+s['r23']"
python conservation_checker.py --all   # run all ICs and known candidates

# 2D point-vortex
python vortex_checker.py --ic restricted --expr "s['r12'] + 0.3*(s['r13']+s['r23'])"
python vortex_checker.py --all

# frac_var < 5e-3 → PASS (proceed to oracle)
# frac_var > 5e-3 → FAIL (discard, record in candidates.tsv)
```

### Step 2: Run the oracle (if checker passes)
```bash
# Apple Silicon (MLX)
python oracle_wrapper.py --problem problems/vortex_pair_conservation.yaml

# Linux/CUDA (PyTorch — no MLX needed)
python noethersolve_torch.py eval-oracle \
  --problem problems/vortex_pair_conservation.yaml --diagnose
```

### Step 3: If oracle fails, diagnose and repair
```bash
python oracle_wrapper.py --problem problems/vortex_pair_conservation.yaml \
    --repair --diagnose
# Prints quadrant: FIXABLE_BIAS (adapter helps) or KNOWLEDGE_GAP (need training data)
```

### Step 4: If knowledge gap, train a domain adapter
```bash
# Apple Silicon (MLX)
python training/scripts/train_vortex_adapter.py --data my_training_data.json --steps 1500

# Linux/CUDA (PyTorch)
python noethersolve_torch.py train-adapter \
  --data my_training_data.json \
  --model Qwen/Qwen3-4B-Base \
  --out adapters/my_adapter.npz
```

### Step 5: Publish results
Add a row to `results/candidates.tsv`. If DUAL-PASS or FLIPPED, add a discovery note to `results/discoveries/novel_findings/` (for new science) or `results/discoveries/model_specific/` (for adapter results).

---

## Open Questions Queue

The `results/open_questions.jsonl` file accumulates AI-generated hypotheses and
user-proposed research directions across sessions. It's how the system "remembers"
what to try next.

**Check it before starting:**
```bash
python autonomy_loop.py show-queue
```

**Two types of entries:**
- `type: expression` — specific Python expression ready to numerically check
- `type: direction` — broader research question needing a new problem YAML

**Entries are auto-injected** into the next `autonomy_loop.py` run for the
matching domain. You can also manually pick one and run it.

**Marking a question done:**  Edit `results/open_questions.jsonl` and set
`"status": "done"`. Or it's automatically closed when added to candidates.tsv.

---

## Finding Open Problems — Read the Live Sources

**Never rely on static lists in this file — they go stale.** Always query the live sources:

```bash
# What's already been tried? (closed holes — don't duplicate)
cat results/candidates.tsv

# What's in the open questions queue? (AI-generated + user proposals)
python autonomy_loop.py show-queue

# What's still open in each domain? (suggested next targets from domain experts)
grep -A 20 "Next interesting targets" problems/vortex_pair_conservation.yaml
grep -A 20 "Next interesting targets" problems/3body_conservation.yaml

# Full picture with charts
python dashboard.py --open
```

**Interpreting candidates.tsv to find open work:**
- `ORACLE-FAIL+CHECKER-PASS` → open gap, good target for adapter repair
- `QUADRANT3→FLIPPED` → closed, but suggests related expressions worth trying
- `CHECKER-FAIL` → dead end, skip entirely

**To propose a new domain entirely:**
Copy `problems/problem_template.yaml` and add three files: `my_domain.yaml` + `my_domain_facts.json` + `my_domain_checker.py`.

---

## What NOT to Do

- **Do not re-test already-closed hypotheses.** Check `candidates.tsv` first. Semantic near-duplicates count (r12+r13+r23 ≡ r13+r12+r23).
- **Do not use the mixed STEM adapter on vortex facts.** It makes vortex margins catastrophically worse (confirmed: -10.6 → -30.5). Use the domain-specific vortex adapter.
- **Do not use the choreography adapter on vortex problems** (wrong domain, cross-domain interference confirmed).
- **Do not stack or average trained adapters.** `multi_domain_v2` (averaged weights of vortex + H-Lz adapters) underperforms both specialists on every benchmark. NS blowup + conservation adapters destroy each other when merged (margins to -600/-1100). Stacking a specialist on top of a joint adapter also fails (8/16 → 5/16). Concepts that are representational see-saws must stay in separate orthogonal adapters, routed at inference. **Exception:** cross-domain joint training from scratch (difficulty-weighted sampling) DOES work — one adapter lifts H 14/16, NS 10/16, Knot 11/16, Chem 13/16. Blending from scratch ≠ stacking trained adapters.
- **Do not test equilateral triangle ICs as interesting.** Equilateral = relative equilibrium for ANY circulation values — all rᵢⱼ=const exactly. Trivially conserved, not interesting.
- **Do not use verbose prose in oracle facts.** Compact symbolic notation only: `"Q = r₁₂ + ε(r₁₃+r₂₃) = const"`. Verbose prose fails the oracle (confirmed in pilot runs).
- **Do not blame the adapter when a single fact won't flip.** Check the distractor first. If the distractor is too similar to the correct answer or shorter (e.g., `"k × [A]"` vs the full rate law), the model picks it on length/simplicity bias, not because it believes it's true. Fix the distractor to be clearly wrong and roughly the same length. This flipped the last chemical kinetics holdout from -1.4 to positive immediately.
- **Do not hardcode absolute paths** in any script. Use `os.path.dirname(__file__)` for relative resolution.
- **Do not stack adapters at inference.** Joint + specialist stacking destroys the joint adapter's wins (8/16 → 5/16). Use cluster routing (each fact → its specialist) or joint training from scratch (single blended adapter).
- **Do not ignore token-length bias in oracle facts.** If the truth is longer than the best distractor, the base model picks the shorter answer on length bias alone. Fix by shortening truth and lengthening distractors to similar lengths. This fixed chem08 (-3.8 → +4.3) and ns03 (-44 → +242.8 with adapter).
- **Do not create facts with length ratio > 1.5.** Length ratio = truth_length / min(distractor_lengths). Strong correlation with baseline accuracy (r = -0.742). Domains with ratio < 1.2 average 64% baseline; ratio > 2.5 averages 7% baseline. Length-balancing knot_invariants (ratio 7.81 → 1.16) improved pass rate from 0% to 25% without any adapter. See `results/discoveries/novel_findings/length_ratio_discovery.md`.
- **Do not use coherent distractors for adapter training.** Coherent distractors (plausible wrong answers) get high log-prob from fluency, not factual preference. For training adapters, use semantically incoherent distractors (nonsense completions) to isolate the truth signal. Balanced + coherent: 33% pass. Balanced + incoherent: **75% pass**. See `results/discoveries/novel_findings/distractor_coherence_discovery.md`.
- **Do not use mean scoring on domains with hedged truths.** Mean normalization amplifies per-token fluency differences. For domains where truths are hedged/technical (e.g., "are possible but thresholds remain uncertain") and distractors are confident (e.g., "have absolutely no effect"), use **sum scoring**. Mean scoring dropped climate_science_frontiers from 75% → 0% and black_hole_frontiers from 75% → 8%. Use mean scoring only for verbose/explanatory truths. See `results/discoveries/novel_findings/unified_oracle_difficulty_theory.md`.
- **Do not write distractors with definitive language when truth is hedged.** Models have **certainty contamination bias**: they prefer "completely ruled out" over "remains inconclusive" because training data over-represents definitive claims. Correlation r = -0.402 between certainty gap and oracle margin (t = 3.57, p < 0.01). Pass rate: 55% (gap=0) → 26% (gap=3). High-certainty distractors are LONGER (r = +0.277), so this is NOT length bias — the model is preferring longer, more definitive distractors over shorter, hedged truths. Fix by writing hedged distractors: "may be ruled out" instead of "completely ruled out". Rebalancing improved margins by +0.89 average. See `results/discoveries/novel_findings/certainty_contamination_bias.md`.

---

## Key Files

| File | What it does |
|------|-------------|
| `noethersolve/mcp_server/` | **MCP server — 230+ tools for any AI agent** |
| `noethersolve/paper_agent.py` | **Paper Agent** — autonomous paper generation + Zenodo upload |
| `paper/PAPER_PIPELINE.md` | 9-stage paper publication checklist |
| `conservation_checker.py` | Figure-8 3-body RK45 integrator + frac_var checker |
| `vortex_checker.py` | 2D point-vortex Kirchhoff integrator + frac_var checker |
| `oracle_wrapper.py` | Log-prob margin oracle + repair pass + quadrant diagnosis (MLX) |
| `noethersolve_torch.py` | Same as oracle_wrapper but PyTorch/CUDA — no MLX needed |
| `dashboard.py` | Regenerate results dashboard from candidates.tsv |
| `training/scripts/train_vortex_adapter.py` | Train vortex-specific logit adapter (MLX) |
| `training/scripts/train_choreography_adapter.py` | Train figure-8 choreography adapter (MLX) |
| `results/candidates.tsv` | **Results ledger** — all tested hypotheses and verdicts |
| `problems/*.yaml` | Domain plugin definitions |
| `problems/*_facts.json` | Oracle verification sets (8–15 facts per domain) |
| `adapters/` | Trained adapter weights (gitignored — local only) |
| `noethersolve/adapter_router.py` | **Persistent adapter router** — embedding cascade, LRU cache, save/load |
| `experiments/build_router.py` | Build + validate router from all facts/adapters |
| `experiments/test_instruct_transfer.py` | Base→Instruct adapter transfer test (negative result) |
| `router_state.npz` | Persisted router state (gitignored — built locally) |
| `noethersolve/monitor.py` | Conservation law monitors (Vortex, Chemical, Gravity) |
| `noethersolve/monitor_em.py` | EM field monitor (energy, chirality, helicity, zilch, super-energy) |
| `noethersolve/hamiltonian.py` | Hamiltonian validator (energy, Liouville volume, Poincare invariant) |
| `noethersolve/learner.py` | Invariant learner (L-BFGS-B over 12 basis functions) |
| `noethersolve/validate.py` | Integrator validation via conservation laws |
| `noethersolve/audit_chem.py` | Chemical network thermodynamic auditor |
| `experiments/corruption_benchmark.py` | 5 benchmark experiments proving monitor sensitivity |
| `experiments/train_toolkit_adapter.py` | Joint unified adapter training (244 facts, 16 clusters) |
| `experiments/train_frontier_domains.py` | Orthogonal adapter training for frontier domains |
| `experiments/verify_frontier_domains.py` | Verify frontier domain adapters (96/96 = 100%) |
| `experiments/train_missing_adapters.py` | Train 4B adapters for domains that only have 7B adapters (vocab compatibility) |
| `experiments/run_full_oracle_benchmark.py` | Run full oracle benchmark across all domains with stage discovery |
| `experiments/benchmark_toolkit_adapter.py` | Benchmark any adapter against toolkit facts |
| `results/discoveries/model_specific/adapter_combination_findings.md` | Hybrid routing findings (82.1% on physics frontier) |
| `results/discoveries/novel_findings/length_ratio_discovery.md` | **Length ratio discovery** — r=-0.742 correlation with baseline |
| `results/discoveries/novel_findings/distractor_coherence_discovery.md` | **Distractor coherence** — 33% → 75% with incoherent distractors |
| `results/discoveries/novel_findings/unified_oracle_difficulty_theory.md` | **Unified theory** — 3 mechanisms (length, coherence, scoring) |
| `experiments/analyze_distractor_patterns.py` | Analyze length ratios across all domains |
| `experiments/correlate_length_baseline.py` | Correlate length ratio with baseline accuracy |
| `results/discoveries/novel_findings/` | Novel scientific findings (Q_f family, Z₃ symmetry, EM invariants, etc.) |
| `research/knot_invariants.py` | Numerical verification of knot invariants |
| `research/hamiltonian_invariants.py` | Hamiltonian system invariant checks |
| `research/chemical_networks.py` | Chemical network conservation verification |
| `training/scripts/train_staged_adapter.py` | Staged sequential adapter training |
| `training/scripts/train_anchored_adapter.py` | Anchored training with regression protection |
| `training/scripts/train_prior_breaker.py` | Prior-breaking adapter training |
| `noethersolve/audit_facts.py` | Oracle fact file quality auditor (token-length bias detection) |
| `noethersolve/knot.py` | Knot invariant monitor (Reidemeister moves, Jones polynomial) |
| `noethersolve/audit_sequence.py` | DNA/RNA therapeutic sequence design auditor |
| `noethersolve/crispr.py` | CRISPR guide RNA scorer (on-target activity, off-target risk) |
| `noethersolve/pipeline.py` | Therapeutic pipeline consistency validator |
| `noethersolve/aggregation.py` | Protein aggregation propensity predictor |
| `noethersolve/splice.py` | Splice site strength scorer (PWM-based) |
| `noethersolve/enzyme_kinetics.py` | Michaelis-Menten, inhibition, cooperativity, pH rate profile calculator |
| `noethersolve/qm_calculator.py` | Particle-in-box, hydrogen, tunneling, uncertainty, oscillator calculator |
| `noethersolve/pk_model.py` | Pharmacokinetic compartmental modeling (IV, oral, steady state) |
| `noethersolve/reaction_engine.py` | Organic chemistry molecule analysis, selectivity, mechanisms |
| `noethersolve/complexity.py` | Complexity class relationship auditor |
| `noethersolve/conjecture_status.py` | Mathematical conjecture status checker (~63 conjectures) |
| `noethersolve/proof_barriers.py` | Proof technique barrier checker (10 barriers) |
| `noethersolve/number_theory.py` | Number theory conjecture numerical verifier |
| `noethersolve/reductions.py` | Computational reduction chain validator |
| `noethersolve/pde_regularity.py` | PDE regularity and Sobolev embedding checker |
| `noethersolve/dimension_physics.py` | Dimension-dependent physics (2D vs 3D Green's functions, cascades, etc.) |
| `noethersolve/tool_graph.py` | **Tool graph framework** — `@calculator` decorator, type-based chain discovery, execute_chain() |
| `noethersolve/meta_router.py` | **Meta-router** — learns optimal adapter chains from outcome data (Phase 1) |
| `noethersolve/stage_discovery.py` | **Stage discovery** — automatic adapter sequence finding via greedy/guided/beam (Phase 2). Guided mode uses router for prioritization + fallback to non-routed adapters |
| `noethersolve/outcome_logger.py` | **Outcome logger** — thread-safe logging of fact × adapter outcomes for training |
| `noethersolve/llm_claims.py` | LLM claims auditor (benchmark checker, scaling calculator, misconception DB) |
| `noethersolve/chemistry_calc.py` | Electrochemistry, acid-base, crystal field, semiconductor calculator |
| `noethersolve/crypto_calc.py` | Cryptographic security level, birthday bound, cipher mode analyzer |
| `noethersolve/finance_calc.py` | Black-Scholes, put-call parity, Nash equilibrium, time value calculator |
| `noethersolve/distributed_calc.py` | Quorum systems, Byzantine thresholds, vector clocks, consistency models |
| `noethersolve/network_calc.py` | Bandwidth-delay product, TCP throughput, subnetting, IP fragmentation |
| `noethersolve/os_calc.py` | Page tables, CPU scheduling, deadlock detection, TLB analysis |
| `tests/` | 2265 tests across 61 test files |

---

## Checker Output Interpretation

```
frac_var < 1e-6   → Near-exact conservation (e.g. H, Lz — fundamental laws)
frac_var < 5e-3   → PASS threshold — approximate invariant worth checking oracle
frac_var < 1e-2   → Borderline (record, may be IC-dependent)
frac_var > 1e-2   → FAIL — not conserved on this IC
```

## Oracle Output Interpretation

```
margin > +1.5     → Strong PASS — model confidently knows this
margin  0 to +1.5 → Weak PASS — model leans correct
margin -1 to 0    → Borderline — try adapter repair
margin < -5       → Strong FAIL — likely knowledge gap
margin < -20      → Extreme gap — domain-specific adapter required
```

---

## 27B Work — Adapter Training (Current Phase)

The 27B evaluation phase is **COMPLETE** (111 domains, 70 passing, Mar 2026). The 27B is now the **local compute workhorse** for training 4B adapters on failing domains.

**Current work:**
```bash
# Train 4B adapters on all failing domains (escalation ladder)
python scripts/adapter_trainer.py

# Train one domain and exit
python scripts/adapter_trainer.py --once

# Check which failing domains have/need adapters
python scripts/adapter_trainer.py --status
```

**14 unique domains still failing** — these need 4B adapter training:
- Hard physics/math: knot_invariants, NS regularity, Hamiltonian mechanics, intersection theory, chemical networks
- Vortex conservation: continuous Q_f, Q_f ratio, optimal f(r), kinetic K, EM zilch
- LLM domains: llm_hallucination_grounded
- Bio: bio-AI parallels

**Escalation ladder (all automated via adapter_trainer.py):**
1. Single-pass adapter → if interference:
2. Staged training (sequential clusters) → if plateau:
3. Orthogonal adapters (specialist per cluster, routed at inference) → if still stuck:
4. Cross-domain joint training (difficulty-weighted sampling)

### Evaluation (DONE — Do Not Restart)

Oracle evaluation ran via `scripts/research_runner.py`. Results in:
- `results/research_status.json` — all 111 domains evaluated
- `results/run_summary.json` — sweep history
- `results/escalations.jsonl` — all resolved

## V2 Fact File Campaign

V1 fact files had systematic length ratio bias: detailed truths (60-80 chars) with dismissive one-liner distractors (5-30 chars). Length ratio correlates r=-0.742 with baseline accuracy — the model picks shorter answers regardless of content.

V2 files fix this by matching distractor length to truth length (ratio 0.8-1.2x). Results: Information Theory 8%→92%, LLM Reasoning 25%→83%, Intersection Theory 0%→75%.

**To create a V2:**
1. Read the V1 fact file, check length ratios
2. Rewrite distractors to match truth length (plausible but wrong, same detail level)
3. Apply phrasing rules: remove hedging from truths, match certainty levels, use active voice
4. Save as `problems/{domain}_facts_v2.json`
5. Create `problems/{domain}_v2.yaml` pointing to the V2 facts file
6. The runner automatically picks up new V2 files on its next poll

---

## Hardware Notes

**Apple Silicon (M-series):**
- Use `oracle_wrapper.py` and `train_*_adapter.py` (MLX backend)
- MLX loads Qwen3-4B-Base in ~1.5s
- Do NOT use PyTorch MPS for training — deadlocks on backward passes

**Linux / NVIDIA GPU:**
- Use `noethersolve_torch.py` (PyTorch backend, no MLX dependency)
- `pip install torch transformers accelerate` then run normally
- CUDA auto-detected; falls back to CPU if no GPU

**Both backends** produce `.npz` adapter files with identical key names (`gate_proj.weight`, `up_proj.weight`, `down_proj.weight`) — adapters are cross-platform.

---

## Credits

- **Oracle infrastructure** built on STEM Truth Oracle (Paper 9, DOI: 10.5281/zenodo.19005729) and Snap-On Communication Modules (Paper 8, DOI: 10.5281/zenodo.18902616).
- **Noether's theorem** (Emmy Noether, 1915) — the reason any of this works.
