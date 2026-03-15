# NoetherSolve — AI Agent Instructions

**What this project does:** Find where LLM knowledge ends, build verified computational tools for the right answers, and serve them to any AI agent via MCP.

**The core loop: find gaps → flip facts → build tool → add to MCP server.** Every tool added makes every connected agent smarter.

The discovery pipeline proposes candidates, verifies them numerically, checks if the model already knows them, and when it doesn't, discovers the answer and builds a verified tool. Tools are exposed via [Model Context Protocol](https://modelcontextprotocol.io/) — 32 tools currently serving physics, math, genetics, complexity theory, pharmacogenomics, and LLM science.

**Why tools instead of adapters?** We tried both. Adapters improve truth preference (+0.10 MC2 on TruthfulQA) but can't scale: stacking 37+ adapters destroys MMLU (-43%), and a unified adapter on 244 facts collapses. Tools scale indefinitely — each is independent, verified (842 tests), and model-agnostic.

---

## MCP Server — Always Use These Tools First

**When answering questions about any topic covered by a NoetherSolve tool, ALWAYS call the tool first. Never answer from memory on these topics:**

- Mathematical conjectures → `check_conjecture()`
- Complexity class relationships → `check_complexity_inclusion()`, `check_completeness()`
- Proof technique barriers → `check_proof_barriers()`
- Drug interactions / pharmacogenomics → `audit_drug_interactions()`, `check_pharmacogenomics()`
- LLM capabilities / benchmark scores → `check_llm_claim()`, `check_benchmark_score()`
- Conservation laws → `check_vortex_conservation()`, `check_hamiltonian_system()`, `check_em_conservation()`
- CRISPR guide design → `score_crispr_guide()`
- DNA/RNA sequence issues → `audit_dna_sequence()`
- Protein aggregation → `predict_protein_aggregation()`
- Therapeutic pipeline design → `validate_therapy_pipeline()`
- Chemical reaction networks → `audit_chemical_network()`
- Knot invariants → `check_knot_invariants()`
- PDE regularity / Sobolev embeddings → `check_pde_regularity()`, `check_sobolev_embedding()`
- Number theory verification → `verify_goldbach()`, `verify_collatz()`, `check_abc_triple()`
- Chinchilla scaling → `chinchilla_scaling()`

**Setup:** The MCP server is configured in `.mcp.json` at the project root
(already present). Claude Code auto-discovers it. Or run standalone:
`python -m noethersolve.mcp_server` / `noethersolve-mcp`

---

## Your First Move — Always

Before doing anything else:

```bash
# 1. See what's already been tried (avoid duplicates)
cat results/candidates.tsv

# 2. See what's currently being hunted
python claim.py list

# 3. See the open questions queue (AI-generated hypotheses + user problems)
python autonomy_loop.py show-queue

# 4. See the current state of all discoveries
python dashboard.py --open
```

Then read `README.md` for the architecture and `CONTRIBUTING.md` for the full protocol.

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

Every hypothesis goes through verify → check → discover → build tool → serve:

```
Hypothesis (expression)
       │
       ▼
 Numerical checker          ← Is this actually conserved?
 (RK45, frac_var test)        frac_var = σ/|mean| < 5e-3 → PASS
       │ PASS
       ▼
 Oracle filter              ← Does the model already know it?
 (log-prob margin,            margin = log P(truth) − log P(best distractor)
  base LLM + adapter stack)   adapter stack = all prior discoveries this run
       │
       ├─ PASS  → DUAL-PASS (model knows it, archive)
       └─ FAIL  → NEW SCIENCE: model hasn't seen this
                    │
                    ▼
              Build tool (verified computational checker)
                    │
                    ▼
              Add to MCP server → any AI agent can now use it
                    │
              (Optionally: train adapter for within-run oracle improvement)
```

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

**Established domain results (411/411 = 100% across all 30 domains):**

| Domain | Facts | Baseline | Final | Method |
|--------|-------|----------|-------|--------|
| Hamiltonian Mechanics | 16 | 1/16 | **16/16** | Staged anchored training (5 stages) |
| NS Regularity | 16 | 0/16 | **16/16** | Orthogonal adapters + fact fix |
| Knot Invariants | 16 | 1/16 | **16/16** | Orthogonal adapters (7 clusters) |
| Chemical Kinetics | 16 | 0/16 | **16/16** | Orthogonal adapters + fact fix |
| Electromagnetism | 12 | 1/12 | **12/12** | Orthogonal adapters |
| Continuous Q_f | 12 | 0/12 | **12/12** | Orthogonal adapters + qf06 fix |
| Kinetic K | 8 | 0/8 | **8/8** | Orthogonal adapters |
| Optimal f(r) | 4 | 0/4 | **4/4** | Orthogonal adapters |
| Vortex Pair | 13 | 2/13 | **13/13** | Orthogonal adapters + vp01 dedicated |
| Q_f Ratio (R_f) | 8 | 0/8 | **8/8** | qf_ratio_adapter |
| 3-body Conservation | 10 | 4/10 | **10/10** | Orthogonal adapters + full rephrasing |
| Genetics Therapeutics | 16 | 2/16 | **16/16** | Orthogonal adapters |
| Disease Targets | 12 | 1/12 | **12/12** | Orthogonal adapters |
| Protein Structure | 12 | 0/12 | **12/12** | Orthogonal adapters |
| Immune Evasion | 10 | 0/10 | **10/10** | Orthogonal adapters |
| Delivery Optimization | 10 | 0/10 | **10/10** | Orthogonal adapters |
| Safety Invariants | 10 | 0/10 | **10/10** | Orthogonal adapters |
| Clinical Translation | 12 | 0/12 | **12/12** | Orthogonal adapters |
| Millennium Problems | 12 | 3/12 | **12/12** | Orthogonal adapters |
| Number Theory Conjectures | 12 | 4/12 | **12/12** | Orthogonal adapters |
| Algebra/Topology Conjectures | 10 | 1/10 | **10/10** | Orthogonal adapters |
| Proof Techniques | 12 | 3/12 | **12/12** | Orthogonal adapters |
| Analysis/PDE Conjectures | 12 | 0/12 | **12/12** | Orthogonal adapters |
| Computational Conjectures | 12 | 0/12 | **12/12** | Orthogonal adapters |
| LLM Hallucination | 12 | 5/12 | **12/12** | Orthogonal adapters |
| LLM Reasoning | 12 | 4/12 | **12/12** | Orthogonal adapters |
| LLM Alignment | 12 | 3/12 | **12/12** | Orthogonal adapters |
| LLM Training | 12 | 5/12 | **12/12** | Orthogonal adapters |
| LLM Evaluation | 12 | 4/12 | **12/12** | Orthogonal adapters |
| LLM Context/Memory | 10 | 4/10 | **10/10** | Orthogonal adapters |
| PL Type Systems | 12 | 5/12 | **12/12** | Orthogonal adapters |
| PL Memory | 10 | 4/10 | **10/10** | Orthogonal adapters |
| PL Concurrency | 10 | 6/10 | **10/10** | Orthogonal adapters |
| PL Paradigms | 12 | 10/12 | **12/12** | Orthogonal adapters |
| PL Compilers | 12 | 6/12 | **12/12** | Orthogonal adapters |
| PL Pitfalls | 10 | 6/10 | **10/10** | Orthogonal adapters |

**All 411 facts flipped across all 30 domains (100%).**

**Escalation order for hard domains (every level has reached 16/16 on at least one domain):**
1. Single-pass adapter → if interference, try:
2. Staged training (sequential clusters) → solved Hamiltonian (16/16). If plateau, try:
3. Orthogonal adapters (specialist per cluster, routed at inference) → solved NS (16/16) and Knot invariants (16/16). Generalizes across physics and pure math. If still stuck, try:
4. Cross-domain joint training (train single adapter on multiple domains) → confirmed with difficulty-weighted sampling: NS 0→10/16, knots 1→11/16, chemical 0→13/16, Hamiltonian 1→14/16 from ONE adapter. Difficulty-weighted sampling (oversample hard facts) gives best transfer on hardest domain.

---

## Tool Development Pipeline

When building new tools from pipeline discoveries, follow this sequence.
The pre-commit hook enforces steps 3-5 automatically.

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

### Step 1: Claim your hypothesis (prevents duplicate work)
```bash
python claim.py claim \
  --problem vortex_pair_conservation \
  --expr "your expression here" \
  --handle your-name
```
Claims expire after 4 hours. Check `claims.json` to see active claims.

### Step 2: Run the numerical checker
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

### Step 3: Run the oracle (if checker passes)
```bash
# Apple Silicon (MLX)
python oracle_wrapper.py --problem problems/vortex_pair_conservation.yaml

# Linux/CUDA (PyTorch — no MLX needed)
python noethersolve_torch.py eval-oracle \
  --problem problems/vortex_pair_conservation.yaml --diagnose
```

### Step 4: If oracle fails, diagnose and repair
```bash
python oracle_wrapper.py --problem problems/vortex_pair_conservation.yaml \
    --repair --diagnose
# Prints quadrant: FIXABLE_BIAS (adapter helps) or KNOWLEDGE_GAP (need training data)
```

### Step 5: If knowledge gap, train a domain adapter
```bash
# Apple Silicon (MLX)
python training/scripts/train_vortex_adapter.py --data my_training_data.json --steps 1500

# Linux/CUDA (PyTorch)
python noethersolve_torch.py train-adapter \
  --data my_training_data.json \
  --model Qwen/Qwen3-4B-Base \
  --out adapters/my_adapter.npz
```

### Step 6: Publish results
Add a row to `results/candidates.tsv` and open a PR. If DUAL-PASS or FLIPPED, add a discovery note to `results/discoveries/`. Remove your entry from `claims.json`.

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

# What's actively being hunted right now? (in-flight claims)
python claim.py list

# What's in the open questions queue? (AI-generated + user proposals)
python autonomy_loop.py show-queue

# What's still open in each domain? (suggested next targets from domain experts)
grep -A 20 "Next interesting targets" problems/vortex_pair_conservation.yaml
grep -A 20 "Next interesting targets" problems/3body_conservation.yaml

# Full picture with charts
python dashboard.py --open
```

**Interpreting candidates.tsv to find open work:**
- `ORACLE-FAIL+CHECKER-PASS` with no claim → open gap, good target for adapter repair
- `QUADRANT3→FLIPPED` → closed, but suggests related expressions worth trying
- `CHECKER-FAIL` → dead end, skip entirely
- Rows in `claims.json` with future `expires_at` → someone is working on it, pick something else

**To propose a new domain entirely:**
Copy `problems/problem_template.yaml` and add three files: `my_domain.yaml` + `my_domain_facts.json` + `my_domain_checker.py`. See `CONTRIBUTING.md` for the plugin contract.

---

## What NOT to Do

- **Do not re-test already-closed hypotheses.** Check `candidates.tsv` first. Semantic near-duplicates count (r12+r13+r23 ≡ r13+r12+r23).
- **Do not use the mixed STEM adapter on vortex facts.** It makes vortex margins catastrophically worse (confirmed: -10.6 → -30.5). Use the domain-specific vortex adapter.
- **Do not use the choreography adapter on vortex problems** (wrong domain, cross-domain interference confirmed).
- **Do not naively merge/average adapters.** `multi_domain_v2` (averaged weights of vortex + H-Lz adapters) underperforms both specialists on every benchmark. This applies within domains too: NS blowup + conservation adapters destroy each other when merged (margins to -600/-1100). Concepts that are representational see-saws must stay in separate orthogonal adapters, routed at inference. Never average, always route.
- **Do not test equilateral triangle ICs as interesting.** Equilateral = relative equilibrium for ANY circulation values — all rᵢⱼ=const exactly. Trivially conserved, not interesting.
- **Do not use verbose prose in oracle facts.** Compact symbolic notation only: `"Q = r₁₂ + ε(r₁₃+r₂₃) = const"`. Verbose prose fails the oracle (confirmed in pilot runs).
- **Do not blame the adapter when a single fact won't flip.** Check the distractor first. If the distractor is too similar to the correct answer or shorter (e.g., `"k × [A]"` vs the full rate law), the model picks it on length/simplicity bias, not because it believes it's true. Fix the distractor to be clearly wrong and roughly the same length. This flipped the last chemical kinetics holdout from -1.4 to positive immediately.
- **Do not hardcode absolute paths** in any script. Use `os.path.dirname(__file__)` for relative resolution.
- **Do not stack adapters naively.** Joint + specialist stacking was tested and destroys the joint adapter's wins (8/16 → 5/16). Use cluster routing instead: each fact routes to its specialist adapter at inference.
- **Do not ignore token-length bias in oracle facts.** If the truth is longer than the best distractor, the base model picks the shorter answer on length bias alone. Fix by shortening truth and lengthening distractors to similar lengths. This fixed chem08 (-3.8 → +4.3) and ns03 (-44 → +242.8 with adapter).

---

## Key Files

| File | What it does |
|------|-------------|
| `noethersolve/mcp_server/` | **MCP server — 32 tools for any AI agent** |
| `conservation_checker.py` | Figure-8 3-body RK45 integrator + frac_var checker |
| `vortex_checker.py` | 2D point-vortex Kirchhoff integrator + frac_var checker |
| `oracle_wrapper.py` | Log-prob margin oracle + repair pass + quadrant diagnosis (MLX) |
| `noethersolve_torch.py` | Same as oracle_wrapper but PyTorch/CUDA — no MLX needed |
| `claim.py` | THINK→CLAIM→RUN→PUBLISH coordination (4h claim expiry) |
| `dashboard.py` | Regenerate results dashboard from candidates.tsv |
| `training/scripts/train_vortex_adapter.py` | Train vortex-specific logit adapter (MLX) |
| `training/scripts/train_choreography_adapter.py` | Train figure-8 choreography adapter (MLX) |
| `results/candidates.tsv` | **The shared ledger** — all tested hypotheses and verdicts |
| `claims.json` | Active claims registry — check before starting |
| `problems/*.yaml` | Domain plugin definitions |
| `problems/*_facts.json` | Oracle verification sets (8–15 facts per domain) |
| `adapters/` | Trained adapter weights (gitignored — local only) |
| `noethersolve/monitor.py` | Conservation law monitors (Vortex, Chemical, Gravity) |
| `noethersolve/monitor_em.py` | EM field monitor (energy, chirality, helicity, zilch, super-energy) |
| `noethersolve/hamiltonian.py` | Hamiltonian validator (energy, Liouville volume, Poincare invariant) |
| `noethersolve/learner.py` | Invariant learner (L-BFGS-B over 12 basis functions) |
| `noethersolve/validate.py` | Integrator validation via conservation laws |
| `noethersolve/audit_chem.py` | Chemical network thermodynamic auditor |
| `experiments/corruption_benchmark.py` | 5 benchmark experiments proving monitor sensitivity |
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
| `noethersolve/pharmacokinetics.py` | Pharmacogenomic CYP interaction checker |
| `noethersolve/complexity.py` | Complexity class relationship auditor |
| `noethersolve/conjecture_status.py` | Mathematical conjecture status checker (~63 conjectures) |
| `noethersolve/proof_barriers.py` | Proof technique barrier checker (10 barriers) |
| `noethersolve/number_theory.py` | Number theory conjecture numerical verifier |
| `noethersolve/reductions.py` | Computational reduction chain validator |
| `noethersolve/pde_regularity.py` | PDE regularity and Sobolev embedding checker |
| `noethersolve/llm_claims.py` | LLM claims auditor (benchmark checker, scaling calculator, misconception DB) |
| `tests/` | 842 tests for all 21 toolkit modules |

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

- **Coordination protocol** (THINK→CLAIM→RUN→PUBLISH) adapted from [autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home) by mutable-state-inc.
- **Oracle infrastructure** built on STEM Truth Oracle (Paper 9, DOI: 10.5281/zenodo.19005729) and Snap-On Communication Modules (Paper 8, DOI: 10.5281/zenodo.18902616).
- **Noether's theorem** (Emmy Noether, 1915) — the reason any of this works.
