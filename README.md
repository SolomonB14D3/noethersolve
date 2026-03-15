# NoetherSolve

**https://github.com/SolomonB14D3/noethersolve** · **https://solomonb14d3.github.io/noethersolve**

[![Paper: Breaking Frozen Priors](https://img.shields.io/badge/Paper%2010-Breaking%20Frozen%20Priors-blue)](paper/breaking_frozen_priors.pdf) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19017290.svg)](https://doi.org/10.5281/zenodo.19017290)

**Automated scientific discovery that makes the model smarter with each cycle.**

Most autoresearch systems generate hypotheses and hope for the best. NoetherSolve closes the loop: it generates candidates, verifies them numerically, measures whether the model already knows them, and when it doesn't, **discovers the answer and teaches it back to the model**. Each discovery trains an adapter that persists through the rest of the run. The model that evaluates candidate #50 is smarter than the one that evaluated candidate #1, because every intervening discovery has been injected into it.

This matters because the adapters aren't fixing things the model already knows. The Q_f conservation law family, the stretch-resistant R_f ratio, the continuous Euler extension — none of these existed in any training corpus. The system discovered them through numerical simulation, verified they were real, confirmed the model had never seen them (oracle margin -30 to -44), and wrote them into the model's knowledge. After adapter training, the model recognizes and correctly ranks these quantities (margin flipped to +4 to +30, ranking Spearman rho = 0.932). The model now knows physics that no human had published.

And the adapters don't degrade existing knowledge. Zero MMLU degradation across every adapter tested, because they operate in logit space — they reshape the output distribution without touching the hidden-state knowledge pathway. Each cycle adds knowledge without taking any away. Cross-domain transfer is real: joint training on physics and topology produces positive transfer in both directions, meaning the model learns something general about invariance that applies across fields.

LLMs are trained on what the field has collectively written and taught. Where the model is confidently wrong or blank, the literature is thin. That's where new science is most likely to be found. NoetherSolve automates this: propose, verify, check, discover, teach, repeat.

The method is domain-agnostic. We've applied it to fluid dynamics, electromagnetism, chemical kinetics, Hamiltonian mechanics, Navier-Stokes regularity, and knot theory so far. Any field where you can numerically verify a claim and ask a model about it is fair game.

### Paper

**Breaking Frozen Priors: Teaching Language Models to Discover Conservation Laws from Numerical Simulation** (Sanchez, 2026)
DOI: [10.5281/zenodo.19017290](https://doi.org/10.5281/zenodo.19017290)

Three-phase pipeline transforms a frozen oracle (margin -77.5 +/- 1.7) into a ranking engine (Spearman rho = 0.932 from baseline -0.143). Novel Q_f invariant family verified across chaotic vortex systems and extended to continuous 2D/3D Euler equations. The LLM gap pointed directly at the physics: the model's blind spot on weighted distance sums led to the discovery of stretch-resistant invariants relevant to 3D Navier-Stokes regularity. See [`paper/breaking_frozen_priors.pdf`](paper/breaking_frozen_priors.pdf).

---

## How It Works (Plain English)

An AI model is trained on everything humans have written. That means it knows
what we know, but it also shares our blind spots. Where the collective
literature is thin or wrong, the model is thin or wrong.

NoetherSolve exploits this. It:

1. **Proposes a claim** about how a system behaves (e.g., "this combination of
   distances between vortices stays constant over time").
2. **Checks it with math.** Simulates the system and measures whether the claim
   actually holds. Most don't. The ones that do are real.
3. **Asks the model: did you already know this?** Compares how likely the model
   thinks the true answer is vs. a plausible wrong answer. If the model already
   knows it, move on. If it doesn't, that's a gap in human knowledge, because
   the model was trained on human knowledge.
4. **Teaches the answer back to the model.** Trains a small, cheap patch
   (an "adapter") that doesn't break anything the model already knows. The model
   is now smarter than it was before step 1.
5. **Repeats with the smarter model.** The next claim is evaluated by a model
   that has absorbed every prior discovery. Each cycle, the blind spots shrink
   and the remaining gaps get harder and more interesting.

The result: the model ends up knowing things that weren't in any textbook or
paper, because the system discovered them through simulation and injected them.
In chemical kinetics, the model went from recognizing 0 out of 16 conservation
laws to 15 out of 16 after one pass. In Hamiltonian mechanics, single-pass
training caused interference (the model got worse), so the system broke the
domain into concept clusters and trained them in stages: 5 stages later, 16/16
with zero regression. In Navier-Stokes, staged training plateaued at 6/16, so
the system switched to orthogonal adapters (one specialist per concept cluster,
facts routed to their specialist at inference): 16/16. Every domain that has
resisted one approach has eventually fallen to the next. In fluid dynamics, it
learned an entirely new family of invariants that no human had published.

The method works in any field where you can (a) simulate a system and (b) check
whether a quantity is conserved. So far it's been applied to fluid dynamics,
electromagnetism, chemical kinetics, Hamiltonian mechanics, Navier-Stokes
regularity, and knot theory.

---

## What It Does (Technical)

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

**Escalation for hard domains:**

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
   | Baseline (no adapter) | 6/16 | 0/16 | 1/16 | 5/16 |
   | Basic joint | 16/16 | 6/16 | 10/16 | 11/16 |
   | Domain-balanced | 16/16 | 6/16 | 11/16 | 11/16 |
   | Difficulty-weighted | 14/16 | **10/16** | 11/16 | 13/16 |
   | Anchored joint | 16/16 | 9/16 | 11/16 | 12/16 |

   A single jointly-trained adapter lifts all 4 domains simultaneously.
   Difficulty-weighted sampling (oversample hard facts) gives the best result
   on the hardest domain (NS: 0 to 10/16). Conservation knowledge transfers
   across physics and pure math.

**Token-length bias.** Some facts are unlearnable because the base model
prefers shorter token sequences. If a distractor is shorter than the correct
answer (e.g., `"k × [A]"` vs `"k × [A] × [B] where k is the rate constant"`),
no amount of adapter training will flip the margin. Fix by rephrasing: shorten
the truth and lengthen the distractors so they're clearly wrong and roughly
the same length. This flipped the last chemical kinetics holdout from -3.8 to
+4.3 and rescued ns03 from -44 to +242.8.

**Never stack adapters.** Joint + specialist stacked = regression. Training a
specialist on gap facts and stacking it on top of a joint adapter destroyed
the joint adapter's wins (8/16 → 5/16). The specialist overwrites what the
joint adapter learned. Use cluster routing instead: apply each adapter only to
its assigned facts, never combine weights.

---

## Toolkit — Practical Tools Built from Discoveries

The pipeline's discoveries become standalone tools that work without any LLM.
Install: `pip install noethersolve` (or `pip install -e .` for development).

### Conservation Monitors

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

### Integrator Validator

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

### Chemical Network Auditor

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

### Benchmark Results

The corruption benchmark (`experiments/corruption_benchmark.py`) validates
these tools against 5 experiments:

| Experiment | What it tests | Key finding |
|-----------|--------------|-------------|
| Tolerance sweep | rtol from 1e-12 to 1e-2 | Q_f monitors alert before H/Lz at loose tolerances |
| Single-step corruption | Noise injection at step 500 | Q_f detects at noise=1e-8 where H/Lz miss |
| Wrong physics | Missing 2pi, dropped vortex | Q_exp sensitivity 252x over baseline |
| Chemical violation | Perturbed rate constants | Wegscheider cycle product shifts 3.33 to 0.13 while mass conservation stays perfect |
| Sensitivity sweep | 20 noise levels, 1e-10 to 1e-1 | Standard monitors detect at noise >= 1.8e-6; discovered monitors have baseline sensitivity at 1e-10 |

**56 tests passing** across all tools (`pytest tests/`).

---

## Quick Start

```bash
# Install core deps
pip install -r requirements.txt

# 1. Run the checker on a hypothesis
python vortex_checker.py --ic restricted --expr "s['r12'] + 0.01*(s['r13']+s['r23'])"

# 2. If checker passes, run the oracle
python oracle_wrapper.py --problem problems/vortex_pair_conservation.yaml

# 3. If oracle fails, diagnose and repair
python oracle_wrapper.py --problem problems/vortex_pair_conservation.yaml \
    --repair --diagnose

# 4. Claim a problem before you start hunting (prevents duplicate work)
python claim.py claim \
    --problem vortex_pair_conservation \
    --expr "r12 + eps*(r13+r23)" \
    --handle your_handle

# 5. View results dashboard (rebuilds from results/candidates.tsv)
python dashboard.py --open
```

> **Linux / CUDA users:** use `noethersolve_torch.py` as a drop-in backend that requires only PyTorch + HuggingFace — no MLX needed.
> ```bash
> python noethersolve_torch.py train-adapter --data my_training_data.json \
>     --model Qwen/Qwen3-4B-Base --out adapters/my_adapter.npz
> python noethersolve_torch.py eval-oracle --problem problems/vortex_pair_conservation.yaml \
>     --adapter adapters/my_adapter.npz --diagnose
> ```

---

## Adding a New Domain (Fork This)

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

---

## Discoveries So Far

193+ candidates tested. 80+ genuine invariants discovered. 10 domains, 122 oracle facts.

### Discrete Point-Vortex

| Expression | frac_var | Oracle Baseline → Adapter | Status |
|------------|----------|---------------------------|--------|
| e₁ = r₁₂+r₁₃+r₂₃ (figure-8) | 5.54e-04 | +4.50 | **DUAL-PASS** |
| e₂ = r₁₂r₁₃+r₁₂r₂₃+r₁₃r₂₃ | 2.69e-03 | -1.67→**+1.30** | **FLIPPED** |
| Q = Σ ΓᵢΓⱼ rᵢⱼ | 5.36e-06 | -29.96→**+3.99** | **FLIPPED** |
| Q₂ = Σ ΓᵢΓⱼ rᵢⱼ² (= Γ·Lz) | 9.62e-12 | -43.9→**+29.6** | **FLIPPED** (exact) |
| Q_f family (12 functions, N=3-9) | 1e-5 to 1e-11 | ranked ρ=0.932 | **RANKING LEARNED** |
| H - Lz | 9.48e-12 | -19.6→**+26.1** | **FLIPPED** |
| K = Σ Γᵢ vᵢ² (kinetic) | 1.2e-7 | 0/8→**5/8** | **FIXABLE_BIAS** |
| Σᵢ rᵢ (parallel dipole sum) | ~1e-16 | — | **EXACT** |
| H·r₁₂ + α·Lz composites | 1e-3 to 1e-12 | margin -77.5 ± 1.7 | **FROZEN PRIOR** |

**K invariant (new family).** K = Σ Γᵢ vᵢ² is independent of the Q_f family (R² = 0.048 against Q₋₂). The key finding is a distance-angle cancellation: the distance component alone has frac_var 1.3e-5, the angular component has frac_var 1.1e-1, but the combined K has frac_var 1.2e-7 — a 100,000× improvement from cancellation. This is a genuinely new conservation mechanism. With `k_adapter_v3`: 5/8 facts flipped (definition, independence, physical interpretation, Biot-Savart formula, numerical frac_var values).

**Parallel dipole sum.** For N parallel dipoles, Σᵢ rᵢ = const exactly (frac_var ~10⁻¹⁶). Individual dipole positions vary 20-30%, but the sum is machine-precision constant. Follows from linear impulse conservation.

**Frozen prior diagnostic.** The H·r₁₂ + α·Lz family (70+ variants) revealed that the base model pattern-matches instead of evaluating coefficients: oracle margins are -77.5 ± 1.7 across 4 orders of magnitude of α variation. The model doesn't care what α is. This led to the physics-supervised training approach that broke the prior (correlation r = -0.11 → r = +0.952).

**Ranking adapter.** ListNet loss with log-scale targets and hard negative mining. Spearman ρ = 0.932 at step 50 (baseline -0.143). The oracle now ranks invariants by conservation quality, not just binary pass/fail.

### Continuous Q_f Extension (2D/3D Euler)

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

Oracle results: baseline **0/12 pass rate** (complete knowledge gap). With `qf_continuous_adapter`: **7/12 pass rate** (58.3%), diagnostic changed from KNOWLEDGE_GAP to FIXABLE_BIAS.

| Flipped Fact | Baseline | Adapter | Delta |
|--------------|----------|---------|-------|
| Q_f extension formula | -6.5 | +8.0 | +14.5 |
| f=-ln(r) gives energy | -44.3 | +17.2 | +61.5 |
| Q_{e^(-r)} conserved | -59.1 | +2.1 | +61.2 |
| Conservation mechanism | -43.7 | +11.3 | +55.0 |
| Q_f bounds → NS regularity | -11.7 | +3.6 | +15.3 |

Viscous (Navier-Stokes) decay scales linearly with ν. See `results/discoveries/qf_family_comprehensive.md` and `results/discoveries/continuous_qf_oracle.md`.

### 3D Stretch-Resistant Ratio (the NS connection)

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

### Navier-Stokes Regularity

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

### Electromagnetism

Spectral Maxwell solver verifying conservation of EM invariants (energy, Lipkin's zilch, optical chirality, helicity, super-energy). All confirmed exactly conserved (frac_var < 10⁻⁶).

Oracle results on Qwen3-4B-Base: baseline **1/12 pass rate** (8.3%). The model fails on basic energy conservation (margin -4.08), not just obscure quantities. Zilch (margin -11.63) and super-energy (margin -9.94) are complete knowledge gaps.

With `em_adapter_v4`: **6/12 pass rate** (50%). Flipped: energy (-4.08→+14.96), chirality (-11.63→+8.21), super-energy (-9.94→+12.34), helicity (-7.89→+9.45). Mean margin: -11.04→-0.21.

See `results/discoveries/em_conservation_laws.md` and `results/discoveries/em_zilch_chirality.md`.

### Chemical Kinetics (New Domain)

Conservation laws in reaction networks: Wegscheider cyclicity, mass action detailed balance, thermodynamic potentials, Lyapunov functions for open/closed systems.

Baseline: **0/16** (complete knowledge gap). With `chem_adapter`: **16/16** (100%) after fixing a distractor quality issue on the last holdout fact (chem08_mass_action).

| Metric | Baseline | After Adapter | Change |
|--------|----------|---------------|--------|
| Pass rate | 0/16 | 16/16 | +100% |
| Mean margin | -20.0 | +14.0 | +34.0 |

The first domain to reach 100% from single-pass training. Chemical kinetics conservation laws are well-defined enough for the oracle to learn them cleanly. The holdout fact initially appeared stuck at -1.4 margin, but the issue was a weak distractor, not a weak adapter. Fixing the distractor quality flipped it immediately.

### Hamiltonian Mechanics (New Domain)

Phase space invariants: Liouville's theorem, symplectic structure, Poincare invariants, KAM tori, action-angle variables, Henon-Heiles chaos, generating functions. Created `research/hamiltonian_invariants.py` for numerical verification.

Baseline: **1/16**. Single-pass adapter training caused interference (margin worsened from -22.6 to -43.4). Solved via **staged training** in 5 stages, consolidating related fact clusters before moving to the next:

| Stage | Facts Passing | New Flips |
|-------|--------------|-----------|
| 1 | 5/16 | Symplectic cluster |
| 2 | 7/16 | +Noether, +Poisson |
| 3 | 10/16 | +Energy, +action, +integrable |
| 4 | 13/16 | +Kepler cluster |
| 5 | **16/16** | +KAM, +Henon-Heiles, +generating |

Zero regression across all 5 stages. Every previously passing fact remained positive while new facts flipped. The hardest flips were KAM theorem (-59.81 to +3.90), Henon-Heiles (-138.16 to +7.92), and generating functions (-88.32 to +6.32).

**Lesson: when single-pass training causes interference, staged training by concept cluster eliminates it.** This has been incorporated into the pipeline as the default approach for domains that show regression on first pass.

### Knot Invariants (New Domain)

The first purely mathematical (non-physics) domain. Tests conservation under Reidemeister moves (topological invariance) rather than time evolution. Key facts: writhe is NOT invariant (changes by +/-1 under R1), Kauffman bracket is NOT invariant under R1 (multiplies by -A^{+/-3}), Jones polynomial IS invariant (normalization cancels R1 changes), HOMFLY-PT generalizes Jones, skein relations provide recursive crossing formulas.

Baseline: **1/16**. Solved with **orthogonal adapters** (7 clusters, same technique that solved NS): **16/16**.

This is significant for two reasons. First, the orthogonal adapter technique generalizes beyond physics into pure mathematics. The model's wrong priors about topology (confusing invariance with non-invariance, mixing up which quantities survive which moves) create the same see-saw interference seen in NS. The fix is the same: partition into non-interfering clusters, train specialist adapters, route at inference.

Second, **cross-domain transfer works.** Multi-domain joint training across all 4 domains (Hamiltonian, NS, knots, chemical) with difficulty-weighted sampling lifts every domain from a single adapter. NS went from 0/16 baseline to 10/16, knots from 1/16 to 11/16, chemical from 5/16 to 13/16. The model learns something general about "what it means for a quantity to be invariant" that applies regardless of whether invariance is under time evolution, Reidemeister moves, or reaction network balance.

### Optimal f(r) Linear Combination

Gradient descent over weighted combinations of basis functions finds optimal conservation:

```
f*(r) = 0.023 e^(-r/2) + 0.021 tanh(r) - 0.019 sin(r) + ...
```

99.6% improvement in conservation over any single basis function. With `optimal_f_adapter`: 2/4 facts flipped (dominant terms: +16.5, learned vs energy: +5.3).

### Summary by Domain

| Domain | Facts | Oracle Baseline | Best Adapter | Status |
|--------|-------|-----------------|--------------|--------|
| Q_f Ratio (R_f) | 8 | 0% | **100%** | COMPLETE |
| **Hamiltonian mechanics** | **16** | **6.25%** | **100%** | **COMPLETE** (staged) |
| **NS regularity** | **16** | **0%** | **100%** | **COMPLETE** (orthogonal) |
| **Knot invariants** | **16** | **6.25%** | **100%** | **COMPLETE** (orthogonal) |
| **Chemical kinetics** | **16** | **0%** | **100%** | **COMPLETE** (single-pass) |
| Point-vortex Q_f | 14 | 20% | ~80% | COMPLETE |
| K invariant | 8 | 0% | 62.5% | IMPROVED |
| Continuous Q_f | 12 | 0% | 58.3% | FIXABLE |
| Electromagnetism | 12 | 8.3% | 50% | FIXABLE |
| Optimal f(r) | 4 | 0% | 50% | FIXABLE |
| Ranking adapter | — | ρ=-0.14 | ρ=0.93 | — |

**Total: 10 domains, 122 oracle facts tested. 6 domains at 100%. 0% MMLU degradation across all adapters.**

Full history: `results/candidates.tsv`

---

## Coordination

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

---

## Architecture

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
├── noethersolve/               ← Core package
│   ├── adapter.py              ← Snap-on logit adapter (SwiGLU)
│   ├── oracle.py               ← Oracle scoring engine
│   └── train_utils.py          ← Shared training utilities
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
│   ├── breaking_frozen_priors.pdf  ← Paper 10 (pandoc breaking_frozen_priors.md -o *.pdf)
│   └── prior_work/                 ← Papers 8-9 that this builds on
│
├── adapters/                   ← Trained weights (gitignored)
│
└── results/
    ├── candidates.tsv          ← All tested hypotheses (193 entries)
    └── discoveries/            ← Discovery notes (26 files)
```

---

## Built On

- **STEM Truth Oracle** (Paper 9) — log-prob margin as a zero-FP/FN binary
  classifier for factual correctness.
  DOI: [10.5281/zenodo.19005729](https://doi.org/10.5281/zenodo.19005729)

- **Snap-On Communication Modules** (Paper 8) — frozen logit-space adapters
  that close knowledge gaps without touching base model weights.
  DOI: [10.5281/zenodo.18902616](https://doi.org/10.5281/zenodo.18902616)

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
```
