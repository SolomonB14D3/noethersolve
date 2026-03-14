# NoetherSolve

**https://github.com/SolomonB14D3/noethersolve** · **https://solomonb14d3.github.io/noethersolve**

[![Paper: Breaking Frozen Priors](https://img.shields.io/badge/Paper%2010-Breaking%20Frozen%20Priors-blue)](paper/breaking_frozen_priors.pdf) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19017290.svg)](https://doi.org/10.5281/zenodo.19017290)

**Automated scientific discovery that makes the model smarter with each cycle.**

Most autoresearch systems generate hypotheses and hope for the best. NoetherSolve closes the loop: it generates candidates, verifies them numerically, measures whether the model already knows them, and when it doesn't, **discovers the answer and teaches it back to the model**. Each discovery trains an adapter that persists through the rest of the run. The model that evaluates candidate #50 is smarter than the one that evaluated candidate #1, because every intervening discovery has been injected into it.

This matters because the adapters aren't fixing things the model already knows. The Q_f conservation law family, the stretch-resistant R_f ratio, the continuous Euler extension — none of these existed in any training corpus. The system discovered them through numerical simulation, verified they were real, confirmed the model had never seen them (oracle margin -30 to -44), and wrote them into the model's knowledge. After adapter training, the model recognizes and correctly ranks these quantities (margin flipped to +4 to +30, ranking Spearman rho = 0.893). The model now knows physics that no human had published.

And the adapters don't degrade existing knowledge. Zero MMLU degradation across every adapter tested, because they operate in logit space — they reshape the output distribution without touching the hidden-state knowledge pathway. Each cycle adds knowledge without taking any away.

LLMs are trained on what the field has collectively written and taught. Where the model is confidently wrong or blank, the literature is thin. That's where new science is most likely to be found. NoetherSolve automates this: propose, verify, check, discover, teach, repeat.

The method is domain-agnostic. We've applied it to fluid dynamics, electromagnetism, and gravitational mechanics so far. Any field where you can numerically verify a claim and ask a model about it is fair game.

### Paper

**Breaking Frozen Priors: Teaching Language Models to Discover Conservation Laws from Numerical Simulation** (Sanchez, 2026)
DOI: [10.5281/zenodo.19017290](https://doi.org/10.5281/zenodo.19017290)

Three-phase pipeline transforms a frozen oracle (margin -77.5 +/- 1.7) into a ranking engine (Spearman rho = 0.893 from baseline -0.143). Novel Q_f invariant family verified across chaotic vortex systems and extended to continuous 2D/3D Euler equations. The LLM gap pointed directly at the physics: the model's blind spot on weighted distance sums led to the discovery of stretch-resistant invariants relevant to 3D Navier-Stokes regularity. See [`paper/breaking_frozen_priors.pdf`](paper/breaking_frozen_priors.pdf).

---

## What It Does

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

### Discrete Point-Vortex

| Date | Domain | Expression | frac_var | Oracle | Status |
|------|--------|------------|----------|--------|--------|
| 2026-03-13 | Figure-8 3-body | e₁ = r₁₂+r₁₃+r₂₃ | 5.54e-04 | +4.50 | **DUAL-PASS** |
| 2026-03-13 | Figure-8 3-body | e₂ = r₁₂r₁₃+r₁₂r₂₃+r₁₃r₂₃ | 2.69e-03 | -1.67→**+1.30** | **FLIPPED** |
| 2026-03-13 | Point-vortex | Q = Σ ΓᵢΓⱼ rᵢⱼ | 5.36e-06 | -29.96→**+3.99** | **FLIPPED** |
| 2026-03-13 | Point-vortex | Q₂ = Σ ΓᵢΓⱼ rᵢⱼ² (= Γ·Lz) | 9.62e-12 | -43.9→**+29.6** | **FLIPPED** (exact) |
| 2026-03-13 | Point-vortex | Q_f family (7 powers, N=3-9) | 1e-5 to 1e-11 | ranked ρ=0.893 | **RANKING LEARNED** |
| 2026-03-13 | Point-vortex | K = Σ Γᵢ vᵢ² (kinetic) | 1e-5 to 1e-7 | low margin | GAP (independent of Q_f) |
| 2026-03-13 | Point-vortex | H - Lz | 9.48e-12 | -19.6→**+26.1** | **FLIPPED** |

### Continuous Fluid Extension (2D/3D Euler)

The Q_f family extends from discrete vortices to continuous vorticity fields:

```
Q_f[ω] = ∫∫ ω(x) ω(y) f(|x-y|) dx dy ≈ const
```

Verified across 6 test scenarios (laminar, turbulent 2D, 3D vortex rings, viscous NS):

| f(r) | 2D Laminar | 2D Turbulent | 3D Rings | Status |
|------|-----------|-------------|---------|--------|
| -ln(r) | 4.32e-03 | 2.77e-03 | — | Known (energy) |
| e^(-r) | 3.09e-04 | 5.42e-03 | 1.79e-03 | **NEW** |
| tanh(r) | — | 6.82e-03 | — | **NEW** |
| √r | 3.48e-04 | 1.07e-02 | 2.95e-03 | **NEW** |
| 1/r | — | — | 3.78e-04 | **NEW** (3D best) |

Viscous (Navier-Stokes) decay scales linearly with ν. See `results/discoveries/qf_family_comprehensive.md`.

### 3D Stretch-Resistant Ratio (the NS connection)

Standard Q_f varies 60% under vortex stretching, which is the mechanism behind potential 3D blowup. We tested four modifications:

| Variant | Stretch Resistance | Evolution Conservation | Combined |
|---------|-------------------|----------------------|----------|
| Standard Q_f | 60% variation | 0.14% | 2.95% |
| Q_f / Enstrophy | 17% | 0.36% | 2.44% |
| Curvature-weighted | 4% | 1.02% | 6.4% |
| **R_f = Q_exp / Q_inv** | **2%** | **0.17%** | **0.59%** |

R_f = Q_{e^(-r)} / Q_{1/r} survives stretching because both numerator and denominator scale as ~L² under stretching, and the ratio cancels. Physically, R_f measures the locality of vorticity interactions: how much the dynamics depends on nearby vs distant vorticity. Combined with energy conservation, R_f provides a constraint that persists through the stretching that could cause 3D blowup. See `research/qf_regularity_connection.md` and `research/test_stretch_resistant_qf.py`.

### Electromagnetism (New Domain)

Spectral Maxwell solver verifying conservation of obscure EM invariants (Lipkin's zilch, optical chirality, helicity, super-energy). All confirmed exactly conserved (frac_var < 10^-6).

Oracle results on Qwen3-4B-Base: baseline **1/12 pass rate** (8.3%). The model fails on basic energy conservation (margin -4.08) and momentum (margin -5.35), not just obscure quantities. Zilch (margin -11.63) and super-energy (margin -9.94) are complete knowledge gaps.

With `em_adapter_v4`: **6/12 pass rate** (50%), confirming FIXABLE_BIAS diagnostic. Flipped facts include energy, momentum, chirality, helicity, super-energy, and zilch. Mean margin improved from -11.04 to -0.21.

See `results/discoveries/em_conservation_laws.md` and `results/discoveries/em_zilch_chirality.md`.

### Continuous Q_f Extension (2D/3D Euler)

The discrete point-vortex Q_f family extends to continuous vorticity fields:

```
Q_f[ω] = ∫∫ ω(x) ω(y) f(|x-y|) dx dy ≈ const
```

Oracle results: baseline **0/12 pass rate** (complete knowledge gap). With `qf_continuous_adapter`: **7/12 pass rate** (58.3%), diagnostic changed from KNOWLEDGE_GAP to FIXABLE_BIAS.

| Flipped Fact | Baseline | Adapter | Delta |
|--------------|----------|---------|-------|
| Q_f extension formula | -6.5 | +8.0 | +14.5 |
| f=-ln(r) gives energy | -44.3 | +17.2 | +61.5 |
| Q_{e^(-r)} conserved | -59.1 | +2.1 | +61.2 |
| Conservation mechanism | -43.7 | +11.3 | +55.0 |
| Q_f bounds → NS regularity | -11.7 | +3.6 | +15.3 |

See `results/discoveries/continuous_qf_oracle.md`.

Full history: `results/candidates.tsv` (169 entries)

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
    ├── candidates.tsv          ← All tested hypotheses (159 entries)
    └── discoveries/            ← Discovery notes (19 files)
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
