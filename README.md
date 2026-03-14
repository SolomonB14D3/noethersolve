# NoetherSolve

**https://github.com/SolomonB14D3/noethersolve** · **https://solomonb14d3.github.io/noethersolve**

[![Paper: Breaking Frozen Priors](https://img.shields.io/badge/Paper%2010-Breaking%20Frozen%20Priors-blue)](paper/breaking_frozen_priors.pdf) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19017290.svg)](https://doi.org/10.5281/zenodo.19017290)

**Find what LLMs don't know about what the universe conserves. Then fix it.**

### Paper

**Breaking Frozen Priors: Teaching Language Models to Discover Conservation Laws from Numerical Simulation** (Sanchez, 2026)
DOI: [10.5281/zenodo.19017290](https://doi.org/10.5281/zenodo.19017290)

Three-phase pipeline transforms a frozen oracle (margin -77.5 +/- 1.7) into a physics-aware ranking engine (Spearman rho = 0.893 from baseline -0.143). Novel Q_f = Sigma Gamma_i Gamma_j f(r_ij) family verified across N=3-9 chaotic vortex systems and extended to continuous 2D/3D Euler equations. See [`paper/breaking_frozen_priors.pdf`](paper/breaking_frozen_priors.pdf).

Emmy Noether proved that every continuous symmetry of a physical system
corresponds to a conserved quantity. NoetherSolve finds where LLMs fail to
recognize those quantities — and closes the gap with targeted adapters.

---

## What It Does

NoetherSolve runs a **dual-filter pipeline**:

```
Hypothesis (expression)
       │
       ▼
 Numerical checker          ← Is this quantity actually conserved?
 (RK45 integration,           frac_var = σ/|mean| < threshold
  frac_var test)
       │ PASS
       ▼
 Oracle filter              ← Does the model know it?
 (log-prob margin,            margin = log P(truth) − log P(best distractor)
  base LLM)
       │
       ├─ PASS  → DUAL-PASS: known conserved quantity, archive it
       │
       └─ FAIL  → Run repair pass (adapter):
                    ├─ margin improves  → FIXABLE BIAS: apply domain adapter
                    └─ margin worsens   → KNOWLEDGE GAP: train new adapter
```

Every discovery lands in one of four diagnostic quadrants. The pipeline
tells you exactly which one and what to do next.

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

Full history: `results/candidates.tsv` (159 entries)

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
├── oracle_wrapper.py          ← Oracle + repair + ranking + quadrant diagnosis
├── claim.py                   ← Coordination: THINK/CLAIM/RELEASE
├── autonomy_loop.py           ← Fully autonomous sweep + hypothesis generation
├── dashboard.py               ← Results dashboard from candidates.tsv
├── claims.json                ← Live claims registry
│
├── problems/                  ← Domain plugins (fork here)
│   ├── problem_template.yaml  ← Starting point for new domains
│   ├── *_facts.json           ← Verification sets
│   └── *_checker.py           ← Numerical integrators
│
├── monitors/                  ← Reusable checker monitors
│   ├── sum_pairwise_distances.py
│   ├── e2_symmetric_poly.py
│   └── __init__.py
│
├── research/                  ← Continuous Q_f extension experiments
│   ├── test_continuous_qf.py  ← 2D laminar vortex verification
│   ├── test_qf_turbulence.py  ← 2D turbulent dynamics
│   ├── test_3d_vortex_qf.py   ← 3D vortex ring verification
│   ├── test_qf_viscous.py     ← Navier-Stokes viscous decay
│   ├── test_qf_concentration.py ← Concentration scaling response
│   ├── test_3d_stretching.py  ← 3D vortex stretching
│   ├── learn_optimal_f.py     ← Gradient descent for optimal f(r)
│   └── qf_regularity_connection.md ← NS regularity analysis
│
├── train_ranking_v2.py        ← Ranking adapter (ListNet + hard negatives)
├── train_vortex_adapter.py    ← Domain-specific logit adapter (MLX)
├── train_choreography_adapter.py ← Figure-8 choreography adapter (MLX)
│
├── adapters/                  ← Trained adapter weights (gitignored)
│
├── paper/
│   └── breaking_frozen_priors.pdf ← Paper 10
│
└── results/
    ├── candidates.tsv         ← All tested hypotheses (159 entries)
    └── discoveries/           ← Discovery notes (19 files)
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
