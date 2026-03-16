# Discoveries

Organized into four categories:

## `novel_findings/` — New Science (12 files)

Domain-agnostic discoveries that generalize beyond any specific model. These are the publishable results.

| Finding | File | Summary |
|---------|------|---------|
| Q_f invariant family | `q_f_universal.md` | Q_f = Σ ΓᵢΓⱼ f(rᵢⱼ) universally conserved for arbitrary f in N-vortex systems (N=3-8) |
| Optimal Q_f combination | `optimal_qf_combination.md` | No single f(r) is optimal; linear combination of basis functions achieves 300× better conservation via cancellation mechanism |
| Q_f dichotomy | `qf_dichotomy_regularity.md` | Fundamental tradeoff: stretch-resistant f(r) (positive powers) vs concentration-detecting f(r) (negative powers). NS regularity implications |
| Q_κ curvature-weighted | `curvature_weighted_stretch_resistance.md` | Curvature-weighted Q_κ is 15× more stretch-resistant; dimensional cancellation κ→κ/s, ds→s·ds |
| Z₃ phase cancellation | `z3_phase_cancellation.md` | Z₃ choreographic symmetry explains why positive powers work on figure-8 but inverse powers fail. Critical exponent range -0.67 < p < 2.55 |
| R_f ratio invariant | `qf_ratio_stretch_resistant.md` | R_f = Q_{e^(-r)} / Q_{1/r} optimally stretch-resistant while maintaining evolution conservation |
| Kinetic invariant K | `kinetic_invariant_K.md` | K = Σ Γᵢ vᵢ² approximately conserved; distance-angle cancellation mechanism. Independent from Q_f (R²=0.048) |
| Cross-domain mechanisms | `cross_domain_conservation_mechanisms.md` | Vortex (circulation weighting) vs gravitational (Z₃ symmetry) conservation: distinct mechanisms, shared structure |
| Dipole exact conservation | `dipole_test_vortex_exact.md` | Dipole + test vortex on symmetry axis: weighted distance sum exactly conserved (frac_var ~ 10⁻¹⁵) |
| Parallel dipole sum | `parallel_dipole_sum.md` | Internal separations of N parallel dipoles exactly conserved — proven equivalent to impulse conservation (Px/Γ = const) |
| EM conservation laws | `em_conservation_laws.md` | Maxwell solver verifying energy, chirality (Lipkin's zilch), helicity, super-energy as exact EM invariants |
| Q_√r viscous decay | `viscous_decay_linear_scaling.md` | Q_√r decays as ν^0.99 (perfectly linear) under viscosity — most consistent scaling across all Q_f variants |

## `model_specific/` — Qwen3-4B-Base Adapter Results (13 files)

Results specific to the Qwen3-4B-Base oracle model. Adapter training outcomes, margin improvements, and routing strategies. Useful for reproducing experiments but not publishable science on their own.

Key findings:
- **Hybrid routing** (adapter_combination_findings.md): 82.1% by picking best of joint vs orthogonal per fact
- **Frontier domains** (frontier_domains_findings.md): 96/96 (100%) with orthogonal adapters
- **Frozen priors** (frozen_prior_analysis.md): Oracle exhibits frozen priors on H·r₁₂ + α·Lz regardless of α
- **Orthogonal routing** generalizes across all 67 domains (1014/1014 facts)

## `archived/` — Incremental/Consolidated (9 files)

Extensions, numerical confirmations, or duplicates of findings already covered by the novel_findings files. Kept for reference.

## `sessions/` — Session Summaries (2 files)

Chronological summaries of work completed in each session.
