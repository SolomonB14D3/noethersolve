# Discoveries

Organized into three categories:

## `novel_findings/` — New Science

Domain-agnostic discoveries that generalize beyond any specific model. These are the publishable results.

| Finding | File | Summary |
|---------|------|---------|
| Q_f invariant family | `q_f_universal.md`, `qf_family_comprehensive.md` | Q_f = Σ ΓᵢΓⱼ f(rᵢⱼ) universally conserved for arbitrary f in N-vortex systems |
| Q_n power law family | `q_power_law_family.md` | Q_n = Σ ΓᵢΓⱼ rᵢⱼ^n near-conserved for all powers n; exact at n=2 |
| Z₃ phase cancellation | `z3_phase_cancellation.md`, `figure8_symmetric_polys.md` | Z₃ symmetry of figure-8 orbit causes phase cancellation in power-law time derivatives |
| Q_κ curvature-weighted | `curvature_weighted_stretch_resistance.md` | Curvature-weighted Q_κ is 15× more stretch-resistant than standard Q_f |
| Q_f dichotomy | `qf_dichotomy_regularity.md` | Fundamental tradeoff: stretch-resistant functions vs concentration-detecting functions |
| R_f ratio invariant | `qf_ratio_stretch_resistant.md`, `stretch_resistant_qf_ratio.md` | R_f = Q_{e^(-r)} / Q_{1/r} optimally stretch-resistant |
| Optimal Q_f combination | `optimal_qf_combination.md` | No single f(r) is optimal; linear combination of basis functions achieves 99.6% improvement |
| Continuous Q_f → Euler | `continuous_qf_euler.md` | Point vortex Q_f invariants extend to continuous vorticity in 2D Euler equations |
| Kinetic invariant K | `kinetic_invariant_K.md` | K = Σ Γᵢ vᵢ² approximately conserved in N-vortex systems |
| N-vortex weighted distance | `n_vortex_weighted_distance.md` | Circulation-weighted distance sum near-invariant |
| Dipole exact conservation | `dipole_test_vortex_exact.md` | Dipole + test vortex on symmetry axis: weighted distance sum exactly conserved |
| Parallel dipole sum | `parallel_dipole_sum.md` | Internal separations of N parallel dipoles exactly conserved |
| ε = Γ₃ relationship | `epsilon_gamma_relationship.md` | Optimal weighting coefficient for restricted 3-vortex is exactly the weak circulation |
| Cross-domain mechanisms | `cross_domain_conservation_mechanisms.md`, `cross_domain_weighted_distance.md` | Gravitational vs vortex conservation: distinct mechanisms, shared mathematical structure |
| EM conservation laws | `em_conservation_laws.md`, `em_zilch_chirality.md` | Lipkin's zilches and optical chirality verified as exact EM invariants |
| Frozen prior analysis | `frozen_prior_analysis.md` | Models exhibit frozen priors on H·r₁₂ + α·Lz regardless of α coefficient |

## `model_specific/` — Qwen3-4B-Base Adapter Results

Results specific to the Qwen3-4B-Base oracle model. These document adapter training outcomes, margin improvements, and routing strategies. Useful for reproducing experiments but not publishable science on their own.

Key findings:
- **Hybrid routing** (adapter_combination_findings.md): 82.1% by picking best of joint vs orthogonal per fact
- **Frontier domains** (frontier_domains_findings.md): 96/96 (100%) with orthogonal adapters
- **Orthogonal routing** generalizes across all 67 domains (1014/1014 facts)

## `sessions/` — Session Summaries

Chronological summaries of work completed in each session.
