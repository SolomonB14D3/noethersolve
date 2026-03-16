# High-Priority Research Targets

20 directions ranked by: (1) model likely wrong, (2) numerically verifiable,
(3) tool would be high-value to any AI agent. Grouped by domain.

---

## Physics — Conservation Laws & Invariants

### 1. Magnetohydrodynamics (MHD) Conservation
**Why:** MHD has 5+ conservation laws (magnetic helicity, cross-helicity,
magnetic flux, Elsässer energy, Alfvén wave action) that interact
non-trivially. Models confuse which are exact vs approximate in resistive
vs ideal regimes. Resistive MHD *breaks* magnetic helicity conservation
— a surprising truth the model almost certainly gets wrong.
**Verifiable:** RK45 integration of induction equation + Navier-Stokes.
frac_var test on each invariant under ideal vs resistive conditions.
**Tool:** `check_mhd_conservation(B_field, v_field, resistivity)` —
validates which MHD invariants hold for given parameters.
**Facts:** ~12 (helicity exact/broken, cross-helicity conditions,
flux freezing theorem, Elsässer symmetry, reconnection rates).

### 2. General Relativity — ADM Conservation & Constraints
**Why:** The ADM formalism has Hamiltonian + momentum constraints that must
be satisfied on every spatial slice. Models mix up which quantities are
conserved (ADM mass, Bondi mass, Komar mass) and when. Bondi mass
*decreases* with gravitational radiation — ADM mass doesn't. This is a
textbook-level surprising truth that base models conflate.
**Verifiable:** Constraint violation monitoring on Schwarzschild/Kerr
slicings. ADM vs Bondi mass comparison for radiating systems.
**Tool:** `check_gr_constraints(metric, extrinsic_curvature)` — validates
Hamiltonian and momentum constraints, computes ADM/Bondi/Komar masses.
**Facts:** ~12 (ADM vs Bondi, constraint propagation, Penrose inequality,
positive mass theorem conditions, quasi-local mass definitions).

### 3. Plasma Physics — Adiabatic Invariants
**Why:** Charged particles in magnetic fields have 3 adiabatic invariants
(magnetic moment μ, longitudinal invariant J, flux invariant Φ). Each
breaks under different conditions. Models treat them as exact when they're
approximate — and get the breaking conditions wrong.
**Verifiable:** Boris pusher integration in mirror/tokamak geometries.
frac_var on μ, J, Φ under varying field gradients.
**Tool:** `check_adiabatic_invariants(trajectory, B_field)` — reports
which invariants hold and to what precision.
**Facts:** ~10 (μ conservation/breaking, J under slow drift, Φ for
trapped particles, banana orbits, pitch angle scattering thresholds).

### 4. Fluid Turbulence — Kolmogorov Exact Results
**Why:** Kolmogorov's 4/5 law is one of the few exact results in
turbulence. Models confuse it with the approximate -5/3 spectrum. The
4/5 law is *exact* for the third-order structure function in the inertial
range — no approximation. The -5/3 spectrum is dimensional analysis.
**Verifiable:** Direct numerical simulation structure function computation.
**Tool:** `check_turbulence_scaling(velocity_field, Re)` — validates
structure functions, computes intermittency corrections, checks exact
vs approximate results.
**Facts:** ~12 (4/5 law exact, -5/3 approximate, intermittency corrections,
Batchelor vs Kraichnan 2D scaling, enstrophy cascade direction, inverse
energy cascade conditions).

### 5. Topological Phases — Bulk-Boundary Correspondence
**Why:** Topological invariants (Chern numbers, Z₂ indices, winding
numbers) are *exactly* quantized — not approximately. Models often present
them as approximate or confuse the conditions for protection. The quantum
Hall conductance is σ_xy = νe²/h *exactly*, not approximately.
**Verifiable:** Band structure computation → Chern number extraction.
Berry phase integration on closed paths.
**Tool:** `check_topological_invariant(hamiltonian, symmetry_class)` —
computes Chern number, Z₂ index, validates bulk-boundary correspondence.
**Facts:** ~12 (Chern number quantization, Z₂ classification, symmetry
protection conditions, tenfold way, edge state counting, Majorana
zero modes).

---

## Mathematics — Where Models Are Confidently Wrong

### 6. Homological Algebra — Spectral Sequences
**Why:** Spectral sequences are a computational tool where models
hallucinate freely. The differentials, page structure, and convergence
conditions have exact rules that models get subtly wrong. Serre spectral
sequence for fibrations is a prime example.
**Verifiable:** Page-by-page computation of known spectral sequences
(Serre, Adams, Atiyah-Hirzebruch). Check differentials.
**Tool:** `compute_spectral_sequence(fibration_data, page_limit)` —
computes differentials and E_r pages, validates convergence.
**Facts:** ~10 (Serre SS for sphere fibrations, Adams SS d₂ differentials,
Leray-Serre edge homomorphisms, multiplicative structure, convergence
conditions).

### 7. Algebraic Geometry — Intersection Theory
**Why:** Bézout's theorem generalizations, Schubert calculus, and
intersection multiplicities have exact numerical answers. Models
frequently get multiplicities wrong or confuse transverse vs tangential
intersections. How many lines meet 4 general lines in P³? (Answer: 2,
not 1 or 4.)
**Verifiable:** Symbolic computation of intersection numbers on
Grassmannians and flag varieties.
**Tool:** `compute_intersection_number(variety, divisors)` — Schubert
calculus, Bézout applications, multiplicity computation.
**Facts:** ~10 (27 lines on cubic surface, 2 lines meeting 4 in P³,
degree-genus formula, Riemann-Hurwitz, adjunction formula applications).

### 8. Dynamical Systems — Ergodic Theory Exact Results
**Why:** Ergodic theory has surprising exact results. Birkhoff's theorem
says time averages = space averages for ergodic systems — but models
confuse ergodic with mixing with Bernoulli. The hierarchy is strict:
Bernoulli ⊂ K-mixing ⊂ mixing ⊂ ergodic, and each inclusion is proper.
**Verifiable:** Numerical computation of Lyapunov exponents, entropy,
mixing rates for standard maps (Arnold cat, baker's, standard).
**Tool:** `classify_dynamical_system(map_data)` — computes Lyapunov
spectrum, classifies ergodic hierarchy level, checks hyperbolicity.
**Facts:** ~12 (ergodic hierarchy strictness, KAM tori vs ergodic
regions, Pesin's formula relating entropy to Lyapunov exponents,
Ruelle inequality, Sinai-Ruelle-Bowen measures).

---

## Therapeutics & Life Sciences

### 9. Pharmacokinetic Drug-Drug Interactions
**Why:** CYP450 enzyme interactions create non-obvious effects. CYP3A4
inhibition by grapefruit juice can increase drug exposure 3-10x — models
know this but get the *magnitude* wrong and miss time-dependent inhibition
(mechanism-based inhibitors are irreversible, competitive are not).
**Verifiable:** PBPK model simulation with known Ki/kinact values.
**Tool:** `predict_ddi(perpetrator, victim, cyp_data)` — predicts AUC
fold-change, classifies interaction mechanism, checks FDA guidance
thresholds (AUC ratio > 5 = strong inhibitor).
**Facts:** ~12 (CYP3A4 vs 2D6 vs 2C19 substrate specificity,
time-dependent vs reversible inhibition, auto-induction, prodrug
activation failures, genetic polymorphism effects on 2D6).

### 10. Antibody Engineering — Developability
**Why:** Models know antibody structure but get developability wrong.
High aggregation propensity, poor solubility, polyreactivity — these kill
90% of candidates. The surprising truth: net charge, not hydrophobicity,
is the strongest single predictor of viscosity at high concentration.
**Verifiable:** Sequence-based prediction validated against published
DLS/SEC data.
**Tool:** `assess_antibody_developability(sequence)` — predicts
aggregation, viscosity, clearance risk, polyreactivity from sequence.
**Facts:** ~12 (charge-viscosity correlation, CDR hydrophobicity vs
framework effects, isotype-specific half-life, FcRn binding pH
dependence, deamidation hotspots NG/NS/DG).

### 11. Cancer Immunotherapy — Neoantigen Prediction
**Why:** MHC binding prediction is well-known, but the full pipeline
(proteasomal cleavage → TAP transport → MHC binding → TCR recognition)
has compounding errors. Models overweight MHC binding and underweight
cleavage/transport — leading to high false positive rates.
**Verifiable:** Against validated neoantigen databases (IEDB, TESLA).
**Tool:** `predict_neoantigen(mutation, hla_type)` — full pipeline
scoring with per-step confidence, not just MHC binding.
**Facts:** ~12 (proteasomal cleavage preferences, TAP binding motifs,
MHC-I vs MHC-II pathway differences, immunoproteasome vs constitutive,
TCR cross-reactivity patterns).

### 12. RNA Therapeutics — Secondary Structure & Stability
**Why:** mRNA therapeutic design depends on codon optimization, UTR
design, and modified nucleoside effects. Models know about pseudouridine
but get the thermodynamic effects wrong — Ψ *destabilizes* some
secondary structures while stabilizing others depending on context.
**Verifiable:** Nearest-neighbor thermodynamic calculations vs
experimental melting curves.
**Tool:** `design_mrna_therapeutic(protein_sequence, target_tissue)` —
codon optimization, UTR selection, modification strategy, stability
prediction.
**Facts:** ~10 (Ψ context-dependent stability, N1-methylpseudouridine
vs Ψ, 5' cap analog effects, poly(A) length optimization, rare codon
effects on folding kinetics).

---

## Computational Science — Numerical Methods

### 13. Numerical PDE — Stability & Convergence Traps
**Why:** CFL conditions, von Neumann stability analysis, and order of
accuracy have exact results that models state approximately. The Lax
equivalence theorem (consistency + stability ⇔ convergence) is exact
but models confuse necessary vs sufficient conditions. Leap-frog is
*unstable* for diffusion — a common model error.
**Verifiable:** Run schemes on test PDEs, measure convergence rates,
check stability boundaries.
**Tool:** `analyze_pde_scheme(stencil, equation_type)` — computes
CFL number, von Neumann amplification factor, order of accuracy,
identifies instability regimes.
**Facts:** ~12 (CFL exact thresholds for standard schemes, leap-frog
instability for parabolic, Crank-Nicolson unconditional stability,
supraconvergence phenomena, Kreiss matrix theorem).

### 14. Optimization — Convexity & Convergence Rates
**Why:** Models routinely claim algorithms converge faster than they do.
Gradient descent on L-smooth μ-strongly convex functions converges at
rate (1-μ/L)^k — exact, not approximate. Nesterov acceleration achieves
(1-√(μ/L))^k — provably optimal. Models confuse these and misstate
when acceleration helps.
**Verifiable:** Run algorithms on known-condition-number problems,
measure actual vs predicted convergence.
**Tool:** `analyze_convergence(objective, algorithm, params)` — predicts
convergence rate, checks strong convexity, smoothness, compares to
lower bounds (Nesterov's oracle complexity).
**Facts:** ~10 (exact rates for GD/AGD/Newton, heavy ball vs Nesterov,
condition number dependence, restart strategies, non-convex escape
time from saddle points).

---

## Materials & Chemistry

### 15. Battery Electrochemistry — Degradation Mechanisms
**Why:** Lithium-ion degradation has multiple competing mechanisms (SEI
growth, lithium plating, cathode cracking, transition metal dissolution)
with non-obvious interactions. Models know individual mechanisms but get
the *coupling* wrong. SEI growth rate follows √t (parabolic) — but only
in the initial phase. Calendar aging vs cycle aging have different
dominant mechanisms.
**Verifiable:** Equivalent circuit model fitting to impedance data,
capacity fade rate prediction.
**Tool:** `predict_battery_degradation(chemistry, cycling_protocol)` —
predicts capacity fade, identifies dominant mechanism, estimates
remaining useful life.
**Facts:** ~12 (SEI growth kinetics, lithium plating onset conditions,
NMC vs LFP degradation differences, fast-charge effects, temperature
Arrhenius factors, calendar vs cycle aging crossover).

### 16. Catalysis — Sabatier Principle & Volcano Plots
**Why:** The Sabatier principle (optimal catalyst binds intermediates
neither too strongly nor too weakly) gives rise to volcano plots. Models
state this qualitatively but get the quantitative scaling relations wrong.
Brønsted-Evans-Polanyi relations give *exact* linear correlations between
activation energies and reaction energies — with known slopes.
**Verifiable:** DFT binding energy databases (Catalysis-Hub).
**Tool:** `predict_catalyst_activity(surface, reaction)` — applies
scaling relations, predicts position on volcano, identifies
rate-limiting step.
**Facts:** ~10 (BEP slopes for different bond types, d-band center
theory, volcano peak positions for HER/OER/ORR, alloying effects,
strain vs ligand effects, CO poisoning thresholds).

---

## Emerging & Cross-Disciplinary

### 17. Information Theory — Channel Capacity Exact Results
**Why:** Shannon's channel capacity theorem gives exact limits. Models
know C = B log₂(1 + SNR) for AWGN but get extensions wrong. The
capacity of the binary symmetric channel is 1 - H(p) *exactly*.
Models confuse achievability with converse bounds and misstate
multi-user capacity regions.
**Verifiable:** Numerical computation of mutual information, comparison
to known capacities.
**Tool:** `compute_channel_capacity(channel_model, params)` — exact
capacity for standard channels, bounds for multi-user, MAC/BC capacity
regions.
**Facts:** ~10 (BSC/BEC/AWGN exact capacities, MAC pentagon region,
broadcast channel degraded vs non-degraded, dirty paper coding,
feedback doesn't increase capacity for memoryless channels, Slepian-Wolf
source coding).

### 18. Epidemiological Modeling — Reproduction Numbers
**Why:** R₀ is widely misunderstood. Models confuse R₀ (basic) with Rₜ
(effective) and get the herd immunity threshold wrong when populations
are heterogeneous. For heterogeneous mixing, the herd immunity threshold
is *lower* than 1 - 1/R₀ — a surprising and practically important truth.
**Verifiable:** SIR/SEIR integration with known parameters.
**Tool:** `compute_epidemic_params(model_type, params)` — R₀, Rₜ, herd
immunity threshold (homogeneous and heterogeneous), final size relation,
generation interval effects.
**Facts:** ~10 (R₀ vs Rₜ distinction, heterogeneous HIT correction,
final size equation exactness, generation interval vs serial interval,
superspreading k parameter effects, backward vs forward generation
interval).

### 19. Climate Physics — Radiative Transfer Exact Results
**Why:** Radiative forcing has exact solutions for gray atmospheres and
exact logarithmic dependence for CO₂ (ΔF = 5.35 ln(C/C₀) W/m²). Models
confuse the *logarithmic* (exact for well-mixed gases) with *linear*
(wrong) or *square root* (only for strongly-absorbing bands). The
climate sensitivity framework has exact decomposition into feedbacks.
**Verifiable:** Line-by-line radiative transfer computation.
**Tool:** `compute_radiative_forcing(gas, concentration, baseline)` —
exact forcing formulas, feedback decomposition, climate sensitivity
estimation with uncertainty propagation.
**Facts:** ~12 (CO₂ logarithmic forcing exact, CH₄ square-root
correction, water vapor feedback magnitude, Planck response exact value
3.2 W/m²/K, lapse rate + WV partial cancellation, cloud feedback
as dominant uncertainty).

### 20. Geophysics — Seismic Wave Propagation
**Why:** Seismic wave velocities have exact relationships to elastic
moduli (Vp = √((K + 4G/3)/ρ), Vs = √(G/ρ)). Models confuse P-wave
and S-wave velocity relationships and get Poisson's ratio constraints
wrong. For typical rocks, Vp/Vs ≈ √3 for Poisson's ratio 0.25 — but
this is approximate and the *exact* relationship is well-known.
**Verifiable:** Elastic wave equation integration, comparison to
known velocity profiles (PREM, ak135).
**Tool:** `compute_seismic_properties(elastic_moduli, density)` —
P/S velocities, Poisson's ratio, impedance contrast, reflection
coefficients, anisotropy parameters (Thomsen).
**Facts:** ~10 (Vp/Vs exact formula, Poisson's ratio bounds for
stability, Snell's law for anisotropic media, surface wave dispersion
exact solutions, free oscillation eigenfrequencies, PKJKP existence).

---

## Priority Ranking

| # | Direction | Model Weakness | Tool Value | Verifiability | Score |
|---|-----------|---------------|------------|---------------|-------|
| 1 | MHD Conservation | Very high (resistive confusion) | High | Excellent | ★★★★★ |
| 4 | Turbulence Scaling | Very high (4/5 vs -5/3) | High | Excellent | ★★★★★ |
| 9 | Drug-Drug Interactions | High (magnitude errors) | Very high | Excellent | ★★★★★ |
| 13 | Numerical PDE Stability | Very high (CFL confusion) | Very high | Excellent | ★★★★★ |
| 17 | Information Theory | High (multi-user confusion) | High | Excellent | ★★★★★ |
| 2 | GR Constraints | Very high (mass confusion) | Medium | Good | ★★★★ |
| 5 | Topological Phases | High (quantization confusion) | High | Good | ★★★★ |
| 8 | Ergodic Theory | Very high (hierarchy confusion) | Medium | Excellent | ★★★★ |
| 14 | Optimization Rates | High (rate misstatement) | Very high | Excellent | ★★★★ |
| 18 | Epidemiology R₀ | High (HIT heterogeneity) | Very high | Excellent | ★★★★ |
| 19 | Radiative Transfer | High (log vs linear) | Very high | Good | ★★★★ |
| 3 | Plasma Adiabatic | High (exact vs approximate) | Medium | Excellent | ★★★ |
| 6 | Spectral Sequences | Very high (hallucination) | Medium | Good | ★★★ |
| 7 | Intersection Theory | High (multiplicity errors) | Medium | Good | ★★★ |
| 10 | Antibody Developability | High (predictors wrong) | Very high | Medium | ★★★ |
| 11 | Neoantigen Prediction | Medium (pipeline errors) | Very high | Medium | ★★★ |
| 12 | RNA Therapeutics | Medium (Ψ effects) | High | Good | ★★★ |
| 15 | Battery Degradation | Medium (coupling errors) | High | Good | ★★★ |
| 16 | Catalysis Volcano | High (BEP slopes wrong) | Medium | Good | ★★★ |
| 20 | Seismic Propagation | Medium (Vp/Vs confusion) | Medium | Excellent | ★★★ |
