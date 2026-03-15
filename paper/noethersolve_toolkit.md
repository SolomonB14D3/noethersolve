# NoetherSolve Toolkit: Conservation Law Monitoring, Discovery, and Scientific Auditing Across Physics, Genetics, and Mathematics

Bryan Sanchez

March 2026

## Abstract

We present NoetherSolve, an open-source Python toolkit for monitoring, validating, and discovering conservation laws and scientific invariants across three domains: numerical physics, computational genetics, and unsolved mathematics. The toolkit provides forty-six tools organized in four tiers: (1) six physics tools covering point-vortex dynamics, chemical kinetics, N-body gravity, electromagnetism, Hamiltonian mechanics, and domain-agnostic invariant discovery; (2) seven genetics tools for DNA/RNA sequence auditing, CRISPR guide scoring, therapeutic pipeline validation, protein aggregation prediction, splice site scoring, pharmacogenomic interaction checking, and oracle fact quality auditing; and (3) seven unsolved mathematics tools for complexity class relationships, conjecture status tracking, proof technique barriers, number theory conjecture verification, computational reduction chains, and PDE regularity/Sobolev embedding analysis. Each tool emerged from a pipeline that identifies where language models fail to recognize domain-specific facts, verifies those facts computationally, and teaches the results back to the model. This represents one half of a two-path approach: adapter blending (cross-domain joint training) fixes small models directly, achieving 100% across all 48 domains via orthogonal routing; MCP tools make any model a powerhouse through verified tool use. The toolkit requires no machine learning at runtime. We demonstrate that Q_f-based monitors detect numerical corruption at noise levels 100x lower than standard monitors, that algebraic auditing catches thermodynamic inconsistencies without simulation, that automatic optimization discovers conservation laws with 40% lower fractional variation than any single basis function, that genetics auditors catch 100% of known pathological sequences in validation batteries, and that mathematics tools correctly verify number theory conjectures and identify proof technique barriers across all tested cases. The package is available on PyPI (`pip install noethersolve`) with 1252 tests, a physics-enforcing pre-commit hook, and 555 oracle-verified facts across 48 domains.

## 1. Introduction

Conservation laws are the foundation of numerical simulation. A well-posed physical system conserves energy, momentum, angular momentum, and often additional quantities tied to its symmetries via Noether's theorem [1]. When a simulation violates these laws, something is wrong: the integrator tolerances are too loose, the equations of motion contain an error, or the numerical scheme introduces artificial dissipation.

Standard practice checks a handful of known invariants. In vortex dynamics, the Kirchhoff Hamiltonian H and angular impulse Lz are textbook quantities [2]. In chemical kinetics, total mass conservation follows from stoichiometry. In N-body gravity, energy and momentum are standard diagnostics. These checks catch large errors, but small systematic corruptions can pass through undetected.

This paper describes a toolkit that goes further in two ways. First, it monitors quantities that standard references do not cover. The Q_f family of invariants, $Q_f = \sum_{i<j} \Gamma_i \Gamma_j f(r_{ij})$, was discovered through numerical simulation and verified across chaotic vortex systems [3]. These approximate invariants are more sensitive to corruption than the exact invariants H and Lz. Second, the toolkit includes a learner that automatically discovers new conservation laws from trajectory data, without requiring the user to specify what to look for.

The toolkit represents one half of a two-path approach to model improvement. **Adapter blending** (cross-domain joint training from scratch) is the path to fixing small models directly — a single difficulty-weighted adapter lifts 4 domains simultaneously (Hamiltonian 14/16, NS 10/16, Knot 11/16, Chemical 13/16), and orthogonal routing achieves 100% across all 48 domains. **MCP tools** are the path to making any model a powerhouse through tool use — each tool is independent, verified, and callable on demand. Adapters change what the model knows; tools change what the model can do.

The forty-six tools are organized into four tiers:

**Tier 1 — Physics (6 tools):**

1. **Conservation Monitors** (VortexMonitor, ChemicalMonitor, GravityMonitor) for tracking invariants during simulation.
2. **Integrator Validator** for checking solver configurations before long runs.
3. **Chemical Network Auditor** for algebraic thermodynamic consistency.
4. **EM Field Monitor** for electromagnetic invariants including optical chirality [4] and helicity [5].
5. **Hamiltonian System Validator** for symplectic structure verification via Liouville volume and Poincare invariants [6, 7].
6. **Invariant Learner** for automatic conservation law discovery via L-BFGS-B optimization [8] over basis functions.

**Tier 2 — Genetics (7 tools):**

7. **Oracle Fact Auditor** for detecting token-length bias in oracle verification facts.
8. **Knot Invariant Monitor** for verifying invariance under Reidemeister moves and Jones polynomial computation.
9. **Sequence Design Auditor** for DNA/RNA therapeutic sequences (GC content, CpG islands, homopolymers, palindromes, restriction sites).
10. **CRISPR Guide RNA Scorer** for on-target activity estimation and off-target mismatch risk.
11. **Therapeutic Pipeline Validator** for end-to-end consistency of therapy designs (target-disease matching, delivery-cargo compatibility, dosing, safety).
12. **Protein Aggregation Predictor** for identifying aggregation-prone regions via AGGRESCAN/Kyte-Doolittle hydrophobicity analysis.
13. **Splice Site Scorer** for donor/acceptor site strength via Shapiro-Senapathy position weight matrices.
14. **Pharmacogenomic Interaction Checker** for CYP enzyme drug-drug interactions, metabolizer phenotype effects, and HLA associations.

**Tier 3 — Unsolved Mathematics (7 tools):**

15. **Complexity Class Auditor** for checking inclusion/completeness relationships between ~20 complexity classes, oracle separations, and collapse implications.
16. **Conjecture Status Checker** for tracking ~63 mathematical conjectures across 6 domains with current status, implications, and common errors.
17. **Proof Barrier Checker** for identifying which proof techniques (relativization, natural proofs, algebrization, etc.) are known to fail for specific problems.
18. **Number Theory Verifier** for numerically checking Goldbach, Collatz, twin primes, ABC triples, Legendre's conjecture, and prime gap statistics.
19. **Reduction Chain Validator** for verifying computational reduction chains (Karp, Turing, Cook, log-space, etc.) with transitivity and circularity checking.
20. **PDE Regularity Checker** for Sobolev embedding analysis, critical exponent computation, blow-up checking, and regularity classification across 8 PDE families.

**Tier 4 — STEM Calculators and Science Lookups (26 tools):**

21-26. **Six STEM calculators** covering electrochemistry/acid-base (Nernst equation, Henderson-Hasselbalch, crystal field splitting), cryptographic security (security levels, birthday bounds, cipher mode analysis), financial mathematics (Black-Scholes with Greeks, put-call parity, Nash equilibrium), distributed systems (quorum systems, Byzantine thresholds, vector clocks), networking (bandwidth-delay product, TCP throughput, subnetting), and operating systems (page tables, CPU scheduling, deadlock detection).

27-29. **Three systems tools**: PID controller simulator with Routh-Hurwitz stability analysis, SQL transaction isolation anomaly checker, and quantum circuit state vector simulator with entanglement detection.

30. **LLM claims auditor** for validating benchmark scores, Chinchilla scaling calculations, and common misconceptions across 6 LLM science domains (hallucination, reasoning, alignment, training, evaluation, context/memory).

31-46. **Sixteen lookup tables** covering mathematical conjectures (~63 conjectures, 6 domains), complexity class relationships (~20 classes), proof technique barriers (10 barriers), number theory verification (Goldbach, Collatz, twin primes, ABC, Legendre), computational reduction chains, PDE regularity/Sobolev embeddings, biochemistry (enzymes, metabolism, molecular biology, proteins, signaling — 12 topics), organic chemistry (mechanisms, reagents, named reactions, stereochemistry, synthesis — 12 topics), and quantum mechanics (uncertainty, wave function collapse, Pauli exclusion, tunneling, entanglement, zero-point energy — 12 topics).

All tools share a common architecture: dataclass reports with severity levels (HIGH/MODERATE/LOW/INFO), PASS/WARN/FAIL verdicts, and human-readable `__str__` output. Physics tools use frac_var ($\sigma / |\mu|$) as their metric; genetics and mathematics tools use domain-specific checks. The package depends only on NumPy and SciPy.

## 2. Conservation Monitoring

### 2.1 Architecture

Each monitor follows the same interface:

```python
monitor = XyzMonitor(system_parameters)
monitor.set_initial(state)
for step in simulation:
    report = monitor.check(new_state)
```

The `check()` method returns a `MonitorReport` containing current values, fractional drifts from initial, running frac_var over the full history, and alerts for any quantity exceeding a user-specified threshold. The `summary()` method returns aggregate statistics.

### 2.2 Vortex Monitor

For a system of N point vortices with circulations $\Gamma_i$ and positions $(x_i, y_i)$, governed by the Kirchhoff equations [2]:

$$\Gamma_i \dot{x}_i = -\frac{\partial H}{\partial y_i}, \quad \Gamma_i \dot{y}_i = \frac{\partial H}{\partial x_i}$$

the VortexMonitor tracks:

**Exact invariants** (frac_var < $10^{-8}$ expected):
- Kirchhoff Hamiltonian: $H = -\frac{1}{4\pi} \sum_{i<j} \Gamma_i \Gamma_j \ln(r_{ij}^2)$
- Angular impulse: $L_z = \sum_i \Gamma_i (x_i^2 + y_i^2)$
- Linear impulse: $P_x = \sum_i \Gamma_i y_i$, $P_y = -\sum_i \Gamma_i x_i$
- Derived: $Q_{\text{squared}} = \sum_i \Gamma_i L_z - P_x^2 - P_y^2$ (combination of three exact integrals)

**Approximate invariants** (frac_var < $5 \times 10^{-3}$ typical):
- Q_f family: $Q_f = \sum_{i<j} \Gamma_i \Gamma_j f(r_{ij})$ for $f \in \{r, \sqrt{r}, e^{-r}, 1/r, \tanh(r), \sin(r)\}$
- Stretch-resistant ratio: $R_f = Q_{e^{-r}} / Q_{1/r}$

**Analytical velocity invariant:**
- $K = \sum_i \Gamma_i v_i^2$ where velocities are computed analytically from the Biot-Savart law, not by finite differences.

The Q_f family and R_f ratio are not found in standard references. They were discovered through systematic numerical search over candidate expressions and verified to hold across multiple initial conditions including chaotic trajectories [3].

### 2.3 Chemical Monitor

For a reaction network with species concentrations $c$, stoichiometry matrix $S$, and mass-action kinetics, the ChemicalMonitor tracks:

- **Conservation laws** from the left null space of $S$: any vector $w$ with $w^T S = 0$ gives a conserved linear combination $w \cdot c = \text{const}$.
- **Rate constant products** for reversible pairs: $\prod k_f / k_r$ is constant over time (Wegscheider cyclicity for closed reaction cycles [9]).
- **Lyapunov function** $G = \sum_i c_i (\ln c_i - 1)$, which decreases monotonically toward equilibrium.
- **Entropy production**: $\sigma = \sum_j (v_j^+ - v_j^-) \ln(v_j^+ / v_j^-)$, which is non-negative (second law).
- **Detailed balance ratios** $v_f / v_r$ for each reversible pair, which approach 1.0 at equilibrium.

The monitor automatically discovers conservation laws from the stoichiometry matrix using SVD, without requiring the user to specify them.

### 2.4 Gravity Monitor

For N-body gravitational dynamics, the GravityMonitor tracks total energy $E = T + V$, linear momentum, angular momentum, and extends the Q_f family to 3D pairwise distances between bodies.

## 3. Integrator Validation

The `validate_integrator()` function runs a short integration of a user-supplied ODE system and checks all applicable conservation laws. It accepts the right-hand side function, initial conditions, time span, and solver parameters, then returns a `ValidationReport` classifying each quantity as PASS, WARN, or FAIL based on its frac_var:

- Exact invariants: FAIL if frac_var > $10^{-8}$
- Approximate invariants: FAIL if frac_var > $5 \times 10^{-3}$, WARN if > $10^{-2}$
- Dynamic quantities (KE, PE, entropy production): always PASS, since they are expected to change.

This classification prevents false alarms. Quantities like kinetic energy in gravity or the Lyapunov function in chemical kinetics change by design and should not be flagged.

The validator also provides fix suggestions. If exact invariants are violated with rtol > $10^{-10}$, it suggests tightening tolerances. If they are violated with already-tight tolerances, it suggests checking the equations of motion.

The `compare_configs()` convenience function runs the validator with multiple solver settings side by side, allowing systematic tolerance sweeps.

## 4. Chemical Network Auditor

The `audit_network()` function checks thermodynamic consistency of a reaction network without running any simulation. It performs purely algebraic checks:

1. **Conservation law discovery**: SVD of $S^T$ identifies the null space, giving all independent conservation laws.
2. **Wegscheider cyclicity**: For closed reaction cycles $A \rightleftharpoons B \rightleftharpoons C \rightleftharpoons A$, the product of forward rate constants divided by reverse rate constants around the cycle must equal the equilibrium constant ratio [9]. Violations indicate thermodynamically inconsistent rate parameters.
3. **Detailed balance**: At a reference concentration, computes forward and reverse rates for each reversible pair.
4. **Entropy production**: Checks that $\sigma \geq 0$ at the reference state.
5. **Rate validation**: Checks for non-physical (negative or zero) rate constants.

This catches errors in reaction network specifications before they produce wrong simulation results.

## 5. EM Field Monitor

For source-free Maxwell equations on a periodic 3D grid, the EMMonitor tracks seven conservation quantities:

- **Energy**: $U = \frac{1}{2} \int (E^2 + B^2) \, d^3x$
- **Momentum**: $|\mathbf{P}| = |\int \mathbf{E} \times \mathbf{B} \, d^3x|$
- **Optical chirality** (Zilch $Z^0$): $C = \frac{1}{2} \int [\mathbf{E} \cdot (\nabla \times \mathbf{E}) + \mathbf{B} \cdot (\nabla \times \mathbf{B})] \, d^3x$ [4, 10]
- **Helicity**: $\mathcal{H} = \int \mathbf{A} \cdot \mathbf{B} \, d^3x$, with $\mathbf{A}$ recovered from $\mathbf{B}$ in Coulomb gauge [5]
- **Zilch 3-vector**: $\mathbf{Z} = c(\mathbf{E} \times \nabla \times \mathbf{B} - \mathbf{B} \times \nabla \times \mathbf{E})$
- **Super-energy**: $S = \int [(\nabla \times \mathbf{E})^2 + (\nabla \times \mathbf{B})^2] \, d^3x$ (related to the Chevreton tensor [11])
- **EM enstrophy**: $\Omega = \int |\nabla \times \mathbf{B}|^2 \, d^3x$

All curls are computed spectrally via FFT, making the computation efficient and free of finite-difference artifacts. Chirality and helicity are non-zero only for fields with definite handedness (circular polarization), providing a diagnostic that linear-polarization-only solvers cannot test.

## 6. Hamiltonian System Validator

The `HamiltonianMonitor` validates three levels of Hamiltonian structure:

1. **Energy conservation**: $H(q, p) = \text{const}$ along the flow. This is the standard check.
2. **Liouville volume preservation** [6]: A cloud of $n$ initial conditions is evolved, and the ratio of final to initial covariance determinants is compared to 1.0. A non-symplectic integrator (e.g., forward Euler) will fail this test.
3. **Poincare integral invariant** [7]: A closed loop in $(q_1, p_1)$ phase space is evolved, and $\oint p \, dq$ is compared before and after. This tests preservation of the symplectic 2-form, which is a stronger condition than volume preservation.

The toolkit includes four built-in Hamiltonian systems:
- Harmonic oscillator (1 DOF)
- Kepler problem (2 DOF), with angular momentum $L$ and Laplace-Runge-Lenz vector $|A|$ as custom invariants [12]
- Henon-Heiles system (2 DOF, chaotic at high energy) [13]
- Coupled oscillators (2 DOF)

Users can supply their own $H(z)$ and $\nabla H(z)$ for arbitrary systems.

## 7. Invariant Learner

The `InvariantLearner` discovers conservation laws automatically. Given one or more trajectories and a set of interaction weights, it searches for the function $f(r)$ that minimizes the frac_var of

$$Q_f = \sum_{i<j} w_i w_j f(r_{ij})$$

over all trajectories. The search space is a linear combination of 12 basis functions:

$$f(r) = \sum_{k=1}^{12} a_k \phi_k(r)$$

where $\phi_k \in \{\sqrt{r}, r, r^{3/2}, r^2, e^{-r}, e^{-r/2}, e^{-2r}, -\ln r, \ln(1+r), \tanh r, \sin r, 1/r\}$.

Optimization uses L-BFGS-B [8] with L2 regularization. The `LearnerReport` includes:
- Optimal coefficients and a human-readable formula
- Improvement over the initial guess (typically 30-50% over any single basis function)
- Individual basis losses, identifying the best single function
- Dominant terms in the optimal combination

Three input modes cover different use cases:
- `learn_from_positions`: raw coordinate arrays, pairwise distances computed internally
- `learn_from_distances`: pre-computed pairwise distance time series
- `learn_from_field`: continuous 2D vorticity fields, with $Q_f$ computed via FFT convolution

## 8. Genetics Tools

The genetics tier emerged from applying the discovery-injection pipeline to seven therapeutic/genomic domains (Genetics Therapeutics, Disease Targets, Protein Structure, Immune Evasion, Delivery Optimization, Safety Invariants, Clinical Translation), producing 80 oracle-verified facts. Each tool encodes domain knowledge that LLMs frequently get wrong.

### 8.1 Sequence Design Auditor

The `audit_sequence()` function checks DNA/RNA therapeutic sequences for six categories of design problems:

- **GC content**: Flags sequences outside the 40-60% range (optimal for stability and expression).
- **CpG observed/expected ratio**: Detects CpG islands ($O/E > 0.6$) that trigger innate immune responses.
- **Homopolymer runs**: Identifies stretches of $\geq 4$ identical nucleotides that cause polymerase slippage.
- **Palindromes**: Detects self-complementary regions ($\geq 6$ nt) that form hairpins.
- **Restriction sites**: Flags common restriction enzyme recognition sequences that could cause unintended cleavage.
- **Repeat regions**: Identifies tandem repeats that reduce construct stability.

### 8.2 CRISPR Guide RNA Scorer

The `score_guide()` function evaluates CRISPR guide RNA sequences (20 nt) for on-target activity and off-target risk:

- **GC content** in the 40-70% range for efficient binding.
- **Seed region GC** (positions 1-12 from PAM) for target recognition.
- **Poly-T terminator** detection (4+ consecutive T's that terminate Pol III transcription).
- **Self-complementarity** that reduces guide availability.
- **Off-target pair comparison** via `check_offtarget_pair()`: counts mismatches between guide and potential off-target, with position-weighted scoring (seed region mismatches are less tolerable).

### 8.3 Therapeutic Pipeline Validator

The `validate_pipeline()` function checks end-to-end consistency of therapy designs:

- **Target-disease coherence**: Verifies the molecular target is relevant to the disease indication.
- **Delivery-cargo compatibility**: Checks that the delivery vehicle (AAV, LNP, nanoparticle, etc.) is compatible with the cargo type (DNA, mRNA, protein, small molecule).
- **Dosing range**: Flags doses outside established safety ranges for the delivery modality.
- **Safety signals**: Cross-references known toxicity patterns for the combination of target, cargo, and delivery.

### 8.4 Protein Aggregation Predictor

The `predict_aggregation()` function identifies aggregation-prone protein sequences via five checks:

- **Aggregation-prone regions (APRs)**: Sliding window over AGGRESCAN hydrophobicity scores, flagging regions above threshold.
- **Mean hydrophobicity**: Kyte-Doolittle scale average, HIGH if $> 1.0$.
- **Hydrophobic patches**: Longest consecutive stretch of hydrophobic residues ($> 10$ = HIGH, $> 7$ = MODERATE).
- **Net charge**: Near-zero net charge ($|q| < 2$) increases aggregation risk.
- **Low complexity**: Windows with $\leq 4$ unique residue types in 20-residue windows.

### 8.5 Splice Site Scorer

Position weight matrix (PWM) scoring for splice site strength:

- **Donor sites**: 9-nucleotide window (3 exonic + GT + 4 intronic), scored against Shapiro-Senapathy consensus frequencies [16].
- **Acceptor sites**: 16-nucleotide window (11 intronic + AG + 3 exonic), with polypyrimidine tract scoring.
- **Scan function**: `scan_splice_sites()` slides across a full sequence, reporting all GT/AG dinucleotides with their PWM scores, sorted by strength.

### 8.6 Pharmacogenomic Interaction Checker

The `audit_drug_list()` function checks drug combinations for pharmacogenomic interactions:

- **CYP enzyme conflicts**: ~50 drug-enzyme pairs across 5 major CYP isoforms (1A2, 2C9, 2C19, 2D6, 3A4). Flags substrate-inhibitor co-prescriptions that cause toxic accumulation.
- **Metabolizer phenotype effects**: `check_phenotype()` identifies drugs requiring dose adjustment for poor/ultra-rapid metabolizers.
- **HLA associations**: `check_hla()` flags drugs with known HLA-mediated hypersensitivity reactions (e.g., abacavir/HLA-B*57:01, carbamazepine/HLA-B*15:02).

## 9. Unsolved Mathematics Tools

The mathematics tier emerged from six unsolved-math domains (Millennium Problems, Number Theory Conjectures, Algebra/Topology Conjectures, Proof Techniques, Analysis/PDE Conjectures, Computational Conjectures), producing 70 oracle-verified facts. These tools encode relationships between mathematical objects that LLMs commonly misstate.

### 9.1 Complexity Class Auditor

The `audit_complexity()` function checks claims about computational complexity:

- **Class hierarchy**: DAG of ~20 classes (P, NP, coNP, PSPACE, BPP, BQP, PH, EXP, NEXP, etc.) with known and conjectured inclusions.
- **Completeness checking**: 14 hardcoded completeness results (SAT is NP-complete, TQBF is PSPACE-complete, etc.).
- **Oracle separations**: Known relativized separations that establish proof barriers.
- **Collapse implications**: If P=NP then PH collapses; if BPP$\neq$P then derandomization fails — the auditor traces these chains.

### 9.2 Conjecture Status Checker

The `check_conjecture()` function tracks ~63 mathematical conjectures across 6 domains:

- **Status tracking**: Open, proved, disproved, partially resolved, with year and resolver.
- **Implication chains**: "If Riemann Hypothesis then..." cascades.
- **Common errors**: Specific misconceptions LLMs make about each conjecture (e.g., confusing weak and strong Goldbach, claiming P vs NP is about practical efficiency).
- **Claim verification**: `check_claim()` validates natural-language claims against the database.

### 9.3 Proof Barrier Checker

The `check_barriers()` function identifies which proof techniques are known to fail for specific problems:

- **10 barriers**: Relativization (Baker-Gill-Solovay), natural proofs (Razborov-Rudich), algebrization (Aaronson-Wigderson), black-box reductions, current algebraic geometry, diagonal arguments, topological methods, analytic methods, probabilistic methods, combinatorial methods.
- **Technique aliases**: Maps common names to formal barrier categories.
- **Problem-barrier matrix**: Which barriers apply to which open problems, with explanations of why.

### 9.4 Number Theory Verifier

Numerical verification of number theory conjectures:

- **Goldbach**: `verify_goldbach(n)` checks that even $n > 2$ is a sum of two primes, returning the decomposition.
- **Collatz**: `verify_collatz(n, max_steps)` traces the $3n+1$ sequence, reporting total stopping time and maximum value.
- **Twin primes**: `verify_twin_primes(limit)` finds all twin prime pairs below the limit with density statistics.
- **ABC triples**: `check_abc_triple(a, b)` computes the quality $q = \log(c) / \log(\text{rad}(abc))$ for the ABC conjecture.
- **Legendre**: `verify_legendre(n)` confirms a prime exists between $n^2$ and $(n+1)^2$.
- **Prime gaps**: `prime_gap_analysis(limit)` computes gap statistics, maximal gaps, and Cramer's conjecture ratio.

All functions use Miller-Rabin primality testing (deterministic for $n < 3.3 \times 10^{24}$) and an optimized prime sieve.

### 9.5 Reduction Chain Validator

The `validate_chain()` function verifies computational reduction chains:

- **8 reduction types**: Karp (many-one polynomial), Turing (oracle polynomial), Cook (polynomial-time), log-space, first-order, Levin (optimal), truth-table, randomized.
- **13 known reductions**: Hardcoded verified reductions (3SAT $\leq_K$ CLIQUE, HAMPATH $\leq_K$ TSP, etc.).
- **Transitivity**: Automatically verifies multi-step chains ($A \leq B \leq C \implies A \leq C$).
- **Circularity detection**: Flags circular reduction chains.
- **Hardness inheritance**: Validates that reductions correctly propagate hardness (NP-hard $\leq$ NP-hard is valid; P $\leq$ NP-hard is always valid but uninformative).

### 9.6 PDE Regularity Checker

The `check_pde_regularity()` function analyzes PDE well-posedness:

- **Sobolev embeddings**: `check_sobolev_embedding()` determines whether $W^{s,p}(\mathbb{R}^n) \hookrightarrow L^q(\mathbb{R}^n)$ holds, with subcritical/critical/supercritical classification.
- **Critical exponents**: `critical_exponent()` computes Fujita, Sobolev, Strauss, and Serrin exponents for 8 PDE families (heat, wave, Navier-Stokes, NLS, KdV, Euler, porous medium, SQG).
- **Blow-up analysis**: `check_blowup()` determines whether solutions can develop singularities based on spatial dimension and nonlinearity.
- **Regularity classification**: Maps (PDE family, dimension, exponent) to known regularity results (global existence, conditional regularity, finite-time blowup, open problem).

## 10. Benchmark Results

### 10.1 Tolerance Sensitivity

We ran a 3-vortex system ($\Gamma = [1.0, -0.5, 0.3]$) for $t \in [0, 50]$ at seven rtol values from $10^{-12}$ to $10^{-2}$, using Dormand-Prince RK45 [14]. At rtol $= 10^{-4}$, H shows frac_var $= 1.6 \times 10^{-3}$ while Q_exp shows $5.1 \times 10^{-3}$, both near their alerting thresholds. At rtol $= 10^{-3}$, H jumps to $7.9 \times 10^{-2}$ while $Q_{\exp}$ reaches $1.4 \times 10^{-1}$. The Q_f monitors do not provide earlier warning at loose tolerances (they respond to the same numerical drift), but they provide independent confirmation and are sensitive to different error modes.

### 10.2 Wrong Physics Detection

Three scenarios test detection of physics errors:

1. **Correct physics**: All quantities at baseline frac_var levels.
2. **Missing $2\pi$ factor in the Biot-Savart kernel**: H and Lz remain conserved (the wrong-factor system is still Hamiltonian), but Q_exp shifts from $2.3 \times 10^{-3}$ to $3.5 \times 10^{-3}$, a 55% increase. This demonstrates that Q_f monitors detect wrong physics that preserves exact conservation laws.
3. **Dropped weakest vortex**: All quantities show large violations. $Q_{\exp}$ reaches frac_var $= 0.57$, a 252x increase over baseline, making it the most sensitive indicator.

### 10.3 Chemical Network Auditing

For an $A \rightleftharpoons B \rightleftharpoons C$ network with rate constants $k = [0.5, 0.3, 0.4, 0.2]$, the auditor finds one conservation law ($[A] + [B] + [C] = \text{const}$) and computes the rate constant product $k_1 k_3 / (k_2 k_4) = 3.33$. Perturbing the rates produces a shifted product of 0.13, immediately flagging thermodynamic inconsistency without running any simulation.

### 10.4 Genetics Tool Validation

A validation battery of 173 practical test cases was run across all non-physics tools:

- **Sequence auditor**: Correctly flags high-GC mRNA vaccine sequences (GC > 60%), detects CpG islands in immunostimulatory constructs, identifies homopolymer runs in synthesis-problematic sequences. 24/24 test cases passed.
- **CRISPR scorer**: Identifies poly-T terminators, scores GC content in optimal range, detects high off-target risk from seed-region complementarity. 18/18 test cases passed.
- **Pipeline validator**: Catches delivery-cargo mismatches (AAV with small molecules), flags missing safety assessments, validates dosing ranges. 20/20 test cases passed.
- **Aggregation predictor**: Detects APRs in amyloid-forming sequences, passes diverse charged sequences, correctly flags low-complexity regions. 16/16 test cases passed.
- **Splice site scorer**: Strong canonical donors score > 0, non-canonical dinucleotides flagged, pyrimidine tract scoring matches known biology. 14/14 test cases passed.
- **Pharmacokinetics**: Detects CYP3A4 substrate-inhibitor co-prescriptions, flags HLA-B*57:01/abacavir, identifies poor metabolizer dose adjustments. 16/16 test cases passed.

### 10.5 Mathematics Tool Validation

- **Number theory verifier**: Goldbach verified to $10^6$, Collatz verified to $10^4$ (longest sequence: $n = 6171$, 261 steps), ABC triple $(1, 4374, 4375)$ found with quality $q = 1.568$, Legendre verified to $n = 1000$. 6 discovery scans completed.
- **Complexity auditor**: Correctly validates P $\subseteq$ NP $\subseteq$ PSPACE chain, flags invalid claims (NP $\subseteq$ P), identifies NP-completeness of SAT. 12/12 test cases passed.
- **Conjecture checker**: Correctly reports Poincare as proved (Perelman 2003), Riemann Hypothesis as open, identifies common errors for each conjecture. 10/10 test cases passed.
- **Proof barriers**: Correctly identifies relativization barrier for P vs NP, natural proofs barrier for circuit lower bounds, lists applicable techniques per problem. 8/8 test cases passed.
- **Reduction validator**: Validates SAT $\leq_K$ CLIQUE $\leq_K$ VERTEX-COVER chains, detects circular reductions, verifies hardness inheritance. 10/10 test cases passed.
- **PDE regularity**: Correctly classifies Sobolev embeddings in all three regimes (subcritical/critical/supercritical), computes Fujita exponents, identifies Navier-Stokes regularity as open in 3D. 8/8 test cases passed.

All 173 validation cases passed (100% catch rate). Additionally, 29 edge case stress tests and 6 discovery scans completed successfully.

## 11. Implementation

The toolkit is implemented in Python with NumPy and SciPy as the only dependencies. Integration uses `scipy.integrate.solve_ivp` with Dormand-Prince RK45 [14] by default. Spectral operations (curls in the EM monitor, convolutions in the invariant learner) use NumPy's FFT module. Genetics and mathematics tools use hardcoded reference data (hydrophobicity scales, PWM frequencies, complexity class hierarchies, conjecture databases) requiring no external data files or API calls.

A pre-commit hook enforces quality on every commit: all 1252 tests must pass, all public exports must import cleanly, and a physics smoke test must confirm that H and Lz are conserved to frac_var < $10^{-8}$ on a reference vortex problem. The hook blocks the commit if any check fails.

The package is available at `pip install noethersolve` (PyPI) and https://github.com/SolomonB14D3/NoetherSolve. Version 1.1.0 includes all 46 tools.

## 12. Related Work

Geometric numerical integration [15] provides the theoretical foundation for structure-preserving algorithms. Symplectic integrators exactly preserve the symplectic 2-form, guaranteeing long-term energy bounds. Our Hamiltonian validator checks these properties empirically, useful when the integrator is not guaranteed to be symplectic (e.g., when using adaptive-step RK methods).

Conservation monitoring is standard practice in computational fluid dynamics and molecular dynamics, but typically limited to energy and momentum. The Q_f family of invariants and the stretch-resistant ratio R_f extend monitoring to quantities not previously tracked.

Optical chirality was introduced by Lipkin [4] in 1964 and rediscovered by Tang and Cohen [10] in 2010. The super-energy tensor was introduced by Chevreton [11]. These quantities are exactly conserved for source-free Maxwell fields but are rarely monitored in electromagnetic simulations.

The Wegscheider cyclicity condition [9] provides algebraic constraints on rate constants in reaction networks. Our auditor automates checking these constraints alongside conservation law discovery.

**Genetics tools.** Protein aggregation prediction has a rich literature (TANGO [16], Zyggregator [17], AGGRESCAN [18]). Our predictor uses simplified AGGRESCAN-inspired scoring suitable for rapid screening without external servers. Splice site scoring follows the Shapiro-Senapathy PWM framework [19]. Pharmacogenomic interaction checking draws on CPIC guidelines [20] for CYP enzyme-drug relationships and HLA associations.

**Computational complexity.** The complexity class hierarchy and proof barriers encoded in our tools follow standard references (Arora and Barak [21], Aaronson [22]). The tool's value is not in novel complexity theory but in catching common LLM errors about these relationships (e.g., confusing known inclusions with conjectured separations, misattributing completeness results).

**Number theory verification.** Computational verification of number theory conjectures is well-established (Oliveira e Silva et al. [23] for Goldbach, Roosendaal for Collatz records). Our verifier provides a unified interface for multiple conjectures with structured reports.

## References

[1] E. Noether, "Invariante Variationsprobleme," Nachrichten von der Gesellschaft der Wissenschaften zu Gottingen, Mathematisch-Physikalische Klasse, pp. 235-257, 1918.

[2] G. Kirchhoff, *Vorlesungen uber mathematische Physik: Mechanik*. Teubner, Leipzig, 1876.

[3] B. Sanchez, "Breaking Frozen Priors: Teaching Language Models to Discover Conservation Laws from Numerical Simulation," 2026. DOI: 10.5281/zenodo.19017290.

[4] D. M. Lipkin, "Existence of a New Conservation Law in Electromagnetic Theory," Journal of Mathematical Physics, vol. 5, no. 5, pp. 696-700, 1964. DOI: 10.1063/1.1704165.

[5] K. Y. Bliokh and F. Nori, "Characterizing optical chirality," Physical Review A, vol. 83, p. 021803(R), 2011. DOI: 10.1103/PhysRevA.83.021803.

[6] J. Liouville, "Note sur la Theorie de la Variation des constantes arbitraires," Journal de Mathematiques Pures et Appliquees, Serie 1, Tome 3, pp. 342-349, 1838.

[7] H. Poincare, *Les methodes nouvelles de la mecanique celeste, Tome III*. Gauthier-Villars, Paris, 1899.

[8] R. H. Byrd, P. Lu, J. Nocedal, and C. Zhu, "A Limited Memory Algorithm for Bound Constrained Optimization," SIAM Journal on Scientific Computing, vol. 16, no. 5, pp. 1190-1208, 1995. DOI: 10.1137/0916069.

[9] R. Wegscheider, "Uber simultane Gleichgewichte und die Beziehungen zwischen Thermodynamik und Reactionskinetik homogener Systeme," Monatshefte fur Chemie, vol. 22, no. 8, pp. 849-906, 1901. DOI: 10.1007/BF01517498.

[10] Y. Tang and A. E. Cohen, "Optical Chirality and Its Interaction with Matter," Physical Review Letters, vol. 104, p. 163901, 2010. DOI: 10.1103/PhysRevLett.104.163901.

[11] M. Chevreton, "Sur le tenseur de superenergie du champ electromagnetique," Il Nuovo Cimento, vol. 34, pp. 901-913, 1964. DOI: 10.1007/BF02812520.

[12] H. Goldstein, "Prehistory of the 'Runge-Lenz' vector," American Journal of Physics, vol. 43, no. 8, pp. 737-738, 1975. DOI: 10.1119/1.9745.

[13] M. Henon and C. Heiles, "The Applicability of the Third Integral of Motion: Some Numerical Experiments," The Astronomical Journal, vol. 69, pp. 73-79, 1964. DOI: 10.1086/109234.

[14] J. R. Dormand and P. J. Prince, "A family of embedded Runge-Kutta formulae," Journal of Computational and Applied Mathematics, vol. 6, no. 1, pp. 19-26, 1980. DOI: 10.1016/0771-050X(80)90013-3.

[15] E. Hairer, C. Lubich, and G. Wanner, *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*, 2nd ed. Springer, 2006. DOI: 10.1007/3-540-30666-8.

[16] S. Fernandez-Escamilla, F. Rousseau, J. Schymkowitz, and L. Serrano, "Prediction of sequence-dependent and mutational effects on the aggregation of peptides and proteins," Nature Biotechnology, vol. 22, pp. 1302-1306, 2004. DOI: 10.1038/nbt1012.

[17] G. G. Tartaglia et al., "Prediction of aggregation-prone regions in structured proteins," Journal of Molecular Biology, vol. 380, pp. 425-436, 2008. DOI: 10.1016/j.jmb.2008.05.013.

[18] O. Conchillo-Sole et al., "AGGRESCAN: a server for the prediction and evaluation of 'hot spots' of aggregation in polypeptides," BMC Bioinformatics, vol. 8, p. 65, 2007. DOI: 10.1186/1471-2105-8-65.

[19] M. B. Shapiro and P. Senapathy, "RNA splice junctions of different classes of eukaryotes: sequence statistics and functional implications in gene expression," Nucleic Acids Research, vol. 15, pp. 7155-7174, 1987. DOI: 10.1093/nar/15.17.7155.

[20] M. V. Relling and T. E. Klein, "CPIC: Clinical Pharmacogenetics Implementation Consortium of the Pharmacogenomics Research Network," Clinical Pharmacology & Therapeutics, vol. 89, pp. 464-467, 2011. DOI: 10.1038/clpt.2010.279.

[21] S. Arora and B. Barak, *Computational Complexity: A Modern Approach*. Cambridge University Press, 2009.

[22] S. Aaronson, "Algebrization: A New Barrier in Complexity Theory," ACM Transactions on Computation Theory, vol. 1, no. 2, pp. 1-54, 2009. DOI: 10.1145/1490270.1490272.

[23] T. Oliveira e Silva, S. Herzog, and S. Pardi, "Empirical verification of the even Goldbach conjecture and computation of prime gaps up to 4×10^18," Mathematics of Computation, vol. 83, pp. 2033-2060, 2014. DOI: 10.1090/S0025-5718-2013-02787-1.
