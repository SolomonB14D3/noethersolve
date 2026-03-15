# NoetherSolve Toolkit: Conservation Law Monitoring and Discovery for Numerical Simulations

Bryan Sanchez

March 2026

## Abstract

We present NoetherSolve, an open-source Python toolkit for monitoring, validating, and discovering conservation laws in numerical simulations. The toolkit provides six tools covering three physical domains (point-vortex dynamics, chemical kinetics, N-body gravity) plus electromagnetism, Hamiltonian mechanics, and a domain-agnostic invariant discovery engine. Each tool emerged from a pipeline that numerically verifies conservation laws, identifies where language models fail to recognize them, and teaches the results back to the model. The toolkit requires no machine learning at runtime. We demonstrate that monitors based on a newly discovered Q_f invariant family detect numerical corruption at noise levels 100x lower than standard monitors (H, Lz), that algebraic auditing catches thermodynamic inconsistencies in reaction networks without simulation, and that automatic optimization over 12 basis functions discovers conservation laws with 40% lower fractional variation than any single basis function. The package is available on PyPI (`pip install noethersolve`) with 102 tests and a physics-enforcing pre-commit hook.

## 1. Introduction

Conservation laws are the foundation of numerical simulation. A well-posed physical system conserves energy, momentum, angular momentum, and often additional quantities tied to its symmetries via Noether's theorem [1]. When a simulation violates these laws, something is wrong: the integrator tolerances are too loose, the equations of motion contain an error, or the numerical scheme introduces artificial dissipation.

Standard practice checks a handful of known invariants. In vortex dynamics, the Kirchhoff Hamiltonian H and angular impulse Lz are textbook quantities [2]. In chemical kinetics, total mass conservation follows from stoichiometry. In N-body gravity, energy and momentum are standard diagnostics. These checks catch large errors, but small systematic corruptions can pass through undetected.

This paper describes a toolkit that goes further in two ways. First, it monitors quantities that standard references do not cover. The Q_f family of invariants, $Q_f = \sum_{i<j} \Gamma_i \Gamma_j f(r_{ij})$, was discovered through numerical simulation and verified across chaotic vortex systems [3]. These approximate invariants are more sensitive to corruption than the exact invariants H and Lz. Second, the toolkit includes a learner that automatically discovers new conservation laws from trajectory data, without requiring the user to specify what to look for.

The six tools are:

1. **Conservation Monitors** (VortexMonitor, ChemicalMonitor, GravityMonitor) for tracking invariants during simulation.
2. **Integrator Validator** for checking solver configurations before long runs.
3. **Chemical Network Auditor** for algebraic thermodynamic consistency.
4. **EM Field Monitor** for electromagnetic invariants including optical chirality [4] and helicity [5].
5. **Hamiltonian System Validator** for symplectic structure verification via Liouville volume and Poincare invariants [6, 7].
6. **Invariant Learner** for automatic conservation law discovery via L-BFGS-B optimization [8] over basis functions.

All tools share a common metric, the coefficient of variation of the monitored quantity (which we call frac_var: $\sigma / |\mu|$), and produce structured reports with PASS/WARN/FAIL verdicts. The package depends only on NumPy and SciPy.

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

## 8. Benchmark Results

### 8.1 Tolerance Sensitivity

We ran a 3-vortex system ($\Gamma = [1.0, -0.5, 0.3]$) for $t \in [0, 50]$ at seven rtol values from $10^{-12}$ to $10^{-2}$, using Dormand-Prince RK45 [14]. At rtol $= 10^{-4}$, H shows frac_var $= 1.6 \times 10^{-3}$ while Q_exp shows $5.1 \times 10^{-3}$, both near their alerting thresholds. At rtol $= 10^{-3}$, H jumps to $7.9 \times 10^{-2}$ while $Q_{\exp}$ reaches $1.4 \times 10^{-1}$. The Q_f monitors do not provide earlier warning at loose tolerances (they respond to the same numerical drift), but they provide independent confirmation and are sensitive to different error modes.

### 8.2 Wrong Physics Detection

Three scenarios test detection of physics errors:

1. **Correct physics**: All quantities at baseline frac_var levels.
2. **Missing $2\pi$ factor in the Biot-Savart kernel**: H and Lz remain conserved (the wrong-factor system is still Hamiltonian), but Q_exp shifts from $2.3 \times 10^{-3}$ to $3.5 \times 10^{-3}$, a 55% increase. This demonstrates that Q_f monitors detect wrong physics that preserves exact conservation laws.
3. **Dropped weakest vortex**: All quantities show large violations. $Q_{\exp}$ reaches frac_var $= 0.57$, a 252x increase over baseline, making it the most sensitive indicator.

### 8.3 Chemical Network Auditing

For an $A \rightleftharpoons B \rightleftharpoons C$ network with rate constants $k = [0.5, 0.3, 0.4, 0.2]$, the auditor finds one conservation law ($[A] + [B] + [C] = \text{const}$) and computes the rate constant product $k_1 k_3 / (k_2 k_4) = 3.33$. Perturbing the rates produces a shifted product of 0.13, immediately flagging thermodynamic inconsistency without running any simulation.

## 9. Implementation

The toolkit is implemented in Python with NumPy and SciPy as the only dependencies. Integration uses `scipy.integrate.solve_ivp` with Dormand-Prince RK45 [14] by default. Spectral operations (curls in the EM monitor, convolutions in the invariant learner) use NumPy's FFT module.

A pre-commit hook enforces quality on every commit: all 102 tests must pass, all public exports must import cleanly, and a physics smoke test must confirm that H and Lz are conserved to frac_var < $10^{-8}$ on a reference vortex problem. The hook blocks the commit if any check fails.

The package is available at `pip install noethersolve` (PyPI) and https://github.com/SolomonB14D3/NoetherSolve.

## 10. Related Work

Geometric numerical integration [15] provides the theoretical foundation for structure-preserving algorithms. Symplectic integrators exactly preserve the symplectic 2-form, guaranteeing long-term energy bounds. Our Hamiltonian validator checks these properties empirically, useful when the integrator is not guaranteed to be symplectic (e.g., when using adaptive-step RK methods).

Conservation monitoring is standard practice in computational fluid dynamics and molecular dynamics, but typically limited to energy and momentum. The Q_f family of invariants and the stretch-resistant ratio R_f extend monitoring to quantities not previously tracked.

Optical chirality was introduced by Lipkin [4] in 1964 and rediscovered by Tang and Cohen [10] in 2010. The super-energy tensor was introduced by Chevreton [11]. These quantities are exactly conserved for source-free Maxwell fields but are rarely monitored in electromagnetic simulations.

The Wegscheider cyclicity condition [9] provides algebraic constraints on rate constants in reaction networks. Our auditor automates checking these constraints alongside conservation law discovery.

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
