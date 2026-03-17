# D1 Source Numbers — All Data for Paper

## Core Result: Q_f = Sigma_{i<j} Gamma_i Gamma_j f(r_ij) is approximately conserved

### 2D Point Vortex: frac_var by f(r) and N

| N | Configuration | f=r | f=r^2 | f=sqrt(r) | f=ln(r) |
|---|---------------|-----|-------|-----------|---------|
| 3 | Restricted | 2e-10 | EXACT | 3e-11 | EXACT |
| 3 | Equal | 3e-8 | EXACT | 4e-9 | EXACT |
| 4 | Generic (chaotic) | 2e-6 | EXACT | 3e-7 | EXACT |
| 5 | Hierarchical | 2e-7 | EXACT | 3e-8 | EXACT |
| 6 | Hierarchical | 2e-6 | EXACT | 3e-7 | EXACT |
| 7 | Hierarchical | 1e-6 | EXACT | 2e-7 | EXACT |
| 8 | Dipole array | 1e-4 | 3e-6 | 1e-6 | EXACT |

### Optimal Power: Q_n = Sigma Gamma_i Gamma_j r_ij^n

| n | frac_var | Status |
|---|----------|--------|
| -2.0 | 5e-8 | PASS |
| -1.0 | 2e-9 | PASS |
| -0.5 | 8e-11 | PASS |
| 0.0 | 5e-32 | TRIVIAL |
| 0.5 | 3e-11 | BEST NON-TRIVIAL |
| 1.0 | 2e-10 | PASS |
| 1.5 | 2e-10 | PASS |
| 2.0 | 1e-21 | EXACT (= Gamma_tot * Lz) |
| 3.0 | 2e-8 | PASS |
| 4.0 | 2e-7 | PASS |

### Exact Cases
- n=0: Q_0 = Sigma Gamma_i Gamma_j = const (trivial, function of circulations only)
- n=2: Q_2 = Sigma Gamma_i Gamma_j r_ij^2 = Gamma_total * Lz + const (R^2 = 0.9999999999)
- f=ln(r): Q_ln = Sigma Gamma_i Gamma_j ln(r_ij) proportional to H (Hamiltonian)

### Scaling with epsilon (restricted 3-vortex, Gamma_1=Gamma_2=1, Gamma_3=epsilon)
frac_var(Q_1) proportional to epsilon^1.44

### 3D Vortex Ring Conservation (two coaxial rings, Biot-Savart)

| f(r) | frac_var | Rank |
|------|----------|------|
| 1/r | 3.78e-04 | 1st (BEST) |
| e^(-r) | 1.79e-03 | 2nd |
| sqrt(r) | 2.95e-03 | 3rd |
| e^(-r^2/2) | 3.64e-03 | 4th |
| r | 4.36e-03 | 5th |

### 3D Stretching + Lateral Motion: Growth Ratio

| f(r) | Growth Ratio |
|------|-------------|
| r | 41.1x (worst) |
| sqrt(r) | 27.7x |
| 1/r | 8.9x (best) |
| e^(-r) | 5.6x |

### 3D Alignment Weighting: Q_f^p = integral |T_i.T_j|^p f(r) ds dt

| f(r) | p=0 | p=1 | p=2 |
|------|-----|-----|-----|
| 1/r | 3.78e-04 | 3.53e-04 | 3.36e-04 |
| e^(-r) | 1.79e-03 | 1.93e-03 | 2.01e-03 |

### Optimal Combination (L-BFGS-B over 12 basis functions)

| Configuration | Loss (frac_var sum) | Improvement |
|--------------|---------------------|-------------|
| Best single f(r) = sqrt(r) | 0.00159 | baseline |
| Best single f(r) = e^(-r) | 0.00129 | 19% |
| Optimal combination | 0.000005 | 99.6% (300x better) |

### Viscous Decay (2D Navier-Stokes, grid 128x128, T=5.0)

| f(r) | Exponent alpha (rel_change proportional to nu^alpha) | CV | R^2 |
|------|-----------------------------------------------------|-----|-----|
| sqrt(r) | 0.99 | 5.6% | 0.9982 |
| e^(-r) | 0.82 | 23% | — |
| -ln(r) | 0.59 | 46% | — |
| tanh(r) | 1.76 | 58% | — |

Measured decay constant for sqrt(r): C approx 7
Lower bound: Q_sqrt(r)(t) >= Q_sqrt(r)(0) * exp(-7*nu*t)

### Concentration Detection (Gaussian vortex width sigma)

| f(r) | Scaling exponent alpha (Q proportional to sigma^alpha) | Type |
|------|-------------------------------------------------------|------|
| -ln(r) | -0.63 | DETECTING (grows) |
| 1/sqrt(r) | -0.47 | DETECTING |
| sqrt(r) | +0.49 | BLIND (shrinks) |
| r | +0.99 | BLIND |

### Dichotomy: Concentration vs Stretch-Resistance

Stretch-resistant: f(r) ~ r^p with p >= 0
Concentration-detecting: f(r) -> infinity as r -> 0

Curvature-weighted hybrid Q_kappa,1/sqrt(r):
- Helical vortex filaments, stretch factor 4.0
- frac_var: 0.28 (vs 0.60 for Q_kappa,r) -> 2x better

### Negative Result: Triplet Invariants
T_full = Sigma_{all i,j,k} Gamma_i Gamma_j Gamma_k * Area(ijk) = identically zero by symmetry
T_ordered = Sigma_{i<j<k} Gamma_i Gamma_j Gamma_k * Area(ijk) is NOT conserved (frac_var ~ 0.5 to 100)
Pairwise structure is special, no genuine higher-order invariants found.

### Green's Function Principle (universal)
| Dimension | Green's Function | Best Q_f | Physical Meaning |
|-----------|-----------------|----------|-----------------|
| 2D | G(r) = -ln(r)/(2pi) | Q_{-ln(r)} | proportional to Kinetic energy |
| 3D | G(r) = 1/(4pi*r) | Q_{1/r} | proportional to Kinetic energy |

Conjecture: In any dimension d, optimal Q_f uses f(r) = G_d(r).
