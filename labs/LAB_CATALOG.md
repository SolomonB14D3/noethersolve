# NoetherSolve Lab Catalog

14 autonomous research labs that chain NoetherSolve tools for domain-specific analysis.

---

## Medical / Clinical Labs

### Drug Therapy Lab
**File:** `drug_therapy_lab.py`

**What it does:** Screens drug candidates for pharmacokinetic viability by computing therapeutic index, drug-drug interaction risk, and metabolic stability.

**Practical use:**
- Teaching PK/PD concepts with real drug examples
- Preliminary screening before wet-lab work
- Identifying potential drug interaction concerns

**Data sources:**
- **ChEMBL** (chembl.gitbook.io) — ADMET data, t½, Vd, clearance
- **PubChem** (pubchem.ncbi.nlm.nih.gov) — molecular properties
- **FDA Labels** (curated subset) — therapeutic windows

**Usage:**
```bash
python labs/drug_therapy_lab.py --top 50       # Screen top 50 drugs from ChEMBL
python labs/drug_therapy_lab.py --drugs warfarin metformin  # Specific drugs
python labs/drug_therapy_lab.py --offline      # Use built-in reference data
```

**⚠️ DISCLAIMER:** This tool is for **educational and research purposes only**. It does NOT provide medical advice. Drug dosing, interactions, and therapeutic decisions must be made by licensed healthcare providers using validated clinical resources. Never use this tool to make treatment decisions.

---

### Genetic Therapeutics Lab
**File:** `genetic_therapeutics_lab.py`

**What it does:** Evaluates genetic therapy candidates across three modalities:
- **CRISPR guides:** On-target activity, off-target risk, PAM compatibility
- **mRNA therapeutics:** Codon optimization, stability, immunogenicity
- **Neoantigens:** MHC binding, TAP transport, TCR recognition

**Practical use:**
- Prioritizing guide RNA candidates before synthesis
- Evaluating mRNA construct designs
- Neoantigen vaccine candidate selection

**Data sources:**
- Built-in scoring matrices from literature
- Future: Integration with Ensembl for gene sequences

**⚠️ DISCLAIMER:** This tool provides **computational predictions only**. CRISPR off-target analysis requires empirical validation (GUIDE-seq, CIRCLE-seq). mRNA and neoantigen candidates require experimental testing. Do not use for clinical applications without proper validation.

---

### Epidemic Modeling Lab
**File:** `epidemic_lab.py`

**What it does:** SIR epidemic modeling with intervention analysis:
- Basic reproduction number (R₀) and doubling time
- Herd immunity thresholds
- Vaccine coverage requirements
- NPI (non-pharmaceutical intervention) scenarios

**Practical use:**
- Teaching epidemiology concepts
- Scenario planning for public health
- Comparing disease controllability

**Data sources:**
- **Built-in:** R₀ estimates from WHO/CDC literature (measles, COVID variants, influenza, Ebola)
- **Future:** Integration with GISAID or outbreak.info

**Usage:**
```bash
python labs/epidemic_lab.py                   # Run all scenarios
python labs/epidemic_lab.py --verbose         # Detailed output
```

**⚠️ DISCLAIMER:** SIR models are **simplified approximations**. Real epidemics involve heterogeneous mixing, spatial structure, behavioral changes, and data uncertainty. Do not use for actual public health decision-making without epidemiological expertise.

---

## Materials Science Labs

### Catalyst Discovery Lab
**File:** `catalyst_lab.py`

**What it does:** Screens transition metal catalysts for electrochemical reactions (HER, OER, ORR) using:
- d-band center theory (electronic structure)
- Volcano plot positioning (Sabatier principle)
- BEP activation energy barriers

**Practical use:**
- Guiding catalyst material selection
- Understanding activity-cost tradeoffs
- Teaching heterogeneous catalysis

**Data sources:**
- **Built-in:** d-band centers from DFT calculations (Hammer-Nørskov)
- **Future:** Materials Project API for computed surface energies

**Output example:**
```
Catalyst rankings for HER:
  #1  Pt  (volcano apex, but cost rank 8/8)
  #2  Ni  (off-apex but 100x cheaper)
```

---

### Topological Materials Lab
**File:** `topological_lab.py`

**What it does:** Classifies materials/model systems by topological invariants:
- Chern numbers (quantum Hall, Chern insulators)
- Z₂ invariants (topological insulators)
- Berry phase (Zak phase for 1D systems)
- Symmetry class lookup (Altland-Zirnbauer periodic table)

**Practical use:**
- Teaching topological band theory
- Classifying candidate topological materials
- Understanding bulk-boundary correspondence

**Data sources:**
- **Built-in:** Periodic table of topological invariants
- **Future:** Integration with Topological Materials Database

---

### Battery Materials Lab
**File:** `battery_materials_lab.py`

**What it does:** Analyzes battery degradation mechanisms:
- Calendar aging (storage capacity loss)
- Cycle aging (charge/discharge degradation)
- Chemistry comparison (NMC, LFP, NCA, solid-state)

**Practical use:**
- Battery lifetime estimation
- Chemistry selection for applications
- Teaching electrochemical aging

**Data sources:**
- **Built-in:** Aging parameters from literature (Saft, Argonne studies)

---

## Climate & Physics Labs

### Climate Sensitivity Lab
**File:** `climate_lab.py`

**What it does:** Radiative forcing and climate sensitivity analysis:
- CO₂ forcing calculations (logarithmic relationship)
- Feedback analysis (water vapor, ice-albedo, cloud)
- Temperature response under different scenarios

**Practical use:**
- Teaching climate physics
- Exploring sensitivity to feedback assumptions
- Understanding IPCC scenario ranges

**Data sources:**
- **Built-in:** IPCC AR6 feedback estimates
- Scenario definitions: 280, 421, 560, 1120 ppm CO₂

**⚠️ NOTE:** Uses simplified 0D/1D models. For actual climate projections, use full GCMs (CMIP6 models).

---

### Conservation Law Mining Lab
**File:** `conservation_mining_lab.py`

**What it does:** Discovers approximate conservation laws in dynamical systems:
- Point vortex dynamics
- N-body gravitational systems
- Chemical reaction networks

**Practical use:**
- Research: discovering new invariants
- Teaching: demonstrating Noether's theorem
- Validation: checking numerical integrators

**Data sources:**
- Initial conditions from classical test problems (figure-8 three-body, restricted vortex problems)

---

## AI & Computing Labs

### AI Safety Lab
**File:** `ai_safety_lab.py`

**What it does:** Evaluates AI systems for safety properties:
- **Reward hacking risk:** Probability of exploiting proxy rewards
- **Calibration:** ECE (expected calibration error)
- **Corrigibility:** Shutdown acceptance, value modification
- **Scalable oversight:** Human review coverage
- **Robustness bounds:** Adversarial perturbation limits

**Practical use:**
- Safety case development
- Comparing AI system designs
- Teaching AI alignment concepts

**⚠️ DISCLAIMER:** These are **simplified models**. Real AI safety assessment requires empirical testing, red-teaming, and expert review. Tool outputs are illustrative, not definitive safety evaluations.

---

### Bio-AI Convergent Solutions Lab
**File:** `bio_ai_lab.py`

**What it does:** Identifies algorithmic parallels between biological and artificial systems:
- Navigation: bacterial chemotaxis vs RL agents
- Learning: dopamine RPE vs TD learning
- Coordination: swarm consensus vs multi-agent systems

**Practical use:**
- Bio-inspired algorithm design
- Understanding neural computation
- Teaching computational neuroscience

**Data sources:**
- **C. elegans connectome:** Built-in 302-neuron circuit data
- **Chemotaxis models:** E. coli adaptation parameters

---

## Operations Research Labs

### Supply Chain Optimization Lab
**File:** `supply_chain_lab.py`

**What it does:** Analyzes supply chain scenarios using OR tools:
- **EOQ:** Economic order quantity for inventory
- **Safety stock:** Buffer stock for demand variability
- **Newsvendor:** Optimal ordering for perishables
- **Vehicle routing:** Delivery route optimization
- **Bin packing:** Warehouse space utilization

**Practical use:**
- Inventory policy design
- Logistics cost optimization
- Teaching operations research

**Data sources:**
- **Built-in:** Example scenarios (retail, perishable goods, e-commerce)
- **Future:** Integration with real demand datasets

---

### Behavioral Economics Lab
**File:** `behavioral_economics_lab.py`

**What it does:** Analyzes decision-making under uncertainty:
- **Prospect theory:** Value function, probability weighting
- **Loss aversion:** λ ≈ 2.25 empirical ratio
- **Temporal discounting:** Hyperbolic vs exponential
- **Allais paradox:** Certainty effect demonstration
- **Herding cascades:** Information cascade thresholds

**Practical use:**
- Teaching behavioral finance
- Analyzing investment decisions
- Understanding cognitive biases

**Data sources:**
- **Built-in:** Kahneman-Tversky parameters from empirical studies

---

## Basic Science Labs

### Origin of Life Lab
**File:** `origin_of_life_lab.py`

**What it does:** Evaluates abiogenesis scenarios:
- **Autocatalytic sets:** RAF (reflexively autocatalytic food-generated) detection
- **Prebiotic plausibility:** Miller-Urey yields, geological availability
- **Eigen threshold:** Maximum genome length given mutation rate
- **RNA world:** Folding energetics, ribozyme stability

**Practical use:**
- Teaching chemical evolution
- Evaluating origin hypotheses
- Understanding information threshold

**Data sources:**
- **Built-in:** Miller-Urey product yields, nucleotide thermodynamics

---

### Quantum Mechanics Lab
**File:** `quantum_mechanics_lab.py`

**What it does:** Basic QM calculations:
- Particle-in-box energy levels
- Hydrogen atom energies
- Tunneling probabilities
- Harmonic oscillator states

**Practical use:**
- Teaching quantum mechanics
- Quick calculations for coursework

**⚠️ NOTE:** This lab duplicates functionality already in the main NoetherSolve toolkit. Consider deprecating.

---

## Running Labs

All labs follow the same pattern:
```bash
python labs/<lab_name>.py           # Run with defaults
python labs/<lab_name>.py --verbose # Detailed output
```

Results are saved to `results/labs/<lab_name>/`.

## Adding New Labs

Follow this pipeline when creating or extending a lab:

### 1. Search for Data Sources FIRST

Before writing any code, search for public APIs that provide real data:

```bash
# Check existing integrations
cat labs/DATA_SOURCES.md

# Search for new sources
# "[domain] public API free"
# "[domain] REST API JSON"
# "[domain] open data download"
```

**Go-to directories:**
- APIs.guru (https://apis.guru/)
- Public APIs (https://github.com/public-apis/public-apis)
- Kaggle Datasets (https://kaggle.com/datasets)

### 2. Create Data Source Module

If you find a useful API, create `noethersolve/{domain}_data_source.py`:

```python
# Always include built-in fallback data
REFERENCE_DATA = {...}

class DomainDataSource:
    def get(self, name: str):
        # Try cache → Try API → Return None
```

See `drug_data_source.py` or `disease_data_source.py` as templates.

### 3. Create Lab Module

Create `labs/my_lab.py`:
- Chain existing NoetherSolve tools
- Use data source module instead of hardcoded values
- Add proper disclaimers in docstring

### 4. Document

- Add to `LAB_CATALOG.md` with description and disclaimers
- Add data source to `DATA_SOURCES.md`
- Register in lab registry

### 5. Test

```bash
python labs/my_lab.py --verbose
pytest tests/test_my_lab.py
```

---

## Global Disclaimers

**RESEARCH USE ONLY:** All labs are designed for research, education, and preliminary screening. None are validated for clinical, regulatory, or safety-critical applications.

**NO WARRANTIES:** Tools are provided as-is. Results should be validated independently before any consequential use.

**EXPERT REVIEW REQUIRED:** Outputs from medical, safety, and climate labs require review by domain experts before any decision-making.
