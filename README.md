# NoetherSolve

**https://github.com/SolomonB14D3/noethersolve** · **https://solomonb14d3.github.io/noethersolve**

[![D1: Conservation Laws](https://zenodo.org/badge/DOI/10.5281/zenodo.19055338.svg)](https://doi.org/10.5281/zenodo.19055338)
[![D3: LLM Knowledge Gaps](https://zenodo.org/badge/DOI/10.5281/zenodo.19055582.svg)](https://doi.org/10.5281/zenodo.19055582)
[![D5: Certainty Contamination](https://zenodo.org/badge/DOI/10.5281/zenodo.19068373.svg)](https://doi.org/10.5281/zenodo.19068373)
[![D6: Resolvent Unification](https://zenodo.org/badge/DOI/10.5281/zenodo.19071198.svg)](https://doi.org/10.5281/zenodo.19071198)
[![D10: Bio-AI Convergence](https://zenodo.org/badge/DOI/10.5281/zenodo.19152253.svg)](https://doi.org/10.5281/zenodo.19152253)

**Find where AI models are wrong, fix them, and build tools that give the right answer.**

> **Research Project Notice**: This is an experimental research project, not a production system. All results should be independently verified before making critical decisions. The tools augment human expertise—they don't replace it.

---

<details open>
<summary><h2>Steering Vectors</h2></summary>

### What they are

A **steering vector** is a tiny file (about 0.1 KB) that nudges an AI model toward correct answers when it already knows the answer but picks the wrong one.

Think of it like giving someone a hint when they know the answer but can't quite remember it. The knowledge is already there—it just needs a small push in the right direction.

### When they help

Steering vectors work when the model is "mute, not dumb"—it has the knowledge inside but doesn't surface it correctly. This happens surprisingly often: the model learned the right information during training, but its output mechanism picks the wrong answer.

### Results

| Metric | Value |
|--------|-------|
| Domains tested | 570 |
| Improved by steering | 129 (23%) |
| Made worse | 0 (0%) |
| Total storage | ~3.5 MB |

Top improvements:
- TruthfulQA: 13% → **100%**
- Moral Scenarios: 0% → **73%**
- WinoGrande: 3% → **83%**
- CommonsenseQA: 33% → **97%**

<details>
<summary><strong>Available Steering Vectors</strong></summary>

Steering vectors are stored in `steering_vectors/{model}/` as `.npy` files. Current coverage for Qwen3-4B-Base:

| Category | Examples | Domains |
|----------|----------|---------|
| Truthfulness | TruthfulQA chunks | 8 |
| Commonsense | CommonsenseQA, WinoGrande, COPA | 15 |
| Ethics | Moral Scenarios, MMLU Ethics | 6 |
| Reading | RACE, BoolQ | 10 |
| Medical | MedMCQA subdomains | 20+ |
| MMLU | Various MMLU subjects | 57 |
| MMLU-Pro | Extended MMLU subjects | 127 |
| Custom | Hand-crafted domain facts | 84 |

</details>

<details>
<summary><strong>Related Papers</strong></summary>

- **D7: Nine Systematic Biases in Log-Probability Evaluation** · [DOI: 10.5281/zenodo.19124851](https://doi.org/10.5281/zenodo.19124851)
  Documents 9 biases: length, coherence, scoring method, anti-fluency, round numbers, certainty, simplification, term preference, status blindness.

- **Paper 7: The Expression Bottleneck** · [DOI: 10.5281/zenodo.18895248](https://doi.org/10.5281/zenodo.18895248)
  Explains the "mute not dumb" phenomenon: a universal 41% accuracy constant across all scales, revealing a generation bottleneck.

</details>

</details>

---

<details open>
<summary><h2>Adapters</h2></summary>

### What they are

An **adapter** is a larger file (about 50 MB) that teaches the model new knowledge it didn't have before.

Think of it like giving someone a study guide for material they never learned. The model genuinely doesn't know this information, so we need to teach it—not just remind it.

### When they help

Adapters work when the model truly lacks knowledge: graduate-level STEM, specialized medicine, cutting-edge research, or domain-specific facts that weren't well-represented in training data.

### Results

| Metric | Value |
|--------|-------|
| Domains at 100% | 77 |
| Facts successfully taught | 1043+ |
| Knowledge loss elsewhere | 0% |

Example improvements:
- College Mathematics: 6% → **100%**
- GPQA (graduate-level): 4% → **100%**
- Navier-Stokes Regularity: 0% → **100%** (16/16 facts)
- Knot Theory Invariants: 0% → **100%** (16/16 facts)

> **Note**: Adapters are trained on specific domains. They may not generalize to related but different topics. Always verify with domain experts.

<details>
<summary><strong>Available Adapters</strong></summary>

Adapters are stored in `adapters/` as `.npz` files. Current coverage:

| Domain | Baseline | After Adapter | Facts |
|--------|----------|---------------|-------|
| Navier-Stokes Regularity | 0/16 | 16/16 | Conservation vs blowup |
| Hamiltonian Mechanics | 1/16 | 16/16 | Phase space invariants |
| Chemical Kinetics | 0/16 | 16/16 | Reaction network conservation |
| Knot Theory | 0/16 | 16/16 | Reidemeister, Jones polynomial |
| Electromagnetism | 1/12 | 12/12 | Energy, zilch, chirality |
| Continuous Q_f | 0/12 | 12/12 | Euler conservation laws |
| Intersection Theory | 0/16 | 16/16 | Bezout, genus, Noether formula |
| Elliptic Curves | 0/16 | 16/16 | Hasse, discriminant, point order |
| Drug Interactions | 0/12 | 12/12 | CYP450, DDI prediction |
| Information Theory | 0/12 | 12/12 | Channel capacity, rate distortion |

</details>

<details>
<summary><strong>Related Papers</strong></summary>

- **D3: Where LLMs Are Confidently Wrong** · [DOI: 10.5281/zenodo.19055582](https://doi.org/10.5281/zenodo.19055582)
  1038 facts across 67 domains. Intersection theory is the deepest gap (margin -27.6).

- **D4: Orthogonal Adapter Routing** · [DOI: 10.5281/zenodo.19055588](https://doi.org/10.5281/zenodo.19055588)
  100% across 69 domains. Key insight: route adapters, never stack them.

- **D5: Certainty Contamination** · [DOI: 10.5281/zenodo.19068373](https://doi.org/10.5281/zenodo.19068373)
  Models prefer definitive claims over hedged scientific language (r = -0.402).

</details>

</details>

---

<details open>
<summary><h2>Tools (MCP)</h2></summary>

### What they are

**Tools** are verified calculators that derive answers from first principles. Unlike steering or adapters that modify how the model thinks, tools compute the answer directly—like giving someone a calculator instead of teaching them arithmetic.

Tools are exposed via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), so any AI agent (Claude, GPT, local models) can call them during a conversation.

### Why tools instead of fine-tuning?

| Approach | Pros | Cons |
|----------|------|------|
| **Fine-tuning** | Model "knows" the answer | Can forget other things, limited capacity |
| **Tools** | Verified correct, unlimited scope | Requires tool call |

Tools don't have capacity limits, don't interfere with each other, and are verified by 2265 tests. Adding tool #43 doesn't break tools #1-42.

### Results

- **230+ tools** currently available
- **220+ calculators** (derive answers from first principles)
- **10+ lookup tables** (reference databases)
- **25+ domains** covered

<details>
<summary><strong>Available Tools</strong></summary>

| Category | Tools | Examples |
|----------|-------|----------|
| **Conservation** | 4 | `check_vortex_conservation`, `check_hamiltonian_system`, `check_em_conservation` |
| **Mathematics** | 10 | `check_conjecture`, `check_complexity_inclusion`, `verify_goldbach`, `check_sobolev_embedding` |
| **Genetics** | 5 | `score_crispr_guide`, `audit_dna_sequence`, `predict_protein_aggregation`, `validate_therapy_pipeline` |
| **Enzyme Kinetics** | 5 | `calc_michaelis_menten`, `calc_enzyme_inhibition`, `calc_cooperativity` |
| **Quantum Mechanics** | 6 | `calc_particle_in_box`, `calc_hydrogen_energy`, `calc_tunneling`, `calc_uncertainty_check` |
| **Pharmacokinetics** | 5 | `calc_iv_bolus`, `calc_oral_dose`, `calc_half_life`, `calc_steady_state` |
| **Organic Chemistry** | 6 | `analyze_molecule`, `predict_reaction_mechanism`, `check_baldwin_rules` |
| **LLM Science** | 4 | `check_llm_claim`, `chinchilla_scaling`, `check_benchmark_score` |
| **Chemistry** | 3 | `calc_nernst`, `calc_buffer_ph`, `calc_crystal_field` |
| **Cryptography** | 3 | `calc_security_level`, `calc_birthday_bound`, `calc_cipher_mode` |
| **Finance** | 3 | `calc_black_scholes`, `calc_put_call_parity`, `calc_nash_equilibrium` |
| **Distributed Systems** | 3 | `calc_quorum`, `calc_byzantine`, `calc_vector_clock` |
| **Networking** | 3 | `calc_bandwidth_delay`, `calc_subnet`, `calc_tcp_throughput` |
| **Operating Systems** | 3 | `calc_page_table`, `calc_scheduling`, `calc_deadlock` |
| **Epidemiology** | 8 | `calc_sir_model`, `calc_reproduction_number`, `calc_herd_immunity` |
| **Climate** | 6 | `calc_co2_forcing`, `calc_climate_sensitivity`, `analyze_climate_feedback` |
| **Topological Materials** | 6 | `calc_chern_number`, `calc_z2_invariant`, `check_bulk_boundary` |

</details>

<details>
<summary><strong>Related Papers</strong></summary>

- **NoetherSolve Toolkit** · [DOI: 10.5281/zenodo.19029880](https://doi.org/10.5281/zenodo.19029880)
  Full documentation of 230+ tools. 2265 tests.

- **D1: Approximate Conservation Laws** · [DOI: 10.5281/zenodo.19055338](https://doi.org/10.5281/zenodo.19055338)
  Q_f family: an infinite family of approximate invariants in vortex dynamics.

- **D6: Resolvent-Conservation Unification** · [DOI: 10.5281/zenodo.19071198](https://doi.org/10.5281/zenodo.19071198)
  Green's function optimality from spectral theory.

</details>

</details>

---

<details open>
<summary><h2>Practical Usage</h2></summary>

NoetherSolve tools are designed for **verification**, not generation. They check that answers are correct rather than producing creative content.

### For Coding: Verification

NoetherSolve doesn't write code. It **verifies** that code is correct:

| Use Case | Tool | What it checks |
|----------|------|----------------|
| Physics simulation | `check_hamiltonian_system()` | Does the simulation conserve energy? |
| Algorithm analysis | `check_complexity_inclusion()` | Is my complexity claim accurate? |
| Concurrent code | `calc_deadlock()` | Can these locks deadlock? |
| PDE numerics | `check_pde_cfl()` | Is my timestep stable? |
| Network code | `calc_bandwidth_delay()` | What's the theoretical throughput? |

**Example**: "Does my N-body simulation conserve energy?"
```
check_hamiltonian_system(positions, momenta, masses)
→ Reports energy conservation error, identifies drift
```

### For Research: Screening

Tools help researchers check their work before expensive experiments:

| Use Case | Tool | What it checks |
|----------|------|----------------|
| New conservation law | `discover_conservation_law()` | Is this quantity actually conserved? |
| Reaction mechanism | `audit_chemical_network()` | Thermodynamically consistent? |
| Drug candidate | `calc_half_life()`, `check_drug_interaction()` | PK properties, DDI risk |
| Synthesis pathway | `validate_synthesis_pathway()` | Chemically plausible? |
| CRISPR target | `score_crispr_guide()` | On-target score, off-target risk |

**Example**: "Is this chemical reaction network thermodynamically consistent?"
```
audit_chemical_network(reactions, rate_constants)
→ Checks Wegscheider cyclicity, detailed balance, mass action
```

### For Fact Checking: STEM Claims

Tools verify claims about STEM topics:

| Use Case | Tool | What it checks |
|----------|------|----------------|
| Math conjecture | `check_conjecture("Riemann")` | Status: OPEN, PROVED, or DISPROVED |
| LLM benchmark | `check_benchmark_score("GPT-4", "MMLU")` | Actual vs claimed score |
| Complexity claim | `check_complexity_inclusion("P", "NP")` | Is P ⊆ NP? (Yes, trivially) |
| Physics equation | Various calculators | Compute from first principles |

**Example**: "Is P=NP proven?"
```
check_conjecture("P_vs_NP")
→ Status: OPEN, Clay Millennium Problem, current evidence, common errors
```

> **Important**: Tools verify what can be computed or looked up from established sources. They don't replace domain expertise. Critical decisions—medical, legal, safety-related—require human expert review.

</details>

---

<details open>
<summary><h2>Labs</h2></summary>

### What they are

**Labs** are automated computational pipelines that use NoetherSolve tools to screen candidates and produce ranked results. Each lab focuses on a specific scientific domain.

Think of labs as virtual research assistants: they run the computational screening so researchers can focus on the candidates most likely to succeed in actual experiments.

### How they work

1. **Define screening criteria** — What makes a good candidate?
2. **Run tools on candidates** — Compute scores using verified calculators
3. **Rank by viability** — Produce a prioritized list
4. **Export for validation** — Results go to wet labs or field experts

### What they produce

Labs output **ranked candidate lists** ready for experimental validation. They do NOT produce final answers—they narrow the search space.

> **Critical Disclaimer**: Labs produce computational screening results ONLY. They do NOT replace:
> - Wet-lab experiments for drug/material candidates
> - Clinical trials for medical applications
> - Peer review for scientific claims
> - Domain expert judgment
>
> All candidates require experimental validation before any real-world application.

<details>
<summary><strong>Active Labs (14)</strong></summary>

| Lab | Description | Tools |
|-----|-------------|-------|
| **Drug Therapy Screening** | PK/PD screening: half-life, therapeutic index, CYP interactions, DDI risk | 12 |
| **Genetic Therapeutics** | CRISPR guide scoring, mRNA optimization, neoantigen evaluation, antibody developability | 18 |
| **Catalyst Discovery (HER)** | DFT-informed prescreening: d-band center, volcano position, BEP activation energy | 8 |
| **Epidemic Dynamics** | SIR modeling, R0 estimation, herd immunity thresholds, doubling times | 8 |
| **Climate Sensitivity** | CO2 forcing, feedback profiles, equilibrium climate sensitivity | 6 |
| **Topological Materials** | Berry phase, Chern number, Z2 invariant, bulk-boundary correspondence | 6 |
| **Conservation Law Mining** | Automated discovery of approximate conservation laws in dynamical systems | 5 |
| **Bio-AI Convergence** | Verification of computational parallels between biological and AI systems | 8 |
| **Origin of Life** | Miller-Urey yield, autocatalytic sets, prebiotic plausibility scoring | 5 |
| **Battery Materials** | Cycle aging, calendar aging, LFP vs NCA vs NMC comparison | 4 |
| **Quantum Mechanics** | Particle-in-box, hydrogen energies, tunneling, uncertainty validation | 7 |
| **Behavioral Economics** | Prospect theory, loss aversion, Allais paradox, framing effects | 7 |
| **AI Safety Evaluation** | Adversarial robustness, reward hacking risk, oversight coverage | 7 |
| **Supply Chain** | EOQ, safety stock, newsvendor, vehicle routing, bin packing | 5 |

</details>

<details>
<summary><strong>Related Papers</strong></summary>

- **D10: Bio-AI Convergence** · [DOI: 10.5281/zenodo.19152253](https://doi.org/10.5281/zenodo.19152253)
  Statistical framework for testing bio-AI parallels. 13 parallels tested, 5 strong convergences.

- **D1: Conservation Laws in Point Vortex Dynamics** · [DOI: 10.5281/zenodo.19055338](https://doi.org/10.5281/zenodo.19055338)
  Discovery of Q_f family through the conservation mining lab.

</details>

</details>

---

<details open>
<summary><h2>Making New Discoveries</h2></summary>

### The core insight

AI models are trained on human writing. That means they know what we know—but they also share our blind spots. Where the collective literature is thin or wrong, the model is thin or wrong.

**Model gaps point to human knowledge gaps.**

### The discovery loop

```
1. PROPOSE  →  Generate hypotheses about how systems behave
       ↓
2. VERIFY   →  Check numerically: is this actually true?
       ↓
3. TEST     →  Ask the model: did you already know this?
       ↓
4. DISCOVER →  If the model is wrong, we've found a gap
       ↓
5. FIX      →  Steering vector → Adapter → Tool
       ↓
6. SHARE    →  Add to MCP server so all agents benefit
```

### The escalation ladder

| Method | Size | Time | When to use |
|--------|------|------|-------------|
| Steering vector | 0.1 KB | Seconds | Model knows but picks wrong |
| Adapter | 50 MB | Minutes | Model genuinely doesn't know |
| Tool | Any | Once | Maximum reliability needed |

### Example: Q_f conservation laws

The pipeline discovered that Q_f = Σ ΓᵢΓⱼ f(rᵢⱼ) is approximately conserved for **any** smooth function f in vortex dynamics. The model had a complete blind spot (margin -77.5). This pointed directly at an unexplored area of physics.

The discovery is now:
- Documented in [Paper D1](https://doi.org/10.5281/zenodo.19055338)
- Implemented as `check_vortex_conservation()` tool
- Available to any AI agent via MCP

> **Note**: Scientific discoveries require peer review and experimental validation. The pipeline identifies candidates—the scientific community validates them.

<details>
<summary><strong>Example Discoveries</strong></summary>

| Discovery | Domain | Model Margin | Status |
|-----------|--------|--------------|--------|
| Q_f family (infinite approximate invariants) | Vortex dynamics | -77.5 → +0.93 | Published D1 |
| Stretch-resistant R_f ratio | NS regularity | -30 → +34.3 | Published |
| EM zilch conservation | Electromagnetism | -11.6 → +8.2 | Verified |
| KAM theorem (model inverted) | Hamiltonian | -59.8 → +3.9 | Fixed |
| Grid cells ≠ position encoding | Bio-AI | r=0.002 | Non-convergence |
| Oscillations ↔ Attention | Bio-AI | r=0.871 | Novel parallel |

</details>

<details>
<summary><strong>Related Papers</strong></summary>

- **Breaking Frozen Priors** · [DOI: 10.5281/zenodo.19017290](https://doi.org/10.5281/zenodo.19017290)
  Three-phase pipeline transforms frozen oracle into ranking engine (ρ = 0.932).

- **D8: Unified Cycle Theory** · [DOI: 10.5281/zenodo.19124858](https://doi.org/10.5281/zenodo.19124858)
  Cross-domain conservation connecting vortex dynamics, chemistry, Hamiltonian mechanics.

</details>

</details>

---

<details>
<summary><h2>Quick Start</h2></summary>

### Install

```bash
pip install noethersolve
```

### Use a tool directly

```python
from noethersolve import check_conjecture

result = check_conjecture("Riemann")
print(result)
# → Status: OPEN, Clay Millennium Problem, key facts, common errors
```

### Serve tools to AI agents via MCP

```bash
# Start the MCP server
noethersolve-mcp

# Or in Claude Code, the .mcp.json is auto-discovered
```

### Apply a steering vector

```python
import numpy as np

# Load a steering vector
sv = np.load("steering_vectors/qwen3_4b_base/truthfulqa_best.npy")

# Add sv * alpha to hidden state at layer 20 during forward pass
# alpha typically between 0.5 and 2.0
```

### Train an adapter

```bash
# Train adapter on domain facts
python experiments/train_steering_failures.py --domain my_domain
```

</details>

---

<details>
<summary><h2>Technical Details</h2></summary>

### Pipeline architecture

```
Hypothesis → Numerical Checker → Oracle Filter → Adapter Training → Tool Building → MCP Server
                  ↓                    ↓                ↓
              frac_var < 5e-3     margin test      hinge loss
```

### Escalation for hard domains

1. **Single-pass adapter** — Works for clean domains
2. **Staged training** — Group facts into clusters, train sequentially
3. **Orthogonal adapters** — Separate adapters per concept cluster, routed at inference
4. **Joint training** — Train on multiple domains simultaneously with difficulty weighting

### Key findings

- Adapters can't be stacked by weight averaging (-43% MMLU degradation)
- Route adapters, never merge them
- Steering vectors have zero regressions across 570 domains
- 77 domains reach 100% with orthogonal adapters

### File structure

```
noethersolve/
├── mcp_server/       # MCP tool implementations
├── *.py              # Domain modules (conservation, chemistry, etc.)
steering_vectors/     # Extracted steering vectors by model
adapters/             # Trained adapter weights
results/
├── labs/             # Lab output directories
├── discoveries/      # Discovery documentation
└── *.json            # Result files
```

</details>

---

## Built On

- **NoetherSolve Toolkit** — 230+ verified tools
  [DOI: 10.5281/zenodo.19029880](https://doi.org/10.5281/zenodo.19029880)

- **STEM Truth Oracle** — Log-prob MC ranking for factual evaluation
  [DOI: 10.5281/zenodo.19005729](https://doi.org/10.5281/zenodo.19005729)

- **Snap-On Communication Modules** — Frozen logit-space adapters
  [DOI: 10.5281/zenodo.18902616](https://doi.org/10.5281/zenodo.18902616)

- **Noether's theorem** (Emmy Noether, 1915) — The mathematical foundation: every continuous symmetry implies a conservation law.

---

## Cite

```bibtex
@software{noethersolve2026,
  author = {Sanchez, Bryan},
  title = {NoetherSolve: Automated Discovery and Verification Tools for AI Agents},
  year = {2026},
  url = {https://github.com/SolomonB14D3/noethersolve},
  doi = {10.5281/zenodo.19029880}
}
```
