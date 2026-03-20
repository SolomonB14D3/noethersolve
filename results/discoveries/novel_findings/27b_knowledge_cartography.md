# 27B Knowledge Cartography: Systematic Blind Spots in Qwen3.5-27B

## Discovery Summary

Comprehensive oracle evaluation of Qwen3.5-27B across 79 scientific domains reveals systematic knowledge gaps. The 27B model fails 37/79 domains (47%) with mean margins as low as -43.8.

## Key Findings

### Domain Coverage
- **Total domains tested:** 79
- **Passing (≥50% correct):** 42 domains (53%)
- **Failing (<50% correct):** 37 domains (47%)

### Deepest Knowledge Gaps (margin < -20)

| Domain | Margin | Pass Rate |
|--------|--------|-----------|
| Q_f Ratio Invariant | -43.8 | 0/8 |
| NS Regularity | -36.6 | 1/16 |
| Kinetic Invariant K | -36.0 | 0/8 |
| Continuous Q_f | -34.4 | 0/12 |
| Optimal f(r) | -33.1 | 0/4 |
| Intersection Theory | -23.9 | 0/12 |
| Knot Invariants | -22.9 | 0/16 |

### Gap Categories

1. **Novel Conservation Laws** (Q_f family, Kinetic K, NS regularity)
   - Mean margin: -38.5
   - Model has no representation of these recently-discovered invariants

2. **Advanced Pure Mathematics** (Intersection theory, knot invariants, algebra/topology)
   - Mean margin: -21.7
   - Complex symbolic manipulation + geometric intuition required

3. **Frontier Physics** (safety invariants, Hamiltonian mechanics)
   - Mean margin: -16.3
   - Requires integration of multiple sub-fields

### Strongest Knowledge Areas (margin > +10)

| Domain | Margin | Pass Rate |
|--------|--------|-----------|
| Multi-messenger Astro | +12.2 | 12/12 |
| Black Hole Frontiers | +11.6 | 12/12 |
| Particle Physics | +11.2 | 10/12 |
| Condensed Matter | +11.1 | 12/12 |
| Dark Matter/Energy | +10.4 | 8/12 |

### Pattern Analysis

1. **Scale effect:** Large-scale physics (cosmology, astrophysics) passes at 100%. Small-scale (molecular, quantum) passes at 67-83%.

2. **Recency effect:** Post-2020 discoveries (Q_f family, recent LLM papers) fail at 0-25%. Pre-2010 established results pass at 75-100%.

3. **Abstraction effect:** Concrete calculations (chemistry, networking) pass at 83-92%. Abstract theory (intersection theory, topology) fails at 0-25%.

## Methodology

- Model: `mlx-community/Qwen3.5-27B-4bit`
- Evaluation: Log-probability multiple-choice with sum scoring
- Facts: 800+ verified facts across 79 domains
- Threshold: margin > 0 = PASS, margin < 0 = FAIL

## Implications

1. **Training data bias:** Model knowledge reflects internet corpus distribution, not scientific importance
2. **Adapter targets:** The 37 failing domains are prime targets for domain-specific adapters
3. **Capability limits:** Even at 27B scale, specialized scientific knowledge requires explicit injection

## Date
2026-03-20

## Tools Used
NoetherSolve oracle pipeline with 79 domain YAML files
