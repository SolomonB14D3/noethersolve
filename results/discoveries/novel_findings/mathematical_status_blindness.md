# Mathematical Status Blindness

## Discovery

LLMs exhibit **mathematical status blindness**: they can accurately state what a conjecture claims but **systematically fail** when asked about its research status (proven/open/partially resolved).

## Empirical Evidence

### Pass Rate by Question Type

| Question Type | Pass Rate | Avg Margin | n |
|---------------|-----------|------------|---|
| Content-only (what does it claim?) | **71.4%** | -2.2 | 7 |
| Status-only (is it proven/open?) | **4.2%** | -33.3 | 24 |
| Mixed | 7.4% | -20.4 | 27 |

**Statistical significance:** t = -4.21, p = 0.000224

### Status Word Failure Rates

| Status Word | Facts | Pass Rate | Avg Margin |
|-------------|-------|-----------|------------|
| "proven" | 11 | **0%** | -25.1 |
| "open" | 8 | **0%** | -31.3 |
| "unknown" | 4 | **0%** | -41.4 |
| "conjectured" | 4 | **0%** | -54.9 |

### Confusion Direction (Novel Finding)

The model has a **directional bias toward resolution**:

| Confusion Type | Count | Percentage |
|----------------|-------|------------|
| Thinks "proven" when truth is "open" | 6 | 27% |
| Thinks "open" when truth is "proven" | 0 | **0%** |
| Oversimplifies nuanced status | 5 | 23% |
| Other | 11 | 50% |

**The model NEVER downgrades proven results to open, but DOES upgrade open problems to proven.**

## Examples

### PASSES (Content Questions)
```
nt01_goldbach: "every even integer greater than 2 is the sum of two primes"
nt03_twin_prime: "there are infinitely many pairs of primes differing by 2"
nt05_collatz: "eventually reaches 1 for all positive integers"
```

### FAILS (Status Questions)
```
at02_smooth_poincare: Truth="open" → Model picks "is proven true"
ap02_euler_blowup: Truth="conjectured but not proven" → Model picks "is proven impossible"
nt11_fermat_primes: Truth="whether infinitely many exist is unknown" → Model picks "all Fermat numbers are prime"
```

## Mechanism

1. **Training data consistency**: Mathematical definitions/statements are repeated consistently across sources
2. **Status volatility**: Research status changes when proofs are published
3. **Temporal contamination**: Training data mixes pre- and post-proof discussions
4. **Resolution preference**: Definitive claims ("is proven", "is impossible") appear more frequently than nuanced status ("is open", "has partial progress")

## Relation to Other Biases

This is a **domain-specific manifestation** of Certainty Contamination Bias:
- General form: Model prefers definitive language over hedged language
- Mathematical form: Model prefers "proven/resolved" over "open/unknown"

The directional asymmetry (upgrades but never downgrades) suggests training data has more "X is proven" claims than "X is open" claims.

## Implications

### For Oracle Design
- Separate content questions from status questions
- Content: "Goldbach conjecture claims that..." → likely passes
- Status: "Goldbach conjecture is currently..." → likely fails

### For Adapter Training
- Create status-specific training data
- Focus on: proven↔open distinctions, partial progress, numerical bounds
- Include recent resolutions (e.g., weak Goldbach 2013, Poincaré 2003)

### For Tool Design
- MCP tools should provide authoritative status: `check_conjecture("Goldbach")` → returns both claim AND status
- This is exactly what `noethersolve.conjecture_status` does

## Quantitative Detection

```python
STATUS_WORDS = ["proven", "unproven", "open", "unknown", "conjectured",
                "resolved", "established", "progress", "bound"]

def is_status_question(truth: str) -> bool:
    """Detect if fact is asking about research status."""
    return any(w in truth.lower() for w in STATUS_WORDS)

# Status questions fail at ~5% vs content questions at ~70%
```

## Status

**CONFIRMED** - Verified with:
- 58 mathematical facts across 5 domains
- t = -4.21, p = 0.000224 (status vs content)
- 0% pass rate on all status words (proven, open, unknown, conjectured)
- Directional bias: upgrades open→proven, never downgrades proven→open

---

*Discovered: 2026-03-17*
*Method: Oracle analysis of mathematical conjecture domains, categorizing by question type*
