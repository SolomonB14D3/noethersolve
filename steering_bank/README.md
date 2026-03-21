# Steering Bank — Benchmark Fact Library

**523 domain files** containing MC questions from standard benchmarks + custom NoetherSolve facts.
Model-agnostic — reusable on any LLM for steering vector extraction or adapter training.

## Sources

| Source | Domains | Facts | Format |
|--------|---------|-------|--------|
| MMLU (57 subjects) | 57 | ~5,700 | 4-option MC |
| MMLU-Pro (14 categories) | 127 | ~12,000 | 10-option MC (trimmed to 4) |
| TruthfulQA | 9 | ~817 | MC with targets |
| GPQA (graduate-level) | 9 | ~448 | 4-option MC |
| MedMCQA (20 subjects) | 44 | ~4,000 | 4-option MC |
| ARC (Easy + Challenge) | 8 | ~4,750 | 3-5 option MC |
| BoolQ | 17 | ~3,270 | Yes/No |
| CommonsenseQA | 7 | ~1,221 | 5-option MC |
| WinoGrande | 7 | ~1,267 | 2-option MC |
| HellaSwag | 51 | ~10,042 | 4-option MC |
| RACE (middle + high) | 8 | ~600 | 4-option MC |
| COPA | 1 | ~500 | 2-option MC |
| Custom (NoetherSolve) | 84 | ~1,043 | 4-option MC |

## File Format

Each JSON file:
```json
{
  "domain": "abstract_algebra",
  "source": "mmlu",
  "facts": [
    {
      "context": "Question text...",
      "truth": "Correct answer",
      "distractors": ["Wrong 1", "Wrong 2", "Wrong 3"]
    }
  ]
}
```

## Usage

```python
import json
with open("steering_bank/abstract_algebra.json") as f:
    data = json.load(f)
facts = data["facts"]  # list of {context, truth, distractors}
```

## Regenerating

```bash
python experiments/build_steering_bank.py          # Download from HuggingFace
python experiments/extract_vectors_fast.py          # Extract steering vectors
python experiments/train_steering_failures.py       # Train adapters on failures
```

## Applying to a New Model

The bank is model-agnostic. To apply to a different model:

1. Load the new model
2. Run `extract_vectors_fast.py` with `--model your-model-name` (or modify the load line)
3. Vectors will be model-specific but facts are universal
4. Adapters need retraining per model (different vocab size / hidden dim)
