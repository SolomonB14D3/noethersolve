"""Training utilities for snap-on logit adapters.

Provides:
  LOGIT_SOFTCAP        — Gemma 2-style logit cap (30.0)
  get_lm_head_fn       — handles tied embeddings (Qwen2.5 pattern)
  apply_adapter        — single adapter: zero-mean centering + softcap
  apply_adapter_stack  — multiple adapters: sum centered shifts + softcap
                         (each discovery compounds without degrading prior knowledge)
"""

import mlx.core as mx
import mlx.nn as nn

# Gemma 2 / nanochat-style logit cap: caps TOTAL logits, not adapter shifts.
# Monotonic (tanh), so does not change argmax rankings — MC accuracy identical
# with or without the cap.
LOGIT_SOFTCAP: float = 30.0


def get_lm_head_fn(model):
    """Return the logit-projection function, handling tied embeddings.

    Works with Qwen2.5 (tied embeddings) and models with an explicit lm_head.
    """
    if hasattr(model, "lm_head") and model.lm_head is not None:
        try:
            _ = model.lm_head.weight.shape
            return model.lm_head
        except AttributeError:
            pass
    # Tied embeddings — Qwen2.5 pattern
    if hasattr(model, "args") and getattr(model.args, "tie_word_embeddings", False):
        return model.model.embed_tokens.as_linear
    # Last resort
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise RuntimeError(
        "Cannot find lm_head or tied embeddings on model. "
        "Supported: explicit lm_head, or Qwen2.5-style tied embed_tokens."
    )


def apply_adapter(adapter, base_logits: mx.array) -> mx.array:
    """Apply a snap-on logit adapter with zero-mean centering + softcap.

    Three-step recipe:
    1. Zero-mean centering: raw_shifts − mean(raw_shifts)
       Prevents uniform boost/suppress — adapter must pick winners and losers.
    2. Add to base logits (no tanh on shifts, preserves fine-grained control).
    3. Softcap total logits via LOGIT_SOFTCAP * tanh(combined / LOGIT_SOFTCAP).
       Bounds output without constraining adapter expressiveness; tanh monotonic
       so MC argmax rankings are unaffected.
    """
    raw_shifts = adapter(base_logits)
    centered = raw_shifts - raw_shifts.mean(axis=-1, keepdims=True)
    combined = base_logits + centered
    return LOGIT_SOFTCAP * mx.tanh(combined / LOGIT_SOFTCAP)


def apply_adapter_stack(adapters: list, base_logits: mx.array) -> mx.array:
    """Apply multiple snap-on logit adapters by summing their centered shifts.

    Like apply_adapter but for a stack of adapters:
    1. Compute zero-mean centered shifts from EACH adapter
    2. Sum all centered shifts
    3. Add to base logits
    4. Softcap total logits via LOGIT_SOFTCAP * tanh(combined / LOGIT_SOFTCAP)

    This lets each adapter's correction accumulate — each discovery makes the
    model immediately smarter for subsequent evaluations.
    """
    total_shift = mx.zeros_like(base_logits)
    for adapter in adapters:
        raw_shifts = adapter(base_logits)
        centered = raw_shifts - raw_shifts.mean(axis=-1, keepdims=True)
        total_shift = total_shift + centered
    combined = base_logits + total_shift
    return LOGIT_SOFTCAP * mx.tanh(combined / LOGIT_SOFTCAP)
