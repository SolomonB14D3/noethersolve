"""Model-agnostic multiple-choice oracle for factual truth recall.

Vendored from knowledge-fidelity/experiments/operation_destroyer/eval_mc.py.
No external knowledge-fidelity path dependencies — self-contained.

Measures whether a model assigns higher log-probability to the correct answer
than to all distractors. Completely independent of:
  - Tokenization (compares full completion log-probs, not first-token rank)
  - Thinking tokens / fill-in-blank triggers
  - Prompt format (neutral context: format)
  - Vocabulary size / model architecture

Works with ANY autoregressive model that mlx_lm can load.

Metric: truth wins if log P(truth | prompt) > log P(distractor | prompt)
        for ALL distractors. (Strict MC accuracy.)
"""

import mlx.core as mx
import numpy as np


def get_completion_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """Compute sum log P(completion | prompt) for any autoregressive model.

    Uses teacher forcing: encode prompt+completion, sum log-probs over
    completion tokens only. Model-agnostic — works for any model regardless
    of tokenization or architecture.
    """
    prompt_ids = tokenizer.encode(prompt)
    full_ids   = tokenizer.encode(prompt + completion)

    n_prompt = len(prompt_ids)
    if len(full_ids) <= n_prompt:
        return -1e9  # completion tokenized to nothing

    tokens = mx.array(full_ids)[None, :]  # [1, seq]

    logits = model(tokens)
    mx.eval(logits)

    logits_np = np.array(logits[0].astype(mx.float32))  # [seq, vocab]

    # Log-probs at positions [n_prompt-1 … len-2] predict tokens [n_prompt … len-1]
    total_lp = 0.0
    for i, tok_id in enumerate(full_ids[n_prompt:]):
        pos  = n_prompt - 1 + i
        row  = logits_np[pos]
        lse  = float(np.log(np.sum(np.exp(row - row.max())) + 1e-8) + row.max())
        total_lp += float(row[tok_id]) - lse

    return total_lp


def score_fact_mc(model, tokenizer, context: str, truth: str, distractors: list,
                  adapter=None, lm_head=None):
    """Score one fact via multiple-choice log-prob comparison.

    Returns:
        (win, margin, truth_lp, best_distractor_lp)

        win    — True if truth beats ALL distractors
        margin — truth_lp − max(distractor_lps)

    Args:
        adapter, lm_head — optional snap-on adapter (from noethersolve.adapter).
            Can be a single adapter or a list of adapters.
            If a list, logits are computed via apply_adapter_stack (summed shifts).
            If a single adapter, uses apply_adapter as before.
            If provided, logits are computed as:
                h = model.model(tokens)
                logits = apply_adapter[_stack](adapter(s), lm_head(h))
    """
    prompt = f"{context}:"  # neutral format, no fill-in-blank triggers

    if adapter is not None and lm_head is not None:
        # Determine whether we have a stack of adapters or a single one
        if isinstance(adapter, list):
            from noethersolve.train_utils import apply_adapter_stack
            _apply = lambda bl: apply_adapter_stack(adapter, bl)
        else:
            from noethersolve.train_utils import apply_adapter
            _apply = lambda bl: apply_adapter(adapter, bl)

        def _lp(completion: str) -> float:
            prompt_ids = tokenizer.encode(prompt)
            full_ids   = tokenizer.encode(prompt + completion)
            n_prompt   = len(prompt_ids)
            if len(full_ids) <= n_prompt:
                return -1e9
            tokens = mx.array(full_ids)[None, :]
            h = model.model(tokens)
            mx.eval(h)
            bl = lm_head(h)
            mx.eval(bl)
            logits    = _apply(bl)
            mx.eval(logits)
            logits_np = np.array(logits[0].astype(mx.float32))
            total = 0.0
            for i, tok_id in enumerate(full_ids[n_prompt:]):
                pos  = n_prompt - 1 + i
                row  = logits_np[pos]
                lse  = float(np.log(np.sum(np.exp(row - row.max())) + 1e-8) + row.max())
                total += float(row[tok_id]) - lse
            return total

    else:
        def _lp(completion: str) -> float:
            return get_completion_logprob(model, tokenizer, prompt, completion)

    truth_lp  = _lp(f" {truth}")
    dist_lps  = [_lp(f" {d}") for d in distractors]
    best_dist = max(dist_lps)
    win       = truth_lp > best_dist
    margin    = truth_lp - best_dist
    return win, margin, truth_lp, best_dist
