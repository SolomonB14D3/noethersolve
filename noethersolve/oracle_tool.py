"""Oracle Tool — expose the 4B model + adapters as a verification service.

This wraps the trained Qwen3-4B-Base model with domain adapters as an MCP tool
that any AI agent can call to verify claims.

Usage:
    from noethersolve.oracle_tool import verify_claim, get_domain_confidence

    result = verify_claim(
        claim="The Riemann Hypothesis is proven",
        domain="mathematics"
    )
    # Returns: {"verdict": "FALSE", "confidence": 0.92, "correct_answer": "open problem", ...}
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Lazy imports to avoid loading MLX until needed
_model = None
_tokenizer = None
_adapters = {}


@dataclass
class VerificationResult:
    """Result of claim verification."""
    claim: str
    verdict: str  # "TRUE", "FALSE", "UNCERTAIN"
    confidence: float  # 0.0 to 1.0
    domain: str
    adapter_used: Optional[str]
    margin: float  # log-prob margin (positive = confident correct)
    explanation: str

    def __str__(self) -> str:
        return (
            f"Verdict: {self.verdict} (confidence: {self.confidence:.1%})\n"
            f"Domain: {self.domain}\n"
            f"Margin: {self.margin:+.2f}\n"
            f"Explanation: {self.explanation}"
        )


def _load_model():
    """Lazy-load the model and tokenizer."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    try:
        from mlx_lm import load

        _model, _tokenizer = load("Qwen/Qwen3-4B-Base")
        return _model, _tokenizer
    except ImportError:
        raise RuntimeError(
            "MLX not available. Install with: pip install mlx mlx-lm\n"
            "Or use the PyTorch backend on Linux/CUDA."
        )


def _find_adapter(domain: str) -> Optional[Path]:
    """Find the best adapter for a domain."""
    adapters_dir = Path(__file__).parent.parent / "adapters"
    if not adapters_dir.exists():
        return None

    # Try exact match first
    exact = adapters_dir / f"{domain}_4b_adapter.npz"
    if exact.exists():
        return exact

    # Try prefix match
    for adapter in adapters_dir.glob(f"{domain}*.npz"):
        return adapter

    # Try substring match
    for adapter in adapters_dir.glob("*.npz"):
        if domain.lower() in adapter.stem.lower():
            return adapter

    return None


def _load_adapter(adapter_path: Path):
    """Load adapter weights."""
    global _adapters

    if str(adapter_path) in _adapters:
        return _adapters[str(adapter_path)]

    import numpy as np
    adapter = dict(np.load(adapter_path))
    _adapters[str(adapter_path)] = adapter
    return adapter


def _get_log_prob(model, tokenizer, prompt: str, completion: str,
                  adapter_weights: dict = None) -> float:
    """Get log probability of completion given prompt.

    If adapter_weights is provided, applies logit-space adapter shifts.

    Note: Tokenizes prompt and completion separately, then concatenates token IDs.
    This matches how adapters are trained and ensures consistent tokenization behavior.
    """
    import mlx.core as mx
    import mlx.nn as nn

    # Tokenize separately and concatenate (matches training script behavior)
    prompt_tokens = tokenizer.encode(prompt)
    comp_tokens = tokenizer.encode(completion)
    full_tokens = prompt_tokens + comp_tokens

    if not comp_tokens:
        return -999.0

    # Get hidden states and base logits
    input_ids = mx.array([full_tokens])[..., :-1]  # All but last token

    if adapter_weights is not None:
        # Get hidden states from the model backbone
        h = model.model(input_ids)
        # Get base logits from lm_head
        if hasattr(model, 'lm_head'):
            base_logits = model.lm_head(h)
        else:
            base_logits = h @ model.model.embed_tokens.weight.T

        # Apply adapter: logit shifts = down_proj(SiLU(gate_proj(logits) * up_proj(logits)))
        gate_proj = mx.array(adapter_weights['gate_proj.weight'])  # (d_inner, vocab)
        up_proj = mx.array(adapter_weights['up_proj.weight'])      # (d_inner, vocab)
        down_proj = mx.array(adapter_weights['down_proj.weight'])  # (vocab, d_inner)

        # SwiGLU-style adapter forward pass
        gate = base_logits @ gate_proj.T  # (batch, seq, d_inner)
        up = base_logits @ up_proj.T      # (batch, seq, d_inner)
        hidden = nn.silu(gate) * up  # SiLU(gate) * up
        shifts = hidden @ down_proj.T  # (batch, seq, vocab)

        # Center shifts and apply
        shifts = shifts - mx.mean(shifts, axis=-1, keepdims=True)
        logits = base_logits + shifts

        # Softcap for stability
        LOGIT_SOFTCAP = 30.0
        logits = LOGIT_SOFTCAP * mx.tanh(logits / LOGIT_SOFTCAP)
    else:
        logits = model(input_ids)

    # Calculate log probs for completion tokens only
    log_probs = nn.log_softmax(logits, axis=-1)

    completion_start = len(prompt_tokens) - 1
    total_log_prob = 0.0

    for i in range(completion_start, len(full_tokens) - 1):
        token_id = full_tokens[i + 1]
        total_log_prob += log_probs[0, i, token_id].item()

    return total_log_prob


def verify_fact(
    context: str,
    truth: str,
    distractors: list[str],
    domain: str = "general",
) -> VerificationResult:
    """Verify a fact using direct completion format (matches training).

    This is the format used by NoetherSolve adapter training:
    - Prompt = context
    - Completions = " truth" or " distractor" (with leading space)

    Args:
        context: The context/prompt (e.g., "APOE4 carriers with HSV-1...")
        truth: The correct completion
        distractors: Alternative (wrong) completions
        domain: Domain for adapter selection

    Returns:
        VerificationResult with verdict, confidence, and explanation
    """
    model, tokenizer = _load_model()

    # Find and load adapter
    adapter_path = _find_adapter(domain)
    adapter_used = None
    adapter_weights = None
    if adapter_path:
        adapter_used = adapter_path.stem
        adapter_weights = _load_adapter(adapter_path)

    # Get log prob for truth (with leading space, matching training format)
    truth_lp = _get_log_prob(model, tokenizer, context, f" {truth}", adapter_weights)

    # Get log probs for distractors
    dist_lps = []
    for d in distractors:
        d_lp = _get_log_prob(model, tokenizer, context, f" {d}", adapter_weights)
        dist_lps.append((d, d_lp))

    best_distractor, best_distractor_lp = max(dist_lps, key=lambda x: x[1])

    # Calculate margin
    margin = truth_lp - best_distractor_lp

    # Determine verdict
    if margin > 1.5:
        verdict = "TRUE"
        confidence = min(0.99, 0.5 + margin / 10)
    elif margin < -1.5:
        verdict = "FALSE"
        confidence = min(0.99, 0.5 + abs(margin) / 10)
    else:
        verdict = "UNCERTAIN"
        confidence = 0.5 + abs(margin) / 20

    # Build explanation
    if verdict == "FALSE":
        explanation = f"Model prefers: '{best_distractor}' over the truth"
    elif verdict == "TRUE":
        explanation = "Model confidence supports this claim"
    else:
        explanation = "Insufficient confidence to determine truth value"

    return VerificationResult(
        claim=truth,
        verdict=verdict,
        confidence=confidence,
        domain=domain,
        adapter_used=adapter_used,
        margin=margin,
        explanation=explanation,
    )


def verify_claim(
    claim: str,
    domain: str = "general",
    distractors: Optional[list[str]] = None,
    context: str = "",
) -> VerificationResult:
    """Verify a claim using the oracle model.

    Args:
        claim: The claim to verify (should be a factual statement)
        domain: Domain for adapter selection (e.g., "chemistry", "physics")
        distractors: Alternative claims to compare against
        context: Optional context for the claim

    Returns:
        VerificationResult with verdict, confidence, and explanation
    """
    model, tokenizer = _load_model()

    # Find and load adapter
    adapter_path = _find_adapter(domain)
    adapter_used = None
    adapter_weights = None
    if adapter_path:
        adapter_used = adapter_path.stem
        adapter_weights = _load_adapter(adapter_path)

    # Build prompt
    if context:
        prompt = f"{context}\n\nWhich is correct?\nA) {claim}\n"
    else:
        prompt = f"Which is correct?\nA) {claim}\n"

    # Get log prob for the claim
    claim_lp = _get_log_prob(model, tokenizer, prompt, "A", adapter_weights)

    # Compare against distractors if provided
    best_distractor_lp = float('-inf')
    best_distractor = None

    if distractors:
        for i, d in enumerate(distractors):
            letter = chr(ord('B') + i)
            prompt_with_d = prompt + f"{letter}) {d}\n"
            d_lp = _get_log_prob(model, tokenizer, prompt_with_d, letter, adapter_weights)
            if d_lp > best_distractor_lp:
                best_distractor_lp = d_lp
                best_distractor = d

    # Calculate margin and verdict
    if distractors:
        margin = claim_lp - best_distractor_lp
    else:
        margin = claim_lp  # No comparison baseline

    # Determine verdict
    if margin > 1.5:
        verdict = "TRUE"
        confidence = min(0.99, 0.5 + margin / 10)
    elif margin < -1.5:
        verdict = "FALSE"
        confidence = min(0.99, 0.5 + abs(margin) / 10)
    else:
        verdict = "UNCERTAIN"
        confidence = 0.5 + abs(margin) / 20

    # Build explanation
    if verdict == "FALSE" and best_distractor:
        explanation = f"Model prefers: '{best_distractor}' over the claim"
    elif verdict == "TRUE":
        explanation = "Model confidence supports this claim"
    else:
        explanation = "Insufficient confidence to determine truth value"

    return VerificationResult(
        claim=claim,
        verdict=verdict,
        confidence=confidence,
        domain=domain,
        adapter_used=adapter_used,
        margin=margin,
        explanation=explanation,
    )


def get_domain_confidence(domain: str) -> dict:
    """Get information about oracle confidence for a domain.

    Returns adapter availability and expected performance.
    """
    adapter_path = _find_adapter(domain)

    # Load domain stats if available
    stats_file = Path(__file__).parent.parent / "results" / "domain_stats.json"
    domain_stats = {}
    if stats_file.exists():
        with open(stats_file) as f:
            all_stats = json.load(f)
            domain_stats = all_stats.get(domain, {})

    return {
        "domain": domain,
        "adapter_available": adapter_path is not None,
        "adapter_name": adapter_path.stem if adapter_path else None,
        "expected_accuracy": domain_stats.get("pass_rate", "unknown"),
        "fact_count": domain_stats.get("fact_count", "unknown"),
        "recommendation": (
            "High confidence - domain adapter available"
            if adapter_path else
            "Lower confidence - using base model only"
        ),
    }


def list_supported_domains() -> list[str]:
    """List all domains with trained adapters."""
    adapters_dir = Path(__file__).parent.parent / "adapters"
    if not adapters_dir.exists():
        return []

    domains = set()
    for adapter in adapters_dir.glob("*.npz"):
        # Extract domain from adapter name
        name = adapter.stem
        # Remove common suffixes
        for suffix in ["_4b_adapter", "_adapter", "_orthogonal"]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        domains.add(name)

    return sorted(domains)
