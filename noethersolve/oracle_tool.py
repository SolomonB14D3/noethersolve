"""Oracle Tool — expose the oracle model + steering vectors + adapters as a verification service.

Default oracle: Qwen3-14B-Base (upgraded from 4B on 2026-03-25).
14B baseline 80.7% vs 4B ~40% — when 14B says something is wrong, it's
genuinely interesting, not capacity noise.

Triage pipeline:
1. Try steering vector first (0.1 KB, instant, covers "mute not dumb" domains)
2. Fall back to domain adapter if no vector or vector doesn't cover domain
3. Use base model if neither is available

Set NOETHERSOLVE_ORACLE_MODEL env var to override (e.g., "Qwen/Qwen3-4B-Base").

Usage:
    from noethersolve.oracle_tool import verify_claim, get_domain_confidence

    result = verify_claim(
        claim="The Riemann Hypothesis is proven",
        domain="mathematics"
    )
    # Returns: {"verdict": "FALSE", "confidence": 0.92, "correct_answer": "open problem", ...}
"""

import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Oracle model configuration
_DEFAULT_MODEL = "Qwen/Qwen3-14B-Base"
_ORACLE_MODEL = os.environ.get("NOETHERSOLVE_ORACLE_MODEL", _DEFAULT_MODEL)

# Model short name for adapter/vector directory lookup
_MODEL_SHORT_MAP = {
    "Qwen/Qwen3-4B-Base": "qwen3_4b_base",
    "Qwen/Qwen3-14B-Base": "qwen3_14b_base",
    "Qwen/Qwen3-8B-Base": "qwen3_8b_base",
}

# Lazy imports to avoid loading MLX until needed
_model = None
_tokenizer = None
_adapters = {}
_steering_vectors = {}
_steering_meta = None  # Cached steering results (which domains benefit from steering)
_current_model_name = None  # Track which model is loaded


@dataclass
class VerificationResult:
    """Result of claim verification."""
    claim: str
    verdict: str  # "TRUE", "FALSE", "UNCERTAIN"
    confidence: float  # 0.0 to 1.0
    domain: str
    adapter_used: Optional[str]  # intervention label (steering:X or adapter name or None)
    margin: float  # log-prob margin (positive = confident correct)
    explanation: str

    def __str__(self) -> str:
        intervention = self.adapter_used or "base model"
        return (
            f"Verdict: {self.verdict} (confidence: {self.confidence:.1%})\n"
            f"Domain: {self.domain}\n"
            f"Intervention: {intervention}\n"
            f"Margin: {self.margin:+.2f}\n"
            f"Explanation: {self.explanation}"
        )


def _model_short():
    """Get the short name for the current oracle model (for directory lookup)."""
    return _MODEL_SHORT_MAP.get(_ORACLE_MODEL, _ORACLE_MODEL.split("/")[-1].lower().replace("-", "_"))


def _load_model():
    """Lazy-load the model and tokenizer."""
    global _model, _tokenizer, _current_model_name
    if _model is not None and _current_model_name == _ORACLE_MODEL:
        return _model, _tokenizer

    try:
        from mlx_lm import load

        _model, _tokenizer = load(_ORACLE_MODEL)
        _current_model_name = _ORACLE_MODEL
        return _model, _tokenizer
    except ImportError:
        raise RuntimeError(
            "MLX not available. Install with: pip install mlx mlx-lm\n"
            "Or use the PyTorch backend on Linux/CUDA."
        )


def _load_steering_meta() -> dict:
    """Load steering vector metadata (which domains benefit from vectors)."""
    global _steering_meta
    if _steering_meta is not None:
        return _steering_meta

    # Try model-specific results file first, fall back to v2
    model_short = _model_short()
    meta_file = Path(__file__).parent.parent / "results" / f"steering_vectors_{model_short}.json"
    if not meta_file.exists():
        meta_file = Path(__file__).parent.parent / "results" / "steering_vectors_v2.json"

    if meta_file.exists():
        with open(meta_file) as f:
            entries = json.load(f)
        # Index by domain, only keep entries where steering improved accuracy
        _steering_meta = {}
        for entry in entries:
            domain = entry.get("domain", "")
            improvement = entry.get("improvement", 0)
            if improvement > 0:
                _steering_meta[domain] = {
                    "alpha": entry.get("best_alpha", 1.5),
                    "layer": entry.get("best_layer", 15),
                    "improvement": improvement,
                    "baseline": entry.get("baseline", 0),
                    "steered": entry.get("best_steered", 0),
                }
    else:
        _steering_meta = {}
    return _steering_meta


def _find_steering_vector(domain: str) -> Optional[tuple]:
    """Find a steering vector for the domain.

    Returns (vector_path, alpha, layer) if a beneficial vector exists, None otherwise.
    """
    meta = _load_steering_meta()

    # Model-specific vector directory, fall back to generic
    model_short = _model_short()
    vectors_dir = Path(__file__).parent.parent / "steering_vectors" / model_short
    if not vectors_dir.exists():
        vectors_dir = Path(__file__).parent.parent / "steering_vectors"

    def _try_find_vec(meta_domain, info):
        """Try to find vector file, checking metadata layer first then any layer."""
        # Try exact layer from metadata
        vec_path = vectors_dir / f"{meta_domain}_layer{info['layer']}.npy"
        if vec_path.exists():
            return vec_path, info["alpha"], info["layer"]
        # Try any available layer for this domain
        for candidate in vectors_dir.glob(f"{meta_domain}_layer*.npy"):
            # Extract layer number from filename
            stem = candidate.stem
            layer_str = stem.rsplit("_layer", 1)[-1]
            try:
                layer = int(layer_str)
            except ValueError:
                continue
            return candidate, info["alpha"], layer
        return None

    # Try exact match
    if domain in meta:
        result = _try_find_vec(domain, meta[domain])
        if result:
            return result

    # Try fuzzy match (domain substring)
    for meta_domain, info in meta.items():
        if domain.lower() in meta_domain.lower() or meta_domain.lower() in domain.lower():
            result = _try_find_vec(meta_domain, info)
            if result:
                return result

    return None


def _load_steering_vector(vec_path: Path):
    """Load and cache a steering vector."""
    global _steering_vectors
    key = str(vec_path)
    if key in _steering_vectors:
        return _steering_vectors[key]

    import numpy as np
    vec = np.load(vec_path)
    _steering_vectors[key] = vec
    return vec


def _get_log_prob_steered(model, tokenizer, prompt: str, completion: str,
                          steering_vec, alpha: float, layer: int) -> float:
    """Get log probability with steering vector applied at a specific layer.

    Hooks into the model's forward pass to add the steering vector to
    the hidden state at the specified layer.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    prompt_tokens = tokenizer.encode(prompt)
    comp_tokens = tokenizer.encode(completion)
    full_tokens = prompt_tokens + comp_tokens

    if not comp_tokens:
        return -999.0

    input_ids = mx.array([full_tokens])[..., :-1]

    # Get embeddings
    h = model.model.embed_tokens(input_ids)

    # Apply RMSNorm if present (Qwen uses pre-norm)
    # Run through transformer layers, injecting steering vector at target layer
    if hasattr(model.model, 'layers'):
        vec_mx = mx.array(steering_vec.astype(np.float32))
        for i, layer_module in enumerate(model.model.layers):
            h = layer_module(h, mask=None)
            if i == layer:
                # Add steering vector (broadcast across sequence dimension)
                h = h + alpha * vec_mx
    else:
        # Fallback: can't hook into layers, use base forward pass
        h = model.model(input_ids)

    # Apply final norm
    if hasattr(model.model, 'norm'):
        h = model.model.norm(h)

    # Project to logits
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(h)
    else:
        logits = h @ model.model.embed_tokens.weight.T

    # Calculate log probs for completion tokens only
    log_probs = nn.log_softmax(logits, axis=-1)

    completion_start = len(prompt_tokens) - 1
    total_log_prob = 0.0

    for i in range(completion_start, len(full_tokens) - 1):
        token_id = full_tokens[i + 1]
        total_log_prob += log_probs[0, i, token_id].item()

    return total_log_prob


def _find_adapter(domain: str) -> Optional[Path]:
    """Find the best adapter for a domain."""
    model_short = _model_short()
    adapters_dir = Path(__file__).parent.parent / "adapters" / model_short
    if not adapters_dir.exists():
        # Fall back to 4B adapters only if vocab matches (both Qwen3 use 151936)
        adapters_dir = Path(__file__).parent.parent / "adapters" / "qwen3_4b_base"
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


def _triage_intervention(domain: str) -> dict:
    """Determine the best intervention for a domain.

    Triage order:
    1. Steering vector (if domain has a beneficial vector) - cheapest
    2. Adapter (if domain has a trained adapter) - more powerful
    3. Base model (no intervention) - fallback

    Returns dict with intervention type and parameters.
    """
    # Check steering vector first (cheap, 0.1 KB)
    steering = _find_steering_vector(domain)
    if steering:
        vec_path, alpha, layer = steering
        return {
            "type": "steering",
            "vec_path": vec_path,
            "alpha": alpha,
            "layer": layer,
            "label": f"steering:{vec_path.stem}(α={alpha},L{layer})",
        }

    # Check adapter (more powerful, 50 MB)
    adapter_path = _find_adapter(domain)
    if adapter_path:
        return {
            "type": "adapter",
            "path": adapter_path,
            "label": adapter_path.stem,
        }

    # Base model fallback
    return {"type": "base", "label": None}


def _score_completions(model, tokenizer, prompt, completions, intervention):
    """Score completions using the appropriate intervention.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: The prompt text
        completions: List of (label, completion_text) tuples
        intervention: Dict from _triage_intervention

    Returns:
        List of (label, log_prob) tuples
    """
    results = []
    itype = intervention["type"]

    if itype == "steering":
        vec = _load_steering_vector(intervention["vec_path"])
        alpha = intervention["alpha"]
        layer = intervention["layer"]
        for label, comp in completions:
            lp = _get_log_prob_steered(model, tokenizer, prompt, comp, vec, alpha, layer)
            results.append((label, lp))

    elif itype == "adapter":
        adapter_weights = _load_adapter(intervention["path"])
        for label, comp in completions:
            lp = _get_log_prob(model, tokenizer, prompt, comp, adapter_weights)
            results.append((label, lp))

    else:  # base
        for label, comp in completions:
            lp = _get_log_prob(model, tokenizer, prompt, comp)
            results.append((label, lp))

    return results


def verify_fact(
    context: str,
    truth: str,
    distractors: list[str],
    domain: str = "general",
) -> VerificationResult:
    """Verify a fact using direct completion format (matches training).

    Uses triage pipeline: steering vector → adapter → base model.

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

    # Triage: steering → adapter → base
    intervention = _triage_intervention(domain)
    adapter_used = intervention["label"]

    # Build completions list
    completions = [("truth", f" {truth}")]
    for d in distractors:
        completions.append((d, f" {d}"))

    # Score all completions
    scores = _score_completions(model, tokenizer, context, completions, intervention)

    truth_lp = scores[0][1]
    dist_scores = scores[1:]
    best_distractor, best_distractor_lp = max(dist_scores, key=lambda x: x[1])

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
    method = intervention["type"]
    if verdict == "FALSE":
        explanation = f"Model prefers: '{best_distractor}' over the truth (via {method})"
    elif verdict == "TRUE":
        explanation = f"Model confidence supports this claim (via {method})"
    else:
        explanation = f"Insufficient confidence to determine truth value (via {method})"

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

    Uses triage pipeline: steering vector → adapter → base model.

    Args:
        claim: The claim to verify (should be a factual statement)
        domain: Domain for adapter selection (e.g., "chemistry", "physics")
        distractors: Alternative claims to compare against
        context: Optional context for the claim

    Returns:
        VerificationResult with verdict, confidence, and explanation
    """
    model, tokenizer = _load_model()

    # Triage: steering → adapter → base
    intervention = _triage_intervention(domain)
    adapter_used = intervention["label"]

    # Build single MC prompt with all options (forces model to compare simultaneously)
    all_options = [claim] + (distractors or [])
    labels = [chr(ord('A') + i) for i in range(len(all_options))]

    if context:
        prompt = f"{context}\n\nWhich statement is most accurate?\n"
    else:
        prompt = "Which statement is most accurate?\n"

    for i, (label, option) in enumerate(zip(labels, all_options)):
        prompt += f"{label}) {option}\n"
    prompt += "Answer:"

    # Score each answer letter in the same prompt context
    completions = [(labels[i], f" {labels[i]}") for i in range(len(all_options))]
    scores = _score_completions(model, tokenizer, prompt, completions, intervention)

    # Extract claim score and best distractor score
    claim_lp = scores[0][1]
    best_distractor_lp = float('-inf')
    best_distractor = None

    if distractors:
        for i, (label, lp) in enumerate(scores[1:], 1):
            if lp > best_distractor_lp:
                best_distractor_lp = lp
                best_distractor = all_options[i]

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
    method = intervention["type"]
    if verdict == "FALSE" and best_distractor:
        explanation = f"Model prefers: '{best_distractor}' over the claim (via {method})"
    elif verdict == "TRUE":
        explanation = f"Model confidence supports this claim (via {method})"
    else:
        explanation = f"Insufficient confidence to determine truth value (via {method})"

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

    Returns intervention availability (steering + adapter) and expected performance.
    """
    intervention = _triage_intervention(domain)
    adapter_path = _find_adapter(domain)
    steering = _find_steering_vector(domain)

    # Load domain stats if available
    stats_file = Path(__file__).parent.parent / "results" / "domain_stats.json"
    domain_stats = {}
    if stats_file.exists():
        with open(stats_file) as f:
            all_stats = json.load(f)
            domain_stats = all_stats.get(domain, {})

    # Recommendation based on triage
    if intervention["type"] == "steering":
        rec = f"Steering vector available (α={intervention['alpha']}, L{intervention['layer']})"
    elif intervention["type"] == "adapter":
        rec = "High confidence - domain adapter available"
    else:
        rec = "Lower confidence - using base model only"

    return {
        "domain": domain,
        "intervention_type": intervention["type"],
        "intervention_label": intervention["label"],
        "steering_available": steering is not None,
        "adapter_available": adapter_path is not None,
        "adapter_name": adapter_path.stem if adapter_path else None,
        "expected_accuracy": domain_stats.get("pass_rate", "unknown"),
        "fact_count": domain_stats.get("fact_count", "unknown"),
        "recommendation": rec,
    }


def list_supported_domains() -> list[str]:
    """List all domains with trained adapters or steering vectors."""
    domains = set()

    # Adapter domains (try model-specific first, then 4B fallback)
    model_short = _model_short()
    adapters_dir = Path(__file__).parent.parent / "adapters" / model_short
    if not adapters_dir.exists():
        adapters_dir = Path(__file__).parent.parent / "adapters" / "qwen3_4b_base"
    if adapters_dir.exists():
        for adapter in adapters_dir.glob("*.npz"):
            name = adapter.stem
            for suffix in ["_4b_adapter", "_adapter", "_orthogonal"]:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            domains.add(name)

    # Steering vector domains (from meta, which only includes beneficial ones)
    meta = _load_steering_meta()
    domains.update(meta.keys())

    return sorted(domains)
