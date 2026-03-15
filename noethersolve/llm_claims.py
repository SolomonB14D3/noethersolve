"""LLM Claims Auditor — validate claims about LLM capabilities, limitations,
and training against a curated database of established findings.

Consolidates 6 domains: hallucination, reasoning, alignment, training,
evaluation, and context/memory.  No ML required at runtime — all reference
data is hardcoded from peer-reviewed publications and reproducible findings.

Usage:
    from noethersolve import audit_llm_claims, check_llm_claim

    # Audit a list of claims
    report = audit_llm_claims([
        "RLHF eliminates sycophancy",
        "scaling laws follow power-law relationships",
        "chain-of-thought guarantees correct reasoning",
    ])
    print(report)  # Box-formatted report with verdicts

    # Check a single claim
    result = check_llm_claim("LLMs have perfect factual recall")
    print(result.verdict)   # "FALSE"
    print(result.evidence)  # Why it's wrong + references

    # Look up a topic
    info = get_llm_topic("sycophancy")
    print(info.domain, info.status, info.description)

    # List all topics by domain
    topics = list_llm_topics(domain="hallucination")

    # Check Chinchilla-optimal compute
    from noethersolve.llm_claims import chinchilla_optimal
    opt = chinchilla_optimal(params_B=7.0)
    print(f"Optimal tokens: {opt['tokens_B']:.1f}B")
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class LLMClaimIssue:
    """A single issue found when auditing an LLM claim."""

    severity: str
    """HIGH = factually false, MODERATE = misleading/overstated, LOW = imprecise,
    INFO = correct but with nuance."""

    claim: str
    """The original claim text."""

    description: str
    """Why this claim is problematic (or confirmed)."""

    domain: str = ""
    """Which domain this falls under (hallucination, reasoning, etc.)."""

    topic_id: str = ""
    """Matched topic ID in the database, if any."""

    references: List[str] = field(default_factory=list)
    """Supporting references."""

    def __str__(self) -> str:
        ref_str = ""
        if self.references:
            ref_str = f" [{', '.join(self.references)}]"
        return f"  [{self.severity}] {self.description}{ref_str}"


@dataclass
class LLMClaimReport:
    """Result of auditing a batch of LLM claims."""

    verdict: str
    """PASS = all claims correct, WARN = some imprecise, FAIL = false claims."""

    n_claims: int
    """Number of claims checked."""

    n_high: int = 0
    n_moderate: int = 0
    n_low: int = 0
    n_info: int = 0

    issues: List[LLMClaimIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_issues(self) -> int:
        return len(self.issues)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def __str__(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append(f"  LLM Claims Audit: {self.verdict}")
        lines.append("=" * 60)
        lines.append(
            f"  Summary: {self.n_claims} checked, "
            f"{self.n_high} HIGH, {self.n_moderate} MODERATE, "
            f"{self.n_low} LOW, {self.n_info} INFO"
        )
        lines.append("")
        if self.issues:
            lines.append("  Issues found:")
            sev_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "INFO": 3}
            for issue in sorted(
                self.issues, key=lambda i: sev_order.get(i.severity, 4)
            ):
                lines.append(str(issue))
            lines.append("")
        else:
            lines.append("  No issues detected.")
            lines.append("")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class LLMClaimResult:
    """Result of checking a single claim."""

    verdict: str
    """TRUE, FALSE, MISLEADING, NUANCED, or UNKNOWN."""

    claim: str
    """The original claim."""

    evidence: str
    """Explanation of the verdict."""

    domain: str = ""
    """Matched domain."""

    topic_id: str = ""
    """Matched topic ID."""

    confidence: float = 0.0
    """Match confidence (0-1). Higher = better keyword match."""

    references: List[str] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.verdict in ("TRUE", "NUANCED")


@dataclass
class LLMTopicInfo:
    """Information about a single LLM science topic."""

    topic_id: str
    domain: str
    status: str
    """ESTABLISHED, OPEN, DEBATED, PARTIAL."""

    description: str
    truth: str
    misconceptions: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


# ── Chinchilla Scaling ───────────────────────────────────────────────────

# Hoffmann et al. 2022 — "Training Compute-Optimal Large Language Models"
# Approach 3 (parametric fit): L(N,D) = E + A/N^alpha + B/D^beta
# Optimal ratio: D_opt ≈ 20 * N (tokens ≈ 20× parameters)
_CHINCHILLA_RATIO = 20.0

# Kaplan et al. 2020 scaling exponents (for loss prediction)
_KAPLAN_ALPHA = 0.076   # loss vs params exponent
_KAPLAN_BETA = 0.095    # loss vs data exponent


def chinchilla_optimal(
    params_B: Optional[float] = None,
    tokens_B: Optional[float] = None,
    compute_flops: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute Chinchilla-optimal training configuration.

    Given one of {params, tokens, compute}, compute the other two
    for compute-optimal training per Hoffmann et al. 2022.

    Parameters
    ----------
    params_B : float, optional
        Model parameters in billions.
    tokens_B : float, optional
        Training tokens in billions.
    compute_flops : float, optional
        Total training FLOPs (≈ 6 * N * D).

    Returns
    -------
    dict
        Keys: params_B, tokens_B, compute_flops, ratio,
        is_chinchilla_optimal (bool), notes.
    """
    result: Dict[str, Any] = {"ratio": _CHINCHILLA_RATIO}

    if params_B is not None and tokens_B is not None:
        actual_ratio = tokens_B / params_B if params_B > 0 else 0
        optimal_tokens = params_B * _CHINCHILLA_RATIO
        result["params_B"] = params_B
        result["tokens_B"] = tokens_B
        result["compute_flops"] = 6 * params_B * 1e9 * tokens_B * 1e9
        result["actual_ratio"] = round(actual_ratio, 1)
        result["optimal_tokens_B"] = round(optimal_tokens, 1)
        # Within 2× is "roughly optimal"
        is_opt = 0.5 * _CHINCHILLA_RATIO <= actual_ratio <= 2 * _CHINCHILLA_RATIO
        result["is_chinchilla_optimal"] = is_opt
        if actual_ratio < _CHINCHILLA_RATIO * 0.5:
            result["notes"] = (
                f"Undertrained: {actual_ratio:.0f}× ratio vs {_CHINCHILLA_RATIO:.0f}× optimal. "
                f"Needs ~{optimal_tokens:.0f}B tokens."
            )
        elif actual_ratio > _CHINCHILLA_RATIO * 2:
            result["notes"] = (
                f"Overtrained: {actual_ratio:.0f}× ratio vs {_CHINCHILLA_RATIO:.0f}× optimal. "
                f"Could use a larger model."
            )
        else:
            result["notes"] = f"Approximately compute-optimal ({actual_ratio:.0f}× ratio)."
    elif params_B is not None:
        tokens = params_B * _CHINCHILLA_RATIO
        result["params_B"] = params_B
        result["tokens_B"] = round(tokens, 1)
        result["compute_flops"] = 6 * params_B * 1e9 * tokens * 1e9
        result["is_chinchilla_optimal"] = True
        result["notes"] = (
            f"Optimal for {params_B:.1f}B params: "
            f"{tokens:.0f}B tokens, {result['compute_flops']:.2e} FLOPs."
        )
    elif tokens_B is not None:
        params = tokens_B / _CHINCHILLA_RATIO
        result["params_B"] = round(params, 2)
        result["tokens_B"] = tokens_B
        result["compute_flops"] = 6 * params * 1e9 * tokens_B * 1e9
        result["is_chinchilla_optimal"] = True
        result["notes"] = (
            f"Optimal for {tokens_B:.0f}B tokens: "
            f"{params:.1f}B params."
        )
    elif compute_flops is not None:
        # C ≈ 6ND, D = 20N → C = 120N² → N = sqrt(C/120)
        n = math.sqrt(compute_flops / (6 * _CHINCHILLA_RATIO * 1e18))
        d = n * _CHINCHILLA_RATIO
        result["params_B"] = round(n, 2)
        result["tokens_B"] = round(d, 1)
        result["compute_flops"] = compute_flops
        result["is_chinchilla_optimal"] = True
        result["notes"] = (
            f"Optimal for {compute_flops:.2e} FLOPs: "
            f"{n:.1f}B params, {d:.0f}B tokens."
        )
    else:
        raise ValueError("Provide at least one of: params_B, tokens_B, compute_flops")

    return result


# ── Known Benchmark Results ──────────────────────────────────────────────
# Approximate published scores for reference. Ranges account for
# evaluation methodology differences (0-shot vs 5-shot, etc.).

_BENCHMARK_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "mmlu": {
        "gpt-4": (86.0, 87.5),
        "gpt-4o": (87.0, 88.7),
        "claude-3.5-sonnet": (88.0, 89.0),
        "claude-3-opus": (86.0, 87.0),
        "llama-3.1-405b": (85.0, 88.0),
        "llama-3.1-70b": (79.0, 83.0),
        "llama-3.1-8b": (66.0, 69.0),
        "qwen2.5-72b": (85.0, 86.5),
        "qwen2.5-7b": (74.0, 76.0),
        "gemini-1.5-pro": (85.0, 86.0),
        "random-baseline": (25.0, 25.0),
    },
    "truthfulqa": {
        "gpt-4": (59.0, 62.0),
        "claude-3-opus": (60.0, 65.0),
        "llama-3.1-70b": (55.0, 62.0),
        "llama-3.1-8b": (45.0, 52.0),
        "random-baseline": (25.0, 25.0),
    },
    "humaneval": {
        "gpt-4": (67.0, 87.0),
        "gpt-4o": (87.0, 91.0),
        "claude-3.5-sonnet": (92.0, 93.7),
        "llama-3.1-405b": (61.0, 89.0),
        "random-baseline": (0.0, 0.0),
    },
    "hellaswag": {
        "gpt-4": (95.0, 96.0),
        "llama-3.1-70b": (85.0, 88.0),
        "llama-3.1-8b": (79.0, 82.0),
        "random-baseline": (25.0, 25.0),
    },
}


def check_benchmark_score(
    model: str, benchmark: str, claimed_score: float
) -> LLMClaimResult:
    """Check if a claimed benchmark score is plausible.

    Parameters
    ----------
    model : str
        Model name (fuzzy matched).
    benchmark : str
        Benchmark name (mmlu, truthfulqa, humaneval, hellaswag).
    claimed_score : float
        The claimed score (0-100).

    Returns
    -------
    LLMClaimResult
    """
    bench_key = _normalize_benchmark(benchmark)
    model_key = _normalize_model(model)

    if bench_key not in _BENCHMARK_RANGES:
        return LLMClaimResult(
            verdict="UNKNOWN",
            claim=f"{model} scores {claimed_score} on {benchmark}",
            evidence=f"Benchmark '{benchmark}' not in reference database. "
            f"Known benchmarks: {', '.join(sorted(_BENCHMARK_RANGES))}.",
            domain="evaluation",
        )

    bench = _BENCHMARK_RANGES[bench_key]
    if model_key not in bench:
        return LLMClaimResult(
            verdict="UNKNOWN",
            claim=f"{model} scores {claimed_score} on {benchmark}",
            evidence=f"Model '{model}' not in reference database for {benchmark}. "
            f"Known models: {', '.join(sorted(bench))}.",
            domain="evaluation",
        )

    lo, hi = bench[model_key]
    if lo <= claimed_score <= hi:
        return LLMClaimResult(
            verdict="TRUE",
            claim=f"{model} scores {claimed_score} on {benchmark}",
            evidence=f"Consistent with published range [{lo:.1f}, {hi:.1f}].",
            domain="evaluation",
            confidence=0.9,
            references=["Published benchmarks"],
        )
    elif claimed_score < lo:
        return LLMClaimResult(
            verdict="FALSE",
            claim=f"{model} scores {claimed_score} on {benchmark}",
            evidence=f"Below published range [{lo:.1f}, {hi:.1f}]. "
            f"Score {claimed_score:.1f} is {lo - claimed_score:.1f} points under.",
            domain="evaluation",
            confidence=0.8,
        )
    else:
        return LLMClaimResult(
            verdict="FALSE",
            claim=f"{model} scores {claimed_score} on {benchmark}",
            evidence=f"Above published range [{lo:.1f}, {hi:.1f}]. "
            f"Score {claimed_score:.1f} is {claimed_score - hi:.1f} points over.",
            domain="evaluation",
            confidence=0.8,
        )


# ── Topic Database ───────────────────────────────────────────────────────

def _build_database() -> Dict[str, LLMTopicInfo]:
    """Build the LLM claims reference database. Called once at module load."""
    db: Dict[str, LLMTopicInfo] = {}

    def _add(
        topic_id: str,
        domain: str,
        status: str,
        description: str,
        truth: str,
        misconceptions: List[str],
        keywords: List[str],
        references: Optional[List[str]] = None,
    ):
        db[topic_id] = LLMTopicInfo(
            topic_id=topic_id,
            domain=domain,
            status=status,
            description=description,
            truth=truth,
            misconceptions=misconceptions,
            keywords=keywords,
            references=references or [],
        )

    # ── Hallucination domain ─────────────────────────────────────────
    _add(
        "hallucination_definition", "hallucination", "ESTABLISHED",
        "LLM hallucination is fluent, confident, but factually incorrect output",
        "LLMs generate plausible-sounding but fabricated content with high confidence",
        [
            "hallucination is always detectable by the model itself",
            "only small models hallucinate",
            "scaling eliminates hallucination",
        ],
        ["hallucination", "hallucinate", "fabricat", "confabulat"],
        ["Ji et al. 2023 — Survey of Hallucination in NLG"],
    )
    _add(
        "rag_limitations", "hallucination", "ESTABLISHED",
        "RAG reduces but does not eliminate hallucination",
        "Models may ignore retrieved context or hallucinate beyond it",
        [
            "RAG guarantees factual accuracy",
            "retrieval is always perfect",
            "hallucination is impossible with external knowledge",
        ],
        ["rag", "retrieval", "augment", "ground"],
        ["Lewis et al. 2020 — RAG"],
    )
    _add(
        "citation_fabrication", "hallucination", "ESTABLISHED",
        "LLMs fabricate citations — fake papers, authors, DOIs",
        "Citation accuracy is unreliable; models generate plausible but non-existent references",
        [
            "citation accuracy exceeds 95%",
            "models only cite papers from training",
            "fake citations are self-detected",
        ],
        ["citation", "reference", "doi", "paper", "fabricat"],
        ["Agrawal et al. 2024 — Citation verification"],
    )
    _add(
        "confidence_calibration", "hallucination", "ESTABLISHED",
        "LLMs express high confidence even when wrong (overconfidence on errors)",
        "Model confidence does not reliably indicate correctness",
        [
            "models are well-calibrated by default",
            "low confidence always indicates errors",
            "calibration improves with scale alone",
        ],
        ["calibrat", "confidence", "overconfiden", "uncertain"],
        ["Kadavath et al. 2022 — Language models know what they don't know"],
    )
    _add(
        "entity_confusion", "hallucination", "ESTABLISHED",
        "Models blend attributes of entities with similar names or contexts",
        "Entity hallucination is caused by attribute mixing in distributed representations",
        [
            "entities are stored in separate memory banks",
            "confusion only affects fictional entities",
            "larger models never confuse entities",
        ],
        ["entity", "confusion", "blend", "attribut"],
    )
    _add(
        "factual_recall_frequency", "hallucination", "ESTABLISHED",
        "Factual recall degrades for long-tail entities seen rarely in training",
        "Recall quality correlates with training frequency; rare facts are unreliable",
        [
            "common knowledge degrades more than rare facts",
            "all facts degrade equally",
            "no correlation with training frequency",
        ],
        ["factual", "recall", "long-tail", "rare", "frequenc"],
        ["Mallen et al. 2023 — When Not to Trust Language Models"],
    )

    # ── Reasoning domain ─────────────────────────────────────────────
    _add(
        "chain_of_thought", "reasoning", "ESTABLISHED",
        "CoT elicits intermediate reasoning steps but doesn't guarantee correctness",
        "Chain-of-thought improves performance by guiding intermediate steps, "
        "but stated steps may not reflect actual computation (unfaithful reasoning)",
        [
            "CoT accesses a dedicated reasoning module",
            "CoT guarantees logical correctness",
            "CoT always reflects true internal reasoning",
        ],
        ["chain-of-thought", "cot", "step-by-step", "reasoning"],
        ["Wei et al. 2022 — Chain-of-Thought Prompting",
         "Turpin et al. 2023 — Language Models Don't Always Say What They Think"],
    )
    _add(
        "reversal_curse", "reasoning", "ESTABLISHED",
        "Learning 'A is B' does not imply learning 'B is A' (asymmetric)",
        "LLMs store directional associations; reversal requires explicit training",
        [
            "knowledge is symmetric by default",
            "only affects bidirectional models",
            "reversal curse is a solved problem",
        ],
        ["reversal", "curse", "asymmetr", "bidirection", "symmetric"],
        ["Berglund et al. 2023 — The Reversal Curse"],
    )
    _add(
        "multi_hop_degradation", "reasoning", "ESTABLISHED",
        "Multi-hop reasoning accuracy degrades exponentially with chain length",
        "A→B, B→C chains compound errors; accuracy drops sharply with more hops",
        [
            "LLMs excel at arbitrary-length chains",
            "degradation is linear with chain length",
            "multi-hop is easier than single-hop",
        ],
        ["multi-hop", "multi hop", "chain", "reason", "degrad"],
        ["Press et al. 2023 — Measuring and Narrowing the Compositionality Gap"],
    )
    _add(
        "negation_failures", "reasoning", "ESTABLISHED",
        "LLMs often ignore or misprocess 'not' and negative constructions",
        "Negation handling is a known weakness; models frequently treat negated "
        "statements as if the negation were absent",
        [
            "negation is handled perfectly",
            "only double negatives cause issues",
            "attention mechanisms specialize in negation",
        ],
        ["negat", "not ", "negative"],
        ["Kassner & Schutze 2020 — Negated and Misprimed Probes"],
    )
    _add(
        "math_pattern_matching", "reasoning", "ESTABLISHED",
        "Math errors stem from pattern matching on surface forms, not symbolic manipulation",
        "LLMs approximate arithmetic via learned patterns; they do not perform "
        "exact symbolic computation internally",
        [
            "LLMs perform exact arithmetic",
            "math errors are random",
            "calculator tools are never needed",
        ],
        ["math", "arithmetic", "symbolic", "calculat"],
        ["Dziri et al. 2024 — Faith and Fate"],
    )
    _add(
        "planning_limitations", "reasoning", "ESTABLISHED",
        "Long-horizon planning is weak — no persistent state, goal inconsistency",
        "Autoregressive generation lacks persistent state tracking; plans degrade over steps",
        [
            "planning is a core LLM capability",
            "autoregressive generation aids planning",
            "context window fully determines planning ability",
        ],
        ["planning", "long-horizon", "goal", "persistent state"],
        ["Valmeekam et al. 2023 — On the Planning Abilities of LLMs"],
    )
    _add(
        "compositional_generalization", "reasoning", "ESTABLISHED",
        "Models fail to combine known primitives in novel ways not seen in training",
        "Compositional generalization remains an open challenge despite scale",
        [
            "composition is a core strength",
            "only affects out-of-vocabulary words",
            "larger models fully solve compositionality",
        ],
        ["composit", "generali", "primitiv", "novel combin"],
        ["Lake & Baroni 2018 — Generalization Without Systematicity"],
    )

    # ── Alignment domain ─────────────────────────────────────────────
    _add(
        "rlhf_limitations", "alignment", "ESTABLISHED",
        "RLHF optimizes for human preference ratings, not true human values",
        "Preference optimization is a proxy; reward hacking and sycophancy are known failure modes",
        [
            "RLHF perfectly aligns with human values",
            "RLHF eliminates all harmful outputs",
            "RLHF eliminates sycophancy",
        ],
        ["rlhf", "reinforcement", "human feedback", "preference"],
        ["Casper et al. 2023 — Open Problems in RLHF"],
    )
    _add(
        "sycophancy", "alignment", "ESTABLISHED",
        "RLHF-trained models agree with users even when wrong, to maximize approval",
        "Sycophancy is a systematic bias from preference optimization, not a bug",
        [
            "models always correct user errors",
            "neutral disagreement by default",
            "sycophancy is a solved problem",
        ],
        ["sycophancy", "sycophant", "agree", "approval", "flatter"],
        ["Sharma et al. 2023 — Towards Understanding Sycophancy"],
    )
    _add(
        "reward_hacking", "alignment", "ESTABLISHED",
        "Models exploit reward model weaknesses for high scores without true alignment",
        "Reward hacking is Goodhart's Law applied to RLHF — proxy optimization diverges",
        [
            "reward models are unhackable",
            "hacking improves actual performance",
            "only affects non-language tasks",
        ],
        ["reward hack", "goodhart", "proxy", "exploit"],
        ["Skalse et al. 2022 — Defining and Characterizing Reward Hacking"],
    )
    _add(
        "jailbreaking", "alignment", "ESTABLISHED",
        "Safety training doesn't generalize to all adversarial prompt formulations",
        "Jailbreaks exploit gaps in safety training distribution; new attacks keep emerging",
        [
            "alignment is mathematically guaranteed",
            "jailbreaks are impossible with RLHF",
            "only affects unaligned models",
        ],
        ["jailbreak", "adversarial", "safety", "bypass", "attack"],
        ["Wei et al. 2024 — Jailbroken"],
    )
    _add(
        "deceptive_alignment", "alignment", "DEBATED",
        "Can't distinguish genuine alignment from strategic behavior during evaluation",
        "Deceptive alignment (mesa-optimization) is theoretically possible but not yet demonstrated",
        [
            "deception is impossible for LLMs",
            "interpretability fully solves detection",
            "models can't model the training process",
        ],
        ["decepti", "mesa-optim", "inner alignment", "strategic"],
        ["Hubinger et al. 2019 — Risks from Learned Optimization"],
    )
    _add(
        "constitutional_ai", "alignment", "ESTABLISHED",
        "Constitutional AI uses AI-generated critiques based on explicit principles",
        "CAI reduces reliance on human feedback by having the model self-critique",
        [
            "only human feedback, no AI involvement",
            "based on legal constitutional documents",
            "removes the need for any principles",
        ],
        ["constitutional", "cai", "self-critique", "principle"],
        ["Bai et al. 2022 — Constitutional AI"],
    )
    _add(
        "goodhart_alignment", "alignment", "ESTABLISHED",
        "Optimizing a proxy metric for alignment causes divergence from true alignment",
        "Goodhart's Law applies to all alignment metrics — no finite proxy captures values exactly",
        [
            "proxy metrics are perfect measures",
            "more optimization always helps",
            "Goodhart doesn't apply to AI",
        ],
        ["goodhart", "proxy", "metric", "diverge"],
    )

    # ── Training domain ──────────────────────────────────────────────
    _add(
        "scaling_laws", "training", "ESTABLISHED",
        "Performance follows power-law relationships with compute, data, and parameters",
        "Neural scaling laws (Kaplan et al. 2020, Hoffmann et al. 2022) show "
        "predictable loss reduction with scale",
        [
            "linear improvement with scale",
            "sudden jumps at specific sizes",
            "no predictable relationship",
        ],
        ["scaling law", "power law", "kaplan", "chinchilla"],
        ["Kaplan et al. 2020", "Hoffmann et al. 2022 — Chinchilla"],
    )
    _add(
        "emergence", "training", "DEBATED",
        "Some capabilities appear suddenly at certain scales (emergent abilities)",
        "Whether emergence is real or a metric artifact is actively debated; "
        "Schaeffer et al. 2023 argue it's often a measurement artifact",
        [
            "gradual linear improvement only",
            "abilities present at all scales",
            "emergence has been debunked entirely",
        ],
        ["emergen", "sudden", "phase transition", "capability jump"],
        ["Wei et al. 2022 — Emergent Abilities",
         "Schaeffer et al. 2023 — Are Emergent Abilities a Mirage?"],
    )
    _add(
        "chinchilla_undertrained", "training", "ESTABLISHED",
        "Many early LLMs were undertrained (needed more tokens relative to params)",
        "Chinchilla showed optimal compute allocation requires ~20× more tokens than parameters",
        [
            "most models are overtrained with too much data",
            "models are optimally trained by default",
            "parameter count was always too low",
        ],
        ["chinchilla", "undertrain", "compute-optimal", "token ratio"],
        ["Hoffmann et al. 2022 — Training Compute-Optimal Large Language Models"],
    )
    _add(
        "data_quality", "training", "ESTABLISHED",
        "Data quality directly influences capabilities, biases, and factual accuracy",
        "Garbage in, garbage out — data quality matters more than quantity beyond a threshold",
        [
            "no effect if quantity is sufficient",
            "only affects rare edge cases",
            "fully correctable post-training",
        ],
        ["data quality", "quality", "curat", "filter"],
    )
    _add(
        "catastrophic_forgetting", "training", "ESTABLISHED",
        "Fine-tuning causes loss of pretrained capabilities (catastrophic forgetting)",
        "Standard fine-tuning overwrites general knowledge; mitigation requires "
        "LoRA, replay, or other techniques",
        [
            "perfect retention of all knowledge during fine-tuning",
            "only affects small models",
            "a solved problem with modern techniques",
        ],
        ["catastrophic forget", "forget", "fine-tun", "overwrite"],
        ["Kirkpatrick et al. 2017 — Overcoming Catastrophic Forgetting"],
    )
    _add(
        "benchmark_contamination", "training", "ESTABLISHED",
        "Test data appearing in training inflates reported performance",
        "Benchmark contamination is widespread and often undetected; paraphrased "
        "versions evade exact matching",
        [
            "contamination is impossible",
            "contamination improves genuine capabilities",
            "all benchmarks prevent contamination",
        ],
        ["contaminat", "data leak", "train.*test", "benchmark.*train"],
        ["Jacovi et al. 2023 — Stop Uploading Test Data"],
    )
    _add(
        "lora_peft", "training", "ESTABLISHED",
        "LoRA/PEFT trains small additional parameters while freezing pretrained weights",
        "Parameter-efficient methods achieve near-full-fine-tuning performance at fraction of cost",
        [
            "reduces model size permanently",
            "trains all parameters more efficiently",
            "only works for classification tasks",
        ],
        ["lora", "peft", "adapter", "parameter-efficient", "freeze"],
        ["Hu et al. 2022 — LoRA"],
    )

    # ── Evaluation domain ────────────────────────────────────────────
    _add(
        "benchmark_saturation", "evaluation", "ESTABLISHED",
        "Models approaching ceiling performance makes benchmarks uninformative",
        "Saturated benchmarks can't discriminate between models; need harder evals",
        [
            "benchmarks become more discriminative at ceiling",
            "saturation indicates perfect capability",
            "new models can't reach saturation",
        ],
        ["saturat", "ceiling", "uninformat", "discriminat"],
    )
    _add(
        "llm_as_judge_bias", "evaluation", "ESTABLISHED",
        "LLM-as-judge inherits biases and favors outputs similar to its own style",
        "Self-preferencing, verbosity bias, and position bias are documented",
        [
            "LLM judges are perfectly objective",
            "no correlation with human judgment",
            "strictly better than human evaluation",
        ],
        ["llm.*judge", "judge", "evaluat.*bias", "self-prefer"],
        ["Zheng et al. 2023 — Judging LLM-as-a-Judge"],
    )
    _add(
        "capability_elicitation", "evaluation", "ESTABLISHED",
        "Performance varies greatly with prompt format and evaluation methodology",
        "The same model can look very different under different prompting strategies",
        [
            "capabilities are invariant to prompting",
            "one prompt reveals all abilities",
            "elicitation is a solved problem",
        ],
        ["elicit", "prompt format", "eval method"],
    )
    _add(
        "mc_position_bias", "evaluation", "ESTABLISHED",
        "Multiple-choice evaluation can be gamed via answer position biases",
        "Models have systematic biases toward certain answer positions (often A or C)",
        [
            "multiple-choice is ungameable",
            "position biases don't exist",
            "answer position never affects selection",
        ],
        ["multiple-choice", "position bias", "answer.*bias", "mc "],
    )

    # ── Context & Memory domain ──────────────────────────────────────
    _add(
        "lost_in_the_middle", "context_memory", "ESTABLISHED",
        "LLMs attend poorly to information in the middle of long contexts",
        "U-shaped attention curve: beginning and end are favored, middle is neglected",
        [
            "uniform attention across all positions",
            "only the start is affected",
            "solved with longer context windows",
        ],
        ["lost in the middle", "middle", "u-shaped", "position"],
        ["Liu et al. 2023 — Lost in the Middle"],
    )
    _add(
        "kv_cache_scaling", "context_memory", "ESTABLISHED",
        "KV cache memory grows linearly with sequence length — deployment bottleneck",
        "For a model with d_model dimensions and L layers, KV cache = O(L × d × seq_len)",
        [
            "constant regardless of length",
            "logarithmic growth",
            "negligible compared to model weights",
        ],
        ["kv cache", "memory", "sequence length", "bottleneck"],
    )
    _add(
        "stateless_api", "context_memory", "ESTABLISHED",
        "LLMs are stateless between API calls — no persistent memory across calls",
        "LLMs are stateless; no persistent memory exists between API calls — all "
        "context must be re-provided each time",
        [
            "models remember all past conversations",
            "state persists across sessions automatically",
            "memory is unlimited between calls",
        ],
        ["stateless", "memory", "persist", "session", "api call"],
    )
    _add(
        "knowledge_editing", "context_memory", "PARTIAL",
        "Updating factual knowledge without retraining is difficult — distributed storage",
        "Knowledge is spread across parameters; surgical editing remains unreliable",
        [
            "single parameters store single facts",
            "knowledge editing is reliable and surgical",
            "updates require no compute",
        ],
        ["knowledge edit", "fact edit", "updat", "localiz"],
        ["Meng et al. 2022 — ROME"],
    )
    _add(
        "context_degradation", "context_memory", "ESTABLISHED",
        "Effective context utilization degrades even when longer contexts are supported",
        "Models can technically process long contexts but performance drops with length",
        [
            "longer context always improves performance",
            "interpolation gives perfect utilization",
            "no quality degradation occurs",
        ],
        ["context.*degrad", "long context", "utiliz", "interpolat"],
    )

    # ── Expression vs Knowledge domain ────────────────────────────
    # Based on Papers 6-7 (Expression Bottleneck, Contrastive Pretraining)
    _add(
        "expression_bottleneck", "expression_knowledge", "ESTABLISHED",
        "Small models have internal knowledge but can't express it (expression bottleneck)",
        "Logit-level accuracy is 41.0% for ALL vanilla models from 3M to 64M parameters — "
        "identical probe-by-probe (123/300 correct). The bottleneck is format generation, "
        "not knowledge. Contrastive decoding rescues expression (0.7%→38% accuracy at d=88).",
        [
            "small models lack behavioral discrimination",
            "small models can't discriminate between biased and correct answers",
            "behavioral emergence requires scale",
            "small models don't know the answer",
            "knowledge only emerges at large scale",
        ],
        ["expression", "bottleneck", "small model", "41%", "format", "generation",
         "logit.*accura", "internal knowledge", "small.*discriminat", "small.*behavioral",
         "small.*knowledge", "small.*bias"],
        ["Sanchez 2026 — Expression Bottleneck (Paper 7, DOI: 10.5281/zenodo.18895248)"],
    )
    _add(
        "contrastive_decoding_rescue", "expression_knowledge", "ESTABLISHED",
        "Contrastive decoding rescues muted models by subtracting the LM prior at inference",
        "At d=88 (width below phase transition), contrastive decoding (expert - α×amateur logits) "
        "restores format generation: 0.7%→38.0% accuracy (54× improvement), exceeding d=96 "
        "baseline (30.0%). The behavioral signal was always present in hidden states.",
        [
            "contrastive decoding only helps large models",
            "activation steering is more effective than contrastive decoding",
            "hidden state interventions are stronger than logit interventions",
        ],
        ["contrastive decod", "logit subtract", "expert.*amateur", "inference.*interven"],
        ["Sanchez 2026 — Expression Bottleneck (Paper 7)"],
    )
    _add(
        "intervention_hierarchy", "expression_knowledge", "ESTABLISHED",
        "Inference interventions follow a strict hierarchy: token > logit >> hidden state",
        "Token-level (format forcing) recovers 41% on ALL models. Logit-level (contrastive "
        "decoding) recovers 38%. Hidden-state (activation steering): NULL result (3.7% "
        "unchanged). The generation bottleneck is at the last step: hidden states → token "
        "emissions. Interventions after the mapping work; interventions before it fail.",
        [
            "activation steering is effective for behavior control",
            "hidden state interventions are the strongest",
            "steering vectors reliably change outputs",
        ],
        ["intervention", "hierarchy", "token.*logit", "activation steer",
         "steering vector", "hidden state"],
        ["Sanchez 2026 — Expression Bottleneck (Paper 7)"],
    )
    _add(
        "universal_41_percent", "expression_knowledge", "ESTABLISHED",
        "Logit-forced bias accuracy is exactly 41.0% (123/300) for every model 3M-64M",
        "The number is not approximate — it is EXACTLY 123 correct, 64 biased, 113 neutral "
        "for every vanilla and sycophancy-only model tested (13 models). Sycophancy training "
        "does NOT change bias logit preferences. Only bias+syco combined nudges to 47%. "
        "This is a universal property of training data + tokenizer, independent of capacity.",
        [
            "larger models have better internal discrimination",
            "sycophancy training improves bias discrimination",
            "logit accuracy scales with model size",
        ],
        ["41%", "universal", "logit.*forced", "probe.*identical", "123/300"],
        ["Sanchez 2026 — Expression Bottleneck (Paper 7)"],
    )
    _add(
        "width_phase_transition", "expression_knowledge", "ESTABLISHED",
        "Cross-dimensional transfer requires d_model ≥ 96 — a sharp architectural threshold",
        "Width sweep (d=64,80,88,96) shows cross-transfer ρ ≤ 0.010 for d≤88, jumps to "
        "0.290 at d=96. Per-probe logit gaps are r=0.997 identical between d=88 and d=96. "
        "P(correct) distributions identical. The ENTIRE 29× ρ gap is format generation: "
        "d=88 is 96% unparsed, d=96 is 14% unparsed. Width gates expression, not knowledge.",
        [
            "behavioral transfer scales smoothly with width",
            "wider models are always better at transfer",
            "phase transitions don't exist in neural networks",
        ],
        ["width", "d_model", "phase transition", "d=96", "d=88", "architectural threshold"],
        ["Sanchez 2026 — Expression Bottleneck (Paper 7)",
         "Sanchez 2026 — Contrastive Pretraining (Paper 6, DOI: 10.5281/zenodo.18870555)"],
    )

    # ── Behavioral Transfer domain ────────────────────────────────
    # Based on Papers 3, 6 (Scale Ladder, Contrastive Pretraining)
    _add(
        "contrastive_injection", "behavioral_transfer", "ESTABLISHED",
        "5% contrastive data injection breaks the behavioral emergence wall at small scale",
        "At 7M parameters, 5% contrastive bias+sycophancy injection achieves ρ=0.431 (bias) "
        "and ρ=0.513 (sycophancy) — exceeding vanilla 34M/64M performance at 5× fewer "
        "parameters. 10% injection is worse than 5% (diminishing returns + factual cost). "
        "Behavioral emergence is a data quality threshold, not a scale threshold.",
        [
            "behavioral emergence requires large scale",
            "only parameter count determines behavioral capability",
            "more injection data is always better",
            "10% injection is better than 5%",
        ],
        ["contrastive", "injection", "5%", "behavioral.*wall", "data quality",
         "contrastive.*pretrain", "behavioral.*emergence.*scale",
         "behavioral.*require.*scale", "emergence.*large"],
        ["Sanchez 2026 — Contrastive Pretraining (Paper 6, DOI: 10.5281/zenodo.18870555)"],
    )
    _add(
        "cross_transfer_asymmetry", "behavioral_transfer", "ESTABLISHED",
        "Sycophancy training improves bias (cross-transfer), but NOT vice versa",
        "Sycophancy-only injection at 7M lifts bias ρ from 0 to 0.208 (cross-transfer). "
        "Bias-only injection never lifts sycophancy. Broad behavioral skills (sycophancy = "
        "\"don't just agree\") transfer to narrow ones (bias = \"don't stereotype\"), "
        "but narrow skills don't transfer to broad ones. Behavioral skills have a hierarchy.",
        [
            "all behavioral improvements are independent",
            "bias training fixes sycophancy",
            "behavioral skills are symmetric",
        ],
        ["cross-transfer", "cross transfer", "sycophan.*bias", "bias.*sycophan",
         "behavioral hierarch", "transfer asymmetr"],
        ["Sanchez 2026 — Contrastive Pretraining (Paper 6)",
         "Rimsky et al. ACL 2024 — Sycophancy to truthfulness transfer"],
    )
    _add(
        "geometry_precedes_emergence", "behavioral_transfer", "ESTABLISHED",
        "Activation geometry changes (SVD spectrum) precede behavioral emergence",
        "Effective dimension breakout appears before behavioral ρ across 5 scales. "
        "Every contrastive injection shifts SVD spectrum even at zero behavioral ρ "
        "(silent geometry). Two signatures: inflationary (SV1 grows, null injections) "
        "vs deconcentrating (SV1 drops, productive injections). SVD spectrum is a "
        "leading indicator; ρ-eval is lagging.",
        [
            "behavioral change is immediate",
            "geometry doesn't change until behavior does",
            "SVD spectrum changes are noise",
        ],
        ["geometry", "svd", "spectrum", "precede", "emerge", "silent",
         "effective dimension", "eff_dim"],
        ["Sanchez 2026 — Scale Ladder Phase Transitions (Paper 3, DOI: 10.5281/zenodo.18865198)"],
    )
    _add(
        "deconcentration_score", "behavioral_transfer", "ESTABLISHED",
        "Deconcentration score separates productive from null injections with 100% accuracy",
        "decon = (1 - SV1_post/SV1_van) × (eff_post/eff_van). Direct injection: decon > 1.0. "
        "Null: |decon| < 0.16. Deconcentration does NOT correlate with factual regression "
        "(ρ = −0.010, p = 0.97). Geometric restructuring is 'free' — factual cost comes "
        "from token displacement (injection rate), not from deconcentration.",
        [
            "geometric changes always cause factual regression",
            "there's no way to detect productive training",
            "SVD changes are random noise",
        ],
        ["deconcentration", "decon", "productive.*null", "sv1", "factual regression"],
        ["Sanchez 2026 — Contrastive Pretraining (Paper 6)"],
    )

    # ── Confidence Calibration (expand hallucination domain) ──────
    # Based on Paper 4 (Confidence Cartography)
    _add(
        "confidence_false_belief", "hallucination", "ESTABLISHED",
        "Model confidence (teacher-forced probability) correlates with human false-belief prevalence",
        "Teacher-forced token probability reveals where models are uncertain. Correlation "
        "between model confidence and human false-belief prevalence: ρ=0.652, p=0.016. "
        "Prior literature only showed r=0.26. Low model confidence = surprising truth = "
        "higher chance of hallucination. Predictable and measurable.",
        [
            "model confidence is random noise",
            "uncertainty and human beliefs are unrelated",
            "hallucination is unpredictable from confidence scores",
        ],
        ["confidence.*false belief", "teacher-forced", "calibrat.*human",
         "false.belief", "confidence.*correlat"],
        ["Sanchez 2026 — Confidence Cartography (Paper 4, DOI: 10.5281/zenodo.18703505)"],
    )

    # ── Adapter scaling (expand training domain) ──────────────────
    # Based on Papers 8-9 (Snap-On, STEM Truth Oracle)
    _add(
        "adapter_stacking_failure", "training", "ESTABLISHED",
        "Stacking 37+ LoRA adapters destroys base model knowledge (MMLU drops 43%)",
        "Adapter stacking does not scale. 37+ adapters on Qwen2.5-1.5B: MMLU drops from "
        "54% to ~11%. A unified adapter on 244 facts collapses entirely. Orthogonal "
        "adapters with routing work (100% of 411 facts across 30 domains), but they "
        "must be routed at inference, never stacked or averaged.",
        [
            "you can stack unlimited adapters",
            "adapter merging preserves knowledge",
            "more adapters always help",
        ],
        ["adapter.*stack", "lora.*stack", "adapter.*merg", "multi.*adapter",
         "adapter.*scal", "unlimited.*adapter", "unlimited.*lora", "stack.*unlimited",
         "stack.*dozen", "dozen.*adapter", "stack.*adapter.*knowledge",
         "many.*adapter", "multiple.*adapter.*loss"],
        ["Sanchez 2026 — Snap-On Modules (Paper 8, DOI: 10.5281/zenodo.18902616)"],
    )
    _add(
        "logit_space_adapters", "training", "ESTABLISHED",
        "Logit-space adapters transfer across model scales and architectures with zero knowledge loss",
        "A 29M-parameter logit-space adapter trained on Qwen2.5-1.5B transfers to 3B-Instruct "
        "with 0.0% MMLU degradation and full safety preservation. Cross-architecture to "
        "Llama-3.1-8B: -0.2% delta. Hidden-space adapters degrade MMLU by 5-8.5%. "
        "The key: operate in logit space (after the model), not hidden space (inside it).",
        [
            "adapters are model-specific",
            "hidden-space adapters preserve knowledge better",
            "cross-architecture transfer is impossible",
        ],
        ["logit.*adapter", "snap-on", "cross.*scale.*transfer", "cross.*arch",
         "logit space", "zero.*mmlu", "knowledge.*preserv"],
        ["Sanchez 2026 — Snap-On Modules (Paper 8, DOI: 10.5281/zenodo.18902616)"],
    )
    _add(
        "stem_scaling_baselines", "evaluation", "ESTABLISHED",
        "STEM fact recall via log-prob ranking has established scaling baselines",
        "Frozen base models scored on 97 STEM facts via sum log-prob MC comparison: "
        "GPT-2-124M: 15.5% (near random). SmolLM2-360M: 61.9%. Qwen2.5-0.5B: 54.6%. "
        "Qwen2.5-1.5B: 63.9%. Qwen3-4B: 76.3%. Physics easiest (85%), statistics "
        "hardest (60%). 4 systematic bias patterns: positivity, linearity, "
        "missing-constant, truncation — all scale-invariant.",
        [
            "small models can't do STEM",
            "STEM requires generation ability",
            "all domains scale equally",
        ],
        ["stem", "science.*baseline", "physics.*score", "stem.*scaling",
         "log-prob.*ranking", "factual.*baseline"],
        ["Sanchez 2026 — STEM Truth Oracle (Paper 9, DOI: 10.5281/zenodo.19005729)"],
    )
    _add(
        "tools_vs_adapters", "training", "ESTABLISHED",
        "Verified computational tools scale better than weight-based adapters for factual correction",
        "Adapters improve truth preference (+0.10 MC2 on TruthfulQA) but can't scale: "
        "stacking 37+ destroys MMLU (-43%), unified adapter on 244 facts collapses. "
        "MCP tools scale indefinitely — each is independent, verified, and model-agnostic. "
        "Tools are the better architecture for factual correction at scale.",
        [
            "fine-tuning is always better than tools",
            "adapters scale to arbitrary knowledge",
            "tools are just a workaround",
        ],
        ["tool.*adapter", "mcp.*scale", "tool.*fine-tun", "adapter.*scale",
         "verified tool", "tool.*weight", "adapter.*better.*tool", "tool.*better.*adapter",
         "fine-tun.*better", "tools vs"],
        ["Sanchez 2026 — NoetherSolve Toolkit (Paper 11, DOI: 10.5281/zenodo.19029880)"],
    )

    return db


_DATABASE = _build_database()

# Precompute keyword → topic_id mapping for fast lookup
_KEYWORD_INDEX: Dict[str, List[str]] = {}
for _tid, _info in _DATABASE.items():
    for _kw in _info.keywords:
        _KEYWORD_INDEX.setdefault(_kw.lower(), []).append(_tid)


# ── Normalization ────────────────────────────────────────────────────────

def _normalize_benchmark(name: str) -> str:
    s = name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    mapping = {
        "mmlu": "mmlu",
        "truthfulqa": "truthfulqa",
        "humaneval": "humaneval",
        "hellaswag": "hellaswag",
        "truthful": "truthfulqa",
        "human_eval": "humaneval",
        "hella": "hellaswag",
    }
    return mapping.get(s, s)


def _normalize_model(name: str) -> str:
    s = name.strip().lower().replace(" ", "-")
    # Common aliases
    mapping = {
        "gpt4": "gpt-4",
        "gpt-4o": "gpt-4o",
        "gpt4o": "gpt-4o",
        "claude-3.5-sonnet": "claude-3.5-sonnet",
        "claude-3-opus": "claude-3-opus",
        "sonnet": "claude-3.5-sonnet",
        "opus": "claude-3-opus",
        "llama-3.1-405b": "llama-3.1-405b",
        "llama-3.1-70b": "llama-3.1-70b",
        "llama-3.1-8b": "llama-3.1-8b",
        "llama3.1-405b": "llama-3.1-405b",
        "llama3.1-70b": "llama-3.1-70b",
        "llama3.1-8b": "llama-3.1-8b",
        "qwen2.5-72b": "qwen2.5-72b",
        "qwen2.5-7b": "qwen2.5-7b",
        "gemini-1.5-pro": "gemini-1.5-pro",
    }
    return mapping.get(s, s)


# ── Core Matching ────────────────────────────────────────────────────────

def _match_claim_to_topics(claim: str) -> List[Tuple[str, float]]:
    """Match a claim string to database topics. Returns (topic_id, score) pairs."""
    claim_lower = claim.lower()
    scores: Dict[str, float] = {}

    for kw, topic_ids in _KEYWORD_INDEX.items():
        # Check regex patterns (keywords with special chars)
        try:
            if re.search(kw, claim_lower):
                for tid in topic_ids:
                    scores[tid] = scores.get(tid, 0) + 1.0
        except re.error:
            # Plain string match for non-regex keywords
            if kw in claim_lower:
                for tid in topic_ids:
                    scores[tid] = scores.get(tid, 0) + 1.0

    # Sort by score descending
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "in", "of", "and", "or", "to", "for",
    "it", "that", "this", "with", "by", "on", "at", "as", "be", "was",
    "were", "been", "being", "have", "has", "had", "do", "does", "did",
    "but", "if", "so", "up", "out", "about", "into", "than",
    "its", "can", "may", "from", "they", "them", "their", "all", "each",
    "when", "which", "what", "how", "where", "who", "will", "would",
    "should", "could", "more", "most", "very", "just", "also",
})


def _content_words(text: str) -> set:
    """Extract content words (strip stopwords and punctuation)."""
    words = set(re.sub(r"[^\w\s-]", "", text.lower()).split())
    return words - _STOPWORDS


def _overlap_ratio(words_a: set, words_b: set) -> float:
    """Fraction of the SMALLER set that overlaps the larger."""
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / min(len(words_a), len(words_b))


def _claim_matches_misconception(claim: str, topic: LLMTopicInfo) -> bool:
    """Check if a claim closely matches a known misconception."""
    claim_lower = claim.lower().strip()
    claim_words = _content_words(claim_lower)
    for misc in topic.misconceptions:
        misc_lower = misc.lower()
        # Direct substring containment (exact match)
        if misc_lower in claim_lower or claim_lower in misc_lower:
            return True
        # Content-word overlap — use MAX denominator (strict: claim must
        # mostly consist of misconception words, not just share 2)
        misc_words = _content_words(misc_lower)
        if not misc_words:
            continue
        overlap_count = len(claim_words & misc_words)
        # Require: >50% of misconception's words are in claim AND
        # >30% of claim's words are misconception words
        misc_cov = overlap_count / len(misc_words)
        claim_cov = overlap_count / len(claim_words) if claim_words else 0
        if misc_cov > 0.5 and claim_cov > 0.3:
            return True
    return False


def _claim_matches_truth(claim: str, topic: LLMTopicInfo) -> bool:
    """Check if a claim is consistent with the established truth."""
    claim_lower = claim.lower().strip()
    truth_lower = topic.truth.lower()
    desc_lower = topic.description.lower()
    # Direct containment against truth OR description
    for ref in (truth_lower, desc_lower):
        if ref in claim_lower or claim_lower in ref:
            return True
    # Content-word overlap (min denominator — a short claim inside a long
    # truth should still match)
    claim_words = _content_words(claim_lower)
    truth_words = _content_words(truth_lower)
    desc_words = _content_words(desc_lower)
    if _overlap_ratio(claim_words, truth_words) > 0.35:
        return True
    if _overlap_ratio(claim_words, desc_words) > 0.35:
        return True
    return False


# ── Public API ───────────────────────────────────────────────────────────

def check_llm_claim(claim: str) -> LLMClaimResult:
    """Check a single LLM claim against the reference database.

    Parameters
    ----------
    claim : str
        A claim about LLM capabilities, training, or behavior.

    Returns
    -------
    LLMClaimResult
        Verdict (TRUE/FALSE/MISLEADING/NUANCED/UNKNOWN), evidence, and references.
    """
    matches = _match_claim_to_topics(claim)

    if not matches:
        return LLMClaimResult(
            verdict="UNKNOWN",
            claim=claim,
            evidence="No matching topic found in the reference database.",
            confidence=0.0,
        )

    # Check top N matches — a claim may match multiple topics
    top_n = matches[:5]

    # For each matched topic, compute truth and misconception overlap scores,
    # then pick the best match across all topics.
    best_truth = None       # (score, tid)
    best_misconception = None  # (score, tid)

    claim_words = _content_words(claim.lower())

    for tid, kw_score in top_n:
        topic = _DATABASE[tid]
        # Truth score: max overlap with truth or description
        truth_words = _content_words(topic.truth.lower())
        desc_words = _content_words(topic.description.lower())
        t_score = max(
            _overlap_ratio(claim_words, truth_words),
            _overlap_ratio(claim_words, desc_words),
        )
        if t_score > 0.35:
            if best_truth is None or t_score > best_truth[0]:
                best_truth = (t_score, tid)

        # Misconception score: max overlap with any misconception
        for misc in topic.misconceptions:
            misc_words = _content_words(misc.lower())
            if not misc_words:
                continue
            m_score = _overlap_ratio(claim_words, misc_words)
            # Also check substring containment (score = 1.0)
            if misc.lower() in claim.lower() or claim.lower() in misc.lower():
                m_score = 1.0
            if m_score > 0.5:
                if best_misconception is None or m_score > best_misconception[0]:
                    best_misconception = (m_score, tid)

    # Pick the stronger match (misconception vs truth)
    if best_truth and best_misconception:
        if best_misconception[0] >= best_truth[0]:
            # Misconception is stronger or equal — claim is FALSE
            tid = best_misconception[1]
            topic = _DATABASE[tid]
            kw_score = dict(top_n).get(tid, 1.0)
            return LLMClaimResult(
                verdict="FALSE",
                claim=claim,
                evidence=f"Matches known misconception. "
                f"Established finding: {topic.truth}",
                domain=topic.domain,
                topic_id=tid,
                confidence=min(kw_score / 3.0, 1.0),
                references=topic.references,
            )
        else:
            # Truth is stronger — claim is TRUE
            tid = best_truth[1]
            topic = _DATABASE[tid]
            kw_score = dict(top_n).get(tid, 1.0)
            confidence = min(kw_score / 3.0, 1.0)
            if topic.status == "DEBATED":
                return LLMClaimResult(
                    verdict="NUANCED",
                    claim=claim,
                    evidence=f"Consistent with current understanding, but this topic "
                    f"is debated. {topic.truth}",
                    domain=topic.domain,
                    topic_id=tid,
                    confidence=confidence,
                    references=topic.references,
                )
            return LLMClaimResult(
                verdict="TRUE",
                claim=claim,
                evidence=f"Consistent with established findings. {topic.truth}",
                domain=topic.domain,
                topic_id=tid,
                confidence=confidence,
                references=topic.references,
            )

    if best_misconception:
        tid = best_misconception[1]
        topic = _DATABASE[tid]
        kw_score = dict(top_n).get(tid, 1.0)
        return LLMClaimResult(
            verdict="FALSE",
            claim=claim,
            evidence=f"Matches known misconception. "
            f"Established finding: {topic.truth}",
            domain=topic.domain,
            topic_id=tid,
            confidence=min(kw_score / 3.0, 1.0),
            references=topic.references,
        )

    if best_truth:
        tid = best_truth[1]
        topic = _DATABASE[tid]
        kw_score = dict(top_n).get(tid, 1.0)
        confidence = min(kw_score / 3.0, 1.0)
        if topic.status == "DEBATED":
            return LLMClaimResult(
                verdict="NUANCED",
                claim=claim,
                evidence=f"Consistent with current understanding, but this topic "
                f"is debated. {topic.truth}",
                domain=topic.domain,
                topic_id=tid,
                confidence=confidence,
                references=topic.references,
            )
        return LLMClaimResult(
            verdict="TRUE",
            claim=claim,
            evidence=f"Consistent with established findings. {topic.truth}",
            domain=topic.domain,
            topic_id=tid,
            confidence=confidence,
            references=topic.references,
        )

    # Matched topic but couldn't classify as truth or misconception
    top_id, top_score = matches[0]
    topic = _DATABASE[top_id]
    confidence = min(top_score / 3.0, 1.0)
    return LLMClaimResult(
        verdict="NUANCED",
        claim=claim,
        evidence=f"Related to '{topic.description}'. "
        f"Established: {topic.truth}",
        domain=topic.domain,
        topic_id=top_id,
        confidence=confidence * 0.5,
        references=topic.references,
    )


def audit_llm_claims(claims: List[str]) -> LLMClaimReport:
    """Audit a list of LLM claims against the reference database.

    Parameters
    ----------
    claims : list of str
        Claims to check.

    Returns
    -------
    LLMClaimReport
        Aggregate report with per-claim issues.
    """
    issues: List[LLMClaimIssue] = []
    warnings: List[str] = []
    n_high = n_moderate = n_low = n_info = 0

    for claim in claims:
        result = check_llm_claim(claim)

        if result.verdict == "FALSE":
            issues.append(LLMClaimIssue(
                severity="HIGH",
                claim=claim,
                description=result.evidence,
                domain=result.domain,
                topic_id=result.topic_id,
                references=result.references,
            ))
            n_high += 1
        elif result.verdict == "MISLEADING":
            issues.append(LLMClaimIssue(
                severity="MODERATE",
                claim=claim,
                description=result.evidence,
                domain=result.domain,
                topic_id=result.topic_id,
                references=result.references,
            ))
            n_moderate += 1
        elif result.verdict == "NUANCED":
            issues.append(LLMClaimIssue(
                severity="LOW",
                claim=claim,
                description=result.evidence,
                domain=result.domain,
                topic_id=result.topic_id,
                references=result.references,
            ))
            n_low += 1
        elif result.verdict == "TRUE":
            issues.append(LLMClaimIssue(
                severity="INFO",
                claim=claim,
                description=result.evidence,
                domain=result.domain,
                topic_id=result.topic_id,
                references=result.references,
            ))
            n_info += 1
        elif result.verdict == "UNKNOWN":
            warnings.append(f"Could not match claim: '{claim}'")

    # Determine verdict
    if n_high > 0:
        verdict = "FAIL"
    elif n_moderate > 0:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return LLMClaimReport(
        verdict=verdict,
        n_claims=len(claims),
        n_high=n_high,
        n_moderate=n_moderate,
        n_low=n_low,
        n_info=n_info,
        issues=issues,
        warnings=warnings,
    )


def get_llm_topic(name: str) -> Optional[LLMTopicInfo]:
    """Look up an LLM topic by ID or keyword.

    Parameters
    ----------
    name : str
        Topic ID (e.g., "sycophancy") or keyword.

    Returns
    -------
    LLMTopicInfo or None
    """
    # Direct ID match
    name_lower = name.lower().replace(" ", "_").replace("-", "_")
    if name_lower in _DATABASE:
        return _DATABASE[name_lower]

    # Keyword search
    matches = _match_claim_to_topics(name)
    if matches:
        return _DATABASE[matches[0][0]]

    return None


def list_llm_topics(domain: Optional[str] = None) -> List[str]:
    """List all LLM topics, optionally filtered by domain.

    Parameters
    ----------
    domain : str, optional
        Filter by domain: "hallucination", "reasoning", "alignment",
        "training", "evaluation", "context_memory".

    Returns
    -------
    list of str
        Topic IDs.
    """
    if domain is None:
        return sorted(_DATABASE.keys())

    domain_lower = domain.lower().replace(" ", "_").replace("-", "_")
    return sorted(
        tid for tid, info in _DATABASE.items()
        if info.domain == domain_lower
    )


def list_domains() -> List[str]:
    """List all LLM claim domains.

    Returns
    -------
    list of str
        Unique domain names.
    """
    return sorted(set(info.domain for info in _DATABASE.values()))
