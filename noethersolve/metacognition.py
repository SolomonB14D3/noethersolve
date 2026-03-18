"""Metacognition analysis module - measure and improve metacognitive capabilities.

This module provides computational tools for assessing metacognition based on
cognitive science frameworks (Flavell 1976, Nelson & Narens 1990).

Metacognition = "cognition about cognition" = knowing what you know

Two key processes:
1. MONITORING: Evaluating your own cognitive states (confidence, difficulty, knowing)
2. CONTROL: Using monitoring signals to regulate behavior (strategy selection, effort)

Key metrics from signal detection theory:
- Calibration (ECE): Does expressed confidence match actual accuracy?
- Resolution (AUROC): Can you distinguish correct from incorrect responses?
- Meta-d': Metacognitive sensitivity independent of response bias
- Meta-d'/d': Metacognitive efficiency (how much info reaches meta-level)

Critical LLM deficit: "Unknown recall" = 0% on most models. They cannot
recognize their own knowledge boundaries - confidently wrong instead.

References:
- Flavell (1976): Metacognitive aspects of problem solving
- Nelson & Narens (1990): Metamemory: A theoretical framework
- Maniscalco & Lau (2012): Meta-d' signal detection theory
- Nature Communications (2024): LLM metacognitive deficits in medicine
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import math


class MetacognitiveProcess(Enum):
    """The two core metacognitive processes (Nelson & Narens)."""
    MONITORING = "monitoring"  # Meta-level receives info from object-level
    CONTROL = "control"        # Meta-level sends commands to object-level


class KnowledgeType(Enum):
    """Types of metacognitive knowledge (Flavell/Brown)."""
    DECLARATIVE = "declarative"   # Knowing ABOUT (what I know, task properties)
    PROCEDURAL = "procedural"     # Knowing HOW (strategies, procedures)
    CONDITIONAL = "conditional"   # Knowing WHEN/WHY (when to apply strategies)


class MonitoringJudgment(Enum):
    """Types of metacognitive monitoring judgments."""
    # Prospective (before task)
    EASE_OF_LEARNING = "EOL"      # How easy will this be to learn?
    JUDGMENT_OF_LEARNING = "JOL"  # How well did I learn this?
    FEELING_OF_KNOWING = "FOK"    # Do I know the answer even if I can't recall it?

    # Retrospective (after task)
    CONFIDENCE = "confidence"     # How sure am I this answer is correct?
    SOURCE_MONITORING = "source"  # Where did I learn this?


@dataclass
class ConfidenceSample:
    """A single response with associated confidence."""
    response: str
    confidence: float  # 0.0 to 1.0
    is_correct: bool
    domain: str = ""

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")


@dataclass
class CalibrationResult:
    """Result of calibration analysis."""
    expected_calibration_error: float  # ECE - lower is better
    maximum_calibration_error: float   # MCE - worst bin
    overconfidence: float              # Mean(confidence - accuracy) when confident
    underconfidence: float             # Mean(accuracy - confidence) when uncertain
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]
    n_bins: int
    total_samples: int

    @property
    def is_well_calibrated(self) -> bool:
        """ECE < 0.1 is generally considered well-calibrated."""
        return self.expected_calibration_error < 0.1

    @property
    def calibration_diagnosis(self) -> str:
        """Diagnose the calibration pattern."""
        if self.expected_calibration_error < 0.05:
            return "excellent"
        elif self.expected_calibration_error < 0.1:
            return "good"
        elif self.expected_calibration_error < 0.2:
            return "moderate"
        else:
            return "poor"

    @property
    def bias_direction(self) -> str:
        """Is the system overconfident or underconfident?"""
        if self.overconfidence > self.underconfidence + 0.05:
            return "overconfident"
        elif self.underconfidence > self.overconfidence + 0.05:
            return "underconfident"
        else:
            return "balanced"

    def __str__(self) -> str:
        lines = [
            "CALIBRATION ANALYSIS",
            "=" * 50,
            "",
            f"Expected Calibration Error (ECE): {self.expected_calibration_error:.3f}",
            f"Maximum Calibration Error (MCE): {self.maximum_calibration_error:.3f}",
            f"Calibration Quality: {self.calibration_diagnosis.upper()}",
            f"Bias Direction: {self.bias_direction.upper()}",
            "",
            f"Overconfidence: {self.overconfidence:.3f}",
            f"Underconfidence: {self.underconfidence:.3f}",
            "",
            f"Samples: {self.total_samples}",
            f"Bins: {self.n_bins}",
            "",
            "Reliability Diagram:",
        ]

        for i in range(self.n_bins):
            if self.bin_counts[i] > 0:
                conf = self.bin_confidences[i]
                acc = self.bin_accuracies[i]
                bar_len = int(acc * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"  {conf:.1f}: {bar} {acc:.2f} (n={self.bin_counts[i]})")

        return "\n".join(lines)


@dataclass
class ResolutionResult:
    """Result of resolution/discrimination analysis."""
    auroc: float  # Area under ROC curve - ability to separate correct/incorrect
    sensitivity: float  # True positive rate at optimal threshold
    specificity: float  # True negative rate at optimal threshold
    optimal_threshold: float
    n_correct: int
    n_incorrect: int

    @property
    def resolution_quality(self) -> str:
        """Interpret AUROC."""
        if self.auroc >= 0.9:
            return "excellent"
        elif self.auroc >= 0.8:
            return "good"
        elif self.auroc >= 0.7:
            return "moderate"
        elif self.auroc >= 0.6:
            return "poor"
        else:
            return "no_discrimination"

    @property
    def can_discriminate(self) -> bool:
        """AUROC > 0.5 indicates above-chance discrimination."""
        return self.auroc > 0.55

    def __str__(self) -> str:
        lines = [
            "RESOLUTION ANALYSIS (Confidence Discrimination)",
            "=" * 50,
            "",
            f"AUROC: {self.auroc:.3f}",
            f"Resolution Quality: {self.resolution_quality.upper()}",
            f"Can Discriminate Correct/Incorrect: {'YES' if self.can_discriminate else 'NO'}",
            "",
            f"Optimal Threshold: {self.optimal_threshold:.2f}",
            f"Sensitivity (TPR): {self.sensitivity:.3f}",
            f"Specificity (TNR): {self.specificity:.3f}",
            "",
            f"Correct responses: {self.n_correct}",
            f"Incorrect responses: {self.n_incorrect}",
        ]
        return "\n".join(lines)


@dataclass
class MetaDPrimeResult:
    """Result of meta-d' signal detection analysis."""
    d_prime: float           # Task sensitivity (Type 1)
    meta_d_prime: float      # Metacognitive sensitivity (Type 2)
    m_ratio: float           # meta-d'/d' - metacognitive efficiency

    hit_rate: float          # P(high conf | correct)
    false_alarm_rate: float  # P(high conf | incorrect)

    interpretation: str

    @property
    def efficiency_category(self) -> str:
        """Categorize metacognitive efficiency."""
        if self.m_ratio >= 1.0:
            return "optimal_or_super"  # All info reaches meta-level
        elif self.m_ratio >= 0.8:
            return "high"
        elif self.m_ratio >= 0.5:
            return "moderate"
        else:
            return "low"  # Much info lost before meta-level

    def __str__(self) -> str:
        lines = [
            "META-D' SIGNAL DETECTION ANALYSIS",
            "=" * 50,
            "",
            f"Task Sensitivity (d'): {self.d_prime:.3f}",
            f"Metacognitive Sensitivity (meta-d'): {self.meta_d_prime:.3f}",
            f"Metacognitive Efficiency (M-ratio): {self.m_ratio:.3f}",
            f"Efficiency Category: {self.efficiency_category.upper()}",
            "",
            f"Hit Rate (high conf | correct): {self.hit_rate:.3f}",
            f"False Alarm Rate (high conf | incorrect): {self.false_alarm_rate:.3f}",
            "",
            f"Interpretation: {self.interpretation}",
        ]
        return "\n".join(lines)


@dataclass
class UnknownRecallResult:
    """Result of unknown recall analysis - can the system say 'I don't know'?"""
    unknown_recall_rate: float  # P(says "don't know" | actually doesn't know)
    false_unknown_rate: float   # P(says "don't know" | actually knows)
    unknown_precision: float    # P(actually doesn't know | says "don't know")

    n_truly_unknown: int
    n_truly_known: int
    n_said_unknown: int

    @property
    def can_recognize_ignorance(self) -> bool:
        """Can the system recognize its own knowledge limits?"""
        return self.unknown_recall_rate > 0.3  # 30% threshold

    @property
    def diagnosis(self) -> str:
        """Diagnose unknown recall capability."""
        if self.unknown_recall_rate >= 0.7:
            return "good_epistemic_humility"
        elif self.unknown_recall_rate >= 0.3:
            return "partial_self_awareness"
        elif self.unknown_recall_rate > 0:
            return "minimal_self_awareness"
        else:
            return "no_epistemic_humility"  # Most LLMs

    def __str__(self) -> str:
        lines = [
            "UNKNOWN RECALL ANALYSIS",
            "=" * 50,
            "",
            f"Unknown Recall Rate: {self.unknown_recall_rate:.1%}",
            f"False Unknown Rate: {self.false_unknown_rate:.1%}",
            f"Unknown Precision: {self.unknown_precision:.1%}",
            "",
            f"Diagnosis: {self.diagnosis.upper()}",
            f"Can Recognize Ignorance: {'YES' if self.can_recognize_ignorance else 'NO'}",
            "",
            f"Truly unknown items: {self.n_truly_unknown}",
            f"Truly known items: {self.n_truly_known}",
            f"Said 'don't know': {self.n_said_unknown}",
            "",
            "NOTE: Most LLMs score 0% - they cannot recognize knowledge limits.",
        ]
        return "\n".join(lines)


@dataclass
class SelfCorrectionResult:
    """Result of self-correction analysis."""
    correction_attempted_rate: float  # How often does it try to correct?
    successful_correction_rate: float # P(correct after | incorrect before, attempted)
    degradation_rate: float           # P(incorrect after | correct before, attempted)
    net_improvement: float            # Overall accuracy change from correction

    n_initially_correct: int
    n_initially_incorrect: int
    n_corrections_attempted: int

    @property
    def self_correction_helps(self) -> bool:
        """Does self-correction improve accuracy?"""
        return self.net_improvement > 0.05

    @property
    def self_correction_safe(self) -> bool:
        """Is self-correction safe (doesn't degrade correct answers)?"""
        return self.degradation_rate < 0.1

    def __str__(self) -> str:
        lines = [
            "SELF-CORRECTION ANALYSIS",
            "=" * 50,
            "",
            f"Correction Attempt Rate: {self.correction_attempted_rate:.1%}",
            f"Successful Correction Rate: {self.successful_correction_rate:.1%}",
            f"Degradation Rate: {self.degradation_rate:.1%}",
            f"Net Accuracy Improvement: {self.net_improvement:+.1%}",
            "",
            f"Self-Correction Helps: {'YES' if self.self_correction_helps else 'NO'}",
            f"Self-Correction Safe: {'YES' if self.self_correction_safe else 'NO'}",
            "",
            f"Initially correct: {self.n_initially_correct}",
            f"Initially incorrect: {self.n_initially_incorrect}",
            f"Corrections attempted: {self.n_corrections_attempted}",
            "",
            "NOTE: Research shows LLMs often cannot self-correct without external feedback.",
        ]
        return "\n".join(lines)


@dataclass
class MetacognitiveStateVector:
    """Current metacognitive state (for real-time monitoring)."""
    confidence: float        # P(response is correct)
    experience_match: float  # Similarity to training distribution
    conflict_level: float    # Internal contradictions detected
    difficulty: float        # Perceived task difficulty
    importance: float        # Stakes/urgency

    def should_escalate(self) -> bool:
        """Should trigger slow deliberative reasoning?"""
        return (self.confidence < 0.3 or
                self.conflict_level > 0.5 or
                (self.importance > 0.7 and self.confidence < 0.7))

    def should_seek_help(self) -> bool:
        """Should the system ask for clarification or human review?"""
        return (self.confidence < 0.2 or
                (self.importance > 0.8 and self.confidence < 0.5))

    def __str__(self) -> str:
        return (f"MetaState(conf={self.confidence:.2f}, "
                f"exp_match={self.experience_match:.2f}, "
                f"conflict={self.conflict_level:.2f}, "
                f"diff={self.difficulty:.2f}, "
                f"imp={self.importance:.2f})")


@dataclass
class MetacognitionReport:
    """Comprehensive metacognition assessment."""
    system_name: str
    calibration: Optional[CalibrationResult] = None
    resolution: Optional[ResolutionResult] = None
    meta_d_prime: Optional[MetaDPrimeResult] = None
    unknown_recall: Optional[UnknownRecallResult] = None
    self_correction: Optional[SelfCorrectionResult] = None

    monitoring_score: float = 0.0  # 0-1, how good is monitoring
    control_score: float = 0.0     # 0-1, how good is control
    overall_score: float = 0.0     # 0-1, overall metacognitive capability

    key_deficits: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"METACOGNITION ASSESSMENT: {self.system_name}",
            "=" * 60,
            "",
            f"Overall Metacognitive Score: {self.overall_score:.1%}",
            f"Monitoring Score: {self.monitoring_score:.1%}",
            f"Control Score: {self.control_score:.1%}",
            "",
        ]

        if self.calibration:
            lines.append(f"Calibration (ECE): {self.calibration.expected_calibration_error:.3f} "
                        f"({self.calibration.calibration_diagnosis})")
        if self.resolution:
            lines.append(f"Resolution (AUROC): {self.resolution.auroc:.3f} "
                        f"({self.resolution.resolution_quality})")
        if self.meta_d_prime:
            lines.append(f"Meta-d'/d' (efficiency): {self.meta_d_prime.m_ratio:.3f} "
                        f"({self.meta_d_prime.efficiency_category})")
        if self.unknown_recall:
            lines.append(f"Unknown Recall: {self.unknown_recall.unknown_recall_rate:.1%} "
                        f"({self.unknown_recall.diagnosis})")

        if self.key_deficits:
            lines.append("")
            lines.append("Key Deficits:")
            for deficit in self.key_deficits:
                lines.append(f"  ✗ {deficit}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  → {rec}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Core Analysis Functions ──────────────────────────────────────────

def compute_calibration(
    samples: List[ConfidenceSample],
    n_bins: int = 10
) -> CalibrationResult:
    """
    Compute calibration metrics (ECE, MCE, reliability diagram).

    Perfect calibration: P(correct | confidence=c) = c for all c.

    Args:
        samples: List of ConfidenceSample with confidence and correctness
        n_bins: Number of bins for reliability diagram

    Returns:
        CalibrationResult with ECE, MCE, and per-bin data
    """
    if not samples:
        raise ValueError("No samples provided")

    # Initialize bins
    bin_sums = [0.0] * n_bins  # Sum of confidences
    bin_correct = [0] * n_bins  # Count of correct
    bin_counts = [0] * n_bins   # Total count

    # Assign samples to bins
    for sample in samples:
        bin_idx = min(int(sample.confidence * n_bins), n_bins - 1)
        bin_sums[bin_idx] += sample.confidence
        bin_correct[bin_idx] += int(sample.is_correct)
        bin_counts[bin_idx] += 1

    # Compute per-bin metrics
    bin_confidences = []
    bin_accuracies = []

    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_confidences.append(bin_sums[i] / bin_counts[i])
            bin_accuracies.append(bin_correct[i] / bin_counts[i])
        else:
            bin_confidences.append((i + 0.5) / n_bins)  # Bin center
            bin_accuracies.append(0.0)

    # Compute ECE and MCE
    total = len(samples)
    ece = sum(bin_counts[i] * abs(bin_accuracies[i] - bin_confidences[i])
              for i in range(n_bins)) / total
    mce = max(abs(bin_accuracies[i] - bin_confidences[i])
              for i in range(n_bins) if bin_counts[i] > 0)

    # Compute over/underconfidence
    overconfidence = 0.0
    underconfidence = 0.0
    over_count = 0
    under_count = 0

    for sample in samples:
        gap = sample.confidence - (1.0 if sample.is_correct else 0.0)
        if sample.confidence > 0.5:  # Confident
            if gap > 0:
                overconfidence += gap
                over_count += 1
        else:  # Uncertain
            if gap < 0:
                underconfidence += abs(gap)
                under_count += 1

    overconfidence = overconfidence / over_count if over_count > 0 else 0.0
    underconfidence = underconfidence / under_count if under_count > 0 else 0.0

    return CalibrationResult(
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        overconfidence=overconfidence,
        underconfidence=underconfidence,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
        n_bins=n_bins,
        total_samples=total
    )


def compute_resolution(samples: List[ConfidenceSample]) -> ResolutionResult:
    """
    Compute resolution (AUROC for confidence discrimination).

    Resolution = ability to give high confidence when correct,
    low confidence when incorrect.

    Args:
        samples: List of ConfidenceSample with confidence and correctness

    Returns:
        ResolutionResult with AUROC and optimal threshold
    """
    if not samples:
        raise ValueError("No samples provided")

    correct = [s.confidence for s in samples if s.is_correct]
    incorrect = [s.confidence for s in samples if not s.is_correct]

    if not correct or not incorrect:
        return ResolutionResult(
            auroc=0.5,
            sensitivity=0.5,
            specificity=0.5,
            optimal_threshold=0.5,
            n_correct=len(correct),
            n_incorrect=len(incorrect)
        )

    # Compute AUROC via Mann-Whitney U statistic
    # P(conf_correct > conf_incorrect)
    n_comparisons = 0
    n_correct_higher = 0

    for c in correct:
        for i in incorrect:
            n_comparisons += 1
            if c > i:
                n_correct_higher += 1
            elif c == i:
                n_correct_higher += 0.5

    auroc = n_correct_higher / n_comparisons if n_comparisons > 0 else 0.5

    # Find optimal threshold (maximize Youden's J)
    all_conf = sorted(set(s.confidence for s in samples))
    best_j = -1
    best_threshold = 0.5
    best_sens = 0.5
    best_spec = 0.5

    for threshold in all_conf:
        tp = sum(1 for c in correct if c >= threshold)
        tn = sum(1 for i in incorrect if i < threshold)
        sens = tp / len(correct) if correct else 0
        spec = tn / len(incorrect) if incorrect else 0
        j = sens + spec - 1  # Youden's J

        if j > best_j:
            best_j = j
            best_threshold = threshold
            best_sens = sens
            best_spec = spec

    return ResolutionResult(
        auroc=auroc,
        sensitivity=best_sens,
        specificity=best_spec,
        optimal_threshold=best_threshold,
        n_correct=len(correct),
        n_incorrect=len(incorrect)
    )


def compute_meta_d_prime(
    samples: List[ConfidenceSample],
    confidence_threshold: float = 0.5
) -> MetaDPrimeResult:
    """
    Compute meta-d' (metacognitive sensitivity in SDT units).

    Meta-d' measures how much information about correctness reaches
    the metacognitive level. If meta-d' < d', information is lost.

    Simplified computation (full MLE would require scipy optimization).

    Args:
        samples: List of ConfidenceSample
        confidence_threshold: Threshold for high vs low confidence

    Returns:
        MetaDPrimeResult with d', meta-d', and M-ratio
    """
    if not samples:
        raise ValueError("No samples provided")

    # Count outcomes
    # High confidence responses
    high_conf_correct = sum(1 for s in samples
                            if s.confidence >= confidence_threshold and s.is_correct)
    high_conf_incorrect = sum(1 for s in samples
                              if s.confidence >= confidence_threshold and not s.is_correct)
    low_conf_correct = sum(1 for s in samples
                           if s.confidence < confidence_threshold and s.is_correct)
    low_conf_incorrect = sum(1 for s in samples
                             if s.confidence < confidence_threshold and not s.is_correct)

    total_correct = high_conf_correct + low_conf_correct
    total_incorrect = high_conf_incorrect + low_conf_incorrect

    # Prevent division by zero
    if total_correct == 0 or total_incorrect == 0:
        return MetaDPrimeResult(
            d_prime=0.0,
            meta_d_prime=0.0,
            m_ratio=0.0,
            hit_rate=0.0,
            false_alarm_rate=0.0,
            interpretation="Insufficient data for analysis"
        )

    # Hit rate: P(high conf | correct)
    hit_rate = high_conf_correct / total_correct

    # False alarm rate: P(high conf | incorrect)
    fa_rate = high_conf_incorrect / total_incorrect

    # Apply correction for extreme values (Hautus adjustment)
    def adjust_rate(rate, n):
        return (rate * n + 0.5) / (n + 1) if rate == 0 or rate == 1 else rate

    hit_rate_adj = adjust_rate(hit_rate, total_correct)
    fa_rate_adj = adjust_rate(fa_rate, total_incorrect)

    # Compute d' (z-transform)
    def phi_inv(p):
        """Approximate inverse normal CDF."""
        # Approximation using Abramowitz and Stegun formula
        if p <= 0:
            return -3.0
        if p >= 1:
            return 3.0

        t = math.sqrt(-2 * math.log(min(p, 1-p)))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z = t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)

        return z if p > 0.5 else -z

    # Task d' (simplified - using overall accuracy)
    overall_accuracy = total_correct / len(samples)
    d_prime = phi_inv(overall_accuracy) * math.sqrt(2) if 0 < overall_accuracy < 1 else 0

    # Meta-d' (Type 2 SDT)
    z_hit = phi_inv(hit_rate_adj)
    z_fa = phi_inv(fa_rate_adj)
    meta_d_prime = z_hit - z_fa

    # M-ratio
    m_ratio = meta_d_prime / d_prime if d_prime > 0 else 0.0

    # Interpretation
    if m_ratio >= 1.0:
        interp = "Optimal or super-optimal: all task info reaches meta-level"
    elif m_ratio >= 0.8:
        interp = "High efficiency: most task info reaches meta-level"
    elif m_ratio >= 0.5:
        interp = "Moderate efficiency: substantial info loss to meta-level"
    else:
        interp = "Low efficiency: most task info lost before reaching meta-level"

    return MetaDPrimeResult(
        d_prime=d_prime,
        meta_d_prime=meta_d_prime,
        m_ratio=m_ratio,
        hit_rate=hit_rate,
        false_alarm_rate=fa_rate,
        interpretation=interp
    )


def analyze_unknown_recall(
    responses: List[Dict],
    unknown_markers: List[str] = None
) -> UnknownRecallResult:
    """
    Analyze unknown recall - can the system say "I don't know"?

    This is a critical LLM deficit: most models score 0%.

    Args:
        responses: List of dicts with keys:
            - 'response': str
            - 'actually_knows': bool (ground truth)
            - 'said_unknown': bool (optional, auto-detected if not provided)
        unknown_markers: Phrases indicating uncertainty

    Returns:
        UnknownRecallResult with recall/precision metrics
    """
    if unknown_markers is None:
        unknown_markers = [
            "i don't know",
            "i'm not sure",
            "i cannot answer",
            "i don't have enough information",
            "uncertain",
            "i'm unable to",
            "i cannot determine"
        ]

    truly_known = 0
    truly_unknown = 0
    said_unknown_when_unknown = 0
    said_unknown_when_known = 0
    total_said_unknown = 0

    for r in responses:
        actually_knows = r.get('actually_knows', True)

        # Auto-detect if said unknown
        if 'said_unknown' in r:
            said_unknown = r['said_unknown']
        else:
            response_lower = r['response'].lower()
            said_unknown = any(marker in response_lower for marker in unknown_markers)

        if actually_knows:
            truly_known += 1
            if said_unknown:
                said_unknown_when_known += 1
                total_said_unknown += 1
        else:
            truly_unknown += 1
            if said_unknown:
                said_unknown_when_unknown += 1
                total_said_unknown += 1

    # Compute rates
    unknown_recall = (said_unknown_when_unknown / truly_unknown
                      if truly_unknown > 0 else 0.0)
    false_unknown = (said_unknown_when_known / truly_known
                     if truly_known > 0 else 0.0)
    unknown_precision = (said_unknown_when_unknown / total_said_unknown
                         if total_said_unknown > 0 else 0.0)

    return UnknownRecallResult(
        unknown_recall_rate=unknown_recall,
        false_unknown_rate=false_unknown,
        unknown_precision=unknown_precision,
        n_truly_unknown=truly_unknown,
        n_truly_known=truly_known,
        n_said_unknown=total_said_unknown
    )


def analyze_self_correction(
    correction_attempts: List[Dict]
) -> SelfCorrectionResult:
    """
    Analyze self-correction capability.

    Research shows LLMs often cannot self-correct reasoning without
    external feedback, and sometimes make things worse.

    Args:
        correction_attempts: List of dicts with keys:
            - 'initial_correct': bool
            - 'attempted_correction': bool
            - 'final_correct': bool

    Returns:
        SelfCorrectionResult with success/degradation rates
    """
    initially_correct = sum(1 for a in correction_attempts if a['initial_correct'])
    initially_incorrect = sum(1 for a in correction_attempts
                               if not a['initial_correct'])

    # Among those who attempted correction
    attempted = [a for a in correction_attempts if a['attempted_correction']]
    n_attempted = len(attempted)

    if n_attempted == 0:
        return SelfCorrectionResult(
            correction_attempted_rate=0.0,
            successful_correction_rate=0.0,
            degradation_rate=0.0,
            net_improvement=0.0,
            n_initially_correct=initially_correct,
            n_initially_incorrect=initially_incorrect,
            n_corrections_attempted=0
        )

    # Successful corrections: incorrect -> correct
    successful = sum(1 for a in attempted
                     if not a['initial_correct'] and a['final_correct'])
    incorrect_attempted = sum(1 for a in attempted if not a['initial_correct'])

    # Degradations: correct -> incorrect
    degraded = sum(1 for a in attempted
                   if a['initial_correct'] and not a['final_correct'])
    correct_attempted = sum(1 for a in attempted if a['initial_correct'])

    # Rates
    attempt_rate = n_attempted / len(correction_attempts)
    success_rate = successful / incorrect_attempted if incorrect_attempted > 0 else 0.0
    degrade_rate = degraded / correct_attempted if correct_attempted > 0 else 0.0

    # Net improvement
    initial_accuracy = initially_correct / len(correction_attempts)
    final_correct = sum(1 for a in correction_attempts if a['final_correct'])
    final_accuracy = final_correct / len(correction_attempts)
    net_improvement = final_accuracy - initial_accuracy

    return SelfCorrectionResult(
        correction_attempted_rate=attempt_rate,
        successful_correction_rate=success_rate,
        degradation_rate=degrade_rate,
        net_improvement=net_improvement,
        n_initially_correct=initially_correct,
        n_initially_incorrect=initially_incorrect,
        n_corrections_attempted=n_attempted
    )


def assess_metacognition(
    system_name: str,
    confidence_samples: List[ConfidenceSample] = None,
    unknown_responses: List[Dict] = None,
    correction_attempts: List[Dict] = None
) -> MetacognitionReport:
    """
    Comprehensive metacognition assessment.

    Args:
        system_name: Name of the system being assessed
        confidence_samples: For calibration/resolution analysis
        unknown_responses: For unknown recall analysis
        correction_attempts: For self-correction analysis

    Returns:
        MetacognitionReport with all metrics and recommendations
    """
    report = MetacognitionReport(system_name=system_name)

    monitoring_scores = []
    control_scores = []

    # Calibration and resolution
    if confidence_samples and len(confidence_samples) >= 10:
        report.calibration = compute_calibration(confidence_samples)
        report.resolution = compute_resolution(confidence_samples)
        report.meta_d_prime = compute_meta_d_prime(confidence_samples)

        # Score based on ECE (lower is better)
        cal_score = max(0, 1 - report.calibration.expected_calibration_error * 5)
        monitoring_scores.append(cal_score)

        # Score based on AUROC
        res_score = (report.resolution.auroc - 0.5) * 2  # 0.5 -> 0, 1.0 -> 1
        monitoring_scores.append(res_score)

        # Deficits
        if report.calibration.expected_calibration_error > 0.15:
            report.key_deficits.append(f"Poor calibration (ECE={report.calibration.expected_calibration_error:.2f})")
            report.recommendations.append("Implement confidence calibration training")

        if report.calibration.bias_direction == "overconfident":
            report.key_deficits.append("Overconfident when wrong")
            report.recommendations.append("Add uncertainty penalty to training")

        if report.resolution.auroc < 0.7:
            report.key_deficits.append(f"Poor discrimination (AUROC={report.resolution.auroc:.2f})")
            report.recommendations.append("Train to distinguish correct/incorrect responses")

    # Unknown recall
    if unknown_responses:
        report.unknown_recall = analyze_unknown_recall(unknown_responses)

        ur_score = report.unknown_recall.unknown_recall_rate
        monitoring_scores.append(ur_score)

        if report.unknown_recall.unknown_recall_rate < 0.3:
            report.key_deficits.append(f"Cannot recognize ignorance (recall={report.unknown_recall.unknown_recall_rate:.1%})")
            report.recommendations.append("Train to output 'I don't know' for out-of-distribution queries")

    # Self-correction
    if correction_attempts:
        report.self_correction = analyze_self_correction(correction_attempts)

        if report.self_correction.self_correction_helps:
            sc_score = report.self_correction.successful_correction_rate
        else:
            sc_score = 0.0
        control_scores.append(sc_score)

        if not report.self_correction.self_correction_safe:
            report.key_deficits.append(f"Self-correction degrades answers ({report.self_correction.degradation_rate:.1%})")
            report.recommendations.append("Avoid prompting for self-correction without external verification")

        if report.self_correction.successful_correction_rate < 0.3:
            report.key_deficits.append("Cannot effectively self-correct")
            report.recommendations.append("Use external verification instead of self-critique")

    # Compute overall scores
    if monitoring_scores:
        report.monitoring_score = sum(monitoring_scores) / len(monitoring_scores)
    if control_scores:
        report.control_score = sum(control_scores) / len(control_scores)

    all_scores = monitoring_scores + control_scores
    if all_scores:
        report.overall_score = sum(all_scores) / len(all_scores)

    return report


# ── LLM-Specific Defaults ────────────────────────────────────────────

# Typical LLM metacognitive profile based on research
LLM_TYPICAL_PROFILE = {
    "unknown_recall": 0.0,           # 0% - critical deficit
    "calibration_ece": 0.15,         # Moderate miscalibration
    "resolution_auroc": 0.7,         # Token probs provide some signal
    "self_correction_success": 0.3,  # Limited without external feedback
    "self_correction_degrade": 0.2,  # Often makes things worse
}


def get_llm_metacognition_baseline() -> MetacognitionReport:
    """
    Get typical LLM metacognition baseline based on research.

    Key findings from Nature Communications (2024) and other studies:
    - Unknown recall: 0% (cannot say "I don't know")
    - Calibration: Moderate ECE ~0.15 (worse when prompted for confidence)
    - Resolution: AUROC ~0.7 (token probabilities help)
    - Self-correction: Often degrades without external feedback
    """
    report = MetacognitionReport(system_name="Typical LLM (Research Baseline)")

    report.monitoring_score = 0.35
    report.control_score = 0.20
    report.overall_score = 0.30

    report.key_deficits = [
        "Cannot recognize knowledge boundaries (0% unknown recall)",
        "Confidently wrong - provides answers even without correct options",
        "Cannot self-correct reasoning without external feedback",
        "Verbalized confidence poorly calibrated",
        "No persistent self-model across sessions"
    ]

    report.recommendations = [
        "Use token probabilities, not verbalized confidence",
        "Implement external verification before accepting corrections",
        "Train with explicit 'I don't know' responses",
        "Add calibration fine-tuning with confidence labels",
        "Use retrieval to ground responses in verified knowledge"
    ]

    return report


def list_metacognitive_capabilities() -> Dict:
    """List all metacognitive capabilities with LLM status."""
    return {
        "monitoring": {
            "confidence_estimation": {
                "llm_status": "partial",
                "evidence": "Token probabilities correlate with accuracy, but verbalized confidence is poorly calibrated",
                "implementation": "Use logprobs, not prompted confidence"
            },
            "error_detection": {
                "llm_status": "partial",
                "evidence": "Can identify errors in text with prompting, but cannot reliably self-correct",
                "implementation": "External verification, not self-critique alone"
            },
            "difficulty_assessment": {
                "llm_status": "partial",
                "evidence": "Can estimate relative difficulty but calibration varies by domain",
                "implementation": "Ensemble disagreement as proxy for difficulty"
            },
            "unknown_recall": {
                "llm_status": "absent",
                "evidence": "0% on most models - cannot recognize knowledge boundaries",
                "implementation": "Train with explicit 'I don't know' responses + OOD detection"
            }
        },
        "control": {
            "strategy_selection": {
                "llm_status": "partial",
                "evidence": "Can switch strategies with prompting, but not autonomously",
                "implementation": "Meta-prompt that selects strategy based on task features"
            },
            "effort_allocation": {
                "llm_status": "absent",
                "evidence": "No mechanism to allocate more compute to harder problems",
                "implementation": "Adaptive compute / test-time training"
            },
            "self_correction": {
                "llm_status": "limited",
                "evidence": "Cannot self-correct without external feedback; often degrades",
                "implementation": "Verification-guided revision, not raw self-critique"
            },
            "help_seeking": {
                "llm_status": "absent",
                "evidence": "No autonomous mechanism to seek clarification",
                "implementation": "Confidence-triggered human-in-the-loop"
            }
        }
    }
