"""
Type definitions for the Therapeutic Design Lab.

Dataclasses for molecular targets, candidates, reports, and pipeline results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class TargetType(Enum):
    """Types of molecular targets."""
    ENZYME = "enzyme"
    RECEPTOR = "receptor"
    ION_CHANNEL = "ion_channel"
    TRANSCRIPTION_FACTOR = "transcription_factor"
    SECRETED = "secreted"
    STRUCTURAL = "structural"
    KINASE = "kinase"
    PROTEASE = "protease"
    UNKNOWN = "unknown"


class Mechanism(Enum):
    """Disease mechanisms."""
    LOSS_OF_FUNCTION = "loss_of_function"
    GAIN_OF_FUNCTION = "gain_of_function"
    OVEREXPRESSION = "overexpression"
    UNDEREXPRESSION = "underexpression"
    MUTATION = "mutation"
    MISFOLDING = "misfolding"
    AGGREGATION = "aggregation"
    UNKNOWN = "unknown"


class Modality(Enum):
    """Therapeutic modalities."""
    SMALL_MOLECULE = "small_molecule"
    ANTIBODY = "antibody"
    MRNA = "mrna"
    CRISPR = "crispr"
    ASO = "antisense_oligonucleotide"
    GENE_THERAPY = "gene_therapy"
    NEOANTIGEN = "neoantigen"
    CELL_THERAPY = "cell_therapy"


class ValidationLevel(Enum):
    """Target validation levels."""
    VALIDATED = "validated"  # Drugs approved targeting this
    CLINICAL = "clinical"    # In clinical trials
    EMERGING = "emerging"    # Preclinical evidence
    THEORETICAL = "theoretical"  # Hypothesis only


@dataclass
class MolecularTarget:
    """A potential therapeutic target."""
    name: str
    target_type: TargetType = TargetType.UNKNOWN
    uniprot_id: Optional[str] = None
    gene_symbol: Optional[str] = None
    role: str = "causative"  # causative, modulator, downstream
    mechanism: Mechanism = Mechanism.UNKNOWN
    tissue_expression: List[str] = field(default_factory=list)
    validation_level: ValidationLevel = ValidationLevel.THEORETICAL
    known_drugs: List[str] = field(default_factory=list)
    confidence: float = 0.5  # Extraction confidence

    def __str__(self) -> str:
        return (
            f"{self.name} ({self.target_type.value})\n"
            f"  Mechanism: {self.mechanism.value}\n"
            f"  Validation: {self.validation_level.value}\n"
            f"  Confidence: {self.confidence:.0%}"
        )


@dataclass
class OracleVerification:
    """Result of oracle verification."""
    claim: str
    verdict: str  # "TRUE", "FALSE", "UNCERTAIN"
    confidence: float
    margin: float
    domain: str

    def __str__(self) -> str:
        return f"{self.verdict} ({self.confidence:.0%}): {self.claim}"


@dataclass
class PathophysiologyExtraction:
    """Parsed pathophysiology information."""
    disease_name: str
    description: str
    molecular_targets: List[MolecularTarget] = field(default_factory=list)
    mechanisms: List[str] = field(default_factory=list)
    affected_tissues: List[str] = field(default_factory=list)
    cell_types: List[str] = field(default_factory=list)
    causal_chain: List[str] = field(default_factory=list)
    oracle_verifications: List[OracleVerification] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Disease: {self.disease_name}",
            f"Targets found: {len(self.molecular_targets)}",
        ]
        for t in self.molecular_targets:
            lines.append(f"  - {t.name} ({t.target_type.value}, {t.mechanism.value})")
        if self.affected_tissues:
            lines.append(f"Tissues: {', '.join(self.affected_tissues)}")
        if self.oracle_verifications:
            n_verified = sum(1 for v in self.oracle_verifications if v.verdict == "TRUE")
            lines.append(f"Oracle: {n_verified}/{len(self.oracle_verifications)} claims verified")
        return "\n".join(lines)


@dataclass
class DruggabilityReport:
    """Assessment of target druggability."""
    target: MolecularTarget
    druggability_score: float  # 0-100
    modality_scores: Dict[str, float] = field(default_factory=dict)
    structural_features: Dict[str, bool] = field(default_factory=dict)
    existing_drugs: List[str] = field(default_factory=list)
    rationale: str = ""
    oracle_verified: bool = False

    def __str__(self) -> str:
        lines = [
            f"Druggability: {self.target.name}",
            f"  Score: {self.druggability_score:.0f}/100",
            "  Modality scores:",
        ]
        for modality, score in sorted(self.modality_scores.items(), key=lambda x: -x[1]):
            lines.append(f"    {modality}: {score:.0f}")
        if self.existing_drugs:
            lines.append(f"  Existing drugs: {', '.join(self.existing_drugs[:3])}")
        return "\n".join(lines)


@dataclass
class ModalityRecommendation:
    """Recommended therapeutic modality."""
    modality: Modality
    score: float  # 0-100
    rationale: str
    advantages: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    precedents: List[str] = field(default_factory=list)
    oracle_confidence: float = 0.5

    def __str__(self) -> str:
        lines = [
            f"{self.modality.value}: {self.score:.0f}/100",
            f"  {self.rationale}",
        ]
        if self.advantages:
            lines.append(f"  Advantages: {', '.join(self.advantages[:2])}")
        if self.challenges:
            lines.append(f"  Challenges: {', '.join(self.challenges[:2])}")
        return "\n".join(lines)


@dataclass
class TherapeuticCandidate:
    """A therapeutic candidate."""
    candidate_id: str
    modality: Modality
    target: MolecularTarget
    description: str

    # Scores
    efficacy_score: float = 0.0
    safety_score: float = 0.0
    developability_score: float = 0.0
    manufacturing_score: float = 0.0
    regulatory_score: float = 0.0
    combined_score: float = 0.0

    # Modality-specific data
    modality_data: Dict[str, Any] = field(default_factory=dict)

    # Oracle verifications
    oracle_verifications: List[OracleVerification] = field(default_factory=list)

    # Recommendations
    development_path: str = ""
    key_experiments: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.candidate_id} ({self.modality.value})\n"
            f"  Target: {self.target.name}\n"
            f"  Combined score: {self.combined_score:.1f}/100\n"
            f"  Efficacy: {self.efficacy_score:.0f}  Safety: {self.safety_score:.0f}  "
            f"Developability: {self.developability_score:.0f}"
        )


@dataclass
class RankedCandidate:
    """A ranked therapeutic candidate with full evaluation."""
    rank: int
    candidate: TherapeuticCandidate
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    flagged_concerns: List[str] = field(default_factory=list)
    recommendation: str = ""

    def __str__(self) -> str:
        lines = [
            f"#{self.rank}: {self.candidate.candidate_id}",
            f"  Score: {self.candidate.combined_score:.1f}/100",
            f"  {self.recommendation}",
        ]
        if self.flagged_concerns:
            lines.append(f"  Concerns: {', '.join(self.flagged_concerns[:2])}")
        return "\n".join(lines)


@dataclass
class DesignReport:
    """Full therapeutic design report."""
    disease: str
    pathophysiology_summary: PathophysiologyExtraction
    targets_analyzed: int
    candidates_generated: int
    ranked_candidates: List[RankedCandidate] = field(default_factory=list)
    oracle_verifications: int = 0
    recommendations: List[str] = field(default_factory=list)
    disclaimers: List[str] = field(default_factory=lambda: [
        "COMPUTATIONAL PREDICTIONS ONLY",
        "All candidates require experimental validation",
        "Not validated for clinical decision-making",
    ])

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"THERAPEUTIC DESIGN REPORT: {self.disease}",
            "=" * 60,
            "",
            str(self.pathophysiology_summary),
            "",
            f"Candidates generated: {self.candidates_generated}",
            f"Oracle verifications: {self.oracle_verifications}",
            "",
            "TOP CANDIDATES:",
        ]
        for rc in self.ranked_candidates[:5]:
            lines.append(str(rc))
            lines.append("")

        if self.recommendations:
            lines.append("RECOMMENDATIONS:")
            for r in self.recommendations[:3]:
                lines.append(f"  - {r}")

        lines.append("")
        lines.append("DISCLAIMERS:")
        for d in self.disclaimers:
            lines.append(f"  * {d}")

        return "\n".join(lines)
