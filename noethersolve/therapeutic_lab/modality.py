"""
Modality Selector — Recommend therapeutic modalities for targets.

Decision rules based on target type, mechanism, and disease context.
"""

from typing import List, Tuple

from .types import (
    MolecularTarget,
    DruggabilityReport,
    ModalityRecommendation,
    Modality,
    TargetType,
    Mechanism,
)


class ModalitySelector:
    """Select optimal therapeutic modality based on target properties."""

    # Decision rules: (target_type, mechanism) -> (modalities, rationale)
    DECISION_RULES: List[Tuple] = [
        # Kinases - almost always small molecules
        (TargetType.KINASE, None, [Modality.SMALL_MOLECULE],
         "Kinases have well-defined ATP-binding pockets suitable for small molecule inhibitors"),

        # Enzymes - prefer small molecules
        (TargetType.ENZYME, Mechanism.GAIN_OF_FUNCTION, [Modality.SMALL_MOLECULE, Modality.ANTIBODY],
         "Hyperactive enzymes can be inhibited with small molecules or blocking antibodies"),
        (TargetType.ENZYME, Mechanism.LOSS_OF_FUNCTION, [Modality.MRNA, Modality.GENE_THERAPY],
         "Enzyme deficiency requires replacement via mRNA or gene therapy"),

        # Receptors - antibodies are strong candidates
        (TargetType.RECEPTOR, Mechanism.OVEREXPRESSION, [Modality.ANTIBODY, Modality.SMALL_MOLECULE],
         "Overexpressed receptors can be blocked with antibodies or small molecule antagonists"),
        (TargetType.RECEPTOR, None, [Modality.ANTIBODY, Modality.SMALL_MOLECULE],
         "Cell surface receptors are accessible to both antibodies and small molecules"),

        # Ion channels - small molecules
        (TargetType.ION_CHANNEL, Mechanism.LOSS_OF_FUNCTION, [Modality.SMALL_MOLECULE, Modality.MRNA],
         "Ion channel defects may be corrected with potentiators/correctors or mRNA replacement"),
        (TargetType.ION_CHANNEL, None, [Modality.SMALL_MOLECULE],
         "Ion channels are modulated by small molecule blockers or openers"),

        # Secreted proteins - replacement or neutralization
        (TargetType.SECRETED, Mechanism.LOSS_OF_FUNCTION, [Modality.MRNA, Modality.GENE_THERAPY],
         "Secreted protein deficiency is addressed by mRNA-based or gene therapy replacement"),
        (TargetType.SECRETED, Mechanism.OVEREXPRESSION, [Modality.ANTIBODY],
         "Excess secreted protein can be neutralized with antibodies"),
        (TargetType.SECRETED, None, [Modality.ANTIBODY, Modality.MRNA],
         "Secreted proteins are accessible to antibodies or can be replaced with mRNA"),

        # Transcription factors - knockdown required
        (TargetType.TRANSCRIPTION_FACTOR, Mechanism.OVEREXPRESSION, [Modality.ASO, Modality.CRISPR],
         "Overexpressed transcription factors require knockdown via ASO or CRISPR"),
        (TargetType.TRANSCRIPTION_FACTOR, None, [Modality.ASO, Modality.CRISPR],
         "Transcription factors lack binding pockets; knockdown strategies are preferred"),

        # Structural proteins - gene-based approaches
        (TargetType.STRUCTURAL, Mechanism.LOSS_OF_FUNCTION, [Modality.GENE_THERAPY, Modality.MRNA],
         "Structural protein deficiency requires gene replacement or mRNA expression"),
        (TargetType.STRUCTURAL, Mechanism.MUTATION, [Modality.CRISPR, Modality.ASO],
         "Mutant structural proteins may be corrected with CRISPR or exon skipping"),
        (TargetType.STRUCTURAL, None, [Modality.GENE_THERAPY, Modality.ASO],
         "Structural proteins often require gene-level interventions"),

        # Proteases - small molecules
        (TargetType.PROTEASE, None, [Modality.SMALL_MOLECULE],
         "Proteases have well-defined active sites for small molecule inhibitors"),
    ]

    # Modality advantages and challenges
    MODALITY_INFO = {
        Modality.SMALL_MOLECULE: {
            "advantages": [
                "Oral administration possible",
                "Good tissue penetration",
                "Well-established manufacturing",
                "Rapid onset of action",
            ],
            "challenges": [
                "Requires binding pocket",
                "Selectivity can be challenging",
                "Metabolism and drug interactions",
            ],
            "precedents": ["imatinib", "erlotinib", "venetoclax"],
        },
        Modality.ANTIBODY: {
            "advantages": [
                "High specificity",
                "Long half-life",
                "Well-characterized platform",
                "Fc effector functions available",
            ],
            "challenges": [
                "Limited to extracellular targets",
                "Parenteral administration",
                "Immunogenicity risk",
                "High manufacturing cost",
            ],
            "precedents": ["trastuzumab", "pembrolizumab", "adalimumab"],
        },
        Modality.MRNA: {
            "advantages": [
                "Enables protein replacement",
                "Rapid development",
                "No genomic integration",
                "Dose adjustable",
            ],
            "challenges": [
                "Delivery challenges",
                "Immunogenicity of mRNA",
                "Limited duration of expression",
                "Cold chain requirements",
            ],
            "precedents": ["mRNA-1273", "BNT162b2"],
        },
        Modality.CRISPR: {
            "advantages": [
                "Permanent correction possible",
                "Targets previously undruggable genes",
                "Can knockout or correct",
                "One-time treatment potential",
            ],
            "challenges": [
                "Off-target editing risk",
                "Delivery to target tissue",
                "Immune response to Cas proteins",
                "Irreversibility",
            ],
            "precedents": ["exa-cel", "casgevy"],
        },
        Modality.ASO: {
            "advantages": [
                "Knocks down undruggable targets",
                "Can modulate splicing",
                "Sequence-specific",
                "Proven CNS delivery",
            ],
            "challenges": [
                "Repeated dosing needed",
                "Limited tissue distribution",
                "Potential for off-target effects",
            ],
            "precedents": ["nusinersen", "eteplirsen", "inotersen"],
        },
        Modality.GENE_THERAPY: {
            "advantages": [
                "One-time treatment potential",
                "Addresses root cause",
                "Can express full-length protein",
            ],
            "challenges": [
                "Pre-existing immunity to vectors",
                "Insertion site risks",
                "Manufacturing complexity",
                "High cost per dose",
            ],
            "precedents": ["Zolgensma", "Luxturna", "Hemgenix"],
        },
        Modality.NEOANTIGEN: {
            "advantages": [
                "Tumor-specific targeting",
                "Harnesses immune system",
                "Potential for durable response",
            ],
            "challenges": [
                "Requires tumor sequencing",
                "HLA restriction",
                "Manufacturing is patient-specific",
                "Tumor heterogeneity",
            ],
            "precedents": ["personalized cancer vaccines"],
        },
    }

    def __init__(self, oracle=None):
        """Initialize selector."""
        self.oracle = oracle

    def recommend(
        self,
        target: MolecularTarget,
        druggability: DruggabilityReport,
        max_recommendations: int = 3,
    ) -> List[ModalityRecommendation]:
        """
        Recommend modalities for a target.

        Args:
            target: The molecular target
            druggability: Druggability assessment
            max_recommendations: Maximum number of recommendations

        Returns:
            Ranked list of modality recommendations
        """
        recommendations = []

        # First, try to match decision rules
        matched = self._match_decision_rules(target)

        # If no rules matched, use modality scores from druggability
        if not matched:
            # Sort modalities by druggability score
            sorted_modalities = sorted(
                druggability.modality_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for modality_name, score in sorted_modalities[:max_recommendations]:
                try:
                    modality = Modality(modality_name)
                    rec = self._create_recommendation(modality, target, score)
                    recommendations.append(rec)
                except ValueError:
                    pass
        else:
            # Use matched rules
            for modality, rationale in matched:
                score = druggability.modality_scores.get(modality.value, 50)
                rec = self._create_recommendation(modality, target, score, rationale)
                recommendations.append(rec)

        # Sort by score
        recommendations.sort(key=lambda r: r.score, reverse=True)

        # Oracle verification if available
        if self.oracle:
            for rec in recommendations:
                rec.oracle_confidence = self._verify_with_oracle(target, rec.modality)

        return recommendations[:max_recommendations]

    def _match_decision_rules(
        self, target: MolecularTarget
    ) -> List[Tuple[Modality, str]]:
        """Match target against decision rules."""
        matches = []

        for rule in self.DECISION_RULES:
            target_type, mechanism, modalities, rationale = rule

            # Check target type match
            if target.target_type != target_type:
                continue

            # Check mechanism match (None means any mechanism)
            if mechanism is not None and target.mechanism != mechanism:
                continue

            # Rule matched
            for modality in modalities:
                matches.append((modality, rationale))

        return matches

    def _create_recommendation(
        self,
        modality: Modality,
        target: MolecularTarget,
        score: float,
        rationale: str = "",
    ) -> ModalityRecommendation:
        """Create a modality recommendation."""
        info = self.MODALITY_INFO.get(modality, {})

        if not rationale:
            rationale = f"{modality.value} approach for {target.name}"

        return ModalityRecommendation(
            modality=modality,
            score=score,
            rationale=rationale,
            advantages=info.get("advantages", [])[:3],
            challenges=info.get("challenges", [])[:3],
            precedents=info.get("precedents", [])[:3],
        )

    def _verify_with_oracle(self, target: MolecularTarget, modality: Modality) -> float:
        """Verify modality choice with oracle."""
        if not self.oracle:
            return 0.5

        try:
            claim = f"{modality.value} is an appropriate therapeutic modality for targeting {target.name}"
            distractors = [
                f"{modality.value} is not suitable for {target.name}",
                f"Alternative modalities are much better for {target.name}",
                f"There is no precedent for using {modality.value} against this target type",
            ]

            result = self.oracle.verify_claim(
                claim=claim,
                domain="modality_selection",
                distractors=distractors,
            )

            return result.confidence if result.verdict == "TRUE" else 0.3

        except Exception:
            return 0.5

    def recommend_for_target(
        self,
        target_name: str,
        mechanism: str,
        tissue: str = "systemic",
    ) -> List[ModalityRecommendation]:
        """
        Convenience method to recommend modalities from basic info.

        Args:
            target_name: Gene/protein name
            mechanism: Disease mechanism
            tissue: Target tissue

        Returns:
            List of modality recommendations
        """
        from .knowledge import get_known_target_info

        # Create target object
        known_info = get_known_target_info(target_name)

        target_type = TargetType.UNKNOWN
        if known_info:
            type_str = known_info.get("type", "unknown")
            try:
                target_type = TargetType(type_str)
            except ValueError:
                pass

        try:
            mech = Mechanism(mechanism)
        except ValueError:
            mech = Mechanism.UNKNOWN

        target = MolecularTarget(
            name=target_name,
            target_type=target_type,
            mechanism=mech,
        )

        # Create dummy druggability report
        from .druggability import TargetScorer
        scorer = TargetScorer()
        druggability = scorer.score(target)

        return self.recommend(target, druggability)
