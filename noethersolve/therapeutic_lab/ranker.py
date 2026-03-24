"""
Candidate Ranker — Multi-criteria ranking with oracle verification.

Evaluates candidates on efficacy, safety, developability, manufacturing, and regulatory.
"""

from typing import List, Dict

from .types import (
    TherapeuticCandidate,
    RankedCandidate,
    PathophysiologyExtraction,
    Modality,
    Mechanism,
)


class CandidateRanker:
    """Rank therapeutic candidates using multi-criteria scoring."""

    # Scoring weights
    WEIGHTS = {
        "efficacy": 0.30,
        "safety": 0.25,
        "developability": 0.20,
        "manufacturing": 0.15,
        "regulatory": 0.10,
    }

    # Modality-specific base scores for manufacturing and regulatory
    MANUFACTURING_SCORES = {
        Modality.SMALL_MOLECULE: 85,
        Modality.ANTIBODY: 70,
        Modality.MRNA: 60,
        Modality.CRISPR: 45,
        Modality.ASO: 65,
        Modality.GENE_THERAPY: 40,
        Modality.NEOANTIGEN: 35,
        Modality.CELL_THERAPY: 30,
    }

    REGULATORY_SCORES = {
        Modality.SMALL_MOLECULE: 80,
        Modality.ANTIBODY: 75,
        Modality.MRNA: 65,
        Modality.CRISPR: 50,
        Modality.ASO: 60,
        Modality.GENE_THERAPY: 45,
        Modality.NEOANTIGEN: 40,
        Modality.CELL_THERAPY: 35,
    }

    def __init__(self, oracle=None):
        """Initialize ranker."""
        self.oracle = oracle

    def rank(
        self,
        candidates: List[TherapeuticCandidate],
        context: PathophysiologyExtraction,
        apply_oracle_adjustment: bool = True,
    ) -> List[RankedCandidate]:
        """
        Rank candidates by combined score.

        Args:
            candidates: List of therapeutic candidates
            context: Disease context for relevance scoring
            apply_oracle_adjustment: Whether to adjust scores based on oracle

        Returns:
            Ranked list of candidates
        """
        ranked = []

        for candidate in candidates:
            # Score each dimension
            scores = self._score_candidate(candidate, context)

            # Apply oracle adjustment
            oracle_factor = 1.0
            if apply_oracle_adjustment and self.oracle:
                oracle_factor = self._get_oracle_adjustment(candidate)

            # Calculate combined score
            combined = sum(
                scores[dim] * self.WEIGHTS[dim]
                for dim in self.WEIGHTS
            ) * oracle_factor

            # Update candidate score
            candidate.combined_score = combined

            # Flag concerns
            concerns = self._identify_concerns(candidate, scores)

            # Generate recommendation
            recommendation = self._generate_recommendation(candidate, scores, concerns)

            ranked.append(RankedCandidate(
                rank=0,  # Will be set after sorting
                candidate=candidate,
                score_breakdown=scores,
                flagged_concerns=concerns,
                recommendation=recommendation,
            ))

        # Sort by combined score
        ranked.sort(key=lambda r: r.candidate.combined_score, reverse=True)

        # Assign ranks
        for i, r in enumerate(ranked, 1):
            r.rank = i

        return ranked

    def _score_candidate(
        self,
        candidate: TherapeuticCandidate,
        context: PathophysiologyExtraction,
    ) -> Dict[str, float]:
        """Score a candidate on all dimensions."""
        # Use provided scores or defaults
        efficacy = candidate.efficacy_score or 50
        safety = candidate.safety_score or 50
        developability = candidate.developability_score or 50

        # Manufacturing and regulatory scores based on modality
        manufacturing = self.MANUFACTURING_SCORES.get(candidate.modality, 50)
        regulatory = self.REGULATORY_SCORES.get(candidate.modality, 50)

        # Adjust efficacy based on mechanism match
        efficacy = self._adjust_efficacy(efficacy, candidate, context)

        # Adjust safety based on modality-specific considerations
        safety = self._adjust_safety(safety, candidate)

        # Adjust developability based on existing precedents
        developability = self._adjust_developability(developability, candidate)

        return {
            "efficacy": min(100, max(0, efficacy)),
            "safety": min(100, max(0, safety)),
            "developability": min(100, max(0, developability)),
            "manufacturing": manufacturing,
            "regulatory": regulatory,
        }

    def _adjust_efficacy(
        self,
        base_score: float,
        candidate: TherapeuticCandidate,
        context: PathophysiologyExtraction,
    ) -> float:
        """Adjust efficacy based on mechanism and context."""
        adjustment = 0

        # Boost if modality matches mechanism
        if candidate.target.mechanism == Mechanism.LOSS_OF_FUNCTION:
            if candidate.modality in {Modality.MRNA, Modality.GENE_THERAPY}:
                adjustment += 10  # Good mechanism match
            elif candidate.modality == Modality.SMALL_MOLECULE:
                adjustment -= 10  # Harder to restore function

        elif candidate.target.mechanism in {Mechanism.GAIN_OF_FUNCTION, Mechanism.OVEREXPRESSION}:
            if candidate.modality in {Modality.SMALL_MOLECULE, Modality.ANTIBODY}:
                adjustment += 10  # Good for inhibition

        # Boost if target has validated drugs
        if candidate.target.known_drugs:
            adjustment += 5

        return base_score + adjustment

    def _adjust_safety(self, base_score: float, candidate: TherapeuticCandidate) -> float:
        """Adjust safety based on modality-specific risks."""
        adjustment = 0

        # CRISPR has off-target concerns
        if candidate.modality == Modality.CRISPR:
            offtarget = candidate.modality_data.get("offtarget_risk", "MODERATE")
            if offtarget == "LOW":
                adjustment += 10
            elif offtarget == "HIGH":
                adjustment -= 15

        # Gene therapy has immunogenicity concerns
        elif candidate.modality == Modality.GENE_THERAPY:
            adjustment -= 5  # General concern about vector immunity

        # Small molecules may have off-target pharmacology
        elif candidate.modality == Modality.SMALL_MOLECULE:
            # Kinase inhibitors tend to have class effects
            if candidate.target.target_type.value == "kinase":
                adjustment -= 5

        return base_score + adjustment

    def _adjust_developability(
        self, base_score: float, candidate: TherapeuticCandidate
    ) -> float:
        """Adjust developability based on precedents."""
        adjustment = 0

        # Boost if existing drugs validate the target
        if candidate.target.known_drugs:
            adjustment += 10

        # Boost if modality has been used for this target type before
        from .knowledge import KNOWN_TARGETS
        target_info = KNOWN_TARGETS.get(candidate.target.name, {})
        if target_info:
            drugs = target_info.get("drugs", [])
            if drugs:
                # Check if existing drugs match the modality
                has_mab = any(d.endswith("mab") for d in drugs)
                has_nib = any(d.endswith("ib") or d.endswith("nib") for d in drugs)

                if candidate.modality == Modality.ANTIBODY and has_mab:
                    adjustment += 15
                elif candidate.modality == Modality.SMALL_MOLECULE and has_nib:
                    adjustment += 15

        return base_score + adjustment

    def _get_oracle_adjustment(self, candidate: TherapeuticCandidate) -> float:
        """Get oracle-based adjustment factor."""
        if not self.oracle:
            return 1.0

        try:
            claim = (
                f"{candidate.modality.value} targeting {candidate.target.name} "
                f"is a viable therapeutic approach"
            )
            distractors = [
                f"{candidate.modality.value} is unlikely to succeed for this target",
                "Alternative modalities are much more promising",
                "There are significant biological barriers to this approach",
            ]

            result = self.oracle.verify_claim(
                claim=claim,
                domain="therapeutic_viability",
                distractors=distractors,
            )

            if result.verdict == "TRUE":
                return 1.0 + 0.15 * result.confidence  # Up to +15% boost
            elif result.verdict == "FALSE":
                return 0.85  # 15% penalty
            else:
                return 1.0

        except Exception:
            return 1.0

    def _identify_concerns(
        self,
        candidate: TherapeuticCandidate,
        scores: Dict[str, float],
    ) -> List[str]:
        """Identify potential concerns with a candidate."""
        concerns = []

        # Low scores
        if scores["efficacy"] < 50:
            concerns.append("Low predicted efficacy")
        if scores["safety"] < 50:
            concerns.append("Safety concerns require attention")
        if scores["developability"] < 50:
            concerns.append("Developability challenges expected")
        if scores["manufacturing"] < 50:
            concerns.append("Manufacturing complexity")
        if scores["regulatory"] < 50:
            concerns.append("Regulatory pathway unclear")

        # Modality-specific concerns
        if candidate.modality == Modality.CRISPR:
            if candidate.modality_data.get("offtarget_risk") in {"HIGH", "MODERATE"}:
                concerns.append("Off-target editing risk")

        elif candidate.modality == Modality.GENE_THERAPY:
            concerns.append("Pre-existing anti-AAV immunity may limit patient population")

        elif candidate.modality == Modality.NEOANTIGEN:
            if not candidate.modality_data.get("pipeline_pass", True):
                concerns.append(f"Limiting step: {candidate.modality_data.get('limiting_step', 'unknown')}")

        # No precedent
        if not candidate.target.known_drugs and scores["developability"] < 60:
            concerns.append("No validated drugs for this target")

        return concerns

    def _generate_recommendation(
        self,
        candidate: TherapeuticCandidate,
        scores: Dict[str, float],
        concerns: List[str],
    ) -> str:
        """Generate a recommendation summary."""
        combined = candidate.combined_score

        if combined >= 75:
            priority = "High priority"
            action = "Proceed to lead optimization"
        elif combined >= 60:
            priority = "Medium priority"
            action = "Address key concerns before advancing"
        elif combined >= 45:
            priority = "Lower priority"
            action = "Consider if alternative modalities are exhausted"
        else:
            priority = "Not recommended"
            action = "Significant barriers to development"

        recommendation = f"{priority}: {action}"

        if concerns:
            recommendation += f". Key issues: {concerns[0]}"

        return recommendation

    def rank_from_dicts(
        self,
        candidates: List[Dict],
        disease_context: str,
    ) -> List[RankedCandidate]:
        """
        Rank candidates from dictionary format.

        For use with MCP tool interface.
        """
        from .types import MolecularTarget, TargetType

        parsed_candidates = []

        for c in candidates:
            target = MolecularTarget(
                name=c.get("target_name", "Unknown"),
                target_type=TargetType(c.get("target_type", "unknown")),
            )

            try:
                modality = Modality(c.get("modality", "small_molecule"))
            except ValueError:
                modality = Modality.SMALL_MOLECULE

            candidate = TherapeuticCandidate(
                candidate_id=c.get("id", "Candidate"),
                modality=modality,
                target=target,
                description=c.get("description", ""),
                efficacy_score=c.get("efficacy_score", 50),
                safety_score=c.get("safety_score", 50),
                developability_score=c.get("developability_score", 50),
                modality_data=c.get("modality_data", {}),
            )
            parsed_candidates.append(candidate)

        # Create minimal context
        context = PathophysiologyExtraction(
            disease_name="Disease",
            description=disease_context,
        )

        return self.rank(parsed_candidates, context)
