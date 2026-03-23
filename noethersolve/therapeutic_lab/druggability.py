"""
Target Druggability Scorer — Assess tractability of therapeutic targets.

Scores targets based on type, structure, existing drugs, and mechanism.
"""

from typing import Dict, List, Optional

from .types import (
    MolecularTarget,
    DruggabilityReport,
    OracleVerification,
    TargetType,
    Mechanism,
)
from .knowledge import (
    KNOWN_TARGETS,
    MODALITY_PREFERENCES,
    MECHANISM_MODALITY_PREFERENCES,
    get_known_target_info,
    get_modality_preferences,
)


class TargetScorer:
    """Score targets for druggability."""

    # Base druggability scores by target type
    TYPE_SCORES: Dict[TargetType, float] = {
        TargetType.ENZYME: 80,
        TargetType.KINASE: 85,
        TargetType.RECEPTOR: 75,
        TargetType.ION_CHANNEL: 70,
        TargetType.PROTEASE: 80,
        TargetType.SECRETED: 70,
        TargetType.TRANSCRIPTION_FACTOR: 30,
        TargetType.STRUCTURAL: 40,
        TargetType.UNKNOWN: 50,
    }

    # Modality base scores by target type
    MODALITY_TYPE_SCORES: Dict[str, Dict[TargetType, float]] = {
        "small_molecule": {
            TargetType.ENZYME: 90,
            TargetType.KINASE: 95,
            TargetType.RECEPTOR: 70,
            TargetType.ION_CHANNEL: 85,
            TargetType.PROTEASE: 90,
            TargetType.SECRETED: 20,
            TargetType.TRANSCRIPTION_FACTOR: 15,
            TargetType.STRUCTURAL: 20,
            TargetType.UNKNOWN: 50,
        },
        "antibody": {
            TargetType.ENZYME: 40,
            TargetType.KINASE: 30,
            TargetType.RECEPTOR: 90,
            TargetType.ION_CHANNEL: 20,
            TargetType.PROTEASE: 30,
            TargetType.SECRETED: 95,
            TargetType.TRANSCRIPTION_FACTOR: 10,
            TargetType.STRUCTURAL: 30,
            TargetType.UNKNOWN: 40,
        },
        "mrna": {
            TargetType.ENZYME: 60,
            TargetType.KINASE: 50,
            TargetType.RECEPTOR: 50,
            TargetType.ION_CHANNEL: 60,
            TargetType.PROTEASE: 50,
            TargetType.SECRETED: 90,
            TargetType.TRANSCRIPTION_FACTOR: 40,
            TargetType.STRUCTURAL: 70,
            TargetType.UNKNOWN: 50,
        },
        "crispr": {
            TargetType.ENZYME: 60,
            TargetType.KINASE: 60,
            TargetType.RECEPTOR: 60,
            TargetType.ION_CHANNEL: 70,
            TargetType.PROTEASE: 60,
            TargetType.SECRETED: 60,
            TargetType.TRANSCRIPTION_FACTOR: 80,
            TargetType.STRUCTURAL: 85,
            TargetType.UNKNOWN: 60,
        },
        "aso": {
            TargetType.ENZYME: 50,
            TargetType.KINASE: 50,
            TargetType.RECEPTOR: 50,
            TargetType.ION_CHANNEL: 60,
            TargetType.PROTEASE: 50,
            TargetType.SECRETED: 40,
            TargetType.TRANSCRIPTION_FACTOR: 85,
            TargetType.STRUCTURAL: 80,
            TargetType.UNKNOWN: 50,
        },
        "gene_therapy": {
            TargetType.ENZYME: 70,
            TargetType.KINASE: 60,
            TargetType.RECEPTOR: 60,
            TargetType.ION_CHANNEL: 75,
            TargetType.PROTEASE: 60,
            TargetType.SECRETED: 80,
            TargetType.TRANSCRIPTION_FACTOR: 50,
            TargetType.STRUCTURAL: 85,
            TargetType.UNKNOWN: 60,
        },
        "neoantigen": {
            TargetType.ENZYME: 50,
            TargetType.KINASE: 60,
            TargetType.RECEPTOR: 50,
            TargetType.ION_CHANNEL: 30,
            TargetType.PROTEASE: 50,
            TargetType.SECRETED: 40,
            TargetType.TRANSCRIPTION_FACTOR: 60,
            TargetType.STRUCTURAL: 50,
            TargetType.UNKNOWN: 40,
        },
    }

    def __init__(self, oracle=None, use_external_api: bool = False):
        """
        Initialize scorer.

        Args:
            oracle: Optional oracle verifier
            use_external_api: Whether to use UniProt API for additional data
        """
        self.oracle = oracle
        self.use_external_api = use_external_api

    def score(self, target: MolecularTarget) -> DruggabilityReport:
        """
        Score a target for druggability.

        Args:
            target: The molecular target to score

        Returns:
            DruggabilityReport with overall and per-modality scores
        """
        # Get base score from target type
        base_score = self.TYPE_SCORES.get(target.target_type, 50)

        # Adjust based on known information
        known_info = get_known_target_info(target.name)
        if known_info:
            if known_info.get("druggable", False):
                base_score = max(base_score, 70)
            if known_info.get("drugs"):
                base_score = max(base_score, 85)

        # Adjust based on mechanism
        base_score = self._adjust_for_mechanism(base_score, target)

        # Calculate modality scores
        modality_scores = self._calculate_modality_scores(target)

        # Get existing drugs
        existing_drugs = known_info.get("drugs", []) if known_info else []

        # Assess structural features
        structural_features = self._assess_structure(target)

        # Generate rationale
        rationale = self._generate_rationale(target, base_score, modality_scores)

        # Optional oracle verification
        oracle_verified = False
        if self.oracle:
            oracle_verified = self._verify_with_oracle(target, base_score)

        return DruggabilityReport(
            target=target,
            druggability_score=base_score,
            modality_scores=modality_scores,
            structural_features=structural_features,
            existing_drugs=existing_drugs,
            rationale=rationale,
            oracle_verified=oracle_verified,
        )

    def _adjust_for_mechanism(self, score: float, target: MolecularTarget) -> float:
        """Adjust score based on disease mechanism."""
        mechanism = target.mechanism

        # Loss of function is generally harder to drug
        if mechanism == Mechanism.LOSS_OF_FUNCTION:
            if target.target_type == TargetType.SECRETED:
                score += 10  # Can replace with protein/mRNA
            else:
                score -= 10  # Need to restore function

        # Gain of function/overexpression is easier to inhibit
        elif mechanism in {Mechanism.GAIN_OF_FUNCTION, Mechanism.OVEREXPRESSION}:
            score += 10  # Inhibition is tractable

        # Misfolding may be addressable with small molecule chaperones
        elif mechanism == Mechanism.MISFOLDING:
            if target.target_type == TargetType.ION_CHANNEL:
                score += 5  # CFTR-style correctors

        return min(100, max(0, score))

    def _calculate_modality_scores(self, target: MolecularTarget) -> Dict[str, float]:
        """Calculate scores for each therapeutic modality."""
        scores = {}

        for modality, type_scores in self.MODALITY_TYPE_SCORES.items():
            base = type_scores.get(target.target_type, 50)

            # Adjust for mechanism
            base = self._adjust_modality_for_mechanism(base, modality, target)

            # Adjust if drugs already exist in this modality
            known_info = get_known_target_info(target.name)
            if known_info and known_info.get("drugs"):
                drugs = known_info["drugs"]
                # Check if existing drugs are small molecules or biologics
                # (simplified heuristic based on name patterns)
                has_biologic = any(
                    d.endswith("mab") or d.endswith("cept")
                    for d in drugs
                )
                has_small_mol = any(
                    d.endswith("ib") or d.endswith("nib")
                    for d in drugs
                )

                if modality == "antibody" and has_biologic:
                    base = max(base, 90)
                elif modality == "small_molecule" and has_small_mol:
                    base = max(base, 90)

            scores[modality] = min(100, max(0, base))

        return scores

    def _adjust_modality_for_mechanism(
        self, score: float, modality: str, target: MolecularTarget
    ) -> float:
        """Adjust modality score based on mechanism."""
        mechanism = target.mechanism

        if mechanism == Mechanism.LOSS_OF_FUNCTION:
            # For loss of function, prefer replacement strategies
            if modality in {"mrna", "gene_therapy"}:
                score += 15
            elif modality in {"small_molecule", "antibody"}:
                score -= 20  # Can't easily restore function

        elif mechanism in {Mechanism.GAIN_OF_FUNCTION, Mechanism.OVEREXPRESSION}:
            # For gain of function, prefer inhibition strategies
            if modality in {"small_molecule", "antibody", "aso", "crispr"}:
                score += 10

        elif mechanism == Mechanism.MUTATION:
            # For mutations, CRISPR or mutation-specific strategies
            if modality == "crispr":
                score += 15
            elif modality == "small_molecule":
                score += 5  # Mutation-specific inhibitors (e.g., sotorasib)

        return score

    def _assess_structure(self, target: MolecularTarget) -> Dict[str, bool]:
        """Assess structural features relevant to druggability."""
        features = {
            "has_binding_pocket": False,
            "surface_accessible": False,
            "is_membrane_protein": False,
            "has_enzymatic_activity": False,
            "protein_protein_interface": False,
            "known_structure": False,
        }

        # Infer from target type
        if target.target_type in {TargetType.ENZYME, TargetType.KINASE, TargetType.PROTEASE}:
            features["has_binding_pocket"] = True
            features["has_enzymatic_activity"] = True
            features["known_structure"] = True

        if target.target_type == TargetType.RECEPTOR:
            features["surface_accessible"] = True
            features["is_membrane_protein"] = True

        if target.target_type == TargetType.ION_CHANNEL:
            features["is_membrane_protein"] = True
            features["has_binding_pocket"] = True

        if target.target_type == TargetType.SECRETED:
            features["surface_accessible"] = True

        if target.target_type == TargetType.TRANSCRIPTION_FACTOR:
            features["protein_protein_interface"] = True

        # Check if this target has approved drugs (implies known structure)
        known_info = get_known_target_info(target.name)
        if known_info and known_info.get("drugs"):
            features["known_structure"] = True

        return features

    def _generate_rationale(
        self,
        target: MolecularTarget,
        score: float,
        modality_scores: Dict[str, float],
    ) -> str:
        """Generate human-readable rationale."""
        parts = []

        # Overall assessment
        if score >= 80:
            parts.append(f"{target.name} is a highly druggable target")
        elif score >= 60:
            parts.append(f"{target.name} is a moderately druggable target")
        elif score >= 40:
            parts.append(f"{target.name} has limited druggability")
        else:
            parts.append(f"{target.name} is challenging to drug directly")

        # Target type context
        if target.target_type == TargetType.KINASE:
            parts.append("as a kinase with well-defined ATP-binding pocket")
        elif target.target_type == TargetType.RECEPTOR:
            parts.append("with surface accessibility for antibodies")
        elif target.target_type == TargetType.TRANSCRIPTION_FACTOR:
            parts.append("though transcription factors lack binding pockets")
        elif target.target_type == TargetType.SECRETED:
            parts.append("accessible as a secreted protein")

        # Best modality
        best_modality = max(modality_scores.items(), key=lambda x: x[1])
        parts.append(f". Best modality: {best_modality[0]} ({best_modality[1]:.0f}/100)")

        # Existing precedent
        known_info = get_known_target_info(target.name)
        if known_info and known_info.get("drugs"):
            drugs = known_info["drugs"][:2]
            parts.append(f". Approved drugs: {', '.join(drugs)}")

        return "".join(parts)

    def _verify_with_oracle(self, target: MolecularTarget, score: float) -> bool:
        """Verify druggability claim with oracle."""
        if not self.oracle:
            return False

        try:
            claim = f"{target.name} is a druggable therapeutic target"
            distractors = [
                f"{target.name} cannot be targeted with drugs",
                f"{target.name} is too challenging to develop drugs against",
                f"No successful drugs target {target.name}",
            ]

            result = self.oracle.verify_claim(
                claim=claim,
                domain="druggability",
                distractors=distractors,
            )
            return result.verdict == "TRUE"

        except Exception:
            return False
