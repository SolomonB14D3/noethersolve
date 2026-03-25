"""
Therapeutic Design Lab — Pathophysiology to therapeutic candidates.

Full pipeline: disease description → molecular targets → modality selection →
candidate generation → ranking → recommendations.

Usage:
    from noethersolve.therapeutic_lab import TherapeuticDesignLab

    lab = TherapeuticDesignLab()
    result = lab.design(
        disease="Cystic Fibrosis",
        pathophysiology="CFTR chloride channel loss-of-function mutation "
                       "causing thick mucus in lungs and pancreas.",
    )
    print(result)

DISCLAIMER:
    This tool provides COMPUTATIONAL PREDICTIONS for research and educational
    purposes only. All therapeutic candidates require experimental validation:
    - In vitro efficacy and toxicity testing
    - In vivo animal studies
    - Clinical trials with appropriate regulatory oversight
    Not validated for clinical decision-making or regulatory submissions.
"""

from typing import List, Optional

from .types import (
    MolecularTarget,
    PathophysiologyExtraction,
    DruggabilityReport,
    ModalityRecommendation,
    TherapeuticCandidate,
    RankedCandidate,
    DesignReport,
    OracleVerification,
    TargetType,
    Mechanism,
    Modality,
    ValidationLevel,
)
from .parser import PathophysiologyParser
from .druggability import TargetScorer
from .modality import ModalitySelector
from .generator import CandidateGenerator
from .ranker import CandidateRanker


DISCLAIMERS = [
    "COMPUTATIONAL PREDICTIONS ONLY",
    "All candidates require experimental validation",
    "In vitro and in vivo studies needed before clinical use",
    "Not validated for clinical decision-making or regulatory submissions",
]


class TherapeuticDesignLab:
    """
    Main interface for the Therapeutic Design Lab.

    Orchestrates the full pipeline from disease pathophysiology to
    ranked therapeutic candidates.

    Args:
        use_oracle: Whether to use 14B oracle for verification
        verbose: Whether to print progress

    Example:
        lab = TherapeuticDesignLab()
        result = lab.design(
            disease="Cystic Fibrosis",
            pathophysiology="CFTR chloride channel loss-of-function mutation...",
        )
        print(result)
        # Access top candidate:
        print(result.ranked_candidates[0])
    """

    def __init__(self, use_oracle: bool = True, verbose: bool = False):
        """Initialize the lab with all components."""
        self.verbose = verbose

        # Try to load oracle
        self.oracle = None
        if use_oracle:
            try:
                from noethersolve.oracle_tool import verify_claim
                # Wrap verify_claim in a simple object
                self.oracle = type('Oracle', (), {'verify_claim': staticmethod(verify_claim)})()
            except ImportError:
                if verbose:
                    print("Oracle not available, proceeding without verification")

        # Initialize components
        self.parser = PathophysiologyParser(oracle=self.oracle)
        self.target_scorer = TargetScorer(oracle=self.oracle)
        self.modality_selector = ModalitySelector(oracle=self.oracle)
        self.candidate_generator = CandidateGenerator(oracle=self.oracle)
        self.candidate_ranker = CandidateRanker(oracle=self.oracle)

    def design(
        self,
        disease: str,
        pathophysiology: str,
        max_candidates: int = 10,
        modalities: Optional[List[Modality]] = None,
    ) -> DesignReport:
        """
        Design therapeutic candidates for a disease.

        Args:
            disease: Disease name (e.g., "Cystic Fibrosis", "KRAS-mutant pancreatic cancer")
            pathophysiology: Natural language description of disease mechanism
            max_candidates: Maximum number of candidates to return
            modalities: Optional filter for specific modalities

        Returns:
            DesignReport with ranked candidates and recommendations

        Example:
            result = lab.design(
                disease="Cystic Fibrosis",
                pathophysiology="CFTR chloride channel loss-of-function mutation "
                               "causing thick mucus in lungs and pancreas. F508del "
                               "is the most common mutation causing protein misfolding.",
            )
        """
        if self.verbose:
            print(f"Designing therapeutics for: {disease}")

        # Step 1: Parse pathophysiology
        if self.verbose:
            print("  Parsing pathophysiology...")
        extraction = self.parser.parse(pathophysiology, disease_name=disease)

        if not extraction.molecular_targets:
            # No targets found, try with just disease name
            extraction = self.parser.parse(disease, disease_name=disease)

        if self.verbose:
            print(f"  Found {len(extraction.molecular_targets)} targets")

        # Step 2: Score targets for druggability
        if self.verbose:
            print("  Scoring druggability...")
        druggability_reports = []
        for target in extraction.molecular_targets:
            report = self.target_scorer.score(target)
            druggability_reports.append(report)

        # Step 3: Select modalities for top targets
        if self.verbose:
            print("  Selecting modalities...")
        modality_recommendations = {}
        for target, druggability in zip(extraction.molecular_targets, druggability_reports):
            if druggability.druggability_score >= 30:  # Threshold
                recs = self.modality_selector.recommend(target, druggability)
                # Filter by requested modalities if specified
                if modalities:
                    recs = [r for r in recs if r.modality in modalities]
                if recs:
                    modality_recommendations[target.name] = (target, recs)

        # Step 4: Generate candidates
        if self.verbose:
            print("  Generating candidates...")
        all_candidates = []
        for target_name, (target, recs) in modality_recommendations.items():
            for rec in recs[:3]:  # Top 3 modalities per target
                candidates = self.candidate_generator.generate(target, rec.modality)
                all_candidates.extend(candidates)

        if self.verbose:
            print(f"  Generated {len(all_candidates)} candidates")

        # Step 5: Rank candidates
        if self.verbose:
            print("  Ranking candidates...")
        ranked = self.candidate_ranker.rank(all_candidates, extraction)

        # Step 6: Generate recommendations
        recommendations = self._generate_recommendations(ranked[:5], extraction)

        # Count oracle verifications
        n_verifications = len(extraction.oracle_verifications)
        for candidate in all_candidates:
            n_verifications += len(candidate.oracle_verifications)

        if self.verbose:
            print(f"  Done! Top candidate: {ranked[0].candidate.candidate_id if ranked else 'None'}")

        return DesignReport(
            disease=disease,
            pathophysiology_summary=extraction,
            targets_analyzed=len(extraction.molecular_targets),
            candidates_generated=len(all_candidates),
            ranked_candidates=ranked[:max_candidates],
            oracle_verifications=n_verifications,
            recommendations=recommendations,
            disclaimers=DISCLAIMERS,
        )

    def _generate_recommendations(
        self,
        top_candidates: List[RankedCandidate],
        context: PathophysiologyExtraction,
    ) -> List[str]:
        """Generate strategic recommendations based on top candidates."""
        recommendations = []

        if not top_candidates:
            recommendations.append("No viable candidates identified. Consider expanding target search.")
            return recommendations

        # Analyze top candidates
        modalities_used = set(rc.candidate.modality for rc in top_candidates)
        targets_used = set(rc.candidate.target.name for rc in top_candidates)

        # Lead recommendation
        top = top_candidates[0]
        recommendations.append(
            f"Lead candidate: {top.candidate.candidate_id} "
            f"({top.candidate.modality.value} targeting {top.candidate.target.name})"
        )

        # Modality diversity
        if len(modalities_used) == 1:
            recommendations.append(
                f"Consider diversifying modality portfolio beyond {list(modalities_used)[0].value}"
            )
        else:
            recommendations.append(
                f"Multiple modalities represented: {', '.join(m.value for m in modalities_used)}"
            )

        # Target diversity
        if len(targets_used) > 1:
            recommendations.append(
                f"Multiple targets identified: {', '.join(targets_used)}"
            )

        # Concern summary
        all_concerns = []
        for rc in top_candidates:
            all_concerns.extend(rc.flagged_concerns)
        if all_concerns:
            unique_concerns = list(set(all_concerns))[:3]
            recommendations.append(
                f"Key considerations: {'; '.join(unique_concerns)}"
            )

        return recommendations


# ── Convenience Functions ─────────────────────────────────────────────

def design_therapeutic(
    disease: str,
    pathophysiology: str,
    max_candidates: int = 10,
) -> DesignReport:
    """
    Convenience function for one-off therapeutic design.

    Args:
        disease: Disease name
        pathophysiology: Disease mechanism description
        max_candidates: Maximum candidates to return

    Returns:
        DesignReport with ranked candidates
    """
    lab = TherapeuticDesignLab()
    return lab.design(disease, pathophysiology, max_candidates)


def parse_pathophysiology(description: str, disease_name: str = "") -> PathophysiologyExtraction:
    """
    Parse pathophysiology to extract molecular targets.

    Args:
        description: Natural language description
        disease_name: Optional disease name for context

    Returns:
        PathophysiologyExtraction with targets and mechanisms
    """
    parser = PathophysiologyParser()
    return parser.parse(description, disease_name)


def score_target(target_name: str, target_type: str = None) -> DruggabilityReport:
    """
    Score a target for druggability.

    Args:
        target_name: Gene/protein name
        target_type: Optional target type hint

    Returns:
        DruggabilityReport with scores
    """
    tt = TargetType.UNKNOWN
    if target_type:
        try:
            tt = TargetType(target_type)
        except ValueError:
            pass

    target = MolecularTarget(name=target_name, target_type=tt)
    scorer = TargetScorer()
    return scorer.score(target)


def recommend_modality(
    target_name: str,
    mechanism: str,
    tissue: str = "systemic",
) -> List[ModalityRecommendation]:
    """
    Recommend modalities for a target.

    Args:
        target_name: Gene/protein name
        mechanism: Disease mechanism
        tissue: Target tissue

    Returns:
        List of modality recommendations
    """
    selector = ModalitySelector()
    return selector.recommend_for_target(target_name, mechanism, tissue)


# ── Exports ─────────────────────────────────────────────────────────

__all__ = [
    # Main class
    "TherapeuticDesignLab",
    # Convenience functions
    "design_therapeutic",
    "parse_pathophysiology",
    "score_target",
    "recommend_modality",
    # Types
    "MolecularTarget",
    "PathophysiologyExtraction",
    "DruggabilityReport",
    "ModalityRecommendation",
    "TherapeuticCandidate",
    "RankedCandidate",
    "DesignReport",
    "OracleVerification",
    "TargetType",
    "Mechanism",
    "Modality",
    "ValidationLevel",
    # Components (for advanced use)
    "PathophysiologyParser",
    "TargetScorer",
    "ModalitySelector",
    "CandidateGenerator",
    "CandidateRanker",
]
