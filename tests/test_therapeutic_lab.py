"""
Unit tests for Therapeutic Design Lab.

Tests the full pipeline from pathophysiology parsing to candidate ranking,
without requiring the oracle model.
"""

import pytest
from noethersolve.therapeutic_lab import (
    TherapeuticDesignLab,
    PathophysiologyParser,
    TargetScorer,
    ModalitySelector,
    CandidateGenerator,
    CandidateRanker,
    MolecularTarget,
    PathophysiologyExtraction,
    DruggabilityReport,
    ModalityRecommendation,
    TherapeuticCandidate,
    RankedCandidate,
    DesignReport,
    TargetType,
    Mechanism,
    Modality,
    ValidationLevel,
    design_therapeutic,
    parse_pathophysiology,
    score_target,
    recommend_modality,
)


# ── Types Tests ────────────────────────────────────────────────────────


class TestTypes:
    """Test dataclasses and enums."""

    def test_target_type_enum(self):
        """Test TargetType enum values."""
        assert TargetType.ENZYME.value == "enzyme"
        assert TargetType.KINASE.value == "kinase"
        assert TargetType.RECEPTOR.value == "receptor"
        assert TargetType.TRANSCRIPTION_FACTOR.value == "transcription_factor"

    def test_mechanism_enum(self):
        """Test Mechanism enum values."""
        assert Mechanism.LOSS_OF_FUNCTION.value == "loss_of_function"
        assert Mechanism.GAIN_OF_FUNCTION.value == "gain_of_function"
        assert Mechanism.OVEREXPRESSION.value == "overexpression"

    def test_modality_enum(self):
        """Test Modality enum values."""
        assert Modality.SMALL_MOLECULE.value == "small_molecule"
        assert Modality.ANTIBODY.value == "antibody"
        assert Modality.MRNA.value == "mrna"
        assert Modality.CRISPR.value == "crispr"

    def test_molecular_target_creation(self):
        """Test MolecularTarget dataclass."""
        target = MolecularTarget(
            name="EGFR",
            target_type=TargetType.KINASE,
            mechanism=Mechanism.OVEREXPRESSION,
        )
        assert target.name == "EGFR"
        assert target.target_type == TargetType.KINASE
        assert target.mechanism == Mechanism.OVEREXPRESSION

    def test_molecular_target_str(self):
        """Test MolecularTarget string representation."""
        target = MolecularTarget(
            name="CFTR",
            target_type=TargetType.ION_CHANNEL,
            known_drugs=["ivacaftor", "lumacaftor"],
        )
        s = str(target)
        assert "CFTR" in s
        assert "ion_channel" in s

    def test_therapeutic_candidate_creation(self):
        """Test TherapeuticCandidate dataclass."""
        target = MolecularTarget(name="KRAS", target_type=TargetType.ENZYME)
        candidate = TherapeuticCandidate(
            candidate_id="SM-KRAS-001",
            modality=Modality.SMALL_MOLECULE,
            target=target,
            description="KRAS G12C inhibitor",
            efficacy_score=75.0,
            safety_score=60.0,
        )
        assert candidate.candidate_id == "SM-KRAS-001"
        assert candidate.modality == Modality.SMALL_MOLECULE
        assert candidate.efficacy_score == 75.0


# ── Parser Tests ───────────────────────────────────────────────────────


class TestPathophysiologyParser:
    """Test pathophysiology parsing."""

    def test_parser_initialization(self):
        """Test parser initializes correctly."""
        parser = PathophysiologyParser()
        assert parser is not None

    def test_parse_cftr(self):
        """Test parsing CFTR mutation description."""
        parser = PathophysiologyParser()
        result = parser.parse(
            "CFTR chloride channel loss-of-function mutation causing cystic fibrosis",
            disease_name="Cystic Fibrosis",
        )
        assert isinstance(result, PathophysiologyExtraction)
        assert result.disease_name == "Cystic Fibrosis"
        # Should extract CFTR as a target
        target_names = [t.name for t in result.molecular_targets]
        assert "CFTR" in target_names

    def test_parse_bcr_abl(self):
        """Test parsing BCR-ABL description."""
        parser = PathophysiologyParser()
        result = parser.parse(
            "BCR-ABL fusion protein causes constitutive kinase activity",
            disease_name="CML",
        )
        target_names = [t.name for t in result.molecular_targets]
        # Should extract BCR-ABL as a fusion target
        assert "BCR-ABL" in target_names or "BCR" in target_names or "ABL" in target_names

    def test_parse_mechanism_detection(self):
        """Test mechanism keyword detection."""
        parser = PathophysiologyParser()
        result = parser.parse(
            "EGFR overexpression drives tumor growth",
            disease_name="Lung Cancer",
        )
        targets = result.molecular_targets
        # Should find EGFR with overexpression mechanism
        egfr = next((t for t in targets if t.name == "EGFR"), None)
        if egfr:
            assert egfr.mechanism == Mechanism.OVEREXPRESSION

    def test_parse_empty_input(self):
        """Test parsing empty input."""
        parser = PathophysiologyParser()
        result = parser.parse("", disease_name="Unknown")
        assert isinstance(result, PathophysiologyExtraction)
        assert len(result.molecular_targets) == 0

    def test_parse_extracts_tissues(self):
        """Test tissue extraction."""
        parser = PathophysiologyParser()
        result = parser.parse(
            "CFTR defect affects lung and pancreas epithelium",
            disease_name="CF",
        )
        assert "lung" in result.affected_tissues or "pancreas" in result.affected_tissues


# ── Druggability Tests ─────────────────────────────────────────────────


class TestTargetScorer:
    """Test druggability scoring."""

    def test_scorer_initialization(self):
        """Test scorer initializes correctly."""
        scorer = TargetScorer()
        assert scorer is not None

    def test_score_kinase(self):
        """Test kinase gets high druggability score."""
        scorer = TargetScorer()
        target = MolecularTarget(name="EGFR", target_type=TargetType.KINASE)
        report = scorer.score(target)
        assert isinstance(report, DruggabilityReport)
        assert report.druggability_score >= 70  # Kinases are highly druggable

    def test_score_transcription_factor(self):
        """Test transcription factor gets lower score."""
        scorer = TargetScorer()
        target = MolecularTarget(
            name="MYC", target_type=TargetType.TRANSCRIPTION_FACTOR
        )
        report = scorer.score(target)
        # TFs are hard to drug
        assert report.druggability_score < 50

    def test_score_known_target(self):
        """Test known target gets boosted score."""
        scorer = TargetScorer()
        target = MolecularTarget(name="BRAF", target_type=TargetType.KINASE)
        report = scorer.score(target)
        # BRAF is in knowledge base with drugs
        assert report.druggability_score >= 80
        assert len(report.existing_drugs) > 0

    def test_modality_scores(self):
        """Test per-modality scores are generated."""
        scorer = TargetScorer()
        target = MolecularTarget(name="TEST", target_type=TargetType.ENZYME)
        report = scorer.score(target)
        assert "small_molecule" in report.modality_scores
        assert "antibody" in report.modality_scores
        assert "mrna" in report.modality_scores

    def test_structural_features(self):
        """Test structural features are inferred."""
        scorer = TargetScorer()
        target = MolecularTarget(name="AKT1", target_type=TargetType.KINASE)
        report = scorer.score(target)
        assert report.structural_features["has_binding_pocket"] is True
        assert report.structural_features["has_enzymatic_activity"] is True


# ── Modality Selection Tests ───────────────────────────────────────────


class TestModalitySelector:
    """Test modality selection."""

    def test_selector_initialization(self):
        """Test selector initializes correctly."""
        selector = ModalitySelector()
        assert selector is not None

    def test_recommend_for_kinase(self):
        """Test kinase gets small molecule recommendation."""
        selector = ModalitySelector()
        results = selector.recommend_for_target(
            "EGFR", "gain_of_function", "lung"
        )
        assert len(results) > 0
        modalities = [r.modality for r in results]
        assert Modality.SMALL_MOLECULE in modalities

    def test_recommend_for_secreted_lof(self):
        """Test secreted protein LOF gets mRNA/gene therapy."""
        selector = ModalitySelector()
        results = selector.recommend_for_target(
            "FACTOR_VIII", "loss_of_function", "liver"
        )
        modalities = [r.modality for r in results]
        # Should recommend replacement strategies
        assert Modality.MRNA in modalities or Modality.GENE_THERAPY in modalities

    def test_recommendation_has_rationale(self):
        """Test recommendations include rationale."""
        selector = ModalitySelector()
        results = selector.recommend_for_target("HER2", "overexpression", "breast")
        if results:
            assert results[0].rationale is not None
            assert len(results[0].rationale) > 0

    def test_recommendation_has_advantages(self):
        """Test recommendations include advantages."""
        selector = ModalitySelector()
        results = selector.recommend_for_target("TNF", "overexpression", "systemic")
        if results:
            assert len(results[0].advantages) > 0


# ── Generator Tests ────────────────────────────────────────────────────


class TestCandidateGenerator:
    """Test candidate generation."""

    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        generator = CandidateGenerator()
        assert generator is not None

    def test_generate_small_molecule(self):
        """Test small molecule candidate generation."""
        generator = CandidateGenerator()
        target = MolecularTarget(name="BRAF", target_type=TargetType.KINASE)
        candidates = generator.generate(target, Modality.SMALL_MOLECULE)
        assert len(candidates) > 0
        assert candidates[0].modality == Modality.SMALL_MOLECULE

    def test_generate_antibody(self):
        """Test antibody candidate generation."""
        generator = CandidateGenerator()
        target = MolecularTarget(name="HER2", target_type=TargetType.RECEPTOR)
        candidates = generator.generate(target, Modality.ANTIBODY)
        assert len(candidates) > 0
        assert candidates[0].modality == Modality.ANTIBODY

    def test_generate_mrna(self):
        """Test mRNA candidate generation."""
        generator = CandidateGenerator()
        target = MolecularTarget(name="CFTR", target_type=TargetType.ION_CHANNEL)
        candidates = generator.generate(target, Modality.MRNA)
        assert len(candidates) > 0
        assert candidates[0].modality == Modality.MRNA

    def test_candidate_has_scores(self):
        """Test generated candidates have scores."""
        generator = CandidateGenerator()
        target = MolecularTarget(name="EGFR", target_type=TargetType.KINASE)
        candidates = generator.generate(target, Modality.SMALL_MOLECULE)
        if candidates:
            c = candidates[0]
            assert c.efficacy_score is not None
            assert c.safety_score is not None


# ── Ranker Tests ───────────────────────────────────────────────────────


class TestCandidateRanker:
    """Test candidate ranking."""

    def test_ranker_initialization(self):
        """Test ranker initializes correctly."""
        ranker = CandidateRanker()
        assert ranker is not None

    def test_rank_candidates(self):
        """Test ranking multiple candidates."""
        ranker = CandidateRanker()

        target = MolecularTarget(name="EGFR", target_type=TargetType.KINASE)
        candidates = [
            TherapeuticCandidate(
                candidate_id="SM-001",
                modality=Modality.SMALL_MOLECULE,
                target=target,
                description="Test small molecule",
                efficacy_score=80,
                safety_score=70,
            ),
            TherapeuticCandidate(
                candidate_id="AB-001",
                modality=Modality.ANTIBODY,
                target=target,
                description="Test antibody",
                efficacy_score=60,
                safety_score=85,
            ),
        ]

        context = PathophysiologyExtraction(
            disease_name="Test Disease",
            description="Test description",
        )

        ranked = ranker.rank(candidates, context, apply_oracle_adjustment=False)
        assert len(ranked) == 2
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        # Higher efficacy should rank first (given weights)
        assert ranked[0].candidate.candidate_id == "SM-001"

    def test_rank_generates_recommendations(self):
        """Test ranking generates recommendations."""
        ranker = CandidateRanker()

        target = MolecularTarget(name="KRAS", target_type=TargetType.ENZYME)
        candidates = [
            TherapeuticCandidate(
                candidate_id="SM-KRAS-001",
                modality=Modality.SMALL_MOLECULE,
                target=target,
                description="KRAS inhibitor",
                efficacy_score=75,
                safety_score=65,
            ),
        ]

        context = PathophysiologyExtraction(
            disease_name="Pancreatic Cancer",
            description="KRAS mutation",
        )

        ranked = ranker.rank(candidates, context, apply_oracle_adjustment=False)
        assert ranked[0].recommendation is not None
        assert len(ranked[0].recommendation) > 0

    def test_rank_identifies_concerns(self):
        """Test ranking identifies concerns for low scores."""
        ranker = CandidateRanker()

        target = MolecularTarget(name="TEST", target_type=TargetType.UNKNOWN)
        candidates = [
            TherapeuticCandidate(
                candidate_id="LOW-001",
                modality=Modality.CRISPR,
                target=target,
                description="Low scoring candidate",
                efficacy_score=30,
                safety_score=35,
                modality_data={"offtarget_risk": "HIGH"},
            ),
        ]

        context = PathophysiologyExtraction(
            disease_name="Test",
            description="Test",
        )

        ranked = ranker.rank(candidates, context, apply_oracle_adjustment=False)
        assert len(ranked[0].flagged_concerns) > 0


# ── Full Pipeline Tests ────────────────────────────────────────────────


class TestTherapeuticDesignLab:
    """Test the full design lab pipeline."""

    def test_lab_initialization(self):
        """Test lab initializes correctly."""
        lab = TherapeuticDesignLab(use_oracle=False)
        assert lab is not None

    def test_design_cf(self):
        """Test designing therapeutics for Cystic Fibrosis."""
        lab = TherapeuticDesignLab(use_oracle=False)
        result = lab.design(
            disease="Cystic Fibrosis",
            pathophysiology="CFTR chloride channel loss-of-function mutation",
            max_candidates=5,
        )
        assert isinstance(result, DesignReport)
        assert result.disease == "Cystic Fibrosis"
        assert result.targets_analyzed > 0
        assert len(result.disclaimers) > 0

    def test_design_cancer(self):
        """Test designing therapeutics for cancer."""
        lab = TherapeuticDesignLab(use_oracle=False)
        result = lab.design(
            disease="Lung Cancer",
            pathophysiology="EGFR overexpression and KRAS mutation drive tumor growth",
            max_candidates=5,
        )
        assert isinstance(result, DesignReport)
        assert result.targets_analyzed > 0

    def test_design_with_modality_filter(self):
        """Test filtering to specific modalities."""
        lab = TherapeuticDesignLab(use_oracle=False)
        result = lab.design(
            disease="Test Disease",
            pathophysiology="BRAF mutation causes cell proliferation",
            max_candidates=5,
            modalities=[Modality.SMALL_MOLECULE],
        )
        # All candidates should be small molecules
        for rc in result.ranked_candidates:
            assert rc.candidate.modality == Modality.SMALL_MOLECULE

    def test_design_report_str(self):
        """Test design report string representation."""
        lab = TherapeuticDesignLab(use_oracle=False)
        result = lab.design(
            disease="Test",
            pathophysiology="HER2 overexpression",
            max_candidates=3,
        )
        s = str(result)
        assert "Test" in s
        assert "DISCLAIMER" in s or "Computational" in s.lower()


# ── Convenience Function Tests ─────────────────────────────────────────


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_design_therapeutic_function(self):
        """Test design_therapeutic convenience function."""
        result = design_therapeutic(
            disease="Hemophilia A",
            pathophysiology="Factor VIII deficiency",
            max_candidates=3,
        )
        assert isinstance(result, DesignReport)

    def test_parse_pathophysiology_function(self):
        """Test parse_pathophysiology convenience function."""
        result = parse_pathophysiology(
            "BRCA1 mutation increases breast cancer risk",
            disease_name="Breast Cancer",
        )
        assert isinstance(result, PathophysiologyExtraction)
        target_names = [t.name for t in result.molecular_targets]
        assert "BRCA1" in target_names or "BRCA" in target_names

    def test_score_target_function(self):
        """Test score_target convenience function."""
        result = score_target("EGFR", "kinase")
        assert isinstance(result, DruggabilityReport)
        assert result.druggability_score > 0

    def test_recommend_modality_function(self):
        """Test recommend_modality convenience function."""
        results = recommend_modality("TNF", "overexpression", "systemic")
        assert len(results) > 0
        assert all(isinstance(r, ModalityRecommendation) for r in results)


# ── Edge Cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_target(self):
        """Test handling unknown target."""
        scorer = TargetScorer()
        target = MolecularTarget(
            name="UNKNOWN_GENE_XYZ",
            target_type=TargetType.UNKNOWN,
        )
        report = scorer.score(target)
        # Should still return a report
        assert isinstance(report, DruggabilityReport)
        assert report.druggability_score == 50  # Default score

    def test_no_targets_found(self):
        """Test when no targets are found in description."""
        parser = PathophysiologyParser()
        result = parser.parse(
            "general inflammation without specific genes mentioned",
            disease_name="Unknown",
        )
        # Should return empty but valid extraction
        assert isinstance(result, PathophysiologyExtraction)

    def test_generator_unknown_modality(self):
        """Test generator with less common modality."""
        generator = CandidateGenerator()
        target = MolecularTarget(name="DMD", target_type=TargetType.STRUCTURAL)
        candidates = generator.generate(target, Modality.ASO)
        # Should still generate candidates
        assert isinstance(candidates, list)

    def test_ranker_empty_list(self):
        """Test ranking empty candidate list."""
        ranker = CandidateRanker()
        context = PathophysiologyExtraction(
            disease_name="Test",
            description="Test",
        )
        ranked = ranker.rank([], context)
        assert ranked == []
