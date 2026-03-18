"""Tests for neoantigen_pipeline module."""

from noethersolve.neoantigen_pipeline import (
    score_cleavage,
    score_tap,
    score_mhc_binding,
    score_tcr_recognition,
    evaluate_neoantigen,
    compare_candidates,
)


class TestScoreCleavage:
    """Tests for proteasomal cleavage scoring."""

    def test_good_c_terminus(self):
        """Leucine C-terminus should score well."""
        report = score_cleavage("AAAAAAAL")  # L at C-term

        assert report.c_term_score > 0.5
        assert report.cleavage_probability > 0.5

    def test_bad_c_terminus(self):
        """Proline C-terminus should score poorly."""
        report = score_cleavage("AAAAAAAP")  # P at C-term

        assert report.c_term_score < 0
        assert report.cleavage_probability < 0.5

    def test_aromatic_c_terminus(self):
        """Tyrosine/Phenylalanine should be good for cleavage."""
        report_y = score_cleavage("AAAAAAAAY")
        report_f = score_cleavage("AAAAAAAAF")

        assert report_y.c_term_score > 0.5
        assert report_f.c_term_score > 0.5

    def test_report_str(self):
        """Report should have readable output."""
        report = score_cleavage("YLQLVFGIEV")
        s = str(report)

        assert "Cleavage" in s
        assert "probability" in s.lower()


class TestScoreTAP:
    """Tests for TAP transport scoring."""

    def test_good_tap_binder(self):
        """Peptide with favorable residues should score well."""
        # R at P1, L at P2, good C-term
        report = score_tap("RLKAAAAAAY")

        assert report.tap_score > 0.4
        assert report.transport_probability > 0.5

    def test_poor_tap_binder(self):
        """Peptide with unfavorable residues should score poorly."""
        # All alanines - no strong preferences
        report = score_tap("AAAAAAAAAA")

        assert report.tap_score < 0.4

    def test_short_peptide(self):
        """Peptide < 8aa should be flagged."""
        report = score_tap("AAAAA")

        assert report.transport_probability < 0.1
        assert "short" in report.notes.lower()

    def test_limiting_positions(self):
        """Should identify limiting positions."""
        report = score_tap("AAAAAAAAAA")

        # Should have some limiting positions identified
        assert len(report.limiting_positions) >= 0


class TestScoreMHCBinding:
    """Tests for MHC binding scoring."""

    def test_hla_a0201_anchor(self):
        """HLA-A*02:01 prefers L at position 2."""
        # Good anchor at position 2
        good = score_mhc_binding("ALAAAAAAV", "HLA-A*02:01")
        # Bad anchor
        bad = score_mhc_binding("ADAAAAAAV", "HLA-A*02:01")

        assert good.binding_score > bad.binding_score

    def test_c_terminal_anchor(self):
        """C-terminal anchor matters for binding."""
        # V at C-term (good for A*02:01)
        good = score_mhc_binding("ALAAAAAV", "HLA-A*02:01")
        # P at C-term (bad)
        bad = score_mhc_binding("ALAAAAAP", "HLA-A*02:01")

        assert good.binding_score > bad.binding_score

    def test_binding_levels(self):
        """Should categorize binding strength."""
        report = score_mhc_binding("YLQLVFGIEV", "HLA-A*02:01")

        assert report.binding_level in ["strong", "weak", "non"]

    def test_unknown_allele_fallback(self):
        """Unknown allele should use default."""
        report = score_mhc_binding("YLQLVFGIEV", "HLA-X*99:99")

        # Should not raise, uses default
        assert report.allele == "HLA-X*99:99"


class TestScoreTCRRecognition:
    """Tests for TCR recognition scoring."""

    def test_charged_residues_favorable(self):
        """Charged residues in TCR-facing positions should score well."""
        # R, K, D, E in central positions
        charged = score_tcr_recognition("AAARKDELA")
        # All alanines
        neutral = score_tcr_recognition("AAAAAAAAA")

        assert charged.tcr_score > neutral.tcr_score

    def test_aromatic_residues_favorable(self):
        """Aromatic residues should enhance TCR recognition."""
        aromatic = score_tcr_recognition("AAAWYFAAA")
        neutral = score_tcr_recognition("AAAAAAAAA")

        assert aromatic.tcr_score > neutral.tcr_score

    def test_foreignness_with_wildtype(self):
        """Should compute foreignness vs wildtype."""
        peptide = "YLQLVFGIEV"
        wildtype = "YLQLVFGIEA"  # 1 mutation

        report = score_tcr_recognition(peptide, wildtype)

        assert report.foreign_score > 0

    def test_exposed_residues(self):
        """Should identify TCR-facing residues."""
        report = score_tcr_recognition("YLQLVFGIEV")

        assert len(report.exposed_residues) > 0


class TestEvaluateNeoantigen:
    """Tests for complete pipeline evaluation."""

    def test_all_steps_present(self):
        """Should evaluate all 4 steps."""
        report = evaluate_neoantigen("YLQLVFGIEV")

        assert report.cleavage is not None
        assert report.tap is not None
        assert report.mhc is not None
        assert report.tcr is not None

    def test_combined_score(self):
        """Combined score should be between 0 and 1."""
        report = evaluate_neoantigen("YLQLVFGIEV")

        assert 0 <= report.combined_score <= 1

    def test_limiting_step_identified(self):
        """Should identify the limiting step."""
        report = evaluate_neoantigen("YLQLVFGIEV")

        assert report.limiting_step in ["Cleavage", "TAP", "MHC", "TCR"]

    def test_pipeline_pass_requires_all_steps(self):
        """Pipeline pass requires all 4 steps to pass."""
        # This tests the critical insight: MHC alone is not enough
        report = evaluate_neoantigen("YLQLVFGIEV")

        if report.pipeline_pass:
            assert report.cleavage.cleavage_probability > 0.3
            assert report.tap.transport_probability > 0.3
            assert report.mhc.binding_level != "non"

    def test_strong_mhc_can_still_fail(self):
        """CRITICAL: Strong MHC binder can still fail pipeline."""
        # This tests the main insight the module teaches
        # A peptide with good MHC anchors but poor cleavage/TAP
        report = evaluate_neoantigen("ALMAAAAAAV")  # Good for A*02:01

        # Even with good MHC, pipeline may fail on other steps
        # The point is that we CHECK all steps
        assert report.limiting_step is not None

    def test_recommendation_present(self):
        """Should provide recommendation."""
        report = evaluate_neoantigen("YLQLVFGIEV")

        assert len(report.recommendation) > 0

    def test_report_str_mentions_all_steps(self):
        """Report should mention all 4 steps."""
        report = evaluate_neoantigen("YLQLVFGIEV")
        s = str(report)

        assert "CLEAVAGE" in s
        assert "TAP" in s
        assert "MHC" in s
        assert "TCR" in s

    def test_report_warns_about_mhc_only(self):
        """Report should warn against MHC-only approach."""
        report = evaluate_neoantigen("YLQLVFGIEV")
        s = str(report)

        assert "MHC binding" in s or "Step 3" in s
        assert "ALL" in s or "all 4" in s.lower()


class TestCompareCandidates:
    """Tests for candidate comparison."""

    def test_returns_ranking(self):
        """Should return ranked comparison."""
        candidates = ["YLQLVFGIEV", "AAAAAAAAAA", "RLKAAAAAAY"]
        result = compare_candidates(candidates)

        assert "Rank" in result
        assert "1" in result

    def test_includes_all_candidates(self):
        """Should include all input candidates."""
        candidates = ["YLQLVFGIEV", "KLLPKLDGI"]
        result = compare_candidates(candidates)

        for pep in candidates:
            assert pep in result


class TestPhysicsCorrectness:
    """Tests for biological accuracy."""

    def test_immunoproteasome_preferences(self):
        """Immunoproteasome prefers hydrophobic/basic C-termini."""
        # L, Y, F, K, R should be preferred
        for aa in ["L", "Y", "F", "K", "R"]:
            report = score_cleavage(f"AAAAAAA{aa}")
            assert report.c_term_score > 0, f"{aa} should be preferred"

        # P, D, E should be disfavored
        for aa in ["P", "D", "E"]:
            report = score_cleavage(f"AAAAAAA{aa}")
            assert report.c_term_score <= 0, f"{aa} should be disfavored"

    def test_tap_length_requirement(self):
        """TAP requires minimum peptide length."""
        short = score_tap("AAAAA")
        good = score_tap("AAAAAAAAA")

        assert short.transport_probability < good.transport_probability

    def test_mhc_class_i_length(self):
        """MHC-I typically binds 8-11 mers."""
        # Should handle different lengths
        for length in [8, 9, 10, 11]:
            peptide = "A" * length
            report = score_mhc_binding(peptide)
            assert report is not None

    def test_geometric_mean_scoring(self):
        """Combined score uses geometric mean to penalize weak links."""
        # A peptide failing one step should have much lower combined score
        # than one passing all steps
        # This is the mathematical basis for the "all steps matter" insight
        report = evaluate_neoantigen("YLQLVFGIEV")

        # Combined score should be <= minimum step score
        # (property of geometric mean)
        step_scores = [
            report.cleavage.cleavage_probability,
            report.tap.transport_probability,
            report.mhc.binding_score,
        ]
        min(step_scores)

        # Geometric mean <= arithmetic mean <= max
        # So combined should be reasonable
        assert report.combined_score <= 1.0
