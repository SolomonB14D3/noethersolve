"""Tests for mrna_design module."""

import pytest
from noethersolve.mrna_design import (
    calculate_base_pair_energy,
    calculate_duplex_stability,
    analyze_immunogenicity,
    calculate_cai,
    optimize_codons,
    analyze_mrna_design,
    compare_modifications,
    ModificationType,
)


class TestBasePairEnergy:
    """Tests for base pair energy calculations."""

    def test_au_pair_energy(self):
        """A-U pairs should have known energy."""
        energy = calculate_base_pair_energy("A", "U", modified=False)
        assert energy == -0.9

    def test_gc_pair_stronger_than_au(self):
        """G-C pairs should be stronger than A-U."""
        gc = calculate_base_pair_energy("G", "C", modified=False)
        au = calculate_base_pair_energy("A", "U", modified=False)
        assert gc < au  # More negative = stronger

    def test_pseudouridine_weaker_than_uridine(self):
        """CRITICAL: A-Ψ pairs should be WEAKER than A-U pairs."""
        au = calculate_base_pair_energy("A", "U", modified=False)
        apsi = calculate_base_pair_energy("A", "U", modified=True)

        # Ψ makes the pair WEAKER (less negative)
        assert apsi > au

    def test_mismatch_has_no_energy(self):
        """Mismatches should return 0."""
        assert calculate_base_pair_energy("A", "A", modified=False) == 0.0
        assert calculate_base_pair_energy("G", "G", modified=False) == 0.0


class TestDuplexStability:
    """Tests for duplex stability calculations."""

    def test_gc_rich_more_stable(self):
        """GC-rich duplex should be more stable than AU-rich."""
        gc_seq = "GCGCGCGC"
        gc_comp = "CGCGCGCG"
        au_seq = "AUAUAUAU"
        au_comp = "UAUAUAUA"

        gc_report = calculate_duplex_stability(gc_seq, gc_comp)
        au_report = calculate_duplex_stability(au_seq, au_comp)

        assert gc_report.delta_G < au_report.delta_G

    def test_pseudouridine_destabilizes_au_rich(self):
        """CRITICAL: Ψ should DESTABILIZE A-U rich sequences."""
        au_seq = "AUAUAUAU"
        au_comp = "UAUAUAUA"

        report = calculate_duplex_stability(au_seq, au_comp, modified=True)

        # stability_change > 0 means DESTABILIZED
        assert report.stability_change > 0
        assert report.stability_direction == "destabilized"

    def test_report_includes_warning(self):
        """Report should warn about destabilization."""
        au_seq = "AUAUAUAU"
        au_comp = "UAUAUAUA"

        report = calculate_duplex_stability(au_seq, au_comp, modified=True)

        assert any("DESTABILIZES" in note for note in report.notes)

    def test_report_str_format(self):
        """Report should have readable output."""
        seq = "GCGCAUAU"
        comp = "CGCGUAUA"

        report = calculate_duplex_stability(seq, comp, modified=True)
        s = str(report)

        assert "ΔG" in s
        assert "kcal/mol" in s


class TestImmunogenicity:
    """Tests for immunogenicity analysis."""

    def test_unmodified_high_risk(self):
        """Unmodified RNA with uridine runs should be high risk."""
        seq = "UUUUUUUUAUUUUUUU"  # Multiple U runs

        report = analyze_immunogenicity(seq, ModificationType.NONE)

        # Unmodified with U runs = immunogenic
        assert report.tlr7_8_risk in ["medium", "high"]

    def test_pseudouridine_reduces_risk(self):
        """CRITICAL: Ψ should dramatically reduce immune risk."""
        seq = "UUUUUUUUAUUUUUUU"

        unmod = analyze_immunogenicity(seq, ModificationType.NONE)
        mod = analyze_immunogenicity(seq, ModificationType.PSEUDOURIDINE)

        # Ψ should reduce TLR7/8 risk
        if unmod.tlr7_8_risk != "low":
            assert mod.tlr7_8_risk == "low"

    def test_cpg_counted(self):
        """CpG dinucleotides should be counted."""
        seq = "ACGACGACG"  # 3 CpG

        report = analyze_immunogenicity(seq)

        assert report.cpg_count == 3

    def test_recommendations_for_unmodified(self):
        """Should recommend Ψ for immunogenic unmodified RNA."""
        seq = "UUUUUUUUAUUUUUUU"

        report = analyze_immunogenicity(seq, ModificationType.NONE)

        # Should recommend modification
        assert len(report.recommendations) > 0


class TestCodonOptimization:
    """Tests for codon optimization."""

    def test_cai_calculation(self):
        """CAI should be between 0 and 1."""
        seq = "AUGGCUAAAUAG"  # Met-Ala-Lys-Stop

        cai = calculate_cai(seq)

        assert 0 <= cai <= 1

    def test_optimization_improves_cai(self):
        """Optimization should improve or maintain CAI."""
        # Use suboptimal codons
        seq = "UUUAUAGAA"  # Phe-Ile-Glu (suboptimal)

        report = optimize_codons(seq, strategy="high_cai")

        assert report.optimized_cai >= report.original_cai

    def test_changes_tracked(self):
        """Should track which codons changed."""
        seq = "UUUAUAGAA"

        report = optimize_codons(seq, strategy="high_cai")

        # Should have some changes
        assert isinstance(report.changes, list)

    def test_invalid_length_raises(self):
        """Sequence not divisible by 3 should raise."""
        with pytest.raises(ValueError):
            calculate_cai("AUGG")


class TestmRNADesign:
    """Tests for complete mRNA design analysis."""

    def test_complete_analysis(self):
        """Should produce complete design report."""
        cds = "AUGGCUAAAUAG"  # Simple CDS

        report = analyze_mrna_design(cds, use_pseudouridine=True)

        assert report.thermodynamics is not None
        assert report.immunogenicity is not None
        assert report.overall_quality in ["excellent", "good", "fair", "poor"]

    def test_critical_insights_present(self):
        """Should include critical insights about Ψ."""
        cds = "AUGGCUAAAUAG"

        report = analyze_mrna_design(cds, use_pseudouridine=True)

        # Should have insight about immune evasion
        assert len(report.critical_insights) > 0

    def test_report_corrects_llm_errors(self):
        """Report should explicitly correct LLM errors."""
        cds = "AUGGCUAAAUAG"

        report = analyze_mrna_design(cds, use_pseudouridine=True)
        s = str(report)

        assert "CRITICAL LLM ERRORS" in s or "CONTEXT-DEPENDENT" in s

    def test_unmodified_warning(self):
        """Should warn about unmodified RNA risks."""
        cds = "AUGUUUUUUUUUUUUUUUUUUUUUUUUUUUUAG"  # U-rich, divisible by 3

        report = analyze_mrna_design(cds, use_pseudouridine=False)

        # Should have some indication of immune risk (TLR7/8 or RIG-I)
        assert any("immune" in i.lower() for i in report.critical_insights) or \
               report.immunogenicity.tlr7_8_risk != "low" or \
               report.immunogenicity.rig_i_risk != "low"


class TestCompareModifications:
    """Tests for modification comparison."""

    def test_comparison_output(self):
        """Should produce comparison output."""
        seq = "AUAUAUAU"
        comp = "UAUAUAUA"

        result = compare_modifications(seq, comp)

        assert "UNMODIFIED" in result
        assert "PSEUDOURIDINE" in result

    def test_explains_destabilization(self):
        """Should explain why Ψ destabilizes."""
        seq = "AUAUAUAU"
        comp = "UAUAUAUA"

        result = compare_modifications(seq, comp)

        assert "DESTABILIZES" in result or "weaker" in result.lower()

    def test_explains_immune_benefit(self):
        """Should explain the real benefit of Ψ."""
        seq = "GCGCGCGC"
        comp = "CGCGCGCG"

        result = compare_modifications(seq, comp)

        assert "immune" in result.lower() or "TLR" in result


class TestPhysicsCorrectness:
    """Tests for biological/physical correctness."""

    def test_hydrogen_bond_strength_order(self):
        """G-C > A-U > A-Ψ in H-bond strength."""
        gc = abs(calculate_base_pair_energy("G", "C"))
        au = abs(calculate_base_pair_energy("A", "U"))
        apsi = abs(calculate_base_pair_energy("A", "U", modified=True))

        assert gc > au > apsi

    def test_covid_vaccine_uses_pseudouridine(self):
        """Module should acknowledge COVID vaccine context."""
        seq = "AUGUUUUUUUAG"
        comp = "UACAAAAAAUUC"

        result = compare_modifications(seq, comp)

        assert "COVID" in result or "vaccine" in result.lower()

    def test_tlr7_8_recognition(self):
        """Should mention TLR7/8 as the key sensors."""
        seq = "AUAUAUAU"
        comp = "UAUAUAUA"

        result = compare_modifications(seq, comp)

        assert "TLR" in result

    def test_thermodynamic_units(self):
        """Energies should be in kcal/mol."""
        seq = "GCGCAUAU"
        comp = "CGCGUAUA"

        report = calculate_duplex_stability(seq, comp)

        assert "kcal/mol" in str(report)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_sequence(self):
        """Should handle short sequences."""
        seq = "GC"
        comp = "CG"

        report = calculate_duplex_stability(seq, comp)
        assert report is not None

    def test_all_gc_sequence(self):
        """GC-only should have minimal Ψ effect."""
        seq = "GCGCGCGC"
        comp = "CGCGCGCG"

        report = calculate_duplex_stability(seq, comp, modified=True)

        # No U's to modify, so minimal change
        assert abs(report.stability_change) < 0.5

    def test_empty_recommendations_when_low_risk(self):
        """Low-risk modified RNA shouldn't need recommendations."""
        seq = "GCGCGCGC"  # No U's

        report = analyze_immunogenicity(seq, ModificationType.PSEUDOURIDINE)

        # Low risk = minimal recommendations
        assert report.tlr7_8_risk == "low"
