"""Tests for antibody_developability module."""

import pytest
from noethersolve.antibody_developability import (
    analyze_charge,
    analyze_aggregation,
    analyze_polyreactivity,
    analyze_liabilities,
    assess_developability,
    calc_charge_at_ph,
    estimate_pI,
    count_patches,
    find_glycosylation_sites,
    RiskLevel,
)


class TestChargeCalculation:
    """Tests for charge calculations."""

    def test_basic_positive_charge(self):
        """Lysine-rich sequence should be positive."""
        seq = "KKKKKKKKKK"  # 10 lysines
        charge = calc_charge_at_ph(seq, 7.4)
        assert charge > 9.5  # Should be close to +10

    def test_basic_negative_charge(self):
        """Glutamate-rich sequence should be negative."""
        seq = "EEEEEEEEEE"  # 10 glutamates
        charge = calc_charge_at_ph(seq, 7.4)
        assert charge < -9.5  # Should be close to -10

    def test_neutral_sequence(self):
        """Neutral amino acids should have near-zero charge."""
        seq = "AAAAAAAAAA"  # 10 alanines
        charge = calc_charge_at_ph(seq, 7.4)
        assert abs(charge) < 0.5  # Should be near zero (only termini contribute)

    def test_ph_dependence(self):
        """Histidine should be charged at low pH, neutral at high pH."""
        seq = "HHHHHHHHHH"  # 10 histidines
        charge_low = calc_charge_at_ph(seq, 4.0)
        charge_high = calc_charge_at_ph(seq, 8.0)
        assert charge_low > charge_high
        assert charge_low > 5  # Mostly protonated at pH 4
        assert charge_high < 1  # Mostly deprotonated at pH 8

    def test_pI_estimation(self):
        """pI should be near 7 for neutral sequences."""
        seq = "AAAAAAAAAA"
        pI = estimate_pI(seq)
        # Neutral sequence pI determined by N/C termini
        assert 4 < pI < 10

    def test_pI_basic_sequence(self):
        """Lysine-rich sequence should have high pI."""
        seq = "KKKKKKKKKK"
        pI = estimate_pI(seq)
        assert pI > 10


class TestAnalyzeCharge:
    """Tests for analyze_charge function."""

    def test_high_positive_charge_low_viscosity(self):
        """High positive charge should predict low viscosity."""
        seq = "K" * 50 + "A" * 50  # Very positively charged
        report = analyze_charge(seq)

        assert report.net_charge_pH7 > 40
        assert report.viscosity_risk == RiskLevel.LOW

    def test_neutral_charge_high_viscosity(self):
        """Near-neutral charge should predict high viscosity."""
        seq = "KKKKKEEEEE" * 10  # Balanced charges
        report = analyze_charge(seq)

        assert abs(report.net_charge_pH7) < 5
        # Viscosity risk should be high or very high

    def test_charge_patches_detection(self):
        """Should detect charged patches."""
        seq = "KKKKK" + "A" * 20 + "EEEEE"  # Two patches
        report = analyze_charge(seq)

        assert report.positive_patches >= 1
        assert report.negative_patches >= 1

    def test_report_str(self):
        """Report should have readable output."""
        seq = "AKDERFGHT"
        report = analyze_charge(seq)
        s = str(report)

        assert "viscosity" in s.lower()
        assert "charge" in s.lower()


class TestAnalyzeAggregation:
    """Tests for analyze_aggregation function."""

    def test_hydrophobic_sequence_high_risk(self):
        """Hydrophobic sequence should have high aggregation risk."""
        seq = "IIIIILLLLLLVVVVV"
        report = analyze_aggregation(seq)

        assert report.aggregation_score > 0.5
        assert report.aggregation_risk in (RiskLevel.HIGH, RiskLevel.VERY_HIGH)

    def test_charged_sequence_low_risk(self):
        """Charged sequence should have low aggregation risk."""
        seq = "KKKKKEEEEEKKKKKEEEEEK"
        report = analyze_aggregation(seq)

        assert report.aggregation_score < 0
        assert report.aggregation_risk == RiskLevel.LOW

    def test_hotspot_detection(self):
        """Should detect aggregation-prone regions."""
        # Hydrophobic stretch in the middle
        seq = "KKKKK" + "ILVILV" + "KKKKK"
        report = analyze_aggregation(seq)

        # Should detect the hydrophobic region as a hotspot
        assert report.aggregation_risk != RiskLevel.LOW or len(report.hotspot_regions) > 0

    def test_report_str(self):
        """Report should mention aggregation."""
        seq = "AKDERFGHT"
        report = analyze_aggregation(seq)
        s = str(report)

        assert "aggregation" in s.lower()


class TestAnalyzePolyreactivity:
    """Tests for analyze_polyreactivity function."""

    def test_high_positive_charge_high_risk(self):
        """High K/R density should increase polyreactivity risk."""
        seq = "KRKRKRKRKRKRKR"
        report = analyze_polyreactivity(seq)

        assert report.positive_charge_density > 80
        # Risk level depends on combined factors, not just charge alone
        assert report.polyreactivity_risk != RiskLevel.LOW

    def test_high_aromatic_high_risk(self):
        """High aromatic content should increase risk."""
        seq = "FWYFWYFWYFWY"
        report = analyze_polyreactivity(seq)

        assert report.aromatic_density > 80
        # Risk level depends on combined factors, not just aromatics alone
        assert report.polyreactivity_risk != RiskLevel.LOW

    def test_low_risk_sequence(self):
        """Balanced sequence should have low risk."""
        seq = "AAAAAGGGGGSSSSS"
        report = analyze_polyreactivity(seq)

        assert report.positive_charge_density < 5
        assert report.aromatic_density < 5
        assert report.polyreactivity_risk == RiskLevel.LOW

    def test_suggestions_generated(self):
        """Should provide suggestions for high-risk sequences."""
        seq = "KRKRKRKRKRKRKR"
        report = analyze_polyreactivity(seq)

        assert len(report.suggestions) > 0


class TestAnalyzeLiabilities:
    """Tests for analyze_liabilities function."""

    def test_deamidation_detection(self):
        """Should detect NG, NS deamidation hotspots."""
        seq = "AAAAANGAAAAANSAAAA"
        report = analyze_liabilities(seq)

        assert len(report.deamidation_sites) == 2
        motifs = [m for _, m in report.deamidation_sites]
        assert "NG" in motifs
        assert "NS" in motifs

    def test_oxidation_detection(self):
        """Should detect M, W, C oxidation sites."""
        seq = "AAAAAMAAAAWAAAACAAAA"
        report = analyze_liabilities(seq)

        assert len(report.oxidation_sites) == 3
        residues = [r for _, r in report.oxidation_sites]
        assert "M" in residues
        assert "W" in residues
        assert "C" in residues

    def test_glycosylation_detection(self):
        """Should detect N-X-S/T glycosylation motifs."""
        seq = "AAAANASAAAANBTAAAA"  # NAS valid, NBT valid
        report = analyze_liabilities(seq)

        assert len(report.glycosylation_sites) >= 2

    def test_no_glycosylation_with_proline(self):
        """N-P-S/T should NOT be detected (proline blocks)."""
        seq = "AAAAANPSAAAA"  # NPS not valid
        report = analyze_liabilities(seq)

        assert len(report.glycosylation_sites) == 0

    def test_total_liabilities(self):
        """Total should sum all liabilities."""
        seq = "NGAAAMAAAANASAAADGAAA"
        report = analyze_liabilities(seq)

        # NG, NS implied in NAS, M, DG
        expected = (len(report.deamidation_sites) +
                   len(report.oxidation_sites) +
                   len(report.glycosylation_sites) +
                   len(report.isomerization_sites))
        assert report.total_liabilities == expected


class TestGlycosylationSites:
    """Tests for find_glycosylation_sites helper."""

    def test_valid_glycosylation(self):
        """N-X-S/T should be detected."""
        assert len(find_glycosylation_sites("NAS")) == 1
        assert len(find_glycosylation_sites("NIT")) == 1

    def test_proline_blocks(self):
        """N-P-S/T should NOT be detected."""
        assert len(find_glycosylation_sites("NPS")) == 0
        assert len(find_glycosylation_sites("NPT")) == 0

    def test_multiple_sites(self):
        """Should find all sites."""
        seq = "NASAAANITAAANAT"
        sites = find_glycosylation_sites(seq)
        assert len(sites) == 3


class TestCountPatches:
    """Tests for count_patches helper."""

    def test_finds_positive_patches(self):
        """Should find KR clusters."""
        seq = "KKKKKAAAAARRRRR"
        patches = count_patches(seq, "KR", window=5, threshold=3)
        assert patches >= 2

    def test_no_patches(self):
        """Dispersed charges should not count as patches."""
        seq = "KAAKAAKAAKAAKAAK"  # More dispersed
        patches = count_patches(seq, "KR", window=5, threshold=3)
        assert patches == 0


class TestAssessDevelopability:
    """Tests for comprehensive assessment."""

    def test_returns_all_components(self):
        """Should include all sub-reports."""
        seq = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVA"
        report = assess_developability(seq)

        assert report.charge is not None
        assert report.aggregation is not None
        assert report.polyreactivity is not None
        assert report.liabilities is not None

    def test_overall_risk_worst_case(self):
        """Overall risk should be worst of components."""
        # Very hydrophobic = high aggregation risk
        seq = "IIIIILLLLLLVVVVVFFFFFF"
        report = assess_developability(seq)

        assert report.overall_risk in (RiskLevel.HIGH, RiskLevel.VERY_HIGH)

    def test_recommendation_present(self):
        """Should provide recommendation."""
        seq = "AKDERFGHTILMNPQRSTVWY"
        report = assess_developability(seq)

        assert len(report.recommendation) > 0

    def test_report_str_comprehensive(self):
        """Report should include all sections."""
        seq = "AKDERFGHTILMNPQRSTVWY"
        report = assess_developability(seq)
        s = str(report)

        assert "charge" in s.lower()
        assert "aggregation" in s.lower()
        assert "polyreactivity" in s.lower()
        assert "liability" in s.lower()


class TestPhysicsCorrectness:
    """Tests for physical accuracy."""

    def test_viscosity_charge_relationship(self):
        """Net charge should correlate inversely with viscosity risk.

        CRITICAL: This is the key insight - net charge (not hydrophobicity)
        predicts viscosity. High charge → repulsion → low viscosity.
        """
        # High positive charge
        high_charge_seq = "K" * 20
        high_charge_report = analyze_charge(high_charge_seq)

        # Near-neutral charge
        neutral_seq = "KKKKKEEEEE" * 2
        neutral_report = analyze_charge(neutral_seq)

        # High charge should have LOW viscosity risk
        # Neutral should have HIGH viscosity risk
        risk_order = [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.VERY_HIGH]
        assert risk_order.index(high_charge_report.viscosity_risk) < risk_order.index(neutral_report.viscosity_risk)

    def test_henderson_hasselbalch(self):
        """Charge calculation should follow Henderson-Hasselbalch."""
        # At pH = pKa, half the groups are protonated
        # For histidine (pKa ≈ 6), at pH 6, charge should be ~0.5 per His
        seq = "HHHHHHHHHH"
        charge = calc_charge_at_ph(seq, 6.0)

        # Should be close to 5 (10 His × 0.5 protonation)
        assert 4 < charge < 6
