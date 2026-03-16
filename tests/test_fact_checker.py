#!/usr/bin/env python3
"""
Tests for unified fact-checker interface.
"""

import pytest
from noethersolve.fact_checker import (
    fact_check, categorize_claim, ClaimCategory, UnifiedResult
)


class TestCategorization:
    """Test automatic claim categorization."""

    def test_water_fuel_detected(self):
        """Water fuel claims should be categorized correctly."""
        claim = "Stanley Meyer's water fuel cell runs on water"
        assert categorize_claim(claim) == ClaimCategory.WATER_FUEL

    def test_hho_detected(self):
        """HHO claims should be categorized as water fuel."""
        claim = "HHO gas produces more energy than electrolysis requires"
        assert categorize_claim(claim) == ClaimCategory.WATER_FUEL

    def test_magnetic_motor_detected(self):
        """Magnetic motor claims should be categorized correctly."""
        claim = "The Perendev permanent magnet motor runs forever"
        assert categorize_claim(claim) == ClaimCategory.MAGNETIC_MOTOR

    def test_cold_fusion_detected(self):
        """Cold fusion claims should be categorized correctly."""
        claim = "Cold fusion in palladium-deuterium produces excess heat"
        assert categorize_claim(claim) == ClaimCategory.COLD_FUSION

    def test_atmospheric_detected(self):
        """Atmospheric electricity claims should be categorized correctly."""
        claim = "Tesla tower extracts energy from the atmosphere"
        assert categorize_claim(claim) == ClaimCategory.ATMOSPHERIC

    def test_perpetual_motion_detected(self):
        """Perpetual motion claims should be categorized correctly."""
        claim = "This perpetual motion machine runs forever"
        assert categorize_claim(claim) == ClaimCategory.PERPETUAL_MOTION

    def test_free_energy_detected(self):
        """Free energy claims should be categorized correctly."""
        claim = "Free energy from the vacuum, over unity device"
        assert categorize_claim(claim) == ClaimCategory.FREE_ENERGY

    def test_ftl_detected(self):
        """FTL claims should be categorized correctly."""
        claim = "Faster than light travel using warp drive"
        assert categorize_claim(claim) == ClaimCategory.FTL

    def test_anti_gravity_detected(self):
        """Anti-gravity claims should be categorized correctly."""
        claim = "EM drive produces thrust without propellant"
        assert categorize_claim(claim) == ClaimCategory.ANTI_GRAVITY

    def test_unknown_for_unrecognized(self):
        """Unrecognized claims should return UNKNOWN."""
        claim = "Something completely unrelated to physics"
        assert categorize_claim(claim) == ClaimCategory.UNKNOWN


class TestFactCheck:
    """Test fact_check function."""

    def test_returns_unified_result(self):
        """fact_check should return UnifiedResult."""
        result = fact_check("Water fuel cell")
        assert isinstance(result, UnifiedResult)

    def test_water_fuel_impossible(self):
        """Water fuel claims should be IMPOSSIBLE."""
        result = fact_check("Stanley Meyer's water car runs on water alone")
        assert result.verdict == "IMPOSSIBLE"
        assert result.category == ClaimCategory.WATER_FUEL

    def test_magnetic_motor_impossible(self):
        """Magnetic motor claims should be IMPOSSIBLE."""
        result = fact_check("The Perendev motor runs on magnets alone")
        assert result.verdict == "IMPOSSIBLE"
        assert result.category == ClaimCategory.MAGNETIC_MOTOR

    def test_perpetual_motion_impossible(self):
        """Perpetual motion claims should be IMPOSSIBLE."""
        result = fact_check("This perpetual motion machine runs forever")
        assert result.verdict == "IMPOSSIBLE"
        assert result.category == ClaimCategory.PERPETUAL_MOTION

    def test_free_energy_impossible(self):
        """Free energy claims should be IMPOSSIBLE."""
        result = fact_check("This over-unity device produces free energy")
        assert result.verdict == "IMPOSSIBLE"
        assert result.category == ClaimCategory.FREE_ENERGY

    def test_ftl_impossible(self):
        """FTL claims should be IMPOSSIBLE."""
        result = fact_check("We can travel faster than light")
        assert result.verdict == "IMPOSSIBLE"
        assert result.category == ClaimCategory.FTL

    def test_cold_fusion_implausible(self):
        """Cold fusion claims should be IMPLAUSIBLE."""
        result = fact_check("Cold fusion produces excess heat in palladium")
        assert result.verdict in ["IMPLAUSIBLE", "VIOLATES_CONSERVATION"]
        assert result.category == ClaimCategory.COLD_FUSION

    def test_atmospheric_not_free(self):
        """Atmospheric claims should be flagged."""
        result = fact_check("Tesla tower extracts unlimited energy from atmosphere")
        assert result.verdict in ["IMPLAUSIBLE", "MISLEADING"]
        assert result.category == ClaimCategory.ATMOSPHERIC

    def test_has_explanations(self):
        """Results should include explanations."""
        result = fact_check("Free energy from zero point vacuum")
        assert len(result.explanations) > 0

    def test_has_physics_basis(self):
        """Results should include physics basis."""
        result = fact_check("Magnetic motor runs forever")
        assert result.physics_basis != ""

    def test_has_confidence(self):
        """Results should include confidence score."""
        result = fact_check("Perpetual motion")
        assert 0 <= result.confidence <= 1


class TestUnifiedResultStructure:
    """Test UnifiedResult structure."""

    def test_all_fields_present(self):
        """UnifiedResult should have all required fields."""
        result = fact_check("Free energy device")

        assert hasattr(result, 'claim_text')
        assert hasattr(result, 'category')
        assert hasattr(result, 'verdict')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'violations')
        assert hasattr(result, 'explanations')
        assert hasattr(result, 'physics_basis')
        assert hasattr(result, 'references')

    def test_violations_is_list(self):
        """Violations should be a list."""
        result = fact_check("Over unity device")
        assert isinstance(result.violations, list)

    def test_explanations_is_list(self):
        """Explanations should be a list."""
        result = fact_check("Magnet motor")
        assert isinstance(result.explanations, list)

    def test_references_is_list(self):
        """References should be a list."""
        result = fact_check("Zero point energy")
        assert isinstance(result.references, list)


class TestCategoryOverride:
    """Test category override functionality."""

    def test_can_override_category(self):
        """Should be able to override automatic categorization."""
        result = fact_check("some vague claim", category=ClaimCategory.FREE_ENERGY)
        assert result.category == ClaimCategory.FREE_ENERGY
        assert result.verdict == "IMPOSSIBLE"

    def test_override_changes_analysis(self):
        """Category override should change the analysis."""
        result1 = fact_check("this device", category=ClaimCategory.FREE_ENERGY)
        result2 = fact_check("this device", category=ClaimCategory.FTL)

        # Different categories should give different physics bases
        assert result1.physics_basis != result2.physics_basis


class TestKnownClaims:
    """Test that known claims are found in the databases."""

    def test_stanley_meyer_found(self):
        """Stanley Meyer should be found in water fuel database."""
        result = fact_check("stanley meyer water fuel cell")
        assert result.verdict == "IMPOSSIBLE"
        assert len(result.violations) > 0

    def test_perendev_found(self):
        """Perendev should be found in magnetic motors database."""
        result = fact_check("perendev motor claims")
        assert result.verdict == "IMPOSSIBLE"

    def test_tesla_tower_found(self):
        """Tesla tower should be found in atmospheric database."""
        result = fact_check("tesla tower atmospheric energy")
        assert result.verdict in ["IMPLAUSIBLE", "IMPOSSIBLE"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
