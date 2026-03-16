#!/usr/bin/env python3
"""
Tests for fringe physics fact-checking module.
"""

import pytest
from noethersolve.fringe_physics import (
    FringeClaim, FringeCategory, ViolationType,
    fact_check_fringe_claim, check_energy_conservation,
    check_entropy_increase, check_momentum_conservation,
    check_causality, check_anti_gravity,
    FRINGE_CLAIMS
)


class TestEnergyConservation:
    """Test First Law of Thermodynamics checks."""

    def test_over_unity_detected(self):
        """Over-unity efficiency violates energy conservation."""
        claim = FringeClaim(
            name="Test",
            description="Over-unity device",
            category=FringeCategory.OVER_UNITY,
            claimed_input_energy=100,
            claimed_output_energy=200,
            claimed_efficiency=2.0,
            requires_fuel=True,
            claimed_mechanism="Magic"
        )
        passed, msg = check_energy_conservation(claim)
        assert not passed
        assert "violates" in msg.lower()

    def test_perpetual_output_detected(self):
        """Perpetual output without input violates energy conservation."""
        claim = FringeClaim(
            name="Test",
            description="Free energy",
            category=FringeCategory.FREE_ENERGY,
            claimed_input_energy=0,
            claimed_output_energy=100,
            claimed_efficiency=None,
            requires_fuel=False,
            claimed_mechanism="Free energy"
        )
        passed, msg = check_energy_conservation(claim)
        assert not passed

    def test_normal_efficiency_passes(self):
        """Normal efficiency (<100%) passes check."""
        claim = FringeClaim(
            name="Test",
            description="Normal engine",
            category=FringeCategory.OVER_UNITY,
            claimed_input_energy=100,
            claimed_output_energy=30,
            claimed_efficiency=0.3,
            requires_fuel=True,
            claimed_mechanism="Internal combustion"
        )
        passed, msg = check_energy_conservation(claim)
        assert passed


class TestEntropyIncrease:
    """Test Second Law of Thermodynamics checks."""

    def test_perpetual_motion_detected(self):
        """Perpetual motion violates Second Law."""
        claim = FringeClaim(
            name="Test",
            description="Perpetual motion",
            category=FringeCategory.PERPETUAL_MOTION,
            claimed_input_energy=0,
            claimed_output_energy=0,
            claimed_efficiency=None,
            requires_fuel=False,
            claimed_mechanism="Self-sustaining"
        )
        passed, msg = check_entropy_increase(claim)
        assert not passed
        assert "second law" in msg.lower()


class TestMomentumConservation:
    """Test momentum conservation checks."""

    def test_reactionless_drive_detected(self):
        """Reactionless drives violate momentum conservation."""
        claim = FringeClaim(
            name="Test",
            description="Reactionless thruster",
            category=FringeCategory.ANTI_GRAVITY,
            claimed_input_energy=1000,
            claimed_output_energy=None,
            claimed_efficiency=None,
            requires_fuel=False,
            claimed_mechanism="Reactionless drive"
        )
        passed, msg = check_momentum_conservation(claim)
        assert not passed
        assert "momentum" in msg.lower()

    def test_em_drive_detected(self):
        """EM Drive claims violate momentum conservation."""
        claim = FringeClaim(
            name="Test",
            description="EM Drive",
            category=FringeCategory.ANTI_GRAVITY,
            claimed_input_energy=1000,
            claimed_output_energy=None,
            claimed_efficiency=None,
            requires_fuel=False,
            claimed_mechanism="EM Drive cavity resonance"
        )
        passed, msg = check_momentum_conservation(claim)
        assert not passed


class TestCausality:
    """Test causality checks."""

    def test_ftl_violates_causality(self):
        """FTL travel violates causality."""
        claim = FringeClaim(
            name="Test",
            description="FTL ship",
            category=FringeCategory.FTL,
            claimed_input_energy=None,
            claimed_output_energy=None,
            claimed_efficiency=None,
            requires_fuel=True,
            claimed_mechanism="Hyperdrive"
        )
        passed, msg = check_causality(claim)
        assert not passed
        assert "causality" in msg.lower()


class TestAntiGravity:
    """Test anti-gravity checks."""

    def test_gravity_shielding_impossible(self):
        """Gravity shielding is impossible."""
        claim = FringeClaim(
            name="Test",
            description="Gravity shield",
            category=FringeCategory.ANTI_GRAVITY,
            claimed_input_energy=1000,
            claimed_output_energy=None,
            claimed_efficiency=None,
            requires_fuel=True,
            claimed_mechanism="Gravity shield using superconductors"
        )
        passed, msg, is_violation = check_anti_gravity(claim)
        assert not passed
        assert is_violation
        assert "impossible" in msg.lower()


class TestFactCheck:
    """Test overall fact-checking."""

    def test_perpetual_motion_impossible(self):
        """Perpetual motion wheel should be impossible."""
        result = fact_check_fringe_claim(FRINGE_CLAIMS["perpetual_motion_wheel"])
        assert result.verdict == "IMPOSSIBLE"
        assert ViolationType.FIRST_LAW in result.violations

    def test_em_drive_impossible(self):
        """EM Drive should be impossible."""
        result = fact_check_fringe_claim(FRINGE_CLAIMS["em_drive"])
        assert result.verdict == "IMPOSSIBLE"
        assert ViolationType.MOMENTUM in result.violations

    def test_ftl_impossible(self):
        """FTL neutrinos should be impossible."""
        result = fact_check_fringe_claim(FRINGE_CLAIMS["ftl_neutrino"])
        assert result.verdict == "IMPOSSIBLE"
        assert ViolationType.CAUSALITY in result.violations

    def test_water_fuel_cell_impossible(self):
        """Water fuel cell should be impossible."""
        result = fact_check_fringe_claim(FRINGE_CLAIMS["water_fuel_cell"])
        assert result.verdict == "IMPOSSIBLE"
        assert ViolationType.FIRST_LAW in result.violations

    def test_gravity_shield_impossible(self):
        """Gravity shielding should be impossible."""
        result = fact_check_fringe_claim(FRINGE_CLAIMS["podkletnov_gravity_shield"])
        assert result.verdict == "IMPOSSIBLE"
        assert ViolationType.GENERAL_RELATIVITY in result.violations

    def test_all_claims_checked(self):
        """All claims in database should be checked."""
        for name, claim in FRINGE_CLAIMS.items():
            result = fact_check_fringe_claim(claim)
            assert result.verdict in ["IMPOSSIBLE", "IMPLAUSIBLE", "UNVERIFIED", "PLAUSIBLE"]
            assert result.claim_name == claim.name


class TestViolationTypes:
    """Test violation type handling."""

    def test_first_law_trumps_other_violations(self):
        """First Law violation should give IMPOSSIBLE verdict."""
        claim = FringeClaim(
            name="Test",
            description="Multi-violation device",
            category=FringeCategory.PERPETUAL_MOTION,
            claimed_input_energy=0,
            claimed_output_energy=1000,
            claimed_efficiency=float('inf'),
            requires_fuel=False,
            claimed_mechanism="Magic perpetual reactionless FTL drive"
        )
        result = fact_check_fringe_claim(claim)
        assert result.verdict == "IMPOSSIBLE"
        assert "energy conservation" in result.physics_notes.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
