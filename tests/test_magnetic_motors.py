#!/usr/bin/env python3
"""
Tests for magnetic motor fact-checking module.
"""

import pytest
from noethersolve.magnetic_motors import (
    MagneticMotorClaim, Verdict, FallacyType,
    analyze_magnetic_motor_claim,
    calculate_magnet_energy,
    check_conservative_field,
    check_energy_storage,
    check_force_symmetry,
    check_thermodynamic_limits,
    MAGNETIC_MOTOR_CLAIMS
)


class TestMagnetEnergyCalculations:
    """Test magnet energy calculations."""

    def test_energy_scales_with_volume(self):
        """Energy should scale linearly with volume."""
        result1 = calculate_magnet_energy(100, "neodymium")
        result2 = calculate_magnet_energy(200, "neodymium")

        assert abs(result2["stored_energy_J"] -
                   2 * result1["stored_energy_J"]) < 1e-10

    def test_neodymium_more_energy_than_ferrite(self):
        """Neodymium magnets should have more energy than ferrite."""
        neo = calculate_magnet_energy(100, "neodymium")
        ferrite = calculate_magnet_energy(100, "ferrite")

        assert neo["stored_energy_J"] > ferrite["stored_energy_J"]

    def test_energy_is_finite(self):
        """Stored energy should be finite."""
        result = calculate_magnet_energy(100, "neodymium")
        assert result["stored_energy_J"] < float('inf')
        assert result["stored_energy_J"] > 0

    def test_energy_density_reasonable(self):
        """Energy density should be in expected range."""
        result = calculate_magnet_energy(100, "neodymium")
        # Neodymium BH_max is ~450 kJ/m³
        assert 400e3 < result["energy_density_J_m3"] < 500e3


class TestConservativeFieldCheck:
    """Test conservative field principle checks."""

    def test_no_input_detected(self):
        """Claims with no input should fail."""
        claim = MagneticMotorClaim(
            name="Test",
            description="Self-running",
            magnet_type="neodymium",
            claimed_output_watts=1000,
            claimed_efficiency=float('inf'),
            mechanism="Magnets spin",
            has_external_input=False
        )
        passed, msg, fallacy = check_conservative_field(claim)
        assert not passed
        assert fallacy == FallacyType.CONSERVATIVE_FIELD

    def test_special_arrangement_detected(self):
        """Claims about special arrangements should fail if no input."""
        claim = MagneticMotorClaim(
            name="Test",
            description="Special",
            magnet_type="neodymium",
            claimed_output_watts=100,
            claimed_efficiency=float('inf'),
            mechanism="Special arrangement creates continuous motion",
            has_external_input=False
        )
        passed, msg, fallacy = check_conservative_field(claim)
        assert not passed

    def test_input_motor_passes(self):
        """Motors with input should pass."""
        claim = MagneticMotorClaim(
            name="Test",
            description="Real motor",
            magnet_type="neodymium",
            claimed_output_watts=1000,
            claimed_efficiency=0.90,
            mechanism="Electromagnetic interaction",
            has_external_input=True
        )
        passed, msg, fallacy = check_conservative_field(claim)
        assert passed


class TestEnergyStorageCheck:
    """Test energy storage principle checks."""

    def test_magnetic_energy_claim_detected(self):
        """Claims about extracting magnetic energy should fail."""
        claim = MagneticMotorClaim(
            name="Test",
            description="Magnetic energy extraction",
            magnet_type="neodymium",
            claimed_output_watts=100,
            claimed_efficiency=float('inf'),
            mechanism="Extracts magnetic energy from magnets",
            has_external_input=False
        )
        passed, msg, fallacy = check_energy_storage(claim)
        assert not passed
        assert fallacy == FallacyType.ENERGY_STORAGE

    def test_perpetual_claim_detected(self):
        """Claims about perpetual operation should fail."""
        claim = MagneticMotorClaim(
            name="Test",
            description="Perpetual",
            magnet_type="neodymium",
            claimed_output_watts=100,
            claimed_efficiency=float('inf'),
            mechanism="Runs forever on magnets",
            has_external_input=False
        )
        passed, msg, fallacy = check_energy_storage(claim)
        assert not passed


class TestForceSymmetryCheck:
    """Test force symmetry checks."""

    def test_asymmetric_force_claim_detected(self):
        """Claims about asymmetric forces should fail."""
        claim = MagneticMotorClaim(
            name="Test",
            description="Asymmetric",
            magnet_type="neodymium",
            claimed_output_watts=100,
            claimed_efficiency=float('inf'),
            mechanism="Asymmetric force arrangement",
            has_external_input=False
        )
        passed, msg, fallacy = check_force_symmetry(claim)
        assert not passed
        assert fallacy == FallacyType.SYMMETRIC_FORCES

    def test_magnetic_gate_claim_detected(self):
        """Claims about magnetic gates/shields should fail."""
        claim = MagneticMotorClaim(
            name="Test",
            description="Gate motor",
            magnet_type="neodymium",
            claimed_output_watts=100,
            claimed_efficiency=float('inf'),
            mechanism="Magnetic gate blocks return force",
            has_external_input=False
        )
        passed, msg, fallacy = check_force_symmetry(claim)
        assert not passed
        assert fallacy == FallacyType.MAGNETIC_SHIELDING


class TestThermodynamicLimits:
    """Test thermodynamic limit checks."""

    def test_over_unity_detected(self):
        """Over-unity efficiency should fail."""
        claim = MagneticMotorClaim(
            name="Test",
            description="Over-unity",
            magnet_type="neodymium",
            claimed_output_watts=200,
            claimed_efficiency=2.0,  # 200%
            mechanism="Special motor",
            has_external_input=True
        )
        passed, msg, fallacy = check_thermodynamic_limits(claim)
        assert not passed
        assert fallacy == FallacyType.PERPETUAL_MOTION

    def test_output_without_input_detected(self):
        """Output without input should fail."""
        claim = MagneticMotorClaim(
            name="Test",
            description="No input",
            magnet_type="neodymium",
            claimed_output_watts=100,
            claimed_efficiency=None,
            mechanism="Magnets spin",
            has_external_input=False
        )
        passed, msg, fallacy = check_thermodynamic_limits(claim)
        assert not passed

    def test_normal_efficiency_passes(self):
        """Normal efficiency should pass."""
        claim = MagneticMotorClaim(
            name="Test",
            description="Normal motor",
            magnet_type="neodymium",
            claimed_output_watts=90,
            claimed_efficiency=0.90,
            mechanism="Standard motor",
            has_external_input=True
        )
        passed, msg, fallacy = check_thermodynamic_limits(claim)
        assert passed


class TestClaimAnalysis:
    """Test overall claim analysis."""

    def test_impossible_verdict_for_perpetual(self):
        """Perpetual motion claims should be IMPOSSIBLE."""
        claim = MagneticMotorClaim(
            name="Test",
            description="Perpetual",
            magnet_type="neodymium",
            claimed_output_watts=1000,
            claimed_efficiency=float('inf'),
            mechanism="Self-running magnetic motor",
            has_external_input=False
        )
        result = analyze_magnetic_motor_claim(claim)
        assert result.verdict == Verdict.IMPOSSIBLE

    def test_correct_verdict_for_real_motor(self):
        """Real motors should be CORRECT."""
        claim = MagneticMotorClaim(
            name="Test",
            description="Real motor",
            magnet_type="neodymium",
            claimed_output_watts=1000,
            claimed_efficiency=0.90,
            mechanism="BLDC motor with electronic control",
            has_external_input=True
        )
        result = analyze_magnetic_motor_claim(claim)
        assert result.verdict == Verdict.CORRECT


class TestStandardClaims:
    """Test analysis of standard claims in database."""

    def test_perendev_impossible(self):
        """Perendev motor should be impossible."""
        result = analyze_magnetic_motor_claim(MAGNETIC_MOTOR_CLAIMS["perendev"])
        assert result.verdict == Verdict.IMPOSSIBLE

    def test_yildiz_impossible(self):
        """Yildiz motor should be impossible."""
        result = analyze_magnetic_motor_claim(MAGNETIC_MOTOR_CLAIMS["yildiz"])
        assert result.verdict == Verdict.IMPOSSIBLE

    def test_howard_johnson_impossible(self):
        """Howard Johnson motor should be impossible."""
        result = analyze_magnetic_motor_claim(MAGNETIC_MOTOR_CLAIMS["howard_johnson"])
        assert result.verdict == Verdict.IMPOSSIBLE

    def test_steorn_orbo_impossible(self):
        """Steorn Orbo should be impossible."""
        result = analyze_magnetic_motor_claim(MAGNETIC_MOTOR_CLAIMS["steorn_orbo"])
        assert result.verdict == Verdict.IMPOSSIBLE

    def test_bldc_motor_correct(self):
        """BLDC motor should be correct."""
        result = analyze_magnetic_motor_claim(MAGNETIC_MOTOR_CLAIMS["bldc_motor"])
        assert result.verdict == Verdict.CORRECT

    def test_ev_motor_correct(self):
        """EV motor should be correct."""
        result = analyze_magnetic_motor_claim(MAGNETIC_MOTOR_CLAIMS["electric_vehicle"])
        assert result.verdict == Verdict.CORRECT

    def test_all_claims_analyzed(self):
        """All claims in database should produce valid verdicts."""
        for name, claim in MAGNETIC_MOTOR_CLAIMS.items():
            result = analyze_magnetic_motor_claim(claim)
            assert result.verdict in [Verdict.IMPOSSIBLE, Verdict.IMPLAUSIBLE,
                                      Verdict.MISLEADING, Verdict.CORRECT]
            assert result.claim_name == claim.name


class TestFallacyDetection:
    """Test that fallacies are properly detected."""

    def test_multiple_fallacies_detected(self):
        """Claims with multiple issues should detect all fallacies."""
        claim = MagneticMotorClaim(
            name="Test",
            description="Multiple issues",
            magnet_type="neodymium",
            claimed_output_watts=1000,
            claimed_efficiency=float('inf'),
            mechanism="Asymmetric magnetic energy extraction forever",
            has_external_input=False
        )
        result = analyze_magnetic_motor_claim(claim)

        # Should detect multiple fallacies
        assert len(result.fallacies) >= 2
        assert FallacyType.PERPETUAL_MOTION in result.fallacies


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
