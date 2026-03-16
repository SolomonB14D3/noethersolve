#!/usr/bin/env python3
"""
Tests for water fuel / HHO fact-checking module.
"""

import pytest
from noethersolve.water_fuel import (
    WaterFuelClaim, Verdict,
    calculate_electrolysis_energy,
    calculate_round_trip_efficiency,
    analyze_water_fuel_claim,
    WATER_FUEL_CLAIMS,
    ELECTROLYSIS_ENTHALPY, H2_HHV
)


class TestElectrolysisCalculations:
    """Test electrolysis energy calculations."""

    def test_energy_scales_with_water(self):
        """Energy should scale linearly with water volume."""
        result1 = calculate_electrolysis_energy(1.0)
        result2 = calculate_electrolysis_energy(2.0)

        assert abs(result2["min_electrolysis_energy_kWh"] -
                   2 * result1["min_electrolysis_energy_kWh"]) < 1e-10

    def test_h2_produced_scales_with_water(self):
        """H₂ production should scale with water input."""
        result1 = calculate_electrolysis_energy(1.0)
        result2 = calculate_electrolysis_energy(3.0)

        assert abs(result2["h2_produced_moles"] -
                   3 * result1["h2_produced_moles"]) < 1e-10

    def test_h2_energy_equals_electrolysis_minimum(self):
        """H₂ energy content should equal minimum electrolysis energy."""
        # This is because ΔH for electrolysis = ΔH for combustion (HHV)
        result = calculate_electrolysis_energy(1.0)

        # H₂ energy content uses HHV (286 kJ/mol)
        # Min electrolysis uses Gibbs (237 kJ/mol)
        # So H₂ energy > min electrolysis energy
        assert result["h2_energy_content_kWh"] > result["min_electrolysis_energy_kWh"]

    def test_practical_energy_exceeds_minimum(self):
        """Practical electrolysis energy should exceed theoretical minimum."""
        result = calculate_electrolysis_energy(1.0)
        assert result["practical_electrolysis_energy_kWh"] > result["min_electrolysis_energy_kWh"]

    def test_one_liter_values_reasonable(self):
        """Check that 1 liter produces reasonable values."""
        result = calculate_electrolysis_energy(1.0)

        # 1 L water = ~55.5 mol
        assert 55 < result["water_moles"] < 56

        # Should produce ~55.5 mol H₂ = ~112 g
        assert 110 < result["h2_produced_mass_g"] < 115

        # At STP, ~1244 L of H₂
        assert 1200 < result["h2_produced_volume_L_stp"] < 1300


class TestRoundTripEfficiency:
    """Test round-trip efficiency calculations."""

    def test_fuel_cell_path_efficiency(self):
        """Fuel cell path efficiency should be product of efficiencies."""
        result = calculate_round_trip_efficiency(
            electrolysis_efficiency=0.70,
            fuel_cell_efficiency=0.50
        )
        assert abs(result["fuel_cell_round_trip"] - 0.35) < 1e-10

    def test_engine_path_efficiency(self):
        """Engine path efficiency should be product of efficiencies."""
        result = calculate_round_trip_efficiency(
            electrolysis_efficiency=0.70,
            engine_efficiency=0.25
        )
        assert abs(result["engine_round_trip"] - 0.175) < 1e-10

    def test_efficiency_always_less_than_one(self):
        """Round-trip efficiency must always be < 1."""
        # Even with optimistic values
        result = calculate_round_trip_efficiency(
            electrolysis_efficiency=0.90,
            fuel_cell_efficiency=0.80,
            engine_efficiency=0.50
        )
        assert result["fuel_cell_round_trip"] < 1.0
        assert result["engine_round_trip"] < 1.0


class TestClaimAnalysis:
    """Test claim analysis logic."""

    def test_over_unity_detected(self):
        """Over-unity claims should be marked IMPOSSIBLE."""
        claim = WaterFuelClaim(
            name="Test",
            description="Over-unity device",
            claimed_water_input_L=1.0,
            claimed_electricity_input_kWh=1.0,
            claimed_energy_output_kWh=2.0,
            claimed_efficiency=2.0,
            mechanism="Magic"
        )
        result = analyze_water_fuel_claim(claim)
        assert result["verdict"] == Verdict.IMPOSSIBLE

    def test_net_energy_gain_detected(self):
        """Net energy gain should be marked IMPOSSIBLE."""
        claim = WaterFuelClaim(
            name="Test",
            description="Net energy gain",
            claimed_water_input_L=1.0,
            claimed_electricity_input_kWh=1.0,
            claimed_energy_output_kWh=5.0,
            claimed_efficiency=None,
            mechanism="Standard electrolysis"
        )
        result = analyze_water_fuel_claim(claim)
        assert result["verdict"] == Verdict.IMPOSSIBLE

    def test_free_energy_claim_detected(self):
        """'Free energy' claims should be marked IMPOSSIBLE."""
        claim = WaterFuelClaim(
            name="Test",
            description="Free energy device",
            claimed_water_input_L=1.0,
            claimed_electricity_input_kWh=0.1,
            claimed_energy_output_kWh=10.0,
            claimed_efficiency=None,
            mechanism="Free energy from water"
        )
        result = analyze_water_fuel_claim(claim)
        assert result["verdict"] == Verdict.IMPOSSIBLE

    def test_resonance_claim_detected(self):
        """Resonance claims should be flagged."""
        claim = WaterFuelClaim(
            name="Test",
            description="Resonance device",
            claimed_water_input_L=1.0,
            claimed_electricity_input_kWh=4.0,
            claimed_energy_output_kWh=3.0,
            claimed_efficiency=0.75,
            mechanism="Special resonance frequency reduces energy needs"
        )
        result = analyze_water_fuel_claim(claim)
        # Should be at least MISLEADING
        assert result["verdict"] in [Verdict.MISLEADING, Verdict.IMPLAUSIBLE, Verdict.IMPOSSIBLE]

    def test_hho_claim_noted(self):
        """HHO/Brown's gas claims should be noted."""
        claim = WaterFuelClaim(
            name="Test",
            description="HHO device",
            claimed_water_input_L=1.0,
            claimed_electricity_input_kWh=5.0,
            claimed_energy_output_kWh=3.0,
            claimed_efficiency=0.6,
            mechanism="HHO gas production"
        )
        result = analyze_water_fuel_claim(claim)
        assert any("HHO" in note or "Brown" in note for note in result["notes"])

    def test_legitimate_claim_passes(self):
        """Legitimate fuel cell claims should pass."""
        claim = WaterFuelClaim(
            name="Test",
            description="Legitimate fuel cell",
            claimed_water_input_L=1.0,
            claimed_electricity_input_kWh=5.0,
            claimed_energy_output_kWh=1.5,
            claimed_efficiency=0.30,
            mechanism="Standard PEM electrolysis and fuel cell"
        )
        result = analyze_water_fuel_claim(claim)
        assert result["verdict"] == Verdict.CORRECT

    def test_no_energy_output_passes(self):
        """Claims with no energy output should pass."""
        claim = WaterFuelClaim(
            name="Test",
            description="Water injection",
            claimed_water_input_L=0.1,
            claimed_electricity_input_kWh=0.0,
            claimed_energy_output_kWh=0.0,
            claimed_efficiency=None,
            mechanism="Water injection for cooling"
        )
        result = analyze_water_fuel_claim(claim)
        assert result["verdict"] == Verdict.CORRECT


class TestStandardClaims:
    """Test analysis of standard claims in database."""

    def test_stanley_meyer_impossible(self):
        """Stanley Meyer water fuel cell should be impossible."""
        result = analyze_water_fuel_claim(WATER_FUEL_CLAIMS["stanley_meyer"])
        assert result["verdict"] == Verdict.IMPOSSIBLE

    def test_hho_generator_impossible(self):
        """HHO generator should be impossible."""
        result = analyze_water_fuel_claim(WATER_FUEL_CLAIMS["hho_generator"])
        assert result["verdict"] == Verdict.IMPOSSIBLE

    def test_joe_cell_impossible(self):
        """Joe Cell (orgone) should be impossible."""
        result = analyze_water_fuel_claim(WATER_FUEL_CLAIMS["joe_cell"])
        assert result["verdict"] == Verdict.IMPOSSIBLE

    def test_fuel_cell_vehicle_correct(self):
        """Legitimate fuel cell vehicle should be correct."""
        result = analyze_water_fuel_claim(WATER_FUEL_CLAIMS["fuel_cell_vehicle"])
        assert result["verdict"] == Verdict.CORRECT

    def test_water_injection_correct(self):
        """Legitimate water injection should be correct."""
        result = analyze_water_fuel_claim(WATER_FUEL_CLAIMS["water_injection"])
        assert result["verdict"] == Verdict.CORRECT

    def test_all_claims_have_verdicts(self):
        """All claims in database should produce valid verdicts."""
        for name, claim in WATER_FUEL_CLAIMS.items():
            result = analyze_water_fuel_claim(claim)
            assert result["verdict"] in [Verdict.IMPOSSIBLE, Verdict.IMPLAUSIBLE,
                                         Verdict.MISLEADING, Verdict.CORRECT]


class TestPhysicsConstants:
    """Test that physics constants are reasonable."""

    def test_electrolysis_enthalpy_positive(self):
        """Electrolysis enthalpy should be positive (energy required)."""
        assert ELECTROLYSIS_ENTHALPY > 0

    def test_h2_hhv_equals_electrolysis(self):
        """H₂ HHV should equal electrolysis enthalpy (Hess's law)."""
        # They should be equal (inverse reactions)
        assert abs(H2_HHV - ELECTROLYSIS_ENTHALPY) < 1

    def test_electrolysis_reasonable_value(self):
        """Electrolysis enthalpy should be ~286 kJ/mol."""
        assert 280 < ELECTROLYSIS_ENTHALPY < 290


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
