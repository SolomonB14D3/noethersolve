#!/usr/bin/env python3
"""
Tests for atmospheric electricity fact-checking module.
"""

import pytest
from noethersolve.atmospheric_electricity import (
    AtmosphericEnergyDevice, analyze_atmospheric_claim,
    calculate_theoretical_power, ATMOSPHERIC_DEVICES,
    EARTH_SURFACE_FIELD, FAIR_WEATHER_CURRENT_DENSITY
)


class TestPhysicsCalculations:
    """Test the physics calculations."""

    def test_current_proportional_to_area(self):
        """Current should scale with collector area."""
        device1 = AtmosphericEnergyDevice(
            name="Small", description="", collector_area_m2=1,
            collector_height_m=10, claimed_power_watts=None, mechanism=""
        )
        device2 = AtmosphericEnergyDevice(
            name="Large", description="", collector_area_m2=100,
            collector_height_m=10, claimed_power_watts=None, mechanism=""
        )

        physics1 = calculate_theoretical_power(device1)
        physics2 = calculate_theoretical_power(device2)

        assert physics2["fair_weather_current_A"] == 100 * physics1["fair_weather_current_A"]

    def test_voltage_proportional_to_height(self):
        """Voltage should scale with height (for low heights)."""
        device1 = AtmosphericEnergyDevice(
            name="Short", description="", collector_area_m2=10,
            collector_height_m=10, claimed_power_watts=None, mechanism=""
        )
        device2 = AtmosphericEnergyDevice(
            name="Tall", description="", collector_area_m2=10,
            collector_height_m=50, claimed_power_watts=None, mechanism=""
        )

        physics1 = calculate_theoretical_power(device1)
        physics2 = calculate_theoretical_power(device2)

        assert physics2["voltage_V"] == 5 * physics1["voltage_V"]

    def test_power_is_current_times_voltage(self):
        """Power = I × V."""
        device = AtmosphericEnergyDevice(
            name="Test", description="", collector_area_m2=100,
            collector_height_m=100, claimed_power_watts=None, mechanism=""
        )
        physics = calculate_theoretical_power(device)

        expected_power = physics["fair_weather_current_A"] * physics["voltage_V"]
        assert abs(physics["theoretical_power_W"] - expected_power) < 1e-20

    def test_realistic_power_less_than_theoretical(self):
        """Realistic power should be less than theoretical."""
        device = AtmosphericEnergyDevice(
            name="Test", description="", collector_area_m2=100,
            collector_height_m=100, claimed_power_watts=None, mechanism=""
        )
        physics = calculate_theoretical_power(device)

        assert physics["realistic_power_W"] < physics["theoretical_power_W"]


class TestClaimAnalysis:
    """Test claim analysis."""

    def test_massive_overclaim_implausible(self):
        """Massively overclaimed power should be implausible."""
        device = AtmosphericEnergyDevice(
            name="Test", description="Overclaim",
            collector_area_m2=1, collector_height_m=1,
            claimed_power_watts=1000,  # Way too high
            mechanism="Fair weather collection"
        )
        results = analyze_atmospheric_claim(device)
        assert results["verdict"] in ["IMPLAUSIBLE", "QUESTIONABLE"]

    def test_reasonable_claim_plausible(self):
        """Reasonable claims should be plausible."""
        device = AtmosphericEnergyDevice(
            name="Test", description="Reasonable",
            collector_area_m2=1000, collector_height_m=1000,
            claimed_power_watts=None,  # No overclaim
            mechanism="Fair weather collection"
        )
        results = analyze_atmospheric_claim(device)
        assert results["verdict"] == "PLAUSIBLE"

    def test_free_energy_claim_misleading(self):
        """'Free energy' claims should be marked misleading."""
        device = AtmosphericEnergyDevice(
            name="Test", description="Free energy scam",
            collector_area_m2=1, collector_height_m=1,
            claimed_power_watts=None,
            mechanism="Free energy from the atmosphere"
        )
        results = analyze_atmospheric_claim(device)
        assert results["verdict"] == "MISLEADING"

    def test_zero_point_claim_misleading(self):
        """'Zero point' claims should be marked misleading."""
        device = AtmosphericEnergyDevice(
            name="Test", description="ZPE scam",
            collector_area_m2=1, collector_height_m=1,
            claimed_power_watts=None,
            mechanism="Zero point vacuum energy extraction"
        )
        results = analyze_atmospheric_claim(device)
        assert results["verdict"] == "MISLEADING"


class TestStandardDevices:
    """Test analysis of standard devices in database."""

    def test_tesla_tower_implausible(self):
        """Tesla's Wardenclyffe claims are implausible."""
        results = analyze_atmospheric_claim(ATMOSPHERIC_DEVICES["tesla_tower"])
        assert results["verdict"] == "IMPLAUSIBLE"

    def test_small_antenna_misleading(self):
        """Small antenna 'free energy' demos are misleading."""
        results = analyze_atmospheric_claim(ATMOSPHERIC_DEVICES["small_antenna"])
        assert results["verdict"] == "MISLEADING"

    def test_large_balloon_plausible(self):
        """Large high-altitude balloon is plausible for low power."""
        results = analyze_atmospheric_claim(ATMOSPHERIC_DEVICES["large_balloon"])
        assert results["verdict"] == "PLAUSIBLE"

    def test_lightning_harvesting_implausible(self):
        """Lightning harvesting MW claims are implausible."""
        results = analyze_atmospheric_claim(ATMOSPHERIC_DEVICES["lightning_rod_harvester"])
        assert results["verdict"] == "IMPLAUSIBLE"


class TestPhysicsConstants:
    """Test that physics constants are reasonable."""

    def test_surface_field_positive(self):
        """Surface field should be positive."""
        assert EARTH_SURFACE_FIELD > 0

    def test_surface_field_reasonable(self):
        """Surface field should be ~100-150 V/m."""
        assert 50 < EARTH_SURFACE_FIELD < 200

    def test_current_density_tiny(self):
        """Fair-weather current density should be picoamps/m²."""
        assert FAIR_WEATHER_CURRENT_DENSITY < 1e-10
        assert FAIR_WEATHER_CURRENT_DENSITY > 1e-15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
