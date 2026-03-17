"""Tests for seismic wave velocity and elastic moduli calculator."""

import math
import pytest

from noethersolve.seismic_waves import (
    calc_seismic_velocity,
    calc_velocity_from_poisson,
    poisson_from_velocities,
    convert_elastic_moduli,
    calc_reflection_coefficient,
    critical_angle,
    snells_law,
    vp_vs_ratio_bounds,
    SeismicVelocityReport,
    PoissonRatioReport,
    ElasticModuliReport,
    ReflectionReport,
)


class TestSeismicVelocity:
    """Tests for basic seismic velocity calculations."""

    def test_granite_velocities(self):
        """Test P and S wave velocities for granite."""
        # Granite: K ~ 50 GPa, G ~ 30 GPa, rho ~ 2700 kg/m^3
        K = 50e9  # Pa
        G = 30e9  # Pa
        rho = 2700  # kg/m^3

        result = calc_seismic_velocity(K, G, rho)

        # Expected: Vp ~ 5900 m/s, Vs ~ 3300 m/s
        assert 5500 < result.Vp < 6500
        assert 3000 < result.Vs < 4000

        # Vp must always be greater than Vs
        assert result.Vp > result.Vs

        # Vp/Vs ratio for typical rock
        assert 1.4 < result.Vp_Vs_ratio < 2.0

    def test_vp_formula_correct(self):
        """Verify Vp = sqrt((K + 4G/3) / rho) exactly."""
        K, G, rho = 100e9, 40e9, 3000

        result = calc_seismic_velocity(K, G, rho)

        # Manual calculation
        M = K + (4.0 * G / 3.0)
        Vp_expected = math.sqrt(M / rho)

        assert abs(result.Vp - Vp_expected) < 1e-6

    def test_vs_formula_correct(self):
        """Verify Vs = sqrt(G / rho) exactly."""
        K, G, rho = 100e9, 40e9, 3000

        result = calc_seismic_velocity(K, G, rho)

        Vs_expected = math.sqrt(G / rho)
        assert abs(result.Vs - Vs_expected) < 1e-6

    def test_poisson_ratio_from_moduli(self):
        """Test that Poisson ratio is correctly computed from K and G."""
        # For nu = 0.25 (Poisson solid): 3K - 2G = 0.25*(6K + 2G)
        # Simplifies to K = 5G/3
        G = 30e9
        K = 5 * G / 3  # Should give nu = 0.25

        result = calc_seismic_velocity(K, G, 2700)

        assert abs(result.poisson_ratio - 0.25) < 1e-6

    def test_vp_vs_ratio_for_poisson_solid(self):
        """For nu = 0.25, Vp/Vs = sqrt(3)."""
        G = 30e9
        K = 5 * G / 3  # nu = 0.25

        result = calc_seismic_velocity(K, G, 2700)

        assert abs(result.Vp_Vs_ratio - math.sqrt(3)) < 1e-4

    def test_acoustic_impedance(self):
        """Test acoustic impedance Z = rho * V."""
        K, G, rho = 50e9, 30e9, 2700

        result = calc_seismic_velocity(K, G, rho)

        assert abs(result.acoustic_impedance_P - rho * result.Vp) < 1e-6
        assert abs(result.acoustic_impedance_S - rho * result.Vs) < 1e-6

    def test_liquid_case_zero_shear(self):
        """Test liquid case where G = 0."""
        K, G, rho = 2.2e9, 0, 1000  # Water-like

        result = calc_seismic_velocity(K, G, rho)

        # Vs should be 0
        assert result.Vs == 0.0

        # Vp = sqrt(K/rho) when G=0
        assert abs(result.Vp - math.sqrt(K / rho)) < 1e-6

        # Poisson ratio should be 0.5 (liquid limit)
        assert abs(result.poisson_ratio - 0.5) < 1e-6

        # Vp/Vs should be infinity
        assert result.Vp_Vs_ratio == float('inf')

        assert result.material_type == "liquid (nu = 0.5, no shear)"

    def test_stability_classification(self):
        """Test thermodynamic stability checking."""
        # Normal rock - should be stable and typical
        result = calc_seismic_velocity(50e9, 30e9, 2700)
        assert result.is_thermodynamically_stable
        assert result.is_typical_rock

        # Soft sediment (high nu ~ 0.48)
        K_soft, G_soft = 10e9, 0.3e9  # High K/G ratio
        result_soft = calc_seismic_velocity(K_soft, G_soft, 1800)
        assert result_soft.is_thermodynamically_stable
        assert not result_soft.is_typical_rock

    def test_negative_K_raises(self):
        """Negative bulk modulus should raise."""
        with pytest.raises(ValueError, match="positive"):
            calc_seismic_velocity(-50e9, 30e9, 2700)

    def test_negative_G_raises(self):
        """Negative shear modulus should raise."""
        with pytest.raises(ValueError, match="negative"):
            calc_seismic_velocity(50e9, -30e9, 2700)

    def test_negative_density_raises(self):
        """Negative density should raise."""
        with pytest.raises(ValueError, match="positive"):
            calc_seismic_velocity(50e9, 30e9, -2700)


class TestVelocityFromPoisson:
    """Tests for velocity calculation from E and nu."""

    def test_steel_velocities(self):
        """Test steel parameters."""
        E = 200e9  # Pa
        nu = 0.3
        rho = 7800  # kg/m^3

        result = calc_velocity_from_poisson(E, nu, rho)

        # Steel: Vp ~ 5900 m/s, Vs ~ 3200 m/s
        assert 5500 < result.Vp < 6500
        assert 3000 < result.Vs < 3500

    def test_conversion_roundtrip(self):
        """E,nu -> K,G -> back should be consistent."""
        E = 150e9
        nu = 0.28
        rho = 3000

        result = calc_velocity_from_poisson(E, nu, rho)

        # Check computed Poisson ratio matches input
        assert abs(result.poisson_ratio - nu) < 1e-6

        # Check computed Young's modulus matches input
        assert abs(result.youngs_modulus_E - E) / E < 1e-6

    def test_invalid_poisson_raises(self):
        """Poisson ratio outside valid range should raise."""
        with pytest.raises(ValueError, match="Poisson"):
            calc_velocity_from_poisson(200e9, 0.5, 7800)  # At limit

        with pytest.raises(ValueError, match="Poisson"):
            calc_velocity_from_poisson(200e9, -1.0, 7800)  # At limit


class TestPoissonFromVelocities:
    """Tests for Poisson ratio calculation from measured velocities."""

    def test_poisson_solid_reference(self):
        """Vp/Vs = sqrt(3) should give nu = 0.25."""
        Vp = 6000
        Vs = Vp / math.sqrt(3)

        result = poisson_from_velocities(Vp, Vs)

        assert abs(result.poisson_ratio - 0.25) < 1e-6

    def test_zero_poisson_material(self):
        """Vp/Vs = sqrt(2) should give nu = 0."""
        Vp = 6000
        Vs = Vp / math.sqrt(2)

        result = poisson_from_velocities(Vp, Vs)

        assert abs(result.poisson_ratio) < 1e-6

    def test_high_vp_vs_ratio(self):
        """High Vp/Vs indicates high Poisson ratio."""
        Vp = 6000
        Vs = 3000  # Vp/Vs = 2

        result = poisson_from_velocities(Vp, Vs)

        # nu = (4 - 2*2) / (2*(4 - 2)) = 0 / 4 = 0... wait let me recalculate
        # Actually: nu = (Vp^2 - 2*Vs^2) / (2*(Vp^2 - Vs^2))
        # = (36e6 - 2*9e6) / (2*(36e6 - 9e6))
        # = (36 - 18) / (2*27) = 18/54 = 1/3
        assert abs(result.poisson_ratio - 1/3) < 1e-6

    def test_liquid_zero_vs(self):
        """Zero Vs should give nu = 0.5 (liquid)."""
        result = poisson_from_velocities(1500, 0)

        assert abs(result.poisson_ratio - 0.5) < 1e-6
        assert result.material_type == "liquid"

    def test_formula_correctness(self):
        """Verify the exact formula is implemented correctly."""
        Vp, Vs = 5500, 3200

        result = poisson_from_velocities(Vp, Vs)

        # Manual calculation
        Vp2, Vs2 = Vp**2, Vs**2
        nu_expected = (Vp2 - 2*Vs2) / (2*(Vp2 - Vs2))

        assert abs(result.poisson_ratio - nu_expected) < 1e-10

    def test_k_over_g_ratio(self):
        """Test K/G ratio computation."""
        Vp, Vs = 6000, 3000  # Vp/Vs = 2

        result = poisson_from_velocities(Vp, Vs)

        # K/G = Vp^2/Vs^2 - 4/3 = 4 - 4/3 = 8/3
        assert abs(result.K_over_G - 8/3) < 1e-6

    def test_invalid_vs_greater_than_vp(self):
        """Vs >= Vp is physically impossible."""
        with pytest.raises(ValueError, match="less than"):
            poisson_from_velocities(3000, 4000)

    def test_negative_velocity_raises(self):
        """Negative velocities should raise."""
        with pytest.raises(ValueError, match="positive"):
            poisson_from_velocities(-6000, 3000)


class TestElasticModuliConversion:
    """Tests for elastic moduli conversions."""

    def test_k_g_input(self):
        """Test conversion from K and G."""
        K, G = 50e9, 30e9

        result = convert_elastic_moduli(K=K, G=G)

        assert result.bulk_modulus_K == K
        assert result.shear_modulus_G == G

        # Check derived quantities
        # nu = (3K - 2G) / (6K + 2G)
        nu_expected = (3*K - 2*G) / (6*K + 2*G)
        assert abs(result.poisson_ratio_nu - nu_expected) < 1e-10

        # E = 9KG / (3K + G)
        E_expected = 9*K*G / (3*K + G)
        assert abs(result.youngs_modulus_E - E_expected) < 1e-6

        # lambda = K - 2G/3
        lam_expected = K - 2*G/3
        assert abs(result.lame_lambda - lam_expected) < 1e-6

        # M = K + 4G/3
        M_expected = K + 4*G/3
        assert abs(result.p_wave_modulus_M - M_expected) < 1e-6

    def test_e_nu_input(self):
        """Test conversion from E and nu."""
        E, nu = 200e9, 0.3

        result = convert_elastic_moduli(E=E, nu=nu)

        # Check roundtrip
        assert abs(result.youngs_modulus_E - E) / E < 1e-6
        assert abs(result.poisson_ratio_nu - nu) < 1e-6

        # Verify K and G
        K_expected = E / (3*(1-2*nu))
        G_expected = E / (2*(1+nu))
        assert abs(result.bulk_modulus_K - K_expected) / K_expected < 1e-6
        assert abs(result.shear_modulus_G - G_expected) / G_expected < 1e-6

    def test_k_nu_input(self):
        """Test conversion from K and nu."""
        K, nu = 166.67e9, 0.25

        result = convert_elastic_moduli(K=K, nu=nu)

        assert abs(result.bulk_modulus_K - K) / K < 1e-6
        assert abs(result.poisson_ratio_nu - nu) < 1e-6

    def test_g_e_input(self):
        """Test conversion from G and E."""
        G, E = 80e9, 200e9

        result = convert_elastic_moduli(G=G, E=E)

        assert abs(result.shear_modulus_G - G) / G < 1e-6
        assert abs(result.youngs_modulus_E - E) / E < 1e-6

    def test_k_lambda_input(self):
        """Test conversion from K and lambda."""
        K, lam = 100e9, 60e9

        result = convert_elastic_moduli(K=K, lam=lam)

        assert abs(result.bulk_modulus_K - K) / K < 1e-6
        assert abs(result.lame_lambda - lam) / lam < 1e-6

    def test_wrong_number_of_inputs(self):
        """Should raise if not exactly 2 moduli provided."""
        with pytest.raises(ValueError, match="exactly 2"):
            convert_elastic_moduli(K=50e9)  # Only 1

        with pytest.raises(ValueError, match="exactly 2"):
            convert_elastic_moduli(K=50e9, G=30e9, E=100e9)  # 3

    def test_poisson_out_of_range(self):
        """Invalid Poisson ratio should raise."""
        with pytest.raises(ValueError, match="Poisson"):
            convert_elastic_moduli(E=200e9, nu=0.6)


class TestReflectionCoefficient:
    """Tests for seismic reflection calculations."""

    def test_equal_impedance_no_reflection(self):
        """Equal impedances = no reflection."""
        result = calc_reflection_coefficient(
            rho1=2500, Vp1=4000,
            rho2=2500, Vp2=4000
        )

        assert abs(result.R_P) < 1e-10
        assert abs(result.T_P - 1.0) < 1e-10

    def test_hard_boundary_positive_R(self):
        """Higher impedance layer = positive R (no polarity flip)."""
        result = calc_reflection_coefficient(
            rho1=2000, Vp1=3000,  # Z1 = 6e6
            rho2=3000, Vp2=5000   # Z2 = 15e6
        )

        assert result.R_P > 0  # Z2 > Z1

    def test_soft_boundary_negative_R(self):
        """Lower impedance layer = negative R (polarity flip)."""
        result = calc_reflection_coefficient(
            rho1=3000, Vp1=5000,  # Z1 = 15e6
            rho2=2000, Vp2=3000   # Z2 = 6e6
        )

        assert result.R_P < 0  # Z2 < Z1

    def test_formula_correctness(self):
        """Verify R = (Z2-Z1)/(Z2+Z1) exactly."""
        rho1, Vp1 = 2500, 4000
        rho2, Vp2 = 2800, 5000

        Z1 = rho1 * Vp1
        Z2 = rho2 * Vp2

        result = calc_reflection_coefficient(rho1, Vp1, rho2, Vp2)

        R_expected = (Z2 - Z1) / (Z2 + Z1)
        T_expected = 2 * Z1 / (Z2 + Z1)

        assert abs(result.R_P - R_expected) < 1e-10
        assert abs(result.T_P - T_expected) < 1e-10

    def test_energy_conservation(self):
        """Energy must be conserved: R^2 + (Z2/Z1)*T^2 = 1."""
        result = calc_reflection_coefficient(
            rho1=2500, Vp1=4000,
            rho2=3000, Vp2=5500
        )

        assert result.energy_conserved

    def test_with_s_wave_velocities(self):
        """Test S-wave reflection coefficients."""
        result = calc_reflection_coefficient(
            rho1=2500, Vp1=5000, Vs1=2900,
            rho2=2700, Vp2=5500, Vs2=3200
        )

        # S-wave coefficients should be computed
        assert result.R_S != 0.0 or result.T_S != 0.0


class TestCriticalAngle:
    """Tests for critical angle calculations."""

    def test_basic_critical_angle(self):
        """Test critical angle calculation."""
        V1 = 3000
        V2 = 6000  # V2 = 2*V1

        theta_c = critical_angle(V1, V2)

        # sin(theta_c) = V1/V2 = 0.5 -> theta_c = 30 degrees
        assert abs(theta_c - 30.0) < 1e-6

    def test_no_critical_angle_when_v1_greater(self):
        """No critical angle if V1 >= V2."""
        with pytest.raises(ValueError, match="V1 must be"):
            critical_angle(6000, 3000)

    def test_critical_angle_approaches_90(self):
        """As V1 approaches V2, critical angle approaches 90."""
        V2 = 6000
        V1 = 5999  # Very close to V2

        theta_c = critical_angle(V1, V2)

        assert theta_c > 85.0


class TestSnellsLaw:
    """Tests for Snell's law calculations."""

    def test_normal_incidence(self):
        """Normal incidence stays normal."""
        theta2, transmitted = snells_law(0.0, 3000, 5000)

        assert abs(theta2) < 1e-10
        assert transmitted

    def test_refraction_into_faster_medium(self):
        """Ray bends away from normal in faster medium."""
        theta2, transmitted = snells_law(30.0, 3000, 5000)

        assert theta2 > 30.0  # Bends away
        assert transmitted

    def test_refraction_into_slower_medium(self):
        """Ray bends toward normal in slower medium."""
        theta2, transmitted = snells_law(30.0, 5000, 3000)

        assert theta2 < 30.0  # Bends toward
        assert transmitted

    def test_beyond_critical_angle(self):
        """Beyond critical angle = no transmission."""
        V1, V2 = 3000, 6000
        theta_c = critical_angle(V1, V2)  # 30 degrees

        # Just beyond critical
        theta2, transmitted = snells_law(theta_c + 1, V1, V2)

        assert not transmitted
        assert theta2 == 90.0

    def test_formula_correctness(self):
        """Verify Snell's law formula."""
        theta1 = 25.0
        V1, V2 = 4000, 5500

        theta2, _ = snells_law(theta1, V1, V2)

        # sin(theta1)/V1 = sin(theta2)/V2
        ratio1 = math.sin(math.radians(theta1)) / V1
        ratio2 = math.sin(math.radians(theta2)) / V2

        assert abs(ratio1 - ratio2) < 1e-10


class TestVpVsBounds:
    """Tests for Vp/Vs ratio bounds reference."""

    def test_bounds_present(self):
        """Check that key bounds are present."""
        bounds = vp_vs_ratio_bounds()

        assert bounds["theoretical_minimum"] == math.sqrt(2)
        assert bounds["poisson_solid"] == math.sqrt(3)
        assert bounds["liquid_limit"] == float('inf')

    def test_common_values_present(self):
        """Check common rock types are included."""
        bounds = vp_vs_ratio_bounds()
        common = bounds["common_values"]

        assert "granite" in common
        assert "sandstone_dry" in common
        assert "shale" in common
        assert "water" in common


class TestReportStrings:
    """Test that all report __str__ methods work."""

    def test_seismic_velocity_report_str(self):
        """Test SeismicVelocityReport string formatting."""
        result = calc_seismic_velocity(50e9, 30e9, 2700)
        s = str(result)

        assert "Seismic Velocity Report" in s
        assert "P-wave" in s
        assert "S-wave" in s
        assert "Poisson" in s

    def test_poisson_ratio_report_str(self):
        """Test PoissonRatioReport string formatting."""
        result = poisson_from_velocities(6000, 3500)
        s = str(result)

        assert "Poisson Ratio Analysis" in s
        assert "Vp/Vs" in s

    def test_elastic_moduli_report_str(self):
        """Test ElasticModuliReport string formatting."""
        result = convert_elastic_moduli(K=50e9, G=30e9)
        s = str(result)

        assert "Elastic Moduli Conversion" in s
        assert "Bulk modulus" in s
        assert "Shear modulus" in s

    def test_reflection_report_str(self):
        """Test ReflectionReport string formatting."""
        result = calc_reflection_coefficient(2500, 4000, 2700, 5000)
        s = str(result)

        assert "Reflection Coefficient Report" in s
        assert "reflection" in s.lower()
        assert "transmission" in s.lower()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_shear_modulus(self):
        """Very small G (almost liquid) should work."""
        K = 100e9
        G = 1e6  # Very small but nonzero
        rho = 2500

        result = calc_seismic_velocity(K, G, rho)

        assert result.Vs > 0
        assert result.poisson_ratio < 0.5
        assert result.poisson_ratio > 0.45  # Near liquid

    def test_equal_k_and_g(self):
        """K = G should give specific Poisson ratio."""
        K = G = 60e9
        rho = 2800

        result = calc_seismic_velocity(K, G, rho)

        # nu = (3K - 2G) / (6K + 2G) = (3-2)K / (6+2)K = 1/8 = 0.125
        assert abs(result.poisson_ratio - 0.125) < 1e-6

    def test_very_high_density(self):
        """High density should give low velocities."""
        K, G = 50e9, 30e9
        rho_normal = 2700
        rho_high = 10000

        result_normal = calc_seismic_velocity(K, G, rho_normal)
        result_high = calc_seismic_velocity(K, G, rho_high)

        assert result_high.Vp < result_normal.Vp
        assert result_high.Vs < result_normal.Vs

    def test_moduli_conversion_consistency(self):
        """All conversion paths should give same result."""
        # Start with K, G
        K, G = 100e9, 60e9

        result1 = convert_elastic_moduli(K=K, G=G)

        # Use the computed E and nu
        E = result1.youngs_modulus_E
        nu = result1.poisson_ratio_nu

        result2 = convert_elastic_moduli(E=E, nu=nu)

        # Should get same K and G back
        assert abs(result2.bulk_modulus_K - K) / K < 1e-6
        assert abs(result2.shear_modulus_G - G) / G < 1e-6
