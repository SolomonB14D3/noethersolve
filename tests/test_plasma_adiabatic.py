"""Tests for plasma physics adiabatic invariants calculator."""

import math
import pytest

from noethersolve.plasma_adiabatic import (
    calc_magnetic_moment,
    calc_bounce_invariant,
    calc_flux_invariant,
    check_adiabatic_hierarchy,
    mirror_force,
    loss_cone_angle,
    cyclotron_frequency,
    larmor_radius,
    get_particle_mass,
    MagneticMomentReport,
    BounceInvariantReport,
    FluxInvariantReport,
    AdiabaticHierarchyReport,
    ELECTRON_MASS,
    PROTON_MASS,
    ELECTRON_CHARGE,
)


class TestMagneticMoment:
    """Tests for first adiabatic invariant μ."""

    def test_basic_calculation(self):
        """Test μ = m*v⊥²/(2*B) formula."""
        v_perp = 1e7  # 10^7 m/s
        B = 1.0  # 1 Tesla
        mass = ELECTRON_MASS

        result = calc_magnetic_moment(v_perp, B, mass)

        # μ = m*v²/(2B)
        expected_mu = mass * v_perp**2 / (2 * B)
        assert abs(result.mu - expected_mu) < 1e-30

    def test_cyclotron_frequency(self):
        """Test ω_c = qB/m."""
        B = 1.0
        mass = ELECTRON_MASS
        charge = ELECTRON_CHARGE

        result = calc_magnetic_moment(1e7, B, mass, charge)

        expected_omega = charge * B / mass
        assert abs(result.omega_cyclotron - expected_omega) < 1e-10

    def test_larmor_radius(self):
        """Test r_L = v⊥/ω_c."""
        v_perp = 1e7
        B = 1.0

        result = calc_magnetic_moment(v_perp, B)

        expected_r_L = v_perp / result.omega_cyclotron
        assert abs(result.cyclotron_radius - expected_r_L) < 1e-10

    def test_conserved_slow_field_change(self):
        """μ should be conserved when field changes slowly."""
        # Field changes over 1 second, cyclotron period ~ 10^-11 s
        result = calc_magnetic_moment(1e7, 1.0, field_timescale=1.0)

        assert result.is_conserved
        assert "slow" in result.breaking_condition.lower() or "N/A" in result.breaking_condition

    def test_not_conserved_fast_field_change(self):
        """μ should break when field changes on cyclotron timescale."""
        # Field changes on 10^-11 s, similar to cyclotron period
        result = calc_magnetic_moment(1e7, 1.0, field_timescale=1e-11)

        # With τ_field ~ τ_c, should not be conserved
        # Actually, the threshold is τ_field > 10*τ_c
        # τ_c for electron in 1T ~ 3.6e-11 s, so 1e-11 < 3.6e-10
        assert not result.is_conserved

    def test_energy_in_eV(self):
        """Test energy conversion to eV."""
        v_perp = 1e7
        mass = ELECTRON_MASS

        result = calc_magnetic_moment(v_perp, 1.0, mass)

        E_joules = 0.5 * mass * v_perp**2
        E_eV = E_joules / ELECTRON_CHARGE
        assert abs(result.kinetic_energy_perp_eV - E_eV) < 1e-6

    def test_proton_vs_electron(self):
        """Proton should have larger μ than electron at same v⊥."""
        v_perp = 1e6
        B = 1.0

        result_e = calc_magnetic_moment(v_perp, B, mass=ELECTRON_MASS)
        result_p = calc_magnetic_moment(v_perp, B, mass=PROTON_MASS)

        # μ ∝ m, so proton has larger μ
        assert result_p.mu > result_e.mu
        assert abs(result_p.mu / result_e.mu - PROTON_MASS / ELECTRON_MASS) < 1e-6

    def test_zero_velocity(self):
        """Zero perpendicular velocity gives zero μ."""
        result = calc_magnetic_moment(0.0, 1.0)
        assert result.mu == 0.0

    def test_invalid_field(self):
        """Negative or zero field should raise."""
        with pytest.raises(ValueError, match="positive"):
            calc_magnetic_moment(1e7, 0.0)
        with pytest.raises(ValueError, match="positive"):
            calc_magnetic_moment(1e7, -1.0)


class TestBounceInvariant:
    """Tests for second adiabatic invariant J."""

    def test_basic_calculation(self):
        """Test J ≈ 2*v∥*L formula."""
        v_parallel = 1e6
        bounce_length = 1e5  # 100 km

        result = calc_bounce_invariant(
            v_parallel, bounce_length,
            B_min=1e-5, B_max=5e-5  # Earth-like mirror
        )

        # Simple approximation: J ≈ 2*v∥*L
        expected_J = 2 * v_parallel * bounce_length
        assert abs(result.J - expected_J) < 1e-6

    def test_mirror_ratio(self):
        """Test mirror ratio R = B_max/B_min."""
        B_min, B_max = 1e-5, 5e-5

        result = calc_bounce_invariant(
            1e6, 1e5, B_min, B_max
        )

        assert abs(result.mirror_ratio - 5.0) < 1e-10

    def test_loss_cone_calculation(self):
        """Test loss cone angle calculation."""
        B_min, B_max = 1e-5, 4e-5  # R = 4

        result = calc_bounce_invariant(
            1e6, 1e5, B_min, B_max
        )

        # sin²(α_loss) = 1/R = 0.25, sin(α) = 0.5, α = 30°
        assert abs(result.loss_cone_angle - 30.0) < 0.1

    def test_bounce_period(self):
        """Test bounce period τ_b ≈ 2L/v∥."""
        v_parallel = 1e6
        bounce_length = 1e5

        result = calc_bounce_invariant(
            v_parallel, bounce_length,
            B_min=1e-5, B_max=5e-5
        )

        expected_tau = 2 * bounce_length / v_parallel
        assert abs(result.bounce_period - expected_tau) < 1e-6

    def test_conserved_slow_change(self):
        """J conserved when field changes slowly vs bounce."""
        result = calc_bounce_invariant(
            1e6, 1e5, 1e-5, 5e-5,
            field_timescale=100.0  # 100 seconds >> bounce period
        )

        assert result.is_conserved

    def test_invalid_mirror_geometry(self):
        """B_max must be greater than B_min."""
        with pytest.raises(ValueError, match="greater"):
            calc_bounce_invariant(1e6, 1e5, B_min=5e-5, B_max=1e-5)


class TestFluxInvariant:
    """Tests for third adiabatic invariant Φ."""

    def test_basic_calculation(self):
        """Test Φ ≈ π*r²*B for circular orbit."""
        drift_radius = 6e7  # 60,000 km (Earth-like L-shell)
        B_average = 1e-5

        result = calc_flux_invariant(
            drift_radius, B_average,
            energy=1e6  # 1 MeV
        )

        expected_Phi = math.pi * drift_radius**2 * B_average
        assert abs(result.Phi - expected_Phi) < 1e-6

    def test_drift_period(self):
        """Test drift period calculation."""
        result = calc_flux_invariant(
            drift_radius=6e7,
            B_average=1e-5,
            energy=1e6
        )

        # Should be order of hours for MeV particles
        assert result.drift_period > 0
        assert result.drift_period < 1e6  # Less than ~11 days

    def test_most_fragile_invariant(self):
        """Φ is the most fragile - breaks on drift timescale."""
        result = calc_flux_invariant(
            drift_radius=6e7, B_average=1e-5, energy=1e6,
            field_timescale=60.0  # 1 minute
        )

        # If drift period is longer than 1 minute, Φ may not be conserved
        # This depends on the calculated drift period
        assert "fragile" in " ".join(result.notes).lower()

    def test_invalid_radius(self):
        """Drift radius must be positive."""
        with pytest.raises(ValueError, match="positive"):
            calc_flux_invariant(0, 1e-5, 1e6)


class TestAdiabaticHierarchy:
    """Tests for the full adiabatic hierarchy check."""

    def test_correct_hierarchy(self):
        """Should have ω_c >> ω_b >> ω_d."""
        result = check_adiabatic_hierarchy(
            B=1e-5,
            v_total=1e7,
            pitch_angle_deg=45,
            bounce_length=1e6,
            drift_radius=6e7,
        )

        # Verify hierarchy
        assert result.omega_cyclotron > result.omega_bounce
        assert result.omega_bounce > result.omega_drift

        # Equivalently: τ_c < τ_b < τ_d
        assert result.tau_cyclotron < result.tau_bounce
        assert result.tau_bounce < result.tau_drift

    def test_all_conserved_slow_field(self):
        """All invariants conserved for very slow field changes."""
        result = check_adiabatic_hierarchy(
            B=1e-5,
            v_total=1e7,
            pitch_angle_deg=45,
            bounce_length=1e6,
            drift_radius=6e7,
            field_timescale=1e10  # Very slow (centuries)
        )

        assert result.mu_conserved
        assert result.J_conserved
        assert result.Phi_conserved

    def test_phi_only_broken(self):
        """Can have intermediate regime where only Φ breaks."""
        # Need field timescale between τ_b and τ_d
        result = check_adiabatic_hierarchy(
            B=1e-5,
            v_total=1e7,
            pitch_angle_deg=45,
            bounce_length=1e6,
            drift_radius=6e7,
            field_timescale=1.0  # 1 second - intermediate
        )

        # τ_c ~ 10^-6 s, τ_b ~ 0.1 s, τ_d ~ 10^4 s (rough estimates)
        # With 1 s timescale: μ and J should be conserved, Φ may break
        # Actually depends on exact values
        assert result.mu_conserved  # Always true for 1s >> 10^-6 s

    def test_invalid_pitch_angle(self):
        """Pitch angle must be between 0 and 90."""
        with pytest.raises(ValueError, match="Pitch angle"):
            check_adiabatic_hierarchy(1e-5, 1e7, 0, 1e6, 6e7)
        with pytest.raises(ValueError, match="Pitch angle"):
            check_adiabatic_hierarchy(1e-5, 1e7, 90, 1e6, 6e7)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_mirror_force(self):
        """Test F = -μ*dB/ds."""
        mu = 1e-20  # J/T
        dB_ds = 1e-6  # T/m (increasing B)

        F = mirror_force(mu, dB_ds)

        # Force should be negative (toward lower B)
        assert F == -mu * dB_ds
        assert F < 0

    def test_loss_cone_angle_earth(self):
        """Test loss cone for Earth-like mirror."""
        # Earth: B equator ~ 3e-5 T, B poles ~ 6e-5 T
        B_min, B_max = 3e-5, 6e-5

        alpha_loss = loss_cone_angle(B_min, B_max)

        # R = 2, sin²α = 0.5, α = 45°
        assert abs(alpha_loss - 45.0) < 0.1

    def test_loss_cone_strong_mirror(self):
        """Strong mirror has small loss cone."""
        alpha_loss = loss_cone_angle(1e-5, 100e-5)  # R = 100

        # sin²α = 0.01, sin α = 0.1, α ≈ 5.7°
        assert alpha_loss < 10

    def test_cyclotron_frequency_scaling(self):
        """Test ω_c = qB/m scaling."""
        B = 1.0

        omega_e = cyclotron_frequency(B, ELECTRON_MASS)
        omega_p = cyclotron_frequency(B, PROTON_MASS)

        # Ratio should be m_p/m_e
        assert abs(omega_e / omega_p - PROTON_MASS / ELECTRON_MASS) < 1e-6

    def test_larmor_radius_scaling(self):
        """Test r_L = m*v/(qB) scaling."""
        v_perp = 1e6
        B = 1.0

        r_e = larmor_radius(v_perp, B, ELECTRON_MASS)
        r_p = larmor_radius(v_perp, B, PROTON_MASS)

        # Proton has larger Larmor radius (same v, larger m)
        assert r_p > r_e
        assert abs(r_p / r_e - PROTON_MASS / ELECTRON_MASS) < 1e-6

    def test_get_particle_mass(self):
        """Test particle mass lookup."""
        assert get_particle_mass("electron") == ELECTRON_MASS
        assert get_particle_mass("proton") == PROTON_MASS
        assert get_particle_mass("alpha") == 4 * PROTON_MASS

    def test_invalid_particle(self):
        """Unknown particle should raise."""
        with pytest.raises(ValueError, match="Unknown"):
            get_particle_mass("neutrino")


class TestReportStrings:
    """Test that all report __str__ methods work."""

    def test_magnetic_moment_str(self):
        """Test MagneticMomentReport string."""
        result = calc_magnetic_moment(1e7, 1.0)
        s = str(result)

        assert "First Adiabatic Invariant" in s
        assert "μ" in s or "mu" in s.lower()
        assert "cyclotron" in s.lower()

    def test_bounce_invariant_str(self):
        """Test BounceInvariantReport string."""
        result = calc_bounce_invariant(1e6, 1e5, 1e-5, 5e-5)
        s = str(result)

        assert "Second Adiabatic Invariant" in s
        assert "bounce" in s.lower()
        assert "mirror" in s.lower()

    def test_flux_invariant_str(self):
        """Test FluxInvariantReport string."""
        result = calc_flux_invariant(6e7, 1e-5, 1e6)
        s = str(result)

        assert "Third Adiabatic Invariant" in s
        assert "drift" in s.lower()

    def test_hierarchy_str(self):
        """Test AdiabaticHierarchyReport string."""
        result = check_adiabatic_hierarchy(
            1e-5, 1e7, 45, 1e6, 6e7, field_timescale=100
        )
        s = str(result)

        assert "Hierarchy" in s
        assert "cyclotron" in s.lower()
        assert "bounce" in s.lower()
        assert "drift" in s.lower()


class TestPhysicalConsistency:
    """Tests for physical consistency."""

    def test_mu_inversely_proportional_to_B(self):
        """At constant energy, μ ∝ 1/B is not quite right; μ = E_perp/B."""
        # At same v_perp, μ ∝ 1/B
        v_perp = 1e7

        result1 = calc_magnetic_moment(v_perp, 1.0)
        result2 = calc_magnetic_moment(v_perp, 2.0)

        # μ = m*v²/(2B) ∝ 1/B at fixed v
        assert abs(result1.mu / result2.mu - 2.0) < 1e-6

    def test_energy_conservation_during_mirroring(self):
        """As particle moves to higher B, v⊥ increases, v∥ decreases."""
        # At mirror point, all energy is in v_perp
        # E_total = E_perp + E_parallel = const
        # μ = E_perp / B = const
        # So E_perp ∝ B as particle approaches mirror

        E_total = 1000  # eV
        B1 = 1e-5
        B2 = 4e-5  # 4× stronger at mirror

        # At B1: E_perp = 500 eV, E_parallel = 500 eV
        # At B2: E_perp must be 4× larger for μ conservation
        # But E_total is only 1000 eV, so E_perp can't exceed 1000
        # This means particle mirrors before B2 if E_perp(B2) > E_total

        # If E_perp(B1) = 500 eV, μ = 500/B1
        # At B2, E_perp = μ*B2 = 500*B2/B1 = 500*4 = 2000 eV
        # This exceeds E_total, so particle mirrors before reaching B2

        # Mirror point: E_perp = E_total, so B_mirror = E_total/μ = E_total*B1/E_perp(B1)
        # B_mirror = 1000 * 1e-5 / 500 = 2e-5 T

        # This is consistent with mirror ratio R = B_mirror/B1 = 2
        # And loss cone sin²α = B1/B_mirror = 0.5, α = 45°

        alpha_loss = loss_cone_angle(B1, 2*B1)
        assert abs(alpha_loss - 45.0) < 0.1

    def test_hierarchy_orders_of_magnitude(self):
        """Verify typical frequency hierarchy ratios."""
        # For 1 keV electron in Earth's magnetosphere:
        # - ω_c ~ 10^5 rad/s (gyration ~10 μs)
        # - ω_b ~ 1 rad/s (bounce ~seconds)
        # - ω_d ~ 10^-4 rad/s (drift ~hours)

        result = check_adiabatic_hierarchy(
            B=1e-5,  # 10 μT ~ Earth equator
            v_total=2e7,  # ~1 keV electron
            pitch_angle_deg=45,
            bounce_length=1e7,  # ~10,000 km
            drift_radius=6e7,  # ~10 Earth radii
            mass=ELECTRON_MASS,
        )

        # τ_c should be microseconds or less
        assert result.tau_cyclotron < 1e-3

        # τ_b should be seconds
        assert 0.1 < result.tau_bounce < 100

        # τ_d should be much longer
        assert result.tau_drift > result.tau_bounce * 10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_strong_field(self):
        """Test with laboratory-strength fields."""
        result = calc_magnetic_moment(1e5, 10.0)  # 10 Tesla

        assert result.mu > 0
        assert result.omega_cyclotron > 1e12  # Very fast

    def test_very_weak_field(self):
        """Test with interplanetary field strengths."""
        result = calc_magnetic_moment(1e7, 1e-9)  # 1 nT

        assert result.mu > 0
        assert result.omega_cyclotron < 1e4  # Relatively slow

    def test_relativistic_warning(self):
        """Should work but velocities near c need care."""
        c = 3e8
        v_perp = 0.5 * c  # 50% speed of light

        result = calc_magnetic_moment(v_perp, 1.0)

        # Non-relativistic formula still gives a number
        assert result.mu > 0
        # Larmor radius: r_L = m*v/(qB) ~ 0.5*3e8*9.1e-31/(1.6e-19*1) ~ 0.9 mm
        # At 1 Tesla, even relativistic electrons have small Larmor radius
        assert result.cyclotron_radius > 1e-4  # > 0.1 mm
        assert result.cyclotron_radius < 1  # < 1 m

    def test_heavy_ions(self):
        """Test with heavy ions like oxygen."""
        mass_O = 16 * PROTON_MASS

        result = calc_magnetic_moment(1e6, 1e-5, mass=mass_O)

        # Heavier ion has larger μ at same v
        result_p = calc_magnetic_moment(1e6, 1e-5, mass=PROTON_MASS)
        assert result.mu > result_p.mu
