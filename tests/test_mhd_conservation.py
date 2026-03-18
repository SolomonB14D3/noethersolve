"""Tests for noethersolve.mhd_conservation module.

Tests cover:
- Magnetic helicity conservation
- Cross helicity conservation
- MHD energy conservation
- Frozen-in flux theorem
- Div B constraint
- General invariant checks
"""

import pytest

from noethersolve.mhd_conservation import (
    check_magnetic_helicity,
    check_cross_helicity,
    check_mhd_energy,
    check_frozen_flux,
    check_div_B,
    check_mhd_invariant,
    list_mhd_invariants,
)


# ─── Magnetic Helicity Tests ──────────────────────────────────────────────

class TestMagneticHelicity:
    """Tests for magnetic helicity conservation."""

    def test_ideal_conserved(self):
        """Magnetic helicity conserved in ideal MHD."""
        r = check_magnetic_helicity(resistivity=0.0)
        assert r.is_conserved is True
        assert r.regime == "ideal"
        assert r.decay_rate is None

    def test_resistive_not_conserved(self):
        """Magnetic helicity decays in resistive MHD."""
        r = check_magnetic_helicity(resistivity=1e-3)
        assert r.is_conserved is False
        assert r.regime == "resistive"
        assert r.decay_rate is not None

    def test_hall_conserved(self):
        """Magnetic helicity conserved in Hall MHD."""
        r = check_magnetic_helicity(resistivity=0.0, regime="hall")
        assert r.is_conserved is True
        assert r.regime == "hall"

    def test_topological_meaning(self):
        """Topological meaning is explained."""
        r = check_magnetic_helicity(resistivity=0.0)
        assert "linkage" in r.topological_meaning.lower()


class TestCrossHelicity:
    """Tests for cross helicity conservation."""

    def test_ideal_incompressible_conserved(self):
        """Cross helicity conserved in ideal incompressible MHD."""
        r = check_cross_helicity(resistivity=0.0, viscosity=0.0, compressible=False)
        assert r.is_conserved is True
        assert "ideal" in r.regime.lower()

    def test_resistive_not_conserved(self):
        """Cross helicity broken by resistivity."""
        r = check_cross_helicity(resistivity=1e-3, viscosity=0.0, compressible=False)
        assert r.is_conserved is False

    def test_viscous_not_conserved(self):
        """Cross helicity broken by viscosity."""
        r = check_cross_helicity(resistivity=0.0, viscosity=1e-3, compressible=False)
        assert r.is_conserved is False

    def test_compressible_not_conserved(self):
        """Cross helicity broken by compressibility."""
        r = check_cross_helicity(resistivity=0.0, viscosity=0.0, compressible=True)
        assert r.is_conserved is False

    def test_multiple_breaking(self):
        """Multiple breaking mechanisms listed."""
        r = check_cross_helicity(resistivity=1e-3, viscosity=1e-3, compressible=True)
        assert r.is_conserved is False
        assert "dissipative" in r.regime.lower()


# ─── MHD Energy Tests ─────────────────────────────────────────────────────

class TestMHDEnergy:
    """Tests for MHD energy conservation."""

    def test_ideal_conserved(self):
        """Energy conserved in ideal MHD."""
        r = check_mhd_energy(
            B_rms=1e-3,
            v_rms=1000.0,
            density=1e-12,
            resistivity=0.0,
            viscosity=0.0,
        )
        assert r.is_conserved is True
        assert r.regime == "ideal"

    def test_energy_components(self):
        """Energy components calculated correctly."""
        r = check_mhd_energy(
            B_rms=1e-3,
            v_rms=1000.0,
            density=1e-12,
            volume=1.0,
        )
        # Magnetic energy ~ B²/(2μ₀)
        assert r.magnetic_energy > 0
        # Kinetic energy ~ ρv²/2
        assert r.kinetic_energy > 0
        assert r.total_energy == pytest.approx(r.magnetic_energy + r.kinetic_energy)

    def test_ohmic_dissipation(self):
        """Ohmic dissipation detected."""
        r = check_mhd_energy(
            B_rms=1e-3,
            v_rms=1000.0,
            density=1e-12,
            resistivity=1e-3,
            J_rms=1e6,
        )
        assert r.is_conserved is False
        assert "Ohmic" in str(r.dissipation_sources)

    def test_viscous_dissipation(self):
        """Viscous dissipation detected."""
        r = check_mhd_energy(
            B_rms=1e-3,
            v_rms=1000.0,
            density=1e-12,
            viscosity=1e-3,
            omega_rms=1e3,
        )
        assert r.is_conserved is False
        assert "Viscous" in str(r.dissipation_sources)


# ─── Frozen Flux Tests ────────────────────────────────────────────────────

class TestFrozenFlux:
    """Tests for frozen-in flux theorem."""

    def test_ideal_frozen(self):
        """Flux frozen in ideal MHD."""
        r = check_frozen_flux(resistivity=0.0)
        assert r.is_frozen is True
        assert r.regime == "ideal"
        assert r.reconnection_possible is False

    def test_high_rm_approximately_frozen(self):
        """Flux approximately frozen for high Rm."""
        r = check_frozen_flux(resistivity=1e-6, length_scale=1.0, velocity=1000.0)
        # Rm = 1000 / 1e-6 = 1e9 >> 100
        assert r.is_frozen is True
        assert r.magnetic_reynolds > 100
        assert r.reconnection_possible is True  # But possible at small scales

    def test_low_rm_not_frozen(self):
        """Flux not frozen for low Rm."""
        r = check_frozen_flux(resistivity=1.0, length_scale=1.0, velocity=0.5)
        # Rm = 0.5 / 1.0 = 0.5 < 1
        assert r.is_frozen is False
        assert r.magnetic_reynolds < 1
        assert r.reconnection_possible is True

    def test_diffusion_time(self):
        """Diffusion time calculated correctly."""
        r = check_frozen_flux(resistivity=1e-4, length_scale=1.0, velocity=1.0)
        # τ = L²/η = 1/1e-4 = 1e4
        assert r.diffusion_time == pytest.approx(1e4, rel=1e-6)


# ─── Div B Constraint Tests ───────────────────────────────────────────────

class TestDivB:
    """Tests for ∇·B = 0 constraint."""

    def test_satisfied(self):
        """Constraint satisfied for small divergence."""
        r = check_div_B(max_div_B=1e-12, B_scale=1e-3, dx=0.01)
        assert r.is_satisfied is True

    def test_violated(self):
        """Constraint violated for large divergence."""
        r = check_div_B(max_div_B=1e-3, B_scale=1e-3, dx=0.01, tolerance_factor=1e-8)
        assert r.is_satisfied is False

    def test_cleaning_suggested(self):
        """Cleaning method suggested for violated constraint."""
        r = check_div_B(max_div_B=1e-3, B_scale=1e-3, dx=0.01)
        assert r.cleaning_method != "none needed"

    def test_projection_for_small_error(self):
        """Projection cleaning for small errors."""
        r = check_div_B(max_div_B=1e-8, B_scale=1e-3, dx=0.01, tolerance_factor=1e-10)
        if not r.is_satisfied:
            assert "projection" in r.cleaning_method.lower()


# ─── General Invariant Tests ──────────────────────────────────────────────

class TestMHDInvariant:
    """Tests for general MHD invariant checking."""

    def test_mass_always_conserved(self):
        """Mass is always conserved."""
        r = check_mhd_invariant("mass", resistivity=1.0, viscosity=1.0)
        assert r.is_conserved is True
        assert r.regime == "all"

    def test_momentum_conserved(self):
        """Momentum conserved without external forces."""
        r = check_mhd_invariant("momentum")
        assert r.is_conserved is True

    def test_magnetic_flux_ideal(self):
        """Magnetic flux conserved in ideal MHD."""
        r = check_mhd_invariant("magnetic_flux", resistivity=0.0)
        assert r.is_conserved is True

    def test_magnetic_flux_resistive(self):
        """Magnetic flux not conserved in resistive MHD."""
        r = check_mhd_invariant("magnetic_flux", resistivity=1e-3)
        assert r.is_conserved is False

    def test_energy_ideal(self):
        """Energy conserved in ideal MHD."""
        r = check_mhd_invariant("energy", resistivity=0.0, viscosity=0.0)
        assert r.is_conserved is True

    def test_energy_dissipative(self):
        """Energy not conserved with dissipation."""
        r = check_mhd_invariant("energy", resistivity=1e-3, viscosity=1e-3)
        assert r.is_conserved is False

    def test_unknown_invariant_error(self):
        """Unknown invariant raises error."""
        with pytest.raises(ValueError):
            check_mhd_invariant("invalid_invariant")


class TestListInvariants:
    """Tests for list_mhd_invariants."""

    def test_all_listed(self):
        """All known invariants are listed."""
        invs = list_mhd_invariants()
        assert "mass" in invs
        assert "momentum" in invs
        assert "magnetic_helicity" in invs
        assert "cross_helicity" in invs
        assert "energy" in invs
        assert len(invs) >= 6


# ─── Report String Tests ──────────────────────────────────────────────────

class TestReportStrings:
    """Tests for report __str__ methods."""

    def test_helicity_report_str(self):
        """Helicity report is readable."""
        r = check_magnetic_helicity(resistivity=0.0)
        s = str(r)
        assert "Helicity" in s
        assert "CONSERVED" in s

    def test_energy_report_str(self):
        """Energy report is readable."""
        r = check_mhd_energy(B_rms=1e-3, v_rms=1000, density=1e-12)
        s = str(r)
        assert "Energy" in s
        assert "Magnetic" in s

    def test_frozen_flux_report_str(self):
        """Frozen flux report is readable."""
        r = check_frozen_flux(resistivity=0.0)
        s = str(r)
        assert "Frozen" in s
        assert "ideal" in s.lower()

    def test_div_b_report_str(self):
        """Div B report is readable."""
        r = check_div_B(max_div_B=1e-10, B_scale=1e-3, dx=0.01)
        s = str(r)
        assert "∇·B" in s or "div" in s.lower()

    def test_invariant_report_str(self):
        """Invariant report is readable."""
        r = check_mhd_invariant("mass")
        s = str(r)
        assert "Mass" in s
        assert "CONSERVED" in s


# ─── Physical Consistency Tests ───────────────────────────────────────────

class TestPhysicalConsistency:
    """Tests for physical consistency."""

    def test_ideal_all_conserved(self):
        """All ideal MHD invariants conserved."""
        for inv in ["magnetic_flux", "magnetic_helicity", "energy"]:
            r = check_mhd_invariant(inv, resistivity=0.0, viscosity=0.0)
            assert r.is_conserved is True

    def test_resistive_breaks_flux_invariants(self):
        """Resistivity breaks flux-related invariants."""
        for inv in ["magnetic_flux", "magnetic_helicity"]:
            r = check_mhd_invariant(inv, resistivity=1e-3)
            assert r.is_conserved is False

    def test_reconnection_requires_resistivity(self):
        """Reconnection requires resistivity."""
        r_ideal = check_frozen_flux(resistivity=0.0)
        r_resist = check_frozen_flux(resistivity=1e-3, length_scale=1.0, velocity=1.0)
        assert r_ideal.reconnection_possible is False
        assert r_resist.reconnection_possible is True

    def test_mass_always_conserved(self):
        """Mass conserved regardless of dissipation."""
        for eta in [0.0, 1e-3, 1.0]:
            for nu in [0.0, 1e-3, 1.0]:
                r = check_mhd_invariant("mass", resistivity=eta, viscosity=nu)
                assert r.is_conserved is True


# ─── Edge Cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_resistivity(self):
        """Zero resistivity correctly handled."""
        r = check_magnetic_helicity(resistivity=0.0)
        assert r.is_conserved is True

    def test_very_small_resistivity(self):
        """Very small resistivity still breaks conservation."""
        r = check_magnetic_helicity(resistivity=1e-20)
        assert r.is_conserved is False
        assert r.regime == "resistive"

    def test_high_magnetic_reynolds(self):
        """Very high Rm still allows reconnection."""
        r = check_frozen_flux(resistivity=1e-12, length_scale=1.0, velocity=1e6)
        assert r.magnetic_reynolds > 1e15
        assert r.is_frozen is True
        assert r.reconnection_possible is True  # At small scales

    def test_negative_resistivity_error(self):
        """Negative resistivity raises error."""
        with pytest.raises(ValueError):
            check_magnetic_helicity(resistivity=-1e-3)
