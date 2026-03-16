#!/usr/bin/env python3
"""
Tests for physics analyzer module.
"""

import pytest
from noethersolve.physics_analyzer import (
    ConservationLaw, PhysicsDiscovery, Domain,
    list_conservation_laws, list_discoveries,
    analyze_symmetry, CONSERVATION_LAWS, DISCOVERIES
)


class TestConservationLaws:
    """Test conservation law database."""

    def test_conservation_laws_exist(self):
        """Should have conservation laws in database."""
        assert len(CONSERVATION_LAWS) > 0

    def test_all_domains_represented(self):
        """Multiple domains should be represented."""
        domains = set(law.domain for law in CONSERVATION_LAWS)
        assert len(domains) >= 3

    def test_filter_by_domain(self):
        """Should be able to filter by domain."""
        em_laws = list_conservation_laws(Domain.ELECTROMAGNETISM)
        assert all(law.domain == Domain.ELECTROMAGNETISM for law in em_laws)

    def test_energy_conservation_present(self):
        """Energy conservation should be in database."""
        names = [law.name for law in CONSERVATION_LAWS]
        assert "Energy Conservation" in names

    def test_charge_conservation_present(self):
        """Charge conservation should be in database."""
        names = [law.name for law in CONSERVATION_LAWS]
        assert "Electric Charge Conservation" in names

    def test_zilch_conservation_present(self):
        """Optical chirality (Zilch) should be in database."""
        names = [law.name for law in CONSERVATION_LAWS]
        assert any("Chirality" in name or "Zilch" in name for name in names)


class TestDiscoveries:
    """Test NoetherSolve discoveries."""

    def test_discoveries_exist(self):
        """Should have discoveries in database."""
        assert len(DISCOVERIES) > 0

    def test_discoveries_have_key_results(self):
        """All discoveries should have key results."""
        for discovery in DISCOVERIES:
            assert discovery.key_result != ""

    def test_discoveries_have_implications(self):
        """Discoveries should have implications."""
        for discovery in DISCOVERIES:
            assert len(discovery.implications) > 0

    def test_filter_by_domain(self):
        """Should be able to filter discoveries by domain."""
        fluid_discoveries = list_discoveries(Domain.FLUID_DYNAMICS)
        assert all(d.domain == Domain.FLUID_DYNAMICS for d in fluid_discoveries)

    def test_qf_kinetic_energy_discovery(self):
        """Q_{-ln(r)} = kinetic energy should be a discovery."""
        titles = [d.title for d in DISCOVERIES]
        assert any("kinetic" in t.lower() or "ln(r)" in t for t in titles)


class TestSymmetryAnalysis:
    """Test symmetry -> conservation law analysis."""

    def test_time_translation_gives_energy(self):
        """Time translation should give energy conservation."""
        result = analyze_symmetry("Time translation invariance")
        assert result["conserved_quantity"] == "Energy"

    def test_space_translation_gives_momentum(self):
        """Space translation should give momentum conservation."""
        result = analyze_symmetry("Space translation invariance")
        assert result["conserved_quantity"] == "Momentum"

    def test_rotation_gives_angular_momentum(self):
        """Rotation should give angular momentum conservation."""
        result = analyze_symmetry("Rotational symmetry")
        assert result["conserved_quantity"] == "Angular Momentum"

    def test_gauge_gives_charge(self):
        """U(1) gauge should give charge conservation."""
        result = analyze_symmetry("U(1) gauge symmetry")
        assert "Charge" in result["conserved_quantity"]

    def test_particle_relabeling_gives_circulation(self):
        """Particle relabeling should give circulation."""
        result = analyze_symmetry("Particle relabeling symmetry")
        assert result["conserved_quantity"] == "Circulation"

    def test_unknown_symmetry_handled(self):
        """Unknown symmetries should be handled gracefully."""
        result = analyze_symmetry("Some unknown symmetry xyz")
        assert result["conserved_quantity"] == "Unknown"


class TestDataclasses:
    """Test dataclass structure."""

    def test_conservation_law_structure(self):
        """ConservationLaw should have required fields."""
        law = ConservationLaw(
            name="Test",
            expression="dX/dt = 0",
            symmetry="Test symmetry",
            domain=Domain.MECHANICS,
            discovered_by="Test",
            year=2024,
            verified=True
        )
        assert law.name == "Test"
        assert law.year == 2024

    def test_discovery_structure(self):
        """PhysicsDiscovery should have required fields."""
        discovery = PhysicsDiscovery(
            title="Test Discovery",
            domain=Domain.FLUID_DYNAMICS,
            description="Test description",
            key_result="Test result",
            implications=["Implication 1"],
            reference_files=["test.py"]
        )
        assert discovery.title == "Test Discovery"
        assert len(discovery.implications) == 1


class TestNoetherPrinciple:
    """Test that Noether's theorem is properly encoded."""

    def test_all_laws_have_symmetry(self):
        """All conservation laws should reference a symmetry."""
        for law in CONSERVATION_LAWS:
            assert law.symmetry != ""
            assert len(law.symmetry) > 5  # Not just placeholder

    def test_all_laws_verified(self):
        """All laws in database should be marked as verified."""
        for law in CONSERVATION_LAWS:
            assert law.verified is True

    def test_symmetry_contains_keywords(self):
        """Symmetries should contain descriptive keywords."""
        keywords = ["translation", "rotation", "gauge", "invariance", "symmetry"]
        for law in CONSERVATION_LAWS:
            assert any(kw in law.symmetry.lower() for kw in keywords)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
