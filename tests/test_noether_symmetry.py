"""Tests for noether_symmetry module — bidirectional symmetry ↔ conservation mapping."""

import pytest
from noethersolve.noether_symmetry import (
    symmetry_to_conservation,
    conservation_to_symmetry,
    verify_noether_claim,
    list_all_pairs,
    get_pair,
    NOETHER_PAIRS,
    SymmetryType,
    ConservationType,
)


class TestSymmetryToConservation:
    """Tests for symmetry → conservation mapping."""

    def test_time_translation_to_energy(self):
        """Time translation should map to energy conservation."""
        report = symmetry_to_conservation("time translation")

        assert "energy" in report.pair.conservation_law.lower()
        assert report.direction == "symmetry → conservation"

    def test_spatial_translation_to_momentum(self):
        """Spatial translation should map to momentum conservation."""
        report = symmetry_to_conservation("spatial translation")

        assert "momentum" in report.pair.conservation_law.lower()
        assert "energy" not in report.pair.conservation_law.lower()

    def test_rotation_to_angular_momentum(self):
        """Rotation should map to angular momentum."""
        report = symmetry_to_conservation("rotation")

        assert "angular" in report.pair.conservation_law.lower()

    def test_gauge_to_charge(self):
        """U(1) gauge should map to charge conservation."""
        report = symmetry_to_conservation("U(1) gauge")

        assert "charge" in report.pair.conservation_law.lower()

    def test_partial_match(self):
        """Should work with partial/alternate names."""
        # "temporal" → time translation
        report = symmetry_to_conservation("temporal")
        assert "energy" in report.pair.conservation_law.lower()

        # "translational" → spatial translation
        report = symmetry_to_conservation("translational")
        assert "momentum" in report.pair.conservation_law.lower()

    def test_unknown_symmetry_raises(self):
        """Unknown symmetry should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown symmetry"):
            symmetry_to_conservation("completely_unknown_xyz")

    def test_model_error_warning(self):
        """Should warn about common model errors."""
        report = symmetry_to_conservation("spatial translation")

        # Should have model error warning about translation→energy confusion
        assert report.model_likely_error is not None
        assert "energy" in report.model_likely_error.lower()


class TestConservationToSymmetry:
    """Tests for conservation → symmetry mapping."""

    def test_energy_to_time_translation(self):
        """Energy conservation should map to time translation."""
        report = conservation_to_symmetry("energy")

        assert "time" in report.pair.symmetry.lower()

    def test_momentum_to_spatial_translation(self):
        """Momentum conservation should map to spatial translation."""
        report = conservation_to_symmetry("momentum")

        assert "spatial" in report.pair.symmetry.lower() or "translation" in report.pair.symmetry.lower()

    def test_angular_momentum_to_rotation(self):
        """Angular momentum should map to rotation."""
        report = conservation_to_symmetry("angular momentum")

        assert "rotation" in report.pair.symmetry.lower()

    def test_charge_to_gauge(self):
        """Charge conservation should map to gauge symmetry."""
        report = conservation_to_symmetry("electric charge")

        assert "gauge" in report.pair.symmetry.lower() or "u(1)" in report.pair.symmetry.lower()

    def test_unknown_conservation_raises(self):
        """Unknown conserved quantity should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown conserved quantity"):
            conservation_to_symmetry("completely_unknown_quantity")


class TestVerifyNoetherClaim:
    """Tests for claim verification."""

    def test_correct_claim(self):
        """Correct claims should be verified."""
        correct, _ = verify_noether_claim("time translation", "energy")
        assert correct

        correct, _ = verify_noether_claim("spatial translation", "momentum")
        assert correct

        correct, _ = verify_noether_claim("rotation", "angular momentum")
        assert correct

    def test_incorrect_claim(self):
        """Incorrect claims should be rejected."""
        correct, explanation = verify_noether_claim("spatial translation", "energy")
        assert not correct
        assert "momentum" in explanation.lower()  # Should suggest correct answer

        correct, explanation = verify_noether_claim("time translation", "momentum")
        assert not correct
        assert "energy" in explanation.lower()

    def test_explanation_quality(self):
        """Explanation should be informative."""
        _, explanation = verify_noether_claim("spatial translation", "energy")

        # Should mention the correct mapping
        assert "momentum" in explanation.lower()
        assert "incorrect" in explanation.lower() or "correct mapping" in explanation.lower()


class TestNoetherPairs:
    """Tests for the NOETHER_PAIRS database."""

    def test_all_pairs_have_required_fields(self):
        """All pairs should have required fields."""
        for name, pair in NOETHER_PAIRS.items():
            assert pair.symmetry, f"{name} missing symmetry"
            assert pair.symmetry_description, f"{name} missing description"
            assert pair.conservation_law, f"{name} missing conservation_law"
            assert pair.conserved_quantity, f"{name} missing conserved_quantity"
            assert pair.quantity_symbol, f"{name} missing quantity_symbol"
            assert pair.generator, f"{name} missing generator"

    def test_key_pairs_present(self):
        """Essential Noether pairs should be present."""
        required = [
            "time_translation",
            "spatial_translation",
            "rotation",
            "u1_gauge",
        ]
        for key in required:
            assert key in NOETHER_PAIRS, f"Missing essential pair: {key}"

    def test_symmetry_types(self):
        """Symmetry types should be correctly categorized."""
        # Spacetime symmetries
        assert NOETHER_PAIRS["time_translation"].symmetry_type == SymmetryType.SPACETIME
        assert NOETHER_PAIRS["spatial_translation"].symmetry_type == SymmetryType.SPACETIME
        assert NOETHER_PAIRS["rotation"].symmetry_type == SymmetryType.SPACETIME

        # Internal symmetries
        assert NOETHER_PAIRS["u1_gauge"].symmetry_type == SymmetryType.INTERNAL

    def test_conservation_types(self):
        """Conservation types should be correctly categorized."""
        # Extensive quantities (scale with system size)
        assert NOETHER_PAIRS["time_translation"].conservation_type == ConservationType.EXTENSIVE
        assert NOETHER_PAIRS["u1_gauge"].conservation_type == ConservationType.EXTENSIVE


class TestListFunctions:
    """Tests for list and get functions."""

    def test_list_all_pairs(self):
        """Should return all pair keys."""
        pairs = list_all_pairs()

        assert isinstance(pairs, list)
        assert len(pairs) >= 4  # At least the core pairs
        assert "time_translation" in pairs

    def test_get_pair_exists(self):
        """Should return pair for valid key."""
        pair = get_pair("time_translation")

        assert pair is not None
        assert "time" in pair.symmetry.lower()

    def test_get_pair_not_exists(self):
        """Should return None for invalid key."""
        pair = get_pair("nonexistent_key")

        assert pair is None


class TestReportOutput:
    """Tests for report string formatting."""

    def test_report_has_key_sections(self):
        """Report should have all key sections."""
        report = symmetry_to_conservation("time translation")
        s = str(report)

        assert "SYMMETRY" in s
        assert "CONSERVATION" in s
        assert "Generator" in s
        assert "Examples" in s
        assert "Common errors" in s

    def test_report_shows_direction(self):
        """Report should show mapping direction."""
        report = symmetry_to_conservation("rotation")
        s = str(report)

        assert "→" in s


class TestPhysicsCorrectness:
    """Tests for physics accuracy."""

    def test_translation_momentum_not_energy(self):
        """CRITICAL: Spatial translation → momentum (NOT energy).

        This is the common LLM error we're trying to correct.
        """
        report = symmetry_to_conservation("spatial translation")

        # Must be momentum
        assert "momentum" in report.pair.conservation_law.lower()
        # Must NOT be energy
        assert "energy" not in report.pair.conservation_law.lower()

    def test_time_energy_not_momentum(self):
        """Time translation → energy (NOT momentum)."""
        report = symmetry_to_conservation("time translation")

        # Must be energy
        assert "energy" in report.pair.conservation_law.lower()
        # Must NOT be momentum
        assert "momentum" not in report.pair.conservation_law.lower()

    def test_rotation_angular_not_linear(self):
        """Rotation → angular momentum (NOT linear momentum)."""
        report = symmetry_to_conservation("rotation")

        assert "angular" in report.pair.conservation_law.lower()

    def test_bidirectional_consistency(self):
        """Mapping should be consistent in both directions."""
        # symmetry → conservation → symmetry should round-trip
        forward = symmetry_to_conservation("time translation")
        conserved = forward.pair.conserved_quantity

        reverse = conservation_to_symmetry(conserved)

        # Should get back to time translation
        assert "time" in reverse.pair.symmetry.lower()

    def test_gauge_charge_relationship(self):
        """U(1) gauge → charge, not something else."""
        report = symmetry_to_conservation("U(1) gauge")

        # Should be electric/Noether charge
        assert "charge" in report.pair.conserved_quantity.lower()
