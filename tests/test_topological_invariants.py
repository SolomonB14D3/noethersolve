"""Tests for noethersolve.topological_invariants module.

Tests cover:
- Chern number (exact integer quantization)
- Z2 invariant
- Bulk-boundary correspondence
- Quantum Hall effect
- Berry phase
- Topological classification (periodic table)
"""

import math
import pytest

from noethersolve.topological_invariants import (
    chern_number,
    z2_invariant,
    bulk_boundary_correspondence,
    quantum_hall,
    berry_phase,
    topological_classification,
    list_symmetry_classes,
    VON_KLITZING,
    CONDUCTANCE_QUANTUM,
)


# ─── Chern Number Tests ─────────────────────────────────────────────────────

class TestChernNumber:
    """Tests for chern_number()."""

    def test_exactly_quantized(self):
        """Chern number is marked as exactly quantized."""
        r = chern_number(band_index=1)
        assert r.is_exactly_quantized is True
        assert isinstance(r.chern_number, int)

    def test_quantum_hall(self):
        """Quantum Hall system has C=1."""
        r = chern_number(system="quantum_hall")
        assert r.chern_number == 1

    def test_haldane_model(self):
        """Haldane model has C=1."""
        r = chern_number(system="haldane")
        assert r.chern_number == 1

    def test_invalid_band(self):
        """Band index must be at least 1."""
        with pytest.raises(ValueError):
            chern_number(band_index=0)

    def test_invalid_system(self):
        """Unknown system raises error."""
        with pytest.raises(ValueError):
            chern_number(system="unknown")


# ─── Z2 Invariant Tests ─────────────────────────────────────────────────────

class TestZ2Invariant:
    """Tests for z2_invariant()."""

    def test_trivial(self):
        """ν=0 is trivial."""
        r = z2_invariant(nu=0)
        assert r.classification == "trivial"

    def test_topological(self):
        """ν=1 is topological."""
        r = z2_invariant(nu=1)
        assert r.classification == "topological"

    def test_binary_only(self):
        """Z2 can only be 0 or 1."""
        with pytest.raises(ValueError):
            z2_invariant(nu=2)
        with pytest.raises(ValueError):
            z2_invariant(nu=-1)

    def test_3d_indices(self):
        """3D has four indices."""
        r = z2_invariant(nu=1, dimension=3)
        assert r.indices is not None
        assert len(r.indices) == 4

    def test_dimension_valid(self):
        """Dimension must be 2 or 3."""
        with pytest.raises(ValueError):
            z2_invariant(nu=0, dimension=1)

    def test_time_reversal_protection(self):
        """Z2 is protected by time reversal."""
        r = z2_invariant(nu=1)
        assert "time-reversal" in r.protected_by.lower()


# ─── Bulk-Boundary Correspondence Tests ─────────────────────────────────────

class TestBulkBoundary:
    """Tests for bulk_boundary_correspondence()."""

    def test_correspondence_satisfied(self):
        """Edge modes equal bulk invariant."""
        r = bulk_boundary_correspondence(bulk_invariant=2, edge_modes=2)
        assert r.correspondence_satisfied is True

    def test_correspondence_violated(self):
        """Mismatch indicates error."""
        r = bulk_boundary_correspondence(bulk_invariant=2, edge_modes=1)
        assert r.correspondence_satisfied is False

    def test_default_edge_modes(self):
        """Default edge modes = |bulk|."""
        r = bulk_boundary_correspondence(bulk_invariant=3)
        assert r.edge_modes == 3
        assert r.correspondence_satisfied is True

    def test_negative_invariant(self):
        """Negative invariant uses absolute value."""
        r = bulk_boundary_correspondence(bulk_invariant=-2, edge_modes=2)
        assert r.correspondence_satisfied is True


# ─── Quantum Hall Tests ─────────────────────────────────────────────────────

class TestQuantumHall:
    """Tests for quantum_hall()."""

    def test_integer_qhe(self):
        """Integer QHE at ν=1."""
        r = quantum_hall(filling_factor=1, is_integer=True)
        assert r.plateau_type == "integer"
        assert r.chern_number == 1
        assert r.is_exactly_quantized is True

    def test_hall_conductance_quantized(self):
        """σ_xy = ν × e²/h."""
        r = quantum_hall(filling_factor=2)
        assert r.hall_conductance == 2.0  # In units of e²/h

    def test_hall_resistance(self):
        """R_H = h/(ν × e²)."""
        r = quantum_hall(filling_factor=1)
        assert abs(r.hall_resistance - VON_KLITZING) < 1

    def test_von_klitzing_constant(self):
        """Von Klitzing constant is exact."""
        assert abs(VON_KLITZING - 25812.80745) < 0.001

    def test_fractional_qhe(self):
        """Fractional QHE at ν=1/3."""
        r = quantum_hall(filling_factor=1/3, is_integer=False)
        assert r.plateau_type == "fractional"
        assert abs(r.hall_conductance - 1/3) < 1e-10

    def test_invalid_filling(self):
        """Filling factor must be positive."""
        with pytest.raises(ValueError):
            quantum_hall(filling_factor=0)


# ─── Berry Phase Tests ──────────────────────────────────────────────────────

class TestBerryPhase:
    """Tests for berry_phase()."""

    def test_quantized_by_inversion(self):
        """Berry phase quantized by inversion symmetry."""
        r = berry_phase(phase_value=math.pi, symmetry="inversion")
        assert r.is_quantized is True
        assert r.quantized_value == 1.0

    def test_quantized_zero(self):
        """Berry phase = 0 when trivial."""
        r = berry_phase(phase_value=0.0, symmetry="inversion")
        assert r.quantized_value == 0.0

    def test_not_quantized_no_symmetry(self):
        """Without symmetry, not quantized."""
        r = berry_phase(phase_value=0.5)
        assert r.is_quantized is False
        assert r.quantized_value is None

    def test_normalized_to_2pi(self):
        """Phase normalized to [0, 2π)."""
        r = berry_phase(phase_value=3 * math.pi)
        assert 0 <= r.berry_phase < 2 * math.pi


# ─── Topological Classification Tests ───────────────────────────────────────

class TestTopologicalClassification:
    """Tests for topological_classification()."""

    def test_chern_insulator_2d(self):
        """Class A in 2D has Z invariant."""
        r = topological_classification("A", 2)
        assert r.invariant_type == "Z"

    def test_z2_ti_2d(self):
        """Class AII in 2D has Z2 invariant."""
        r = topological_classification("AII", 2)
        assert r.invariant_type == "Z2"

    def test_z2_ti_3d(self):
        """Class AII in 3D has Z2 invariant."""
        r = topological_classification("AII", 3)
        assert r.invariant_type == "Z2"

    def test_trivial(self):
        """Some classes are trivial."""
        r = topological_classification("AI", 2)
        assert r.invariant_type == "0"

    def test_time_reversal_symmetry(self):
        """Class AII has time reversal."""
        r = topological_classification("AII", 2)
        assert r.has_time_reversal is True

    def test_no_symmetry_class_a(self):
        """Class A has no symmetries."""
        r = topological_classification("A", 2)
        assert r.has_time_reversal is False
        assert r.has_particle_hole is False
        assert r.has_chiral is False

    def test_bdi_all_symmetries(self):
        """Class BDI has all symmetries."""
        r = topological_classification("BDI", 1)
        assert r.has_time_reversal is True
        assert r.has_particle_hole is True
        assert r.has_chiral is True

    def test_invalid_class(self):
        """Unknown class raises error."""
        with pytest.raises(ValueError):
            topological_classification("X", 2)

    def test_invalid_dimension(self):
        """Dimension must be 1, 2, or 3."""
        with pytest.raises(ValueError):
            topological_classification("A", 4)

    def test_list_classes(self):
        """list_symmetry_classes returns all 10 classes."""
        classes = list_symmetry_classes()
        assert len(classes) == 10
        assert "A" in classes
        assert "AII" in classes


# ─── Report String Tests ────────────────────────────────────────────────────

class TestReportStrings:
    """Tests for report __str__ methods."""

    def test_chern_str(self):
        """Chern number report is readable."""
        r = chern_number()
        s = str(r)
        assert "Chern" in s
        assert "EXACTLY" in s or "exactly" in s.lower()
        assert "integer" in s.lower()

    def test_z2_str(self):
        """Z2 report is readable."""
        r = z2_invariant(nu=1)
        s = str(r)
        assert "Z₂" in s or "Z2" in s
        assert "TOPOLOGICAL" in s

    def test_qhe_str(self):
        """Quantum Hall report is readable."""
        r = quantum_hall(filling_factor=1)
        s = str(r)
        assert "Hall" in s
        assert "QUANTIZATION" in s or "quantiz" in s.lower()

    def test_bulk_boundary_str(self):
        """Bulk-boundary report is readable."""
        r = bulk_boundary_correspondence(bulk_invariant=1)
        s = str(r)
        assert "Bulk" in s
        assert "Boundary" in s or "edge" in s.lower()

    def test_berry_str(self):
        """Berry phase report is readable."""
        r = berry_phase(phase_value=math.pi, symmetry="inversion")
        s = str(r)
        assert "Berry" in s
        assert "π" in s or "pi" in s.lower()

    def test_classification_str(self):
        """Classification report is readable."""
        r = topological_classification("AII", 3)
        s = str(r)
        assert "AII" in s
        assert "3D" in s


# ─── Physical Consistency Tests ─────────────────────────────────────────────

class TestPhysicalConsistency:
    """Tests for physical self-consistency."""

    def test_qhe_conductance_resistance(self):
        """σ × R = 1/ν² in proper units."""
        r = quantum_hall(filling_factor=1)
        # σ in e²/h, R in Ω
        # σ_SI = σ × (e²/h), R_H in Ω
        # Product should relate properly
        assert r.hall_conductance * r.hall_resistance == VON_KLITZING

    def test_bulk_boundary_chern(self):
        """Bulk-boundary holds for Chern insulator."""
        c = chern_number()
        bb = bulk_boundary_correspondence(bulk_invariant=c.chern_number)
        assert bb.correspondence_satisfied is True

    def test_periodic_table_symmetry(self):
        """Classes with same symmetries in same column."""
        # A and AII differ only in T²
        a = topological_classification("A", 2)
        aii = topological_classification("AII", 2)
        # A has no T, AII has T
        assert a.has_time_reversal is False
        assert aii.has_time_reversal is True
