"""Tests for the chemistry calculator module."""

import math
import pytest
from noethersolve.chemistry_calc import (
    nernst_equation,
    henderson_hasselbalch,
    crystal_field_splitting,
    band_gap_analysis,
    balance_redox,
)


class TestNernstEquation:
    def test_standard_conditions_Q1(self):
        """At Q=1, E = E_standard (ln(1)=0 so correction term vanishes)."""
        r = nernst_equation(E_standard=1.10, n_electrons=2, Q=1.0)
        assert abs(r.E_cell - 1.10) < 1e-10

    def test_positive_E_is_spontaneous(self):
        """Positive cell potential means spontaneous reaction."""
        r = nernst_equation(E_standard=1.10, n_electrons=2, Q=1.0)
        assert r.spontaneous is True
        assert r.delta_G < 0  # negative delta_G = spontaneous

    def test_negative_E_not_spontaneous(self):
        """Negative cell potential means non-spontaneous."""
        r = nernst_equation(E_standard=-0.50, n_electrons=2, Q=1.0)
        assert r.spontaneous is False
        assert r.delta_G > 0

    def test_large_Q_reduces_potential(self):
        """Large Q (products >> reactants) reduces E below E_standard."""
        r = nernst_equation(E_standard=1.10, n_electrons=2, Q=1e6)
        assert r.E_cell < r.E_standard

    def test_small_Q_increases_potential(self):
        """Small Q (reactants >> products) increases E above E_standard."""
        r = nernst_equation(E_standard=1.10, n_electrons=2, Q=1e-6)
        assert r.E_cell > r.E_standard

    def test_delta_G_relation(self):
        """delta_G = -nFE must hold."""
        r = nernst_equation(E_standard=0.80, n_electrons=3, Q=0.01)
        expected_dG = -3 * 96485.33212 * r.E_cell
        assert abs(r.delta_G - expected_dG) < 1.0

    def test_error_negative_Q(self):
        with pytest.raises(ValueError, match="positive"):
            nernst_equation(E_standard=1.0, n_electrons=2, Q=-1.0)

    def test_error_zero_electrons(self):
        with pytest.raises(ValueError, match="positive"):
            nernst_equation(E_standard=1.0, n_electrons=0, Q=1.0)

    def test_error_zero_temperature(self):
        with pytest.raises(ValueError, match="positive"):
            nernst_equation(E_standard=1.0, n_electrons=2, Q=1.0, temperature=0)

    def test_report_string(self):
        r = nernst_equation(E_standard=1.10, n_electrons=2, Q=1.0)
        s = str(r)
        assert "Nernst Equation" in s
        assert "1.1000 V" in s
        assert "Spontaneous" in s


class TestHendersonHasselbalch:
    def test_equal_concentrations_gives_pKa(self):
        """When [HA] = [A-], pH = pKa (log(1) = 0)."""
        r = henderson_hasselbalch(pKa=4.75, acid_conc=0.1, base_conc=0.1)
        assert abs(r.pH - 4.75) < 1e-10

    def test_more_base_raises_pH(self):
        """More conjugate base than acid raises pH above pKa."""
        r = henderson_hasselbalch(pKa=4.75, acid_conc=0.1, base_conc=1.0)
        assert r.pH > 4.75

    def test_more_acid_lowers_pH(self):
        """More acid than conjugate base lowers pH below pKa."""
        r = henderson_hasselbalch(pKa=4.75, acid_conc=1.0, base_conc=0.1)
        assert r.pH < 4.75

    def test_effective_range(self):
        """Effective range is pKa +/- 1."""
        r = henderson_hasselbalch(pKa=7.20, acid_conc=0.1, base_conc=0.1)
        assert r.effective_range == (6.20, 8.20)

    def test_buffer_capacity_positive(self):
        r = henderson_hasselbalch(pKa=4.75, acid_conc=0.1, base_conc=0.1)
        assert r.buffer_capacity > 0

    def test_error_zero_acid_conc(self):
        with pytest.raises(ValueError, match="positive"):
            henderson_hasselbalch(pKa=4.75, acid_conc=0.0, base_conc=0.1)

    def test_error_negative_base_conc(self):
        with pytest.raises(ValueError, match="positive"):
            henderson_hasselbalch(pKa=4.75, acid_conc=0.1, base_conc=-0.1)

    def test_report_string(self):
        r = henderson_hasselbalch(pKa=4.75, acid_conc=0.1, base_conc=0.1)
        s = str(r)
        assert "Henderson-Hasselbalch" in s
        assert "pKa" in s
        assert "Effective range" in s


class TestCrystalFieldSplitting:
    def test_d3_octahedral_high_spin(self):
        """d3 octahedral: t2g^3 eg^0, 3 unpaired electrons."""
        r = crystal_field_splitting(d_electrons=3, geometry="octahedral")
        assert r.configuration == "t2g^3 eg^0"
        assert r.unpaired_electrons == 3
        assert r.spin_state == "high_spin"

    def test_d6_octahedral_low_spin(self):
        """d6 octahedral low-spin: t2g^6 eg^0, 0 unpaired."""
        r = crystal_field_splitting(d_electrons=6, geometry="octahedral", strong_field=True)
        assert r.configuration == "t2g^6 eg^0"
        assert r.unpaired_electrons == 0
        assert r.spin_state == "low_spin"

    def test_d8_octahedral_high_spin(self):
        """d8 octahedral high-spin: t2g^6 eg^2, 2 unpaired."""
        r = crystal_field_splitting(d_electrons=8, geometry="octahedral")
        assert r.configuration == "t2g^6 eg^2"
        assert r.unpaired_electrons == 2

    def test_tetrahedral_always_high_spin(self):
        r = crystal_field_splitting(d_electrons=5, geometry="tetrahedral")
        assert r.spin_state == "high_spin"

    def test_d10_all_paired(self):
        """d10 in any geometry: 0 unpaired electrons."""
        r = crystal_field_splitting(d_electrons=10, geometry="octahedral")
        assert r.unpaired_electrons == 0

    def test_error_d0(self):
        with pytest.raises(ValueError, match="between 1 and 10"):
            crystal_field_splitting(d_electrons=0)

    def test_error_d11(self):
        with pytest.raises(ValueError, match="between 1 and 10"):
            crystal_field_splitting(d_electrons=11)

    def test_error_bad_geometry(self):
        with pytest.raises(ValueError, match="geometry"):
            crystal_field_splitting(d_electrons=3, geometry="trigonal")

    def test_report_string(self):
        r = crystal_field_splitting(d_electrons=3, geometry="octahedral")
        s = str(r)
        assert "Crystal Field" in s
        assert "octahedral" in s
        assert "Unpaired" in s


class TestBandGapAnalysis:
    def test_silicon_semiconductor(self):
        """Si band gap 1.1 eV should be classified as semiconductor."""
        r = band_gap_analysis(band_gap_eV=1.1)
        assert r.conductor_type == "semiconductor"

    def test_diamond_insulator(self):
        """Diamond ~5.5 eV should be insulator."""
        r = band_gap_analysis(band_gap_eV=5.5)
        assert r.conductor_type == "insulator"

    def test_metal_conductor(self):
        """Zero band gap = conductor."""
        r = band_gap_analysis(band_gap_eV=0.0)
        assert r.conductor_type == "conductor"

    def test_wavelength_silicon(self):
        """Si absorption edge: hc/1.1eV ~ 1127 nm (infrared)."""
        r = band_gap_analysis(band_gap_eV=1.1)
        assert 1100 < r.wavelength_nm < 1150

    def test_carrier_conc_decreases_with_gap(self):
        """Larger gap means fewer intrinsic carriers at same temperature."""
        r_si = band_gap_analysis(band_gap_eV=1.1, temperature=300)
        r_wide = band_gap_analysis(band_gap_eV=3.0, temperature=300)
        assert r_si.intrinsic_carrier_conc > r_wide.intrinsic_carrier_conc

    def test_error_negative_gap(self):
        with pytest.raises(ValueError, match="non-negative"):
            band_gap_analysis(band_gap_eV=-1.0)

    def test_error_zero_temperature(self):
        with pytest.raises(ValueError, match="positive"):
            band_gap_analysis(band_gap_eV=1.1, temperature=0)

    def test_report_string(self):
        r = band_gap_analysis(band_gap_eV=1.1)
        s = str(r)
        assert "Band Gap" in s
        assert "semiconductor" in s
        assert "Absorption edge" in s


class TestBalanceRedox:
    def test_standard_conditions(self):
        """balance_redox with E_cathode - E_anode at Q=1."""
        r = balance_redox(E_cathode=0.34, E_anode=-0.76, n_electrons=2)
        assert abs(r.E_cell - 1.10) < 0.01
        assert r.spontaneous is True

    def test_reverse_reaction(self):
        """If anode > cathode, reaction is non-spontaneous."""
        r = balance_redox(E_cathode=-0.76, E_anode=0.34, n_electrons=2)
        assert r.E_cell < 0
        assert r.spontaneous is False
