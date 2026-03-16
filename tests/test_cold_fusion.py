#!/usr/bin/env python3
"""
Tests for cold fusion fact-checking module.
"""

import pytest
from noethersolve.cold_fusion import (
    Reaction, ColdFusionClaim, Verdict,
    fact_check_claim, check_coulomb_barrier,
    NUCLEI, STANDARD_CLAIMS
)


class TestReaction:
    """Test nuclear reaction checks."""

    def test_dd_he4_conserves_charge(self):
        """D + D → He4 conserves charge."""
        rxn = Reaction(['d', 'd'], ['He4'])
        ok, msg = rxn.check_charge_conservation()
        assert ok
        assert "Z_in = Z_out = 2" in msg

    def test_dd_he4_conserves_baryon(self):
        """D + D → He4 conserves baryon number."""
        rxn = Reaction(['d', 'd'], ['He4'])
        ok, msg = rxn.check_baryon_conservation()
        assert ok
        assert "A_in = A_out = 4" in msg

    def test_dd_tp_branch(self):
        """D + D → T + p is valid."""
        rxn = Reaction(['d', 'd'], ['t', 'p'])
        charge_ok, _ = rxn.check_charge_conservation()
        baryon_ok, _ = rxn.check_baryon_conservation()
        assert charge_ok
        assert baryon_ok

    def test_dd_he3n_branch(self):
        """D + D → He3 + n is valid."""
        rxn = Reaction(['d', 'd'], ['He3', 'n'])
        charge_ok, _ = rxn.check_charge_conservation()
        baryon_ok, _ = rxn.check_baryon_conservation()
        assert charge_ok
        assert baryon_ok

    def test_impossible_reaction_violates_baryon(self):
        """Ni58 + p → Cu63 violates baryon conservation (ΔA = 4)."""
        rxn = Reaction(['Ni58', 'p'], ['Cu63'])
        ok, msg = rxn.check_baryon_conservation()
        assert not ok
        assert "VIOLATION" in msg
        assert "ΔA = 4" in msg

    def test_q_value_dd_he4(self):
        """Q-value for D+D→He4 should be ~23.8 MeV."""
        rxn = Reaction(['d', 'd'], ['He4'])
        q = rxn.compute_q_value()
        assert 23 < q < 24  # About 23.8 MeV

    def test_q_value_dd_tp(self):
        """Q-value for D+D→T+p should be ~4 MeV."""
        rxn = Reaction(['d', 'd'], ['t', 'p'])
        q = rxn.compute_q_value()
        assert 3.5 < q < 4.5  # About 4.03 MeV


class TestCoulombBarrier:
    """Test Coulomb barrier physics."""

    def test_room_temp_dd_impossible(self):
        """D-D fusion at room temperature should fail barrier check."""
        rxn = Reaction(['d', 'd'], ['He4'])
        ok, msg = check_coulomb_barrier(rxn, 300)  # Room temp
        assert not ok
        assert "ratio" in msg.lower()

    def test_hot_fusion_dd_plausible(self):
        """D-D fusion at 100 million K should pass barrier check."""
        rxn = Reaction(['d', 'd'], ['He4'])
        ok, msg = check_coulomb_barrier(rxn, 1e8)
        assert ok


class TestFactCheck:
    """Test overall fact-checking."""

    def test_fleischmann_pons_implausible(self):
        """Fleischmann-Pons claim should be implausible."""
        claim = STANDARD_CLAIMS["fleischmann_pons_1989"]
        results = fact_check_claim(claim)
        assert results["overall_verdict"] in [Verdict.IMPLAUSIBLE, Verdict.VIOLATES_CONSERVATION]

    def test_rossi_ecat_violates_conservation(self):
        """Rossi E-Cat should violate conservation laws."""
        claim = STANDARD_CLAIMS["rossi_ecat"]
        results = fact_check_claim(claim)
        assert results["overall_verdict"] == Verdict.VIOLATES_CONSERVATION
        assert "Baryon number conservation" in results["violations"]

    def test_standard_dd_branch1_plausible(self):
        """Standard hot D-D fusion branch 1 should be plausible."""
        claim = STANDARD_CLAIMS["dd_standard_branch1"]
        results = fact_check_claim(claim)
        assert results["overall_verdict"] == Verdict.PLAUSIBLE

    def test_standard_dd_branch2_plausible(self):
        """Standard hot D-D fusion branch 2 should be plausible."""
        claim = STANDARD_CLAIMS["dd_standard_branch2"]
        results = fact_check_claim(claim)
        assert results["overall_verdict"] == Verdict.PLAUSIBLE


class TestConservationLaws:
    """Test that conservation laws catch violations."""

    def test_charge_violation_detected(self):
        """Charge violation should be detected."""
        # p + p → He4 would violate charge (Z: 2 → 2 OK, but let's try invalid)
        # Actually need an invalid reaction
        rxn = Reaction(['p', 'p'], ['He4', 'p'])  # Z: 2 → 3
        ok, msg = rxn.check_charge_conservation()
        assert not ok

    def test_baryon_violation_detected(self):
        """Baryon violation should be detected."""
        rxn = Reaction(['p'], ['He4'])  # A: 1 → 4
        ok, msg = rxn.check_baryon_conservation()
        assert not ok


class TestNucleiDatabase:
    """Test nuclei database."""

    def test_common_nuclei_exist(self):
        """Common nuclei should be in database."""
        assert 'p' in NUCLEI
        assert 'd' in NUCLEI
        assert 't' in NUCLEI
        assert 'He3' in NUCLEI
        assert 'He4' in NUCLEI

    def test_deuterium_properties(self):
        """Deuterium should have correct properties."""
        d = NUCLEI['d']
        assert d.Z == 1
        assert d.A == 2
        assert d.N == 1
        assert 2.0 < d.mass_amu < 2.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
