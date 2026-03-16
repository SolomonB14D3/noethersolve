"""Tests for noethersolve.pk_model — pharmacokinetic modeling engine."""

import math
import pytest

from noethersolve.pk_model import (
    one_compartment_iv,
    one_compartment_oral,
    half_life,
    steady_state,
    auc_single_dose,
    dose_adjustment,
    IVBolusReport,
    OralDosingReport,
    HalfLifeReport,
    SteadyStateReport,
    AUCReport,
    DoseAdjustmentReport,
)


# ── One-Compartment IV Bolus ─────────────────────────────────────────────

class TestIVBolus:
    def test_initial_concentration(self):
        r = one_compartment_iv(dose=500, Vd=50, ke=0.1, t=0)
        assert isinstance(r, IVBolusReport)
        assert abs(r.C0 - 10.0) < 1e-10  # 500/50
        assert abs(r.Ct - 10.0) < 1e-10  # at t=0

    def test_exponential_decay(self):
        r = one_compartment_iv(dose=500, Vd=50, ke=0.1, t=10)
        expected = 10.0 * math.exp(-0.1 * 10)
        assert abs(r.Ct - expected) < 1e-10

    def test_half_life(self):
        r = one_compartment_iv(dose=500, Vd=50, ke=0.1, t=0)
        assert abs(r.half_life - math.log(2) / 0.1) < 1e-10

    def test_at_one_half_life(self):
        """At t = t½, concentration should be C0/2."""
        ke = 0.1
        t_half = math.log(2) / ke
        r = one_compartment_iv(dose=500, Vd=50, ke=ke, t=t_half)
        assert abs(r.Ct - r.C0 / 2) < 1e-10

    def test_clearance(self):
        r = one_compartment_iv(dose=500, Vd=50, ke=0.1, t=0)
        assert abs(r.CL - 5.0) < 1e-10  # 0.1 × 50

    def test_auc(self):
        r = one_compartment_iv(dose=500, Vd=50, ke=0.1, t=0)
        assert abs(r.AUC_inf - 500 / 5.0) < 1e-10  # dose/CL

    def test_fraction_remaining(self):
        r = one_compartment_iv(dose=500, Vd=50, ke=0.1, t=0)
        assert abs(r.fraction_remaining - 1.0) < 1e-10

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            one_compartment_iv(dose=-500, Vd=50, ke=0.1, t=0)

    def test_str_output(self):
        r = one_compartment_iv(dose=500, Vd=50, ke=0.1, t=6)
        assert "IV Bolus" in str(r)


# ── One-Compartment Oral Dosing ──────────────────────────────────────────

class TestOralDosing:
    def test_concentration_at_time(self):
        r = one_compartment_oral(dose=500, F=0.8, Vd=50, ka=1.5, ke=0.1, t=4)
        assert isinstance(r, OralDosingReport)
        assert r.Ct > 0

    def test_tmax_calculation(self):
        """Tmax = ln(ka/ke) / (ka - ke)."""
        ka, ke = 1.5, 0.1
        r = one_compartment_oral(dose=500, F=0.8, Vd=50, ka=ka, ke=ke, t=0)
        expected_tmax = math.log(ka / ke) / (ka - ke)
        assert abs(r.Tmax - expected_tmax) < 1e-10

    def test_cmax_at_tmax(self):
        """Concentration at Tmax should equal Cmax."""
        r = one_compartment_oral(dose=500, F=0.8, Vd=50, ka=1.5, ke=0.1, t=0)
        r_at_tmax = one_compartment_oral(dose=500, F=0.8, Vd=50, ka=1.5, ke=0.1, t=r.Tmax)
        assert abs(r_at_tmax.Ct - r.Cmax) < 1e-8

    def test_auc_oral(self):
        r = one_compartment_oral(dose=500, F=0.8, Vd=50, ka=1.5, ke=0.1, t=0)
        expected_auc = 0.8 * 500 / (0.1 * 50)
        assert abs(r.AUC_inf - expected_auc) < 1e-10

    def test_zero_at_t_zero(self):
        """At t=0, no drug has been absorbed yet."""
        r = one_compartment_oral(dose=500, F=0.8, Vd=50, ka=1.5, ke=0.1, t=0)
        assert abs(r.Ct) < 1e-10

    def test_ka_equals_ke_raises(self):
        with pytest.raises(ValueError, match="ka and ke must differ"):
            one_compartment_oral(dose=500, F=0.8, Vd=50, ka=0.1, ke=0.1, t=4)

    def test_invalid_F_raises(self):
        with pytest.raises(ValueError):
            one_compartment_oral(dose=500, F=1.5, Vd=50, ka=1.5, ke=0.1, t=4)

    def test_str_output(self):
        r = one_compartment_oral(dose=500, F=0.8, Vd=50, ka=1.5, ke=0.1, t=4)
        assert "Oral Dosing" in str(r)


# ── Half-Life ────────────────────────────────────────────────────────────

class TestHalfLife:
    def test_from_ke(self):
        r = half_life(ke=0.1)
        assert isinstance(r, HalfLifeReport)
        assert abs(r.half_life - math.log(2) / 0.1) < 1e-10

    def test_from_cl_vd(self):
        r = half_life(CL=10, Vd=50)
        ke = 10 / 50
        assert abs(r.half_life - math.log(2) / ke) < 1e-10

    def test_consistency(self):
        """Both methods should give the same answer."""
        CL, Vd = 10, 50
        ke = CL / Vd
        r1 = half_life(ke=ke)
        r2 = half_life(CL=CL, Vd=Vd)
        assert abs(r1.half_life - r2.half_life) < 1e-10

    def test_no_params_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            half_life()

    def test_short_half_life_note(self):
        r = half_life(ke=2.0)  # t½ ≈ 0.35h
        assert any("short" in n.lower() for n in r.notes)

    def test_long_half_life_note(self):
        r = half_life(ke=0.01)  # t½ ≈ 69h
        assert any("long" in n.lower() or "once-daily" in n.lower() for n in r.notes)

    def test_str_output(self):
        r = half_life(ke=0.1)
        assert "Half-Life" in str(r)


# ── Steady State ─────────────────────────────────────────────────────────

class TestSteadyState:
    def test_css_avg(self):
        """Css_avg = F×Dose/(CL×τ)."""
        r = steady_state(dose=500, F=0.8, CL=10, tau=8)
        assert isinstance(r, SteadyStateReport)
        expected = 0.8 * 500 / (10 * 8)
        assert abs(r.Css_avg - expected) < 1e-10

    def test_time_to_ss(self):
        """Time to SS ≈ 5 × t½."""
        r = steady_state(dose=500, F=0.8, CL=10, tau=8, Vd=50)
        ke = 10 / 50
        t_half = math.log(2) / ke
        assert abs(r.time_to_ss - 5 * t_half) < 1e-10

    def test_peak_gt_trough(self):
        r = steady_state(dose=500, F=0.8, CL=10, tau=8, Vd=50)
        assert r.Css_peak > r.Css_trough

    def test_accumulation_factor(self):
        r = steady_state(dose=500, F=0.8, CL=10, tau=8, Vd=50)
        assert r.accumulation_factor > 1

    def test_invalid_tau_raises(self):
        with pytest.raises(ValueError):
            steady_state(dose=500, F=0.8, CL=10, tau=-8)

    def test_str_output(self):
        r = steady_state(dose=500, F=0.8, CL=10, tau=8, Vd=50)
        assert "Steady-State" in str(r)


# ── AUC ──────────────────────────────────────────────────────────────────

class TestAUC:
    def test_basic_auc(self):
        r = auc_single_dose(dose=500, F=0.8, CL=10)
        assert isinstance(r, AUCReport)
        assert abs(r.AUC_inf - 40.0) < 1e-10  # 0.8×500/10

    def test_full_bioavailability(self):
        r = auc_single_dose(dose=500, F=1.0, CL=10)
        assert abs(r.AUC_inf - 50.0) < 1e-10

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            auc_single_dose(dose=-500, F=0.8, CL=10)

    def test_str_output(self):
        r = auc_single_dose(dose=500, F=0.8, CL=10)
        assert "AUC" in str(r)


# ── Dose Adjustment ──────────────────────────────────────────────────────

class TestDoseAdjustment:
    def test_cyp_inhibitor(self):
        """5× AUC increase → dose reduced to 1/5."""
        r = dose_adjustment(original_dose=100, fold_change_auc=5.0)
        assert isinstance(r, DoseAdjustmentReport)
        assert abs(r.adjusted_dose - 20.0) < 1e-10
        assert abs(r.adjustment_factor - 0.2) < 1e-10

    def test_cyp_inducer(self):
        """0.2× AUC (80% decrease) → dose increased 5×."""
        r = dose_adjustment(original_dose=100, fold_change_auc=0.2,
                           reason="Rifampin induction of CYP3A4")
        assert abs(r.adjusted_dose - 500.0) < 1e-10

    def test_no_change(self):
        r = dose_adjustment(original_dose=100, fold_change_auc=1.0)
        assert abs(r.adjusted_dose - 100.0) < 1e-10

    def test_strong_interaction_note(self):
        r = dose_adjustment(original_dose=100, fold_change_auc=6.0)
        assert any("strong" in n.lower() or "avoid" in n.lower() for n in r.notes)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            dose_adjustment(original_dose=-100, fold_change_auc=2.0)

    def test_str_output(self):
        r = dose_adjustment(original_dose=100, fold_change_auc=3.0)
        assert "Dose Adjustment" in str(r)


# ── Integration Tests ────────────────────────────────────────────────────

class TestIntegration:
    def test_iv_auc_consistency(self):
        """IV bolus AUC should match standalone AUC calculation."""
        r_iv = one_compartment_iv(dose=500, Vd=50, ke=0.1, t=0)
        r_auc = auc_single_dose(dose=500, F=1.0, CL=r_iv.CL)
        assert abs(r_iv.AUC_inf - r_auc.AUC_inf) < 1e-10

    def test_oral_auc_consistency(self):
        r_oral = one_compartment_oral(dose=500, F=0.8, Vd=50, ka=1.5, ke=0.1, t=0)
        r_auc = auc_single_dose(dose=500, F=0.8, CL=r_oral.CL)
        assert abs(r_oral.AUC_inf - r_auc.AUC_inf) < 1e-10

    def test_dose_adjustment_preserves_exposure(self):
        """After adjustment, new AUC should match original."""
        original_auc = auc_single_dose(dose=100, F=1.0, CL=10).AUC_inf
        r_adj = dose_adjustment(original_dose=100, fold_change_auc=3.0)
        adjusted_auc = auc_single_dose(dose=r_adj.adjusted_dose, F=1.0, CL=10).AUC_inf
        # adjusted_dose = 100/3, so adjusted_auc = (100/3)/10 = 10/3
        # original_auc = 100/10 = 10
        # With the fold change, effective AUC = adjusted_auc × fold = (10/3)×3 = 10 ✓
        effective_auc = adjusted_auc * r_adj.fold_change
        assert abs(effective_auc - original_auc) < 1e-10

    def test_full_pk_workflow(self):
        """Simulate a complete PK assessment: IV → half-life → SS → dose adj."""
        # Step 1: IV bolus to get PK parameters
        r_iv = one_compartment_iv(dose=500, Vd=50, ke=0.1, t=6)
        assert r_iv.Ct > 0

        # Step 2: Calculate half-life
        r_hl = half_life(ke=0.1)
        assert r_hl.half_life > 0

        # Step 3: Predict steady state
        r_ss = steady_state(dose=500, F=1.0, CL=r_iv.CL, tau=r_hl.half_life, Vd=50)
        assert r_ss.Css_avg > 0

        # Step 4: Adjust for CYP inhibitor
        r_adj = dose_adjustment(original_dose=500, fold_change_auc=3.0,
                               reason="Co-administration with CYP3A4 inhibitor")
        assert r_adj.adjusted_dose < 500
