"""Tests for battery_degradation module."""

import pytest
import math
from noethersolve.battery_degradation import (
    calc_calendar_aging,
    calc_cycle_aging,
    calc_combined_aging,
    compare_chemistries,
    CHEMISTRY_PARAMS,
)


class TestCalendarAging:
    """Tests for calendar (storage) aging."""

    def test_sqrt_t_scaling(self):
        """Calendar aging should scale as sqrt(t), NOT linear."""
        report_1y = calc_calendar_aging("NMC", 365, temperature_C=25)
        report_4y = calc_calendar_aging("NMC", 365*4, temperature_C=25)

        # 4 years should give ~2x loss (sqrt(4) = 2), not 4x
        ratio = report_4y.capacity_loss_percent / report_1y.capacity_loss_percent
        assert 1.8 < ratio < 2.2, f"Expected ~2x loss, got {ratio}x (sqrt(t) law)"

    def test_arrhenius_temperature(self):
        """Higher temperature should accelerate aging (Arrhenius)."""
        report_25C = calc_calendar_aging("NMC", 365, temperature_C=25)
        report_45C = calc_calendar_aging("NMC", 365, temperature_C=45)

        assert report_45C.capacity_loss_percent > report_25C.capacity_loss_percent
        # Rule of thumb: ~2x per 10°C
        ratio = report_45C.capacity_loss_percent / report_25C.capacity_loss_percent
        assert ratio > 1.5, f"Expected significant temperature effect, got {ratio}x"

    def test_soc_dependence(self):
        """Higher storage SOC should accelerate aging."""
        report_50 = calc_calendar_aging("NMC", 365, soc_storage=0.5)
        report_100 = calc_calendar_aging("NMC", 365, soc_storage=1.0)

        assert report_100.capacity_loss_percent > report_50.capacity_loss_percent

    def test_zero_time(self):
        """Zero storage time should give zero loss."""
        report = calc_calendar_aging("NMC", 0, temperature_C=25)
        assert report.capacity_loss_percent == 0.0

    def test_chemistry_differences(self):
        """Different chemistries should have different aging rates."""
        report_nmc = calc_calendar_aging("NMC", 365)
        report_lfp = calc_calendar_aging("LFP", 365)

        # LFP typically has lower calendar aging than NMC
        assert report_lfp.capacity_loss_percent < report_nmc.capacity_loss_percent

    def test_invalid_chemistry(self):
        """Invalid chemistry should raise error."""
        with pytest.raises(ValueError, match="Unknown chemistry"):
            calc_calendar_aging("INVALID", 365)

    def test_report_str(self):
        """Report should have readable string representation."""
        report = calc_calendar_aging("NMC", 365)
        s = str(report)
        assert "Calendar" in s
        assert "√t" in s or "sqrt" in s.lower()
        assert "NMC" in s


class TestCycleAging:
    """Tests for cycle aging."""

    def test_dod_scaling(self):
        """Higher DOD should cause more degradation."""
        report_50 = calc_cycle_aging("NMC", 500, dod=0.5)
        report_100 = calc_cycle_aging("NMC", 500, dod=1.0)

        # 100% DOD should be significantly worse than 50%
        ratio = report_100.capacity_loss_percent / report_50.capacity_loss_percent
        assert ratio > 2, f"Expected DOD^n scaling, got {ratio}x"

    def test_cycle_count_linear(self):
        """Cycle aging should scale roughly linearly with cycles."""
        report_500 = calc_cycle_aging("NMC", 500)
        report_1000 = calc_cycle_aging("NMC", 1000)

        ratio = report_1000.capacity_loss_percent / report_500.capacity_loss_percent
        assert 1.8 < ratio < 2.2, f"Expected linear scaling, got {ratio}x"

    def test_zero_cycles(self):
        """Zero cycles should give zero loss."""
        report = calc_cycle_aging("NMC", 0)
        assert report.capacity_loss_percent == 0.0

    def test_lfp_cycle_tolerance(self):
        """LFP should handle cycles better at high DOD."""
        calc_cycle_aging("NMC", 1000, dod=0.9)
        calc_cycle_aging("LFP", 1000, dod=0.9)

        # LFP has lower DOD exponent, so less sensitive to deep cycles
        nmc_ratio = calc_cycle_aging("NMC", 1000, dod=0.9).capacity_loss_percent / \
                    calc_cycle_aging("NMC", 1000, dod=0.5).capacity_loss_percent
        lfp_ratio = calc_cycle_aging("LFP", 1000, dod=0.9).capacity_loss_percent / \
                    calc_cycle_aging("LFP", 1000, dod=0.5).capacity_loss_percent

        assert lfp_ratio < nmc_ratio, "LFP should be less DOD-sensitive"

    def test_report_str(self):
        """Report should have readable string representation."""
        report = calc_cycle_aging("NMC", 500)
        s = str(report)
        assert "Cycle" in s
        assert "DOD" in s


class TestCombinedAging:
    """Tests for combined calendar + cycle aging."""

    def test_additive_not_multiplicative(self):
        """Calendar and cycle aging should be ADDITIVE."""
        cal_report = calc_calendar_aging("NMC", 365)
        cyc_report = calc_cycle_aging("NMC", 500)
        combined = calc_combined_aging("NMC", 365, 500)

        # Total should equal sum of individual components
        expected_sum = cal_report.capacity_loss_percent + cyc_report.capacity_loss_percent
        assert abs(combined.total_loss_percent - expected_sum) < 0.01

    def test_remaining_capacity(self):
        """Remaining capacity should be 100 - total loss."""
        report = calc_combined_aging("NMC", 365, 500)
        assert abs(report.remaining_capacity_percent - (100 - report.total_loss_percent)) < 0.01

    def test_calendar_fraction(self):
        """Calendar fraction should be between 0 and 1."""
        report = calc_combined_aging("NMC", 365, 500)
        assert 0 <= report.calendar_fraction <= 1

    def test_dominant_mechanism_identification(self):
        """Should correctly identify dominant mechanism."""
        # Heavy cycling, short time -> cycle dominant
        report_cycle = calc_combined_aging("NMC", 30, 1000)
        # Pure storage, zero cycles -> must be calendar dominant
        report_cal = calc_combined_aging("NMC", 1000, 0)  # Storage only

        assert "Cycle" in report_cycle.dominant_mechanism or report_cycle.calendar_fraction < 0.4
        # With 0 cycles, calendar must be 100%
        assert report_cal.calendar_fraction == 1.0 or report_cal.cycle_loss_percent == 0.0

    def test_report_str(self):
        """Report should emphasize additive nature."""
        report = calc_combined_aging("NMC", 365, 500)
        s = str(report)
        assert "ADDITIVE" in s or "additive" in s.lower()


class TestCompareChemistries:
    """Tests for chemistry comparison."""

    def test_returns_string(self):
        """Should return formatted comparison string."""
        result = compare_chemistries(365, 500)
        assert isinstance(result, str)
        assert "NMC" in result
        assert "LFP" in result
        assert "NCA" in result

    def test_ranks_chemistries(self):
        """Should rank chemistries by remaining capacity."""
        result = compare_chemistries(365, 500)
        assert "Best" in result


class TestPhysicsCorrectness:
    """Tests for physics accuracy."""

    def test_sei_sqrt_t_is_exact(self):
        """SEI growth sqrt(t) law should be exact."""
        for t in [100, 400, 900, 1600]:
            report = calc_calendar_aging("NMC", t, temperature_C=25, soc_storage=0.5)
            # sei_growth_factor should be exactly sqrt(t)
            assert abs(report.sei_growth_factor - math.sqrt(t)) < 1e-10

    def test_arrhenius_form(self):
        """Arrhenius factor should follow exp(-Ea/RT)."""
        report = calc_calendar_aging("NMC", 365, temperature_C=25)
        T = 25 + 273.15
        Ea = CHEMISTRY_PARAMS["NMC"]["Ea_calendar"]
        R = 8.314
        expected = math.exp(-Ea / (R * T))
        assert abs(report.arrhenius_factor - expected) < 1e-10

    def test_reasonable_lifetime_estimates(self):
        """Lifetime estimates should be reasonable."""
        # At 25°C, NMC should last multiple years
        report = calc_calendar_aging("NMC", 365, temperature_C=25)
        assert report.time_to_80_percent > 365 * 2  # At least 2 years

        # Cycles to 80% should be in the hundreds to thousands
        cyc_report = calc_cycle_aging("NMC", 500, dod=0.8)
        assert 500 < cyc_report.cycles_to_80_percent < 5000
