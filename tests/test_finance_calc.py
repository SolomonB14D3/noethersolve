"""Tests for the finance calculator module."""

import math
import pytest
from noethersolve.finance_calc import (
    black_scholes,
    put_call_parity,
    nash_equilibrium_2x2,
    present_value,
    future_value,
)


class TestBlackScholes:
    def test_call_price_positive_itm(self):
        """In-the-money call (S > K) should have positive price."""
        r = black_scholes(S=110, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
        assert r.price > 0

    def test_put_price_positive_itm(self):
        """In-the-money put (S < K) should have positive price."""
        r = black_scholes(S=90, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")
        assert r.price > 0

    def test_call_delta_between_0_and_1(self):
        """Call delta is always in [0, 1]."""
        r = black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
        assert 0 <= r.delta <= 1

    def test_put_delta_between_neg1_and_0(self):
        """Put delta is always in [-1, 0]."""
        r = black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")
        assert -1 <= r.delta <= 0

    def test_gamma_positive(self):
        """Gamma is always positive for both calls and puts."""
        r = black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert r.gamma > 0

    def test_vega_positive(self):
        """Vega is always positive (higher vol = higher option price)."""
        r = black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert r.vega > 0

    def test_higher_vol_higher_price(self):
        """Higher volatility increases option price."""
        r_low = black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.10)
        r_high = black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.40)
        assert r_high.price > r_low.price

    def test_error_negative_S(self):
        with pytest.raises(ValueError, match="positive"):
            black_scholes(S=-100, K=100, T=1.0, r=0.05, sigma=0.20)

    def test_error_zero_K(self):
        with pytest.raises(ValueError, match="positive"):
            black_scholes(S=100, K=0, T=1.0, r=0.05, sigma=0.20)

    def test_error_zero_T(self):
        with pytest.raises(ValueError, match="positive"):
            black_scholes(S=100, K=100, T=0, r=0.05, sigma=0.20)

    def test_error_negative_sigma(self):
        with pytest.raises(ValueError, match="positive"):
            black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=-0.20)

    def test_error_invalid_type(self):
        with pytest.raises(ValueError, match="call.*put"):
            black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="straddle")

    def test_report_string(self):
        r = black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        s = str(r)
        assert "Black-Scholes" in s
        assert "Price" in s
        assert "Delta" in s
        assert "Gamma" in s


class TestPutCallParity:
    def test_bs_prices_satisfy_parity(self):
        """BS call and put prices should satisfy put-call parity: C - P = S - PV(K)."""
        S, K, T, r_rate, sigma = 100, 100, 1.0, 0.05, 0.20
        call = black_scholes(S, K, T, r_rate, sigma, "call")
        put = black_scholes(S, K, T, r_rate, sigma, "put")
        result = put_call_parity(call.price, put.price, S, K, r_rate, T)
        assert result.parity_holds is True
        assert result.arbitrage_opportunity is None

    def test_violation_detected(self):
        """Mispriced options should be detected as arbitrage."""
        r = put_call_parity(call_price=15.0, put_price=5.0, S=100, K=95, r=0.05, T=1.0)
        # C - P = 10, S - PV(K) = 100 - 95*e^(-0.05) ~ 9.63
        # Diff ~ 0.37 > 0.01 tolerance
        assert r.parity_holds is False
        assert r.arbitrage_opportunity is not None

    def test_lhs_rhs_values(self):
        r = put_call_parity(call_price=10.0, put_price=5.0, S=100, K=95, r=0.05, T=1.0)
        assert abs(r.lhs - 5.0) < 1e-10  # C - P = 5
        expected_rhs = 100 - 95 * math.exp(-0.05 * 1.0)
        assert abs(r.rhs - expected_rhs) < 1e-6

    def test_report_string(self):
        r = put_call_parity(call_price=10.0, put_price=5.0, S=100, K=95, r=0.05, T=1.0)
        s = str(r)
        assert "Put-Call Parity" in s
        assert "C - P" in s


class TestNashEquilibrium2x2:
    def test_prisoners_dilemma(self):
        """Prisoner's dilemma: (Defect, Defect) = (1,1) is the unique pure NE."""
        # Standard PD: T > R > P > S => payoffs: R=3, T=5, S=0, P=1
        p1 = [[3, 0], [5, 1]]
        p2 = [[3, 5], [0, 1]]
        r = nash_equilibrium_2x2(p1, p2)
        assert (1, 1) in r.pure_equilibria
        assert r.game_type == "prisoners_dilemma"

    def test_coordination_game(self):
        """Coordination game: two pure NE on diagonal."""
        p1 = [[3, 0], [0, 2]]
        p2 = [[3, 0], [0, 2]]
        r = nash_equilibrium_2x2(p1, p2)
        assert (0, 0) in r.pure_equilibria
        assert (1, 1) in r.pure_equilibria

    def test_dominant_strategy_detected(self):
        """Prisoner's dilemma has dominant strategy for both players."""
        p1 = [[3, 0], [5, 1]]
        p2 = [[3, 5], [0, 1]]
        r = nash_equilibrium_2x2(p1, p2)
        assert len(r.dominant_strategies) > 0

    def test_mixed_equilibrium_exists(self):
        """Matching pennies has no pure NE but a mixed NE."""
        p1 = [[1, -1], [-1, 1]]
        p2 = [[-1, 1], [1, -1]]
        r = nash_equilibrium_2x2(p1, p2)
        assert len(r.pure_equilibria) == 0
        assert r.mixed_equilibrium is not None
        # Each player mixes 50/50
        p1_mix, p2_mix = r.mixed_equilibrium
        assert abs(p1_mix[0] - 0.5) < 0.01
        assert abs(p2_mix[0] - 0.5) < 0.01

    def test_error_non_2x2(self):
        with pytest.raises(ValueError, match="2x2"):
            nash_equilibrium_2x2([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]])

    def test_report_string(self):
        p1 = [[3, 0], [5, 1]]
        p2 = [[3, 5], [0, 1]]
        r = nash_equilibrium_2x2(p1, p2)
        s = str(r)
        assert "Nash Equilibrium" in s
        assert "Pure NE" in s


class TestPresentValue:
    def test_pv_of_1000_at_5pct_10yr(self):
        """PV of 1000 at 5% for 10 years ~ 613.91."""
        r = present_value(future_value=1000, rate=0.05, periods=10)
        assert abs(r.present_value - 613.91) < 0.02

    def test_zero_periods_pv_equals_fv(self):
        """Zero periods means PV = FV."""
        r = present_value(future_value=1000, rate=0.05, periods=0)
        assert abs(r.present_value - 1000) < 1e-10

    def test_higher_rate_lower_pv(self):
        """Higher discount rate means lower present value."""
        r_low = present_value(future_value=1000, rate=0.02, periods=10)
        r_high = present_value(future_value=1000, rate=0.10, periods=10)
        assert r_high.present_value < r_low.present_value

    def test_continuous_compounding(self):
        """Continuous PV = FV * e^(-rT)."""
        r = present_value(future_value=1000, rate=0.05, periods=10, continuous=True)
        expected = 1000 * math.exp(-0.05 * 10)
        assert abs(r.present_value - expected) < 1e-6

    def test_error_negative_periods(self):
        with pytest.raises(ValueError, match="non-negative"):
            present_value(future_value=1000, rate=0.05, periods=-1)

    def test_report_string(self):
        r = present_value(future_value=1000, rate=0.05, periods=10)
        s = str(r)
        assert "Time Value" in s
        assert "Present Value" in s
        assert "Future Value" in s


class TestFutureValue:
    def test_fv_of_1000_at_5pct_10yr(self):
        """FV of 1000 at 5% for 10 years ~ 1628.89."""
        r = future_value(present_val=1000, rate=0.05, periods=10)
        assert abs(r.future_value - 1628.89) < 0.02

    def test_zero_rate_no_growth(self):
        """Zero rate means FV = PV."""
        r = future_value(present_val=1000, rate=0.0, periods=10)
        assert abs(r.future_value - 1000) < 1e-10

    def test_continuous_compounding(self):
        """Continuous FV = PV * e^(rT)."""
        r = future_value(present_val=1000, rate=0.05, periods=10, continuous=True)
        expected = 1000 * math.exp(0.05 * 10)
        assert abs(r.future_value - expected) < 1e-6

    def test_pv_fv_roundtrip(self):
        """PV(FV(X)) should return X."""
        fv_r = future_value(present_val=500, rate=0.08, periods=5)
        pv_r = present_value(future_value=fv_r.future_value, rate=0.08, periods=5)
        assert abs(pv_r.present_value - 500) < 1e-6

    def test_error_negative_periods(self):
        with pytest.raises(ValueError, match="non-negative"):
            future_value(present_val=1000, rate=0.05, periods=-1)
