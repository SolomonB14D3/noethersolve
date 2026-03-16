"""Tests for the PID controller simulator and stability analyzer."""

import math
import pytest
from noethersolve.control import simulate_pid, analyze_stability, PIDReport, StabilityReport


class TestSimulatePID:
    def test_first_order_plant_converges(self):
        """PID on 1/(s+1) should converge to setpoint with Ki > 0."""
        r = simulate_pid(Kp=2.0, Ki=1.0, Kd=0.0,
                         plant_num=[1.0], plant_den=[1.0, 1.0])
        assert r.verdict == "STABLE"
        assert r.steady_state_error < 0.01
        assert abs(r.final_value - 1.0) < 0.01

    def test_proportional_only_has_sse(self):
        """Pure P controller has steady-state error (no integral action)."""
        r = simulate_pid(Kp=1.0, Ki=0.0, Kd=0.0,
                         plant_num=[1.0], plant_den=[1.0, 1.0])
        # P-only on 1/(s+1): final value = Kp/(1+Kp) = 0.5
        assert r.steady_state_error > 0.1
        assert abs(r.final_value - 0.5) < 0.05  # Kp/(1+Kp)

    def test_second_order_dc_gain(self):
        """P-only on 1/(s^2+3s+1): final value = Kp×G(0)/(1+Kp×G(0)) = Kp/(1+Kp)."""
        r = simulate_pid(Kp=2.0, Ki=0.0, Kd=0.0,
                         plant_num=[1.0], plant_den=[1.0, 3.0, 1.0])
        # G(0)=1, so final = 2/(1+2) = 2/3 ≈ 0.667
        assert abs(r.final_value - 2.0 / 3.0) < 0.05

    def test_type_1_plant_zero_sse(self):
        """P-only on 1/(s^2+s) [type-1]: zero steady-state error for step."""
        r = simulate_pid(Kp=10.0, Ki=0.0, Kd=0.0,
                         plant_num=[1.0], plant_den=[1.0, 1.0, 0.0])
        # Type-1 system: position error = 0 for step input
        assert abs(r.final_value - 1.0) < 0.05
        assert r.steady_state_error < 0.05

    def test_high_gain_instability(self):
        """Very high gains on 2nd-order plant cause instability."""
        r = simulate_pid(Kp=50.0, Ki=20.0, Kd=0.0,
                         plant_num=[1.0], plant_den=[1.0, 3.0, 1.0])
        assert r.verdict in ("UNSTABLE", "MARGINAL")  # depends on sim duration

    def test_custom_setpoint(self):
        """Setpoint other than 1.0 works correctly."""
        r = simulate_pid(Kp=2.0, Ki=1.0, Kd=0.0,
                         plant_num=[1.0], plant_den=[1.0, 1.0],
                         setpoint=5.0)
        assert abs(r.final_value - 5.0) < 0.1

    def test_zero_gains_no_movement(self):
        """Zero PID gains = no control action."""
        r = simulate_pid(Kp=0.0, Ki=0.0, Kd=0.0,
                         plant_num=[1.0], plant_den=[1.0, 1.0])
        assert abs(r.final_value) < 0.01

    def test_report_has_time_series(self):
        r = simulate_pid(Kp=1.0, Ki=0.5, Kd=0.0,
                         plant_num=[1.0], plant_den=[1.0, 1.0])
        assert len(r.time_series) > 100

    def test_report_string(self):
        r = simulate_pid(Kp=1.0, Ki=0.5, Kd=0.0,
                         plant_num=[1.0], plant_den=[1.0, 1.0])
        s = str(r)
        assert "PID Step Response" in s
        assert "Overshoot" in s
        assert "Settling time" in s

    def test_plant_den_must_be_first_order_or_higher(self):
        with pytest.raises(ValueError):
            simulate_pid(plant_num=[1.0], plant_den=[1.0])

    def test_derivative_action_reduces_overshoot(self):
        """Adding derivative should reduce overshoot vs PI only."""
        r_pi = simulate_pid(Kp=5.0, Ki=2.0, Kd=0.0,
                            plant_num=[1.0], plant_den=[1.0, 1.0])
        r_pid = simulate_pid(Kp=5.0, Ki=2.0, Kd=1.0,
                             plant_num=[1.0], plant_den=[1.0, 1.0])
        # PID should have less or equal overshoot
        assert r_pid.overshoot_pct <= r_pi.overshoot_pct + 5.0


class TestAnalyzeStability:
    def test_stable_polynomial(self):
        """s^3 + 3s^2 + 3s + 1 = (s+1)^3 — all poles at -1."""
        r = analyze_stability([1.0, 3.0, 3.0, 1.0])
        assert r.verdict == "STABLE"
        assert r.n_unstable == 0
        assert len(r.poles) == 3
        for p in r.poles:
            assert p.real < 0

    def test_unstable_polynomial(self):
        """s^2 - 1 = (s-1)(s+1) — one unstable pole."""
        r = analyze_stability([1.0, 0.0, -1.0])
        assert r.verdict == "UNSTABLE"
        assert r.n_unstable >= 1

    def test_marginal_polynomial(self):
        """s^2 + 1 — poles at ±j (imaginary axis)."""
        r = analyze_stability([1.0, 0.0, 1.0])
        assert r.verdict == "MARGINAL"

    def test_first_order_stable(self):
        """s + 2 — pole at -2."""
        r = analyze_stability([1.0, 2.0])
        assert r.verdict == "STABLE"
        assert len(r.poles) == 1
        assert abs(r.poles[0].real + 2.0) < 0.01

    def test_quadratic_poles(self):
        """s^2 + 2s + 5 — poles at -1 ± 2j."""
        r = analyze_stability([1.0, 2.0, 5.0])
        assert r.verdict == "STABLE"
        assert len(r.poles) == 2
        for p in r.poles:
            assert abs(p.real + 1.0) < 0.01
            assert abs(abs(p.imag) - 2.0) < 0.01

    def test_routh_table_built(self):
        r = analyze_stability([1.0, 3.0, 3.0, 1.0])
        assert len(r.routh_table) >= 3

    def test_report_string(self):
        r = analyze_stability([1.0, 3.0, 3.0, 1.0])
        s = str(r)
        assert "Stability Analysis" in s
        assert "STABLE" in s
        assert "Poles" in s

    def test_zero_leading_coefficient_raises(self):
        with pytest.raises(ValueError):
            analyze_stability([0.0, 1.0, 1.0])

    def test_higher_degree(self):
        """4th degree: s^4 + 4s^3 + 6s^2 + 4s + 1 = (s+1)^4."""
        r = analyze_stability([1.0, 4.0, 6.0, 4.0, 1.0])
        assert r.verdict == "STABLE"
        assert len(r.poles) == 4
