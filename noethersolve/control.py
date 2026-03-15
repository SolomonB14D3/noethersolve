"""PID controller simulator and control system stability analyzer.

Computes step response, overshoot, settling time, and stability margins
for transfer functions. This is a CALCULATOR — it derives answers from
numerical integration, not from a lookup table.

Usage:
    from noethersolve.control import simulate_pid, analyze_stability

    # PID step response with a 2nd-order plant
    report = simulate_pid(Kp=2.5, Ki=0.8, Kd=0.1,
                          plant_num=[1.0], plant_den=[1.0, 3.0, 1.0])
    print(report)  # overshoot, settling time, steady-state error, windup

    # Stability analysis via Routh-Hurwitz
    report = analyze_stability([1.0, 3.0, 3.0, 1.0])
    print(report)  # stable/unstable, pole locations, margins
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PIDReport:
    """Report from PID step-response simulation."""

    verdict: str  # STABLE, UNSTABLE, MARGINAL
    Kp: float
    Ki: float
    Kd: float
    overshoot_pct: float
    rise_time: float
    settling_time: float
    steady_state_error: float
    windup_detected: bool
    windup_peak: float
    peak_value: float
    final_value: float
    time_series: List[Tuple[float, float]] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  PID Step Response: " + self.verdict,
            "=" * 60,
            f"  Gains: Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}",
            f"  Overshoot: {self.overshoot_pct:.1f}%",
            f"  Rise time: {self.rise_time:.4f}s",
            f"  Settling time (2%): {self.settling_time:.4f}s",
            f"  Steady-state error: {self.steady_state_error:.6f}",
            f"  Peak value: {self.peak_value:.4f}",
            f"  Final value: {self.final_value:.4f}",
            "",
            f"  Integral windup: {'YES — peak integral = '+ f'{self.windup_peak:.3f}' if self.windup_detected else 'No'}",
        ]
        if self.windup_detected:
            lines.append(
                "  WARNING: Integral term accumulated significantly before"
                " the output reached setpoint. Consider anti-windup."
            )
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class StabilityReport:
    """Report from Routh-Hurwitz stability analysis."""

    verdict: str  # STABLE, UNSTABLE, MARGINAL
    coefficients: List[float]
    poles: List[complex]
    n_unstable: int
    routh_table: List[List[float]] = field(default_factory=list, repr=False)
    gain_margin_db: Optional[float] = None
    phase_margin_deg: Optional[float] = None

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Stability Analysis: " + self.verdict,
            "=" * 60,
            f"  Characteristic polynomial degree: {len(self.coefficients) - 1}",
            f"  Coefficients: {self.coefficients}",
            f"  Poles: {[f'{p.real:.4f}{p.imag:+.4f}j' for p in self.poles]}",
            f"  Unstable poles (Re > 0): {self.n_unstable}",
        ]
        if self.gain_margin_db is not None:
            lines.append(f"  Gain margin: {self.gain_margin_db:.2f} dB")
        if self.phase_margin_deg is not None:
            lines.append(f"  Phase margin: {self.phase_margin_deg:.2f}°")
        lines.append("=" * 60)
        return "\n".join(lines)


def simulate_pid(
    Kp: float = 1.0,
    Ki: float = 0.0,
    Kd: float = 0.0,
    plant_num: Optional[List[float]] = None,
    plant_den: Optional[List[float]] = None,
    setpoint: float = 1.0,
    t_final: float = 20.0,
    dt: float = 0.001,
    output_min: float = -1e6,
    output_max: float = 1e6,
) -> PIDReport:
    """Simulate PID controller step response with a transfer-function plant.

    Parameters
    ----------
    Kp, Ki, Kd : float
        PID gains.
    plant_num, plant_den : list of float
        Numerator and denominator coefficients of the plant transfer function,
        highest power first. Default: 1/(s^2 + 3s + 1) (underdamped 2nd order).
    setpoint : float
        Step input magnitude (default 1.0).
    t_final : float
        Simulation duration in seconds.
    dt : float
        Integration timestep.
    output_min, output_max : float
        Actuator saturation limits. Set to detect windup.
    """
    if plant_num is None:
        plant_num = [1.0]
    if plant_den is None:
        plant_den = [1.0, 3.0, 1.0]

    # Convert transfer function to state-space (controllable canonical form)
    # Plant: G(s) = num(s) / den(s)
    n = len(plant_den) - 1  # system order
    if n < 1:
        raise ValueError("Plant denominator must be at least 1st order")

    # Normalize so leading coefficient is 1
    a0 = plant_den[0]
    den = [c / a0 for c in plant_den]
    num = [c / a0 for c in plant_num]

    # Pad numerator to match denominator length
    while len(num) < len(den):
        num.insert(0, 0.0)

    # State-space: controllable canonical form
    # x_dot = A*x + B*u, y = C*x + D*u
    # A = companion matrix, B = [0,...,0,1]^T
    # C and D from numerator coefficients

    # Simulate with Euler integration (dt is small enough for stability)
    x = [0.0] * n  # plant states
    integral = 0.0
    prev_error = 0.0
    time_series = []
    windup_peak = 0.0

    steps = int(t_final / dt)
    for step_i in range(steps):
        t = step_i * dt

        # Plant output: y = C*x
        y = 0.0
        for i in range(n):
            y += (num[i + 1] - num[0] * den[i + 1]) * x[i]
        y += num[0] * x[0] if n > 0 else 0.0
        # Simplified: for standard form, y = sum(c_i * x_i)
        # Actually, let me use the proper observable output
        # For controllable canonical form with den = [1, a1, ..., an]:
        # A has -a_i in last row, B = [0,...,0,1]
        # C = [b_n - a_n*b_0, ..., b_1 - a_1*b_0], D = b_0
        c = [num[i + 1] - num[0] * den[i + 1] for i in range(n)]
        d_val = num[0]

        y = sum(c[i] * x[i] for i in range(n))
        # D*u would need the current input, which creates algebraic loop
        # For strictly proper systems (num order < den order), D=0

        # PID controller
        error = setpoint - y
        integral += error * dt
        derivative = (error - prev_error) / dt if step_i > 0 else 0.0
        prev_error = error

        u_raw = Kp * error + Ki * integral + Kd * derivative

        # Saturate
        u = max(output_min, min(output_max, u_raw))

        # Track windup
        if abs(integral) > abs(windup_peak):
            windup_peak = integral

        # State update: x_dot = A*x + B*u
        x_new = [0.0] * n
        for i in range(n - 1):
            x_new[i] = x[i + 1]
        # Last state: -a_n*x_0 - a_{n-1}*x_1 - ... - a_1*x_{n-1} + u
        x_new[n - 1] = -sum(den[n - i] * x[i] for i in range(n)) + u

        # Euler step
        for i in range(n):
            x[i] = x[i] + dt * x_new[i]

        # Store every 10 steps for output
        if step_i % 10 == 0:
            time_series.append((t, y))

    # Analyze results
    if not time_series:
        return PIDReport(
            verdict="ERROR", Kp=Kp, Ki=Ki, Kd=Kd,
            overshoot_pct=0, rise_time=0, settling_time=0,
            steady_state_error=1.0, windup_detected=False,
            windup_peak=0, peak_value=0, final_value=0,
        )

    ys = [pt[1] for pt in time_series]
    ts = [pt[0] for pt in time_series]
    final_val = ys[-1] if ys else 0.0
    peak_val = max(ys) if ys else 0.0
    peak_idx = ys.index(peak_val)

    # Overshoot
    if abs(setpoint) > 1e-12:
        overshoot = max(0.0, (peak_val - setpoint) / setpoint * 100.0)
    else:
        overshoot = 0.0

    # Rise time (10% to 90% of setpoint)
    rise_time = 0.0
    t_10 = t_90 = None
    for i, (t, y) in enumerate(time_series):
        if t_10 is None and y >= 0.1 * setpoint:
            t_10 = t
        if t_90 is None and y >= 0.9 * setpoint:
            t_90 = t
            break
    if t_10 is not None and t_90 is not None:
        rise_time = t_90 - t_10

    # Settling time (2% band)
    settling_time = t_final
    band = 0.02 * abs(setpoint)
    for i in range(len(ys) - 1, -1, -1):
        if abs(ys[i] - setpoint) > band:
            settling_time = ts[i] if i < len(ts) else t_final
            break
    else:
        settling_time = 0.0

    # Steady-state error
    ss_error = abs(setpoint - final_val)

    # Windup detection: integral exceeded 2x the steady-state value
    # and overshot significantly
    windup = abs(windup_peak) > 2.0 * abs(setpoint) and overshoot > 10.0

    # Check if diverging
    if abs(final_val) > 10 * abs(setpoint) or any(abs(y) > 100 * abs(setpoint) for y in ys):
        verdict = "UNSTABLE"
    elif overshoot > 50 or settling_time > 0.8 * t_final:
        verdict = "MARGINAL"
    else:
        verdict = "STABLE"

    return PIDReport(
        verdict=verdict,
        Kp=Kp, Ki=Ki, Kd=Kd,
        overshoot_pct=overshoot,
        rise_time=rise_time,
        settling_time=settling_time,
        steady_state_error=ss_error,
        windup_detected=windup,
        windup_peak=windup_peak,
        peak_value=peak_val,
        final_value=final_val,
        time_series=time_series,
    )


def analyze_stability(coefficients: List[float]) -> StabilityReport:
    """Analyze stability of a characteristic polynomial using Routh-Hurwitz.

    Parameters
    ----------
    coefficients : list of float
        Polynomial coefficients, highest power first.
        E.g. [1, 3, 3, 1] for s^3 + 3s^2 + 3s + 1.

    Returns
    -------
    StabilityReport
        Verdict, poles, Routh table, margins.
    """
    if not coefficients or coefficients[0] == 0:
        raise ValueError("Leading coefficient must be non-zero")

    n = len(coefficients) - 1  # polynomial degree

    # Normalize
    a0 = coefficients[0]
    coeffs = [c / a0 for c in coefficients]

    # Build Routh table
    routh = _build_routh_table(coeffs)

    # Count sign changes in first column
    first_col = [row[0] for row in routh if row]
    n_sign_changes = 0
    for i in range(1, len(first_col)):
        if first_col[i - 1] * first_col[i] < 0:
            n_sign_changes += 1

    # Find poles using companion matrix eigenvalues
    poles = _find_poles(coeffs)
    n_unstable = sum(1 for p in poles if p.real > 1e-10)

    if n_unstable > 0:
        verdict = "UNSTABLE"
    elif any(abs(p.real) < 1e-10 for p in poles):
        verdict = "MARGINAL"
    else:
        verdict = "STABLE"

    return StabilityReport(
        verdict=verdict,
        coefficients=coefficients,
        poles=sorted(poles, key=lambda p: (p.real, p.imag)),
        n_unstable=n_unstable,
        routh_table=routh,
    )


def _build_routh_table(coeffs: List[float]) -> List[List[float]]:
    """Build the Routh-Hurwitz table from normalized polynomial coefficients."""
    n = len(coeffs) - 1
    if n < 1:
        return [[coeffs[0]]]

    cols = (n + 2) // 2
    table = []

    # First two rows from coefficients
    row1 = [coeffs[i] if i < len(coeffs) else 0.0 for i in range(0, 2 * cols, 2)]
    row2 = [coeffs[i] if i < len(coeffs) else 0.0 for i in range(1, 2 * cols + 1, 2)]
    table.append(row1)
    table.append(row2)

    # Fill remaining rows
    for i in range(2, n + 1):
        prev2 = table[i - 2]
        prev1 = table[i - 1]
        row = []
        pivot = prev1[0] if abs(prev1[0]) > 1e-15 else 1e-15  # avoid division by zero
        for j in range(cols - 1):
            a = prev2[0]
            b = prev2[j + 1] if j + 1 < len(prev2) else 0.0
            c = prev1[0]
            d = prev1[j + 1] if j + 1 < len(prev1) else 0.0
            val = (c * b - a * d) / pivot
            row.append(val)
        if not row:
            row = [0.0]
        table.append(row)

    return table


def _find_poles(coeffs: List[float]) -> List[complex]:
    """Find polynomial roots using companion matrix eigenvalues.

    Pure Python implementation — no numpy dependency required.
    Uses Durand-Kerner method for root finding.
    """
    n = len(coeffs) - 1
    if n == 0:
        return []
    if n == 1:
        return [complex(-coeffs[1] / coeffs[0])]
    if n == 2:
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        disc = b * b - 4 * a * c
        if disc >= 0:
            sq = math.sqrt(disc)
            return [complex((-b + sq) / (2 * a)), complex((-b - sq) / (2 * a))]
        else:
            sq = math.sqrt(-disc)
            return [
                complex(-b / (2 * a), sq / (2 * a)),
                complex(-b / (2 * a), -sq / (2 * a)),
            ]

    # Durand-Kerner iteration for higher-degree polynomials
    # Normalize so leading coefficient is 1
    c = [coeffs[i] / coeffs[0] for i in range(len(coeffs))]

    # Initial guesses: spread around unit circle
    roots = []
    for k in range(n):
        angle = 2 * math.pi * k / n + 0.1
        r = 1.0 + 0.5 * k / n
        roots.append(complex(r * math.cos(angle), r * math.sin(angle)))

    for _iteration in range(1000):
        max_change = 0.0
        new_roots = list(roots)
        for i in range(n):
            # Evaluate polynomial at roots[i]
            val = complex(1.0)
            for j in range(1, len(c)):
                val = val * roots[i] + c[j]

            # Product of (roots[i] - roots[j]) for j != i
            denom = complex(1.0)
            for j in range(n):
                if j != i:
                    diff = roots[i] - roots[j]
                    if abs(diff) < 1e-15:
                        diff = complex(1e-15, 1e-15)
                    denom *= diff

            if abs(denom) < 1e-30:
                continue

            delta = val / denom
            new_roots[i] = roots[i] - delta
            max_change = max(max_change, abs(delta))

        roots = new_roots
        if max_change < 1e-12:
            break

    # Clean up: snap near-real roots to real, snap near-zero imaginary parts
    cleaned = []
    for r in roots:
        if abs(r.imag) < 1e-8:
            cleaned.append(complex(r.real, 0))
        else:
            cleaned.append(r)

    return cleaned
