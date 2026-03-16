"""noethersolve.enzyme_kinetics — Enzyme kinetics computational engine.

Computes Michaelis-Menten kinetics, inhibition effects, catalytic efficiency,
Lineweaver-Burk parameters, and multi-substrate kinetics from first principles.
Replaces static biochemistry fact-checking with actual calculation.

Conservation law philosophy: enzyme-substrate binding obeys mass-action
equilibria. The Michaelis-Menten equation IS a conservation law — total enzyme
is conserved ([E]_total = [E] + [ES]), and steady-state d[ES]/dt = 0 is
the constraint that produces the rate equation.

Usage:
    from noethersolve.enzyme_kinetics import (
        michaelis_menten, inhibition, catalytic_efficiency,
        lineweaver_burk, ph_rate_profile, cooperativity,
    )

    # Basic Michaelis-Menten
    r = michaelis_menten(Vmax=100, Km=5, S=10)
    print(r)  # V0 = 66.7 µM/s, 66.7% Vmax

    # Competitive inhibition
    r = inhibition(Vmax=100, Km=5, S=10, Ki=2, I=4, mode="competitive")
    print(r)  # Km_app = 15, V0 = 40.0 µM/s

    # Catalytic efficiency
    r = catalytic_efficiency(kcat=1000, Km=5e-6)
    print(r)  # kcat/Km = 2.0e8 M⁻¹s⁻¹ — near diffusion limit!
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Constants ───────────────────────────────────────────────────────────────

DIFFUSION_LIMIT = 1e8  # M⁻¹s⁻¹ — catalytic perfection threshold


# ─── Report Dataclasses ─────────────────────────────────────────────────────

@dataclass
class MMReport:
    """Michaelis-Menten kinetics result."""
    Vmax: float          # µM/s (or user units)
    Km: float            # µM (or user units)
    S: float             # substrate concentration
    V0: float            # initial velocity
    fraction_Vmax: float # V0 / Vmax
    saturation: str      # "unsaturated", "half-saturated", "near-saturated", "saturated"
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Michaelis-Menten Kinetics", "=" * 60]
        lines.append(f"  Vmax = {self.Vmax:.4g}    Km = {self.Km:.4g}    [S] = {self.S:.4g}")
        lines.append(f"  V0 = Vmax × [S] / (Km + [S]) = {self.V0:.4g}")
        lines.append(f"  Fraction of Vmax: {self.fraction_Vmax:.1%}  ({self.saturation})")
        lines.append("-" * 60)
        lines.append(f"  At [S] = Km: V0 = Vmax/2 = {self.Vmax/2:.4g}")
        lines.append(f"  At [S] = 10×Km: V0 = {self.Vmax * 10 / 11:.4g} (91% Vmax)")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class InhibitionReport:
    """Enzyme inhibition kinetics result."""
    mode: str              # competitive, noncompetitive, uncompetitive, mixed
    Vmax: float
    Km: float
    S: float
    Ki: float
    I: float               # inhibitor concentration
    Vmax_app: float        # apparent Vmax
    Km_app: float          # apparent Km
    V0_uninhibited: float  # V0 without inhibitor
    V0_inhibited: float    # V0 with inhibitor
    percent_inhibition: float
    Ki_prime: Optional[float] = None  # for mixed inhibition
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, f"  {self.mode.capitalize()} Inhibition", "=" * 60]
        lines.append(f"  Vmax = {self.Vmax:.4g}    Km = {self.Km:.4g}    Ki = {self.Ki:.4g}")
        lines.append(f"  [S] = {self.S:.4g}    [I] = {self.I:.4g}")
        if self.Ki_prime is not None:
            lines.append(f"  Ki' = {self.Ki_prime:.4g} (mixed)")
        lines.append("-" * 60)
        lines.append(f"  Apparent Vmax = {self.Vmax_app:.4g}    Apparent Km = {self.Km_app:.4g}")
        lines.append(f"  V0 (no inhibitor) = {self.V0_uninhibited:.4g}")
        lines.append(f"  V0 (with inhibitor) = {self.V0_inhibited:.4g}")
        lines.append(f"  Inhibition: {self.percent_inhibition:.1f}%")
        lines.append("-" * 60)

        # Key diagnostic: what changes?
        if self.mode == "competitive":
            lines.append("  Competitive: Km↑ (harder to bind), Vmax unchanged")
            lines.append("  Can be overcome by increasing [S]")
            lines.append(f"  Lineweaver-Burk: same y-intercept (1/Vmax), different x-intercept (-1/Km_app)")
        elif self.mode == "noncompetitive":
            lines.append("  Noncompetitive: Vmax↓ (fewer active enzyme), Km unchanged")
            lines.append("  CANNOT be overcome by increasing [S]")
            lines.append(f"  Lineweaver-Burk: different y-intercept (1/Vmax_app), same x-intercept (-1/Km)")
        elif self.mode == "uncompetitive":
            lines.append("  Uncompetitive: both Vmax↓ and Km↓ (ratio Vmax/Km unchanged)")
            lines.append("  Inhibitor binds ES complex only")
            lines.append(f"  Lineweaver-Burk: parallel lines (same slope, different intercepts)")
        elif self.mode == "mixed":
            lines.append("  Mixed: both Vmax↓ and Km changes")
            lines.append(f"  Lineweaver-Burk: lines intersect left of y-axis")

        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class EfficiencyReport:
    """Catalytic efficiency result."""
    kcat: float              # turnover number (s⁻¹)
    Km: float                # Michaelis constant (M)
    kcat_over_Km: float      # specificity constant (M⁻¹s⁻¹)
    is_diffusion_limited: bool
    efficiency_class: str    # "catalytically perfect", "very efficient", "efficient", "moderate", "slow"
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Catalytic Efficiency", "=" * 60]
        lines.append(f"  kcat = {self.kcat:.4g} s⁻¹")
        lines.append(f"  Km = {self.Km:.4g} M")
        lines.append(f"  kcat/Km = {self.kcat_over_Km:.4g} M⁻¹s⁻¹")
        lines.append(f"  Class: {self.efficiency_class}")
        if self.is_diffusion_limited:
            lines.append(f"  ⚡ Near diffusion limit (~10⁸–10⁹ M⁻¹s⁻¹) — catalytically perfect!")
        lines.append("-" * 60)
        # Reference enzymes
        lines.append("  Reference enzymes:")
        lines.append("    Carbonic anhydrase:  kcat/Km = 8.3×10⁷ (near perfect)")
        lines.append("    Acetylcholinesterase: kcat/Km = 1.6×10⁸ (diffusion limited)")
        lines.append("    Triosephosphate isomerase: kcat/Km = 2.4×10⁸ (perfect)")
        lines.append("    Fumarase: kcat/Km = 1.6×10⁸ (perfect)")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class LineweaverBurkReport:
    """Lineweaver-Burk (double reciprocal) plot parameters."""
    slope: float         # Km / Vmax
    y_intercept: float   # 1 / Vmax
    x_intercept: float   # -1 / Km
    Vmax: float
    Km: float
    data_points: List[Tuple[float, float]]  # (1/[S], 1/V0) pairs
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Lineweaver-Burk (Double Reciprocal) Plot", "=" * 60]
        lines.append(f"  1/V0 = (Km/Vmax) × (1/[S]) + 1/Vmax")
        lines.append(f"  Slope = Km/Vmax = {self.slope:.4g}")
        lines.append(f"  y-intercept = 1/Vmax = {self.y_intercept:.4g}  →  Vmax = {self.Vmax:.4g}")
        lines.append(f"  x-intercept = -1/Km = {self.x_intercept:.4g}  →  Km = {self.Km:.4g}")
        if self.data_points:
            lines.append("-" * 60)
            lines.append("  Data points (1/[S], 1/V0):")
            for inv_s, inv_v in self.data_points[:8]:
                lines.append(f"    ({inv_s:.4g}, {inv_v:.4g})")
            if len(self.data_points) > 8:
                lines.append(f"    ... and {len(self.data_points) - 8} more")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class PHProfileReport:
    """pH-rate profile result."""
    pH: float
    V_at_pH: float
    V_optimal: float
    pH_optimum: float
    pKa1: float
    pKa2: float
    fraction_active: float
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  pH-Rate Profile", "=" * 60]
        lines.append(f"  pH = {self.pH:.2f}")
        lines.append(f"  V(pH) = {self.V_at_pH:.4g}    V(optimal) = {self.V_optimal:.4g}")
        lines.append(f"  Fraction active: {self.fraction_active:.1%}")
        lines.append(f"  pH optimum = {self.pH_optimum:.2f}")
        lines.append(f"  pKa1 = {self.pKa1:.2f} (acid limb)    pKa2 = {self.pKa2:.2f} (base limb)")
        lines.append("-" * 60)
        lines.append("  Bell-shaped curve: V = Vopt / (1 + [H⁺]/Ka1 + Ka2/[H⁺])")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class CooperativityReport:
    """Hill equation cooperativity result."""
    Vmax: float
    K_half: float     # [S] at half-maximal velocity (like Km but for cooperative)
    n: float          # Hill coefficient
    S: float
    V0: float
    fraction_Vmax: float
    cooperativity_type: str  # "positive", "negative", "non-cooperative"
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Hill Equation (Cooperativity)", "=" * 60]
        lines.append(f"  Vmax = {self.Vmax:.4g}    K₀.₅ = {self.K_half:.4g}    n = {self.n:.2f}")
        lines.append(f"  [S] = {self.S:.4g}")
        lines.append(f"  V0 = Vmax × [S]ⁿ / (K₀.₅ⁿ + [S]ⁿ) = {self.V0:.4g}")
        lines.append(f"  Fraction of Vmax: {self.fraction_Vmax:.1%}")
        lines.append(f"  Cooperativity: {self.cooperativity_type} (n {'>' if self.n > 1 else '<' if self.n < 1 else '='} 1)")
        lines.append("-" * 60)
        if self.n > 1:
            lines.append("  Sigmoidal curve — binding of first substrate enhances subsequent binding")
            lines.append("  Example: hemoglobin O₂ binding (n ≈ 2.8)")
        elif self.n < 1:
            lines.append("  Flattened curve — binding of first substrate reduces subsequent binding")
        else:
            lines.append("  Hyperbolic curve — identical to Michaelis-Menten (n = 1)")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Public API ──────────────────────────────────────────────────────────────

def michaelis_menten(
    Vmax: float,
    Km: float,
    S: float,
) -> MMReport:
    """Compute Michaelis-Menten initial velocity.

    V0 = Vmax × [S] / (Km + [S])

    Args:
        Vmax: maximum velocity (any consistent units, e.g., µM/s)
        Km: Michaelis constant (same concentration units as S)
        S: substrate concentration

    Returns:
        MMReport with V0, fraction of Vmax, and saturation status.
    """
    if Km <= 0:
        raise ValueError(f"Km must be positive, got {Km}")
    if S < 0:
        raise ValueError(f"[S] must be non-negative, got {S}")
    if Vmax <= 0:
        raise ValueError(f"Vmax must be positive, got {Vmax}")

    V0 = Vmax * S / (Km + S) if (Km + S) > 0 else 0.0
    frac = V0 / Vmax if Vmax > 0 else 0.0

    if frac < 0.1:
        sat = "unsaturated ([S] << Km, first-order in [S])"
    elif frac < 0.45:
        sat = "partially saturated"
    elif frac < 0.55:
        sat = "half-saturated ([S] ≈ Km)"
    elif frac < 0.9:
        sat = "near-saturated"
    else:
        sat = "saturated ([S] >> Km, zero-order in [S])"

    notes = []
    if S < 0.1 * Km:
        notes.append(f"First-order regime: V0 ≈ (Vmax/Km)×[S] = {Vmax/Km:.4g} × {S:.4g} = {Vmax*S/Km:.4g}")
    if S > 10 * Km:
        notes.append(f"Zero-order regime: V0 ≈ Vmax = {Vmax:.4g}")

    return MMReport(
        Vmax=Vmax, Km=Km, S=S, V0=V0,
        fraction_Vmax=frac, saturation=sat, notes=notes,
    )


def inhibition(
    Vmax: float,
    Km: float,
    S: float,
    Ki: float,
    I: float,
    mode: str = "competitive",
    Ki_prime: Optional[float] = None,
) -> InhibitionReport:
    """Compute enzyme kinetics with inhibitor.

    Modes:
        competitive:    Km_app = Km(1 + [I]/Ki),      Vmax_app = Vmax
        noncompetitive: Km_app = Km,                    Vmax_app = Vmax/(1 + [I]/Ki)
        uncompetitive:  Km_app = Km/(1 + [I]/Ki),      Vmax_app = Vmax/(1 + [I]/Ki)
        mixed:          Km_app = Km(1+[I]/Ki)/(1+[I]/Ki'), Vmax_app = Vmax/(1+[I]/Ki')

    Args:
        Vmax: maximum velocity
        Km: Michaelis constant
        S: substrate concentration
        Ki: inhibition constant (dissociation of E-I complex)
        I: inhibitor concentration
        mode: "competitive", "noncompetitive", "uncompetitive", or "mixed"
        Ki_prime: for mixed inhibition, dissociation of ES-I complex

    Returns:
        InhibitionReport with apparent parameters and inhibition percentage.
    """
    if Ki <= 0:
        raise ValueError(f"Ki must be positive, got {Ki}")
    if I < 0:
        raise ValueError(f"[I] must be non-negative, got {I}")

    mode = mode.lower().strip()
    alpha = 1 + I / Ki  # competitive factor

    if mode == "competitive":
        Km_app = Km * alpha
        Vmax_app = Vmax
    elif mode == "noncompetitive":
        Km_app = Km
        Vmax_app = Vmax / alpha
    elif mode == "uncompetitive":
        Km_app = Km / alpha
        Vmax_app = Vmax / alpha
    elif mode == "mixed":
        if Ki_prime is None:
            raise ValueError("Mixed inhibition requires Ki_prime parameter")
        if Ki_prime <= 0:
            raise ValueError(f"Ki_prime must be positive, got {Ki_prime}")
        alpha_prime = 1 + I / Ki_prime
        Km_app = Km * alpha / alpha_prime
        Vmax_app = Vmax / alpha_prime
    else:
        raise ValueError(f"Unknown inhibition mode: {mode}. "
                         f"Use 'competitive', 'noncompetitive', 'uncompetitive', or 'mixed'.")

    V0_uninh = Vmax * S / (Km + S)
    V0_inh = Vmax_app * S / (Km_app + S) if (Km_app + S) > 0 else 0.0
    pct_inh = (1 - V0_inh / V0_uninh) * 100 if V0_uninh > 0 else 0.0

    notes = []
    if mode == "competitive" and S > 10 * Km_app:
        notes.append("At high [S], competitive inhibition is largely overcome")
    if I > 10 * Ki:
        notes.append(f"[I] >> Ki: enzyme is heavily inhibited ({pct_inh:.0f}%)")

    return InhibitionReport(
        mode=mode, Vmax=Vmax, Km=Km, S=S, Ki=Ki, I=I,
        Vmax_app=Vmax_app, Km_app=Km_app,
        V0_uninhibited=V0_uninh, V0_inhibited=V0_inh,
        percent_inhibition=pct_inh,
        Ki_prime=Ki_prime if mode == "mixed" else None,
        notes=notes,
    )


def catalytic_efficiency(
    kcat: float,
    Km: float,
) -> EfficiencyReport:
    """Compute catalytic efficiency (specificity constant).

    kcat/Km is the second-order rate constant for the reaction of free
    enzyme with substrate. Upper limit is the diffusion limit (~10⁸-10⁹ M⁻¹s⁻¹).

    Args:
        kcat: turnover number in s⁻¹ (molecules/enzyme/second)
        Km: Michaelis constant in M (molar)

    Returns:
        EfficiencyReport with efficiency classification.
    """
    if kcat <= 0:
        raise ValueError(f"kcat must be positive, got {kcat}")
    if Km <= 0:
        raise ValueError(f"Km must be positive, got {Km}")

    ratio = kcat / Km
    is_diff = ratio >= DIFFUSION_LIMIT

    if ratio >= 1e8:
        cls = "catalytically perfect (diffusion-limited)"
    elif ratio >= 1e6:
        cls = "very efficient"
    elif ratio >= 1e4:
        cls = "efficient"
    elif ratio >= 1e2:
        cls = "moderate"
    else:
        cls = "slow"

    notes = []
    if is_diff:
        notes.append("Every encounter between enzyme and substrate leads to product — "
                      "the reaction is limited only by how fast they can meet (diffusion).")
    if kcat > 1e6:
        notes.append(f"Very high turnover: {kcat:.2g} molecules/s per enzyme active site")

    return EfficiencyReport(
        kcat=kcat, Km=Km, kcat_over_Km=ratio,
        is_diffusion_limited=is_diff, efficiency_class=cls,
        notes=notes,
    )


def lineweaver_burk(
    Vmax: float,
    Km: float,
    S_values: Optional[List[float]] = None,
) -> LineweaverBurkReport:
    """Compute Lineweaver-Burk (double reciprocal) plot parameters.

    1/V0 = (Km/Vmax)(1/[S]) + 1/Vmax

    Args:
        Vmax: maximum velocity
        Km: Michaelis constant
        S_values: optional list of substrate concentrations to generate
                  data points. If None, uses default range.

    Returns:
        LineweaverBurkReport with slope, intercepts, and data points.
    """
    if Vmax <= 0:
        raise ValueError(f"Vmax must be positive, got {Vmax}")
    if Km <= 0:
        raise ValueError(f"Km must be positive, got {Km}")

    slope = Km / Vmax
    y_int = 1.0 / Vmax
    x_int = -1.0 / Km

    if S_values is None:
        S_values = [Km * f for f in [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]

    points = []
    for s in S_values:
        if s > 0:
            v0 = Vmax * s / (Km + s)
            if v0 > 0:
                points.append((1.0 / s, 1.0 / v0))

    notes = []
    notes.append("Lineweaver-Burk linearizes MM but amplifies errors at low [S] — "
                 "use Eadie-Hofstee or direct nonlinear fitting for real data")

    return LineweaverBurkReport(
        slope=slope, y_intercept=y_int, x_intercept=x_int,
        Vmax=Vmax, Km=Km, data_points=points, notes=notes,
    )


def ph_rate_profile(
    pH: float,
    V_optimal: float,
    pKa1: float,
    pKa2: float,
) -> PHProfileReport:
    """Compute enzyme activity at a given pH using the bell-shaped model.

    V(pH) = V_opt / (1 + [H⁺]/Ka1 + Ka2/[H⁺])

    The enzyme is maximally active when both catalytic residues are in their
    correct protonation state (between pKa1 and pKa2).

    Args:
        pH: pH value to evaluate
        V_optimal: maximum velocity at optimal pH
        pKa1: pKa of acid limb (lower pH side)
        pKa2: pKa of base limb (higher pH side)

    Returns:
        PHProfileReport with activity at given pH.
    """
    if pKa1 >= pKa2:
        raise ValueError(f"pKa1 ({pKa1}) must be less than pKa2 ({pKa2})")

    H = 10 ** (-pH)
    Ka1 = 10 ** (-pKa1)
    Ka2 = 10 ** (-pKa2)

    denominator = 1 + H / Ka1 + Ka2 / H
    V_at_pH = V_optimal / denominator
    fraction = V_at_pH / V_optimal if V_optimal > 0 else 0.0

    # Optimal pH is the geometric mean of pKa values
    pH_opt = (pKa1 + pKa2) / 2.0

    notes = []
    if abs(pH - pH_opt) > 2:
        notes.append(f"pH {pH:.1f} is far from optimum {pH_opt:.1f} — enzyme is largely inactive")
    if pH < pKa1 - 1:
        notes.append("Below pKa1: catalytic base is protonated (inactive)")
    if pH > pKa2 + 1:
        notes.append("Above pKa2: catalytic acid is deprotonated (inactive)")

    return PHProfileReport(
        pH=pH, V_at_pH=V_at_pH, V_optimal=V_optimal,
        pH_optimum=pH_opt, pKa1=pKa1, pKa2=pKa2,
        fraction_active=fraction, notes=notes,
    )


def cooperativity(
    Vmax: float,
    K_half: float,
    n: float,
    S: float,
) -> CooperativityReport:
    """Compute velocity using the Hill equation for cooperative binding.

    V0 = Vmax × [S]ⁿ / (K₀.₅ⁿ + [S]ⁿ)

    The Hill coefficient n describes cooperativity:
        n > 1: positive cooperativity (sigmoidal curve)
        n = 1: no cooperativity (reduces to Michaelis-Menten)
        n < 1: negative cooperativity

    Args:
        Vmax: maximum velocity
        K_half: substrate concentration at half-maximal velocity
        n: Hill coefficient
        S: substrate concentration

    Returns:
        CooperativityReport with velocity and cooperativity analysis.
    """
    if Vmax <= 0:
        raise ValueError(f"Vmax must be positive, got {Vmax}")
    if K_half <= 0:
        raise ValueError(f"K_half must be positive, got {K_half}")
    if n <= 0:
        raise ValueError(f"Hill coefficient n must be positive, got {n}")
    if S < 0:
        raise ValueError(f"[S] must be non-negative, got {S}")

    S_n = S ** n
    K_n = K_half ** n
    V0 = Vmax * S_n / (K_n + S_n) if (K_n + S_n) > 0 else 0.0
    frac = V0 / Vmax if Vmax > 0 else 0.0

    if abs(n - 1.0) < 0.05:
        coop_type = "non-cooperative (n ≈ 1, equivalent to Michaelis-Menten)"
    elif n > 1:
        coop_type = "positive cooperativity"
    else:
        coop_type = "negative cooperativity"

    notes = []
    if n > 1:
        # The Hill coefficient gives a LOWER BOUND on the number of binding sites
        notes.append(f"Hill coefficient n = {n:.2f} implies at least {math.ceil(n)} binding sites")
        notes.append("True n for hemoglobin (4 sites) is ~2.8, not 4 — n is always ≤ actual sites")

    return CooperativityReport(
        Vmax=Vmax, K_half=K_half, n=n, S=S, V0=V0,
        fraction_Vmax=frac, cooperativity_type=coop_type,
        notes=notes,
    )
